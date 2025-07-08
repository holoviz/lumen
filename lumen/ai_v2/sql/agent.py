"""
SQL Agent Implementation
"""

from typing import Any

import param

from ...sources.base import BaseSQLSource
from ..core.agent import BaseAgent, action
from ..core.models import ActionResult, Checkpoints, Roadmap
from .models import (
    LineChanges, SQLQueries, SQLQueryOutput, SQLQueryOutputs,
)


class SQLAgent(BaseAgent):
    """Agent that generates and executes SQL queries

    Purpose:
        Specialized agent for SQL query generation and database schema exploration.
        Uses hierarchical planning with efficient milestone execution for SQL goals.
    """

    name = param.String(default="sql_agent")

    source = param.ClassSelector(class_=BaseSQLSource, doc="Data source for schema and execution")

    async def create_roadmap(self, goal: str, checkpoints: Checkpoints) -> Roadmap:
        """{{ super() }}

        IMPORTANT SQL-SPECIFIC GUIDELINES:
        - Combine related SQL operations into single substantial milestones
        - Example: "Generate and execute query to find top 5 customers by total order amount"
        - Consider schema discovery as a prerequisite for complex queries
        - Create 2-3 meaningful milestones rather than many small steps
        - If the goal involves aggregations or joins, consider table structure examination first
        """
        return await super().create_roadmap(goal, checkpoints)

    async def execute_milestone_step(self, messages: list[dict]) -> ActionResult:
        """{{ super() }}

        IMPORTANT SQL-SPECIFIC GUIDELINES:
        - If you need to query data but don't know table structure, use examine_table first
        - If column names are unknown, use examine_table before generating queries
        - Combine related operations when possible (generate then execute queries in sequence)
        - Build upon previous work rather than duplicating it
        - If previous actions failed due to wrong column names, examine table structure first
        - Use artifacts from previous milestones to avoid redundant schema discovery
        """
        return await super().execute_milestone_step(messages)

    # ===== Actions =====

    @action
    async def find_tables(self, table_patterns: list[str], examine: bool = True) -> SQLQueryOutputs:
        """Find tables in the database matching given patterns."""
        table_patterns = [pattern.replace("*", "%") if "*" in pattern or "%" in pattern else f"%{pattern}%" for pattern in table_patterns]
        content = [f"SELECT table_name FROM information_schema.tables WHERE table_name LIKE '{pattern}' LIMIT 10" for pattern in table_patterns]
        queries = SQLQueries(reasoning=f"Discovering schema for patterns: {table_patterns}", content=content)
        return await self.execute_multiple_queries(queries)

    @action
    async def examine_table(self, table_names: list[str]) -> SQLQueryOutputs:
        """Examine specific table structures and sample data to discover schema, like column names and types."""
        content = []
        for table_name in table_names:
            content.extend([f"DESCRIBE {table_name}", f"SELECT * FROM {table_name} LIMIT 5", f"SELECT COUNT(*) as row_count FROM {table_name}"])
        queries = SQLQueries(reasoning=f"Examining table structure and sample data for {table_names}", content=content)
        return await self.execute_multiple_queries(queries)

    @action
    async def generate_multiple_queries(self, instruction: str, execute: bool = False) -> SQLQueries | SQLQueryOutputs:
        """Generate SQL query(s) based on the given instruction and optionally execute them immediately.

        You are a SQL expert. Based on the instruction provided, generate the appropriate SQL queries.
        Consider the database schema and table structures that have been discovered.

        Return the SQL queries that will achieve the goal described in the instruction.
        Focus on writing efficient, correct SQL syntax.
        """
        messages = [{"role": "user", "content": f"{instruction=}"}]
        sql_queries = await self._call_llm("generate_multiple_queries", messages, SQLQueries)
        if execute:
            # If execute is True, execute the generated queries immediately
            outputs = await self.execute_multiple_queries(sql_queries)
            return outputs
        return sql_queries

    @action(locked=True)
    async def apply_line_changes(self, query: str, feedback: str) -> LineChanges:
        """Edit query based on feedback.

        You are editing a SQL query based on feedback. The query is numbered line by line.

        Query with line numbers:
        {{ enumerated_query }}

        Provide the specific line changes needed to address the feedback.
        """
        lines = query.split("\n")
        messages = [{"role": "user", "content": f"{feedback=}"}]
        enumerated_query = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
        line_changes = await self._call_llm("apply_line_changes", messages, LineChanges, enumerated_query=enumerated_query)

        # Apply the changes
        for change in line_changes.content:
            if change.replacement:
                lines[change.line_no - 1] = change.replacement
            else:
                lines[change.line_no - 1] = ""
        return "\n".join(lines).strip()

    @action
    async def execute_multiple_queries(self, queries: SQLQueries) -> SQLQueryOutputs:
        """Execute SQL query(s)."""
        self.logger.info(f"Executing {len(queries.content)} SQL queries")

        # Execute queries and track outputs
        outputs = []
        for i, sql_query in enumerate(queries.content):
            try:
                self.logger.debug(f"Executing query {i + 1}: {sql_query.content}...")
                df = await self.source.execute_async(sql_query=sql_query.content)
                output = SQLQueryOutput(content=df, table_name=sql_query.table_name)
                output.query = sql_query.content  # Store the original query
                outputs.append(output)
                self.logger.debug(f"Query {i + 1} succeeded: {len(df)} rows")
            except Exception as e:
                error_msg = f"Query {i + 1} failed: {e}"
                self.logger.warning(error_msg)
                # Continue with other queries instead of failing completely
                continue

        success_count = len(outputs)
        total_count = len(queries.content)
        self.logger.info(f"Completed: {success_count}/{total_count} queries succeeded")

        # Log outputs for debugging milestone completion
        if outputs:
            self.logger.debug(f"Schema discovery successful: {success_count} queries returned data")
            for i, output in enumerate(outputs):
                if hasattr(output.content, "shape") and len(output.content) > 0:
                    self.logger.debug(f"  Output {i + 1}: {output.table_name} - {len(output.content)} rows")
                    # Log first few table names if available
                    if "table_name" in output.content.columns:
                        tables = output.content["table_name"].tolist()[:5]
                        self.logger.debug(f"    Tables found: {tables}")
                    elif "name" in output.content.columns:
                        tables = output.content["name"].tolist()[:5]
                        self.logger.debug(f"    Tables found: {tables}")
                    elif "tablename" in output.content.columns:
                        tables = output.content["tablename"].tolist()[:5]
                        self.logger.debug(f"    Tables found: {tables}")
        else:
            self.logger.warning("Schema discovery failed: no successful query outputs")

        return SQLQueryOutputs(content=outputs)
