"""
SQL Agent Implementation
"""

import param

from ...sources.base import BaseSQLSource
from ..core.agent import BaseAgent, action
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

    # ===== Actions =====

    @action
    async def generate_multiple_queries(self, instruction: str) -> SQLQueries:
        """Generate SQL query(s)."""
        system_prompt = self._render_prompt("generate_multiple_queries")
        messages = [{"role": "user", "content": instruction}]
        sql_queries = await self.llm.invoke(messages=messages, system=system_prompt, response_model=SQLQueries, model_spec="sql")
        return sql_queries

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

    @action(locked=True)
    async def identify_line_changes(self, query: str, feedback: str) -> LineChanges:
        """Edit query based on feedback."""
        lines = query.split("\n")
        enumerated_query = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
        system_prompt = self._render_prompt("edit_lines", enumerated_query)
        messages = [{"role": "user", "content": f"Query:\n{enumerated_query}\n\nFeedback:\n{feedback}"}]
        line_changes = await self.llm.invoke(messages=messages, system=system_prompt, response_model=LineChanges, model_spec="reasoning")
        return line_changes

    @action
    async def apply_line_changes(self, query: str, line_changes: LineChanges) -> str:
        """Apply line changes."""
        lines = query.split("\n")
        for change in line_changes.content:
            if change.replacement:
                lines[change.line_no - 1] = change.replacement
            else:
                lines[change.line_no - 1] = ""
        return "\n".join(lines).strip()

    @action
    async def find_tables(self, table_patterns: list[str]) -> SQLQueryOutputs:
        """Discover database schema and tables by searching for table names matching patterns."""
        table_patterns = [pattern.replace("*", "%") if "*" in pattern or "%" in pattern else f"%{pattern}%" for pattern in table_patterns]
        content = [f"SELECT table_name FROM information_schema.tables WHERE table_name LIKE '{pattern}' LIMIT 10" for pattern in table_patterns]
        queries = SQLQueries(reasoning=f"Discovering schema for patterns: {table_patterns}", content=content)
        return await self.execute_multiple_queries(queries)

    @action
    async def examine_table(self, table_names: list[str]) -> SQLQueryOutputs:
        """Examine specific table structures and sample data."""
        content = []
        for table_name in table_names:
            content.extend([f"DESCRIBE {table_name}", f"SELECT * FROM {table_name} LIMIT 5", f"SELECT COUNT(*) as row_count FROM {table_name}"])
        queries = SQLQueries(reasoning=f"Examining table structure and sample data for {table_names}", content=content)
        return await self.execute_multiple_queries(queries)
