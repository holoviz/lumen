"""
SQL Agent Implementation
"""

import param

from ...sources.base import BaseSQLSource
from ..core.agent import BaseAgent, action
from ..core.models import ActionResult, Roadmap
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

    # ===== Internal Methods =====

    async def _apply_line_changes(self, query: str, feedback: str) -> LineChanges:
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

    # ===== Agent Methods =====

    async def create_roadmap(self, goal: str) -> Roadmap:
        """{{ super() }}

        Split compound milestones with internal dependencies ("Find tables and columns"
        needs table names first). Typical workflow: Find tables → Examine structures/joins →
        Execute final query. Don't separate operations SQL can do in one query (aggregation, sorting, filtering).
        """
        return await super().create_roadmap(goal)

    async def execute_milestone_step(self, messages: list[dict]) -> ActionResult:
        """{{ super() }}

        Build upon previous work. Use examine_table before querying if structure unknown.
        Consider outputs from previous actions when choosing parameters.
        """
        return await super().execute_milestone_step(messages)

    # ===== Actions =====

    # TODO COMBINE:
    @action
    async def find_tables(self, offset: int = 0, limit: int = 5, table_patterns: list[str] | None = None) -> SQLQueryOutputs:
        """Find tables in the database. If no patterns provided, will search all tables."""
        if table_patterns is None:
            content = [f"SELECT table_name FROM information_schema.tables LIMIT {limit} OFFSET {offset}"]
        else:
            table_patterns = [pattern.replace("*", "%") if "*" in pattern or "%" in pattern else f"%{pattern}%" for pattern in table_patterns]
            content = [
                f"SELECT table_name FROM information_schema.tables WHERE table_name ILIKE '{pattern}' LIMIT {limit} OFFSET {offset}" for pattern in table_patterns
            ]
        queries = SQLQueries(reasoning=f"Discovering schema for patterns: {table_patterns}", content=content)
        return await self.execute_multiple_queries(queries)

    @action
    async def examine_table(self, table_names: list[str]) -> SQLQueryOutputs:
        """Examine specific table structures and sample data to discover schema, like column names and types."""
        content = []
        for table_name in table_names:
            content.extend([f"DESCRIBE {table_name}", f"SELECT * FROM {table_name} LIMIT 3", f"SELECT COUNT(*) as row_count FROM {table_name}"])
        queries = SQLQueries(reasoning=f"Examining table structure and sample data for {table_names}", content=content)
        return await self.execute_multiple_queries(queries)

    @action
    async def execute_multiple_queries(self, queries: SQLQueries) -> SQLQueryOutputs:
        """Execute SQL query(s)."""
        self.logger.info(f"Executing {len(queries.content)} SQL queries")

        # Execute queries and track outputs
        outputs = []
        for i, sql_query in enumerate(queries.content):
            try:
                sql_query_content = sql_query.content
                for _ in range(3):  # Retry up to 3 times
                    try:
                        df = await self.source.execute_async(sql_query=sql_query_content)
                        break  # Exit loop if successful
                    except Exception as e:
                        original_sql_query_content = sql_query_content
                        sql_query_content = self._apply_line_changes(sql_query_content, e)
                        self.logger.warning(f"Retried {original_sql_query_content} due to error: {e}. New query: {sql_query_content}")
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
        return SQLQueryOutputs(content=outputs)
