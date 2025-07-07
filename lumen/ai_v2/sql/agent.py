"""
SQL Agent Implementation
"""

import asyncio

import param

from ...sources.base import BaseSQLSource
from ..core.agent import BaseAgent, action
from .models import (
    LineChanges, LLMMessages, SQLQueries, SQLQueryResult, SQLQueryResults,
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
    async def execute_multiple_queries(self, queries: SQLQueries) -> SQLQueryResults:
        """Execute SQL query(s)."""
        df_list = await asyncio.gather(*(self.source.execute_async(sql_query=sql_query.content) for sql_query in queries.content))
        content = [SQLQueryResult(content=df, table_name=sql_query.table_name) for df, sql_query in zip(df_list, queries.content, strict=False)]
        return SQLQueryResults(content=content)

    @action
    async def identify_line_changes(self, query: str, feedback: str, messages: LLMMessages | None = None) -> LineChanges:
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
