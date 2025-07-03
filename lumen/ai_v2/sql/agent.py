"""
SQL Agent Implementation
"""

import param

from ...sources.base import BaseSQLSource
from ..core.agent import BaseAgent, action
from ..core.models import ExecutionContext, Roadmap, SpecFragment
from .models import (
    QueryBatchExecutionRequest, QueryBatchRequest, QueryEditRequest, RetrySpec,
    SQLQueries,
)


class SQLAgent(BaseAgent):
    """Agent that generates and executes SQL queries

    Purpose:
        Specialized agent for SQL query generation and database schema exploration.
        Uses hierarchical planning with efficient milestone execution for SQL goals.
        Works with Lumen Source objects to understand available tables and generate
        appropriate SQL queries based on natural language instructions.

    Capabilities:
        - Generate SQL queries for data retrieval
        - Edit SQL queries line by line for refinement
        - Execute SQL queries against data sources
        - Filter and aggregate data
        - Join tables across schemas
        - Calculate metrics and KPIs
        - Work with any Lumen Source (DuckDB, PostgreSQL, etc.)
        - Efficient schema exploration with batch operations
        - Decisive completion to prevent endless exploration
    """

    name = param.String(default="sql_agent")

    source = param.ClassSelector(class_=BaseSQLSource, doc="Data source for schema and execution")

    # ===== Override Planning Methods for SQL-Specific Logic =====

    async def create_roadmap(self, goal: str) -> Roadmap:
        """Create a high-level roadmap with words to achieve goal through SQL queries.

        Tables: {{ tables }}

        For example, create 2-3 concrete milestones:
        - [ ] Find relevant tables
        - [ ] Explore database schema (if needed)
        - [ ] Execute query to find customer purchases

        The actions can process multiple SQL queries at once so the roadmap
        should focus on high-level milestones rather than individual queries.
        """
        # Get available tables
        available_tables = self.source.get_tables() if self.source else []

        system_prompt = self._render_prompt("create_roadmap", tables=available_tables)
        messages = [{"role": "user", "content": f"Create SQL roadmap for: {goal}"}]
        return await self.llm.invoke(messages, system=system_prompt, response_model=Roadmap)

    # ===== Override Extension Points =====

    def get_completion_message(self, goal: str, context: ExecutionContext) -> str:
        """Get SQL-specific completion message"""
        return "Query generated and executed successfully!"

    # ===== Actions =====

    @action
    async def generate_queries(self, request: QueryBatchRequest) -> SpecFragment:
        """Generate multiple SQL queries for batch execution.

        {% if context %}
        Context: {{ context }}
        {% endif %}

        {% if goal %}
        Goal: {{ goal }}
        {% endif %}

        Generate a series of SQL queries to accomplish the objective. Consider dependencies between queries.
        Use schema exploration (SELECT * FROM table LIMIT 1) if column names are unknown.
        """
        system_prompt = self._render_prompt("generate_queries", context=request.context, goal=request.goal)
        messages = [{"role": "user", "content": request.instruction}]
        response = await self.llm.invoke(messages=messages, system=system_prompt, response_model=SQLQueries, model_spec="sql")

        # Create spec fragment using factory method
        fragment = SpecFragment.create(
            "sql_queries",
            {
                "queries": [
                    {
                        "query": query.query,
                        "description": query.description,
                        "output_table": query.output_table,
                    }
                    for query in response.queries
                ],
                "batch_description": response.batch_description,
                "chain_of_thought": response.chain_of_thought,
                "should_materialize": response.should_materialize,
            },
        )

        return fragment

    @action
    async def execute_queries(self, request: QueryBatchExecutionRequest) -> SpecFragment:
        """Execute multiple SQL queries in sequence."""
        executed_queries = []
        success_count = 0

        for i, query_data in enumerate(request.queries):
            # Determine if this query should be materialized
            should_materialize = (
                query_data.materialize if hasattr(query_data, "materialize") else (request.materialize_flags[i] if i < len(request.materialize_flags) else False)
            )

            try:
                # Create a SQL expression source for execution - this is what can fail
                self.source.create_sql_expr_source({query_data.output_table: query_data.query}, materialize=should_materialize)
                query_result = {
                    "query": query_data.query,
                    "output_table": query_data.output_table,
                    "description": query_data.description,
                    "status": "executed",
                    "materialized": should_materialize,
                }
                executed_queries.append(query_result)
                success_count += 1
            except Exception as e:
                executed_queries.append(
                    {
                        "query": query_data.query,
                        "output_table": query_data.output_table,
                        "description": query_data.description,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        # Create content for the fragment
        content = {
            "executed_queries": executed_queries,
            "success_count": success_count,
            "total_queries": len(request.queries),
            "batch_description": request.batch_description,
            "description": f"Executed {len(request.queries)} queries ({success_count} successful)",
        }

        fragment = SpecFragment.create(
            "query_execution",
            content,
        )

        return fragment

    @action
    async def edit_lines(self, request: QueryEditRequest) -> SpecFragment:
        """Edit SQL query based on feedback: {{ request.feedback }}

        Original query:
        ```sql
        {{ enumerated_query }}
        ```

        Make minimal line-by-line changes to fix the issues.
        """
        lines = request.original_query.split("\n")
        enumerated_query = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
        system_prompt = self._render_prompt("edit_lines", enumerated_query)
        messages = [{"role": "user", "content": f"Edit the SQL query based on this feedback: {request.feedback}"}]
        response = await self.llm.invoke(messages=messages, system=system_prompt, response_model=RetrySpec, model_spec="reasoning")

        # Apply changes (1-indexed line numbers)
        for change in response.lines_changes:
            if 1 <= change.line_no <= len(lines):
                lines[change.line_no - 1] = change.replacement

        # Reconstruct the query, removing empty lines if they were set to ""
        edited_query = "\n".join(line for line in lines if line is not None)

        # Create spec fragment with the edited query
        fragment = SpecFragment.create(
            "sql_queries",
            {
                "queries": [
                    {
                        "query": edited_query,
                        "description": f"Edited query based on feedback: {request.feedback}",
                        "output_table": "edited_result",
                    }
                ],
                "batch_description": f"Edited query based on feedback: {request.feedback}",
                "chain_of_thought": response.chain_of_thought,
                "should_materialize": [False],
                "changes_made": len(response.lines_changes),
            },
        )

        return fragment
