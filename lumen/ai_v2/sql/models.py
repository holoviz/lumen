"""
SQL Agent Response Models
"""

from pydantic import BaseModel, Field

from ..core.models import PartialBaseModel, SpecFragment


class SQLQuery(PartialBaseModel):
    """Response model for a single SQL query"""

    description: str = Field(description="What this query does")
    query: str = Field(description="The SQL query")
    output_table: str = Field(description="Name for the resulting table")


class SQLQueries(PartialBaseModel):
    """Response model for multiple SQL queries to execute together"""

    chain_of_thought: str = Field(description="Reasoning behind these queries")
    queries: list[SQLQuery] = Field(description="List of SQL queries to execute in order")
    batch_description: str = Field(description="Overall description of what this batch accomplishes")
    should_materialize: list[bool] = Field(description="Whether each query result should be materialized for later use")


class LineChange(BaseModel):
    """Represents a single line change in a SQL query"""

    replacement: str = Field(
        description="""
        The new line that replaces the original line, and if applicable, ensuring the indentation matches the original.
        To remove a line set this to an empty string."""
    )
    line_no: int = Field(description="The line number in the original text that needs to be changed.")


class RetrySpec(PartialBaseModel):
    """Represents a revision of text with its content and changes."""

    chain_of_thought: str = Field(description="In a sentence or two, explain the plan to revise the text based on the feedback provided.")
    lines_changes: list[LineChange] = Field(description="A list of changes made to the lines in the original text based on the chain_of_thought.")


# ===== Action Input Models =====


class QueryEditRequest(PartialBaseModel):
    """Request to edit a SQL query"""

    original_query: str = Field(description="The original SQL query to edit")
    feedback: str = Field(description="Feedback on what needs to be changed")


class QueryBatchRequest(PartialBaseModel):
    """Request to generate and execute multiple SQL queries"""

    instruction: str = Field(description="Natural language instruction for what the queries should accomplish")
    goal: str = Field(default="", description="The overall goal or objective")
    context: str = Field(default="", description="Relevant context or requirements")


class QueryExecutionRequest(PartialBaseModel):
    """Request to execute a specific SQL query"""

    query: str = Field(description="The SQL query to execute")
    output_table: str = Field(description="Name for the resulting table")
    materialize: bool = Field(default=False, description="Whether to materialize the result for later use")


class QueryBatchExecutionRequest(PartialBaseModel):
    """Request to execute multiple SQL queries in sequence"""

    batch_description: str = Field(default="", description="Description of what this batch accomplishes")
    queries: list[QueryExecutionRequest] = Field(description="List of queries to execute")


# ===== SQL Fragment Models =====


class SQLQueriesFragment(SpecFragment):
    """Fragment for SQL queries generation"""

    def __str__(self) -> str:
        queries = self.content.get("queries", [])
        batch_desc = self.content.get("batch_description", "SQL queries batch")
        return f"{batch_desc}: {len(queries)} queries generated"


class QueryExecutionFragment(SpecFragment):
    """Fragment for query execution results"""

    def __str__(self) -> str:
        if self.content.get("executed_queries"):
            # Multi-query execution
            executed_queries = self.content.get("executed_queries", [])
            success_count = self.content.get("success_count", 0)
            total_queries = len(executed_queries)

            # Check for schema discovery
            schema_count = sum(1 for q in executed_queries if q.get("schema_results"))
            if schema_count > 0:
                return f"Executed {total_queries} queries ({schema_count} schema, {success_count} successful)"
            else:
                return f"Executed {total_queries} queries ({success_count} successful)"
        else:
            # Single query execution
            status = self.content.get("status", "unknown")
            table = self.content.get("table", "")
            return f"Query executed: {table} ({status})"


class SchemaDiscoveryFragment(SpecFragment):
    """Fragment for schema discovery results"""

    def __str__(self) -> str:
        tables_count = len(self.content.get("tables", []))
        database = self.content.get("database", "database")
        return f"Discovered {tables_count} tables in {database}"
