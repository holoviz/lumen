"""
SQL Agent Response Models
"""

import pandas as pd

from pydantic import (
    BaseModel, Field, field_serializer, field_validator,
)
from pydantic.json_schema import SkipJsonSchema

from ..core.models import Artifact, PartialBaseModel


class SQLQuery(PartialBaseModel):
    """Represents a SQL query"""

    table_name: str = Field(default="", description="Name for the resulting table if the query should be stored.")
    content: str = Field(description="The SQL query code to execute")


class SQLQueries(PartialBaseModel):
    """Represents query(s) to execute"""

    reasoning: str = Field(description="In a sentence or two, explain the SQL queries.")
    content: list[SQLQuery] = Field(description="List of SQL queries to execute in order based on the reasoning.")

    @field_validator("content", mode="before")
    @classmethod
    def convert_to_sql_query(cls, content: list[str]) -> list[SQLQuery]:
        """Convert a list of strings to a list of SQLQuery objects."""
        if isinstance(content, list) and all(isinstance(item, str) for item in content):
            return [SQLQuery(content=item) for item in content]
        return content

    def __str__(self) -> str:
        """String representation showing actual SQL queries"""
        lines = ["SQL queries:"]
        for query in self.content:
            lines.append(f"  - {query.content}")
        return "\n".join(lines)


# TODO: combine with SQLQuery
class SQLQueryOutput(PartialBaseModel):
    """Represents a single SQL query output"""

    table_name: str = Field(default="", description="Name of the resulting table from the SQL query")
    content: pd.DataFrame = Field(description="The output of the SQL query as a DataFrame")

    # Set programmatically after execution
    query: SkipJsonSchema[str] = ""

    class Config:
        arbitrary_types_allowed = True

    @field_serializer("content")
    def serialize_dataframe(self, df: pd.DataFrame) -> list[dict]:
        return df.to_dict(orient="records")


class SQLQueryOutputs(PartialBaseModel):
    """Represents multiple SQL query outputs"""

    content: list[SQLQueryOutput] = Field(description="A mapping table names to their corresponding SQL query outputs")

    def __str__(self) -> str:
        """String representation showing query results"""
        lines = ["Query results:"]
        for i, result in enumerate(self.content):
            df = result.content
            # Show query info
            if result.table_name:
                lines.append(f"  - {len(df)} rows returned for table '{result.table_name}'")
            elif hasattr(result, 'query') and result.query:
                # Truncate long queries for readability
                query_preview = result.query if len(result.query) < 100 else result.query[:97] + "..."
                lines.append(f"  - {len(df)} rows returned from: `{query_preview}`")
            else:
                lines.append(f"  - {len(df)} rows returned")

            # Include column names if it's schema discovery
            if len(df) > 0:
                if "column_name" in df.columns:
                    columns = df["column_name"].tolist()
                    lines.append(f"    Columns: {', '.join(columns[:5])}")
                    if len(columns) > 5:
                        lines.append(f"    ... and {len(columns) - 5} more columns")
                elif "table_name" in df.columns:
                    tables = df["table_name"].tolist()
                    lines.append(f"    Tables: {', '.join(tables[:5])}")
                    if len(tables) > 5:
                        lines.append(f"    ... and {len(tables) - 5} more tables")
                # For aggregated results, show sample
                elif len(df) < 10:
                    lines.append(f"    Data: {df.to_string()}")

        return "\n".join(lines)


class LineChange(BaseModel):
    """Represents a single line change in a SQL query"""

    replacement: str = Field(
        description="""
        The new line that replaces the original line, and if applicable, ensuring the indentation matches the original.
        To remove a line set this to an empty string."""
    )
    line_no: int = Field(description="The line number in the original text that needs to be changed.")


class LineChanges(PartialBaseModel):
    """Represents a revision of text with its content and changes."""

    reasoning: str = Field(description="In a sentence or two, explain the plan to revise the text based on the feedback provided.")
    content: list[LineChange] = Field(description="A list of changes made to the lines in the original text based on the reasoning.")


class SQLResultData(PartialBaseModel):
    """Represents the final SQL query result data"""

    query: str = Field(description="The final SQL query that was executed")
    results: list[dict] = Field(description="The query results as a list of dictionaries")
    row_count: int = Field(description="Number of rows returned")
    columns: list[str] = Field(description="Column names in the result")

    def to_artifact(self, key: str, description: str) -> Artifact:
        """Convert to an artifact for storage"""
        return Artifact(
            key=key,
            value=f"Query: {self.query}\nResults: {len(self.results)} rows with columns: {', '.join(self.columns)}",
            description=description
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of the results"""
        if not self.results:
            return "No results found"

        # Format first few results for display
        summary_lines = [f"Found {self.row_count} rows:"]
        for i, row in enumerate(self.results[:5]):
            row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
            summary_lines.append(f"{i+1}. {row_str}")

        if len(self.results) > 5:
            summary_lines.append(f"... and {len(self.results) - 5} more rows")

        return "\n".join(summary_lines)
