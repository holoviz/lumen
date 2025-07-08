"""
SQL Agent Response Models
"""

import pandas as pd

from pydantic import (
    BaseModel, Field, field_serializer, field_validator,
)

from ..core.models import PartialBaseModel


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


class SQLQueryOutput(PartialBaseModel):
    """Represents a single SQL query output"""

    table_name: str = Field(description="Name of the resulting table from the SQL query")
    content: pd.DataFrame = Field(description="The output of the SQL query as a DataFrame")

    class Config:
        arbitrary_types_allowed = True

    @field_serializer("content")
    def serialize_dataframe(self, df: pd.DataFrame) -> list[dict]:
        return df.to_dict(orient="records")


class SQLQueryOutputs(PartialBaseModel):
    """Represents multiple SQL query outputs"""

    content: list[SQLQueryOutput] = Field(description="A dictionary mapping table names to their corresponding SQL query outputs")


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
