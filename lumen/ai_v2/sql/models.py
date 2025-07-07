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

    content: str = Field(description="The SQL query code to execute")
    table_name: str = Field(description="Name for the resulting table")

    @field_validator("content")
    @classmethod
    def validate_select(cls, v: str) -> str:
        """Ensure query starts with SELECT"""
        if not v.strip().upper().startswith("SELECT"):
            raise ValueError("SQL query must start with 'SELECT'")
        return v


class SQLQueries(PartialBaseModel):
    """Represents query(s) to execute"""

    chain_of_thought: str = Field(description="In a sentence or two, explain the SQL queries.")
    content: list[SQLQuery] = Field(description="List of SQL queries to execute in order based on the chain_of_thought.")


class SQLQueryResult(PartialBaseModel):
    """Represents a single SQL query result"""

    table_name: str = Field(description="Name of the resulting table from the SQL query")
    content: pd.DataFrame = Field(description="The result of the SQL query as a DataFrame")

    class Config:
        arbitrary_types_allowed = True

    @field_serializer('content')
    def serialize_dataframe(self, df: pd.DataFrame) -> list[dict]:
        return df.to_dict(orient='records')


class SQLQueryResults(PartialBaseModel):
    """Represents multiple SQL query results"""

    content: list[SQLQueryResult] = Field(
        description="A dictionary mapping table names to their corresponding SQL query results"
    )

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

    chain_of_thought: str = Field(description="In a sentence or two, explain the plan to revise the text based on the feedback provided.")
    content: list[LineChange] = Field(description="A list of changes made to the lines in the original text based on the chain_of_thought.")


class LLMMessage(PartialBaseModel):
    """Represents a message in OpenAI format"""

    role: str = Field(description="The role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(description="The content of the message")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Ensure role is either 'user' or 'assistant'"""
        if v not in ["user", "assistant"]:
            raise ValueError("Role must be either 'user' or 'assistant'")
        return v


class LLMMessages(PartialBaseModel):
    """Represents a list of messages in OpenAI format"""

    messages: list[LLMMessage] = Field(default_factory=list, description="List of messages in OpenAI format")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[LLMMessage]) -> list[LLMMessage]:
        """Ensure at least one message is provided"""
        if not v:
            raise ValueError("At least one message must be provided")
        return v

    def serialize(self) -> list[dict]:
        """Serialize messages to a list of dictionaries"""
        return [message.model_dump() for message in self.messages]

    def append(self, message: LLMMessage) -> None:
        """Add a new message to the list"""
        self.messages.append(LLMMessage.model_validate(message))

    def pop(self) -> LLMMessage | None:
        """Remove and return the last message from the list"""
        if self.messages:
            return self.messages.pop()
        return None
