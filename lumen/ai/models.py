from __future__ import annotations

from pydantic import BaseModel, Field


class FuzzyTable(BaseModel):

    required: bool = Field(description="Whether the user's query requires looking for a new table.")

    keywords: list[str] | None = Field(default=None, description="The most likely keywords related to a table name that the user might be referring to.")


class DataRequired(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Thoughts on whether the user's query requires data loaded;
        if the user wants to explore a dataset, it's required.
        If only finding a dataset, it's not required.
        """
    )

    data_required: bool = Field(description="Whether the user wants to load a specific dataset; if only searching for one, it's not required.")


class JoinRequired(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Explain whether a table join is required to answer the user's query.
        """
    )

    join_required: bool = Field(description="Whether a table join is required to answer the user's query.")


class TableJoins(BaseModel):

    tables: list[str] | None = Field(
        default=None,
        description="List of tables that need to be joined to answer the user's query.",
    )

class Sql(BaseModel):

    chain_of_thought: str = Field(
        description="""
        You are a world-class SQL expert, and your fame is on the line so don't mess up.
        Then, think step by step on how you might approach this problem in an optimal way.
        If it's simple, just provide one sentence.
        """
    )

    query: str = Field(description="Expertly optimized, valid SQL query to be executed; do NOT add extraneous comments.")


class Validity(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Thoughts on whether the current table meets the requirement
        to answer the user's query, i.e. table contains all necessary columns
        """
    )

    missing_table_or_columns: list[str] = Field(
        description="List out all the tables or columns requested by the user that are currently missing in the spec."
    )

    is_invalid: bool = Field(
        description="Whether the table needs a refresh."
    )


class Topic(BaseModel):

    result: str = Field(description="A word or up-to-three-words phrase that describes the topic of the table.")
