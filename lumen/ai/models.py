from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FuzzyTable(BaseModel):

    required: bool = Field(description="Whether the user's query requires looking for a new table.")

    keywords: list[str] = Field(description="The most likely keywords related to a table name that the user might be referring to.")


class DataRequired(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Thoughts on whether the user's query requires data loaded for insights;
        if the user wants to explore a dataset, it's required.
        If only finding a dataset, it's not required.
        """
    )

    data_required: bool = Field(description="Whether the user wants to load a specific dataset; if only searching for one, it's not required.")


class JoinRequired(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Concisely explain whether a table join is required to answer the user's query, or
        if the user is requesting a join or merge using a couple sentences.
        """
    )

    join_required: bool = Field(description="Whether a table join is required to answer the user's query.")


class TableJoins(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Concisely consider the tables that need to be joined to answer the user's query.
        """
    )

    tables: list[str] = Field(
        description=(
            "List of tables that need to be joined to answer the user's query. "
            "Use table names verbatim; e.g. if table is `read_csv('table.csv')` "
            "then use `read_csv('table.csv')` and not `table`, but if the table has "
            "no extension, i.e. `table`, then use only `table`."
        ),
    )


class Sql(BaseModel):

    chain_of_thought: str = Field(
        description="""
        You are a world-class SQL expert, and your fame is on the line so don't mess up.
        Then, think step by step on how you might approach this problem in an optimal way.
        If it's simple, just provide one sentence.
        """
    )

    expr_slug: str = Field(
        description="""
        Give the SQL expression a concise, but descriptive, slug that includes whatever transforms were applied to it.
        """
    )

    query: str = Field(description="Expertly optimized, valid SQL query to be executed; do NOT add extraneous comments.")


class Validity(BaseModel):

    correct_assessment: str = Field(
        description="""
        Restate the current table and think through whether the current table meets the requirement
        to answer the user's query, i.e. table contains all the necessary raw columns.
        However, if the query can be solved through SQL, the data is assumed to be valid.
        If the number of rows is insufficient, the table is invalid.
        If the user user explicitly asks for a refresh, then the table is invalid.
        """
    )

    is_invalid: Literal["table", "sql"] | None = Field(
        description="Whether the `table` or `sql` is invalid or no longer relevant depending on correct assessment. None if valid."
    )


class Topic(BaseModel):

    result: str = Field(description="A word or up-to-three-words phrase that describes the topic of the table.")


class VegaLiteSpec(BaseModel):

    json_spec: str = Field(description="A vega-lite JSON specification WITHOUT the data field, which will be added automatically.")
