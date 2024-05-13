from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FuzzyTable(BaseModel):

    keywords: list[str] = Field(description="The table keywords from the user query.")


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
        Focus only on whether the table and/or / SQL contain all the necessary columns
        to answer the user's query.

        Based on the latest user's query, is the table relevant?
        If not, return invalid_key='table'.

        Then, do all the necessary columns exist?
        If not, return invalid_key='sql'.
        """
    )

    is_invalid: bool = Field(
        description="Whether one of the table and/or pipeline is invalid or needs a refresh."
    )

    invalid_key: Literal["table", "sql"] | None = Field(
        default=None,
        description="Whether a table or sql is invalid. None if valid."
    )
