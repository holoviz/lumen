from __future__ import annotations

from pydantic import BaseModel, Field


class DataRequired(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Thoughts on whether the user's query requires data loaded;
        if the user wants to explore a dataset, it's required.
        If only finding a dataset, it's not required.
        """
    )

    data_required: bool = Field(description="Whether the user wants to load a specific dataset; if only searching for one, it's not required.")


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
        Focus only on whether the table contain all the necessary columns
        to answer the user's query.
        """
    )

    is_invalid: bool = Field(
        description="Whether one of the table and/or pipeline is invalid or needs a refresh."
    )
