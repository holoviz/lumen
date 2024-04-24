from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Table(BaseModel):

    table: str = Field(description="The full path name")


class Sql(BaseModel):

    query: str = Field(description="SQL query to be executed")


class Yaml(BaseModel):

    spec: str = Field(description="Lumen spec YAML to reflect user query")


class Validity(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Focus only on whether the table and/or pipeline contain all the necessary columns
        to answer the user's query.

        Based on the latest user's query, is the table relevant?
        If not, return invalid_key='table'.

        Then for pipeline, focus only on the columns, and **disregard** everything else,
        like whether the pipeline contains visualization or not.
        For example, if it's "visualize this", then it's still valid.
        However, if the pipeline contains `MAX(t_cap) as turbine_capacity`, and the user
        asks to plot a map of `lat and lon`, then it's invalid!
        Please only provide up to two sentences at most; be concise and to the point.
        """
    )

    is_invalid: bool = Field(
        description="Whether one of the table and/or pipeline is invalid."
    )

    invalid_key: Literal["pipeline", "table"] | None = Field(
        default=None,
        description="Whether a pipeline or table is invalid. None if valid."
    )
