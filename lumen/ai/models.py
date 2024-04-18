from pydantic import BaseModel, Field


class Table(BaseModel):

    table: str = Field(description="The full path name")


class Sql(BaseModel):

    query: str = Field(description="SQL query to be executed")


class Yaml(BaseModel):

    spec: str = Field(description="Lumen spec YAML to reflect user query")


class PipelineValidity(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Focus only on whether the pipeline contain all the necessary columns
        to answer the user's query? FOCUS only on the columns, and **disregard** everything else,
        like whether the pipeline contains visualization or not.
        For example, if it's "visualize this", then it's still valid.
        However, if the pipeline contains `MAX(t_cap) as turbine_capacity`, and the user
        asks to plot a map of `lat and lon`, then it's invalid!
        Please only provide up to two sentences at most; be concise and to the point.
        """
    )

    valid: bool = Field(
        default=...,
        description="Whether a transformation is required to achieve the user's request"
    )
