from pydantic import BaseModel, Field


class String(BaseModel):

    output: str


class Table(BaseModel):

    table: str = Field(
        description="The full path name"
    )


class Sql(BaseModel):

    query: str = Field(description="SQL query to be executed")


class Yaml(BaseModel):

    spec: str = Field(description="Lumen spec YAML to reflect user query")
