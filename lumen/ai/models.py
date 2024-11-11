from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo


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



def make_plan_models(agent_names: list[str], tables: list[str]):
    step = create_model(
        "Step",
        expert=(Literal[agent_names], FieldInfo(description="The name of the expert to assign a task to.")),
        instruction=(str, FieldInfo(description="Instructions to the expert to assist in the task, and whether rendering is required.")),
        title=(str, FieldInfo(description="Short title of the task to be performed; up to six words.")),
        render_output=(bool, FieldInfo(description="Whether the output of the expert should be rendered. If the user wants to see the table, and the expert is SQL, then this should be `True`.")),
    )
    extras = {}
    if tables:
        extras['tables'] = (
            list[Literal[tuple(tables)]],
            FieldInfo(
                description="A list of tables you want to inspect before coming up with a plan."
            )
        )
    reasoning = create_model(
        'Reasoning',
        chain_of_thought=(
            str,
            FieldInfo(
                description="Describe at a high-level how the actions of each expert will solve the user query."
            ),
        ),
        **extras
    )
    plan = create_model(
        "Plan",
        title=(
            str, FieldInfo(description="A title that describes this plan in a few words")
        ),
        steps=(
            list[step],
            FieldInfo(
                description="A list of steps to perform that will solve user query. Ensure you include ALL the steps needed to solve the task, matching the chain of thought."
            )
        )
    )
    return reasoning, plan


def make_agent_model(agent_names: list[str], primary: bool = False):
    if primary:
        description = "The agent that will provide the output the user requested, e.g. a plot or a table. This should be the FINAL step in your chain of thought."
    else:
        description = "The most relevant agent to use."
    return create_model(
        "Agent",
        chain_of_thought=(
            str,
            FieldInfo(
                description="Describe what this agent should do."
            ),
        ),
        agent=(
            Literal[tuple(agent_names)],
            FieldInfo(default=..., description=description)
        ),
    )


def make_table_model(tables):
    table_model = create_model(
        "Table",
        chain_of_thought=(str, FieldInfo(
            description="A concise, one sentence decision-tree-style analysis on choosing a table."
        )),
        relevant_table=(Literal[tables], FieldInfo(
            description="The most relevant table based on the user query; if none are relevant, select the first."
        ))
    )
    return table_model
