from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo


class JoinRequired(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Concisely explain whether a table join is required to answer the user's query, or
        if the user is requesting a join or merge using a couple sentences.
        """
    )

    requires_joins: bool = Field(description="Whether a table join is required to answer the user's query.")


class TableJoins(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Concisely consider the tables that need to be joined to answer the user's query.
        """
    )

    tables_to_join: list[str] = Field(
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
        Give the SQL expression a concise, but descriptive, slug that includes whatever transforms were applied to it,
        e.g. top_5_athletes_gold_medals
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


class VegaLiteSpec(BaseModel):

    json_spec: str = Field(description="A vega-lite JSON specification. Do not under any circumstances generate the data field.")


def make_plan_models(agent_names: list[str], tables: list[str]):
    step = create_model(
        "Step",
        expert=(Literal[agent_names], FieldInfo(description="The name of the expert to assign a task to.")),
        instruction=(str, FieldInfo(description="Instructions to the expert to assist in the task, and whether rendering is required.")),
        title=(str, FieldInfo(description="Short title of the task to be performed; up to three words.")),
        render_output=(bool, FieldInfo(description="Whether the output of the expert should be rendered. If the user wants to see the table, and the expert is SQL, then this should be `True`.")),
    )
    extras = {}
    if tables:
        extras['tables'] = (
            list[Literal[tuple(tables)]],
            FieldInfo(
                description="A list of tables to load into memory before coming up with a plan. NOTE: Simple queries asking to list the tables/datasets do not require loading the tables. Table names MUST match verbatim including the quotations, apostrophes, periods, or lack thereof."
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
        relevant_table=(Literal[tuple(tables)], FieldInfo(
            description="The most relevant table based on the user query; if none are relevant, select the first. Table names MUST match verbatim including the quotations, apostrophes, periods, or lack thereof."
        ))
    )
    return table_model
