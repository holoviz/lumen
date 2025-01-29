from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo


class JoinRequired(BaseModel):

    chain_of_thought: str = Field(
        description="""
        Concisely explain whether a table join is required to answer the user's query, or
        if the user is requesting a join or merge.
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
        If it's simple, just provide one step. However, if applicable, be sure to carefully
        study the schema, discuss the values in the columns, and whether you need to
        wrangle the data before you can use it, before finally writing a correct and valid
        SQL query that fully answers the user's query. If using CTEs, comment on the purpose of each.
        Everything should be made as simple as possible, but no simpler.
        """
    )

    expr_slug: str = Field(
        description="""
        Give the SQL expression a concise, but descriptive, slug that includes whatever transforms were applied to it,
        e.g. top_5_athletes. The slug must be unique, i.e. should not match other existing table names or slugs.
        """
    )

    query: str = Field(description="""
        Correct, valid SQL query that answers the user's query and is based on
        the chain of thought; do NOT add extraneous comments.""")


class VegaLiteSpec(BaseModel):

    chain_of_thought: str = Field(
        description="Explain how you will use the data to create a vegalite plot."
    )

    json_spec: str = Field(description="A vega-lite JSON specification. Do not under any circumstances generate the data field.")



class RetrySpec(BaseModel):

    chain_of_thought: str = Field(
        description="Explain why the previous spec failed to address the user query."
    )

    corrected_spec: str = Field(
        description="The corrected version of the previous spec without any additional comments, additions or code examples."
    )


def make_context_model(tools: list[str], tables: list[str]):
    fields = {}
    if tables:
        fields['tables'] = (
            list[Literal[tuple(tables)]],
            FieldInfo(
                description="A list of the most relevant tables to explore and load into memory before coming up with a plan. Choose at most three tables. NOTE: Simple queries asking to list the tables/datasets do not require loading the tables. Table names MUST match verbatim including the quotations, apostrophes, periods, or lack thereof."
            )
        )
    if tools:
        tool = create_model(
            "Tool",
            name=(Literal[tuple(tools)], FieldInfo(description="The name of the tool.")),
            instruction=(str, FieldInfo(description="Instructions for the tool.")),
        )
        fields['tools'] = tools=(
            list[tool],
            FieldInfo(
                description="A list of tools to call to provide context before launching into the planning stage. Use tools to gather additional context or clarification, tools should NEVER be used to obtain the actual data you will be working with."
            )
        )
    return create_model("Context", **fields)


def make_plan_models(agents: list[str], tools: list[str]):
    step = create_model(
        "Step",
        expert_or_tool=(Literal[tuple(agents+tools)], FieldInfo(description="The name of the expert or tool to assign a task to.")),
        instruction=(str, FieldInfo(description="Instructions to the expert to assist in the task, and whether rendering is required.")),
        title=(str, FieldInfo(description="Short title of the task to be performed; up to three words.")),
        render_output=(bool, FieldInfo(description="Whether the output of the expert should be rendered. If the user wants to see the table, and the expert is SQL, then this should be `True`.")),
    )
    reasoning = create_model(
        'Reasoning',
        chain_of_thought=(
            str,
            FieldInfo(
                description="Describe at a high-level how the actions of each expert will solve the user query."
            ),
        ),
    )
    plan = create_model(
        "Plan",
        title=(
            str, FieldInfo(description="A title that describes this plan, up to three words.")
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
        agent_or_tool=(
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
