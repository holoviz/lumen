from __future__ import annotations

from typing import Literal

from instructor.dsl.partial import PartialLiteralMixin
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from .config import SOURCE_TABLE_SEPARATOR


class PartialBaseModel(BaseModel, PartialLiteralMixin):
    ...


class YesNo(BaseModel):

    yes: bool = Field(description="True if yes, otherwise False.")


class Sql(BaseModel):

    chain_of_thought: str = Field(
        description="""
        You are a world-class SQL expert, and your fame is on the line so don't mess up.
        If it's simple, give at max one sentence about using `SELECT * FROM ...`.
        However, if applicable, be sure to carefully study the schema, discuss the values in the columns,
        and whether you need to wrangle the data before you can use it, before finally writing a correct and valid
        SQL query that fully answers the user's query. If using CTEs, comment on the purpose of each.
        Everything should be made as simple as possible, but no simpler and be concise.
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
        description="Explain how you will use the data to create a vegalite plot, and address any previous issues you encountered."
    )

    json_spec: str = Field(description="A vega-lite JSON specification based on the user input and chain of thought. Do not include description")



class RetrySpec(BaseModel):

    chain_of_thought: str = Field(
        description="Explain why the previous spec failed to address the user query and what you will do differently this time to ensure it is correct."
    )

    corrected_spec: str = Field(
        description="The corrected version of the previous spec without any additional comments, additions or code examples."
    )


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
                description="Describe at a high-level how the actions of each expert will solve the user query, and whether the current columns is sufficient to answer the query."
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


def make_columns_selection(table_slugs: list[str], **context):

    class TableColumnsIndices(PartialBaseModel):

        table_slug: Literal[tuple(table_slugs)] = Field(
            description=f"The table slug, i.e. '<source>{SOURCE_TABLE_SEPARATOR}<table>'"
        )

        column_indices: list[int] = Field(
            description="A list of 0-based column indices for the table specified by `table_slug`. This indicates which columns should be included in the output."
        )


    class ColumnsSelection(PartialBaseModel):
        """
        Model for selecting a subset of columns from tables.
        """
        chain_of_thought: str = Field(
            description="""
            Break down the user query into parts, and try to map out the columns to
            the parts of the query that they are relevant to.
            """
        )

        tables_columns_indices: list[TableColumnsIndices] = Field(
            description="""
            The list of table slugs, i.e. '<source>{SOURCE_TABLE_SEPARATOR}<table>',
            and their respective columns based on your own chain_of_thought."""
        )
    return ColumnsSelection


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


def make_find_tables_model(tables):
    table_model = create_model(
        "Table",
        chain_of_thought=(str, FieldInfo(
            description="""
            Concisely consider which tables are necessary to answer the user query.
            """
        )),
        selected_tables=(list[Literal[tuple(tables)]], FieldInfo(
            description="""
            The most relevant tables based on the user query; if none are relevant,
            use the first table. At least one table must be provided.
            If a join is necessary, include all the tables that will be used in the join.
            """
        )),
       potential_join_issues=(str, FieldInfo(
           description="""
           If no join is necessary, return an empty string--else
           list potential join issues between tables:
           - Data type mismatches (int vs string, numeric precision)
           - Format differences (case, leading zeros, dates/times, timezones)
           - Semantic differences (IDs vs names, codes vs full text)
           - Quality issues (nulls, duplicates, validation rules)
           Return specific issues found in current tables, and how you plan to address them
           in the most easiest, but accurate way possible.
           """
       )),
        __base__=PartialBaseModel
    )
    return table_model


def make_iterative_selection_model(table_slugs):
    """
    Creates a model for table selection in the coordinator.
    This is separate from the SQLAgent's table selection model to focus on context gathering
    rather than query execution.
    """
    table_model = create_model(
        "IterativeTableSelection",
        chain_of_thought=(str, FieldInfo(
            description="""
            Consider which tables would provide the most relevant context for understanding the user query,
            and be specific about what columns are most relevant to the query. Please be concise.
            """
        )),
        is_done=(bool, FieldInfo(
            description="""
            Indicate whether you have sufficient context about the available data structures.
            Set to True if you understand enough about the data model to proceed with the query.
            Set to False if you need to examine additional tables to better understand the domain.
            Do not reach for perfection; if there are promising tables, but you are not 100% sure,
            still mark as done.
            """
        )),
        requires_join=(bool, FieldInfo(
            description="""
            Indicate whether a join is necessary to answer the user query.
            Set to True if a join is necessary to answer the query.
            Set to False if no join is necessary.
            """
        )),
        selected_slugs=(list[Literal[tuple(table_slugs)]], FieldInfo(
            description="""
            If is_done and a join is necessary, return a list of table names that will be used in the join.
            If is_done and no join is necessary, return a list of a single table name.
            If not is_done return a list of table names that you believe would provide
            the most relevant context for understanding the user query that wasn't already seen before.
            Focus on tables that contain key entities, relationships, or metrics mentioned in the query.
            """
        )),
        __base__=PartialBaseModel
    )
    return table_model


def make_refined_query_model(item_type_name: str = "items"):
    """
    Creates a model for refining search queries in vector lookup tools.
    """
    return create_model(
        "RefinedQuery",
        chain_of_thought=(str, Field(
            description=f"""
            Analyze the current search results for {item_type_name}. Consider whether the terms used
            in the query match the terms that might be found in {item_type_name} names and descriptions.
            Think about more specific or alternative terms that might yield better results.
            """
        )),
        refined_search_query=(str, Field(
            description=f"""
            A refined search query that would help find more relevant {item_type_name}.
            This should be a focused query with specific terms, entities, or concepts
            that might appear in relevant {item_type_name}.
            """
        )),
        __base__=PartialBaseModel
    )
