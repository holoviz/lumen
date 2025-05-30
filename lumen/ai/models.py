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


class ThinkingYesNo(YesNo):

    chain_of_thought: str = Field(
        description="Explain concisely how you arrived at your answer in under a sentence, or even a phrase.")


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


class LineChange(BaseModel):
    line_no: int = Field(description="The line number in the original text that needs to be changed.")
    replacement: str = Field(description="""
        The new line that replaces the original line, and if applicable, ensuring the indentation matches the original.
        To remove a line set this to an empty string."""
    )


class RetrySpec(BaseModel):
    """Represents a revision of text with its content and changes."""

    chain_of_thought: str = Field(description="In a sentence or two, explain the plan to revise the text based on the feedback provided.")
    lines_changes: list[LineChange] = Field(description="A list of changes made to the lines in the original text based on the chain_of_thought.")

def make_plan_models(agents: list[str], tools: list[str]):
    # TODO: make this inherit from PartialBaseModel
    step = create_model(
        "Step",
        actor=(Literal[tuple(agents+tools)], FieldInfo(description="The name of the actor to assign a task to.")),
        instruction=(str, FieldInfo(description="Instructions to the actor to assist in the task, and whether rendering is required.")),
        title=(str, FieldInfo(description="Short title of the task to be performed; up to three words.")),
        render_output=(bool, FieldInfo(description="Whether the output of the actor should be rendered. If the user wants to see the table, and the actor is SQL, then this should be `True`.")),
    )
    reasoning = create_model(
        'Reasoning',
        chain_of_thought=(
            str,
            FieldInfo(
                description="""
                    Briefly summarize the user's goal and categorize the question type:
                    high-level, data-focused, or other. Identify the most relevant and compatible actors,
                    explaining their requirements, and what you already have satisfied. If there were previous failures, discuss them.
                    """
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
            the parts of the query that they are relevant to. Select more
            than you need, and then you can filter them out later.
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


class DbtslQueryParams(BaseModel):
    """
    Model for dbtsl.client.query() parameters.
    """

    chain_of_thought: str = Field(
        description="""You are a world-class dbt Semantic Layer expert. Think step by step about
        what metrics are needed, what dimensions to group by, what time granularity
        to use, and any filters that should be applied; if filters are applied, include those
        filtered dimensions in group_by. If there are errors, mention how you'll address the errors.
        """
    )

    expr_slug: str = Field(
        description="""Give the query a concise, but descriptive, slug that includes the metrics
        and dimensions used, e.g. monthly_revenue_by_region. The slug must be unique."""
    )

    where: list[str] = Field(
        default_factory=list,
        description="A list of conditions to filter the results; dimensions referenced here must also be in group_by, e.g. ['metric_time__month >= date_trunc('month', '2024-09-30'::date)']"
    )

    group_by: list[str] = Field(
        default_factory=list,
        description="A list of dimensions to group by, e.g. ['metric_time__month'], must include dimensions from where."
    )

    limit: int = Field(
        default=None,
        description="The maximum number of rows to return."
    )

    metrics: list[str] = Field(
        default_factory=list,
        description="A list of metrics to include in the query, e.g. ['revenue']"
    )

    order_by: list[str] = Field(
        default_factory=list,
        description="A list of columns or expressions to order the results by, e.g. ['metric_time__month']"
    )


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
