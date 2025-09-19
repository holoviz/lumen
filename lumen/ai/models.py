from typing import Literal

from instructor.dsl.partial import PartialLiteralMixin
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from pydantic.json_schema import SkipJsonSchema

from .config import SOURCE_TABLE_SEPARATOR, MissingContextError


class PartialBaseModel(BaseModel, PartialLiteralMixin):
    ...


class EscapeBaseModel(PartialBaseModel):

    insufficient_context_reason: str = Field(
        description="If lacking sufficient context, explain why; else use ''. Do not base off the user query; only from the data context provided.",
        examples=[
            "A timeseries is requested but SQL only provides customer and order data; please include a time dimension",
            "The previous result is one aggregated value; try a different aggregation or more dimensions",
            ""
        ]
    )

    insufficient_context: bool = Field(
        description="True if lacking context, else False. If True, leave other fields empty.",
    )

    def model_post_init(self, __context):
        """
        After model initialization, check if insufficient_context. If it is,
        raise a MissingContextError with the provided explanation to stop further processing.
        """
        if self.insufficient_context:
            raise MissingContextError(self.insufficient_context_reason)


class YesNo(BaseModel):

    yes: bool = Field(description="True if yes, otherwise False.")


class ThinkingYesNo(BaseModel):

    chain_of_thought: str = Field(
        description="Explain your reasoning as to why you will be answering yes or no.")

    yes: bool = Field(description="True if yes, otherwise False.")


# Essential Discovery Toolkit

class SampleQuery(PartialBaseModel):
    """See actual data content - reveals format issues and patterns."""

    query: SkipJsonSchema[str] = Field(default="SELECT * FROM {table} LIMIT 5")
    table: str = Field(description="Table to sample")


class DistinctQuery(PartialBaseModel):
    """Universal column analysis with optional pattern matching - handles join keys, categories, date ranges."""

    query: SkipJsonSchema[str] = Field(default="SELECT DISTINCT {column} FROM {table} WHERE {column} ILIKE '%{pattern}%' LIMIT 10 OFFSET {offset}")
    table: str = Field(description="Table to query")
    column: str = Field(description="Column to analyze unique values")
    pattern: str = Field(default="", description="Optional pattern to search for (e.g., 'chin' for China/Chinese variations). Leave empty for all distinct values.")
    offset: int = Field(default=0, description="Number of distinct values to skip (0 for initial, 10 for follow-up)")

    def model_post_init(self, __context):
        """Adjust query template based on whether pattern is provided."""
        if not self.pattern.strip():
            # No pattern - get all distinct values
            self.query = "SELECT DISTINCT {column} FROM {table} LIMIT 10 OFFSET {offset}"
        else:
            # Pattern provided - use ILIKE for case-insensitive partial matching
            self.query = "SELECT DISTINCT {column} FROM {table} WHERE {column} ILIKE '%{pattern}%' LIMIT 10 OFFSET {offset}"


class DiscoveryQueries(PartialBaseModel):
    """LLM selects 2-4 essential discoveries using the core toolkit."""

    reasoning: str = Field(description="Brief discovery strategy")
    queries: list[SampleQuery | DistinctQuery] = Field(
        description="Choose 2-4 discovery queries from the Essential Three toolkit"
    )


class DiscoverySufficiency(PartialBaseModel):
    """Evaluate if discovery results provide enough context to fix the original error."""

    reasoning: str = Field(description="Analyze discovery results - do they reveal the cause of the error?")

    sufficient: bool = Field(description="True if discoveries provide enough context to fix the error")

    follow_up_needed: list[SampleQuery | DistinctQuery] = Field(
        default_factory=list,
        max_items=3,
        description="If insufficient, specify 1-3 follow-up discoveries (e.g., OFFSET for more values)"
    )


class SqlQuery(PartialBaseModel):
    """A single SQL query with its associated metadata."""

    query: str = Field(description="""
        One, correct, valid SQL query that answers the user's question;
        should only be one query and do NOT add extraneous comments; no multiple semicolons""")

    expr_slug: str = Field(
        description="""
        Provide a unique, descriptive slug for the SQL expression that clearly indicates the key transformations and source tables involved.
        Include 1 or 2 elements of data lineage in the slug, such as the main transformation and original table names,
        e.g. top_5_athletes_in_2020 or distinct_years_from_wx_table.
        Ensure the slug does not duplicate any existing table names or slugs.
        """
    )


class SqlQueries(PartialBaseModel):
    """Multiple SQL queries to execute in sequence."""

    queries: list[SqlQuery] = Field(
        default_factory=list,
        description="""List of SQL queries to execute."""
    )


def make_sql_model(is_final: bool = False):
    if is_final:
        return SqlQuery
    else:
        return SqlQueries








class VegaLiteSpec(EscapeBaseModel):

    chain_of_thought: str = Field(
        description="Explain how you will use the data to create a vegalite plot, and address any previous issues you encountered."
    )

    yaml_spec: str = Field(description="A vega-lite YAML specification based on the user input and chain of thought. Do not include description")


class LineChange(BaseModel):
    line_no: int = Field(description="The line number in the original text that needs to be changed.")
    replacement: str = Field(description="""
        The new line that replaces the original line, and if applicable, ensuring the indentation matches the original.
        To remove a line set this to an empty string."""
    )


class RawStep(BaseModel):
    actor: str
    instruction: str
    title: str


class RawPlan(BaseModel):
    title: str = Field(description="A title that describes this plan, up to three words.")
    steps: list[RawStep] = Field(
        description="""
        A list of steps to perform that will solve user query. Each step MUST use a DIFFERENT actor than the previous step.
        Review your plan to ensure this constraint is met.
        """
    )


class Reasoning(BaseModel):
    chain_of_thought: str = Field(
        description="""
        Briefly summarize the user's goal and categorize the question type:
        high-level, data-focused, or other. Identify the most relevant and compatible actors,
        explaining their requirements, and what you already have satisfied. If there were previous failures, discuss them.
        IMPORTANT: Ensure no consecutive steps use the same actor in your planned sequence.
        """
    )

class RetrySpec(BaseModel):
    """Represents a revision of text with its content and changes."""

    chain_of_thought: str = Field(description="In a sentence or two, explain the plan to revise the text based on the feedback provided.")
    lines_changes: list[LineChange] = Field(description="A list of changes made to the lines in the original text based on the chain_of_thought.")


def make_plan_model(agents: list[str], tools: list[str]) -> type[RawPlan]:
    # TODO: make this inherit from PartialBaseModel
    step = create_model(
        "Step",
        actor=(Literal[tuple(agents+tools)], FieldInfo(description="The name of the actor to assign a task to.")),
        instruction=(str, FieldInfo(description="Instructions to the actor to assist in the task, and whether rendering is required.")),
        title=(str, FieldInfo(description="Short title of the task to be performed; up to three words.")),
    )
    plan = create_model(
        "Plan",
        title=(
            str, FieldInfo(description="A title that describes this plan, up to three words.")
        ),
        steps=(
            list[step],
            FieldInfo(
                description="""
                A list of steps to perform that will solve user query. Each step MUST use a DIFFERENT actor than the previous step.
                Review your plan to ensure this constraint is met.
                """
            )
        )
    )
    return plan


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


class QueryCompletionValidation(PartialBaseModel):
    """Validation of whether the executed plan answered the user's query"""

    chain_of_thought: str = Field(
        description="Restate intent and results succinctly; then explain your reasoning as to why you will be answering yes or no.")

    missing_elements: list[str] = Field(
        default_factory=list,
        description="List of specific elements from the user's query that weren't addressed"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for additional steps that could complete the query if not fully answered"
    )
    correct: bool = Field(description="True if query correctly solves user request, otherwise False.")


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
