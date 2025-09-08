from typing import Literal

from instructor.dsl.partial import PartialLiteralMixin
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from .config import SOURCE_TABLE_SEPARATOR


class PartialBaseModel(BaseModel, PartialLiteralMixin):
    ...


class YesNo(BaseModel):

    yes: bool = Field(description="True if yes, otherwise False.")


class ThinkingYesNo(BaseModel):

    chain_of_thought: str = Field(
        description="Explain your reasoning as to why you will be answering yes or no.")

    yes: bool = Field(description="True if yes, otherwise False.")


class SqlQuery(PartialBaseModel):
    """A single SQL query with its associated metadata."""

    query: str = Field(description="""
        One, correct, valid SQL query that answers the user's question without limits unless specified;
        should only be one query and do NOT add extraneous comments; no multiple semicolons""")

    expr_slug: str = Field(
        description="""
        Provide a unique, descriptive slug for the SQL expression that clearly indicates the key transformations and source tables involved.
        Include 1 or 2 elements of data lineage in the slug, such as the main transformation and original table names,
        e.g. top_5_athletes_in_2020 or distinct_years_from_wx_table.
        Ensure the slug does not duplicate any existing table names or slugs.
        """
    )


class Sql(PartialBaseModel):
    """Multiple SQL queries to execute in sequence."""

    queries: list[SqlQuery] = Field(
        default_factory=list,
        description="""
        List of SQL queries to execute. For discovery steps, include multiple queries
        to explore different aspects (e.g., distinct values from different columns).
        For final steps, only one query is allowed.
        """
    )



class SQLRoadmap(PartialBaseModel):
    """High-level execution roadmap for SQL query planning."""

    discovery_steps: list[str] = Field(
        description="""List of discovery steps needed (e.g., 'Check date ranges in both datasets',
        'Find distinct categories', 'Validate join keys'). Each step should be specific and actionable."""
    )

    validation_checks: list[str] = Field(
        default_factory=list,
        description="""Critical validation checks before proceeding (e.g., 'Verify temporal overlap',
        'Confirm key format compatibility'). Focus on compatibility between datasets."""
    )

    join_strategy: str = Field(
        default="",
        description="""If joins are needed, describe the strategy and key relationships.
        Include how to handle temporal or categorical mismatches."""
    )

    potential_issues: list[str] = Field(
        default_factory=list,
        description="""Potential issues to watch for (e.g., 'Limited temporal overlap',
        'Different granularities', 'Format mismatches'). Be specific to the datasets."""
    )

    estimated_steps: int = Field(
        description="Number of steps estimated to complete the query (typically 2-5)."
    )


class NextStep(PartialBaseModel):
    """Represents the next single step to take in SQL exploration."""

    pre_step_validation: str = Field(
        description="""
        If no previous steps, leave empty. Check: 1) Info already available?
        2) Combinable with other discoveries? 3) Directly contributes to answer?
        Brief 1-2 sentence assessment.
        """
    )

    reasoning: str = Field(
        description="""
        Strategic rationale: What gap does this fill? How does it progress toward answer?
        Approach and why? Materialization decision? Which existing materialized tables used?
        Focus on "why" and "how", not validation.
        """
    )

    step_type: Literal["discover", "filter", "join", "final"] = Field(
        description="""
        - "discover": Batch multiple discoveries (LIMIT 10 each) + min/max ranges
        - "filter": Filter rows (LIMIT 100000)
        - "join": Combine tables (LIMIT 100000); explore join keys first
        - "final": Final query (LIMIT 100000)
        """
    )

    action_description: str = Field(
        description="""
        Executable SQL operation: Which tables/materialized views? Which columns?
        What filters/joins/aggregations? Expected output and limit?
        Specific enough for SQL generation, building on previous steps.
        """
    )

    should_materialize: bool = Field(
        description="""
        True if result used in subsequent steps (joins/filtering, limit 100k).
        False for one-off exploratory steps. Materialize only when necessary.
        """
    )

    is_final_answer: bool = Field(
        description="""
        True if this step provides the final answer to user's request.
        E.g., simple "show table" requests or when all info is available.
        """
    )


class ReadinessCheck(PartialBaseModel):
    """Check if we're ready to answer the user's question."""

    reasoning: str = Field(
        description="Explain what information we have and what might still be missing"
    )

    is_ready: bool = Field(
        description="True if we have enough information to write the final query"
    )

    missing_info: list[str] = Field(
        default_factory=list,
        description="List of specific information still needed (if not ready)"
    )


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
