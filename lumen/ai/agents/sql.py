from typing import (
    Annotated, Any, Literal, NotRequired,
)

import param

from panel_material_ui.chat import ChatStep
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from ...pipeline import Pipeline
from ...sources.base import BaseSQLSource, Source
from ...sources.duckdb import DuckDBSource
from ...transforms.sql import SQLLimit
from ..config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from ..context import ContextModel, TContext
from ..editors import LumenEditor, SQLEditor
from ..llm import Message
from ..models import RetrySpec
from ..schemas import Metaset
from ..tools import FunctionTool
from ..utils import (
    clean_sql, describe_data, get_data, get_pipeline, parse_table_slug,
    stream_details,
)
from .base_lumen import BaseLumenAgent


def make_source_table_model(sources: list[tuple[str, str]]):
    class LiteralSourceTable(BaseModel):
        source: Literal[tuple(set(src for src, _ in sources))]
        table: Literal[tuple(set(table for _, table in sources))]
    return LiteralSourceTable


def make_table_model(sources: list[tuple[str, str]]):
    """
    Create a table model with constrained table choices.
    """
    available_tables = list(set(table for _, table in sources))
    TableLiteral = Literal[tuple(available_tables)]  # type: ignore
    return TableLiteral


class SQLQuery(BaseModel):
    """A single SQL query with its associated metadata."""

    query: str = Field(description="""
        One, correct, valid SQL query that answers the user's question;
        should only be one query and do NOT add extraneous comments; no multiple semicolons.""")

    table_slug: str = Field(
        description="""
        A short, unique, descriptive snake_case slug (3-5 words max) describing
        WHAT the resulting data contains. Do NOT include the source table name
        in the slug — provenance is tracked separately.
        Follow the naming style of existing derived tables (shown in the data
        summary with `derived_from`/`step` annotations).
        Examples: avg_sst_by_season, top_5_athletes_2020, revenue_by_region.
        Ensure the slug does not duplicate any existing table names or slugs.
        """
    )


class TableSelection(BaseModel):
    """Select tables relevant to answering the user's query."""

    reasoning: str = Field(description="Brief explanation of why these tables are needed to answer the query.")

    tables: list[str] = Field(
        description="List of table slugs needed to answer the query. Select only tables that are directly relevant."
    )


def make_table_selection_model(available_tables: list[str]):
    """Create a table selection model with constrained table choices."""
    if not available_tables:
        return TableSelection

    TableLiteral = Literal[tuple(available_tables)]  # type: ignore
    return create_model(
        "TableSelectionConstrained",
        tables=(list[TableLiteral], FieldInfo(description="Tables needed to answer the query.")),
        __base__=TableSelection
    )


def make_sql_model(sources: list[tuple[str, str]]):
    """
    Create a SQL query model with source/table validation.

    Parameters
    ----------
    sources : list[tuple[str, str]]
        List of (source_name, table_name) tuples for tables with full schemas.
    """
    # Check if all tables are from a single unique source
    unique_sources = set(src for src, _ in sources)
    if len(unique_sources) == 1:
        Table = make_table_model(sources)
        return create_model(
            "SQLQueryWithTables",
            tables=(
                list[Table],
                FieldInfo(description="The table name(s) referenced in the SQL query.")
            ),
            __base__=SQLQuery
        )

    SourceTable = make_source_table_model(sources)
    return create_model(
        "SQLQueryWithSources",
        tables=(
            list[SourceTable],
            FieldInfo(description="The source and table identifier(s) referenced in the SQL query.")
        ),
        __base__=SQLQuery
    )


async def execute_exploration_sql(
    source: str,
    sql_query: str,
    *,
    sources: dict[tuple[str, str], Source],
) -> str:
    """
    Run a read-only SQL statement on the named Lumen source and return a text preview of results.

    Intended for LLM tool calls: the model supplies the logical ``source`` name and ``sql_query``;
    the ``sources`` map must be provided by the caller (e.g. closed over when building the tool).

    Parameters
    ----------
    source : str
        Name of the data source (key prefix before the source/table separator in slugs).
    sql_query : str
        SQL to execute (SELECT or WITH only).
    sources : dict[tuple[str, str], Source]
        Mapping from ``(source_name, table_name)`` to :class:`~lumen.sources.base.Source` instances.

    Returns
    -------
    str
        Tabular preview, or an error message string if execution fails.
    """
    try:
        base = next(obj for (s, _), obj in sources.items() if s == source)
    except StopIteration:
        avail = sorted({s for s, _ in sources})
        return f"Unknown source {source!r}. Available sources: {avail}"

    raw = sql_query.strip()
    leader = raw.lstrip().upper()
    if not leader.startswith(("SELECT", "WITH")):
        return "Only read-only SELECT or WITH queries are allowed for exploration."

    try:
        sql_clean = clean_sql(raw, base.dialect, prettify=False)
    except Exception as e:
        return f"SQL parse/clean error: {e}"

    try:
        df = base.execute(sql_clean)
    except Exception as e:
        return f"{type(e).__name__}: {e}"

    preview = df.head(100)
    text = preview.to_string(max_cols=25, max_rows=50)
    if len(text) > 8000:
        text = text[:8000] + "\n... (truncated)"
    return text


def make_run_exploration_sql_tool(sources: dict[tuple[str, str], Source]) -> FunctionTool:
    """Build a :class:`~lumen.ai.tools.FunctionTool` that runs :func:`execute_exploration_sql` for ``sources``."""

    async def run_exploration_sql(source: str, sql_query: str) -> str:
        return await execute_exploration_sql(source, sql_query, sources=sources)

    names = ", ".join(sorted({s for s, _ in sources})) or "(none)"
    run_exploration_sql.__doc__ = (
        f"Execute read-only SQL on the named source to inspect data (use LIMIT on raw selects). "
        f"Sources: {names}."
    )
    return FunctionTool(
        run_exploration_sql,
        purpose=(
            "Run exploratory read-only SQL (SELECT/WITH) on a datasource by name; "
            "returns a small text preview of the result or an error message."
        ),
    )


class SQLInputs(ContextModel):

    data: NotRequired[Any]

    source: Source

    sources: Annotated[list[Source], ("accumulate", "source")]

    sql: NotRequired[str]

    metaset: Metaset

    visible_slugs: NotRequired[set[str]]


class SQLOutputs(ContextModel):

    data: Any

    table: str

    sql: str

    pipeline: Pipeline


class SQLAgent(BaseLumenAgent):

    conditions = param.List(
        default=[
            "Use for querying, filtering, aggregating, or transforming data with SQL",
            "Use for calculations that require executing SQL (e.g., 'calculate average', 'sum by category')",
            "Use when user asks to 'show', 'get', 'fetch', 'query', 'find', 'filter', 'calculate', 'aggregate', or 'transform' data",
            "NOT when user asks to 'explain', 'interpret', 'analyze', 'summarize', or 'comment on' existing data",
            "NOT useful if the user is using the same data for plotting",
        ]
    )

    exclusions = param.List(default=["dbtsl_metaset"])

    not_with = param.List(default=["DbtslAgent", "MetadataLookup", "TableListAgent"])

    purpose = param.String(
        default="""
        Creates and executes SQL queries to retrieve, filter, aggregate, or transform data.
        Handles table joins, WHERE clauses, GROUP BY, calculations, and other SQL operations.
        Generates new data pipelines from SQL transformations."""
    )

    prompts = param.Dict(
        default={
            "main": {
                "response_model": make_sql_model,
                "template": PROMPTS_DIR / "SQLAgent" / "main.jinja2",
            },
            "select_tables": {
                "response_model": make_table_selection_model,
                "template": PROMPTS_DIR / "SQLAgent" / "select_tables.jinja2",
            },
            "revise_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "SQLAgent" / "revise_output.jinja2"},
        }
    )

    user = param.String(default="SQL")

    _extensions = ("codeeditor", "tabulator")

    _editor_type = SQLEditor

    input_schema = SQLInputs
    output_schema = SQLOutputs

    async def _gather_prompt_context(self, prompt_name: str, messages: list, context: TContext, **kwargs):
        """Pre-compute metaset context for template rendering."""
        prompt_context = await super()._gather_prompt_context(prompt_name, messages, context, **kwargs)

        # Ensure schemas are loaded for schema_tables before rendering
        if "metaset" in context:
            metaset = context["metaset"]
            if metaset.schema_tables:
                await metaset.ensure_schemas(metaset.schema_tables)

        return prompt_context

    async def _validate_sql(
        self,
        context: TContext,
        sql_query: str,
        expr_slug: str,
        source: BaseSQLSource,
        messages: list[Message],
        step: ChatStep,
        max_retries: int = 2,
        discovery_context: str | None = None,
    ) -> str:
        """Validate and potentially fix SQL query."""
        # Clean SQL
        try:
            sql_query = clean_sql(sql_query, source.dialect, prettify=True)
        except Exception as e:
            step.stream(f"\n\n❌ SQL cleaning failed: {e}")

        # Validate with retries
        for i in range(max_retries):
            try:
                step.stream(f"\n\n`{expr_slug}`\n```sql\n{sql_query}\n```")
                source.execute(sql_query)
                step.stream("\n\n✅ SQL validation successful")
                return sql_query
            except Exception as e:
                if i == max_retries - 1:
                    step.stream(f"\n\n❌ SQL validation failed after {max_retries} attempts: {e}")
                    raise e

                # Retry with LLM fix
                step.stream(f"\n\n⚠️ SQL validation failed (attempt {i+1}/{max_retries}): {e}")
                feedback = f"{type(e).__name__}: {e!s}"
                if "KeyError" in feedback:
                    feedback += " The data does not exist; select from available data sources."

                retry_result = await self.revise(
                    feedback, messages, context, spec=sql_query, language=f"sql.{source.dialect}",
                    discovery_context=discovery_context
                )
                sql_query = clean_sql(retry_result, source.dialect, prettify=True)
        return sql_query

    async def _execute_query(
        self, source: BaseSQLSource, context: TContext, expr_slug: str, sql_query: str, tables: list[str],
        is_final: bool, should_materialize: bool, step: ChatStep
    ) -> tuple[Pipeline, Source, str]:
        """Execute SQL query and return pipeline and summary."""
        # Create SQL source
        source_tables = source.tables if source.tables is not None else {}
        table_defs = {table: source_tables[table] for table in tables if table in source_tables}
        table_defs[expr_slug] = sql_query

        # Only pass materialize parameter for DuckDB sources
        if isinstance(source, DuckDBSource):
            sql_expr_source = source.create_sql_expr_source(
                table_defs, materialize=should_materialize
            )
        else:
            sql_expr_source = source.create_sql_expr_source(table_defs)

        if should_materialize:
            context["source"] = sql_expr_source

            if sql_expr_source.metadata is None:
                sql_expr_source.metadata = {}
            existing_entries = sum(
                1 for v in sql_expr_source.metadata.values()
                if isinstance(v, dict) and "derived_from" in v
            )
            sql_expr_source.metadata[expr_slug] = {
                "derived_from": [
                    f"{source.name}{SOURCE_TABLE_SEPARATOR}{t}" if SOURCE_TABLE_SEPARATOR not in t else t
                    for t in tables
                ],
                "created_order": existing_entries + 1,
            }

        # Create pipeline
        if is_final:
            sql_transforms = [SQLLimit(limit=1_000_000, write=source.dialect, pretty=True, identify=False)]
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms)
        else:
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug)

        # Get data summary
        df = await get_data(pipeline)
        summary = await describe_data(df, reduce_enums=False)
        if len(summary) >= 1000:
            summary = summary[:1000-3] + "..."
        summary_formatted = f"\n```\n{summary}\n```"
        if should_materialize:
            summary_formatted += f"\n\nMaterialized data: `{sql_expr_source.name}{SOURCE_TABLE_SEPARATOR}{expr_slug}`"
        stream_details(f"{summary_formatted}", step, title=expr_slug)

        return pipeline, sql_expr_source, summary

    async def _finalize_execution(
        self,
        pipeline: Pipeline,
        sql: str,
        step_title: str | None,
        raise_if_empty: bool = False
    ) -> SQLEditor:
        """Finalize execution for final step."""

        df = await get_data(pipeline)
        if df.empty and raise_if_empty:
            raise ValueError(f"\nQuery `{sql}` returned empty results; ensure all the WHERE filter values exist in the dataset.")

        view = self._editor_type(
            component=pipeline, title=step_title, spec=sql
        )
        return view

    def _merge_sources(
        self, sources: dict[tuple[str, str], BaseSQLSource], tables: list[tuple[str, str]]
    ) -> tuple[BaseSQLSource, list[str]]:
        if len(set(m.source for m in tables)) == 1:
            m = tables[0]
            return sources[(m.source, m.table)], [m.table]
        mirrors = {}
        for m in tables:
            if not any(m.table.rstrip(")").rstrip("'").rstrip('"').endswith(ext)
                       for ext in [".csv", ".parquet", ".parq", ".json", ".xlsx"]):
                renamed_table = m.table.replace(".", "_")
            else:
                renamed_table = m.table
            mirrors[renamed_table] = (sources[(m.source, m.table)], m.table)
        return DuckDBSource(uri=":memory:", mirrors=mirrors), list(mirrors)

    async def _select_tables(
        self,
        messages: list[Message],
        context: TContext,
        step: ChatStep,
    ) -> list[str]:
        """Select relevant tables for the query."""
        metaset = context.get("metaset")
        if not metaset:
            return []

        all_tables = list(metaset.catalog.keys())
        # If 3 or fewer tables, use all of them
        if len(all_tables) <= 3:
            return all_tables

        selection = await self._invoke_prompt("select_tables", messages, context, model_kwargs=dict(available_tables=all_tables))
        selected = selection.tables if selection else metaset.get_top_tables(n=3)
        step.stream(f"Selected: {selected}")
        return selected

    async def _render_execute_query(
        self,
        messages: list[Message],
        context: TContext,
        sources: dict[tuple[str, str], BaseSQLSource],
        step_title: str,
        success_message: str,
        discovery_context: str | None = None,
        raise_if_empty: bool = False,
        output_title: str | None = None,
        errors: list[str] | None = None,
    ) -> SQLEditor:
        """
        Helper method that generates, validates, and executes final SQL queries.

        Parameters
        ----------
        messages : list[Message]
            List of user messages
        sources : dict[tuple[str, str], BaseSQLSource]
            Data sources to execute against
        step_title : str
            Title for the execution step
        success_message : str
            Message to show on successful completion
        discovery_context : str, optional
            Optional discovery context to include in prompt
        raise_if_empty : bool, optional
            Whether to raise error if query returns empty results
        output_title : str, optional
            Title to use for the output
        errors : list[str], optional
            List of previous errors to include in prompt

        Returns
        -------
        SQLEditor
            Output object from successful execution
        """
        with self._add_step(title=step_title, steps_layout=self._steps_layout) as step:
            # Generate SQL using common prompt pattern
            dialects = set(src.dialect for src in sources.values())
            dialect = "duckdb" if len(dialects) > 1 else next(iter(dialects))

            tool_list = [make_run_exploration_sql_tool(sources)]

            output = await self._invoke_prompt(
                "main",
                messages,
                context,
                model_kwargs=dict(sources=list(sources)),
                dialect=dialect,
                step_number=1,
                is_final_step=True,
                current_step="",
                sql_query_history={},
                current_iteration=1,
                sql_plan_context=None,
                errors=errors,
                discovery_context=discovery_context,
                source_names=sorted({s for s, _ in sources}),
                tools=tool_list,
            )

            if not output:
                raise ValueError("No output was generated.")

            # Check if all tables are from a single unique source
            unique_sources = set(src for src, _ in sources.keys())
            if len(unique_sources) == 1:
                # Single source - use tables from LLM output (list of table names)
                source = next(iter(sources.values()))
                tables = output.tables
            else:
                # Multiple sources - need to merge (output.tables contains SourceTable objects)
                source, tables = self._merge_sources(sources, output.tables)
            sql_query = output.query.strip()
            expr_slug = output.table_slug.strip()

            validated_sql = await self._validate_sql(
                context, sql_query, expr_slug, source, messages,
                step, discovery_context=discovery_context
            )

            # Only materialize for DuckDB sources
            should_materialize = isinstance(source, DuckDBSource)

            pipeline, sql_expr_source, summary = await self._execute_query(
                source, context, expr_slug, validated_sql, tables=tables,
                is_final=True, should_materialize=should_materialize, step=step
            )

            view = await self._finalize_execution(
                pipeline,
                validated_sql,
                output_title,
                raise_if_empty=raise_if_empty
            )

            if isinstance(step, ChatStep):
                step.param.update(
                    status="success",
                    success_title=success_message
                )

        return view

    async def revise(
        self,
        feedback: str,
        messages: list[Message],
        context: TContext,
        view: LumenEditor | None = None,
        spec: str | None = None,
        language: str | None = None,
        errors: list[str] | None = None,
        **kwargs
    ) -> str:
        result = await super().revise(
            feedback, messages, context, view=view, spec=spec, language=language, **kwargs
        )
        if view is not None:
            result = clean_sql(result, view.component.source.dialect, prettify=True)
        return result

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], SQLOutputs]:
        """Execute SQL generation with optional table selection and an optional SQL tool on the main call."""
        sources = context["sources"]
        metaset = context["metaset"]
        selected_slugs = list(metaset.catalog)

        if len(selected_slugs) > 3:
            with self._add_step(title="Selecting relevant tables...", steps_layout=self._steps_layout) as step:
                selected_tables = await self._select_tables(messages, context, step)
                if selected_tables:
                    metaset.schema_tables = selected_tables
                    step.stream(f"\n\nWill load schemas for: {selected_tables}")
        else:
            # Use all tables if 3 or fewer
            metaset.schema_tables = selected_slugs

        # Build source lookup, skipping slugs whose source no longer
        # exists in the context (stale entries from prior materializations).
        available_source_names = {s.name for s in sources}
        sources = {}
        for table_slug in selected_slugs:
            source_name = table_slug.split(SOURCE_TABLE_SEPARATOR)[0] if SOURCE_TABLE_SEPARATOR in table_slug else None
            if source_name and source_name not in available_source_names:
                continue
            key = tuple(table_slug.split(SOURCE_TABLE_SEPARATOR))
            src = parse_table_slug(table_slug, context["sources"])[0]
            if src is not None:
                sources[key] = src
        if not sources:
            raise ValueError("No valid SQL sources available for querying.")

        out = await self._render_execute_query(
            messages,
            context,
            sources=sources,
            step_title="Generating SQL...",
            success_message="SQL generation successful",
            discovery_context=None,
            raise_if_empty=True,
            output_title=step_title
        )
        out_context = await out.render_context()
        return [out], out_context
