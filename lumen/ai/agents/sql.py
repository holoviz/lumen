import asyncio
import difflib
import traceback

from typing import (
    Annotated, Any, Literal, NotRequired,
)

import param

from panel_material_ui.chat import ChatStep
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from pydantic.json_schema import SkipJsonSchema

from ...pipeline import Pipeline
from ...sources.base import BaseSQLSource, Source
from ...sources.duckdb import DuckDBSource
from ...transforms.sql import SQLLimit
from ..config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from ..context import ContextModel, TContext
from ..llm import Message
from ..models import EscapeBaseModel, PartialBaseModel, SearchReplaceSpec
from ..schemas import Metaset
from ..utils import (
    clean_sql, describe_data, get_data, get_pipeline, parse_table_slug,
    retry_llm_output, stream_details,
)
from ..views import LumenOutput, SQLOutput
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


class SampleQuery(PartialBaseModel):
    """See actual data content - reveals format issues and patterns."""

    query: SkipJsonSchema[str] = Field(default="SELECT * FROM {slug[table]} LIMIT 5")

    def generate_sql(self, source: Source) -> str:
        """Generate SQL for this query type."""
        format_kwargs = self.dict(exclude={'query'})
        return self.query.format(**format_kwargs)

    def format_result(self, df) -> str:
        """Format query results for display."""
        return f"\n```\n{df.head().to_string(max_cols=10)}\n```"

    def get_description(self) -> str:
        """Get query description for logging."""
        return f"Sample from {self.slug.table}"


class DistinctQuery(PartialBaseModel):
    """Universal column analysis with optional pattern matching - handles join keys, categories, date ranges."""

    query: SkipJsonSchema[str] = Field(default="SELECT DISTINCT {column} FROM {slug[table]} WHERE {column} ILIKE '%{pattern}%' LIMIT 10 OFFSET {offset}")
    column: str = Field(description="Column to analyze unique values")
    pattern: str = Field(default="", description="Optional pattern to search for (e.g., 'chin' for China/Chinese variations). Leave empty for all distinct values.")
    offset: int = Field(default=0, description="Number of distinct values to skip (0 for initial, 10 for follow-up)")

    def model_post_init(self, __context):
        """Adjust query template based on whether pattern is provided."""
        if not self.pattern.strip():
            # No pattern - get all distinct values
            self.query = "SELECT DISTINCT {column} FROM {slug[table]} LIMIT 10 OFFSET {offset}"
        else:
            # Pattern provided - use ILIKE for case-insensitive partial matching
            self.query = "SELECT DISTINCT {column} FROM {slug[table]} WHERE {column} ILIKE '%{pattern}%' LIMIT 10 OFFSET {offset}"

    def generate_sql(self, source: Source) -> str:
        """Generate SQL for this query type."""
        format_kwargs = self.dict(exclude={'query'})
        return self.query.format(**format_kwargs)

    def format_result(self, df) -> str:
        """Format query results for display."""
        return f" `{df.iloc[:10, 0].tolist()}`"

    def get_description(self) -> str:
        """Get query description for logging."""
        return f"Distinct values from {self.slug.table!r} table column {self.column!r}"


class TableQuery(PartialBaseModel):
    """Wildcard search using native source metadata to discover tables and columns by pattern."""

    pattern: str = Field(description="Search pattern (e.g., 'revenue', 'customer')")
    scope: Literal["columns", "tables", "both"] = Field(
        default="both",
        description="Search scope: 'columns' for column names, 'tables' for table names, 'both' for either"
    )

    def _match_string(self, pattern: str, target: str) -> float | None:
        """Calculate match score for a pattern against a target string.

        Returns:
            Score from 0.0 to 1.0 if match is good enough, None otherwise.
            1.0 = exact substring match, 0.6-0.99 = fuzzy match.
        """
        pattern_lower = pattern.lower()
        target_lower = target.lower()

        # Exact substring match gets highest score
        if pattern_lower in target_lower:
            return 1.0

        # Fuzzy match using SequenceMatcher
        ratio = difflib.SequenceMatcher(None, pattern_lower, target_lower).ratio()
        return ratio if ratio > 0.6 else None

    def search_metadata(self, source: Source) -> list[tuple[str, str | None]]:
        """Search tables and columns using native source metadata with fuzzy matching.

        Returns:
            List of (table_name, column_name) tuples sorted by match quality.
            column_name is None for table matches.
        """
        matches = []  # List of (table, column, score)

        # Get all tables and their metadata
        tables = source.get_tables()
        metadata = source._get_table_metadata(tables)

        for table_name, table_meta in metadata.items():
            # Check if table name matches
            if self.scope in ("tables", "both"):
                score = self._match_string(self.pattern, table_name)
                if score is not None:
                    matches.append((table_name, None, score))

            # Check if any column names match
            if self.scope in ("columns", "both") and "columns" in table_meta:
                for col_name in table_meta["columns"].keys():
                    score = self._match_string(self.pattern, col_name)
                    if score is not None:
                        matches.append((table_name, col_name, score))

        # Sort by score (highest first) and limit to top 20
        matches.sort(key=lambda x: x[2], reverse=True)
        return [(table, col) for table, col, _ in matches[:20]]

    def format_result(self, matches: list[tuple[str, str | None]]) -> str:
        """Format search results for display."""
        if not matches:
            return f" No matches found for pattern '{self.pattern}'"

        formatted = []
        for table_name, col_name in matches:
            if col_name:
                formatted.append(f"{table_name}.{col_name}")
            else:
                formatted.append(table_name)

        return f" `{formatted}`"

    def get_description(self) -> str:
        """Get query description for logging."""
        return f"Wildcard search for '{self.pattern}' in {self.scope}"


def make_discovery_model(sources: list[tuple[str, str]]):

    SourceTable = make_source_table_model(sources)

    SampleQueryLiteral = create_model(
        "LiteralSampleQuery",
        slug=(
                SourceTable, FieldInfo(description="The source and table identifier(s) referenced in the SQL query.")
        ),
        __base__=SampleQuery
    )

    DistinctQueryLiteral = create_model(
        "LiteralDistinctQuery",
        slug=(
                SourceTable, FieldInfo(description="The source and table identifier(s) referenced in the SQL query.")
        ),
        __base__=DistinctQuery
    )

    class DiscoveryQueries(PartialBaseModel):
        """LLM selects 2-4 essential discoveries using the core toolkit."""

        reasoning: str = Field(description="Brief discovery strategy")
        queries: list[SampleQueryLiteral | DistinctQueryLiteral | TableQuery] = Field(
            description="Choose 2-4 discovery queries. Use SampleQuery/DistinctQuery for known tables, TableQuery for wildcard searches."
        )

    class DiscoverySufficiency(EscapeBaseModel):
        """Evaluate if discovery results provide enough context to fix the original error."""

        reasoning: str = Field(description="Analyze discovery results - do they reveal the cause of the error?")

        sufficient: bool = Field(description="True if discoveries provide enough context to fix the error")

        follow_up_needed: list[SampleQueryLiteral | DistinctQueryLiteral | TableQuery] = Field(
            default_factory=list,
            description="If insufficient, specify 1-3 follow-up discoveries (e.g., OFFSET for more values, or TableQuery for wildcards)"
        )

    return DiscoveryQueries, DiscoverySufficiency


class SQLQuery(PartialBaseModel):
    """A single SQL query with its associated metadata."""

    query: str = Field(description="""
        One, correct, valid SQL query that answers the user's question;
        should only be one query and do NOT add extraneous comments; no multiple semicolons""")

    table_slug: str = Field(
        description="""
        Provide a unique, descriptive table slug for the SQL expression that clearly indicates the key transformations and source tables involved.
        Include 1 or 2 elements of data lineage in the slug, such as the main transformation and original table names,
        e.g. top_5_athletes_in_2020 or distinct_years_from_wx_table.
        Ensure the slug does not duplicate any existing table names or slugs.
        """
    )


class TableSelection(PartialBaseModel):
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
            "Use for displaying, examining, or querying data resulting in a data pipeline",
            "Use for calculations that require data (e.g., 'calculate average', 'sum by category')",
            "Do not sample or show limited rows of the data, unless explicitly requested",
            "NOT for non-data questions or technical programming help",
            "NOT useful if the user is using the same data for plotting",
        ]
    )

    exclusions = param.List(default=["dbtsl_metaset"])

    not_with = param.List(default=["DbtslAgent", "MetadataLookup", "TableListAgent"])

    exploration_enabled = param.Boolean(default=True, allow_refs=True, doc="""
        Whether to enable SQL exploration mode. When False, only attempts oneshot SQL generation.""")

    purpose = param.String(
        default="""
        Handles the display of data and the creation, modification, and execution
        of SQL queries to address user queries about the data. Executes queries in
        a single step, encompassing tasks such as table joins, filtering, aggregations,
        and calculations. If additional columns are required, SQLAgent can join the
        current table with other tables to fulfill the query requirements."""
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
            "select_discoveries": {
                "response_model": make_discovery_model,
                "template": PROMPTS_DIR / "SQLAgent" / "select_discoveries.jinja2",
            },
            "check_sufficiency": {
                "response_model": make_discovery_model,
                "template": PROMPTS_DIR / "SQLAgent" / "check_sufficiency.jinja2",
            },
            "revise_output": {
                "response_model": SearchReplaceSpec,
                "template": PROMPTS_DIR / "SQLAgent" / "revise_output.jinja2",
            },
        }
    )

    user = param.String(default="SQL")

    _extensions = ("codeeditor", "tabulator")

    _output_type = SQLOutput

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
    ) -> SQLOutput:
        """Finalize execution for final step."""

        df = await get_data(pipeline)
        if df.empty and raise_if_empty:
            raise ValueError(f"\nQuery `{sql}` returned empty results; ensure all the WHERE filter values exist in the dataset.")

        view = self._output_type(
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

        # Otherwise, ask LLM to select relevant tables
        system_prompt = await self._render_prompt(
            "select_tables",
            messages,
            context,
        )

        model_spec = self.prompts["select_tables"].get("llm_spec", self.llm_spec_key)
        selection_model = self._get_model("select_tables", available_tables=all_tables)

        selection = await self.llm.invoke(
            messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=selection_model,
        )

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
    ) -> SQLOutput:
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
        SQLOutput
            Output object from successful execution
        """
        with self._add_step(title=step_title, steps_layout=self._steps_layout) as step:
            # Generate SQL using common prompt pattern
            dialects = set(src.dialect for src in sources.values())
            dialect = "duckdb" if len(dialects) > 1 else next(iter(dialects))
            system_prompt = await self._render_prompt(
                "main",
                messages,
                context,
                dialect=dialect,
                step_number=1,
                is_final_step=True,
                current_step="",
                sql_query_history={},
                current_iteration=1,
                sql_plan_context=None,
                errors=errors,
                discovery_context=discovery_context,
            )

            # Generate SQL
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            sql_response_model = self._get_model("main", sources=list(sources))

            output = await self.llm.invoke(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=sql_response_model,
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

    async def _select_discoveries(
        self,
        messages: list[Message],
        context: TContext,
        sources: dict[tuple[str, str], BaseSQLSource],
        error_context: str,
    ) -> BaseModel:
        """Let LLM choose which discoveries to run based on error and available tables."""
        system_prompt = await self._render_prompt(
            "select_discoveries",
            messages,
            context,
            error_context=error_context,
        )

        discovery_models = self._get_model("select_discoveries", sources=list(sources))
        discovery_model = discovery_models[0]
        model_spec = self.prompts["select_discoveries"].get("llm_spec", self.llm_spec_key)
        selection = await self.llm.invoke(
            messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=discovery_model
        )
        return selection

    async def _check_discovery_sufficiency(
        self,
        messages: list[Message],
        context: TContext,
        error_context: str,
        discovery_results: list[tuple[str, str]],
    ) -> BaseModel:
        """Check if discovery results are sufficient to fix the error."""
        system_prompt = await self._render_prompt(
            "check_sufficiency",
            messages,
            context,
            error_context=error_context,
            discovery_results=discovery_results,
        )

        # Get sources from context for the discovery models
        metaset = context["metaset"]
        selected_slugs = list(metaset.catalog)
        visible_slugs = context.get('visible_slugs', [])
        if visible_slugs:
            selected_slugs = [slug for slug in selected_slugs if slug in visible_slugs]
        sources_list = [
            tuple(table_slug.split(SOURCE_TABLE_SEPARATOR))
            for table_slug in selected_slugs
        ]

        discovery_models = self._get_model("check_sufficiency", sources=sources_list)
        sufficiency_model = discovery_models[1]

        model_spec = self.prompts["check_sufficiency"].get("llm_spec", self.llm_spec_key)
        sufficiency = await self.llm.invoke(
            messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=sufficiency_model,
        )
        return sufficiency

    async def _execute_discovery_query(
        self,
        query_model: SampleQuery | DistinctQuery | TableQuery,
        sources: dict[tuple[str, str], Source],
    ) -> tuple[str, str]:
        """Execute a single discovery query and return formatted result."""
        if isinstance(query_model, TableQuery):
            # TableQuery searches ALL unique sources using native metadata
            unique_sources = {}
            for (source_name, _), source in sources.items():
                if source_name not in unique_sources:
                    unique_sources[source_name] = source

            # Search metadata in all sources in parallel
            tasks = [
                asyncio.create_task(
                    self._execute_table_query_on_source(query_model, source_name, source)
                )
                for source_name, source in unique_sources.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Combine non-empty results
            combined_results = []
            for source_name, result in zip(unique_sources.keys(), results, strict=False):
                if isinstance(result, Exception):
                    continue  # Skip failed sources
                if result and "No matches" not in result:
                    combined_results.append(f"`{source_name}`: {result}")
            return query_model.get_description(), "\n".join(combined_results)

        # SampleQuery/DistinctQuery use the specific table's source
        source = sources[(query_model.slug.source, query_model.slug.table)]
        sql = query_model.generate_sql(source)
        query_type = query_model.get_description()

        # Execute query
        try:
            df = source.execute(sql)
        except Exception as e:
            return f"Discovery query {query_type[0].lower()}{query_type[1:]} failed", str(e)

        # Format results
        result = query_model.format_result(df)
        return query_type, result

    async def _execute_table_query_on_source(
        self,
        query_model: TableQuery,
        source_name: str,
        source: Source,
    ) -> str:
        """Execute TableQuery on a single source using native metadata."""
        try:
            matches = query_model.search_metadata(source)
            return query_model.format_result(matches)
        except Exception:
            # Some sources might not support metadata retrieval
            return ""

    async def _run_discoveries_parallel(
        self,
        discovery_queries: list[SampleQuery | DistinctQuery | TableQuery],
        sources: dict[tuple[str, str], Source],
    ) -> list[str]:
        """Execute all discoveries concurrently and stream results as they complete."""
        # Create all tasks
        tasks = [
            asyncio.create_task(self._execute_discovery_query(query, sources))
            for query in discovery_queries
        ]

        results = []
        for completed_task in asyncio.as_completed(tasks):
            query_type, result = await completed_task
            with self._add_step(title=query_type, steps_layout=self._steps_layout) as step:
                step.stream(result)
            results.append((query_type, result))
        return results

    async def _explore_tables(
        self,
        messages: list[Message],
        context: TContext,
        sources: dict[tuple[str, str], Source],
        error_context: str,
        step_title: str | None = None,
    ) -> SQLOutput:
        """Run adaptive exploration: initial discoveries → sufficiency check → optional follow-ups → final answer."""
        # Step 1: LLM selects initial discoveries
        with self._add_step(title="Selecting initial discoveries", steps_layout=self._steps_layout) as step:
            selection = await self._select_discoveries(messages, context, sources, error_context)
            step.stream(f"Strategy: {selection.reasoning}\n\nSelected {len(selection.queries)} initial discoveries")

        # Step 2: Run initial discoveries in parallel
        initial_results = await self._run_discoveries_parallel(selection.queries, sources)

        # Step 3: Check if discoveries are sufficient
        with self._add_step(title="Evaluating discovery sufficiency", steps_layout=self._steps_layout) as step:
            sufficiency = await self._check_discovery_sufficiency(messages, context, error_context, initial_results)
            step.stream(f"Assessment: {sufficiency.reasoning}")

            if sufficiency.sufficient:
                step.stream("\n✅ Sufficient context - proceeding to final answer")
                final_results = initial_results
            else:
                step.stream(f"\n⚠️ Insufficient - running {len(sufficiency.follow_up_needed)} follow-ups")

        # Step 4: Run follow-up discoveries if needed
        if not sufficiency.sufficient and sufficiency.follow_up_needed:
            follow_up_results = await self._run_discoveries_parallel(sufficiency.follow_up_needed, sources)
            final_results = initial_results + follow_up_results
        else:
            final_results = initial_results

        # Step 5: Generate final answer with all discovery context
        discovery_context = "\n".join([
            f"- **{query_type}:** {result}" for query_type, result in final_results
        ])

        return await self._render_execute_query(
            messages,
            context,
            sources=sources,
            step_title="Generating final answer with discoveries",
            success_message="Final answer generated with discovery context",
            discovery_context=discovery_context,
            raise_if_empty=False,
            output_title=step_title
        )

    async def revise(
        self,
        feedback: str,
        messages: list[Message],
        context: TContext,
        view: LumenOutput | None = None,
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
        """Execute SQL generation with table selection, then one-shot attempt, then exploration if needed."""
        sources = context["sources"]
        metaset = context["metaset"]
        selected_slugs = list(metaset.catalog)
        visible_slugs = context.get('visible_slugs', [])
        if selected_slugs and visible_slugs:
            # hide de-selected slugs
            selected_slugs = [slug for slug in selected_slugs if slug in visible_slugs]
        elif not selected_slugs and visible_slugs:
            # if no closest matches, use visible slugs
            selected_slugs = list(visible_slugs)

        # Select relevant tables if there are many
        if len(selected_slugs) > 3:
            with self._add_step(title="Selecting relevant tables...", steps_layout=self._steps_layout) as step:
                selected_tables = await self._select_tables(messages, context, step)
                if selected_tables:
                    metaset.schema_tables = selected_tables
                    step.stream(f"\n\nWill load schemas for: {selected_tables}")
        else:
            # Use all tables if 3 or fewer
            metaset.schema_tables = selected_slugs

        sources = {
            tuple(table_slug.split(SOURCE_TABLE_SEPARATOR)): parse_table_slug(table_slug, sources)[0]
            for table_slug in selected_slugs
        }
        if not sources:
            raise ValueError("No valid SQL sources available for querying.")
        try:
            if not self.exploration_enabled:
                # Try retry-wrapped one-shot approach
                execute_method = retry_llm_output()(self._render_execute_query)
            else:
                # Try one-shot approach first
                execute_method = self._render_execute_query

            out = await execute_method(
                messages,
                context,
                sources=sources,
                step_title="Generating SQL...",
                success_message="SQL generation successful",
                discovery_context=None,
                raise_if_empty=True,
                output_title=step_title
            )
        except Exception as e:
            if not self.exploration_enabled:
                traceback.print_exc()
                # If exploration is disabled, re-raise the error instead of falling back
                raise e
            # Fall back to exploration mode if enabled
            out = await self._explore_tables(
                messages, context, sources, str(e), step_title
            )
        out_context = await out.render_context()
        return [out], out_context
