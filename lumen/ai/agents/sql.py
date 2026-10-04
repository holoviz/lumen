import typing as t

import narwhals.stable.v2 as nw
import pandas as pd
import param
import sqlglot

from panel.chat import ChatStep
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from ...filters import ConstantFilter
from ...pipeline import Pipeline
from ...sources.base import BaseSQLSource, QueryTimeoutError, Source
from ...sources.duckdb import DuckDBSource
from ...transforms.sql import SQLLimit
from ...util import as_narwhals, as_pandas, is_lazyframe
from ..config import (
    PROMPTS_DIR, SOURCE_TABLE_SEPARATOR, DeterministicError, EmptyResultError,
)
from ..context import ContextModel, TContext
from ..data_quality import lint_data
from ..editors import LumenEditor, SQLEditor
from ..llm import Message
from ..models import RetrySpec
from ..schemas import Metaset
from ..tools import FunctionTool
from ..utils import (
    PROFILE_SAMPLE_ROWS, clean_sql, describe_data, format_error, get_frame,
    get_pipeline, log_debug, normalize_object_dtypes, parse_table_slug,
    retry_llm_output, stream_details, truncate_to_tokens,
)
from .base_lumen import BaseLumenAgent

if t.TYPE_CHECKING:
    from narwhals.stable.v2.typing import Frame, IntoFrame

# Rows/columns shown by format_exploration_result. Exploration reveals a frame's
# shape, dtypes and value domains; a handful of rows does that, and the previous
# 50-row aligned dump cost ~3.4k tokens for a 53x7 frame — 8x this one.
EXPLORATION_PREVIEW_ROWS = 5
EXPLORATION_PREVIEW_COLS = 25
EXPLORATION_MAX_TOKENS = 1200
EXPLORATION_MAX_ROWS = 1000
VALIDATION_MAX_ROWS = PROFILE_SAMPLE_ROWS
# Schema YAML is dense (nested keys, enum lists) and tokenizes near 2.5
# chars/token, so the previous 12k-character cap admitted ~4.5k tokens.
SCHEMA_MAX_TOKENS = 3000
# Source tables profiled for one query. Each costs a query, and a join across
# more inputs than this is not worth the round trips.
SOURCE_PROFILE_MAX_TABLES = 3

EMPTY_RESULT_HINT = (
    "The query returned no rows. If the filters match the question, no rows is a valid "
    "result and the same query may be returned unchanged; otherwise verify the filter values."
)


def make_source_table_model(sources: list[tuple[str, str]]):
    class LiteralSourceTable(BaseModel):
        source: t.Literal[tuple(sorted(set(src for src, _ in sources)))]
        table: t.Literal[tuple(sorted(set(table for _, table in sources)))]
    return LiteralSourceTable


def make_table_model(sources: list[tuple[str, str]]):
    """
    Create a table model with constrained table choices.
    """
    # Sorted so the response schema, part of the cached prompt prefix, is stable across runs.
    available_tables = sorted(set(table for _, table in sources))
    TableLiteral = t.Literal[tuple(available_tables)]  # type: ignore
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


class SQLCleanup(BaseModel):
    """A revision of an existing query that removes data-quality problems from its result."""

    chain_of_thought: str = Field(description="""
        Which findings you are fixing and which you are leaving alone, in one sentence.""")

    query: str = Field(description="""
        The rewritten SQL query. Return the original query byte-for-byte unchanged
        when no finding clearly calls for one of the allowed edits; leaving the
        query alone is a correct and expected answer.""")


def validate_read_only_sql(sql_query: str, dialect: str) -> None:
    statements = sqlglot.parse(sql_query, read=None if dialect == "any" else dialect)
    if len(statements) != 1 or not isinstance(statements[0], sqlglot.exp.Query) or any(
        statements[0].find(kind) for kind in (
            sqlglot.exp.Insert, sqlglot.exp.Update, sqlglot.exp.Delete,
            sqlglot.exp.Create, sqlglot.exp.Drop, sqlglot.exp.Command, sqlglot.exp.Into,
        )
    ):
        raise ValueError("Only one read-only SELECT query is allowed.")


def format_exploration_result(df: "IntoFrame | Frame", *, capped: bool = False) -> str:
    """
    Render an exploration query result as a compact preview for the LLM.

    Reports the fetched row count (or a lower bound if capped), column dtypes,
    and a few example rows. Markdown omits index and alignment padding.

    Only the handful of cells actually rendered is converted to pandas, so a
    polars or pyarrow result reaches the model without the whole frame being
    copied. The dtypes reported are the preview's for that reason, which is
    also what keeps a pandas caller's output exactly as it was.
    """
    frame = as_narwhals(df)
    if is_lazyframe(frame):
        frame = frame.collect()
    n_rows, n_cols = frame.shape
    parts = [f"{'at least ' if capped else ''}{n_rows} rows x {n_cols} columns"]

    # Normalised because which columns land on object varies by library:
    # pandas converts a DECIMAL to float itself, polars and pyarrow keep it,
    # and an object column is reported as str by the dtype line below.
    names = frame.columns[:EXPLORATION_PREVIEW_COLS]
    preview = normalize_object_dtypes(as_pandas(
        frame.select(*[nw.col(name) for name in names]).head(EXPLORATION_PREVIEW_ROWS)
    ))
    if n_cols > EXPLORATION_PREVIEW_COLS:
        parts[0] += f" (showing the first {EXPLORATION_PREVIEW_COLS} columns)"

    def _dtype_name(dtype):
        return "str" if pd.api.types.is_string_dtype(dtype) else str(dtype)

    dtypes = ", ".join(f"{col}: {_dtype_name(dtype)}" for col, dtype in preview.dtypes.items())
    parts.append(f"Columns — {dtypes}")

    if n_rows:
        shown = min(n_rows, EXPLORATION_PREVIEW_ROWS)
        label = "all rows" if shown == n_rows else f"first {shown} of {n_rows} rows"
        parts.append(f"Sample ({label}):\n" + preview.to_markdown(index=False))
    else:
        parts.append("The query returned no rows.")

    return truncate_to_tokens("\n\n".join(parts), EXPLORATION_MAX_TOKENS)


async def execute_exploration_sql(
    source: str,
    sql_query: str,
    *,
    sources: dict[tuple[str, str], BaseSQLSource],
    timeout: float | None = None,
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
    timeout : float, optional
        Seconds after which the query is abandoned and reported as an error.

    Returns
    -------
    str
        Tabular preview, or an error message string if execution fails.
    """
    # Exact source-name match: the intended usage.
    base = next((obj for (s, _), obj in sources.items() if s == source), None)
    if base is None:
        # Weak models frequently pass a *table* name (or the wrong token)
        # instead of the source name — e.g. run_exploration_sql(source="obs")
        # when the source is "AnnDataSource00679". Resolve gracefully rather
        # than forcing a failed round-trip: prefer a table-name match, else
        # fall back to the sole source when the mapping is unambiguous.
        base = next((obj for (_, t), obj in sources.items() if t == source), None)
        if base is None:
            unique_sources = {s for s, _ in sources}
            if len(unique_sources) == 1:
                base = next(iter(sources.values()))
    if base is None:
        avail = sorted({s for s, _ in sources})
        tables = sorted({t for _, t in sources})
        return (
            f"Unknown source {source!r}. Available sources: {avail}; "
            f"tables: {tables}"
        )

    try:
        sql_clean = clean_sql(sql_query.strip(), base.dialect, prettify=False)
        validate_read_only_sql(sql_clean, base.dialect)
        limited = SQLLimit(limit=EXPLORATION_MAX_ROWS + 1, write=base.dialect).apply(sql_clean)
    except (ValueError, sqlglot.errors.ParseError, sqlglot.errors.TokenError) as e:
        return f"SQL parse/clean error: {format_error(e)}"

    try:
        df = await base.execute_with_timeout(limited, timeout, fetch=True)
    except Exception as e:
        return format_error(e)

    capped = len(df) > EXPLORATION_MAX_ROWS
    return format_exploration_result(df.head(EXPLORATION_MAX_ROWS) if capped else df, capped=capped)


def make_run_exploration_sql_tool(
    sources: dict[tuple[str, str], BaseSQLSource], timeout: float | None = None
) -> FunctionTool:
    """Build a :class:`~lumen.ai.tools.FunctionTool` that runs :func:`execute_exploration_sql` for ``sources``."""

    async def run_exploration_sql(source: str, sql_query: str) -> str:
        return await execute_exploration_sql(source, sql_query, sources=sources, timeout=timeout)

    names = ", ".join(f"`{s}`" for s in sorted({s for s, _ in sources})) or "(none)"
    tables = ", ".join(f"`{t}`" for t in sorted({t for _, t in sources})) or "(none)"
    run_exploration_sql.__doc__ = (
        f"Execute read-only SQL on the named source to inspect data (use LIMIT on raw selects). "
        f"Sources: {names}. "
        f"Tables: {tables}. "
        f"Reference tables by name directly (e.g. SELECT * FROM my_table), not with read_csv() or read_parquet()."
    )
    return FunctionTool(
        run_exploration_sql,
        purpose=(
            "Run exploratory read-only SQL (SELECT/WITH) on a datasource by name. "
            f"Returns up to {EXPLORATION_MAX_ROWS} rows (reporting 'at least' when capped), "
            f"column dtypes and {EXPLORATION_PREVIEW_ROWS} example rows. "
            "Use COUNT(*) only if an exact count is needed. "
            "This gathers information for the final SQL query; it does not produce the "
            "result the user sees, so stop exploring once you know the columns, types and "
            "value formats you need."
        ),
    )


def make_browse_data_catalog_tool(metaset: Metaset) -> FunctionTool:
    """
    Tool wrapping :meth:`~lumen.ai.schemas.Metaset.table_list` for incremental catalog browsing.
    """

    def browse_data_catalog(
        n: int = 15,
        offset: int = 0,
        n_others: int = 25,
        include_metadata: bool = False,
        include_lineage: bool = False,
    ) -> str:
        """
        List tables from the metadata catalog (names, similarity ordering, optional docs).
        Does not load SQL engine column types — use load_table_schemas for those.
        """
        return metaset.table_list(
            n=n,
            offset=offset,
            n_others=n_others,
            include_metadata=include_metadata,
            include_lineage=include_lineage,
        )

    browse_data_catalog.__doc__ = (browse_data_catalog.__doc__ or "").strip()
    return FunctionTool(
        browse_data_catalog,
        purpose=(
            "Browse available table slugs via Metaset.table_list (paginated). "
            "Use before choosing final tables; does not include per-column SQL types or documentation text."
        ),
    )


def make_load_table_schemas_tool(metaset: Metaset, stats_timeout: float | None = 60) -> FunctionTool:
    """
    Tool returning full column statistics for specific catalog tables, computing
    them if the prompt only had names and types.
    """

    async def load_table_schemas(table_slugs: list[str]) -> str:
        """
        Return every column of the given tables with SQL type, keys, value ranges,
        literal values and null fractions, plus catalog descriptions.
        Accept table names, catalog slugs, source/table or source.table.
        """
        if not table_slugs:
            return "No table_slugs provided."
        blocks: list[str] = []
        resolved: list[str] = []
        for raw in table_slugs[:8]:
            if raw in metaset.catalog:
                resolved.append(raw)
                continue
            matches = sorted(
                slug for slug in metaset.catalog
                if (slug.split(SOURCE_TABLE_SEPARATOR, 1)[-1] == raw
                    or slug.replace(SOURCE_TABLE_SEPARATOR, "/") == raw
                    or slug.replace(SOURCE_TABLE_SEPARATOR, ".") == raw)
            )
            if len(matches) > 1:
                blocks.append(f"{raw}: error: ambiguous table {raw!r}. Use one of: {matches}")
            elif not matches:
                options = sorted(metaset.display_name(slug) for slug in metaset.catalog)[:10]
                blocks.append(f"{raw}: error: unknown table {raw!r}. Tables include: {options}.")
            else:
                resolved.append(matches[0])
        resolved = list(dict.fromkeys(resolved))
        if resolved:
            await metaset.ensure_stats(resolved, timeout=stats_timeout)
        for slug in resolved:
            blocks.append(truncate_to_tokens(metaset.table_detail(slug), SCHEMA_MAX_TOKENS // 3))
        if len(table_slugs) > 8:
            blocks.append(f"Only the first 8 tables were processed; request the remaining {len(table_slugs) - 8} separately.")
        if len(resolved) > 3:
            blocks.append("Each table is capped independently; request fewer tables for more detail.")
        return "\n\n".join(blocks)

    load_table_schemas.__doc__ = (load_table_schemas.__doc__ or "").strip()
    return FunctionTool(
        load_table_schemas,
        purpose=(
            "Load full column statistics (types, keys, ranges, values, null fractions) for "
            "chosen tables. Prefer this over guessing or exploratory SQL when the data "
            "summary lists a table with names and types only."
        ),
    )


def sql_contains_aggregates(sql_query: str, dialect: str | None = None) -> bool:
    """
    Whether ``sql_query`` collapses many input rows into fewer output rows.

    Such a query has already collapsed its inputs by the time it returns, so
    nothing about the result shows what those inputs carried: a ``-9999``
    placeholder is already inside ``SUM``, and a blank string is already a GROUP
    BY key. Profiling the result therefore says nothing about the data the
    numbers came from, and the source rows have to be profiled separately --
    otherwise a chart of ``SUM(revenue)`` silently inherits every placeholder in
    the table it was built from.

    Unparseable SQL returns False: the caller only uses this to decide whether to
    spend an extra query, and guessing "yes" would spend it on every failed parse.
    """
    try:
        parsed = sqlglot.parse_one(sql_query, read=None if dialect in (None, "any") else dialect)
    except Exception:
        return False
    return bool(parsed.find(sqlglot.exp.Group) or parsed.find(sqlglot.exp.AggFunc))


_RANGE_FIELD_TYPES = ("number", "integer")


def _coerce_filter_value(field_schema: dict[str, t.Any], value: t.Any) -> t.Any:
    """Coerce an LLM-supplied filter value to what the Source expects.

    A two-element ``[lo, hi]`` list on a numeric or datetime field becomes a
    ``(lo, hi)`` tuple (interpreted as an inclusive range / ``BETWEEN``); datetime
    bounds are parsed to timestamps. Scalars (equality) and longer lists
    (membership / ``IN``) are passed through unchanged.
    """
    is_datetime = field_schema.get("format") in ("date", "date-time", "datetime")
    if isinstance(value, (list, tuple)) and len(value) == 2 and (
        field_schema.get("type") in _RANGE_FIELD_TYPES or is_datetime
    ):
        lo, hi = value
        if is_datetime:
            lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        return (lo, hi)
    return value


def make_apply_filter_tool(pipeline: Pipeline) -> FunctionTool:
    """Build a :class:`~lumen.ai.tools.FunctionTool` that filters ``pipeline``.

    The tool lets the chat agent narrow the data already loaded in the current
    exploration -- subsetting an xarray coordinate dimension (``time``/``lat``/
    ``lon``/level) or a tabular column. It adds a
    :class:`~lumen.filters.base.Filter` to the pipeline (the same machinery as
    the manual "Add Filter" UI), which subsets the data without modifying the
    SQL query.

    Known limitation: this tool is registered on :class:`SQLAgent`, whose
    response also emits a SQL query. A single "filter" request can therefore
    both apply this in-place pipeline filter to the current exploration and open
    a new exploration whose SQL carries an equivalent ``WHERE`` clause. Giving
    the tool a dedicated, non-SQL-emitting home is tracked as follow-up work.
    """
    schema = pipeline.schema or {}
    filterable = [c for c in schema if c != "__len__"]

    async def apply_filter(field: str, value: t.Any) -> str:
        if field not in filterable:
            return (
                f"Cannot filter on {field!r}: not a column of this table. "
                f"Filterable fields: {', '.join(filterable) or '(none)'}."
            )
        try:
            coerced = _coerce_filter_value(schema[field], value)
            filt = ConstantFilter(field=field, value=coerced, schema=schema)
            # Replace any existing filter on the same field so repeated calls
            # refine rather than stack contradictory conditions.
            pipeline.filters = [f for f in pipeline.filters if f.field != field]
            pipeline.add_filter(filt)
        except Exception as exc:
            return f"Could not apply filter on {field!r} with value {value!r}: {exc}"
        return f"Applied filter on {field!r} ({value!r})."

    fields = ", ".join(filterable) or "(none)"
    apply_filter.__doc__ = (
        "Filter the current exploration's data on a single field. "
        f"Filterable fields: {fields}. "
        "Pass a [min, max] list for a numeric or datetime range, a single value "
        "for an exact match, or a list of values for membership."
    )
    return FunctionTool(
        apply_filter,
        purpose=(
            "Subset the data already loaded in the current exploration (e.g. an "
            "xarray coordinate such as time/lat/lon, or a tabular column). Prefer "
            "this over rewriting SQL when the user wants to narrow the existing result."
        ),
    )


def make_apply_filter_llm_tool(context: TContext):
    """Expose :func:`make_apply_filter_tool` as a context-gated ``llm_tools`` entry.

    Returns the tool only when a pipeline is present in working memory, so it is
    offered to the LLM exactly when there is an existing exploration to filter.
    """
    pipeline = context.get("pipeline")
    if pipeline is None:
        return []
    return make_apply_filter_tool(pipeline)


class SQLInputs(ContextModel):

    data: t.NotRequired[t.Any]

    source: Source

    sources: t.Annotated[list[Source], ("accumulate", "source")]

    sql: t.NotRequired[str]

    metaset: Metaset

    visible_slugs: t.NotRequired[set[str]]


class SQLOutputs(ContextModel):

    data: t.Any

    table: str

    sql: str

    pipeline: Pipeline


class SQLAgent(BaseLumenAgent):

    clean_data = param.Boolean(default=True, doc="""
        Profile a bounded sample of each query result and, when it contains data-quality problems,
        spend one extra LLM call rewriting the query to clean them up.
        Profiling is deterministic and reuses the frame validation already
        fetched, so a clean result costs nothing; set to False to always take
        the query exactly as first written.""")

    conditions = param.List(
        default=[
            "Use for querying, filtering, aggregating, or transforming data with SQL",
            "Use for calculations that require executing SQL (e.g., 'calculate average', 'sum by category')",
            "Use when user asks to 'show', 'get', 'fetch', 'query', 'find', 'filter', 'calculate', 'aggregate', or 'transform' data",
            "Use after external data has been fetched and the user expects the fetched fields/rows to be presented, selected, filtered, or otherwise queried",
            "Use when user asks about data quality, completeness, missing values, duplicates, or outliers",
            "NOT when user asks to 'explain', 'interpret', 'analyze', 'summarize', or 'comment on' existing data",
            "NOT useful if the user is using the same data for plotting",
        ]
    )

    exclusions = param.List(default=["dbtsl_metaset"])

    schema_max_tokens = param.Integer(default=4000, allow_None=True, doc="""
        Token budget for the data summary in the prompt. Tables beyond it are
        listed with names and types only, and the model loads the rest with
        load_table_schemas.""")

    schema_tables_shown = param.Integer(default=25, bounds=(1, None), doc="""
        Most relevant tables described in the data summary; the rest are
        listed by name. schema_max_tokens usually binds first.""")

    stats_wait = param.Number(default=10, allow_None=True, doc="""
        Seconds to wait for column statistics that are still being computed
        before answering with names and types; profiling continues in the
        background for later questions. None waits until they are ready.""")

    # When an exploration pipeline is already in context, let the LLM narrow it
    # with apply_filter. NOTE: SQLAgent also emits a SQL query, so a filter
    # request may both apply this in-place filter and open a new SQL exploration
    # (see make_apply_filter_tool for the known limitation / follow-up).
    llm_tools = param.List(default=[make_apply_filter_llm_tool])

    not_with = param.List(default=["DbtslAgent", "MetadataLookup", "TableListAgent"])

    purpose = param.String(
        default="""
        Creates and executes SQL queries to retrieve, filter, aggregate, or transform data.
        Handles table joins, WHERE clauses, GROUP BY, calculations, and other SQL operations.
        Generates new data pipelines from SQL transformations.
        Profiles every result for data-quality problems (missing values, duplicates,
        placeholder numbers, outliers) and reports them."""
    )

    query_timeout = param.Number(default=60, bounds=(0, None), allow_None=True, doc="""
        Seconds a query run on the model's behalf may take before it is
        abandoned, or None to wait indefinitely. A timed out query is
        revised once; a second timeout ends the request.""")

    prompts = param.Dict(
        default={
            "main": {
                "response_model": make_sql_model,
                "template": PROMPTS_DIR / "SQLAgent" / "main.jinja2",
            },
            "clean_data": {"response_model": SQLCleanup, "template": PROMPTS_DIR / "SQLAgent" / "clean_data.jinja2"},
            "revise_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "SQLAgent" / "revise_output.jinja2"},
        }
    )

    user = param.String(default="SQL")

    _extensions = ("codeeditor", "tabulator")

    _editor_type = SQLEditor

    input_schema = SQLInputs
    output_schema = SQLOutputs

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
        tools: list[FunctionTool] | None = None,
    ) -> tuple[str, pd.DataFrame | None]:
        """Validate and potentially fix SQL query.

        Returns the validated SQL alongside a bounded frame for profiling.
        The frame is None only when every attempt failed.
        """
        # Reject non-query statements before clean_sql can discard trailing statements.
        try:
            validate_read_only_sql(sql_query, source.dialect)
        except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
            pass
        try:
            sql_query = clean_sql(sql_query, source.dialect, prettify=True)
        except Exception as e:
            step.stream(f"\n\n❌ SQL cleaning failed: {e}")

        # Validate with retries
        timed_out = False
        for i in range(max_retries):
            try:
                validate_read_only_sql(sql_query, source.dialect)
            except ValueError:
                raise
            except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as e:
                if i == max_retries - 1:
                    raise
                retry_result = await self.revise(
                    format_error(e), messages, context, spec=sql_query,
                    language=f"sql.{source.dialect}", discovery_context=discovery_context, tools=tools,
                )
                sql_query = clean_sql(retry_result, source.dialect, prettify=True)
                continue
            try:
                step.stream(f"\n\n`{expr_slug}`\n```sql\n{sql_query}\n```")
                validated = SQLLimit(limit=VALIDATION_MAX_ROWS, write=source.dialect).apply(sql_query)
                result = await source.execute_with_timeout(validated, self.query_timeout)
                step.stream("\n\n✅ SQL validation successful")
                return sql_query, result
            except Exception as e:
                feedback = format_error(e)
                # Rewriting the query cannot make a listed table resolvable, and
                # each timed out attempt may leave a query running on the engine.
                table = source.missing_table(e)
                if table is not None and table in source.get_tables():
                    unrecoverable = (
                        f"Table {table!r} is listed by source {source.name!r} "
                        f"but the engine cannot resolve it: {feedback}"
                    )
                elif isinstance(e, QueryTimeoutError) and (timed_out or i == max_retries - 1):
                    unrecoverable = f"The revised query timed out as well: {feedback}"
                else:
                    unrecoverable = None
                if unrecoverable:
                    step.stream(f"\n\n❌ SQL validation failed: {unrecoverable}")
                    raise DeterministicError(unrecoverable) from e
                if i == max_retries - 1:
                    step.stream(f"\n\n❌ SQL validation failed after {max_retries} attempts: {feedback}")
                    raise
                timed_out = timed_out or isinstance(e, QueryTimeoutError)

                # Retry with LLM fix
                step.stream(f"\n\n⚠️ SQL validation failed (attempt {i+1}/{max_retries}): {feedback}")
                if "KeyError" in feedback:
                    feedback += " The data does not exist; select from available data sources."

                retry_result = await self.revise(
                    feedback, messages, context, spec=sql_query, language=f"sql.{source.dialect}",
                    discovery_context=discovery_context, tools=tools,
                )
                sql_query = clean_sql(retry_result, source.dialect, prettify=True)
        return sql_query, None

    async def _profile_source_rows(self, source: BaseSQLSource, tables: list[str]) -> list[str]:
        """
        Profile a sample of the rows feeding an aggregating query.

        Only called when the result itself cannot show the problem (see
        :func:`sql_contains_aggregates`), because each table costs one extra
        query. Findings are prefixed with their table so the rewriting prompt
        knows the fix belongs before the aggregation rather than in the output
        columns.

        Example: for a query grouping ``sales``, this runs
        ``SELECT * FROM sales LIMIT 5000``, lints those rows, and returns
        findings such as "In the source rows of `sales` (before aggregation):
        Placeholder numbers [-9999, ...] appear as data in "revenue"".
        """
        findings = []
        for table in tables[:SOURCE_PROFILE_MAX_TABLES]:
            try:
                # get_sql_expr turns a table name into the SELECT that defines it
                # (for a CSV-backed table that is a read_csv call, not a name),
                # and SQLLimit appends the row cap in the source's own dialect.
                limited = SQLLimit(
                    limit=PROFILE_SAMPLE_ROWS, write=source.dialect, pretty=False, identify=False
                ).apply(source.get_sql_expr(table))
                sample = await source.execute_with_timeout(limited, self.query_timeout)
            except Exception as e:
                # A source that cannot be sampled simply contributes nothing;
                # the query it feeds has already run successfully.
                log_debug(f"Could not profile source table {table!r}: {e}")
                continue
            findings.extend(
                f"In the source rows of `{table}` (before aggregation): {finding}"
                for finding in lint_data(sample, actionable_only=True)
            )
        return findings

    async def _clean_data_pass(
        self,
        sql_query: str,
        findings: list[str],
        original_rows: int,
        source: BaseSQLSource,
        messages: list[Message],
        context: TContext,
        step: ChatStep,
    ) -> str:
        """Rewrite ``sql_query`` to fix the data-quality problems in ``findings``.

        The second half of the two-shot: the first pass answered the question,
        this one cleans the answer. Returns the original query whenever the
        rewrite is absent, identical, broken or empty, because a query that
        already ran is worth more than a cleaner one that does not.
        """
        try:
            cleanup = await self._invoke_prompt(
                "clean_data", messages, context, sql=sql_query,
                findings=findings, dialect=source.dialect,
            )
            validate_read_only_sql(cleanup.query, source.dialect)
            cleaned = clean_sql(cleanup.query, source.dialect, prettify=True)
        except Exception as e:
            # Deliberately swallowed rather than raised: _render_execute_query is
            # wrapped in retry_llm_output, so letting this escape would throw away
            # a query that already answered the question and regenerate from zero.
            step.stream(f"\n\n⚠️ Could not clean the query, keeping the original: {e}")
            return sql_query
        if cleaned == sql_query:
            step.stream("\n\n✅ Query already handles the findings; left unchanged")
            return sql_query

        try:
            cleaned_result = await source.execute_with_timeout(cleaned, self.query_timeout)
        except Exception as e:
            step.stream(f"\n\n⚠️ Cleaned query failed, keeping the original: {e}")
            return sql_query

        # An empty result means the cleaning filtered away the answer itself.
        # Silently returning nothing is worse than returning dirty data.
        if cleaned_result.empty:
            step.stream("\n\n⚠️ Cleaned query returned no rows, keeping the original")
            return sql_query

        # Always show the row delta: cleaning that quietly discards half the
        # answer must be visible to whoever reads the result.
        step.stream(
            f"\n\n🧹 Cleaned the query: {cleanup.chain_of_thought}"
            f"\n\n```sql\n{cleaned}\n```"
            f"\n\n{original_rows} rows before cleaning, {len(cleaned_result)} after."
        )
        return cleaned

    async def _execute_query(
        self, source: BaseSQLSource, context: TContext, expr_slug: str, sql_query: str, tables: list[str],
        is_final: bool, should_materialize: bool, step: ChatStep, raise_if_empty: bool = False
    ) -> tuple[Pipeline, str]:
        """Execute SQL query and return pipeline and summary."""
        # Create SQL source
        source_tables = source.tables if source.tables is not None else {}
        if isinstance(source_tables, list):
            source_tables = {t: source.get_sql_expr(t) for t in source_tables}
        table_defs = {table: source_tables[table] for table in tables if table in source_tables}
        expr_slug = self._unique_expr_slug(expr_slug, source_tables)
        table_defs[expr_slug] = sql_query

        # Only pass materialize parameter for DuckDB sources
        if isinstance(source, DuckDBSource):
            sql_expr_source = source.create_sql_expr_source(
                table_defs, materialize=should_materialize
            )
        else:
            sql_expr_source = source.create_sql_expr_source(table_defs)

        # Create pipeline
        if is_final:
            sql_transforms = [SQLLimit(limit=1_000_000, write=source.dialect, pretty=True, identify=False)]
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms)
        else:
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug)

        # Get data summary. The frame is only counted and summarised, and
        # describe_data reads any library, so there is nothing here worth
        # converting a polars or pyarrow result to pandas for.
        df = await get_frame(pipeline)

        # Reject before promoting the new source: retry_llm_output re-runs this
        # method on failure, and a rejected query left installed as
        # ``context["source"]`` becomes the base every later attempt builds on.
        if not len(df) and raise_if_empty:
            raise EmptyResultError(f"Query `{sql_query}`: {EMPTY_RESULT_HINT}")

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

        summary = await describe_data(df, reduce_enums=False)
        if len(summary) >= 1000:
            summary = summary[:1000-3] + "..."
        summary_formatted = f"\n```\n{summary}\n```"
        if should_materialize:
            summary_formatted += f"\n\nMaterialized data: `{sql_expr_source.name}{SOURCE_TABLE_SEPARATOR}{expr_slug}`"
        stream_details(f"{summary_formatted}", step, title=expr_slug)

        return pipeline, expr_slug

    @staticmethod
    def _unique_expr_slug(expr_slug: str, source_tables: dict[str, str]) -> str:
        """
        Return a slug that does not collide with an existing table.

        Materialization issues ``CREATE OR REPLACE VIEW <expr_slug>``, so reusing
        the name of a table the query reads redefines that table in terms of the
        query's own output — destroying the original data for the rest of the
        session and, when the query selects from the same name, producing
        "infinite recursion detected" on every later read. Weak models routinely
        echo back an input table name as ``table_slug`` despite the field
        instructing otherwise, so treat the collision as the agent's problem to
        resolve rather than the model's to get right.
        """
        if expr_slug not in source_tables:
            return expr_slug
        for suffix in range(1, 100):
            candidate = f"{expr_slug}_derived_{suffix}"
            if candidate not in source_tables:
                log_debug(
                    f"table_slug {expr_slug!r} collides with an existing table; "
                    f"materializing as {candidate!r} to avoid overwriting it."
                )
                return candidate
        raise ValueError(f"Could not derive a non-colliding table slug for {expr_slug!r}.")

    async def _finalize_execution(
        self,
        pipeline: Pipeline,
        sql: str,
        step_title: str | None,
    ) -> SQLEditor:
        """Finalize execution for final step."""

        view = self._editor_type(
            component=pipeline, title=step_title, spec=sql
        )
        return view

    def _merge_sources(
        self, sources: dict[tuple[str, str], BaseSQLSource], tables: list[tuple[str, str]]
    ) -> tuple[BaseSQLSource, list[str]]:
        if not tables:
            raise ValueError("Select at least one source table for the SQL query.")
        for table in tables:
            if (table.source, table.table) not in sources:
                raise ValueError(f"Unknown source/table pair: {table.source!r}/{table.table!r}")
        if len(set(m.source for m in tables)) == 1:
            return sources[(tables[0].source, tables[0].table)], list(dict.fromkeys(m.table for m in tables))
        mirrors = {}
        for m in tables:
            if not any(m.table.rstrip(")").rstrip("'").rstrip('"').endswith(ext)
                       for ext in [".csv", ".parquet", ".parq", ".json", ".xlsx"]):
                renamed_table = m.table.replace(".", "_")
            else:
                renamed_table = m.table
            if renamed_table in mirrors and mirrors[renamed_table] != (sources[(m.source, m.table)], m.table):
                raise ValueError(f"Ambiguous mirrored table name {renamed_table!r}; select a single source for that name.")
            mirrors[renamed_table] = (sources[(m.source, m.table)], m.table)
        return DuckDBSource(uri=":memory:", mirrors=mirrors), list(mirrors)

    @staticmethod
    def _active_filters(pipeline: Pipeline | None) -> list[str] | None:
        """Describe the interactive filters currently applied to ``pipeline`` as
        WHERE-style conditions so a follow-up query can preserve the subset.
        Returns None when there is no pipeline or no active filter."""
        if pipeline is None:
            return None
        conditions = []
        for filt in pipeline.filters:
            query = filt.query
            if query is None:
                continue
            if isinstance(query, tuple) and len(query) == 2:
                condition = f"between {query[0]} and {query[1]}"
            elif isinstance(query, (list, set)):
                condition = f"in ({', '.join(repr(v) for v in query)})"
            else:
                condition = f"= {query!r}"
            conditions.append(f"{filt.field} {condition}")
        return conditions or None

    @retry_llm_output()
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
            Whether to raise EmptyResultError if the query returns no rows.
            retry_llm_output clears it for the retry, so no rows is accepted
            once the model has been asked to check its filters.
        output_title : str, optional
            Title to use for the output
        errors : list[str], optional
            List of previous errors to include in prompt

        Returns
        -------
        SQLEditor
            Output object from successful execution
        """
        with self._add_step(title=step_title, steps_layout=self._steps_layout, context_exception="raise") as step:
            # Generate SQL using common prompt pattern
            dialects = set(src.dialect for src in sources.values())
            dialect = "duckdb" if len(dialects) > 1 else next(iter(dialects))

            metaset = context.get("metaset")
            exploration = make_run_exploration_sql_tool(sources, timeout=self.query_timeout)
            if metaset is not None:
                tool_list: list[FunctionTool] = [
                    make_browse_data_catalog_tool(metaset),
                    make_load_table_schemas_tool(metaset),
                    exploration,
                ]
            else:
                tool_list = [exploration]

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
                active_filters=self._active_filters(context.get("pipeline")),
                source_names=sorted({s for s, _ in sources}),
                schema_max_tokens=self.schema_max_tokens,
                schema_tables_shown=self.schema_tables_shown,
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
                if not tables:
                    raise ValueError("Select at least one table for the SQL query.")
                for table in tables:
                    if (source.name, table) not in sources:
                        raise ValueError(f"Unknown source/table pair: {source.name!r}/{table!r}")
            else:
                # Multiple sources - need to merge (output.tables contains SourceTable objects)
                source, tables = self._merge_sources(sources, output.tables)
            sql_query = output.query.strip()
            expr_slug = output.table_slug.strip()

            validated_sql, preview = await self._validate_sql(
                context, sql_query, expr_slug, source, messages,
                step, discovery_context=discovery_context, tools=tool_list,
            )

            # Profile the bounded validation sample without loading the full result.
            findings: list[str] = []
            actionable: list[str] = []
            if preview is not None:
                findings = lint_data(preview)
                # Only the actionable subset justifies (and is shown to) the
                # rewriting pass. Constant columns and outliers are reported but
                # must not provoke a query rewrite.
                if self.clean_data:
                    actionable = lint_data(preview, actionable_only=True)

            if self.clean_data and sql_contains_aggregates(validated_sql, source.dialect):
                source_findings = await self._profile_source_rows(source, tables)
                findings += source_findings
                actionable += source_findings

            if findings:
                stream_details(
                    "\n".join(f"- {finding}" for finding in findings),
                    step, title="Data quality findings", auto=False
                )
            if actionable:
                validated_sql = await self._clean_data_pass(
                    validated_sql, actionable, 0 if preview is None else len(preview),
                    source, messages, context, step
                )

            # Only materialize for DuckDB sources
            should_materialize = isinstance(source, DuckDBSource)

            pipeline, expr_slug = await self._execute_query(
                source, context, expr_slug, validated_sql, tables=tables,
                is_final=True, should_materialize=should_materialize, step=step,
                raise_if_empty=raise_if_empty
            )

            view = await self._finalize_execution(
                pipeline,
                validated_sql,
                output_title,
            )

            if isinstance(step, ChatStep):
                step.param.update(
                    status="success",
                    success_title=success_message
                )

        return view

    async def revise(
        self,
        instruction: str,
        messages: list[Message],
        context: TContext,
        view: LumenEditor | None = None,
        spec: str | None = None,
        language: str | None = None,
        errors: list[str] | None = None,
        max_retries: int = 2,
        **kwargs
    ) -> str:
        result = await super().revise(
            instruction, messages, context, view=view, spec=spec, language=language, **kwargs
        )
        if view is None:
            return result
        dialect = view.component.source.dialect
        revise_language = view.language
        for i in range(max_retries):
            try:
                cleaned = clean_sql(result, dialect, prettify=True)
                view.validate_spec(cleaned)
                return cleaned
            except Exception as e:
                if i == max_retries - 1:
                    raise
                feedback = format_error(e)
                result = await super().revise(
                    instruction, messages, context, spec=result, language=revise_language,
                    errors=[feedback], **kwargs
                )
        return clean_sql(result, dialect, prettify=True)

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[t.Any], SQLOutputs]:
        """Generate SQL in one step (catalog browse, schema load, optional exploration, then structured query)."""
        sources = context["sources"]
        metaset = context["metaset"]
        # Do not pin schema_tables: initial prompt uses a capped summary; the model pulls detail via tools.
        metaset.schema_tables = None

        # Build source lookup, skipping slugs whose source no longer
        # exists in the context (stale entries from prior materializations).
        available_source_names = {s.name for s in sources}
        sources = {}
        for table_slug in metaset.catalog:
            source_name = table_slug.split(SOURCE_TABLE_SEPARATOR)[0] if SOURCE_TABLE_SEPARATOR in table_slug else None
            if source_name and source_name not in available_source_names:
                continue
            key = tuple(table_slug.split(SOURCE_TABLE_SEPARATOR))
            src = parse_table_slug(table_slug, context["sources"])[0]
            if src is not None:
                sources[key] = src
        if not sources:
            raise ValueError("No valid SQL sources available for querying.")
        await metaset.ensure_stats(metaset.get_top_tables(self.schema_tables_shown), timeout=self.stats_wait)

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
        out_context = t.cast(SQLOutputs, await out.render_context())
        return [out], out_context
