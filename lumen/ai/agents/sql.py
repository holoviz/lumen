import asyncio
import math
import typing as t

import narwhals.stable.v2 as nw
import pandas as pd
import param
import sqlglot
import yaml

from panel.chat import ChatStep
from pydantic import (
    BaseModel, Field, create_model, field_validator, model_validator,
)
from pydantic.fields import FieldInfo

from ...filters import ConstantFilter
from ...pipeline import Pipeline
from ...sources.base import BaseSQLSource, Source
from ...sources.duckdb import DuckDBSource
from ...transforms.sql import SQLLimit
from ...util import as_narwhals, as_pandas, is_lazyframe
from ..config import (
    PROMPTS_DIR, SOURCE_TABLE_SEPARATOR, RequestBudgetExceededError,
)
from ..context import ContextModel, TContext
from ..data_quality import lint_data
from ..editors import LumenEditor, SQLEditor
from ..llm import Message, SubmitTool
from ..models import RetrySpec
from ..schemas import Metaset, resolve_table_slug
from ..tool_trace import ModelCall, ToolCall
from ..tools import FunctionTool
from ..utils import (
    PROFILE_SAMPLE_ROWS, clean_sql, describe_data, format_unknown_name,
    get_frame, get_pipeline, log_debug, normalize_object_dtypes,
    parse_table_slug, retry_llm_output, stream_details, truncate_string,
    truncate_to_tokens,
)
from .base_lumen import BaseLumenAgent

if t.TYPE_CHECKING:
    from narwhals.stable.v2.typing import Frame, IntoFrame


def _table_reference_resolver(sources: list[tuple[str, str]]):
    """
    Map a model-supplied table reference to its ``(source, table)`` pair, or None.

    Lets the structured ``tables`` field accept the same references as the
    tools (any case, ``source/<table>``), so a near miss is corrected rather
    than rejected by the Literal validation.
    """
    slugs = {f"{src}{SOURCE_TABLE_SEPARATOR}{table}": (src, table) for src, table in sources}

    def resolve(reference: str) -> tuple[str, str] | None:
        try:
            return slugs[resolve_table_slug(reference, slugs)]
        except ValueError:
            return None
    return resolve


def make_source_table_model(sources: list[tuple[str, str]]):
    resolve = _table_reference_resolver(sources)

    class LiteralSourceTable(BaseModel):
        source: t.Literal[tuple(set(src for src, _ in sources))]
        table: t.Literal[tuple(set(table for _, table in sources))]

        @model_validator(mode="before")
        @classmethod
        def _resolve_reference(cls, data: t.Any) -> t.Any:
            if isinstance(data, str):
                reference = data
            elif isinstance(data, dict) and isinstance(data.get("table"), str):
                source = data.get("source")
                reference = f"{source}/{data['table']}" if source else data["table"]
            else:
                return data
            if (match := resolve(reference)) is None and isinstance(data, dict):
                match = resolve(data["table"])
            return {"source": match[0], "table": match[1]} if match else data
    return LiteralSourceTable


def make_table_model(sources: list[tuple[str, str]]):
    """
    Create a table model with constrained table choices.
    """
    available_tables = list(set(table for _, table in sources))
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
        resolve = _table_reference_resolver(sources)

        def resolve_tables(cls, value: t.Any) -> t.Any:
            if not isinstance(value, list):
                return value
            return [
                match[1] if isinstance(item, str) and (match := resolve(item)) else item
                for item in value
            ]

        return create_model(
            "SQLQueryWithTables",
            tables=(
                list[Table],
                FieldInfo(description="The table name(s) referenced in the SQL query.")
            ),
            __base__=SQLQuery,
            __validators__={"resolve_tables": field_validator("tables", mode="before")(resolve_tables)},
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
# Tables the data summary lists with details, and by name only. The catalog
# browsing tool is only worth offering when the catalog is larger than both.
PROMPT_TABLES = 12
PROMPT_OTHER_TABLES = 25
DISTINCT_VALUES_LIMIT = 50
MAIN_TOOL_ROUNDS = 6
REVISE_TOOL_ROUNDS = 3
SUBMIT_SQL_DESCRIPTION = (
    "Submit the final SQL query as your answer. The system executes it: if it fails, the "
    "error comes back as this tool's result so you can fix the query and submit again; if it "
    "succeeds, you are done. Call this instead of replying with text."
)


def summarize_tool_calls(events: list, max_calls: int = 10, max_chars: int = 400) -> str | None:
    """
    Condense an attempt's tool calls into what a retry needs to know.

    A retry that sees what was already looked up can go straight to fixing
    the query instead of repeating the discovery that preceded the failure.
    """
    calls = [event for event in events if isinstance(event, ToolCall)]
    if not calls:
        return None
    lines = []
    for call in calls[-max_calls:]:
        args = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
        result = " ".join(str(call.result).split())
        lines.append(f"- {call.name}({truncate_string(args, max_chars // 2)}) -> {truncate_string(result, max_chars)}")
    if len(calls) > max_calls:
        lines.insert(0, f"(last {max_calls} of {len(calls)} tool calls)")
    return "\n".join(lines)


class RequestBudget:
    """Caps the LLM calls one SQL request spends across all of its attempts."""

    def __init__(self, events: list, max_calls: int, attempts: list[dict[str, t.Any]]):
        self._events = events
        self.max_calls = max_calls
        self._attempts = attempts

    @property
    def calls(self) -> int:
        return sum(isinstance(event, ModelCall) for event in self._events)

    def last_error(self) -> str:
        if not self._attempts or not self._attempts[-1].get("error"):
            return ""
        return f" Last error: {self._attempts[-1]['error']}"

    def check(self):
        if self.calls >= self.max_calls:
            raise RequestBudgetExceededError(
                f"SQL request used {self.calls} LLM calls, the limit is {self.max_calls}.{self.last_error()}"
            )


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


def _source_slugs(sources: dict[tuple[str, str], BaseSQLSource]) -> dict[str, tuple[str, str]]:
    return {f"{source}{SOURCE_TABLE_SEPARATOR}{table}": (source, table) for source, table in sources}


def resolve_source_table(
    reference: str, sources: dict[tuple[str, str], BaseSQLSource]
) -> tuple[BaseSQLSource, str]:
    """Resolve a model-supplied table reference to its source and table name (see :func:`resolve_table_slug`)."""
    slugs = _source_slugs(sources)
    key = slugs[resolve_table_slug(reference, slugs)]
    return sources[key], key[1]


def resolve_exploration_source(
    source: str | None, sources: dict[tuple[str, str], BaseSQLSource]
) -> BaseSQLSource:
    """
    Resolve the ``source`` argument of an exploration tool call.

    Weak models often pass a table name, a generic ``source/<table>`` or a
    differently cased name instead of the source name, so any reference that
    identifies one source unambiguously is accepted.
    """
    names = sorted({s for s, _ in sources})
    if len(names) == 1 and (not source or source.strip().lower() in {"source", names[0].lower()}):
        return next(iter(sources.values()))
    if not source:
        raise ValueError(f"Several sources are available; pass `source` as one of: {', '.join(names)}.")
    by_name = {name.lower(): name for name in names}
    if (name := by_name.get(source.strip().lower())) is not None:
        return next(obj for (s, _), obj in sources.items() if s == name)
    try:
        return resolve_source_table(source, sources)[0]
    except ValueError:
        if len(names) == 1:
            return next(iter(sources.values()))
    raise ValueError(format_unknown_name("source", source, names))


async def execute_exploration_sql(
    source: str | None,
    sql_query: str,
    *,
    sources: dict[tuple[str, str], BaseSQLSource],
) -> str:
    """
    Run a read-only SQL statement on the named Lumen source and return a text preview of results.

    Intended for LLM tool calls: the model supplies the logical ``source`` name and ``sql_query``;
    the ``sources`` map must be provided by the caller (e.g. closed over when building the tool).

    Parameters
    ----------
    source : str | None
        Name of the data source (key prefix before the source/table separator in slugs).
        May be omitted when there is a single source.
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
        base = resolve_exploration_source(source, sources)
    except ValueError as e:
        return f"{e} Tables: {', '.join(sorted({t for _, t in sources}))}."

    try:
        sql_clean = clean_sql(sql_query.strip(), base.dialect, prettify=False)
        validate_read_only_sql(sql_clean, base.dialect)
        limited = SQLLimit(limit=EXPLORATION_MAX_ROWS + 1, write=base.dialect).apply(sql_clean)
    except (ValueError, sqlglot.errors.ParseError, sqlglot.errors.TokenError) as e:
        return f"SQL parse/clean error: {e}"

    try:
        df = base.fetch(limited)
    except Exception as e:
        return f"{type(e).__name__}: {e}"

    capped = len(df) > EXPLORATION_MAX_ROWS
    return format_exploration_result(df.head(EXPLORATION_MAX_ROWS) if capped else df, capped=capped)


def make_run_exploration_sql_tool(sources: dict[tuple[str, str], BaseSQLSource]) -> FunctionTool:
    """Build a :class:`~lumen.ai.tools.FunctionTool` that runs :func:`execute_exploration_sql` for ``sources``."""

    names = sorted({s for s, _ in sources})
    output = (
        f"Returns the row count (reported as 'at least {EXPLORATION_MAX_ROWS}' when capped), the "
        f"column dtypes and the first {EXPLORATION_PREVIEW_ROWS} rows of up to "
        f"{EXPLORATION_PREVIEW_COLS} columns."
    )
    query_doc = (
        "sql_query : str\n"
        "    One read-only SELECT or WITH statement. Reference tables by name "
        "(SELECT * FROM my_table), not with read_csv() or read_parquet()."
    )
    if len(names) > 1:
        async def run_exploration_sql(sql_query: str, source: str | None = None) -> str:
            return await execute_exploration_sql(source, sql_query, sources=sources)

        params_doc = (
            f"{query_doc}\n"
            "source : str | None\n"
            f"    The source the tables belong to: one of {', '.join(names)}."
        )
    else:
        async def run_exploration_sql(sql_query: str) -> str:
            return await execute_exploration_sql(None, sql_query, sources=sources)

        params_doc = query_doc

    run_exploration_sql.__doc__ = (
        f"Run read-only SQL to inspect data before writing the final query. {output}\n\n"
        f"Parameters\n----------\n{params_doc}\n"
    )
    return FunctionTool(
        run_exploration_sql,
        purpose=(
            f"Run exploratory read-only SQL (SELECT/WITH). {output} "
            "Use it to learn column formats and value domains you cannot see in the data "
            "summary. It does not produce the result the user sees, and running the final "
            "query here first wastes a round: submit the final query directly, since "
            "submission executes it and reports any error."
        ),
    )


def make_distinct_values_tool(
    sources: dict[tuple[str, str], BaseSQLSource], metaset: Metaset | None = None
) -> FunctionTool:
    """
    Tool that lists the most frequent values of one column, optionally filtered by a pattern.

    A focused alternative to free-form exploration for the most common reason
    to explore: finding the exact stored spelling of a literal before
    filtering on it.
    """

    async def distinct_values(table: str, column: str, like: str | None = None) -> str:
        """
        List the most frequent stored values of a column with their row counts.

        Parameters
        ----------
        table : str
            The table name.
        column : str
            The column name.
        like : str | None
            Optional case-insensitive substring; only values containing it are listed.
        """
        return await execute_distinct_values(table, column, like, sources=sources, metaset=metaset)

    return FunctionTool(
        distinct_values,
        purpose=(
            f"List up to {DISTINCT_VALUES_LIMIT} most frequent stored values of one column with "
            "row counts, optionally only those containing `like` (case-insensitive). Use it to "
            "confirm the exact spelling, case and whitespace of a value before filtering on it."
        ),
    )


def _known_columns(metaset: Metaset | None, source: BaseSQLSource, table: str) -> list[str]:
    slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}"
    entry = metaset.catalog.get(slug) if metaset is not None else None
    if entry is not None and entry.columns:
        return [col.name for col in entry.columns]
    schema = (metaset.schemas or {}).get(slug) if metaset is not None else None
    return [key for key in (schema or {}) if key != "__len__"]


def build_distinct_values_sql(
    table_expr: str, column: str, dialect: str, like: str | None = None, limit: int = DISTINCT_VALUES_LIMIT
) -> str:
    """Top ``limit + 1`` values of ``column`` by frequency; the extra row reveals truncation."""
    read = None if dialect == "any" else dialect
    col = sqlglot.exp.column(column, quoted=True)
    count = sqlglot.exp.Count(this=sqlglot.exp.Star())
    query = (
        sqlglot.select(col.as_("value", quoted=True), count.as_("count", quoted=True))
        .from_(sqlglot.parse_one(table_expr, read=read).subquery("_t"))
        .group_by(col.copy())
        .order_by(
            sqlglot.exp.Ordered(this=sqlglot.exp.column("count", quoted=True), desc=True),
            sqlglot.exp.Ordered(this=sqlglot.exp.column("value", quoted=True)),
        )
        .limit(limit + 1)
    )
    if like:
        text = sqlglot.exp.Cast(this=col.copy(), to=sqlglot.exp.DataType.build("text"))
        query = query.where(sqlglot.exp.ILike(this=text, expression=sqlglot.exp.Literal.string(f"%{like}%")))
    return query.sql(dialect=read)


def _format_value(value: t.Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NULL"
    # repr keeps padding and quotes visible, which is what the model needs to copy.
    return repr(value) if isinstance(value, str) else str(value)


async def execute_distinct_values(
    table: str,
    column: str,
    like: str | None = None,
    *,
    sources: dict[tuple[str, str], BaseSQLSource],
    metaset: Metaset | None = None,
    limit: int = DISTINCT_VALUES_LIMIT,
) -> str:
    try:
        source, table_name = resolve_source_table(table, sources)
    except ValueError as e:
        return str(e)
    known = _known_columns(metaset, source, table_name)
    if known and column not in known:
        matches = [name for name in known if name.lower() == column.strip().strip('"').lower()]
        if len(matches) != 1:
            return format_unknown_name("column", column, known)
        column = matches[0]
    try:
        sql = build_distinct_values_sql(source.get_sql_expr(table_name), column, source.dialect, like, limit)
        df = as_pandas(source.execute(sql))
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    rows = list(df.itertuples(index=False, name=None))
    truncated = len(rows) > limit
    rows = rows[:limit]
    matching = f" containing {like!r}" if like else ""
    if not rows:
        return f"No values of {column!r} in {table_name!r}{matching}."
    lines = [f"Values of {column!r} in {table_name!r}{matching}, most frequent first (value: rows):"]
    lines += [f"- {_format_value(value)}: {n}" for value, n in rows]
    if truncated:
        lines.append(f"More than {limit} values match; pass `like` to narrow the list.")
    return "\n".join(lines)


def catalog_exceeds_prompt(metaset: Metaset) -> bool:
    """Whether some tables are missing from the data summary, so browsing the catalog can find more."""
    return len(metaset._deduplicated_slugs()) > PROMPT_TABLES + PROMPT_OTHER_TABLES


def make_browse_data_catalog_tool(metaset: Metaset) -> FunctionTool:
    """
    Tool wrapping :meth:`~lumen.ai.schemas.Metaset.table_list` for incremental catalog browsing.
    """

    def browse_data_catalog(
        offset: int = 0,
        limit: int = 25,
        include_metadata: bool = False,
    ) -> str:
        """
        List catalog tables by relevance, one page at a time.

        Parameters
        ----------
        offset : int
            Number of tables to skip; the data summary already covers the first ones.
        limit : int
            Number of tables to list.
        include_metadata : bool
            Include table descriptions and metadata.
        """
        total = len(metaset._deduplicated_slugs())
        offset, limit = max(0, offset), max(1, min(limit, 100))
        if offset >= total:
            return f"The catalog has {total} tables; offset {offset} is past the end."
        listing = metaset.table_list(
            n=limit, offset=offset, n_others=0,
            include_metadata=include_metadata, include_lineage=True,
        )
        end = min(offset + limit, total)
        return f"Tables {offset + 1}-{end} of {total}, most relevant first:\n{listing}"

    return FunctionTool(
        browse_data_catalog,
        purpose=(
            "Page through catalog tables that the data summary does not list. Returns table "
            "names (and optionally descriptions), not columns; use load_table_schemas for those."
        ),
    )


def make_load_table_schemas_tool(metaset: Metaset) -> FunctionTool:
    """
    Tool to load and return schema details for specific catalog tables, preferring catalog metadata
    then source introspection (cached on the metaset), not ad-hoc exploration SQL.
    """

    async def load_table_schemas(table_slugs: list[str], columns: list[str] | None = None) -> str:
        """
        Return column types, value ranges, enums, descriptions and row counts for tables.

        Parameters
        ----------
        table_slugs : list[str]
            Table names, as listed in the data summary.
        columns : list[str] | None
            Only return these columns (matched case-insensitively across the requested tables).
        """
        if not table_slugs:
            return "No table_slugs provided."
        wanted = {c.strip().strip('"').lower() for c in columns} if columns else None
        found: set[str] = set()
        result: dict[str, t.Any] = {}
        for raw in table_slugs[:8]:
            try:
                slug = resolve_table_slug(raw, metaset.catalog)
            except ValueError as e:
                result[raw] = {"error": str(e)}
                continue
            entry = metaset.catalog[slug]
            block: dict[str, t.Any] = {}
            live = await metaset.get_schema(slug)
            if live:
                block["row_count"] = live.get("__len__")
                schema = {}
                col_info = {col.name: col for col in entry.columns}
                for k, v in live.items():
                    if k == "__len__":
                        continue
                    if wanted is not None:
                        if k.lower() not in wanted:
                            continue
                        found.add(k.lower())
                    # col_info only carries cataloged columns; live keys not
                    # in the catalog (e.g. xarray dim coordinates absent from
                    # a STAC datacube extension) get the live schema as-is.
                    col = col_info.get(k)
                    if col is not None and col.description:
                        schema[k] = {"description": col.description, **(v if isinstance(v, dict) else {"value": v})}
                    else:
                        schema[k] = v
                block["schema"] = schema
            else:
                entry_columns = [
                    c for c in entry.columns if wanted is None or c.name.lower() in wanted
                ]
                found.update(c.name.lower() for c in entry_columns)
                if entry_columns and any(c.description for c in entry_columns):
                    block["columns"] = [
                        {"name": c.name, "description": c.description}
                        for c in entry_columns
                    ]
                elif entry_columns:
                    block["catalog_columns"] = [c.name for c in entry_columns]
                elif not entry.columns:
                    block["note"] = "No catalog columns and no live schema could be loaded for this table."
            result[slug] = block
        parts = [
            truncate_to_tokens(
                yaml.dump({slug: block}, default_flow_style=False, allow_unicode=True, sort_keys=False),
                SCHEMA_MAX_TOKENS // 3,
            )
            for slug, block in result.items()
        ]
        if len(table_slugs) > 8:
            parts.append(f"Only the first 8 tables were processed; request the remaining {len(table_slugs) - 8} separately.")
        if len(parts) > 3:
            parts.append("Each table is capped independently; request fewer tables for more detail.")
        if wanted is not None and (missing := sorted(wanted - found)):
            parts.append(f"Columns not found in the requested tables: {', '.join(missing)}.")
        return "\n".join(parts)

    return FunctionTool(
        load_table_schemas,
        purpose=(
            "Load the full schema of tables: column types, value ranges, enums, descriptions and "
            "row counts. Call it only for what the data summary lacks: tables it lists without "
            "columns, or columns whose details it truncates. Pass `columns` to fetch just those."
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


def sql_is_scalar_aggregate(sql_query: str, dialect: str | None = None) -> bool:
    """
    Whether the outermost SELECT collapses everything into a single row.

    True for ``SELECT COUNT(*) FROM t`` or ``SELECT MAX(x) / MIN(x) FROM t``:
    aggregates in every projection and no GROUP BY. Unparseable SQL returns False.
    """
    try:
        parsed = sqlglot.parse_one(sql_query, read=None if dialect in (None, "any") else dialect)
    except Exception:
        return False
    if not isinstance(parsed, sqlglot.exp.Select) or parsed.args.get("group"):
        return False
    projections = parsed.expressions
    return bool(projections) and all(
        proj.find(sqlglot.exp.AggFunc) is not None and proj.find(sqlglot.exp.Window) is None
        for proj in projections
    )


def referenced_columns(sql_query: str, dialect: str | None = None) -> set[str] | None:
    """
    Lower-cased names of the columns ``sql_query`` references.

    None when every column is referenced (a ``*`` projection) or the query
    cannot be parsed, so callers fall back to considering all columns.
    """
    try:
        parsed = sqlglot.parse_one(sql_query, read=None if dialect in (None, "any") else dialect)
    except Exception:
        return None
    for select in parsed.find_all(sqlglot.exp.Select):
        for proj in select.expressions:
            if isinstance(proj, sqlglot.exp.Star) or (
                isinstance(proj, sqlglot.exp.Column) and isinstance(proj.this, sqlglot.exp.Star)
            ):
                return None
    return {col.name.lower() for col in parsed.find_all(sqlglot.exp.Column) if col.name}


# How to turn numeric text into numbers without failing on the rows that are
# not numeric. Dialects without a non-failing cast only get a plain CAST, and
# only when every value parses.
NUMERIC_CAST_EDITS = {
    "duckdb": '`TRY_CAST("col" AS DOUBLE)` where numbers are stored as text',
    "snowflake": '`TRY_CAST("col" AS DOUBLE)` where numbers are stored as text',
    "tsql": '`TRY_CAST("col" AS FLOAT)` where numbers are stored as text',
    "bigquery": '`SAFE_CAST(col AS FLOAT64)` where numbers are stored as text',
    "sqlite": (
        '`CAST("col" AS REAL)` where numbers are stored as text, only when every value is '
        'numeric (SQLite turns other text into 0)'
    ),
    "postgres": (
        '`CAST(NULLIF(TRIM("col"), \'\') AS DOUBLE PRECISION)` where numbers are stored as text, '
        'only when every non-blank value is numeric (the cast fails otherwise)'
    ),
}


def numeric_cast_edit(dialect: str) -> str:
    return NUMERIC_CAST_EDITS.get(dialect, (
        '`CAST("col" AS DOUBLE)` where numbers are stored as text, only when every value is numeric'
    ))


_RANGE_FIELD_TYPES = ("number", "integer")


def _coerce_filter_scalar(field_schema: dict[str, t.Any], value: t.Any) -> t.Any:
    """Validate one filter value against the field type, converting numeric strings."""
    field_type = field_schema.get("type")
    if value is None:
        return value
    if isinstance(value, (dict, list, tuple)):
        raise ValueError(f"{value!r} is not a single value")
    if field_schema.get("format") in ("date", "date-time", "datetime"):
        try:
            pd.Timestamp(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{value!r} is not a date or datetime") from e
        return value
    if field_type in _RANGE_FIELD_TYPES:
        if isinstance(value, bool):
            raise ValueError(f"{value!r} is not a number")
        if isinstance(value, str):
            try:
                number = float(value)
            except ValueError as e:
                raise ValueError(f"{value!r} is not a number") from e
            return int(number) if field_type == "integer" and number.is_integer() else number
        if not isinstance(value, (int, float)):
            raise ValueError(f"{value!r} is not a number")
        return value
    if field_type == "boolean" and not isinstance(value, bool):
        if str(value).lower() not in ("true", "false"):
            raise ValueError(f"{value!r} is not a boolean")
        return str(value).lower() == "true"
    return value


def _coerce_filter_value(field_schema: dict[str, t.Any], value: t.Any) -> t.Any:
    """Coerce an LLM-supplied filter value to what the Source expects.

    A two-element ``[lo, hi]`` list on a numeric or datetime field becomes a
    ``(lo, hi)`` tuple (interpreted as an inclusive range / ``BETWEEN``); datetime
    bounds are parsed to timestamps. Scalars (equality) and longer lists
    (membership / ``IN``) keep their shape.

    Values that do not fit the field type raise ``ValueError``: an accepted
    filter is later rendered into every follow-up query as an active filter,
    so a malformed one would corrupt SQL well beyond this call.
    """
    is_datetime = field_schema.get("format") in ("date", "date-time", "datetime")
    if isinstance(value, (list, tuple)) and len(value) == 2 and (
        field_schema.get("type") in _RANGE_FIELD_TYPES or is_datetime
    ):
        lo, hi = (_coerce_filter_scalar(field_schema, v) for v in value)
        if is_datetime:
            lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"range start {value[0]!r} is after its end {value[1]!r}")
        return (lo, hi)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("an empty list matches nothing")
        return [_coerce_filter_scalar(field_schema, v) for v in value]
    return _coerce_filter_scalar(field_schema, value)


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

    max_request_llm_calls = param.Integer(default=40, bounds=(1, None), doc="""
        Maximum LLM calls one SQL request may spend across all attempts, tool
        rounds and revisions. Checked before each attempt.""")

    request_timeout = param.Number(default=300, bounds=(0, None), allow_None=True, doc="""
        Wall-clock limit in seconds for one SQL request across all attempts.
        None disables the limit.""")

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
        for i in range(max_retries):
            try:
                validate_read_only_sql(sql_query, source.dialect)
            except ValueError:
                raise
            except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as e:
                if i == max_retries - 1:
                    raise
                retry_result = await self.revise(
                    f"{type(e).__name__}: {e}", messages, context, spec=sql_query,
                    language=f"sql.{source.dialect}", discovery_context=discovery_context, tools=tools,
                    max_tool_rounds=REVISE_TOOL_ROUNDS,
                )
                sql_query = clean_sql(retry_result, source.dialect, prettify=True)
                continue
            try:
                step.stream(f"\n\n`{expr_slug}`\n```sql\n{sql_query}\n```")
                validated = SQLLimit(limit=VALIDATION_MAX_ROWS, write=source.dialect).apply(sql_query)
                result = await source.execute_async(validated)
                step.stream("\n\n✅ SQL validation successful")
                return sql_query, result
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
                    discovery_context=discovery_context, tools=tools, max_tool_rounds=REVISE_TOOL_ROUNDS,
                )
                sql_query = clean_sql(retry_result, source.dialect, prettify=True)
        return sql_query, None

    @staticmethod
    def _profile_source_rows(
        source: BaseSQLSource, tables: list[str], sql_query: str | None = None
    ) -> list[str]:
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

        Only the columns ``sql_query`` references are linted: a placeholder in
        a column the query never reads cannot affect its result.
        """
        columns = referenced_columns(sql_query, source.dialect) if sql_query else None
        findings = []
        for table in tables[:SOURCE_PROFILE_MAX_TABLES]:
            try:
                # get_sql_expr turns a table name into the SELECT that defines it
                # (for a CSV-backed table that is a read_csv call, not a name),
                # and SQLLimit appends the row cap in the source's own dialect.
                limited = SQLLimit(
                    limit=PROFILE_SAMPLE_ROWS, write=source.dialect, pretty=False, identify=False
                ).apply(source.get_sql_expr(table))
                sample = source.execute(limited)
            except Exception as e:
                # A source that cannot be sampled simply contributes nothing;
                # the query it feeds has already run successfully.
                log_debug(f"Could not profile source table {table!r}: {e}")
                continue
            if columns is not None:
                sample = as_pandas(sample)
                sample = sample[[col for col in sample.columns if str(col).lower() in columns]]
                if sample.columns.empty:
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
                numeric_cast=numeric_cast_edit(source.dialect),
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
            cleaned_result = source.execute(cleaned)
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
            raise ValueError(
                f"\nQuery `{sql_query}` returned empty results."
                "\nUse `run_exploration_sql` to check what values actually exist before filtering."
            )

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
        attempts: list[dict[str, t.Any]] | None = None,
        budget: RequestBudget | None = None,
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
        attempts : list[dict], optional
            Filled with the SQL, error and tool findings of each failed attempt,
            so the next attempt starts from them instead of from scratch.
        budget : RequestBudget, optional
            Checked before each attempt; stops retrying once exhausted.

        Returns
        -------
        SQLEditor
            Output object from successful execution
        """
        attempts = attempts if attempts is not None else []
        if budget is not None:
            budget.check()
        attempt: dict[str, t.Any] = {"sql": None, "tools": None}
        try:
            return await self._attempt_query(
                messages, context, sources, step_title, success_message, attempt,
                discovery_context=discovery_context, raise_if_empty=raise_if_empty,
                output_title=output_title, errors=errors,
                previous_attempt=attempts[-1] if attempts else None,
            )
        except Exception as e:
            attempt["error"] = f"{type(e).__name__}: {e}"
            attempts.append(attempt)
            raise

    def _sql_tools(
        self, sources: dict[tuple[str, str], BaseSQLSource], metaset: Metaset | None
    ) -> list[FunctionTool]:
        tools: list[FunctionTool] = []
        if metaset is not None:
            if catalog_exceeds_prompt(metaset):
                tools.append(make_browse_data_catalog_tool(metaset))
            tools.append(make_load_table_schemas_tool(metaset))
        tools.append(make_distinct_values_tool(sources, metaset))
        tools.append(make_run_exploration_sql_tool(sources))
        return tools

    def _resolve_output_tables(
        self, sources: dict[tuple[str, str], BaseSQLSource], output: BaseModel
    ) -> tuple[BaseSQLSource, list[str]]:
        """The source to run ``output.query`` on and the table names it reads."""
        if len({src for src, _ in sources}) > 1:
            return self._merge_sources(sources, output.tables)
        source = next(iter(sources.values()))
        tables = list(dict.fromkeys(output.tables))
        if not tables:
            raise ValueError("Select at least one table for the SQL query.")
        for table in tables:
            if (source.name, table) not in sources:
                raise ValueError(f"Unknown source/table pair: {source.name!r}/{table!r}")
        return source, tables

    def _submit_sql_tool(
        self, sources: dict[tuple[str, str], BaseSQLSource], checked: dict[str, t.Any]
    ) -> SubmitTool:
        """
        Offer the SQL response model as ``submit_sql``, executing each submission.

        Running the query inside the tool loop lets the model fix an error with
        the schema and tool results it already has, instead of a separate
        revise prompt that starts without them. The accepted result is kept in
        ``checked`` so validation does not execute it a second time.
        """
        async def check(output: BaseModel):
            source, tables = self._resolve_output_tables(sources, output)
            validate_read_only_sql(output.query, source.dialect)
            read = None if source.dialect == "any" else source.dialect
            if sqlglot.parse_one(output.query, read=read).find(sqlglot.exp.Table) is None:
                # Seen after repeated errors: the model hardcodes the answer it
                # expects, which executes fine but no longer comes from the data.
                raise ValueError(
                    "The query reads no table. Select from the tables listed in `tables` "
                    "instead of returning literal values."
                )
            sql_query = clean_sql(output.query.strip(), source.dialect, prettify=True)
            limited = SQLLimit(limit=VALIDATION_MAX_ROWS, write=source.dialect).apply(sql_query)
            preview = await source.execute_async(limited)
            if not len(preview) and checked.get("empty") != sql_query:
                # Asked once, since an empty answer is often correct and a model
                # pushed to find rows tends to loosen the filters the user asked for.
                checked["empty"] = sql_query
                raise ValueError(
                    "The query returned no rows. If its filters match the question, submit "
                    "it again unchanged; an empty result is a valid answer. Otherwise check "
                    "the literal values (distinct_values) and fix the filters."
                )
            checked.update(output=output, source=source, tables=tables, sql=sql_query, preview=preview)

        return SubmitTool("submit_sql", SUBMIT_SQL_DESCRIPTION, check)

    async def _attempt_query(
        self,
        messages: list[Message],
        context: TContext,
        sources: dict[tuple[str, str], BaseSQLSource],
        step_title: str,
        success_message: str,
        attempt: dict[str, t.Any],
        discovery_context: str | None = None,
        raise_if_empty: bool = False,
        output_title: str | None = None,
        errors: list[str] | None = None,
        previous_attempt: dict[str, t.Any] | None = None,
    ) -> SQLEditor:
        with self._add_step(title=step_title, steps_layout=self._steps_layout, context_exception="raise") as step:
            # Generate SQL using common prompt pattern
            dialects = set(src.dialect for src in sources.values())
            dialect = "duckdb" if len(dialects) > 1 else next(iter(dialects))

            metaset = context.get("metaset")
            tool_list = self._sql_tools(sources, metaset)
            checked: dict[str, t.Any] = {}
            with self.llm.trace() as events:
                try:
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
                        tool_names=[tool.name for tool in tool_list],
                        tool_rounds=MAIN_TOOL_ROUNDS,
                        prompt_tables=PROMPT_TABLES,
                        prompt_other_tables=PROMPT_OTHER_TABLES,
                        previous_attempt=previous_attempt,
                        tools=tool_list,
                        max_tool_rounds=MAIN_TOOL_ROUNDS,
                        submit_tool=self._submit_sql_tool(sources, checked),
                    )
                finally:
                    attempt["tools"] = summarize_tool_calls(events)

            if not output:
                raise ValueError("No output was generated.")

            if checked.get("output") is output:
                # submit_sql already executed this exact query.
                source, tables = checked["source"], checked["tables"]
                sql_query = checked["sql"]
                attempt["sql"] = sql_query
                expr_slug = output.table_slug.strip()
                step.stream(f"\n\n`{expr_slug}`\n```sql\n{sql_query}\n```\n\n✅ SQL validation successful")
                validated_sql, preview = sql_query, checked["preview"]
                # The model resubmitted an empty result after being asked to check it.
                raise_if_empty = raise_if_empty and checked.get("empty") != sql_query
            else:
                source, tables = self._resolve_output_tables(sources, output)
                sql_query = output.query.strip()
                attempt["sql"] = sql_query
                expr_slug = output.table_slug.strip()
                validated_sql, preview = await self._validate_sql(
                    context, sql_query, expr_slug, source, messages,
                    step, discovery_context=discovery_context, tools=tool_list,
                )
            attempt["sql"] = validated_sql

            # Profile the bounded validation sample without loading the full result.
            findings: list[str] = []
            actionable: list[str] = []
            # A single aggregated row gives the profile nothing to work with,
            # and cleaning such queries was neutral on accuracy at extra cost.
            clean = self.clean_data and not sql_is_scalar_aggregate(validated_sql, source.dialect)
            if preview is not None:
                findings = lint_data(preview)
                # Only the actionable subset justifies (and is shown to) the
                # rewriting pass. Constant columns and outliers are reported but
                # must not provoke a query rewrite.
                if clean:
                    actionable = lint_data(preview, actionable_only=True)

            if clean and sql_contains_aggregates(validated_sql, source.dialect):
                source_findings = self._profile_source_rows(source, tables, validated_sql)
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
                feedback = f"{type(e).__name__}: {e!s}"
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

        attempts: list[dict[str, t.Any]] = []
        with self.llm.trace() as events:
            budget = RequestBudget(events, self.max_request_llm_calls, attempts)
            try:
                async with asyncio.timeout(self.request_timeout) as deadline:
                    out = await self._render_execute_query(
                        messages,
                        context,
                        sources=sources,
                        step_title="Generating SQL...",
                        success_message="SQL generation successful",
                        discovery_context=None,
                        raise_if_empty=True,
                        output_title=step_title,
                        attempts=attempts,
                        budget=budget,
                    )
            except TimeoutError as e:
                if not deadline.expired():
                    raise
                raise RequestBudgetExceededError(
                    f"SQL request exceeded its {self.request_timeout}s time limit.{budget.last_error()}"
                ) from e
        out_context = t.cast(SQLOutputs, await out.render_context())
        return [out], out_context
