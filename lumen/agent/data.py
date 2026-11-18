"""Model-free tools over the Lumen source nodes and query results of the open dashboard."""
from __future__ import annotations

import weakref

import sqlglot

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from ..ai.agents.sql import (
    PROMPT_OTHER_TABLES, execute_distinct_values, execute_exploration_sql,
)
from ..ai.data_quality import lint_data
from ..ai.schemas import Metaset, get_metaset
from ..ai.utils import clean_sql, format_error
from ..config import SOURCE_TABLE_SEPARATOR
from ..flowdash import (
    QUERY_TIMEOUT, SOURCE_ID, SQL_ID, VIEW_ID, check_query,
)
from ..sources.base import BaseSQLSource, Source
from ..util import as_pandas
from . import _graph

STATS_TIMEOUT = 30
# Tables SQLAgent's data summary describes, as its schema_tables_shown default; PROMPT_OTHER_TABLES more are named.
SCHEMA_TABLES_SHOWN = 25

DIALECT_NOTES = {
    "duckdb": """\
DuckDB notes:
- Quote column names with double quotes and strings with single quotes.
- Range predicates: "date" >= '2023-06-01' AND "date" < '2023-09-01'.
- Window filtering: QUALIFY ROW_NUMBER() OVER (...) <= 5.
- Use TRY_CAST for conversions that may fail.""",
    "snowflake": "Snowflake notes:\n- No quotes around database, schema or table names.",
}

_METASETS: weakref.WeakKeyDictionary[Source, Metaset] = weakref.WeakKeyDictionary()


def resolve_source(deps: FlowDashDeps, source: str | None) -> tuple[str, BaseSQLSource]:
    """
    The source node named ``source`` and its live Source. With a single
    source node any name, or none, refers to it, since models often pass a
    table name instead.
    """
    ops = deps.ops
    names = _graph.nodes(ops, SOURCE_ID)
    if source not in names:
        if len(names) != 1:
            listed = ", ".join(names) or "none"
            raise ModelRetry(f"No source node {source!r}. The source nodes are: {listed}.")
        source = names[0]
    try:
        value = _graph.output(ops, source, "source")
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    if not isinstance(value, BaseSQLSource):
        raise ModelRetry(f"Source node {source!r} has no SQL source yet.")
    return source, value


async def metaset(source: Source) -> Metaset:
    """The catalog and column statistics of every table of ``source``, computed once per source."""
    if source not in _METASETS:
        slugs = [f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}" for table in source.get_tables()]
        _METASETS[source] = await get_metaset([source], slugs, stats_timeout=STATS_TIMEOUT)
    return _METASETS[source]


def _sources(source: BaseSQLSource) -> dict[tuple[str, str], BaseSQLSource]:
    return {(source.name, table): source for table in source.get_tables()}


def placed_nodes(ops) -> str | None:
    """The SQL and chart nodes on the dashboard, with the columns of each SQL node's result."""
    lines = []
    for name in _graph.nodes(ops):
        component = _graph.component_of(ops, name)
        if component == SQL_ID:
            try:
                frame = _graph.output(ops, name, "data")
                columns = f"columns: {', '.join(map(str, frame.columns))}" if frame is not None else "no result"
            except OperationError as exc:
                columns = str(exc)
            lines.append(f"- {name} (SQL; {columns})")
        elif component == VIEW_ID:
            lines.append(f"- {name} (chart)")
    return "\n".join(lines) or None


async def list_tables(ctx: RunContext[FlowDashDeps], source: str | None = None) -> str:
    """List a source node's tables and columns, its SQL dialect and the nodes already placed.

    Args:
        source: Source node; defaults to the only one.
    """
    name, src = resolve_source(ctx.deps, source)
    tables = src.get_tables()
    meta = await metaset(src)
    others = ", ".join(n for n in _graph.nodes(ctx.deps.ops, SOURCE_ID) if n != name)
    text = ""
    if placed := placed_nodes(ctx.deps.ops):
        text += "Nodes on the dashboard:\n" + placed + "\n\n"
    text += f"{len(tables)} table(s) in source {name!r} ({src.dialect} SQL"
    text += f"; other source nodes: {others}):\n" if others else "):\n"
    text += meta.table_context(n=SCHEMA_TABLES_SHOWN, n_others=PROMPT_OTHER_TABLES, include_lineage=False)
    if src.dialect in DIALECT_NOTES:
        text += "\n\n" + DIALECT_NOTES[src.dialect]
    return text


async def describe_table(
    ctx: RunContext[FlowDashDeps], table: str, source: str | None = None, columns: list[str] | None = None
) -> str:
    """Describe a table's columns: type, keys, ranges or common values, null share.

    Args:
        table: Table name.
        source: Source node; defaults to the only one.
        columns: Only these columns.
    """
    _, src = resolve_source(ctx.deps, source)
    meta = await metaset(src)
    slug = f"{src.name}{SOURCE_TABLE_SEPARATOR}{table.strip()}"
    if slug not in meta.catalog:
        raise ModelRetry(f"No table {table!r}. Tables: {', '.join(src.get_tables())}.")
    await meta.ensure_stats([slug], timeout=STATS_TIMEOUT)
    wanted = {c.strip().strip('"').lower() for c in columns} if columns else None
    return meta.table_detail(slug, include_lineage=False, columns=wanted)


async def distinct_values(
    ctx: RunContext[FlowDashDeps], table: str, column: str, like: str | None = None, source: str | None = None
) -> str:
    """List a column's most frequent stored values with row counts.

    Args:
        table: Table name.
        column: Column name.
        like: Only values containing this text, ignoring case.
        source: Source node; defaults to the only one.
    """
    _, src = resolve_source(ctx.deps, source)
    return await execute_distinct_values(
        table, column, like, sources=_sources(src), metaset=await metaset(src), timeout=QUERY_TIMEOUT,
    )


async def run_sql(ctx: RunContext[FlowDashDeps], sql: str, source: str | None = None) -> str:
    """Run a read-only query for your own checks; returns row count, column types and first rows.

    Args:
        sql: One SELECT or WITH query.
        source: Source node; defaults to the only one.
    """
    _, src = resolve_source(ctx.deps, source)
    try:
        check_query(clean_sql(sql.strip(), src.dialect), src)
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        return f"SQL parse/clean error: {format_error(exc)}"
    return await execute_exploration_sql(None, sql, sources=_sources(src), timeout=QUERY_TIMEOUT)


def lint(frame) -> str:
    findings = lint_data(as_pandas(frame))
    return "\n".join(f"- {finding}" for finding in findings) if findings else "No data-quality problems found."


def check_data(ctx: RunContext[FlowDashDeps], node: str) -> str:
    """Check the result of a SQL node for missing values, duplicate rows, placeholder numbers and similar problems.

    Args:
        node: The name of a SQL node.
    """
    try:
        frame = _graph.output(ctx.deps.ops, node, "data")
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    if frame is None:
        return f"{node!r} has no result yet."
    return lint(frame)


def data_toolset(check: bool = False) -> FunctionToolset[FlowDashDeps]:
    """
    The data tools: list_tables, describe_table, distinct_values and run_sql,
    and with ``check`` check_data, whose findings add_sql already reports.
    """
    return FunctionToolset[FlowDashDeps](
        [list_tables, describe_table, distinct_values, run_sql, *([check_data] if check else [])], max_retries=2,
    )
