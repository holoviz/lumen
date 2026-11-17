"""Model-free tools over the Lumen source nodes and query results of the open dashboard."""
from __future__ import annotations

import weakref

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from ..ai.agents.sql import execute_distinct_values, execute_exploration_sql
from ..ai.data_quality import lint_data
from ..ai.schemas import Metaset, get_metaset
from ..config import SOURCE_TABLE_SEPARATOR
from ..flowdash import SOURCE_ID
from ..sources.base import BaseSQLSource, Source
from ..util import as_pandas
from . import _graph

QUERY_TIMEOUT = 30
STATS_TIMEOUT = 30

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


async def list_tables(ctx: RunContext[FlowDashDeps], source: str | None = None) -> str:
    """List the tables of a source node with their columns.

    Args:
        source: The name of a source node; may be left out when there is only one.
    """
    name, src = resolve_source(ctx.deps, source)
    tables = src.get_tables()
    meta = await metaset(src)
    return f"{len(tables)} table(s) in source {name!r} ({src.dialect}):\n" + meta.table_context(n=None, include_lineage=False)


async def describe_table(
    ctx: RunContext[FlowDashDeps], table: str, source: str | None = None, columns: list[str] | None = None
) -> str:
    """Describe the columns of a table: SQL type, keys, value ranges or most common values, and null share.

    Args:
        table: The table name, as list_tables gives it.
        source: The name of a source node; may be left out when there is only one.
        columns: Only describe these columns.
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
    """List the most frequent stored values of a column with their row counts, to filter on the exact spelling.

    Args:
        table: The table name.
        column: The column name.
        like: Only list values containing this text, ignoring case.
        source: The name of a source node; may be left out when there is only one.
    """
    _, src = resolve_source(ctx.deps, source)
    return await execute_distinct_values(
        table, column, like, sources=_sources(src), metaset=await metaset(src), timeout=QUERY_TIMEOUT,
    )


async def run_sql(ctx: RunContext[FlowDashDeps], sql: str, source: str | None = None) -> str:
    """Run a read-only query to inspect the data and return its row count, column types and first rows.

    The result is not placed on the dashboard; use add_sql for results the user should see.

    Args:
        sql: One SELECT or WITH query.
        source: The name of a source node; may be left out when there is only one.
    """
    _, src = resolve_source(ctx.deps, source)
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


def data_toolset() -> FunctionToolset[FlowDashDeps]:
    """The data tools: list_tables, describe_table, distinct_values, run_sql and check_data."""
    return FunctionToolset[FlowDashDeps](
        [list_tables, describe_table, distinct_values, run_sql, check_data], max_retries=2,
    )
