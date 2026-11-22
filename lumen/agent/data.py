"""Model-free tools over the Lumen source nodes and query results of the open dashboard."""
from __future__ import annotations

import weakref

import sqlglot

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from ..ai.agents.sql import (
    PROMPT_OTHER_TABLES, SCHEMA_MAX_TOKENS, execute_distinct_values,
    execute_exploration_sql,
)
from ..ai.data_quality import lint_data
from ..ai.schemas import Metaset, get_metaset
from ..ai.utils import clean_sql, format_error
from ..config import SOURCE_TABLE_SEPARATOR
from ..flowdash import (
    DATA_ID, FILTER_ID, QUERY_TIMEOUT, SOURCE_ID, SQL_ID, TABLE_ID, VIEW_ID,
    check_query,
)
from ..flowdash.nodes import effective_limit
from ..flowdash.results import expand
from ..sources.base import BaseSQLSource, Source
from ..util import as_pandas
from . import _graph

STATS_TIMEOUT = 30
# Tables SQLAgent's data summary describes, as its schema_tables_shown default; PROMPT_OTHER_TABLES more are named.
SCHEMA_TABLES_SHOWN = 25
# Placed nodes list_tables shows, and the characters of SQL it shows of each.
NODES_SHOWN = 12
# A node from a shared dashboard may hold any amount of SQL.
NODE_SQL_CHARS = 300
NODE_COLUMNS = 12

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


def resolve_input(deps: FlowDashDeps, name: str | None) -> tuple[str, BaseSQLSource]:
    """
    The node named ``name`` and the source a query reading it runs on: an
    earlier SQL node's result, whose result table the query can read by
    name, or a source as ``resolve_source`` finds it. A name that matches no
    node falls back to ``resolve_source``, e.g. to the only source; any other
    node is refused.
    """
    try:
        port = _graph.input_port(deps.ops, name) if name else None
        if port is None:
            return resolve_source(deps, name)
        value = _graph.output(deps.ops, *port.rsplit(".", 1))
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    if not isinstance(value, BaseSQLSource):
        raise ModelRetry(f"SQL node {name!r} has no result yet.")
    return name, value


def _result_line(ops, name: str) -> str | None:
    """A SQL node's result as a table a later query can read: its name, rows and typed columns."""
    try:
        pipeline = _graph.output(ops, name, "pipeline")
        frame = _graph.output(ops, name, "data")
    except OperationError:
        return None
    if pipeline is None or frame is None:
        return None
    limit = _graph.config_of(ops, name).get("limit")
    ceiling = effective_limit(ops.editor, limit if isinstance(limit, int) else None)
    rows = f"{len(frame)} rows shown, the tile's limit; a query reads every row" if len(frame) >= ceiling else f"{len(frame)} row{'s' * (len(frame) != 1)}"
    types = {column: spec.get("type", "") for column, spec in (pipeline.schema or {}).items() if isinstance(spec, dict)}
    columns = ", ".join(f"{column} {types.get(str(column), '')}".strip() for column in frame.columns)
    return f"- {name}: table {pipeline.table!r} ({rows}); columns: {columns}"


def earlier_results(ops) -> str | None:
    """
    The results of the SQL nodes on the dashboard, most recent first, which
    a new query reads by table name when ``add_sql`` gets the node as its
    source. Their columns come from the results, not from the catalog of
    the source's tables.
    """
    names = [n for n in reversed(_graph.nodes(ops, SQL_ID))][:NODES_SHOWN]
    lines = [line for name in names if (line := _result_line(ops, name)) is not None]
    if not lines:
        return None
    return "Earlier results you can query by table name (pass the node as add_sql's source):\n" + "\n".join(lines)


def _node_line(ops, name: str, sql_chars: int | None) -> str | None:
    component = _graph.component_of(ops, name)
    config = _graph.config_of(ops, name)
    if component == SQL_ID:
        try:
            frame = _graph.output(ops, name, "data")
            names = [str(c) for c in frame.columns] if frame is not None else []
            if sql_chars is not None and len(names) > NODE_COLUMNS:
                names = [*names[:NODE_COLUMNS], f"and {len(names) - NODE_COLUMNS} more"]
            columns = f"columns: {', '.join(names)}" if frame is not None else "no result"
        except OperationError as exc:
            columns = str(exc)
        sql = " ".join(str(config.get("sql", "")).split())
        if sql_chars is not None and len(sql) > sql_chars:
            sql = f"{sql[:sql_chars]} [... {len(sql) - sql_chars} more characters]"
        return f"- {name} (SQL; {columns}): {sql}"
    if component == VIEW_ID:
        spec = config.get("spec")
        spec = spec if isinstance(spec, dict) else {}
        mark = spec.get("mark")
        mark = mark.get("type") if isinstance(mark, dict) else mark
        fields = ", ".join(
            f"{channel}={definition['field']}" for channel, definition in (spec.get("encoding") or {}).items()
            if isinstance(definition, dict) and "field" in definition
        )
        return f"- {name} (chart of {_graph.upstream(ops, name)}; {mark or 'composite'}; {fields})"
    if component == TABLE_ID:
        limit = f", limit {config['limit']}" if config.get("limit") else ""
        return f"- {name} (table {config.get('table')!r} of {_graph.upstream(ops, name)}{limit})"
    if component == FILTER_ID:
        return f"- {name} (filter on {config.get('field')!r} of {_graph.upstream(ops, name)})"
    if component == DATA_ID:
        return f"- {name} (rows of {_graph.upstream(ops, name)})"
    return None


# The nodes list_tables shows besides the sources.
LISTED = (SQL_ID, VIEW_ID, TABLE_ID, FILTER_ID, DATA_ID)


def placed_nodes(ops) -> str | None:
    """
    The SQL, chart, table, filter and data nodes on the dashboard, most recently placed first:
    each SQL node's query on one line, cut after ``NODE_SQL_CHARS``
    characters, and its result's columns, and each chart's mark, data node
    and encoded fields, and the table, field or input of each table,
    filter and data node, so the agent can change a node it did not place
    itself. At most ``NODES_SHOWN`` are listed, followed by how many more
    there are.
    """
    limit = NODES_SHOWN
    names = [n for n in reversed(_graph.nodes(ops)) if _graph.component_of(ops, n) in LISTED]
    lines = [_node_line(ops, name, NODE_SQL_CHARS) for name in names[:limit]]
    if len(names) > limit:
        lines.append(f"- and {len(names) - limit} more; list_tables(node=...) shows one in full.")
    if lines and (results := earlier_results(ops)):
        lines += ["", results]
    return "\n".join(lines) or None


async def list_tables(ctx: RunContext[FlowDashDeps], source: str | None = None, node: str | None = None) -> str:
    """List a source node's tables with column types and stats, its SQL dialect and the nodes already placed.

    Args:
        source: Source node; defaults to the only one.
        node: Only show this placed node, with its full SQL.
    """
    if node is not None:
        if node not in _graph.nodes(ctx.deps.ops):
            raise ModelRetry(f"No node {node!r} on the dashboard.")
        return _node_line(ctx.deps.ops, node, None) or f"{node!r} is not a SQL or chart node."
    name, src = resolve_source(ctx.deps, source)
    tables = src.get_tables()
    meta = await metaset(src)
    others = ", ".join(n for n in _graph.nodes(ctx.deps.ops, SOURCE_ID) if n != name)
    text = ""
    if placed := placed_nodes(ctx.deps.ops):
        text += "Nodes on the dashboard, most recent first:\n" + placed + "\n\n"
    text += f"{len(tables)} table(s) in source {name!r} ({src.dialect} SQL"
    text += f"; other source nodes: {others}):\n" if others else "):\n"
    text += meta.compact_context(
        n=SCHEMA_TABLES_SHOWN, n_others=PROMPT_OTHER_TABLES, include_lineage=False, max_tokens=SCHEMA_MAX_TOKENS,
    )
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
    _, src = resolve_input(ctx.deps, source)
    try:
        cleaned = clean_sql(sql.strip(), src.dialect)
        check_query(cleaned, src)
        root, statement = expand(cleaned, src)
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        return f"SQL parse/clean error: {format_error(exc)}"
    return await execute_exploration_sql(None, statement, sources=_sources(root), timeout=QUERY_TIMEOUT)


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
