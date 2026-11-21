"""Model-free tools over the Lumen sources and query results of the open dashboard."""
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
    AGGREGATE_ID, DATA_ID, FILTER_ID, METRIC_ID, QUERY_TIMEOUT, SELECT_ID,
    SORT_ID, SOURCE_ID, SQL_ID, TABLE_ID, VIEW_ID, bound_sources, check_query,
    host_metrics, listed_tables, lookup_source,
)
from ..flowdash.metrics import Metric
from ..flowdash.transforms import TRANSFORM_COMPONENTS
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
# The characters of a heading's, Markdown's or divider's text it shows.
NODE_TEXT_CHARS = 120
NODE_COLUMNS = 12

_METASETS: weakref.WeakKeyDictionary[Source, Metaset] = weakref.WeakKeyDictionary()


def _sources_by_name(deps: FlowDashDeps) -> dict[str, tuple[str | None, Source | None]]:
    """
    The sources the agent can read, by name: each source node by its node
    name, then each source the app provides that no source node shows by its
    ref, or by ``app:<ref>`` when a node has that name, as
    ``(node or None, source)``. A node that has not resolved has no source.
    """
    ops = deps.ops
    found: dict[str, tuple[str | None, Source | None]] = {}
    for name in _graph.nodes(ops, SOURCE_ID):
        try:
            value = _graph.output(ops, name, "source")
        except OperationError:
            value = None
        found[name] = (name, value if isinstance(value, Source) else None)
    shown = {id(value) for _, value in found.values() if value is not None}
    for ref, value in bound_sources(ops.editor).items():
        if id(value) not in shown:
            found[ref if ref not in found else f"app:{ref}"] = (None, value)
            shown.add(id(value))
    return found


def _find(deps: FlowDashDeps, source: str | None, table: str | None = None) -> tuple[str, str | None, Source]:
    found = _sources_by_name(deps)
    if source in found:
        name = source
    elif source is not None and (value := lookup_source(source, deps.ops.editor.context)) is not None:
        name = next((n for n, (_, v) in found.items() if v is value), source)
        found.setdefault(name, (None, value))
    else:
        named = list(found)
        if table is not None and len(found) > 1:
            named = [n for n, (_, v) in found.items() if v is not None and table.strip() in v.get_tables()] or named
        if len(named) != 1:
            listed = ", ".join(found) or "none"
            raise ModelRetry(f"No source {source!r}. The sources are: {listed}.")
        name = named[0]
    node, value = found[name]
    if not isinstance(value, BaseSQLSource):
        raise ModelRetry(f"Source {name!r} has no SQL source yet.")
    return name, node, value


def resolve_source(deps: FlowDashDeps, source: str | None, table: str | None = None) -> tuple[str, BaseSQLSource]:
    """
    The name and live Source of a source the agent can read: a source node
    on the dashboard or a source the app provides, by its ref. With a single
    source any name, or none, refers to it, since models often pass a table
    name instead; with several, ``table`` picks the one that has it.
    """
    name, _, value = _find(deps, source, table)
    return name, value


def source_output(deps: FlowDashDeps, source: str) -> tuple[str, BaseSQLSource, str | None]:
    """
    The ``source`` output a new node connects to for ``source``, its Source
    and the source node placed for it, if the source was not on the
    dashboard yet: a source the app provides is placed as a source node
    named by its ref.
    """
    name, node, value = _find(deps, source)
    if node is not None:
        return f"{node}.source", value, None
    ops = deps.ops
    ref = next((r for r, v in bound_sources(ops.editor).items() if v is value), name)
    taken = set(_graph.nodes(ops))
    node, n = ref, 1
    while node in taken:
        n += 1
        node = f"{ref}_source" if n == 2 else f"{ref}_source_{n - 1}"
    allow_host_sources(ops)
    try:
        ops.add_node(SOURCE_ID, {"ref": ref}, name=node)
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    except PermissionError as exc:
        raise PermissionError(
            f"{exc} Query a source node already on the dashboard, or an earlier SQL node, instead."
        ) from exc
    return f"{node}.source", value, node


def allow_host_sources(ops) -> None:
    """
    Let operations that vet the data sources an agent adds, like
    panel-flowdash's ``ExplorationOps`` through its ``allow_source``, add a
    source node that names a source the app provides by its ref alone. Such
    a node opens no connection of its own; any other source stays up to the
    operations' own check.
    """
    allow = getattr(ops, "allow_source", None)
    if not callable(allow) or getattr(allow, "host_sources", False):
        return

    def allow_source(component: str, config: dict) -> bool:
        if component == SOURCE_ID and set(config) == {"ref"} and config["ref"] in bound_sources(ops.editor):
            return True
        return allow(component, config)

    allow_source.host_sources = True
    ops.allow_source = allow_source


async def metaset(source: Source) -> Metaset:
    """The catalog and column statistics of every table of ``source``, computed once per source."""
    if source not in _METASETS:
        slugs = [f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}" for table in listed_tables(source)]
        _METASETS[source] = await get_metaset([source], slugs, stats_timeout=STATS_TIMEOUT)
    return _METASETS[source]


def _sources(source: BaseSQLSource) -> dict[tuple[str, str], BaseSQLSource]:
    return {(source.name, table): source for table in source.get_tables()}


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
        return f"- {name} (table {config.get('table')!r} of {_graph.upstream(ops, name) or config.get('ref')}{limit})"
    if component == FILTER_ID:
        return f"- {name} (filter on {config.get('field')!r} of {_graph.upstream(ops, name)})"
    if component == DATA_ID:
        return f"- {name} (rows of {_graph.upstream(ops, name)})"
    if component in _graph.TEXT_FIELDS:
        field = _graph.TEXT_FIELDS[component]
        text = " ".join(str(config.get(field) or "").split())
        if sql_chars is not None and len(text) > NODE_TEXT_CHARS:
            text = f"{text[:NODE_TEXT_CHARS]} [... {len(text) - NODE_TEXT_CHARS} more characters]"
        kind = component.split("/")[1].lower()
        return f"- {name} ({kind}; {field}): {text!r}"
    if component == METRIC_ID:
        picked = [f"{key} {', '.join(map(str, config[key]))}" for key in ("measures", "dimensions", "order_by")
                  if config.get(key)]
        picked += [f"{key} {config[key]}" for key in ("grain", "limit") if config.get(key)]
        if parameters := _graph.inputs_of(ops, name).get("parameters"):
            picked.append(f"parameters {parameters}")
        try:
            pipeline = _graph.output(ops, name, "pipeline")
            columns = f"columns: {', '.join(c for c in pipeline.schema if c != '__len__')}"
        except OperationError as exc:
            columns = str(exc)
        return f"- {name} (metric {config.get('metric')!r}; {'; '.join([*picked, columns])})"
    if component in TRANSFORM_COMPONENTS:
        return f"- {name} ({_transform_text(component, config)} of {_graph.upstream(ops, name)})"
    return None


def _transform_text(component: str, config: dict) -> str:
    def names(key: str) -> str:
        return ", ".join(map(repr, config.get(key) or [])) or "nothing"

    if component == AGGREGATE_ID:
        by = f" by {names('by')}" if config.get("by") else ""
        return f"{config.get('method') or 'sum'} of {names('columns')}{by}"
    if component == SORT_ID:
        limit = f", first {config['limit']}" if config.get("limit") else ""
        return f"sorted by {names('by')}{' descending' if config.get('descending') else ''}{limit}"
    if component == SELECT_ID:
        return f"columns {names('columns')}"
    return f"column {config.get('column')!r} = {config.get('expression')}"


# The nodes list_tables shows besides the sources.
LISTED = (SQL_ID, METRIC_ID, VIEW_ID, TABLE_ID, FILTER_ID, DATA_ID, *TRANSFORM_COMPONENTS, *_graph.TEXT_FIELDS)


def _field(name: str, description: str, *notes: str | None) -> str:
    details = [description, *(note for note in notes if note)]
    return f"{name} ({'; '.join(d for d in details if d)})" if any(details) else name


def metric_line(metric: Metric) -> str:
    """A metric as list_tables shows it: what it is, its fields with their descriptions and formats, and its parameters."""
    where = f"source {metric.source}, " + (f"table {metric.table}" if metric.table else "SQL")
    lines = [f"- {metric.name} ({where}): {metric.description}".rstrip(": ")]
    if metric.form == "sql":
        sql = " ".join(metric.sql.split())
        if len(sql) > NODE_SQL_CHARS:
            sql = f"{sql[:NODE_SQL_CHARS]} [... {len(sql) - NODE_SQL_CHARS} more characters]"
        lines.append(f"  query: {sql}")
    if metric.measures:
        lines.append("  measures: " + ", ".join(
            _field(m.name, m.description, m.format and f"format {m.format}") for m in metric.measures.values()))
    if metric.dimensions:
        lines.append("  dimensions: " + ", ".join(
            _field(d.name, d.description, d.time and f"time, grains {'/'.join(d.grains)}") for d in metric.dimensions.values()))
    if metric.parameters:
        lines.append("  parameters: " + ", ".join(
            f"{p.name} ({'; '.join(filter(None, [p.description, p.describe()]))})" for p in metric.parameters.values()))
    return "\n".join(lines)


def metrics_listing(ops) -> str | None:
    """The metrics the app defines, for list_tables, or None without any."""
    metrics = host_metrics(ops.editor.context)
    if not metrics:
        return None
    return "\n".join(metric_line(metric) for metric in metrics.values())


def placed_nodes(ops) -> str | None:
    """
    The SQL, metric, chart, table, filter, data, transform and text nodes
    on the dashboard, most recently placed first: each SQL node's query on
    one line, cut after ``NODE_SQL_CHARS`` characters, and its result's
    columns, each chart's mark, data node and encoded fields, the table,
    field or input of each table, filter and data node, the step of each
    transform node and the start of each text's content, so the agent can
    change a node it did not place
    itself. At most ``NODES_SHOWN`` are listed, followed by how many more
    there are.
    """
    limit = NODES_SHOWN
    names = [n for n in reversed(_graph.nodes(ops)) if _graph.component_of(ops, n) in LISTED]
    lines = [_node_line(ops, name, NODE_SQL_CHARS) for name in names[:limit]]
    if len(names) > limit:
        lines.append(f"- and {len(names) - limit} more; list_tables(node=...) shows one in full.")
    return "\n".join(lines) or None


async def list_tables(ctx: RunContext[FlowDashDeps], source: str | None = None, node: str | None = None) -> str:
    """List the app's metrics, a source's tables with column types and stats, its SQL dialect and the nodes already placed.

    Args:
        source: Source name; defaults to the only one.
        node: Only show this placed node, with its full SQL.
    """
    ops = ctx.deps.ops
    if node is not None:
        if node not in _graph.nodes(ops):
            raise ModelRetry(f"No node {node!r} on the dashboard.")
        return _node_line(ops, node, None) or f"{node!r} is not a SQL or chart node."
    text = ""
    if placed := placed_nodes(ops):
        text += "Nodes on the dashboard, most recent first:\n" + placed + "\n\n"
    if metrics := metrics_listing(ops):
        text += ("Metrics, the app's agreed definitions; place one with add_metric. Column dimensions "
                 "can be filtered by name through parameters:\n" + metrics + "\n\n")
    found = _sources_by_name(ctx.deps)
    if source is None and len(found) > 1:
        lines = [
            f"- {name} ({getattr(value, 'dialect', None) or 'no'} SQL): {', '.join(listed_tables(value)) if value else 'not resolved'}"
            for name, (_, value) in found.items()
        ]
        return text + f"{len(found)} sources; pass one as source for its column stats:\n" + "\n".join(lines)
    name, src = resolve_source(ctx.deps, source)
    tables = listed_tables(src)
    meta = await metaset(src)
    others = ", ".join(n for n in found if n != name)
    text += f"{len(tables)} table(s) in source {name!r} ({src.dialect} SQL"
    text += f"; other sources: {others}):\n" if others else "):\n"
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
        source: Source name; defaults to the one with the table.
        columns: Only these columns.
    """
    _, src = resolve_source(ctx.deps, source, table)
    meta = await metaset(src)
    slug = f"{src.name}{SOURCE_TABLE_SEPARATOR}{table.strip()}"
    if slug not in meta.catalog:
        raise ModelRetry(f"No table {table!r}. Tables: {', '.join(listed_tables(src))}.")
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
        source: Source name; defaults to the one with the table.
    """
    _, src = resolve_source(ctx.deps, source, table)
    return await execute_distinct_values(
        table, column, like, sources=_sources(src), metaset=await metaset(src), timeout=QUERY_TIMEOUT,
    )


async def run_sql(ctx: RunContext[FlowDashDeps], sql: str, source: str | None = None) -> str:
    """Run a read-only query for your own checks; returns row count, column types and first rows.

    Args:
        sql: One SELECT or WITH query.
        source: Source name; defaults to the only one.
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
