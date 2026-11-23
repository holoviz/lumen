"""
Tools that place a KPI card or a local filter control and wire them to a
node's rows, for an agent that lays out the dashboard.
"""
from __future__ import annotations

import math
import re

from typing import Annotated, Any, Literal

import pandas as pd
import sqlglot
import yaml

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic import Field
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from ..flowdash import (
    FILTER_ID, NODE_TIMEOUT, SQL_ID, fetch, filterable_columns,
    pipeline_columns,
)
from ..flowdash.choices import pipeline_column_types
from ..flowdash.nodes import column_values, execute, pipeline_sql
from ..flowdash.transforms import AGGREGATE_ID, COUNT_COLUMN
from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource
from . import _graph
from .authoring import _upstream_port

KPI_ID = "Widgets/KPI"
DATA_ID = "Lumen/data"
CONTROLS = {
    "select": "Widgets/Select", "multiselect": "Widgets/MultiChoice",
    "range": "Widgets/RangeSlider", "date_range": "Widgets/DateRange",
}
# The most options a select or multiselect offers; a column with more takes a range or a SQL node.
MAX_OPTIONS = 200
NUMERIC = ("integer", "number")
TEMPORAL = ("date", "datetime")
CURRENCIES = ("$", "€", "£", "¥")
# Column names of a share or rate, shown as a percentage when their value lies in 0..1.
RATIO_NAME = re.compile(r"(share|ratio|rate|pct|percent|fraction|proportion)", re.IGNORECASE)
# Column names of an entity a row belongs to, whose attributes repeat on each of its rows.
KEY_NAME = re.compile(r"(^|_)(id|name|key|code)$", re.IGNORECASE)

Aggregate = Literal["sum", "mean", "min", "max", "count"]
Decimals = Annotated[int, Field(ge=0, le=10)]
# How a KPI card relates to the filter bar: FlowDash's global filters filter what a view shows,
# not the value a card is given, so a card keeps its number whatever the bar picks.
KPI_FILTERS = "KPI cards do not follow the global filter bar's filters; tell the user if the dashboard has one."


class _Placed:
    """The nodes and edges a tool placed so far, undone in reverse order when a later step fails."""

    def __init__(self, ops):
        self.ops = ops
        self.undo: list[Any] = []

    def node(self, component: str, config: dict[str, Any], name: str) -> str:
        self.ops.add_node(component, config, name=name)
        self.undo.append(lambda: self.ops.remove_node(name))
        return name

    def connect(self, source: str, target: str, transform: list[dict[str, Any]] | None = None) -> None:
        self.ops.connect(source, target, transform)
        self.undo.append(lambda: self.ops.disconnect(source, target))

    def rewire(self, source: str, target: str, new: str) -> None:
        self.ops.disconnect(source, target)
        self.undo.append(lambda: self.ops.connect(source, target))
        self.connect(new, target)

    def rollback(self) -> None:
        for step in reversed(self.undo):
            try:
                step()
            except (OperationError, KeyError, ValueError):
                pass
        self.undo.clear()


def _free(ops, *names: str) -> None:
    taken = [name for name in names if name in _graph.nodes(ops)]
    if taken:
        raise ModelRetry(f"{', '.join(map(repr, taken))} is taken; pick another name.")


def _pipeline(ops, data: str, port: str) -> Pipeline:
    try:
        pipeline = _graph.output(ops, data, port)
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    if not isinstance(pipeline, Pipeline):
        raise ModelRetry(f"{data!r} has no rows yet; fix it or pass another node.")
    return pipeline


def _column_types(pipeline: Pipeline, frame: Any) -> dict[str, str | None]:
    """The type of each column, from the schema or else the fetched rows."""
    types = dict(pipeline_column_types(pipeline) or {})
    if isinstance(frame, pd.DataFrame):
        for column in frame.columns:
            if types.get(column) is None:
                dtype = frame[column].dtype
                types[column] = (
                    "boolean" if pd.api.types.is_bool_dtype(dtype) else "integer" if pd.api.types.is_integer_dtype(dtype)
                    else "number" if pd.api.types.is_numeric_dtype(dtype) else "string"
                )
    return types


def _quote(name: str, dialect: str) -> str:
    return sqlglot.exp.to_identifier(name, quoted=True).sql(dialect=dialect)


def _repeated_per(pipeline: Pipeline, column: str, types: dict[str, str | None]) -> tuple[str, int, int] | None:
    """
    A column naming the entity each row belongs to, e.g. a project, that
    ``column`` is constant within while the entity has several rows, so a
    sum over the rows counts its value once per row; with that entity's
    rows and distinct values. None when no such column is found or the
    rows cannot be queried.
    """
    sql, source = pipeline_sql(pipeline, limits=False), pipeline.source
    if sql is None or not isinstance(source, BaseSQLSource):
        return None
    dialect = source.dialect if source.dialect != "any" else None
    value = _quote(column, dialect)
    keys = [c for c in types if c != column and types[c] == "string" and KEY_NAME.search(c)]
    for key in keys[:5]:
        k = _quote(key, dialect)
        probe = (
            f"SELECT COUNT(*) AS n, COUNT(DISTINCT {k}) AS keys, "
            f"(SELECT COUNT(*) FROM (SELECT {k} FROM ({sql}) AS r GROUP BY {k} "
            f"HAVING COUNT(DISTINCT {value}) > 1) AS v) AS varying FROM ({sql}) AS t"
        )
        try:
            frame = execute(source, probe, NODE_TIMEOUT)
        except Exception:
            return None
        rows, distinct, varying = (int(v) for v in frame.iloc[0])
        if distinct >= 2 and varying == 0 and rows >= 1.5 * distinct:
            return key, rows, distinct
    return None


def _number_format(value: Any, column: str, unit: str, decimals: int | None) -> tuple[str, str, str]:
    """The KPI's format spec, prefix and suffix for ``value`` of ``column`` in ``unit``."""
    unit = unit.strip()
    number = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    ratio = number and 0 <= abs(value) <= 1 and (unit == "%" or (not unit and RATIO_NAME.search(column)))
    if ratio:
        return f".{1 if decimals is None else decimals}%", "", ""
    if decimals is None:
        whole = not number or float(value).is_integer() or abs(value) >= 100
        decimals = 0 if whole else 1 if abs(value) >= 10 else 2
    spec = f",.{decimals}f"
    if unit in CURRENCIES:
        return spec, unit, ""
    if unit == "%":
        return spec, "", "%"
    return spec, "", f" {unit}" if unit else ""


def _label(column: str, aggregate: str | None) -> str:
    if column == COUNT_COLUMN and aggregate == "count":
        return "Rows"
    words = column.replace("_", " ").strip()
    words = words[:1].upper() + words[1:]
    prefix = {"sum": "Total", "mean": "Average", "min": "Lowest", "max": "Highest", "count": "Count of"}.get(aggregate)
    return f"{prefix} {words.lower()}" if prefix else words


def _shown(value: Any, spec: str, prefix: str, suffix: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "empty"
    try:
        return f"{prefix}{format(value, spec)}{suffix}"
    except (TypeError, ValueError):
        return f"{prefix}{value}{suffix}"


def _scalar(frame: Any, column: str) -> Any:
    if not isinstance(frame, pd.DataFrame) or column not in frame.columns or frame.empty:
        return None
    value = frame[column].iloc[-1]
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def add_kpi(
    ctx: RunContext[FlowDashDeps], data: str, name: str, column: str | None = None,
    aggregate: Aggregate | None = None, label: str = "", unit: str = "", decimals: Decimals | None = None,
    format: str = "",
) -> str:
    """Show a headline number as a KPI card in the dashboard's top bar.

    The number is a cell of a node of one row, or the column aggregated over all of the
    node's rows by its source. A one-row node with several numeric columns and no column
    given gets one KPI per column. A SQL node is not a view: use this to show its number.
    A card does not follow the global filter bar's filters, only filter nodes upstream of it.

    Args:
        data: Name of the SQL, metric, table or filter node holding the number (a node name, not a source table).
        name: Descriptive snake_case node name; with one KPI per column, each is named <name>_<column>.
        column: The numeric column to show; leave out for every numeric column of a one-row node, or to count rows.
        aggregate: How to reduce the column over the node's rows; leave out when the node has one row.
        label: Short heading above the number, e.g. "Installed capacity"; defaults to the column in words.
        unit: Unit shown with the number, e.g. "MW", "$" or "%"; a share in 0..1 with "%" shows as a percentage.
        decimals: Digits after the decimal point, 0 to 10; by default 0 for whole or large numbers, at most 2 otherwise.
        format: A Python format spec instead of unit and decimals' choice, e.g. ",.1f".
    """
    ops = ctx.deps.ops
    port = _upstream_port(ctx.deps, data)
    pipeline = _pipeline(ops, data, port)
    try:
        rows = fetch(pipeline, 2)
    except Exception as exc:
        raise ModelRetry(f"{data!r} failed: {exc}") from exc
    types = _column_types(pipeline, rows)
    columns = pipeline_columns(pipeline)
    if column is not None and column not in columns:
        raise ModelRetry(f"{data!r} has no column {column!r}; its columns are {', '.join(columns)}.")
    if aggregate is None:
        if len(rows) != 1:
            raise ModelRetry(
                f"{data!r} has {'no rows' if rows.empty else 'more than one row'}; pass aggregate (sum, mean, min, "
                "max or count) to reduce the column over its rows, or a node of one row."
            )
        picked = [column] if column is not None else [c for c in columns if types.get(c) in NUMERIC]
        if not picked:
            raise ModelRetry(f"{data!r} has no numeric column; a KPI shows a number.")
    elif column is None:
        if aggregate != "count":
            raise ModelRetry(f"Name the column to {aggregate}; leave it out only to count rows.")
        picked = [COUNT_COLUMN]
    else:
        picked = [column]
    for col in picked:
        if aggregate != "count" and types.get(col) not in NUMERIC:
            raise ModelRetry(
                f"{col!r} holds {types.get(col) or 'non-numeric'} values; a KPI shows a number. Pick a numeric "
                "column, or aggregate='count' to count rows."
            )
    if aggregate in ("sum", "mean") and (repeat := _repeated_per(pipeline, column, types)):
        key, n, keys = repeat
        raise ModelRetry(
            f"{aggregate} of {column!r} over {data!r} would double count: {column!r} repeats on every row of each "
            f"{key!r} ({n:,} rows, {keys:,} {key!r} values). Place a SQL node that takes it once per {key!r}, e.g. "
            f"SELECT {aggregate.upper().replace('MEAN', 'AVG')}(v) FROM (SELECT DISTINCT {key}, {column} AS v FROM ...), "
            "and show that node's number."
        )
    if label and len(picked) > 1:
        raise ModelRetry("label names one KPI; leave it out for one KPI per column, or pass column.")
    names = [name] if len(picked) == 1 else [f"{name}_{col}" for col in picked]
    component = _graph.component_of(ops, data)
    helpers = [f"{name}_agg", f"{name}_value"] if aggregate else (
        [] if component == SQL_ID or "data" in _outputs(ops, data) else [f"{name}_rows"]
    )
    _free(ops, *names, *helpers)

    placed = _Placed(ops)
    try:
        if aggregate:
            placed.node(AGGREGATE_ID, {"columns": [] if column is None else [column], "method": aggregate}, helpers[0])
            placed.connect(f"{data}.{port}", f"{helpers[0]}.pipeline")
            placed.node(DATA_ID, {}, helpers[1])
            placed.connect(f"{helpers[0]}.result", f"{helpers[1]}.pipeline")
            frame_port = f"{helpers[1]}.data"
        elif helpers:
            placed.node(DATA_ID, {}, helpers[0])
            placed.connect(f"{data}.{port}", f"{helpers[0]}.pipeline")
            frame_port = f"{helpers[0]}.data"
        else:
            frame_port = f"{data}.data"
        frame = _graph.output(ops, *frame_port.rsplit(".", 1))
        lines = []
        for kpi, col in zip(names, picked, strict=True):
            value = _scalar(frame, col)
            spec, prefix, suffix = _number_format(value, col, unit, decimals)
            spec = format or spec
            text = label or _label(col, aggregate)
            placed.node(KPI_ID, {"label": text, "format": spec, "prefix": prefix, "suffix": suffix}, kpi)
            single = component == SQL_ID and not aggregate and len(columns) == 1
            if single:
                placed.connect(f"{data}.value", f"{kpi}.value")
            else:
                placed.connect(frame_port, f"{kpi}.value", [{"op": "column", "column": col}, {"op": "aggregate", "how": "last"}])
            ops.place_tile(kpi, area="top")
            shown = _shown(value, spec, prefix, suffix)
            empty = (" Its value is empty: the node has no number there (NULL or no rows), so the card shows a dash;"
                     " check the node's query.") if shown == "empty" else ""
            lines.append(f"- {kpi!r}: {text} = {shown}.{empty}")
    except (OperationError, ValueError) as exc:
        placed.rollback()
        raise ModelRetry(str(exc)) from exc
    except BaseException:
        placed.rollback()
        raise
    how = f"the {aggregate} of {column!r}" if aggregate and column else "its row count" if aggregate else "its one row"
    through = f" through {', '.join(map(repr, helpers))}" if helpers else ""
    return (f"Placed {len(names)} KPI card(s) in the top bar, showing {how} of {data!r}{through}:\n"
            + "\n".join(lines) + f"\n{KPI_FILTERS}")


def _outputs(ops, name: str) -> list[str]:
    spec = next((c for c in ops.list_components() if c["component"] == _graph.component_of(ops, name)), {})
    return [port["name"] for port in spec.get("outputs", [])]


def _node(port: str) -> str:
    return port.split(".", maxsplit=1)[0]


def _reader_path(edges: list[dict[str, Any]], view: str, source: str, depth: int = 4) -> list[tuple[str, str]] | None:
    """
    The edges, as (from, to) ports and the one from ``source`` first,
    through which ``view`` reads the output ``source``, directly or a few
    nodes up.
    """
    frontier: list[tuple[str, list[tuple[str, str]]]] = [(view, [])]
    for _ in range(depth):
        nxt = []
        for node, below in frontier:
            for edge in edges:
                if _node(edge["to"]) != node:
                    continue
                path = [(edge["from"], edge["to"]), *below]
                if edge["from"] == source:
                    return path
                nxt.append((_node(edge["from"]), path))
        frontier = nxt
    return None


def _reach(edges: list[dict[str, Any]], node: str, stop: set[str]) -> set[str]:
    """The nodes downstream of ``node``, not following the outputs of those in ``stop``."""
    seen: set[str] = set()
    todo = [node]
    while todo:
        current = todo.pop()
        for edge in edges:
            if _node(edge["from"]) == current and (target := _node(edge["to"])) not in seen:
                seen.add(target)
                if target not in stop:
                    todo.append(target)
    return seen


def _filter_edges(edges: list[dict[str, Any]], paths: dict[str, list[tuple[str, str]]]) -> dict[str, tuple[str, str]]:
    """
    The edge of each view's path to rewire to a filter's rows: the one
    nearest the source whose target feeds only the given views, so a node
    shared with other views, e.g. an earlier filter, keeps its input and
    only the given views read the new filter.
    """
    targets = set(paths)
    allowed = targets | {_node(to) for path in paths.values() for _, to in path}
    chosen = {}
    for view, path in paths.items():
        chosen[view] = next(
            edge for edge in path
            if _node(edge[1]) in targets or _reach(edges, _node(edge[1]), targets) <= allowed
        )
    return chosen


def _global_filter(ops, column: str, views: list[str]) -> dict[str, Any] | None:
    """The global filter on ``column`` that filters any of ``views``, if the dashboard has one."""
    filters = ops.filters() if callable(getattr(ops, "filters", None)) else getattr(ops.editor, "filters", None)
    for entry in filters or []:
        if entry.get("column") == column and not set(views) <= set(entry.get("exclude") or ()):
            return entry
    return None


def _place_before(ops, name: str, views: list[str]) -> None:
    """Move the tile of ``name`` just before the first of ``views``' tiles on the grid."""
    indices = []
    for view in views:
        entry = ops.place_tile(view)
        if entry.get("visible", True) and entry.get("area", "grid") == "grid" and "index" in entry:
            indices.append(entry["index"])
    if not indices:
        return
    first, current = min(indices), ops.place_tile(name).get("index")
    ops.place_tile(name, index=first - 1 if current is not None and current < first else first)


def add_filter_control(
    ctx: RunContext[FlowDashDeps], data: str, column: str, name: str,
    control: Literal["select", "multiselect", "range", "date_range"] = "select", views: list[str] | None = None,
    label: str = "", value: str | int | float | None = None,
) -> str:
    """Add a local filter: a control shown with the views it filters, linked to them through a filter node.

    Where add_filter, the global filter bar, is offered, use this only when the user asks for a
    filter on particular charts or tables. The listed views then read the filtered rows; others
    are unchanged.

    Args:
        data: Name of the SQL, metric, table or filter node the views read (a node name, not a source table).
        column: The column to filter on.
        name: Descriptive snake_case name of the control; its filter node is <name>_filter.
        control: "select" picks one value, "multiselect" any number, "range" a span of a numeric
            column, "date_range" a span of dates. Each starts unfiltered: nothing picked, or the full span.
        views: Charts, tables and KPIs reading data to filter; all views reading it when left out.
        label: The control's label; defaults to the column in words.
        value: Only when the user asks for one, the value a select starts on, filtering from the start.
    """
    ops = ctx.deps.ops
    port = _upstream_port(ctx.deps, data)
    pipeline = _pipeline(ops, data, port)
    if column not in filterable_columns(pipeline):
        raise ModelRetry(f"{data!r} has no column {column!r}; its columns are {', '.join(filterable_columns(pipeline))}.")
    source = f"{data}.{port}"
    targets = list(dict.fromkeys(views)) if views else [
        n for n in _graph.reading(ops, source) if _graph.component_of(ops, n) != FILTER_ID
    ]
    if not targets:
        raise ModelRetry(f"No view reads {data!r}; place the views first, or name them in views.")
    graph_edges = yaml.safe_load(ops.get_dashboard()).get("edges") or []
    paths = {}
    for view in targets:
        if view not in _graph.nodes(ops):
            raise ModelRetry(f"No node {view!r}.")
        if (path := _reader_path(graph_edges, view, source)) is None:
            readers_text = ", ".join(map(repr, _graph.reading(ops, source))) or "none"
            raise ModelRetry(f"{view!r} does not read {data!r}'s rows, so a filter on them cannot change it. "
                             f"The nodes reading {data!r} are: {readers_text}.")
        for _, to in path[:-1]:
            node = _node(to)
            if _graph.component_of(ops, node) == FILTER_ID and _graph.config_of(ops, node).get("field") == column:
                raise ModelRetry(f"{view!r} is already filtered on {column!r} by {node!r}; change the control "
                                 f"that feeds it, or list only views it does not filter.")
        paths[view] = path
    if (entry := _global_filter(ops, column, targets)) is not None:
        raise ModelRetry(
            f"The global filter bar already filters on {column!r} ({entry.get('label') or column}), and would apply "
            "to these views as well as a local filter. Change that filter with add_filter, or remove it with "
            "remove_filter first if the user wants these views filtered on their own."
        )
    edges = _filter_edges(graph_edges, paths)
    feeds = list(dict.fromkeys(old for old, _ in edges.values()))
    filters = {feed: f"{name}_filter" if i == 0 else f"{name}_filter_{i + 1}" for i, feed in enumerate(feeds)}
    _free(ops, name, *filters.values())
    types = _column_types(pipeline, None)
    kind = types.get(column)
    text = label or column.replace("_", " ").strip().capitalize()
    config: dict[str, Any] = {"label": text}
    component = CONTROLS[control]
    kind_text = {"select": "select", "multiselect": "multi-select", "range": "range", "date_range": "date range"}[control]
    if control in ("select", "multiselect"):
        options = column_values(pipeline, column, limit=MAX_OPTIONS + 1)
        if not options:
            raise ModelRetry(f"{column!r} of {data!r} has no values to choose from.")
        if len(options) > MAX_OPTIONS:
            raise ModelRetry(f"{column!r} has more than {MAX_OPTIONS} values, too many to pick from; "
                             "use control='range' for numbers or dates, or filter a smaller column.")
        config["default_options"] = options
        if value is not None and control != "select":
            raise ModelRetry("value sets where a select starts; a multiselect starts with nothing picked.")
        if value is not None:
            if value not in options:
                shown = ", ".join(map(repr, options[:20]))
                raise ModelRetry(f"{value!r} is not a value of {column!r}; its values include {shown}.")
            # A select always holds an option and starts on its first, so the default goes first.
            config["default_options"] = [value, *(o for o in options if o != value)]
            config["searchable"] = len(options) > 10
            start = f"starting on {value!r}, so it filters from the start"
        else:
            if control == "select":
                # A select always holds one of its options, so an unfiltered start takes a
                # multiselect limited to one pick, which keeps every row while nothing is picked.
                component = CONTROLS["multiselect"]
                config.update(max_items=1, placeholder="All")
                kind_text = "multi-select of one pick"
            start = "starting with nothing picked, so every row shows"
        summary = f"{len(options)} options from {column!r}, {start}"
    elif value is not None:
        raise ModelRetry("value sets where a select starts; a range starts on the full span.")
    elif control == "range":
        if kind not in NUMERIC:
            raise ModelRetry(f"{column!r} holds {kind or 'non-numeric'} values; a range control takes a numeric column.")
        span = column_values(pipeline, column, span=True)
        if not span:
            raise ModelRetry(f"{column!r} of {data!r} has no values to span.")
        low, high = span
        whole = kind == "integer" or all(float(v).is_integer() for v in span)
        step = 1 if whole else float(f"{(high - low) / 100:.1g}") or 0.01
        config.update(default_start=low, default_end=high, step=step, number_type="integer" if whole else "float")
        summary = f"{low} to {high} from {column!r}, starting on the full span, which leaves out only rows without a value"
    else:
        if kind not in TEMPORAL:
            raise ModelRetry(f"{column!r} holds {kind or 'non-date'} values; a date_range control takes a date column.")
        span = column_values(pipeline, column, span=True)
        if not span:
            raise ModelRetry(f"{column!r} of {data!r} has no dates to span.")
        low, high = (str(v)[:10] for v in span)
        config.update(default_start=low, default_end=high, default_period="all")
        summary = f"{low} to {high} from {column!r}, starting on the full span, which leaves out only rows without a value"

    placed = _Placed(ops)
    try:
        placed.node(component, config, name)
        for feed, filter_name in filters.items():
            placed.node(FILTER_ID, {"field": column}, filter_name)
            placed.connect(feed, f"{filter_name}.pipeline")
            placed.connect(f"{name}.selected", f"{filter_name}.value")
        for old, target in edges.values():
            placed.rewire(old, target, f"{filters[old]}.filtered")
        for filter_name in filters.values():
            _graph.output(ops, filter_name, "filtered")
        _place_before(ops, name, list(edges))
    except (OperationError, ValueError) as exc:
        placed.rollback()
        raise ModelRetry(str(exc)) from exc
    except BaseException:
        placed.rollback()
        raise
    wired = ", ".join(
        repr(view) if _node(target) == view else f"{view!r} (through {_node(target)!r})"
        for view, (_, target) in edges.items()
    )
    rewired = {_node(target) for _, target in edges.values()}
    others = [n for n in _graph.reading(ops, source) if n not in rewired and _graph.component_of(ops, n) != FILTER_ID]
    rest = f" Unfiltered: {', '.join(map(repr, others))}." if others else ""
    names = ", ".join(map(repr, filters.values()))
    nodes = f"filter node {names}" if len(filters) == 1 else f"filter nodes {names} (one per input the views read)"
    reading = names if len(filters) == 1 else "them"
    return (
        f"Placed {kind_text} {name!r} ({summary}) and {nodes} on {data!r}.{column}, "
        f"fed by {name!r}. Reading {reading} now: {wired}.{rest}"
    )


def controls_toolset() -> FunctionToolset[FlowDashDeps]:
    """The tools of an agent that lays out the dashboard: add_kpi and add_filter_control."""
    return FunctionToolset[FlowDashDeps]([add_kpi, add_filter_control], max_retries=3)
