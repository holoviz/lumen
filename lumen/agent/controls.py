"""
Tools that place a KPI card or a local filter control and wire them to a
node's rows, for an agent that lays out the dashboard.
"""
from __future__ import annotations

import math
import re

from builtins import format as builtins_format
from typing import Annotated, Any, Literal

import pandas as pd
import sqlglot
import yaml

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic import Field
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from ..flowdash import (
    FILTER_ID, METRIC_ID, SQL_ID, fetch, filterable_columns, host_metrics,
    pipeline_columns,
)
from ..flowdash.choices import pipeline_column_types
from ..flowdash.nodes import column_values
from ..flowdash.save_metric import adhoc_caption
from ..flowdash.transforms import AGGREGATE_ID, COUNT_COLUMN
from ..pipeline import Pipeline
from . import _graph
from .authoring import _upstream_port

KPI_ID = "Widgets/KPI"
DATA_ID = "Lumen/data"
CONTROLS = {
    "select": "Widgets/Select", "multiselect": "Widgets/MultiChoice",
    "range": "Widgets/RangeSlider", "date_range": "Widgets/DateRange",
}
# The most metric names an ad hoc KPI's result lists.
MAX_METRICS_LISTED = 10
# The most options a select or multiselect offers; a column with more takes a range or a SQL node.
MAX_OPTIONS = 200
NUMERIC = ("integer", "number")
TEMPORAL = ("date", "datetime")
CURRENCIES = ("$", "€", "£", "¥")
# Column names of a share or rate, shown as a percentage when their value lies in 0..1.
RATIO_NAME = re.compile(r"(share|ratio|rate|pct|percent|fraction|proportion)", re.IGNORECASE)
# The longest format spec and the most decimals a KPI takes.
FORMAT_CHARS = 20
MAX_DECIMALS = 10

Aggregate = Literal["sum", "mean", "min", "max", "count"]
Decimals = Annotated[int, Field(ge=0, le=MAX_DECIMALS)]
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
    format: str = "", distinct_by: str | list[str] | None = None,
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
            For a median or a count of distinct values, aggregate in a SQL node and show its number.
        label: Short heading above the number, e.g. "Installed capacity"; defaults to the column in words.
        unit: Unit shown with the number, e.g. "MW", "$" or "%"; a share in 0..1 with "%" shows as a percentage.
        decimals: Digits after the decimal point, 0 to 10; by default 0 for whole or large numbers, at most 2 otherwise.
        format: A Python format spec instead of unit and decimals' choice, e.g. ",.1f".
        distinct_by: Only when you know the column repeats per entity, e.g. a project's capacity on
            each of its turbines' rows: the column, or columns, naming the entity. The aggregate then
            takes the largest value of each entity once; nothing checks the choice.
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
    if format:
        precision = re.search(r"\.(\d+)", format)
        if len(format) > FORMAT_CHARS or (precision and int(precision.group(1)) > MAX_DECIMALS):
            raise ModelRetry(f"format {format[:40]!r} is not a short format spec with at most {MAX_DECIMALS} "
                             'decimals, e.g. ",.1f" or ".1%".')
        try:
            builtins_format(1234.5, format)
        except (TypeError, ValueError) as exc:
            raise ModelRetry(f"format {format!r} is not a Python format spec for a number: {exc}") from exc
    keys = [distinct_by] if isinstance(distinct_by, str) else list(distinct_by or [])
    if keys:
        if aggregate is None or column is None:
            raise ModelRetry("distinct_by goes with an aggregate of a column; pass both.")
        if missing := [k for k in keys if k not in columns or k == column]:
            raise ModelRetry(f"distinct_by names other columns of {data!r}, not {', '.join(map(repr, missing))}; "
                             f"its columns are {', '.join(columns)}.")
    if label and len(picked) > 1:
        raise ModelRetry("label names one KPI; leave it out for one KPI per column, or pass column.")
    names = [name] if len(picked) == 1 else [f"{name}_{col}" for col in picked]
    component = _graph.component_of(ops, data)
    if aggregate:
        helpers = [*([f"{name}_per_key"] if keys else []), f"{name}_agg", f"{name}_value"]
    else:
        helpers = [] if component == SQL_ID or "data" in _outputs(ops, data) else [f"{name}_rows"]
    _free(ops, *names, *helpers)

    placed = _Placed(ops)
    try:
        value_port = None
        if aggregate:
            upstream = f"{data}.{port}"
            if keys:
                # One row per key, with the key's largest value; the aggregate then reads those.
                placed.node(AGGREGATE_ID, {"by": keys, "columns": [column], "method": "max"}, helpers[0])
                placed.connect(upstream, f"{helpers[0]}.pipeline")
                upstream = f"{helpers[0]}.result"
            agg, rows_node = helpers[-2:]
            placed.node(AGGREGATE_ID, {"columns": [] if column is None else [column], "method": aggregate}, agg)
            placed.connect(upstream, f"{agg}.pipeline")
            placed.node(DATA_ID, {}, rows_node)
            placed.connect(f"{agg}.result", f"{rows_node}.pipeline")
            frame_port = f"{rows_node}.data"
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
            config = {"label": text, "format": spec, "prefix": prefix, "suffix": suffix,
                      "caption": _provenance(ops, data, component, col, aggregate, keys)}
            placed.node(KPI_ID, config, kpi)
            if component == SQL_ID and not aggregate and len(columns) == 1:
                value_port = f"{data}.value"
            if value_port is not None:
                placed.connect(value_port, f"{kpi}.value")
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
    per_key = f", taking the largest {column!r} of each {' and '.join(keys)} once" if keys else ""
    through = f" through {', '.join(map(repr, helpers))}" if helpers else ""
    adhoc = ""
    if component != METRIC_ID:
        adhoc = " The number is computed ad hoc, not an agreed metric; each card's caption says so."
        if metrics := list(host_metrics(ops.editor.context)):
            more = f" and {len(metrics) - MAX_METRICS_LISTED} more" if len(metrics) > MAX_METRICS_LISTED else ""
            adhoc += (f" This app defines metrics: {', '.join(metrics[:MAX_METRICS_LISTED])}{more}. None was used for "
                      "this number; use add_metric if one fits, or ask an admin to define one.")
    return (f"Placed {len(names)} KPI card(s) in the top bar, showing {how} of {data!r}{per_key}{through}:\n"
            + "\n".join([*lines, *([adhoc.strip()] if adhoc else []), KPI_FILTERS]))


def _provenance(ops, data: str, component: str, column: str, aggregate: str | None, keys: list[str]) -> str:
    """
    How a card's number is computed, for its caption: the metric, or "Ad
    hoc: <how>; no metric was used." as the "Ad hoc" provenance label words it.
    """
    if component == METRIC_ID:
        return f"Metric {_graph.config_of(ops, data).get('metric')}"
    if aggregate == "count" and column == COUNT_COLUMN:
        what = "Count of rows"
    elif aggregate == "count" and keys:
        what = f"Count of {' and '.join(keys)} values with a {column}"
    elif aggregate:
        word = {"sum": "Sum", "mean": "Average", "min": "Lowest", "max": "Highest", "count": "Count"}[aggregate]
        what = f"{word} of the largest {column} per {' and '.join(keys)}" if keys else f"{word} of {column} over all rows"
    elif component == SQL_ID and (found := _graph.provenance(ops, data)) is not None and found.kind == "ad hoc":
        return adhoc_caption(found.detail)
    else:
        what = f"Value of {column} from {data}"
    of = f" of {data}" if aggregate else ""
    return adhoc_caption(f"{what}{of}; no metric was used.")


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
            readers = _graph.reading(ops, source)
            readers_text = (f"The nodes reading {data!r} are: {', '.join(map(repr, readers))}." if readers
                            else f"No node reads {data!r}'s rows directly.")
            raise ModelRetry(f"{view!r} does not read {data!r}'s rows, so a filter on them cannot change it. "
                             + readers_text)
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
    """
    The tools of an agent that lays out the dashboard: add_kpi and
    add_filter_control, each run alone, as they change the graph.
    """
    return FunctionToolset[FlowDashDeps]([add_kpi, add_filter_control], max_retries=3, sequential=True)
