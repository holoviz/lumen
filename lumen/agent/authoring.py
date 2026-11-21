"""Tools that validate a query, chart or table before placing it on the dashboard as a node."""
from __future__ import annotations

import dataclasses

from typing import Any, Literal

import sqlglot

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturn

from ..ai.agents.sql import format_exploration_result
from ..ai.utils import clean_sql, format_error
from ..flowdash import (
    METRIC_ID, QUERY_TIMEOUT, SQL_ID, TABLE_VIEW_ID, VIEW_ID, check_query,
    effective_limit, host_metrics, pipeline_columns, primed_result, result_sql,
    selection_warnings,
)
from ..flowdash.metrics import MetricError, prepare
from ..flowdash.nodes import metric_query
from ..pipeline import Pipeline
from . import _graph
from ._graph import (
    DIVIDER_ID, HEADING_ID, MARKDOWN_ID, TEXT_FIELDS,
)
from .data import lint, resolve_source, source_output


def _place(ops, component: str, name: str, config: dict[str, Any], source: str, target: str) -> None:
    try:
        ops.add_node(component, config, name=name)
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    try:
        ops.connect(source, target)
    except OperationError as exc:
        ops.remove_node(name)
        raise ModelRetry(str(exc)) from exc


def _marks(spec: dict[str, Any]) -> list[str]:
    marks = []
    for layer in spec.get("layer", [spec]):
        mark = layer.get("mark") if isinstance(layer, dict) else None
        mark = mark.get("type") if isinstance(mark, dict) else mark
        if isinstance(mark, str):
            marks.append(mark)
    return marks


async def _test_run(src, sql: str, limit: int) -> tuple[str, str, Any]:
    """
    Clean, check and run ``sql`` as a SQL node with ``limit`` rows would,
    returning the cleaned query, the statement run and its result; any
    problem is raised as ``ModelRetry`` with the engine's error.
    """
    try:
        sql = clean_sql(str(sql).strip(), src.dialect, prettify=True)
        check_query(sql, src)
        statement = result_sql(src, sql, limit)
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        raise ModelRetry(format_error(exc)) from exc
    try:
        frame = await src.execute_with_timeout(statement, QUERY_TIMEOUT)
    except Exception as exc:
        raise ModelRetry(format_error(exc)) from exc
    return sql, statement, frame


async def add_sql(ctx: RunContext[FlowDashDeps], source: str, name: str, sql: str) -> str:
    """Place a SQL node with a read-only query and return a preview of its result; errors come back to fix.

    Args:
        source: Source name from list_tables.
        name: Descriptive snake_case node name.
        sql: One SELECT or WITH query.
    """
    ops = ctx.deps.ops
    _, src = resolve_source(ctx.deps, source)
    limit = effective_limit(ops.editor, None)
    sql, statement, frame = await _test_run(src, sql, limit)
    port, src, placed = source_output(ctx.deps, source)
    try:
        with primed_result(src, statement, frame):
            _place(ops, SQL_ID, name, {"sql": sql, "table": name}, port, f"{name}.source")
        try:
            frame = _graph.output(ops, name, "data")
        except OperationError as exc:
            ops.remove_node(name)
            raise ModelRetry(str(exc)) from exc
    except ModelRetry:
        if placed is not None:
            ops.remove_node(placed)
        raise
    capped = f" (capped at {limit} rows)" if len(frame) >= limit else ""
    return f"Placed SQL node {name!r}{capped}: {format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"


def place_view(ctx: RunContext[FlowDashDeps], component: str, data: str, name: str,
               config: dict[str, Any]) -> dict[str, Any]:
    """
    Place the Lumen view node ``component`` on the pipeline of node ``data``
    with ``config``, its spec checked by the view's ``check_spec`` against
    the pipeline's columns first, and return the config placed. A problem
    comes back as ``ModelRetry`` for the model to fix.
    """
    ops = ctx.deps.ops
    port = _upstream_port(ops, data)
    try:
        pipeline = _graph.output(ops, data, port)
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    if not isinstance(pipeline, Pipeline):
        raise ModelRetry(f"{data!r} has no Lumen pipeline output to show; pass the name of a SQL, metric, table or filter node.")
    view = _graph.view_class(component)
    try:
        config = {**config, "spec": view.check_spec(config.get("spec") or {}, pipeline_columns(pipeline))}
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    _place(ops, component, name, config, f"{data}.{port}", f"{name}.pipeline")
    return config


async def add_metric(
    ctx: RunContext[FlowDashDeps], metric: str, name: str, measures: list[str] | None = None,
    dimensions: list[str] | None = None, grain: str | None = None, parameters: dict[str, Any] | None = None,
    order_by: list[str] | None = None, limit: int | None = None,
) -> str:
    """Place a metric node, one of the app's agreed definitions, and return a preview of its result.

    Args:
        metric: Metric name from list_tables.
        name: Descriptive snake_case node name; a metric node of this name is changed instead.
        measures: Measures to compute; a SQL metric has none.
        dimensions: Dimensions to group by.
        grain: Time grain of the time dimensions picked, e.g. year.
        parameters: Parameter values, and column dimensions to filter by with a value or a list.
        order_by: Columns to sort by; a leading - sorts descending.
        limit: Rows at most, e.g. 10 for a top 10.
    """
    ops = ctx.deps.ops
    metrics = host_metrics(ops.editor.context)
    if metric not in metrics:
        raise ModelRetry(f"No metric {metric!r}. The metrics are: {', '.join(metrics) or 'none'}.")
    config = {"metric": metric, "measures": measures or [], "dimensions": dimensions or [], "grain": grain or "",
              "order_by": order_by or [], "limit": limit}
    try:
        query = metric_query(config, parameters)
        prepare(metrics[metric], query)
    except MetricError as exc:
        raise ModelRetry(str(exc)) from exc
    existing = name in _graph.nodes(ops)
    if existing and _graph.component_of(ops, name) != METRIC_ID:
        raise ModelRetry(f"{name!r} is taken by a node that is not a metric; pick another name.")
    before = (_graph.config_of(ops, name), _graph.inputs_of(ops, name).get("parameters")) if existing else None
    placed = set(_graph.nodes(ops))
    target = name
    try:
        if existing:
            ops.set_config(name, config)
            # An edit of an earlier answer in Explore goes to a copy of it.
            target = _graph.edited(ops, name, placed)
        else:
            ops.add_node(METRIC_ID, config, name=name)
        ops.set_inputs(target, {"parameters": parameters or None})
        frame = _graph.output(ops, target, "data")
    except PermissionError as exc:
        raise PermissionError(f"{exc} Place a metric node of your own under another name instead.") from exc
    except OperationError as exc:
        if before is None:
            if name in _graph.nodes(ops):
                ops.remove_node(name)
        else:
            ops.set_config(target, before[0])
            ops.set_inputs(target, {"parameters": before[1]})
        raise ModelRetry(str(exc)) from exc
    ceiling = effective_limit(ops.editor, None)
    capped = f" (capped at {ceiling} rows)" if len(frame) >= ceiling else ""
    verb = "Changed" if existing else "Placed"
    return f"{verb} metric node {target!r}{capped}: {format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"


def add_chart(ctx: RunContext[FlowDashDeps], data: str, name: str, spec: dict[str, Any]) -> str:
    """Place a Vega-Lite chart of a SQL, metric, table or filter node's rows.

    Args:
        data: SQL, metric, table or filter node to chart.
        name: Descriptive snake_case node name.
        spec: Vega-Lite spec using the node's columns, without data or URLs. Leave out
            width and height so the chart fills its tile and follows resizes; set
            them only when the user asks for a fixed size.
    """
    spec = place_view(ctx, VIEW_ID, data, name, {"spec": spec})["spec"]
    placed = f"Placed chart {name!r} ({', '.join(_marks(spec)) or 'composite'}) of {data!r}."
    notes = selection_warnings(spec)
    return placed + "".join(f"\nNote: {note}" for note in notes)


def add_table(ctx: RunContext[FlowDashDeps], data: str, name: str, spec: dict[str, Any] | None = None) -> str:
    """Place a table showing a SQL, table or filter node's rows, when the user asks to see rows rather than a chart.

    Args:
        data: SQL, table or filter node to show.
        name: Descriptive snake_case node name.
        spec: Optional table options naming the node's columns: page_size, frozen_columns,
            hidden_columns, sorters ([{"field": ..., "dir": "desc"}]), titles, widths and
            formatters ({column: "money" | "progress" | "star" | "tickCross" | ...}).
    """
    config = place_view(ctx, TABLE_VIEW_ID, data, name, {"spec": spec or {}})
    options = ", ".join(sorted(config["spec"])) or "default options"
    return f"Placed table {name!r} of {data!r} ({options})."


def _upstream_port(ops, data: str) -> str:
    if data not in _graph.nodes(ops):
        raise ModelRetry(f"No node {data!r}.")
    port = _graph.pipeline_port(ops, data)
    if port is None:
        raise ModelRetry(f"{data!r} has no Lumen pipeline output to show; pass the name of a SQL, metric, table or filter node.")
    return port


TEXT_IDS = {"heading": HEADING_ID, "markdown": MARKDOWN_ID, "divider": DIVIDER_ID}
# The most characters each text field may hold; prose longer than this belongs in several tiles.
TEXT_LIMITS = {
    (HEADING_ID, "text"): 200, (HEADING_ID, "subtitle"): 400,
    (MARKDOWN_ID, "text"): 4000, (DIVIDER_ID, "label"): 100,
}
_PLACEHOLDER = "{{"


def check_text(ops, component: str, config: dict[str, Any], name: str | None = None) -> None:
    """
    Raise ``ModelRetry`` when a text component's ``config`` has a text
    field that is not a string, is too long, or uses a placeholder while
    nothing feeds the node's value input.
    """
    for (cid, field), limit in TEXT_LIMITS.items():
        if cid != component or field not in config:
            continue
        value = config[field]
        if not isinstance(value, str):
            raise ModelRetry(f"{field} must be a string.")
        if len(value) > limit:
            raise ModelRetry(f"{field} has {len(value)} characters; keep it to {limit}.")
        if _PLACEHOLDER in value and (name is None or _graph.upstream(ops, name) is None):
            raise ModelRetry(
                f"{field} uses a {{{{ value }}}} placeholder, but nothing is connected to the node's value "
                "input; write the text itself, with numbers as a node shows them."
            )


def add_text(
    ctx: RunContext[FlowDashDeps], kind: Literal["heading", "markdown", "divider"], name: str, text: str = "",
    subtitle: str = "", size: Literal["Title", "Heading", "Subheading", "Section"] | None = None,
) -> str:
    """Place a heading, Markdown prose or a divider with the text you write for it.

    Args:
        kind: "heading" for a title, "markdown" for prose, "divider" for a rule between groups of tiles.
        name: Descriptive snake_case node name.
        text: The heading's text, the Markdown (at most 4000 characters) or the divider's optional label.
        subtitle: Smaller text below a heading.
        size: A heading's size; "Title" for the dashboard's title.
    """
    ops = ctx.deps.ops
    component = TEXT_IDS[kind]
    if kind != "divider" and not text.strip():
        raise ModelRetry(f"Write the {kind}'s text.")
    config: dict[str, Any] = {TEXT_FIELDS[component]: text} if text else {}
    if kind == "heading":
        if subtitle:
            config["subtitle"] = subtitle
        if size:
            config["size"] = size
    elif subtitle or size:
        raise ModelRetry("Only a heading takes a subtitle or a size.")
    check_text(ops, component, config)
    try:
        ops.add_node(component, config, name=name)
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    return f"Placed {kind} {name!r}."


def authoring_toolset() -> FunctionToolset[FlowDashDeps]:
    """The authoring tools: add_sql, add_metric, add_chart, add_table and add_text."""
    return FunctionToolset[FlowDashDeps]([add_sql, add_metric, add_chart, add_table, add_text], max_retries=3)


class CheckedConfig(AbstractCapability[FlowDashDeps]):
    """
    Checks what ``set_config`` gives a Lumen SQL or view node as add_sql and
    add_chart check it, before the node changes: SQL is cleaned, checked and
    run, and a view's spec is checked by its ``check_spec`` against its
    data's columns, and the text of a heading, Markdown or divider as
    add_text checks it. A problem comes back for the model to fix, and a
    changed query reports its result.
    """

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        config = args.get("config") if tool_def.name == "set_config" else None
        if not isinstance(config, dict):
            return await handler(args)
        ops, name = ctx.deps.ops, args.get("name")
        component = next((node["component"] for node in ops.nodes() if node["name"] == name), None)
        if component in TEXT_FIELDS:
            check_text(ops, component, config, name)
            return await handler(args)
        if (view := _graph.view_class(component)) is not None and "spec" in config:
            data = _graph.upstream(ops, name)
            port = _graph.pipeline_port(ops, data) if data else None
            try:
                pipeline = _graph.output(ops, data, port) if port else None
            except OperationError:
                pipeline = None
            if isinstance(pipeline, Pipeline):
                try:
                    config = {**config, "spec": view.check_spec(config["spec"], pipeline_columns(pipeline))}
                except ValueError as exc:
                    raise ModelRetry(str(exc)) from exc
            return await handler({**args, "config": config})
        if component != SQL_ID or "sql" not in config or (upstream := _graph.upstream(ops, name)) is None:
            return await handler(args)
        _, src = resolve_source(ctx.deps, upstream)
        before = _graph.config_of(ops, name)
        rows = config.get("limit", before.get("limit"))
        limit = effective_limit(ops.editor, int(rows) if isinstance(rows, int) else None)
        sql, statement, frame = await _test_run(src, config["sql"], limit)
        config = {**config, "sql": sql}
        with primed_result(src, statement, frame):
            result = await handler({**args, "config": config})
        try:
            frame = _graph.output(ops, name, "data")
        except OperationError as exc:
            ops.set_config(name, {key: before[key] for key in config if key in before})
            raise ModelRetry(str(exc)) from exc
        capped = f" (capped at {limit} rows)" if len(frame) >= limit else ""
        preview = f"\n\nResult{capped}: {format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"
        if isinstance(result, str):
            return result + preview
        if isinstance(result, ToolReturn) and isinstance(result.return_value, str):
            return dataclasses.replace(result, return_value=result.return_value + preview)
        return result
