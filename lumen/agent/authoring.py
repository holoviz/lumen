"""Tools that validate a query, chart or map before placing it on the dashboard as a node."""
from __future__ import annotations

import dataclasses
import math

from typing import Any

import sqlglot

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturn

from ..ai.agents.sql import format_exploration_result
from ..ai.utils import clean_sql, format_error
from ..flowdash import (
    DECKGL_ID, QUERY_TIMEOUT, SOURCE_ID, SQL_ID, VIEW_ID, check_query,
    effective_limit, fetch, pipeline_columns, primed_result, result_sql,
    row_ceiling, selection_warnings,
)
from ..flowdash.deckgl import (
    fit_view_state, layer_type, map_summary, position_columns,
)
from ..pipeline import Pipeline
from . import _graph
from .data import lint, resolve_source


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
        source: Source node.
        name: Descriptive snake_case node name.
        sql: One SELECT or WITH query.
    """
    ops = ctx.deps.ops
    source_node, src = resolve_source(ctx.deps, source)
    limit = effective_limit(ops.editor, None)
    sql, statement, frame = await _test_run(src, sql, limit)
    with primed_result(src, statement, frame):
        _place(ops, SQL_ID, name, {"sql": sql, "table": name}, f"{source_node}.source", f"{name}.source")
    try:
        frame = _graph.output(ops, name, "data")
    except OperationError as exc:
        ops.remove_node(name)
        raise ModelRetry(str(exc)) from exc
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
    port = _upstream_port(ctx.deps, data)
    try:
        pipeline = _graph.output(ops, data, port)
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    if not isinstance(pipeline, Pipeline):
        raise ModelRetry(f"{data!r} has no Lumen pipeline output to show; pass the name of a SQL, table or filter node.")
    view = _graph.view_class(component)
    try:
        config = {**config, "spec": view.check_spec(config.get("spec") or {}, pipeline_columns(pipeline))}
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    _place(ops, component, name, config, f"{data}.{port}", f"{name}.pipeline")
    return config


def add_chart(ctx: RunContext[FlowDashDeps], data: str, name: str, spec: dict[str, Any]) -> str:
    """Place a Vega-Lite chart of a SQL, table or filter node's rows.

    Args:
        data: SQL, table or filter node to chart.
        name: Descriptive snake_case node name.
        spec: Vega-Lite spec using the node's columns, without data or URLs. Leave out
            width and height so the chart fills its tile and follows resizes; set
            them only when the user asks for a fixed size.
    """
    spec = place_view(ctx, VIEW_ID, data, name, {"spec": spec})["spec"]
    placed = f"Placed chart {name!r} ({', '.join(_marks(spec)) or 'composite'}) of {data!r}."
    notes = selection_warnings(spec)
    return placed + "".join(f"\nNote: {note}" for note in notes)


def add_map(ctx: RunContext[FlowDashDeps], data: str, name: str, spec: dict[str, Any]) -> str:
    """Place a deck.gl map of a SQL, table or filter node's rows with longitude and latitude columns.

    Args:
        data: SQL, table or filter node to map, not a source table: place a SQL node first.
        name: Descriptive snake_case node name.
        spec: deck.gl JSON spec without data: each layer draws the node's rows, has an
            "@@type" and reads columns in "@@=" expressions with arithmetic, comparisons
            and ?: only, no function calls, URLs or "@@function". Pick ScatterplotLayer
            for points (sized or coloured by a value), HexagonLayer or GridLayer to bin
            many points ("extruded": true for 3D), ColumnLayer for 3D bars at points,
            HeatmapLayer for density, ArcLayer for origin-destination pairs. Radii are
            metres unless "radiusUnits": "pixels"; to size points by a value use pixels
            with getRadius scaled to about 2 to 20. Leave out mapStyle, initialViewState
            and colours to follow the page's theme, fit the rows and take the dashboard's
            colours. A fixed colour is an RGBA list. E.g. {"layers": [{"@@type":
            "ScatterplotLayer", "getPosition": "@@=[lon, lat]", "radiusUnits": "pixels",
            "getRadius": "@@=2 + capacity / 200", "pickable": true}], "tooltip": {"text":
            "{name}: {capacity}"}} or {"layers": [{"@@type": "HexagonLayer",
            "getPosition": "@@=[lon, lat]", "radius": 20000, "extruded": true,
            "elevationScale": 50, "pickable": true}]}.
    """
    spec = place_view(ctx, DECKGL_ID, data, name, {"spec": spec})["spec"]
    ops = ctx.deps.ops
    try:
        # The map fetched the same rows to draw them, so this reads them without a query.
        rows = fetch(_graph.output(ops, data, _graph.pipeline_port(ops, data)), row_ceiling(ops.editor))
    except Exception as exc:
        ops.remove_node(name)
        raise ModelRetry(format_error(exc)) from exc
    placed = f"Placed map {name!r} of {data!r}: {map_summary(spec, rows)}."
    view = spec.get("initialViewState")
    notes = []
    if view is None and (view := fit_view_state(spec, rows)) is not None:
        placed += f" Fitted to the rows at longitude {view['longitude']}, latitude {view['latitude']}, zoom {view['zoom']}."
    elif view is None and len(rows) and (lon_lat := position_columns(spec)) is not None:
        notes.append(
            f"no row has a longitude in -180..180 and a latitude in -90..90 in [{', '.join(lon_lat)}], "
            "so the map shows the whole world; getPosition takes [longitude, latitude]."
        )
    if view is not None and (note := _radius_note(spec, view)):
        notes.append(note)
    return placed + "".join(f"\nNote: {note}" for note in notes)


# Metres per pixel at the equator at zoom 0, where deck.gl draws the world 512 pixels wide.
_EQUATOR_METRES_PER_PIXEL = 40_075_016.686 / 512


def _radius_note(spec: dict[str, Any], view: dict[str, Any]) -> str | None:
    """
    A note when a scatterplot sizes its points in metres, which at the
    view's zoom may all fall under ``radiusMinPixels`` and look alike.
    """
    zoom, latitude = view.get("zoom"), view.get("latitude", 0)
    if not isinstance(zoom, (int, float)) or not isinstance(latitude, (int, float)):
        return None
    for layer in spec.get("layers", []):
        radius = layer.get("getRadius")
        if (layer_type(layer) != "ScatterplotLayer" or layer.get("radiusUnits", "meters") != "meters"
                or not isinstance(radius, str)):
            continue
        metres = _EQUATOR_METRES_PER_PIXEL * math.cos(math.radians(latitude)) / 2 ** zoom
        smallest = metres * max(layer.get("radiusMinPixels", 0), 1)
        return (
            f"getRadius is in metres; at zoom {zoom} a pixel is about {metres:,.0f} m, so radii under "
            f"{smallest:,.0f} m all draw at the minimum size. To size points by a value, set "
            '"radiusUnits": "pixels" and scale getRadius to about 2 to 20.'
        )
    return None


def _upstream_port(deps: FlowDashDeps, data: str) -> str:
    ops = deps.ops
    if data not in _graph.nodes(ops):
        shown = [node for node in _graph.nodes(ops) if _graph.pipeline_port(ops, node)]
        hint = f" The SQL, table and filter nodes are: {', '.join(shown)}." if shown else ""
        if any(data in _source_tables(deps, source) for source in _graph.nodes(ops, SOURCE_ID)):
            raise ModelRetry(
                f"No node {data!r}: {data!r} is a source table. Place a SQL node on it with add_sql "
                f"first and pass that node's name.{hint}"
            )
        raise ModelRetry(f"No node {data!r}.{hint or ' Place a SQL node with add_sql first.'}")
    port = _graph.pipeline_port(ops, data)
    if port is None:
        raise ModelRetry(f"{data!r} has no Lumen pipeline output to show; pass the name of a SQL, table or filter node.")
    return port


def _source_tables(deps: FlowDashDeps, source: str) -> list[str]:
    try:
        return resolve_source(deps, source)[1].get_tables()
    except ModelRetry:
        return []


def authoring_toolset() -> FunctionToolset[FlowDashDeps]:
    """The authoring tools: add_sql, add_chart and add_map."""
    return FunctionToolset[FlowDashDeps]([add_sql, add_chart, add_map], max_retries=3)


class CheckedConfig(AbstractCapability[FlowDashDeps]):
    """
    Checks what ``set_config`` gives a Lumen SQL or view node as add_sql and
    add_chart check it, before the node changes: SQL is cleaned, checked and
    run, and a view's spec is checked by its ``check_spec`` against its
    data's columns. A problem comes back for the model to fix, and a changed
    query reports its result.
    """

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        config = args.get("config") if tool_def.name == "set_config" else None
        if not isinstance(config, dict):
            return await handler(args)
        ops, name = ctx.deps.ops, args.get("name")
        component = next((node["component"] for node in ops.nodes() if node["name"] == name), None)
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
