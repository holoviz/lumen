"""Tools that validate a query, chart, table or map before placing it on the dashboard as a node."""
from __future__ import annotations

import dataclasses
import json
import math

from typing import Any, Literal

import sqlglot

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturn

from ..ai.agents.sql import format_exploration_result
from ..ai.utils import clean_sql, format_error
from ..flowdash import (
    DECKGL_ID, METRIC_ID, QUERY_TIMEOUT, SOURCE_ID, SQL_ID, TABLE_VIEW_ID,
    VIEW_ID, CheckDiagnostic, check_query, check_sql_config, effective_limit,
    fetch, host_metrics, pipeline_columns, primed_result, result_sql,
    row_ceiling, selection_warnings,
)
from ..flowdash.deckgl import (
    fit_view_state, layer_type, map_summary, position_columns,
)
from ..flowdash.metrics import MetricError, prepare
from ..flowdash.nodes import default_table, metric_query
from ..flowdash.results import expand, merge
from ..pipeline import Pipeline
from ..util import as_pandas
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
    returning the cleaned query, the statement run (on a result source, the
    expanded statement, run on the root) and its result; any problem is
    raised as ``ModelRetry`` with the engine's error.
    """
    try:
        sql = clean_sql(str(sql).strip(), src.dialect, prettify=True)
        check_query(sql, src)
        root, statement = expand(sql, src)
        statement = result_sql(root, statement, limit)
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        raise ModelRetry(format_error(exc)) from exc
    try:
        frame = await root.execute_with_timeout(statement, QUERY_TIMEOUT)
    except Exception as exc:
        raise ModelRetry(format_error(exc)) from exc
    return sql, statement, frame


def _result_input(ops, source: str | list[str]) -> tuple[list[str], Any] | None:
    """
    The upstream SQL nodes and their results merged into one source when
    ``source`` names an earlier SQL node, or a list of them to join. Some
    models write the list as a JSON string, which counts as the list.
    """
    if isinstance(source, str) and source.strip().startswith("["):
        try:
            source = [str(name) for name in json.loads(source)]
        except (ValueError, TypeError):
            pass
    names = list(dict.fromkeys(source)) if isinstance(source, list) else [source]
    if not names:
        raise ModelRetry("Pass a source, an earlier SQL node, or a list of earlier SQL nodes to join.")
    try:
        ports = [_graph.input_port(ops, name) for name in names]
        if not isinstance(source, list) and ports[0] is None:
            return None
        if missing := [name for name, port in zip(names, ports, strict=True) if port is None]:
            raise ModelRetry(f"To join, pass earlier SQL nodes; {', '.join(map(repr, missing))} is not one.")
        return names, merge([_graph.output(ops, *port.rsplit(".", 1)) for port in ports])
    except (OperationError, ValueError) as exc:
        raise ModelRetry(str(exc)) from exc


def _merged_input(ops, name: str) -> Any:
    """The source the SQL node ``name`` queries: its inputs merged; ``OperationError`` if one has no value."""
    return merge([_graph.output(ops, *port.rsplit(".", 1)) for port in _graph.feeding_all(ops, f"{name}.source")])


async def _count(src: Any, table: str) -> int | None:
    """The rows of result table ``table`` of the result source ``src``, or None if counting fails."""
    count = sqlglot.select("COUNT(*)").from_(sqlglot.exp.Table(this=sqlglot.exp.to_identifier(table)))
    try:
        root, statement = expand(count.sql(dialect=src.dialect), src)
        frame = await root.execute_with_timeout(statement, QUERY_TIMEOUT)
        return int(as_pandas(frame).iloc[0, 0])
    except Exception:
        return None


async def _capped_note(ops, src: Any, upstream: str) -> str:
    """What a query reading ``upstream``'s result should know when its tile shows fewer rows than the result has."""
    shown = _graph.output(ops, upstream, "data")
    rows = _graph.config_of(ops, upstream).get("limit")
    if shown is None or len(shown) < effective_limit(ops.editor, rows if isinstance(rows, int) else None):
        return ""
    total = await _count(src, _graph.output(ops, upstream, "pipeline").table)
    if total is None or total <= len(shown):
        return ""
    return f" It reads all {total} rows of {upstream!r}; its tile shows the first {len(shown)}."


async def _add_follower(ops, upstreams: list[str], src: Any, name: str, sql: str) -> str:
    """Place a SQL node reading the results of ``upstreams``, joined if several, as ``add_sql`` places one on a source."""
    limit = effective_limit(ops.editor, None)
    sql, statement, frame = await _test_run(src, sql, limit)
    with primed_result(src, statement, frame):
        _place(ops, SQL_ID, name, {"sql": sql, "table": name}, f"{upstreams[0]}.result", f"{name}.source")
        try:
            for upstream in upstreams[1:]:
                ops.connect(f"{upstream}.result", f"{name}.source")
        except OperationError as exc:
            ops.remove_node(name)
            raise ModelRetry(str(exc)) from exc
    try:
        frame = _graph.output(ops, name, "data")
        notes = "".join([await _capped_note(ops, src, upstream) for upstream in upstreams])
    except OperationError as exc:
        ops.remove_node(name)
        raise ModelRetry(str(exc)) from exc
    capped = f" (capped at {limit} rows)" if len(frame) >= limit else ""
    read = " and ".join(repr(upstream) for upstream in upstreams)
    return (
        f"Placed SQL node {name!r} on the result{'s' * (len(upstreams) > 1)} of {read}{capped}.{notes} "
        f"{format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"
    )


async def _follower_problems(ops, name: str) -> list[str]:
    """
    What the SQL nodes reading ``name``'s result directly now refuse,
    checked against its new result. A node reading one of those in turn is
    not checked: it fails, and reports it, once that one does.
    """
    followers = [f for f in _graph.reading(ops, f"{name}.result") if _graph.component_of(ops, f) == SQL_ID]
    problems = []
    for follower in followers:
        try:
            result = _merged_input(ops, follower)
        except (OperationError, ValueError):
            continue
        problem = await check_sql_config({"sql": _graph.config_of(ops, follower).get("sql") or ""}, {"source": result})
        if isinstance(problem, CheckDiagnostic):
            problems.append(f"Follower {follower!r} now fails: {problem}")
    return problems


async def add_sql(ctx: RunContext[FlowDashDeps], source: str | list[str], name: str, sql: str) -> str:
    """Place a SQL node with a read-only query and return a preview of its result; errors come back to fix.

    To build on an earlier answer, pass its SQL node as source: the query then reads that
    result by its table name, as list_tables shows it, and runs again when that node changes.
    To join earlier answers, pass their SQL nodes as a list.

    Args:
        source: Source name from list_tables.
        name: Descriptive snake_case node name.
        sql: One SELECT or WITH query.
    """
    if (upstream := _result_input(ctx.deps.ops, source)) is not None:
        return await _add_follower(ctx.deps.ops, *upstream, name, sql)
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
    port = _upstream_port(ctx.deps, data)
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
    try:
        if existing:
            ops.set_config(name, config)
        else:
            ops.add_node(METRIC_ID, config, name=name)
        ops.set_inputs(name, {"parameters": parameters or None})
        frame = _graph.output(ops, name, "data")
    except OperationError as exc:
        if before is None:
            if name in _graph.nodes(ops):
                ops.remove_node(name)
        else:
            ops.set_config(name, before[0])
            ops.set_inputs(name, {"parameters": before[1]})
        raise ModelRetry(str(exc)) from exc
    ceiling = effective_limit(ops.editor, None)
    capped = f" (capped at {ceiling} rows)" if len(frame) >= ceiling else ""
    verb = "Changed" if existing else "Placed"
    return f"{verb} metric node {name!r}{capped}: {format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"


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
        spec: Optional table options naming the node's columns: page_size, layout,
            show_index, header_filters, frozen_columns, hidden_columns, sorters
            ([{"field": ..., "dir": "desc"}], dir defaults to "asc"), titles, widths and formatters
            ({column: "money" | "progress" | "star" | "tickCross" | ...}).
    """
    config = place_view(ctx, TABLE_VIEW_ID, data, name, {"spec": spec or {}})
    options = ", ".join(sorted(config["spec"])) or "default options"
    return f"Placed table {name!r} of {data!r} ({options})."


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


def _source_tables(deps: FlowDashDeps, source: str) -> list[str]:
    try:
        return resolve_source(deps, source)[1].get_tables()
    except ModelRetry:
        return []


def _keep_followers_table(ops, name: str, before: dict[str, Any], config: dict[str, Any]) -> None:
    """Refuse a change to the table name of a SQL node whose result other SQL nodes query by that name."""
    def table(cfg):
        return cfg.get("table") or default_table(cfg.get("sql") or "")

    if table({**before, **config}) == table(before) or not (followers := _graph.reading(ops, f"{name}.result")):
        return
    raise ModelRetry(
        f"{', '.join(repr(f) for f in followers)} query {name!r}'s result as table {table(before)!r}; "
        f"keep `table` as {table(before)!r}."
    )


def authoring_toolset() -> FunctionToolset[FlowDashDeps]:
    """The authoring tools: add_sql, add_metric, add_chart, add_table, add_map and add_text."""
    return FunctionToolset[FlowDashDeps]([add_sql, add_metric, add_chart, add_table, add_map, add_text], max_retries=3)


class CheckedConfig(AbstractCapability[FlowDashDeps]):
    """
    Checks what ``set_config`` gives a Lumen SQL or view node as add_sql and
    add_chart check it, before the node changes: SQL is cleaned, checked and
    run, a view's spec is checked by its ``check_spec`` against its data's
    columns, and the text of a heading, Markdown or divider as add_text
    checks it. A problem comes back for the model to fix, and a changed
    query reports its result and which SQL nodes reading its result
    directly its change breaks.
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
        if component != SQL_ID:
            return await handler(args)
        before = _graph.config_of(ops, name)
        _keep_followers_table(ops, name, before, config)
        if "sql" not in config or not _graph.feeding_all(ops, f"{name}.source"):
            return await handler(args)
        try:
            src = _merged_input(ops, name)
        except (OperationError, ValueError) as exc:
            raise ModelRetry(str(exc)) from exc
        rows = config.get("limit", before.get("limit"))
        limit = effective_limit(ops.editor, int(rows) if isinstance(rows, int) else None)
        sql, statement, frame = await _test_run(src, config["sql"], limit)
        config = {**config, "sql": sql}
        with primed_result(src, statement, frame):
            result = await handler({**args, "config": config})
        try:
            # TODO: with panel-flowdash's ExplorationOps (fd #238), an edit to an earlier answer
            # makes a copy (OwnNodesOnly steps aside for it), so ``name`` would read the frozen
            # original here, not the copy; see test_checked_config_previews_the_copy_explore_makes.
            frame = _graph.output(ops, name, "data")
        except OperationError as exc:
            ops.set_config(name, {key: before[key] for key in config if key in before})
            raise ModelRetry(str(exc)) from exc
        capped = f" (capped at {limit} rows)" if len(frame) >= limit else ""
        preview = f"\n\nResult{capped}: {format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"
        preview += "".join(f"\n\n{problem}" for problem in await _follower_problems(ops, name))
        if isinstance(result, str):
            return result + preview
        if isinstance(result, ToolReturn) and isinstance(result.return_value, str):
            return dataclasses.replace(result, return_value=result.return_value + preview)
        return result
