"""Tools that validate a query, chart, table or map before placing it on the dashboard as a node."""
from __future__ import annotations

import dataclasses
import inspect
import json
import math

from collections.abc import Mapping
from typing import Any, Literal

import sqlglot

from panel_flowdash.agent import DashboardOps, FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturn

from ..ai.agents.sql import format_exploration_result
from ..ai.utils import clean_sql, format_error
from ..flowdash import (
    ANALYSIS_ID, DECKGL_ID, METRIC_ID, QUERY_TIMEOUT, SQL_ID, TABLE_VIEW_ID,
    VIEW_ID, CheckDiagnostic, check_query, check_sql_config, effective_limit,
    fetch, hidden_tables, pipeline_columns, primed_result, result_sql,
    row_ceiling, selection_warnings,
)
from ..flowdash.analysis import (
    _bounded_pipeline, build_analysis, check_analysis_applies, missing_columns,
)
from ..flowdash.access import (
    access_denied_source, can_access_source, sql_access_denied,
)
from ..flowdash.catalog import hidden_reads, turned_off
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
from .data import (
    _sources_by_name, lint, resolve_source, result_notes, source_output,
)
from .edits import (
    PATCH_DEPTH, changed_layers, depth, kind, merge_chart_layers, merge_patch,
    name_notes, patch_problem, unrelated, view_summary,
)
from .facts import answer_notes

try:
    from panel_flowdash.agent import ExplorationOps
except ImportError:
    ExplorationOps = None


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


def _taken(ops, name: str, component: str) -> bool:
    """
    Whether a node of ``component`` is already called ``name``, which an
    add tool then changes; ``ModelRetry`` when a node of another kind is.
    """
    if name not in _graph.nodes(ops):
        return False
    if (other := _graph.component_of(ops, name)) != component:
        raise ModelRetry(
            f"{name!r} is already a {kind(other)} node, so no {kind(component)} can take its name; "
            f"pick another name, or change {name!r} with the tool that placed it."
        )
    return True


def _rewire(ops, ports: list[str], target: str) -> None:
    """Connect the input ``target`` to the outputs ``ports`` instead of what feeds it now."""
    current = _graph.feeding_all(ops, target)
    if current == ports:
        return
    for port in current:
        ops.disconnect(port, target)
    for port in ports:
        ops.connect(port, target)


def _notes(notes: list[str]) -> str:
    return "".join(f"\nNote: {note}" for note in notes)


def _marks(spec: dict[str, Any]) -> list[str]:
    marks = []
    for layer in spec.get("layer", [spec]):
        mark = layer.get("mark") if isinstance(layer, dict) else None
        mark = mark.get("type") if isinstance(mark, dict) else mark
        if isinstance(mark, str):
            marks.append(mark)
    return marks


async def _test_run(src, sql: str, limit: int, context: Mapping[str, Any] | None = None) -> tuple[str, str, Any]:
    """
    Clean, check and run ``sql`` as a SQL node with ``limit`` rows would,
    returning the cleaned query, the statement run (on a result source, the
    expanded statement, run on the root) and its result; any problem is
    raised as ``ModelRetry`` with the engine's error. With ``context``, the
    editor's, a query reading a table its catalog turns off is refused.
    """
    try:
        sql = clean_sql(str(sql).strip(), src.dialect, prettify=True)
        check_query(sql, src)
        if context is not None:
            if not can_access_source(src, context):
                raise ModelRetry(access_denied_source(src.name))
            if denied := sql_access_denied(sql, src, context):
                raise ModelRetry(denied)
            if read := hidden_reads(sql, src, hidden_tables(src, context)):
                raise ModelRetry(turned_off(read))
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
    sql, statement, frame = await _test_run(src, sql, limit, ops.editor.context)
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
    ) + _notes(await result_notes(src, sql, frame))


def _view_problems(ops, name: str) -> list[str]:
    """What the charts, tables and maps reading node ``name``'s rows now refuse, checked against its new columns."""
    try:
        columns = pipeline_columns(_graph.output(ops, name, "pipeline"))
    except (OperationError, AttributeError):
        return []
    readers = list(dict.fromkeys(
        reader for port in ("pipeline", "data") for reader in _graph.reading(ops, f"{name}.{port}")
    ))
    problems = []
    for reader in readers:
        if (view := _graph.view_class(_graph.component_of(ops, reader))) is None:
            continue
        try:
            view.check_spec(_graph.config_of(ops, reader).get("spec") or {}, columns)
        except ValueError as exc:
            problems.append(f"View {reader!r} now fails: {exc} Change it with update_chart, or the query back.")
    return problems


async def _follower_problems(ops, name: str) -> list[str]:
    """
    What the SQL nodes reading ``name``'s result directly and the views
    showing its rows now refuse, checked against its new result. A node
    reading one of those in turn is not checked: it fails, and reports it,
    once that one does.
    """
    followers = [f for f in _graph.reading(ops, f"{name}.result") if _graph.component_of(ops, f) == SQL_ID]
    problems = _view_problems(ops, name)
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
        name: snake_case name saying what the node holds; a SQL node of this name is changed instead.
        sql: One SELECT or WITH query.
    """
    ops = ctx.deps.ops
    if _taken(ops, name, SQL_ID):
        return await _change_sql(ctx, source, name, sql)
    if (upstream := _result_input(ctx.deps.ops, source)) is not None:
        text = await _add_follower(ctx.deps.ops, *upstream, name, sql)
        return text + _sql_notes(ops, name)
    _, src = resolve_source(ctx.deps, source)
    limit = effective_limit(ops.editor, None)
    sql, statement, frame = await _test_run(src, sql, limit, ops.editor.context)
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
    return (f"Placed SQL node {name!r}{capped}: {format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"
            + _notes(await result_notes(src, sql, frame)) + _sql_notes(ops, name))


def _sql_notes(ops, name: str) -> str:
    """Notes on the names of SQL node ``name`` and of the columns its query computes."""
    try:
        frame = _graph.output(ops, name, "data")
        dialect = getattr(_graph.output(ops, name, "pipeline").source, "dialect", None)
    except (OperationError, AttributeError):
        return ""
    columns = [str(column) for column in getattr(frame, "columns", [])]
    return _notes(name_notes(name, columns, _graph.config_of(ops, name).get("sql") or "", dialect))


async def _change_sql(ctx: RunContext[FlowDashDeps], source: str | list[str], name: str, sql: str) -> str:
    """
    Change the query of SQL node ``name``, and what it reads when
    ``source`` names other inputs, as ``add_sql`` would place it; its
    tile, followers and table name stay.
    """
    ops = ctx.deps.ops
    upstream = _result_input(ops, source)
    placed_source = None
    if upstream is not None:
        names, src = upstream
        if name in names:
            raise ModelRetry(f"{name!r} cannot read its own result; pass the source or SQL nodes it reads.")
        ports = [f"{n}.result" for n in names]
    else:
        _, src = resolve_source(ctx.deps, source)
    before, wired = _graph.config_of(ops, name), _graph.feeding_all(ops, f"{name}.source")
    rows = before.get("limit")
    limit = effective_limit(ops.editor, rows if isinstance(rows, int) else None)
    sql, statement, frame = await _test_run(src, sql, limit, ops.editor.context)
    if upstream is None:
        port, src, placed_source = source_output(ctx.deps, source)
        ports = [port]
    existing = set(_graph.nodes(ops))
    target = name
    try:
        with primed_result(src, statement, frame):
            # Rewired first, since the new query is checked against what the node reads.
            _rewire(ops, ports, f"{name}.source")
            ops.set_config(name, {"sql": sql})
            target = _graph.edited(ops, name, existing)
        frame = _graph.output(ops, target, "data")
    except (OperationError, ValueError) as exc:
        if target in _graph.nodes(ops):
            ops.set_config(target, {"sql": before.get("sql", "")})
            _rewire(ops, wired, f"{target}.source")
        if placed_source is not None:
            ops.remove_node(placed_source)
        raise ModelRetry(str(exc)) from exc
    capped = f" (capped at {limit} rows)" if len(frame) >= limit else ""
    problems = "".join(f"\n\n{problem}" for problem in await _follower_problems(ops, target))
    return (f"Changed SQL node {target!r}{capped}: {format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"
            f"{problems}" + _notes(await result_notes(src, sql, frame)) + _sql_notes(ops, target))


def place_view(ctx: RunContext[FlowDashDeps], component: str, data: str, name: str,
               config: dict[str, Any], change: bool = False) -> tuple[str, dict[str, Any]]:
    """
    Place the Lumen view node ``component`` on the pipeline of node ``data``
    with ``config``, its spec checked by the view's ``check_spec`` against
    the pipeline's columns first, and return its name and the config placed.
    With ``change``, the view node ``name`` gets ``config`` and shows
    ``data`` instead, keeping its tile; in Explore, an earlier answer's
    view is changed in a copy, whose name is returned. A problem comes
    back as ``ModelRetry`` for the model to fix.
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
    if not change:
        _place(ops, component, name, config, f"{data}.{port}", f"{name}.pipeline")
        return name, config
    before, wired, existing = _graph.config_of(ops, name), _graph.feeding_all(ops, f"{name}.pipeline"), set(_graph.nodes(ops))
    target = name
    try:
        # Rewired first, since the new spec is checked against the data the view shows.
        _rewire(ops, [f"{data}.{port}"], f"{name}.pipeline")
        ops.set_config(name, config)
        target = _graph.edited(ops, name, existing)
    except OperationError as exc:
        if target in _graph.nodes(ops):
            ops.set_config(target, {key: before.get(key) for key in config})
            _rewire(ops, wired, f"{target}.pipeline")
        raise ModelRetry(str(exc)) from exc
    return target, config


async def add_metric(
    ctx: RunContext[FlowDashDeps], metric: str, name: str, measures: list[str] | None = None,
    dimensions: list[str] | None = None, grain: str | None = None, parameters: dict[str, Any] | None = None,
    order_by: list[str] | None = None, limit: int | None = None,
) -> str:
    """Place one of the app's metrics, as list_tables lists them, and return a preview of its result.

    It only places a metric the app defines and cannot create one: compute a new number with add_sql.

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
    metrics = _graph.metrics(ops)
    if not metrics:
        return "Refused: this app defines no metrics, so there is none to place. Compute the number with add_sql."
    if metric not in metrics:
        raise ModelRetry(f"No metric {metric!r}. The metrics are: {', '.join(metrics)}; compute anything else with add_sql.")
    config = {"metric": metric, "measures": measures or [], "dimensions": dimensions or [], "grain": grain or "",
              "order_by": order_by or [], "limit": limit}
    try:
        query = metric_query(config, parameters)
        prepare(metrics[metric], query)
    except MetricError as exc:
        if metrics[metric].form == "sql" and exc.field in ("measures", "dimensions", "grain"):
            raise ModelRetry(f"{exc} Place it with only its metric, name and parameters: its query already "
                             "returns its columns, e.g. one per group.") from exc
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
    return (f"{verb} metric node {target!r}{capped}: {format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"
            + _notes(answer_notes(frame)))


async def add_analysis(
    ctx: RunContext[FlowDashDeps],
    data: str,
    analysis: str,
    name: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Run one of the app's custom analyses (from list_tables) on a data node's pipeline.

    Returns a preview of its result rows when the analysis outputs a pipeline or DataFrame,
    or confirms the rendered view when the analysis outputs a view.

    Args:
        data: Name of the SQL, metric, table, filter, transform or analysis node to feed in.
        analysis: Analysis name from list_tables.
        name: Descriptive snake_case node name; an analysis node of this name is changed instead.
        params: Optional parameter values for the analysis, matching the params shown in list_tables.
    """
    ops = ctx.deps.ops
    available = _graph.analyses(ops)
    if not available:
        return "Refused: this app defines no analyses, so there is none to place."
    if analysis not in available:
        raise ModelRetry(
            f"No analysis {analysis!r}. The analyses are: {', '.join(sorted(available))}."
        )
    entry = available[analysis]
    port = _upstream_port(ctx.deps, data)
    try:
        pipeline = _graph.output(ops, data, port)
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    if not isinstance(pipeline, Pipeline):
        raise ModelRetry(
            f"{data!r} has no Lumen pipeline output to analyze; pass the name of a SQL, metric, table, "
            "filter or analysis node that outputs a pipeline."
        )
    columns = pipeline_columns(pipeline)
    if missing := missing_columns(entry, columns):
        raise ModelRetry(
            f"Analysis {analysis!r} requires column(s) {', '.join(missing)}, "
            f"not in {data!r}'s columns ({', '.join(columns) or 'none'})."
        )
    try:
        build_analysis(entry, params, name=analysis, columns=columns)
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    try:
        bounded = _bounded_pipeline(pipeline, row_ceiling(ops.editor), QUERY_TIMEOUT)
        applies = await check_analysis_applies(entry, bounded)
    except Exception as exc:
        raise ModelRetry(format_error(exc)) from exc
    if not applies:
        raise ModelRetry(f"Analysis {analysis!r} does not apply to {data!r}.")

    change = _taken(ops, name, ANALYSIS_ID)
    config: dict[str, Any] = {"analysis": analysis, "params": dict(params or {})}
    before = _graph.config_of(ops, name) if change else None
    wired = _graph.feeding_all(ops, f"{name}.pipeline") if change else []
    existing = set(_graph.nodes(ops))
    target = name
    try:
        if change:
            _rewire(ops, [f"{data}.{port}"], f"{name}.pipeline")
            ops.set_config(name, config)
            target = _graph.edited(ops, name, existing)
        else:
            _place(ops, ANALYSIS_ID, name, config, f"{data}.{port}", f"{name}.pipeline")
        view = _graph.output(ops, target, "view")
        frame = _graph.output(ops, target, "data")
    except PermissionError as exc:
        raise PermissionError(f"{exc} Place an analysis node of your own under another name instead.") from exc
    except (OperationError, ValueError) as exc:
        if not change:
            if name in _graph.nodes(ops):
                ops.remove_node(name)
        elif target in _graph.nodes(ops):
            ops.set_config(target, {key: before.get(key) for key in config})
            _rewire(ops, wired, f"{target}.pipeline")
        raise ModelRetry(str(exc)) from exc

    verb = "Changed" if change else "Placed"
    if view is not None:
        explore = ExplorationOps is not None and isinstance(ops, ExplorationOps)
        if not explore and hasattr(ops, "place_tile"):
            try:
                ops.place_tile(target, visible=True)
            except OperationError:
                pass
        return f"{verb} analysis node {target!r} ({analysis}) on {data!r}: view ({type(view).__name__})."
    if frame is not None:
        ceiling = effective_limit(ops.editor, None)
        capped = f" (capped at {ceiling} rows)" if len(frame) >= ceiling else ""
        return (
            f"{verb} analysis node {target!r} ({analysis}) on {data!r}{capped}: "
            f"{format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"
        )
    return f"{verb} analysis node {target!r} ({analysis}) on {data!r}."


def add_chart(ctx: RunContext[FlowDashDeps], data: str, name: str, spec: dict[str, Any]) -> str:
    """Place a Vega-Lite chart of a SQL, metric, table or filter node's rows.

    Args:
        data: SQL, metric, table or filter node to chart.
        name: snake_case name saying what the chart shows; a chart of this name is replaced
            by this one in its tile.
        spec: Vega-Lite spec using the node's columns, without data or URLs. Leave out
            width and height so the chart fills its tile and follows resizes; set
            them only when the user asks for a fixed size.
    """
    change = _taken(ctx.deps.ops, name, VIEW_ID)
    target, config = place_view(ctx, VIEW_ID, data, name, {"spec": spec}, change=change)
    spec = config["spec"]
    verb = "Changed" if change else "Placed"
    placed = f"{verb} chart {target!r} ({', '.join(_marks(spec)) or 'composite'}) of {data!r}."
    notes = selection_warnings(spec) + _chart_name_notes(ctx.deps.ops, name, data, spec)
    return placed + _notes(notes)


def _chart_name_notes(ops, name: str, data: str, spec: dict[str, Any]) -> list[str]:
    """A note when a chart's name shares no word with its data node, fields or title."""
    fields = [value["field"] for layer in [spec, *(spec.get("layer") or [])] if isinstance(layer, dict)
              for value in (layer.get("encoding") or {}).values() if isinstance(value, dict) and isinstance(value.get("field"), str)]
    title = spec.get("title") if isinstance(spec.get("title"), str) else ""
    if not unrelated(name, data, title, *fields):
        return []
    return [f"the name {name!r} shares no word with its data {data!r} or fields ({', '.join(dict.fromkeys(fields))}); "
            "name a chart after what it shows."]


TableFormatter = Literal["number", "money", "plaintext", "textarea", "progress", "star", "tickCross", "traffic", "color"]


def add_table(ctx: RunContext[FlowDashDeps], data: str, name: str, spec: dict[str, Any] | None = None,
              formatters: dict[str, TableFormatter | dict[str, Any]] | None = None) -> str:
    """Place a table showing a SQL, table or filter node's rows, when the user asks to see rows rather than a chart.

    Args:
        data: Name of the SQL, table or filter node to show (a node name, not a source table).
        name: snake_case name saying what the table shows; a table of this name is replaced
            by this one in its tile.
        spec: Optional table options naming the node's columns: page_size, layout,
            show_index, header_filters, frozen_columns, hidden_columns, sorters
            ([{"field": ..., "dir": "desc"}], dir defaults to "asc"), titles and widths.
        formatters: How to draw a column's cells, a name or {"type": name, ...params}:
            "number" with thousands separators and "decimals" (default 0 for whole numbers,
            else 2), "money" ("symbol", "decimals"), "progress" ("min", "max"), "star",
            "tickCross", "traffic", "color", "plaintext" or "textarea" (no params). Leave a year
            column unformatted. E.g. {"capacity_mw": {"type": "number", "decimals": 1}}.
    """
    if formatters:
        spec = {**(spec or {}), "formatters": {**((spec or {}).get("formatters") or {}), **formatters}}
    change = _taken(ctx.deps.ops, name, TABLE_VIEW_ID)
    target, config = place_view(ctx, TABLE_VIEW_ID, data, name, {"spec": spec or {}}, change=change)
    options = ", ".join(sorted(config["spec"])) or "default options"
    return f"{'Changed' if change else 'Placed'} table {target!r} of {data!r} ({options})."


def add_map(ctx: RunContext[FlowDashDeps], data: str, name: str, spec: dict[str, Any]) -> str:
    """Place a deck.gl map of a SQL, table or filter node's rows with longitude and latitude columns.

    Args:
        data: SQL, table or filter node to map, not a source table: place a SQL node first.
        name: snake_case name saying what the map shows; a map of this name is replaced by
            this one in its tile.
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
    ops = ctx.deps.ops
    change = _taken(ops, name, DECKGL_ID)
    before = _graph.config_of(ops, name) if change else None
    target, config = place_view(ctx, DECKGL_ID, data, name, {"spec": spec}, change=change)
    spec = config["spec"]
    try:
        # The map fetched the same rows to draw them, so this reads them without a query.
        rows = fetch(_graph.output(ops, data, _graph.pipeline_port(ops, data)), row_ceiling(ops.editor))
    except Exception as exc:
        if before is None:
            ops.remove_node(name)
        else:
            ops.set_config(target, {"spec": before.get("spec")})
        raise ModelRetry(format_error(exc)) from exc
    placed = f"{'Changed' if change else 'Placed'} map {target!r} of {data!r}: {map_summary(spec, rows)}."
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
        if any(data in _source_tables(deps, source) for source in _sources_by_name(deps)):
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
# FlowDash releases before place_tile took before and after cannot put a tile beside another,
# nor keep a title placed at the top in front of the tiles added after it.
PLACES_BESIDE = "before" in inspect.signature(DashboardOps.place_tile).parameters
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
    position: Literal["top", "bottom"] | None = None, before: str | None = None, after: str | None = None,
) -> str:
    """Place a heading, Markdown prose or a divider with the text you write for it.

    Returns the layout as it is now, row by row. A "Title" heading leads the dashboard by itself.

    Args:
        kind: "heading" for a title, "markdown" for prose, "divider" for a rule between groups of tiles.
        name: Descriptive snake_case node name; a text node of this kind and name gets this text instead.
        text: The heading's text, the Markdown (at most 4000 characters) or the divider's optional label.
        subtitle: Smaller text below a heading.
        size: A heading's size; "Title" for the dashboard's title.
        position: "top" or "bottom" of the dashboard; without it, before or after, the text goes
            where the layout puts it.
        before: Put the text in front of this node's tile, e.g. a section heading above its chart.
        after: Put the text after this node's tile.
    """
    ops = ctx.deps.ops
    component = TEXT_IDS[kind]
    if kind != "divider" and not text.strip():
        raise ModelRetry(f"Write the {kind}'s text.")
    # Explore shows each answer in its feed, which has no layout to place text in.
    explore = ExplorationOps is not None and isinstance(ops, ExplorationOps)
    if not explore and any(p is not None for p in (position, before, after)):
        _check_placement(ops, kind, position, before, after)
    config: dict[str, Any] = {TEXT_FIELDS[component]: text} if text else {}
    if kind == "heading":
        if subtitle:
            config["subtitle"] = subtitle
        if size:
            config["size"] = size
    elif subtitle or size:
        raise ModelRetry("Only a heading takes a subtitle or a size.")
    changed = _taken(ops, name, component)
    if changed:
        check_text(ops, component, config, name)
        existing = set(_graph.nodes(ops))
        try:
            ops.set_config(name, config)
        except OperationError as exc:
            raise ModelRetry(str(exc)) from exc
        placed = f"Changed {kind} {_graph.edited(ops, name, existing)!r}"
    else:
        check_text(ops, component, config)
        try:
            ops.add_node(component, config, name=name)
        except OperationError as exc:
            raise ModelRetry(str(exc)) from exc
        placed = f"Placed {kind} {name!r}"
    if explore:
        return f"{placed}." + (" Explore shows it in the feed, so position, before and after do not apply."
                               if any(p is not None for p in (position, before, after)) else "")
    # A changed text stays where it is unless it is moved, and only a position asked for moves a tile:
    # moving one turns off FlowDash's automatic arrangement of the tiles placed after it, which
    # already puts a Title first.
    try:
        if before is not None or after is not None:
            ops.place_tile(name, before=before, after=after)
            placed += f" {'before' if before else 'after'} {before or after!r}"
        elif position is not None:
            ops.place_tile(name, index=0 if position == "top" else len(_graph.nodes(ops)))
            placed += f" at the {position}"
    except OperationError as exc:
        placed += f", but not moved: {exc}"
    describe = getattr(ops, "describe_layout", None)
    return f"{placed}." + (f"\nLayout now:\n{describe()}" if describe is not None else "")


def _check_placement(ops, kind: str, position: str | None, before: str | None, after: str | None) -> None:
    """Retry a position, before or after that cannot apply, before the text is placed."""
    if sum(p is not None for p in (position, before, after)) > 1:
        raise ModelRetry("Pass one of position, before and after.")
    for other in (before, after):
        if other is not None and other not in _graph.nodes(ops):
            raise ModelRetry(f"No node {other!r} to place the {kind} beside.")
    if (before is not None or after is not None) and not PLACES_BESIDE:
        raise ModelRetry("This FlowDash version cannot place text beside a tile; use position.")
    hidden = ops.hidden_tiles() if hasattr(ops, "hidden_tiles") else []
    for other in (before, after):
        if other in hidden:
            raise ModelRetry(
                f"{other!r} has no tile on the dashboard (it is hidden, e.g. a SQL node), so "
                f"nothing can go beside it; place the {kind} beside a tile that is shown."
            )


def update_chart(ctx: RunContext[FlowDashDeps], name: str, changes: dict[str, Any] | None = None,
                 add_layers: list[dict[str, Any]] | None = None, layer: int | None = None) -> str:
    """Change part of a chart's Vega-Lite spec, or of a table's options or a map's spec, keeping the rest and its tile.

    changes merges into the spec: a key you give replaces that key, an object merges key by key and
    null removes a key, e.g. {"mark": "line"}, {"encoding": {"y": {"title": "GW"}}},
    {"encoding": {"color": null}}, {"title": "Sales by region"} or a table's {"formatters": {...}}.
    A channel given another field drops the old field's title and format; swapping x and y swaps
    them. On a layered chart a mark changes its main data layer, and a channel changes in the
    layers that show the same field there.

    Args:
        name: The chart, table or map node.
        changes: Part of a spec to merge into the node's spec.
        add_layers: Vega-Lite layers to draw over a chart, e.g. a rolling average line or a rule;
            they share the chart's x and y unless they set their own.
        layer: Apply changes to this layer of the chart only, numbered from 0.
    """
    ops = ctx.deps.ops
    if name not in _graph.nodes(ops):
        views = [n for n in _graph.nodes(ops) if _graph.view_class(_graph.component_of(ops, n)) is not None]
        raise ModelRetry(f"No node {name!r}." + (f" The charts, tables and maps are: {', '.join(views)}." if views else ""))
    component = _graph.component_of(ops, name)
    if (view := _graph.view_class(component)) is None:
        raise ModelRetry(f"{name!r} is a {kind(component)} node; update_chart changes charts, tables and maps.")
    if (add_layers or layer is not None) and component != VIEW_ID:
        raise ModelRetry("Only a chart takes add_layers or layer.")
    if not changes and not add_layers:
        raise ModelRetry("Pass the changes to make, or add_layers.")
    if problem := patch_problem({"changes": changes, "add_layers": add_layers}):
        return f"Refused: {problem} Nothing was changed."
    before = _graph.config_of(ops, name).get("spec") or {}
    try:
        if component == VIEW_ID:
            spec, added = merge_chart_layers(before, changes, add_layers, layer)
        else:
            spec, added = merge_patch(before, changes), []
    except ValueError as exc:
        raise ModelRetry(f"{exc} Nothing was changed.") from exc
    except RecursionError:
        return "Refused: the changes nest too deeply to merge. Nothing was changed."
    data = _graph.upstream(ops, name)
    port = _graph.pipeline_port(ops, data) if data else None
    try:
        pipeline = _graph.output(ops, data, port) if port else None
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    # Without data to read, the spec's fields cannot be checked, but what it may hold still is.
    columns = pipeline_columns(pipeline) if isinstance(pipeline, Pipeline) else sorted(view.spec_fields(spec))
    try:
        spec = view.check_spec(spec, columns)
    except ValueError as exc:
        hint = " To show another node's columns, call add_chart with this name and that node as data." if component == VIEW_ID else ""
        raise ModelRetry(f"{exc}{hint} The spec after your changes: {json.dumps(spec)}") from exc
    # Someone may have edited the node while the patched spec was checked; their edit wins.
    if (_graph.config_of(ops, name).get("spec") or {}) != before:
        return (f"Refused: {name!r} changed meanwhile, e.g. by the user, so nothing was changed. "
                "Read it again in the dashboard state or with list_tables and retry the change.")
    existing = set(_graph.nodes(ops))
    try:
        ops.set_config(name, {"spec": spec})
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    target = _graph.edited(ops, name, existing)
    notes = selection_warnings(spec) if component == VIEW_ID else []
    layers = changed_layers(before, spec, added) if component == VIEW_ID else ""
    return (f"Changed {kind(component)} {target!r} of {data!r}{f' ({layers})' if layers else ''}: "
            f"{view_summary(component, spec)}" + _notes(notes))


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
    """
    The authoring tools: add_sql, add_metric, add_analysis, add_chart, add_table,
    add_map, add_text and update_chart. Each runs alone, as the calls of one response
    change the same graph: e.g. update_chart waits for an add_sql adding the
    column a new layer shows, and two add_sql calls do not place a source
    node at once. The data tools, which only read, still run in parallel.
    """
    return FunctionToolset[FlowDashDeps](
        [add_sql, add_metric, add_analysis, add_chart, add_table, add_map, add_text, update_chart],
        max_retries=3, sequential=True,
    )


def _refused(result: Any) -> bool:
    """Whether a FlowDash tool's ``result`` says it changed nothing, e.g. a refusal of Explore."""
    text = result.return_value if isinstance(result, ToolReturn) else result
    return isinstance(text, str) and text.startswith(("Refused:", "Not available:"))


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
        if depth(config) > PATCH_DEPTH:
            return f"Refused: the config nests more than {PATCH_DEPTH} levels deep; a config needs only a few."
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
        if component == ANALYSIS_ID:
            merged = {**_graph.config_of(ops, name), **config}
            analysis_name = merged.get("analysis")
            if analysis_name:
                available = _graph.analyses(ops)
                if analysis_name not in available:
                    raise ModelRetry(
                        f"No analysis {analysis_name!r}. The analyses are: {', '.join(sorted(available)) or 'none'}."
                    )
                upstream = _graph.upstream(ops, name)
                port = _graph.pipeline_port(ops, upstream) if upstream else None
                try:
                    pipeline = _graph.output(ops, upstream, port) if port else None
                except OperationError:
                    pipeline = None
                columns = pipeline_columns(pipeline) if isinstance(pipeline, Pipeline) else None
                try:
                    build_analysis(available[analysis_name], merged.get("params"), name=analysis_name, columns=columns)
                except ValueError as exc:
                    raise ModelRetry(str(exc)) from exc
            return await handler(args)
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
        placed = set(_graph.nodes(ops))
        with primed_result(src, statement, frame):
            result = await handler({**args, "config": config})
        if _refused(result):
            return result
        # An edit of an earlier answer in Explore goes to a copy of it, which the preview shows.
        target = _graph.edited(ops, name, placed)
        try:
            frame = _graph.output(ops, target, "data")
        except OperationError as exc:
            ops.set_config(target, {key: before[key] for key in config if key in before})
            copied = f" The change went to {target!r}, a copy of {name!r}, which is back as it was." if target != name else ""
            raise ModelRetry(f"{exc}{copied}") from exc
        capped = f" (capped at {limit} rows)" if len(frame) >= limit else ""
        preview = f"\n\nResult{capped}: {format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"
        preview += _notes(await result_notes(src, sql, frame))
        preview += "".join(f"\n\n{problem}" for problem in await _follower_problems(ops, target))
        if isinstance(result, str):
            return result + preview
        if isinstance(result, ToolReturn) and isinstance(result.return_value, str):
            return dataclasses.replace(result, return_value=result.return_value + preview)
        return result
