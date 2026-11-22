"""Tools that validate a query or chart before placing it on the dashboard as a node."""
from __future__ import annotations

import dataclasses
import json

from typing import Any

import sqlglot

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturn

from ..ai.agents.sql import format_exploration_result
from ..ai.utils import clean_sql, format_error
from ..flowdash import (
    QUERY_TIMEOUT, SQL_ID, VIEW_ID, CheckDiagnostic, check_chart, check_query,
    check_sql_config, effective_limit, pipeline_columns, primed_result,
    result_sql, selection_warnings,
)
from ..flowdash.nodes import default_table
from ..flowdash.results import expand, merge
from ..pipeline import Pipeline
from ..util import as_pandas
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
        source: Source node.
        name: Descriptive snake_case node name.
        sql: One SELECT or WITH query.
    """
    if (upstream := _result_input(ctx.deps.ops, source)) is not None:
        return await _add_follower(ctx.deps.ops, *upstream, name, sql)
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


def add_chart(ctx: RunContext[FlowDashDeps], data: str, name: str, spec: dict[str, Any]) -> str:
    """Place a Vega-Lite chart of a SQL, table or filter node's rows.

    Args:
        data: SQL, table or filter node to chart.
        name: Descriptive snake_case node name.
        spec: Vega-Lite spec using the node's columns, without data or URLs. Leave out
            width and height so the chart fills its tile and follows resizes; set
            them only when the user asks for a fixed size.
    """
    ops = ctx.deps.ops
    port = _upstream_port(ops, data)
    try:
        pipeline = _graph.output(ops, data, port)
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    if not isinstance(pipeline, Pipeline):
        raise ModelRetry(f"{data!r} has no Lumen pipeline output to chart; pass the name of a SQL, table or filter node.")
    try:
        spec = check_chart(spec, pipeline_columns(pipeline))
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    _place(ops, VIEW_ID, name, {"spec": spec}, f"{data}.{port}", f"{name}.pipeline")
    placed = f"Placed chart {name!r} ({', '.join(_marks(spec)) or 'composite'}) of {data!r}."
    notes = selection_warnings(spec)
    return placed + "".join(f"\nNote: {note}" for note in notes)


def _upstream_port(ops, data: str) -> str:
    if data not in _graph.nodes(ops):
        raise ModelRetry(f"No node {data!r}.")
    port = _graph.pipeline_port(ops, data)
    if port is None:
        raise ModelRetry(f"{data!r} has no Lumen pipeline output to chart; pass the name of a SQL, table or filter node.")
    return port


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
    """The authoring tools: add_sql and add_chart."""
    return FunctionToolset[FlowDashDeps]([add_sql, add_chart], max_retries=3)


class CheckedConfig(AbstractCapability[FlowDashDeps]):
    """
    Checks what ``set_config`` gives a Lumen SQL or chart node as add_sql and
    add_chart check it, before the node changes: SQL is cleaned, checked and
    run, and a chart spec is checked against its data's columns. A problem
    comes back for the model to fix, and a changed query reports its result
    and which SQL nodes reading its result directly its change breaks.
    """

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        config = args.get("config") if tool_def.name == "set_config" else None
        if not isinstance(config, dict):
            return await handler(args)
        ops, name = ctx.deps.ops, args.get("name")
        component = next((node["component"] for node in ops.nodes() if node["name"] == name), None)
        if component == VIEW_ID and "spec" in config:
            data = _graph.upstream(ops, name)
            port = _graph.pipeline_port(ops, data) if data else None
            try:
                pipeline = _graph.output(ops, data, port) if port else None
            except OperationError:
                pipeline = None
            if isinstance(pipeline, Pipeline):
                try:
                    config = {**config, "spec": check_chart(config["spec"], pipeline_columns(pipeline))}
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
