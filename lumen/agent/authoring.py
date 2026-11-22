"""Tools that validate a query or chart before placing it on the dashboard as a node."""
from __future__ import annotations

import dataclasses

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
from ..flowdash.results import expand
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


def _result_input(ops, source: str) -> tuple[str, Any] | None:
    """The upstream SQL node and its result source when ``source`` names an earlier SQL node."""
    try:
        port = _graph.input_port(ops, source)
        return None if port is None else (port.rsplit(".", 1)[0], _graph.output(ops, *port.rsplit(".", 1)))
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc


async def _count(src: Any, table: str) -> int | None:
    """The rows of result table ``table`` of the result source ``src``, or None if counting fails."""
    count = sqlglot.select("COUNT(*)").from_(sqlglot.exp.Table(this=sqlglot.exp.to_identifier(table)))
    try:
        root, statement = expand(count.sql(dialect=src.dialect), src)
        frame = await root.execute_with_timeout(statement, QUERY_TIMEOUT)
        return int(as_pandas(frame).iloc[0, 0])
    except Exception:
        return None


async def _add_follower(ops, upstream: str, src: Any, name: str, sql: str) -> str:
    """Place a SQL node reading the result of ``upstream``, as ``add_sql`` places one on a source."""
    limit = effective_limit(ops.editor, None)
    sql, statement, frame = await _test_run(src, sql, limit)
    with primed_result(src, statement, frame):
        _place(ops, SQL_ID, name, {"sql": sql, "table": name}, f"{upstream}.result", f"{name}.source")
    try:
        frame = _graph.output(ops, name, "data")
        shown = _graph.output(ops, upstream, "data")
    except OperationError as exc:
        ops.remove_node(name)
        raise ModelRetry(str(exc)) from exc
    capped = f" (capped at {limit} rows)" if len(frame) >= limit else ""
    rows = _graph.config_of(ops, upstream).get("limit")
    note = ""
    if shown is not None and len(shown) >= effective_limit(ops.editor, rows if isinstance(rows, int) else None):
        total = await _count(src, _graph.output(ops, upstream, "pipeline").table)
        if total is not None and total > len(shown):
            note = f" It reads all {total} rows of {upstream!r}; its tile shows the first {len(shown)}."
    return (
        f"Placed SQL node {name!r} on the result of {upstream!r}{capped}.{note} "
        f"{format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"
    )


async def _follower_problems(ops, name: str) -> list[str]:
    """
    What the SQL nodes reading ``name``'s result directly now refuse,
    checked against its new result. A node reading one of those in turn is
    not checked: it fails, and reports it, once that one does.
    """
    followers = [f for f in _graph.reading(ops, f"{name}.result") if _graph.component_of(ops, f) == SQL_ID]
    if not followers:
        return []
    try:
        result = _graph.output(ops, name, "result")
    except OperationError:
        return []
    problems = []
    for follower in followers:
        problem = await check_sql_config({"sql": _graph.config_of(ops, follower).get("sql") or ""}, {"source": result})
        if isinstance(problem, CheckDiagnostic):
            problems.append(f"Follower {follower!r} now fails: {problem}")
    return problems


async def add_sql(ctx: RunContext[FlowDashDeps], source: str, name: str, sql: str) -> str:
    """Place a SQL node with a read-only query and return a preview of its result; errors come back to fix.

    To build on an earlier answer, pass its SQL node as source: the query then reads that
    result by its table name, as list_tables shows it, and runs again when that node changes.

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


def _refused(result: Any) -> bool:
    """Whether a FlowDash tool's ``result`` says it changed nothing, e.g. a refusal of Explore."""
    text = result.return_value if isinstance(result, ToolReturn) else result
    return isinstance(text, str) and text.startswith(("Refused:", "Not available:"))


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
        if "sql" not in config or (upstream := _graph.feeding(ops, f"{name}.source")) is None:
            return await handler(args)
        try:
            src = _graph.output(ops, *upstream.rsplit(".", 1))
        except OperationError as exc:
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
        preview += "".join(f"\n\n{problem}" for problem in await _follower_problems(ops, target))
        if isinstance(result, str):
            return result + preview
        if isinstance(result, ToolReturn) and isinstance(result.return_value, str):
            return dataclasses.replace(result, return_value=result.return_value + preview)
        return result
