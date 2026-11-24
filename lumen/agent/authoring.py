"""Tools that validate a query or chart before placing it on the dashboard as a node."""
from __future__ import annotations

from typing import Any

import sqlglot

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from ..ai.agents.sql import format_exploration_result
from ..ai.utils import clean_sql, format_error
from ..flowdash import (
    QUERY_TIMEOUT, SQL_ID, VIEW_ID, check_chart, check_query, primed_result,
    result_sql, row_ceiling,
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


async def add_sql(ctx: RunContext[FlowDashDeps], source: str, name: str, sql: str) -> str:
    """Place a SQL node that runs a read-only query on a source node, and return a preview of its result.

    The query is executed first; if it fails, the engine's error comes back for you to fix the
    query and call again. Use it for every result the user should see.

    Args:
        source: The name of the source node to query.
        name: A short descriptive snake_case name for the new node, e.g. "paid_total_by_region".
        sql: One SELECT or WITH query over the source's tables.
    """
    ops = ctx.deps.ops
    source_node, src = resolve_source(ctx.deps, source)
    try:
        sql = clean_sql(sql.strip(), src.dialect, prettify=True)
        check_query(sql, src)
        limit = row_ceiling(ops.editor)
        statement = result_sql(src, sql, limit)
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        raise ModelRetry(format_error(exc)) from exc
    try:
        frame = await src.execute_with_timeout(statement, QUERY_TIMEOUT)
    except Exception as exc:
        raise ModelRetry(format_error(exc)) from exc
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
    """Place a Vega-Lite chart of a SQL node's result.

    Args:
        data: The name of the SQL node whose result the chart shows.
        name: A short descriptive snake_case name for the chart node, e.g. "revenue_by_region_chart".
        spec: A Vega-Lite spec without `data` or URLs; its encodings use the SQL node's column names.
    """
    ops = ctx.deps.ops
    try:
        pipeline = _graph.output(ops, data, "pipeline")
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    if not isinstance(pipeline, Pipeline):
        raise ModelRetry(f"{data!r} has no Lumen pipeline output to chart; pass the name of a SQL node.")
    try:
        spec = check_chart(spec, [str(c) for c in pipeline.data.columns])
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    _place(ops, VIEW_ID, name, {"spec": spec}, f"{data}.pipeline", f"{name}.pipeline")
    return f"Placed chart {name!r} ({', '.join(_marks(spec)) or 'composite'}) of {data!r}."


def authoring_toolset() -> FunctionToolset[FlowDashDeps]:
    """The authoring tools: add_sql and add_chart."""
    return FunctionToolset[FlowDashDeps]([add_sql, add_chart], max_retries=3)
