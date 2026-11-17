"""Tools that validate a query or chart before placing it on the dashboard as a node."""
from __future__ import annotations

import copy

from typing import Any

import sqlglot

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from ..ai.agents.sql import (
    VALIDATION_MAX_ROWS, format_exploration_result, validate_read_only_sql,
)
from ..ai.agents.vega_lite import VegaLiteAgent
from ..ai.utils import (
    _spec_field_references, clean_sql, format_error, normalize_vegalite_spec,
)
from ..flowdash import SQL_ID, VIEW_ID
from ..pipeline import Pipeline
from ..transforms.sql import SQLLimit
from . import _graph
from .data import QUERY_TIMEOUT, lint, resolve_source


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
        validate_read_only_sql(sql, src.dialect)
        if sqlglot.parse_one(sql, read=None if src.dialect == "any" else src.dialect).find(sqlglot.exp.Table) is None:
            raise ValueError("The query reads no table. Select from the source's tables instead of returning literal values.")
        limited = SQLLimit(limit=VALIDATION_MAX_ROWS + 1, write=src.dialect).apply(sql)
    except (ValueError, sqlglot.errors.ParseError, sqlglot.errors.TokenError) as exc:
        raise ModelRetry(format_error(exc)) from exc
    try:
        await src.execute_with_timeout(limited, QUERY_TIMEOUT)
    except Exception as exc:
        raise ModelRetry(format_error(exc)) from exc
    _place(ops, SQL_ID, name, {"sql": sql, "table": name}, f"{source_node}.source", f"{name}.source")
    try:
        frame = _graph.output(ops, name, "data")
    except OperationError as exc:
        ops.remove_node(name)
        raise ModelRetry(str(exc)) from exc
    preview = format_exploration_result(frame)
    findings = lint(frame)
    return f"Placed SQL node {name!r}: {preview}\n\nData checks:\n{findings}"


def _transform_fields(spec: Any) -> set[str]:
    fields: set[str] = set()
    if isinstance(spec, dict):
        if isinstance(spec.get("transform"), list):
            fields.update(VegaLiteAgent._extract_as_keys(spec["transform"]))
        for value in spec.values():
            fields |= _transform_fields(value)
    elif isinstance(spec, list):
        for item in spec:
            fields |= _transform_fields(item)
    return fields


def _marks(spec: dict[str, Any]) -> list[str]:
    marks = []
    for layer in spec.get("layer", [spec]):
        mark = layer.get("mark") if isinstance(layer, dict) else None
        mark = mark.get("type") if isinstance(mark, dict) else mark
        if isinstance(mark, str):
            marks.append(mark)
    return marks


def check_chart(spec: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """
    The spec to store for a chart of a table with ``columns``, or ``ValueError``
    saying what to fix. The data comes from the node's input, so a spec may not
    bring its own.
    """
    if not isinstance(spec, dict):
        raise ValueError("The spec must be a Vega-Lite JSON object.")
    spec = copy.deepcopy(spec)
    data = spec.pop("data", None)
    spec.pop("datasets", None)
    if isinstance(data, dict) and ("url" in data or "values" in data):
        raise ValueError("The chart's data comes from the data node; remove `data` from the spec.")
    unknown = _spec_field_references(spec, "vega-lite") - set(columns) - _transform_fields(spec)
    if unknown:
        raise ValueError(
            f"The spec references fields the data does not have: {', '.join(sorted(unknown))}. "
            f"Its columns are: {', '.join(columns)}."
        )
    try:
        normalize_vegalite_spec(copy.deepcopy(spec))
    except Exception as exc:
        raise ValueError(f"Invalid Vega-Lite spec: {exc}") from exc
    return spec


def add_chart(ctx: RunContext[FlowDashDeps], data: str, name: str, spec: dict[str, Any]) -> str:
    """Place a Vega-Lite chart of a SQL node's result.

    Args:
        data: The name of the SQL node whose result the chart shows.
        name: A short descriptive snake_case name for the chart node, e.g. "revenue_by_region_chart".
        spec: A Vega-Lite spec without `data`; its encodings use the SQL node's column names.
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
