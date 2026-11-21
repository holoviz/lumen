"""
The ``config_choices=`` hooks of the nodes and the column reader of
pipelines, see ``panel_flowdash.register`` and
``panel_flowdash.register_port_type``. The inspector offers what they
return as dropdowns, so a source ref, a table or a column is picked rather
than typed from memory.

A Filter node's ``value`` input, while unconnected, offers the distinct
values of its field. They are fetched in a worker thread, at most
``DISTINCT_LIMIT`` of them, and kept per pipeline, so asking again while the
inspector stays on the node runs no query.
"""
from __future__ import annotations

import asyncio
import threading
import weakref

from concurrent.futures import Future
from typing import Any

from panel_flowdash import Choices, current_dashboard
from panel_flowdash.util import is_failure

from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource, Source
from ..transforms.sql import SQLDistinct, SQLFilter, SQLLimit
from .checks import pipeline_columns

# Distinct values a Filter node offers for its value; a field with more offers none.
DISTINCT_LIMIT = 200
# Seconds the query reading them may run.
DISTINCT_TIMEOUT = 5

_DISTINCT_LOCK = threading.Lock()
# The distinct values read per pipeline and field, None where there were too many or reading failed.
_DISTINCT: weakref.WeakKeyDictionary[Pipeline, dict[str, list[Any] | None]] = weakref.WeakKeyDictionary()
_DISTINCT_PENDING: dict[tuple[int, str], Future] = {}

_TYPES = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean"}


def pipeline_column_types(pipeline: Any) -> dict[str, str | None] | None:
    """
    The columns of a Lumen pipeline with short type names, from its schema,
    so no rows are fetched: ``integer``, ``number``, ``boolean``,
    ``datetime``, ``date`` or ``string``.
    """
    if pipeline is None or is_failure(pipeline):
        return None
    schema = pipeline.schema
    types: dict[str, str | None] = {}
    for column in pipeline_columns(pipeline):
        spec = schema.get(column) or {}
        kind = spec.get("format") if spec.get("format") in ("datetime", "date") else spec.get("type")
        types[column] = "datetime" if kind == "datetime" else "date" if kind == "date" else _TYPES.get(kind)
    return types


def source_choices(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, list[str]]:
    """Offer the refs of the sources the host provides for a ``Lumen/source`` node."""
    from .nodes import host_sources

    sources = host_sources()
    return {"ref": sorted(sources)} if sources else {}


def table_choices(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, list[str]]:
    """Offer the tables of the connected source for a ``Lumen/table`` node."""
    source = inputs.get("source")
    if not isinstance(source, Source):
        return {}
    return {"table": sorted(source.get_tables())}


def _distinct_sql(pipeline: Pipeline, field: str, limit: int) -> str | None:
    """
    The statement reading at most ``limit`` distinct values of ``field`` of
    a pipeline read straight from a SQL source, over the rows its filters
    keep but not under its row limit, or None for any other pipeline.
    """
    source = pipeline.source
    if pipeline.pipeline is not None or pipeline.transforms or not isinstance(source, BaseSQLSource):
        return None
    if pipeline.table in getattr(source, "table_params", {}):
        return None
    sql = source.get_sql_expr(pipeline.table)
    conditions = list(pipeline._filter_query().items())
    if conditions and getattr(source, "filter_in_sql", True):
        sql = SQLFilter(conditions=conditions, write=source.dialect).apply(sql)
    for transform in pipeline.sql_transforms:
        if not isinstance(transform, SQLLimit):
            sql = transform.apply(sql)
    sql = SQLDistinct(columns=[field], write=source.dialect).apply(sql)
    return SQLLimit(limit=limit, write=source.dialect).apply(sql)


def distinct_values(pipeline: Pipeline, field: str, limit: int | None = None,
                    timeout: float | None = None, rows: int | None = None) -> list[Any] | None:
    """
    The distinct values of ``field`` in ``pipeline``, sorted and without
    nulls, or None when it has more than ``limit``, by default
    ``DISTINCT_LIMIT``. On a SQL source the source computes them, capped in
    its query and stopped after ``timeout`` seconds; otherwise they are read
    from the rows the pipeline fetches, at most ``rows``.
    """
    from .nodes import (
        RESULT_LIMIT, execute, fetch, python_value,
    )

    limit = DISTINCT_LIMIT if limit is None else limit
    timeout = DISTINCT_TIMEOUT if timeout is None else timeout
    sql = _distinct_sql(pipeline, field, limit + 1)
    if sql is not None:
        column = execute(pipeline.source, sql, timeout).iloc[:, 0]
    else:
        column = fetch(pipeline, rows or RESULT_LIMIT, timeout)[field]
    unique = column.dropna().unique()
    if len(unique) > limit:
        return None
    values = [python_value(value) for value in unique]
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def _read_distinct(pipeline: Pipeline, field: str, rows: int | None) -> list[Any] | None:
    try:
        kind = (pipeline_column_types(pipeline) or {}).get(field, "missing")
        # A dashboard file cannot hold a date as a literal, and a range suits them better.
        values = None if kind in ("missing", "datetime", "date") else distinct_values(pipeline, field, rows=rows)
    except Exception:
        values = None
    with _DISTINCT_LOCK:
        _DISTINCT.setdefault(pipeline, {})[field] = values
    return values


def _distinct_future(pipeline: Pipeline, field: str, rows: int | None = None) -> Future:
    """The read of ``field``'s distinct values in ``pipeline``: done if they were read before, shared while one runs."""
    from .nodes import _preview_pool

    with _DISTINCT_LOCK:
        known = _DISTINCT.get(pipeline, {})
        if field in known:
            future: Future = Future()
            future.set_result(known[field])
            return future
        key = (id(pipeline), field)
        if (pending := _DISTINCT_PENDING.get(key)) is not None:
            return pending
        future = _preview_pool().submit(_read_distinct, pipeline, field, rows)
        _DISTINCT_PENDING[key] = future

    def done(_):
        with _DISTINCT_LOCK:
            if _DISTINCT_PENDING.get(key) is future:
                del _DISTINCT_PENDING[key]

    future.add_done_callback(done)
    return future


def forget_distinct(pipeline: Any) -> None:
    """Drop the distinct values read for ``pipeline``."""
    with _DISTINCT_LOCK:
        _DISTINCT.pop(pipeline, None)


async def filter_choices(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Offer the distinct values of the field for a ``Lumen/filter`` node's
    value. They are read in a worker thread, so the event loop stays free,
    and a read the inspector stops waiting for still completes and is kept
    for the next time it asks. Date and datetime fields, whose values a
    dashboard file cannot hold as literals, and fields with more than
    ``DISTINCT_LIMIT`` values offer none.
    """
    pipeline, field = inputs.get("pipeline"), config.get("field")
    if not field or not isinstance(pipeline, Pipeline) or is_failure(pipeline):
        return {}
    from .nodes import row_ceiling

    future = _distinct_future(pipeline, field, row_ceiling(current_dashboard()))
    values = await asyncio.shield(asyncio.wrap_future(future))
    if not values:
        return {}
    return {"value": Choices(values, restrict=False)}
