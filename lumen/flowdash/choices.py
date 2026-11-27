"""
The ``config_choices=`` hooks of the nodes and the column reader of
pipelines, see ``panel_flowdash.register`` and
``panel_flowdash.register_port_type``. The inspector offers what they
return as dropdowns, so a source ref, a table or a column is picked rather
than typed from memory. The ``palette_presets=`` hooks list each source the
host provides, and each of its tables, in the component palette.

A Filter node's ``value`` input, while unconnected, offers the distinct
values of its field. They are fetched in a worker thread, at most
``DISTINCT_LIMIT`` of them, and kept per pipeline, so asking again while the
inspector stays on the node runs no query.
"""
from __future__ import annotations

import asyncio
import inspect
import re
import threading
import weakref

from collections.abc import Callable, Mapping
from concurrent.futures import Future
from typing import Any

from panel_flowdash import Choices, current_dashboard, register
from panel_flowdash.util import is_failure

from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource, Source
from ..transforms.sql import SQLDistinct, SQLFilter, SQLLimit
from .checks import pipeline_columns

IGNORED_REF = "ignored while a source is connected"

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
    """
    Offer the refs of the sources the host provides for a ``Lumen/source``
    node, and the alias it names one by, if it does.
    """
    from panel_flowdash import dashboard_context

    from .access import authorized_sources
    from .nodes import SOURCE_ALIASES_KEY

    sources = authorized_sources()
    if not sources:
        return {}
    ref = config.get("ref")
    target = (dashboard_context().get(SOURCE_ALIASES_KEY) or {}).get(ref)
    if ref and ref not in sources and target in sources:
        return {"ref": {**{r: r for r in sorted(sources)}, f"{ref} (old name of {target})": ref}}
    return {"ref": sorted(sources)}


def table_choices(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, list[str]]:
    """
    Offer the refs of the host's sources for a ``Lumen/table`` node, and the
    tables of its source: the connected one, or else the one its ref names.
    With a source connected, a ref it still holds is only shown as ignored.
    """
    from .access import authorized_tables, can_access_source
    from .nodes import lookup_source

    source, ref = inputs.get("source"), config.get("ref")
    if source is None:
        choices: dict[str, Any] = dict(source_choices(config, inputs))
        source = lookup_source(ref) if ref else None
        if source is not None and not can_access_source(source):
            source = None
    else:
        choices = {"ref": {f"{ref} ({IGNORED_REF})": ref}} if ref else {}
        if isinstance(source, Source) and not can_access_source(source):
            source = None
    if isinstance(source, Source):
        choices["table"] = sorted(authorized_tables(source))
    return choices


# Tables per source, read once per reload: the palette asks for presets whenever it
# is drawn, and listing a remote database's tables is a query.
_TABLES: weakref.WeakKeyDictionary[Source, tuple[Any, Any, list[str]]] = weakref.WeakKeyDictionary()


def _tables(source: Source) -> list[str]:
    """
    The tables of ``source``, cached until it reloads: ``Source.clear_cache``,
    which a change of its reload params calls, replaces its schema cache, and
    a change of its ``tables`` changes the declared tables.
    """
    declared = getattr(source, "tables", None)
    declared = sorted(declared) if isinstance(declared, (dict, list)) else declared
    cached = _TABLES.get(source)
    token = getattr(source, "_schema_cache", None)
    if cached is not None and cached[0] is token and cached[1] == declared:
        return cached[2]
    from .nodes import listed_tables

    try:
        tables = sorted(listed_tables(source))
    except Exception:
        return []
    _TABLES[source] = (token, declared, tables)
    return tables


def _host_sources(context: Mapping[str, Any]) -> dict[str, Source]:
    from .access import authorized_sources

    return {ref: source for ref, source in authorized_sources(context).items() if isinstance(source, Source)}


def _authorized_cached_tables(source: Source, context: Mapping[str, Any]) -> list[str]:
    from .access import restricted_tables

    restricted = restricted_tables(source, context)
    tables = _tables(source)
    return [t for t in tables if t not in restricted] if restricted else tables


def source_presets(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A ``Lumen/source`` palette entry per source the host provides, with its ref filled in."""
    return [
        {
            "title": f"{ref} (source)",
            "description": f"The source {ref} the app provides, for SQL nodes and tables that share it.",
            "tooltip": f"Source {ref}: {', '.join(tables) or 'no tables'}",
            "tags": [ref, *tables],
            "config": {"ref": ref},
        }
        for ref, source in _host_sources(context).items()
        for tables in [_authorized_cached_tables(source, context)]
    ]


def _served_file(source: Source, table: str) -> str | None:
    from .nodes import served_file

    return served_file(source, table)


def table_presets(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    A ``Lumen/table`` palette entry per table of each source the host
    provides, with its ref and the table filled in, so the node reads the
    table by itself.
    """
    return [
        {
            "title": f"{table} ({ref})",
            "description": f"Table {table} of the source {ref}.",
            "tooltip": f"Source {ref}, table {table}" + (f", file {path}" if (path := _served_file(source, table)) else ""),
            "tags": [ref, table],
            "config": {"ref": ref, "table": table},
        }
        for ref, source in _host_sources(context).items()
        for table in _authorized_cached_tables(source, context)
    ]


def metric_field_choices(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Offer a ``Lumen/filter`` node on a metric the columns of the rows the
    metric aggregates besides its own, since it filters them before the
    aggregation. Other pipelines keep the declared column choices.
    """
    from .nodes import _metric_step, filterable_columns

    pipeline = inputs.get("pipeline")
    if pipeline is None or is_failure(pipeline) or not hasattr(pipeline, "sql_transforms") or _metric_step(pipeline) is None:
        return {}
    types = pipeline_column_types(pipeline) or {}
    return {"field": Choices(filterable_columns(pipeline), types={k: v for k, v in types.items() if v})}


def metric_choices(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Offer a ``Lumen/metric`` node the metrics the host defines and, for the
    one picked, its measures, dimensions, grains and the columns to sort by.
    """
    from .access import authorized_metrics

    metrics = authorized_metrics()
    if not metrics:
        return {}
    choices: dict[str, Any] = {"metric": list(metrics)}
    metric = metrics.get(config.get("metric"))
    if metric is None or metric.form == "sql":
        return choices
    choices["measures"] = list(metric.measures)
    choices["dimensions"] = list(metric.dimensions)
    picked = [metric.dimensions[d] for d in config.get("dimensions") or [] if d in metric.dimensions]
    grains = [g for d in picked if d.time for g in d.grains]
    choices["grain"] = {"none": "", **{grain: grain for grain in dict.fromkeys(grains)}}
    outputs = [*(config.get("dimensions") or []), *(config.get("measures") or [])]
    choices["order_by"] = {
        label: value for column in outputs
        for label, value in ((f"{column} (descending)", f"-{column}"), (column, column))
    }
    return choices


def parameter_choices(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """Offer a ``Lumen/parameter`` node the names the host's metrics take as parameters."""
    from .access import authorized_metrics
    from .nodes import metric_parameter_names

    names = metric_parameter_names(authorized_metrics())
    return {"parameter": names} if names else {}


def metric_presets(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A ``Lumen/metric`` palette entry per metric the host defines, with the metric filled in."""
    from .access import authorized_metrics

    presets = []
    for name, metric in authorized_metrics(context).items():
        config: dict[str, Any] = {"metric": name}
        if metric.form == "structured":
            config["measures"] = list(metric.measures)[:1]
        fields = [*metric.measures, *metric.dimensions, *metric.parameters]
        presets.append({
            "title": f"{name} (metric)",
            "description": metric.description or f"The metric {name} the app defines.",
            "tooltip": f"Metric {name}: {', '.join(fields) or 'a SQL query'}",
            "tags": ["metric", name, metric.source],
            "config": config,
        })
    return presets


def palette_presets(hook: Callable[[Mapping[str, Any]], list[dict[str, Any]]]) -> dict[str, Any]:
    """``register`` arguments offering ``hook``'s presets, where panel-flowdash supports them."""
    return {"palette_presets": hook} if "palette_presets" in inspect.signature(register).parameters else {}
def _distinct_sql(pipeline: Pipeline, field: str, limit: int) -> str | None:
    """
    The statement reading at most ``limit`` distinct values of ``field`` of
    a pipeline read straight from a SQL source, over the rows its filters
    keep but not under its row limit, or None for any other pipeline.

    A field the pipeline's result lacks is read from the rows of the first
    SQL transform whose ``input_rows(field, sql_in)`` gives a statement
    holding it, e.g. the rows an aggregation groups; without one, None.
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
    missing = field not in pipeline_columns(pipeline)
    for transform in pipeline.sql_transforms:
        input_rows = getattr(transform, "input_rows", None) if missing else None
        if input_rows is not None and (rows := input_rows(field, sql)) is not None:
            sql, missing = rows, False
            break
        if not isinstance(transform, SQLLimit):
            sql = transform.apply(sql)
    if missing:
        return None
    sql = SQLDistinct(columns=[field], write=source.dialect).apply(sql)
    return SQLLimit(limit=limit, write=source.dialect).apply(sql)


# The values a dashboard file holds as a literal; an object column of dates offers none.
_PLAIN = (str, bool, int, float)

_DIGITS = re.compile(r"(\d+)")


def _natural_key(value: Any) -> tuple:
    """Sort numbers before text and text naturally, so ``region_2`` comes before ``region_10``."""
    if isinstance(value, str):
        return (1, [(0, int(part), "") if part.isdigit() else (1, 0, part.casefold())
                    for part in _DIGITS.split(value) if part])
    return (0, value)


def distinct_values(pipeline: Pipeline, field: str, limit: int | None = None,
                    timeout: float | None = None, rows: int | None = None) -> list[Any] | None:
    """
    The distinct values of ``field`` in ``pipeline``, sorted and without
    nulls, or None when it has more than ``limit``, by default
    ``DISTINCT_LIMIT``, or no such field. On a SQL source the source
    computes them, capped in its query and stopped after ``timeout``
    seconds; otherwise they are read from the rows the pipeline fetches, at
    most ``rows``.
    """
    from .nodes import (
        RESULT_LIMIT, execute, fetch, python_value,
    )

    limit = DISTINCT_LIMIT if limit is None else limit
    timeout = DISTINCT_TIMEOUT if timeout is None else timeout
    sql = _distinct_sql(pipeline, field, limit + 1)
    if sql is not None:
        column = execute(pipeline.source, sql, timeout).iloc[:, 0]
    elif field in pipeline_columns(pipeline):
        column = fetch(pipeline, rows or RESULT_LIMIT, timeout)[field]
    else:
        return None
    unique = column.dropna().unique()
    if len(unique) > limit:
        return None
    values = [python_value(value) for value in unique]
    if not all(isinstance(value, _PLAIN) for value in values):
        return None
    return sorted(values, key=_natural_key)


def _read_distinct(pipeline: Pipeline, field: str, rows: int | None) -> list[Any] | None:
    try:
        kind = (pipeline_column_types(pipeline) or {}).get(field)
        # A dashboard file cannot hold a date as a literal, and a range suits them better.
        values = None if kind in ("datetime", "date") else distinct_values(pipeline, field, rows=rows)
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
    value, and on a metric the columns it can filter. The values are read in a worker thread, so the event loop stays free,
    and a read the inspector stops waiting for still completes and is kept
    for the next time it asks. Date and datetime fields, whose values a
    dashboard file cannot hold as literals, and fields with more than
    ``DISTINCT_LIMIT`` values offer none.
    """
    choices = metric_field_choices(config, inputs)
    pipeline, field = inputs.get("pipeline"), config.get("field")
    if not field or not isinstance(pipeline, Pipeline) or is_failure(pipeline):
        return choices
    from .nodes import row_ceiling

    future = _distinct_future(pipeline, field, row_ceiling(current_dashboard()))
    values = await asyncio.shield(asyncio.wrap_future(future))
    if values:
        choices["value"] = Choices(values, restrict=False)
    return choices
