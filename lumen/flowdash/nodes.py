"""
Lumen sources, tables, filters, SQL queries and Vega-Lite views as FlowDash
nodes. View nodes share ``LumenViewNode``, registered with ``register_view``.

Edges between these nodes carry a Lumen ``Source`` or ``Pipeline`` rather
than a DataFrame, so a query placed on a SQL node runs in the source's
engine, and the filters and row limit of table and filter nodes become part
of the query the source runs rather than being applied after fetching. Table
and filter nodes fetch nothing themselves: rows are fetched where they are
needed, by a data node (the boundary to components that take a DataFrame),
a view node or a tile preview. Each node resolves its Lumen object once per
change and every output reads that object.

Source and table nodes never build a source from their config: the host
hands the editor the sources a dashboard may use, by ref, in the editor's
context under ``"lumen.sources"`` (``FlowDash(context={"lumen.sources":
{...}})``, or ``bind_source`` on an editor that exists), and a node only
names one of them by ``ref``. Whoever edits a dashboard can therefore only
point it at sources the host chose, never open a new connection. A node
whose ref is bound later picks the source up then. A table node names its
source by ``ref`` itself, so one node reads a table; a source node is for
SQL nodes and for table nodes that share it through its output.

The host also sets how many rows a query of these nodes may return
(``set_row_ceiling``); a node's ``limit`` can lower that, never raise it.
"""
from __future__ import annotations

import asyncio
import copy
import datetime as dt
import decimal
import functools
import hashlib
import inspect
import logging
import math
import threading
import time
import weakref

from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import (
    Future, ThreadPoolExecutor, TimeoutError as FutureTimeout,
)
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import panel as pn
import panel_material_ui as pmui
import param
import sqlglot
import yaml

from panel.viewable import Viewer
from panel_flowdash import (
    current_dashboard, current_node_id, dashboard_context, register,
    register_port_type,
)
from panel_flowdash.dataflow_engine import failure_text
from panel_flowdash.selection import (
    DEBOUNCE, EMPTY, SELECTION_TYPE, Selection, link_selection,
)
from panel_flowdash.util import is_failure

from ..data.sql import _legal_order
from ..data.utils import ANSI_ESCAPE
from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource, QueryTimeoutError, Source
from ..sources.duckdb import DuckDBSource
from ..transforms.base import Filter as FilterTransform, Iloc, Range
from ..transforms.sql import (
    NO_BOOLEAN_LITERALS, MetricAggregate, SQLDistinct, SQLFilter, SQLLimit,
    SQLMinMax, SQLSort,
)
from ..util import get_dataframe_schema
from .checks import (
    CheckDiagnostic, _engine_error, _read, _unverified, check_chart,
    check_filter_config, check_query, check_source_config, check_sql_config,
    check_table_config, check_view_spec, forget_probes, pipeline_columns,
    selection_warnings,
)
from .choices import (
    filter_choices, forget_distinct, metric_choices, metric_presets,
    palette_presets, parameter_choices, pipeline_column_types, source_choices,
    source_presets, table_choices, table_presets,
)
from .contracts import (
    DATA_ID, FILTER_ID, METRIC_ID, PARAMETER_ID, SOURCE_ID, SQL_ID, TABLE_ID,
    VIEW_ID, SourceContract, spec_fields as contract_spec_fields,
)
from .defaults import vega_lite_default
from .metrics import (
    METRICS_KEY, Metric, MetricError, MetricPreFilter, Query, aggregate,
    bind_sql, coerce_dimension_values, prepare, table_sql,
)
from .redact import redact_spec
from .results import (
    derive, expand, is_derived, merge, not_a_source, root_of,
)
from .view_settings import VEGA_LITE_SETTINGS, themed_spec

try:
    from panel_flowdash import Choices
except ImportError:
    Choices = None

try:
    from panel_flowdash import request_filters
except ImportError:
    # A panel-flowdash without global filters never asks the nodes for their columns.
    request_filters = None

try:
    from panel_flowdash import placement_context
except ImportError:
    # panel-flowdash 0.1.0rc7 and earlier, where the ``_legacy_`` helpers and ``_next_tick`` stand in for it.
    placement_context = None

SOURCE_COLOR = "#6D4C41"
PIPELINE_COLOR = "#0072B5"

logger = logging.getLogger(__name__)


def _register_port_type(name: str, **options: Any) -> str:
    """
    Register a port type with the ``options`` (``color``, ``columns``) the
    installed panel-flowdash accepts, dropping the ones it does not know yet.
    """
    params = inspect.signature(register_port_type).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        options = {key: value for key, value in options.items() if key in params}
    return register_port_type(name, **options)


SOURCE_TYPE = _register_port_type("lumen.Source", color=SOURCE_COLOR)
# Its columns come from the schema, so the inspector offers them without fetching rows.
PIPELINE_TYPE = _register_port_type(
    "lumen.Pipeline", color=PIPELINE_COLOR, columns=pipeline_column_types
)

# The key of the editor context the host puts its sources in, by ref.
SOURCES_KEY = "lumen.sources"
# The key of further names of those sources, ``{alias: ref}``, e.g. the positional
# refs ``lumen serve`` gave before its refs were stable.
SOURCE_ALIASES_KEY = "lumen.source_aliases"

# Rows a SQL node keeps in memory and a table, filter or data node lets the source
# return, unless the host sets another ceiling.
RESULT_LIMIT = 100_000
# Seconds a query placed by add_sql may run; it runs off the event loop.
QUERY_TIMEOUT = 30
# Rows a table or filter node shows in its own preview.
PREVIEW_ROWS = 50
# Seconds a query may run when a node resolves it itself, e.g. after set_config or an
# import: that happens in a synchronous watcher, which blocks the event loop meanwhile.
NODE_TIMEOUT = 5

_CEILINGS: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
_SOURCE_NODES: weakref.WeakKeyDictionary[Any, weakref.WeakSet] = weakref.WeakKeyDictionary()
_RESULTS: dict[tuple[int, str], Any] = {}
# Rows fetched per pipeline, by the limit they were fetched under, so the nodes that
# read one pipeline (a data node and a chart, say) run its query once.
_FETCHED: weakref.WeakKeyDictionary[Pipeline, dict[float, Any]] = weakref.WeakKeyDictionary()
# What a SQL node's result source is made of, by its pipeline, until the source is first read.
_RESULT_SOURCES: weakref.WeakKeyDictionary[Pipeline, Any] = weakref.WeakKeyDictionary()


# Marks a table that only keeps a name an earlier release gave a served file, so
# saved queries still read it; listings leave it out. Derived sources copy it.
LEGACY_TABLE = "/* lumen.flowdash: earlier name */"


def add_legacy_tables(source: Source, names: Mapping[str, str]) -> None:
    """Let queries on a DuckDB ``source`` read its table ``names[name]`` as ``name`` too, unlisted."""
    connection = getattr(source, "_connection", None)
    if connection is None or not names:
        return
    for name, table in names.items():
        connection.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM "{table}"')
    source.tables = {**source.tables, **{name: f'SELECT * FROM "{table}" {LEGACY_TABLE}' for name, table in names.items()}}


# The file each table of a served data files' source reads, by table.
_SERVED_FILES: weakref.WeakKeyDictionary[Source, dict[str, str]] = weakref.WeakKeyDictionary()


def set_served_files(source: Source, files: Mapping[str, str]) -> None:
    """Record the file, as given, that each table of ``source`` reads."""
    _SERVED_FILES[source] = dict(files)


def served_file(source: Source, table: str) -> str | None:
    """The file, as given, that ``table`` of a served ``source`` reads, if it reads one."""
    return _SERVED_FILES.get(source, {}).get(table)


def listed_tables(source: Source) -> list[str]:
    """The tables of ``source`` to list, without those that only keep an earlier name."""
    declared = getattr(source, "tables", None)
    tables = source.get_tables()
    if not isinstance(declared, dict):
        return tables
    return [t for t in tables if not (isinstance(declared.get(t), str) and LEGACY_TABLE in declared[t])]


def bind_source(editor: Any, source: Source, ref: str = "data") -> str:
    """
    Let source nodes of ``editor`` whose ``ref`` is ``ref`` use ``source``,
    including nodes already placed, by adding it to the editor's context.
    Returns the ref.
    """
    editor.context.setdefault(SOURCES_KEY, {})[ref] = source
    aliases = editor.context.get(SOURCE_ALIASES_KEY) or {}
    for node in list(_SOURCE_NODES.get(editor, ())):
        if ref in (node.ref, aliases.get(node.ref)):
            node._refresh()
    return ref


def unbind_source(editor: Any, ref: str) -> None:
    """Forget a source bound with ``bind_source``."""
    editor.context.get(SOURCES_KEY, {}).pop(ref, None)


def bound_sources(editor: Any) -> dict[str, Source]:
    return dict(editor.context.get(SOURCES_KEY) or {})


def host_sources(context: Mapping[str, Any] | None = None) -> dict[str, Source]:
    """
    The sources the host lets a dashboard use, by ref, from ``context``
    or else ``dashboard_context()``. A node looks one up by ref with
    ``lookup_source``.
    """
    context = dashboard_context() if context is None else context
    return dict(context.get(SOURCES_KEY) or {})


def lookup_source(ref: str, context: Mapping[str, Any] | None = None) -> Source | None:
    """
    The host source ``ref`` names in ``context`` (or ``dashboard_context()``):
    the source of that ref, or else of the ref it is an alias of.
    """
    context = dashboard_context() if context is None else context
    sources = host_sources(context)
    if ref in sources:
        return sources[ref]
    alias = (context.get(SOURCE_ALIASES_KEY) or {}).get(ref)
    return sources.get(alias) if isinstance(alias, str) else None


def host_metrics(context: Mapping[str, Any] | None = None) -> Mapping[str, Metric]:
    """
    The metrics the host lets a dashboard use, by name, from ``context`` or
    else ``dashboard_context()``: one snapshot of the ``MetricSet`` under
    ``METRICS_KEY``, which an admin's save does not change while it is read.
    """
    context = dashboard_context() if context is None else context
    metrics = context.get(METRICS_KEY)
    snapshot = getattr(metrics, "snapshot", None)
    return snapshot() if callable(snapshot) else (metrics or {})


def missing_metric(name: str, metrics: Mapping[str, Metric]) -> str:
    if metrics:
        return f"The app defines no metric `{name}`; its metrics are: {', '.join(f'`{m}`' for m in metrics)}."
    return (
        "The app defines no metrics. Define them in the `metrics:` section of the project file "
        "(`flowdash.yaml`, or `lumen serve --flowdash --project FILE`)."
    )


def missing_source(ref: str, sources: Mapping[str, Source]) -> str:
    """What a source node naming ``ref`` reports when the host provides only ``sources``."""
    if sources:
        return f"The app provides no source `{ref}`; its sources are: {', '.join(f'`{r}`' for r in sorted(sources))}."
    return (
        "The app provides no Lumen sources. Serve the data with `lumen serve <data> --flowdash`, "
        "or set `editor_options = {\"context\": {\"lumen.sources\": {\"data\": source}}}` "
        "in the FlowDash project's `__init__.py`."
    )


def set_row_ceiling(editor: Any, rows: int) -> None:
    """Let the SQL nodes of ``editor`` keep up to ``rows`` rows instead of ``RESULT_LIMIT``."""
    _CEILINGS[editor] = rows


def row_ceiling(editor: Any) -> int:
    return _CEILINGS.get(editor, RESULT_LIMIT) if editor is not None else RESULT_LIMIT


def effective_limit(editor: Any, limit: int | None) -> int:
    ceiling = row_ceiling(editor)
    return ceiling if limit is None else min(limit, ceiling)


def result_sql(source: BaseSQLSource, sql: str, limit: int = RESULT_LIMIT) -> str:
    """The statement a SQL node runs for ``sql``, with its row cap."""
    return SQLLimit(limit=limit, write=source.dialect, pretty=True, identify=False).apply(sql)


def python_value(value: Any) -> Any:
    """
    ``value`` with numpy scalars, also inside lists and tuples, turned into
    Python ones; datetime64 and timedelta64 become ``datetime`` and
    ``timedelta``, which SQL renders as timestamps and pandas compares.
    """
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime()
    if isinstance(value, np.timedelta64):
        return pd.Timedelta(value).to_pytimedelta()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return type(value)(python_value(v) for v in value)
    return value


@contextmanager
def primed_result(source: BaseSQLSource, sql: str, frame: Any) -> Iterator[None]:
    """
    Within the block, hand the result of ``sql`` (a ``result_sql``
    statement), already executed by the caller, to the next SQL node that
    would run it, so it runs once. On a SQL node's result source, ``sql`` is
    the statement ``expand`` gives, which runs on the root source.
    """
    key = (id(root_of(source)), sql)
    _RESULTS[key] = frame
    try:
        yield
    finally:
        _RESULTS.pop(key, None)


def execute(source: BaseSQLSource, sql: str, timeout: float = QUERY_TIMEOUT) -> Any:
    """Run ``sql`` on ``source`` in a worker thread, stopping it after ``timeout`` seconds."""
    if (primed := _RESULTS.pop((id(source), sql), None)) is not None:
        return primed
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, source.execute_with_timeout(sql, timeout)).result()


def compile_sql(source: Source, sql: str, table: str, limit: int = RESULT_LIMIT,
                timeout: float = NODE_TIMEOUT) -> Pipeline:
    """
    A pipeline whose table is ``sql`` evaluated by ``source``'s engine, with
    its result fetched once under ``limit`` rows and ``timeout`` seconds.

    The table is the full statement and the cap the pipeline's last SQL
    transform, as on ``table_pipeline``: a filter or an aggregate downstream
    goes into the query ahead of the cap, so it reads every row of the
    result, not only the ones the node keeps. SQL does not promise that the
    outer query keeps the subquery's ORDER BY; DuckDB keeps it in practice.

    On another SQL node's result source, ``sql`` runs as one statement on
    the root source with the results it reads inlined (``expand``).
    ``result_source`` gives the result as a source of its own.

    Raises ``ValueError`` unless ``sql`` passes ``check_query``.
    """
    if is_failure(source):
        raise ValueError(f"The source failed: {failure_text(source)}")
    if not isinstance(source, BaseSQLSource):
        raise ValueError(f"{type(source).__name__} does not run SQL.")
    check_query(sql, source)
    root, statement = expand(sql, source)
    frame = execute(root, result_sql(root, statement, limit), timeout)
    # The result table joins the source's own tables, so it must not shadow one the query reads,
    # in any case, since engines and the CTEs inlined for it match names regardless of case.
    taken = {name.lower() for name in source.get_tables()}
    while table.lower() in taken:
        table = f"{table}_result"
    kwargs = {"materialize": False} if isinstance(root, DuckDBSource) else {}
    # The full statement with the cap as the pipeline's last step, so a filter or an aggregate
    # downstream goes into the query ahead of the cap and reads every row of the result.
    query_source = root.create_sql_expr_source({table: _derived_table(statement, root.dialect)}, **kwargs)
    schema = get_dataframe_schema(frame)["items"]["properties"]
    pipeline = Pipeline(
        source=query_source, table=table, schema=schema, data=frame, name=table,
        sql_transforms=[SQLLimit(limit=limit, write=root.dialect)],
    )
    _FETCHED[pipeline] = {math.inf: frame}
    _RESULT_SOURCES[pipeline] = (source, table, sql, schema)
    return pipeline


def _derived_table(sql: str, dialect: str) -> str:
    """
    ``sql`` as the steps after it read it, from a derived table: on SQL
    Server a top-level ORDER BY gets ``OFFSET 0 ROWS``, without which it
    refuses one there; elsewhere ``sql`` is unchanged.
    """
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.SqlglotError:
        return sql
    if tree.args.get("offset") is not None:
        return sql
    _legal_order(tree, dialect)
    return sql if tree.args.get("offset") is None else tree.sql(dialect=dialect)


def result_source(pipeline: Pipeline) -> BaseSQLSource | None:
    """
    The source of the SQL node that compiled ``pipeline``, plus its result
    as a table: the full query, not the capped statement, so a query on it
    counts every row. Its table is the pipeline's.
    """
    entry = _RESULT_SOURCES.get(pipeline)
    if isinstance(entry, tuple):
        entry = _RESULT_SOURCES[pipeline] = derive(*entry)
    return entry


def table_pipeline(source: Source, table: str, limit: int = RESULT_LIMIT) -> Pipeline:
    """
    A pipeline over ``table`` of ``source`` that fetches nothing until its
    data is read. On a SQL source the row limit is part of the query the
    source runs, as are the filters a filter node adds later; on other
    sources the rows are cut after fetching.
    """
    if is_failure(source):
        raise ValueError(f"The source failed: {failure_text(source)}")
    if not isinstance(source, Source):
        raise ValueError(f"Expected a Lumen source, got {type(source).__name__}.")
    if table not in source.get_tables():
        raise ValueError(
            f"The source has no table {table!r}. Its tables are: {', '.join(sorted(listed_tables(source)))}."
        )
    if source._supports_sql:
        return Pipeline(
            source=source, table=table, sql_transforms=[SQLLimit(limit=limit, write=source.dialect)], name=table,
        )
    return Pipeline(source=source, table=table, transforms=[Iloc(end=limit)], name=table)


def filter_value(value: Any) -> bool:
    """Whether ``value`` selects something; None and empty strings or lists filter nothing."""
    return not (value is None or (isinstance(value, (str, list, tuple, set)) and len(value) == 0))


_SCALARS = (str, bool, int, float, dt.date)


def filter_condition(value: Any) -> Any:
    """
    ``value`` as a filter condition, or ``ValueError`` if no source can
    apply it: a scalar (text, number, boolean, date or datetime) keeps equal
    rows, a list or set of scalars keeps rows equal to one of them, and a
    ``(start, end)`` tuple of numbers or dates, either of which may be None,
    keeps rows within it. numpy scalars count as the Python ones.
    """
    value = python_value(sorted(value) if isinstance(value, set) else value)
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, list) and all(v is None or isinstance(v, _SCALARS) for v in value):
        return value
    if (isinstance(value, tuple) and len(value) == 2 and any(v is not None for v in value)
            and all(v is None or (isinstance(v, (int, float, dt.date)) and not isinstance(v, bool)) for v in value)):
        return value
    if isinstance(value, Range):
        bounds = filter_condition((value.start, value.end))
        return Range(*bounds, closed=value.closed)
    raise ValueError(
        f"Cannot filter on {value!r}: give a single value, a list of values, or a (start, end) "
        "range of numbers or dates."
    )


def selection_conditions(selection: Selection) -> list[tuple[str, Any]]:
    """
    ``selection``'s predicates as filter conditions: a range closed on both
    ends is a ``(start, end)`` tuple, one closed on the left a ``Range``, and
    ``in`` a list. Raises ``ValueError`` for a selection of rows by position.
    """
    selection.check_filterable()
    conditions = []
    for predicate in selection.predicates:
        if predicate.op == "in":
            conditions.append((predicate.field, list(predicate.values)))
        elif predicate.closed == "left":
            conditions.append((predicate.field, Range(*predicate.values, closed="left")))
        else:
            conditions.append((predicate.field, tuple(predicate.values)))
    return conditions


def filtered_pipeline(pipeline: Pipeline, field: str, value: Any, limit: int = RESULT_LIMIT) -> Pipeline:
    """
    ``pipeline`` keeping the rows whose ``field`` matches ``value``, see
    ``filter_condition``; an empty value keeps every row.

    On a pipeline read straight from a SQL source the condition is added to
    the source's query, before the row limit, so the limit caps the filtered
    rows and every filter added this way narrows the ones before it. The
    query is capped at ``limit`` rows unless the pipeline already caps it.
    On other sources, and on a pipeline derived from another one, the rows
    are filtered after fetching, before any row limit.

    The two paths compare values as their engines do: SQL compares range
    bounds as text the engine casts to the column's type, so a range on a
    text column is lexical, while filtering fetched rows compares the
    Python values.
    """
    _check_pipeline(pipeline)
    columns = filterable_columns(pipeline)
    if field not in columns:
        raise ValueError(f"The data has no column {field!r}. Its columns are: {', '.join(columns)}.")
    if not filter_value(value):
        return pipeline
    return conditioned_pipeline(pipeline, [(field, value)], limit)


def _check_pipeline(pipeline: Any) -> None:
    if is_failure(pipeline):
        raise ValueError(f"The input failed: {failure_text(pipeline)}")
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Expected a Lumen pipeline, got {type(pipeline).__name__}.")


def _metric_step(pipeline: Pipeline) -> tuple[int, Any] | None:
    """Where a metric pipeline aggregates or reads its tables, and that step, if ``pipeline`` is one."""
    if pipeline.pipeline is not None:
        return None
    return next(
        ((i, t) for i, t in enumerate(pipeline.sql_transforms) if isinstance(t, (MetricAggregate, MetricPreFilter))),
        None,
    )


def filterable_columns(pipeline: Pipeline) -> list[str]:
    """
    The columns a filter on ``pipeline`` may name: its own, and on a metric
    the columns of the rows it aggregates or the tables its query reads.
    """
    columns = pipeline_columns(pipeline)
    step = _metric_step(pipeline)
    if step is None:
        return columns
    transform = step[1]
    base = transform.columns if isinstance(transform, MetricAggregate) else [
        column for names in transform.tables.values() for column in names
    ]
    return list(dict.fromkeys([*columns, *base]))


def _pushed_down(pipeline: Pipeline, condition: list[tuple[str, Any]]) -> tuple[Pipeline, list[tuple[str, Any]]]:
    """
    ``pipeline`` with the conditions that filter a metric's rows before it
    aggregates them applied there, and the conditions left to filter its
    result. A condition on a column of the aggregated rows, or on a
    dimension that is one, filters before the aggregation; one on a measure
    or a truncated time dimension filters the result. On a SQL metric a
    condition on a column of a table the query reads that the result does
    not have filters that table inside the query.
    """
    step = _metric_step(pipeline)
    if step is None:
        return pipeline, condition
    at, transform = step
    outputs = pipeline_columns(pipeline)
    before, after = [], []
    if isinstance(transform, MetricAggregate):
        plain = {}
        for name, expr in transform.dimensions:
            parsed = sqlglot.parse_one(expr, read=transform.read)
            if isinstance(parsed, sqlglot.exp.Column) and not parsed.table:
                plain[name] = parsed.name
        for field, value in condition:
            if field in plain:
                before.append((plain[field], value))
            elif field not in outputs and field in transform.columns:
                before.append((field, value))
            else:
                after.append((field, value))
        if not before:
            return pipeline, after
        dialect = pipeline.source.dialect
        sql_transforms = list(pipeline.sql_transforms)
        sql_transforms.insert(at, SQLFilter(
            conditions=before, write=dialect, boolean_literals=dialect not in NO_BOOLEAN_LITERALS,
        ))
        return _clone(pipeline, sql_transforms=sql_transforms), after
    tables: dict[str, list] = {}
    for field, value in condition:
        owners = [table for table, columns in transform.tables.items() if field in columns]
        if field in outputs or not owners:
            after.append((field, value))
            continue
        for table in owners:
            tables.setdefault(table, []).append((field, value))
    if not tables:
        return pipeline, after
    conditions = {table: list(conds) for table, conds in transform.conditions}
    for table, conds in tables.items():
        conditions.setdefault(table, []).extend(conds)
    sql_transforms = list(pipeline.sql_transforms)
    sql_transforms[at] = MetricPreFilter(
        conditions=list(conditions.items()), tables=transform.tables, write=transform.write,
        boolean_literals=transform.boolean_literals,
    )
    return _clone(pipeline, sql_transforms=sql_transforms), after


def conditioned_pipeline(pipeline: Pipeline, conditions: list[tuple[str, Any]],
                         limit: int = RESULT_LIMIT) -> Pipeline:
    """
    ``pipeline`` keeping the rows that match every ``(field, value)``
    condition, as one filter step. On a metric pipeline a condition on the
    rows it aggregates is applied before the aggregation, see ``_pushed_down``.
    """
    _check_pipeline(pipeline)
    columns = filterable_columns(pipeline)
    for field, _ in conditions:
        if field not in columns:
            raise ValueError(f"The data has no column {field!r}. Its columns are: {', '.join(columns)}.")
    condition = [(field, filter_condition(value)) for field, value in conditions if filter_value(value)]
    if not condition:
        return pipeline
    pipeline, condition = _pushed_down(pipeline, condition)
    if not condition:
        return pipeline
    if pipeline.pipeline is not None:
        return pipeline.chain(transforms=[FilterTransform(conditions=condition)])
    if not pipeline.source._supports_sql:
        transforms = list(pipeline.transforms)
        at = next((i for i, t in enumerate(transforms) if isinstance(t, Iloc)), len(transforms))
        transforms.insert(at, FilterTransform(conditions=condition))
        return _clone(pipeline, transforms=transforms)
    sql_transforms = list(pipeline.sql_transforms)
    at = next((i for i, t in enumerate(sql_transforms) if isinstance(t, SQLLimit)), None)
    if at is None:
        at = len(sql_transforms)
        sql_transforms.append(SQLLimit(limit=limit, write=pipeline.source.dialect))
    dialect = pipeline.source.dialect
    sql_transforms.insert(at, SQLFilter(
        conditions=condition, write=dialect, boolean_literals=dialect not in NO_BOOLEAN_LITERALS,
    ))
    return _clone(pipeline, sql_transforms=sql_transforms)


def _clone(pipeline: Pipeline, **params) -> Pipeline:
    """
    ``pipeline`` with ``params`` changed and no data. ``Pipeline.clone``
    copies the parent's data, and reading it to copy it would fetch the
    parent's rows, which is what pushing the filter down avoids.
    """
    values = {name: getattr(pipeline, name) for name in pipeline.param if name not in ("name", "data")}
    return type(pipeline)(**dict(values, **params))


def pipeline_sql(pipeline: Pipeline, limits: bool = True) -> str | None:
    """
    The statement a SQL source runs for ``pipeline``, filters and limit
    included (the limit left out with ``limits=False``), or None when the
    pipeline is not read straight from a SQL source.
    """
    source = pipeline.source
    if pipeline.pipeline is not None or not isinstance(source, BaseSQLSource):
        return None
    sql = source.get_sql_expr(pipeline.table)
    conditions = list(pipeline._filter_query().items())
    if conditions and getattr(source, "filter_in_sql", True):
        sql = SQLFilter(conditions=conditions, write=source.dialect).apply(sql)
    for transform in pipeline.sql_transforms:
        if limits or not isinstance(transform, SQLLimit):
            sql = transform.apply(sql)
    return sql


def fetch(pipeline: Pipeline, limit: int = RESULT_LIMIT, timeout: float = NODE_TIMEOUT) -> Any:
    """
    The rows of ``pipeline``, at most ``limit``, raising what the source
    raises; reading ``Pipeline.data`` instead would turn a failed query
    into a notification and no data. On a SQL source the limit is part of
    the query, which is stopped after ``timeout`` seconds; otherwise
    fetching is abandoned after ``timeout`` seconds and runs on in its
    thread. Each pipeline's rows are fetched once per limit; a node that
    changes builds a new pipeline, and ``forget`` drops what was fetched.

    A filter node's conditions are part of the pipeline's SQL transforms,
    so they are applied in SQL even on a source with ``filter_in_sql``
    off; that setting only affects the pipeline's own ``filters``.
    """
    fetched = _FETCHED.setdefault(pipeline, {})
    # A copy, since a worker fetching a card's preview may add to it meanwhile.
    for fetched_limit, data in list(fetched.items()):
        if fetched_limit >= limit:
            return data if len(data) <= limit else data.head(limit)
    sql = pipeline_sql(pipeline)
    source = pipeline.source
    if sql is not None and pipeline.table not in getattr(source, "table_params", {}) and not pipeline.transforms:
        data = execute(source, SQLLimit(limit=limit, write=source.dialect).apply(sql), timeout)
    else:
        data = _compute(pipeline, timeout)
    data = _parse_temporal(pipeline, data if len(data) <= limit else data.head(limit))
    fetched[limit] = data
    return data


# FlowDash's global filters (panel-flowdash releases after 0.1.0rc7) hand a view the
# predicates of the filter bar, ``{column: Predicate}``, and ask it for its
# columns and the values a filter offers. The nodes below add the predicates
# to the query of what they show rather than filtering fetched rows.

# The most distinct values a select filter offers, as panel-flowdash caps them.
FILTER_OPTIONS = 1000


class FilterMismatch(ValueError):
    """A global filter's values cannot be compared with the column of that name, e.g. text with numbers."""


def _number(value: Any) -> Any:
    """
    ``value`` as a number for a column of numbers. Strings are parsed
    exactly, so an integral one stays an exact ``int`` however large;
    others become floats, the precision of such a column. NaN, infinities
    and values beyond a float's range are refused.
    """
    if isinstance(value, bool):
        raise FilterMismatch(f"holds numbers, and {value!r} is not one")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        number = decimal.Decimal(value) if math.isfinite(value) else None
    elif isinstance(value, str):
        try:
            number = decimal.Decimal(value.strip())
        except decimal.InvalidOperation:
            raise FilterMismatch(f"holds numbers, and {value!r} is not one") from None
    else:
        raise FilterMismatch(f"holds numbers, and {value!r} is not one")
    if number is None or not number.is_finite() or not math.isfinite(float(number)):
        raise FilterMismatch(f"holds numbers, and {value!r} is not a finite one")
    if number == number.to_integral_value():
        return int(number)
    if isinstance(value, float):
        return value
    return float(number)


def _moment(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError:
            raise FilterMismatch(f"holds dates, and {value!r} is not one") from None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, dt.date):
        return value
    raise FilterMismatch(f"holds dates, and {value!r} is not one")


def _text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


def _coerce(value: Any, kind: str | None) -> Any:
    """``value`` made comparable with a column of ``kind``, see ``pipeline_column_types``."""
    value = python_value(value)
    if isinstance(value, str) and "\x00" in value:
        raise FilterMismatch("cannot be compared with a value holding a NUL character")
    if value is None or kind is None:
        return value
    if kind in ("integer", "number"):
        return _number(value)
    if kind in ("datetime", "date"):
        return _moment(value)
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise FilterMismatch(f"holds booleans, and {value!r} is not one")
    return _text(value)


def global_condition(predicate: Any, kind: str | None = None) -> Any:
    """
    The filter condition of a FlowDash global filter ``predicate`` on a
    column of ``kind``, as ``filter_condition`` takes it: ``in`` is a list
    of values, a range closed on both ends a ``(start, end)`` tuple and one
    closed on the left a ``Range``. Values are converted where that is safe,
    as panel-flowdash converts them for a DataFrame: numeric strings to
    numbers, ISO strings to dates, numbers to text. Raises ``FilterMismatch``
    for values the column cannot be compared with, including a range on a
    column of text.
    """
    if predicate.op == "in" and len(predicate.values) > FILTER_OPTIONS:
        raise FilterMismatch(
            f"takes a selection of at most {FILTER_OPTIONS} values, and {len(predicate.values)} were selected"
        )
    if predicate.op != "in" and kind in ("string", "boolean"):
        raise FilterMismatch(f"holds {'text' if kind == 'string' else 'booleans'}, which have no range")
    values = [_coerce(v, kind) for v in predicate.values]
    try:
        if predicate.op == "in":
            if kind == "integer" and any(isinstance(v, float) for v in values):
                raise FilterMismatch("holds integers, which a fractional value never equals")
            return filter_condition(values)
        start, end = values
        if predicate.closed == "left":
            return filter_condition(Range(start, end, closed="left"))
        return filter_condition((start, end))
    except ValueError as exc:
        raise FilterMismatch(str(exc)) from None


def global_conditions(pipeline: Pipeline, filters: Mapping[str, Any]) -> tuple[list[tuple[str, Any]], dict[str, str]]:
    """
    The conditions the global ``filters`` (``{column: Predicate}``) put on
    ``pipeline``, and why the others could not be applied, by column. A
    filter on a column the pipeline does not have is left out, as is one
    whose values its column cannot be compared with.
    """
    types = pipeline_column_types(pipeline) or {}
    conditions, mismatched = [], {}
    for column, predicate in filters.items():
        if column not in types:
            continue
        try:
            conditions.append((column, global_condition(predicate, types[column])))
        except FilterMismatch as exc:
            mismatched[column] = str(exc)
    return conditions, mismatched


def globally_filtered(pipeline: Any, filters: Mapping[str, Any], limit: int = RESULT_LIMIT) -> Any:
    """
    ``pipeline`` keeping the rows that match the global ``filters``, see
    ``global_conditions``. As for a filter node, on a pipeline read from a
    SQL source the conditions become part of the source's query, rendered
    as quoted literals of its dialect, never by formatting values into SQL.
    """
    if not filters or not isinstance(pipeline, Pipeline):
        return pipeline
    conditions, _ = global_conditions(pipeline, filters)
    return conditioned_pipeline(pipeline, conditions, limit) if conditions else pipeline


def _plain(value: Any) -> Any:
    value = python_value(value)
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value


def _bound(value: Any) -> Any:
    value = python_value(value)
    return value.to_pydatetime() if isinstance(value, pd.Timestamp) else value


def column_values(pipeline: Any, column: str, limit: int = FILTER_OPTIONS, span: bool = False,
                  timeout: float = NODE_TIMEOUT) -> list | tuple | None:
    """
    The distinct values of ``column`` in ``pipeline``, sorted and at most
    ``limit`` of them, or with ``span`` its ``(min, max)``; None when the
    pipeline has no such column or no values. On a SQL source the source
    computes them, with the pipeline's filters and without its row limit,
    so no rows are fetched; otherwise they come from the fetched rows.
    Dates are given in ISO format, as panel-flowdash offers them.
    """
    if not isinstance(pipeline, Pipeline) or column not in pipeline_columns(pipeline):
        return None
    source = pipeline.source
    sql = pipeline_sql(pipeline, limits=False)
    if sql is not None and pipeline.table not in getattr(source, "table_params", {}) and not pipeline.transforms:
        dialect = source.dialect
        if span:
            frame = execute(source, SQLMinMax(columns=[column], identify=True, write=dialect).apply(sql), timeout)
            series = frame.iloc[0] if len(frame) else pd.Series(dtype=object)
        else:
            query = SQLDistinct(columns=[column], write=dialect).apply(sql)
            query = SQLSort(by=[column], identify=True, write=dialect).apply(query)
            # One more, in case NULL takes a place.
            query = SQLLimit(limit=limit + 1, write=dialect).apply(query)
            series = execute(source, query, timeout).iloc[:, 0]
    else:
        series = fetch(pipeline, RESULT_LIMIT, timeout)[column]
    series = series.dropna()
    if series.empty:
        return None
    if span:
        return tuple(_bound(v) for v in (series.min(), series.max()))
    values = list(dict.fromkeys(_plain(v) for v in series))[:limit]
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def _parse_temporal(pipeline: Pipeline, data: Any) -> Any:
    """
    ``data`` with the truncated dates of a metric pipeline as timestamps,
    which an engine without a timestamp type, such as SQLite, returns as text.
    """
    step = next((t for t in pipeline.sql_transforms if isinstance(t, MetricAggregate)), None)
    if step is None or not isinstance(data, pd.DataFrame):
        return data
    text = [c for c in step.temporal if c in data.columns and not pd.api.types.is_datetime64_any_dtype(data[c])]
    if not text:
        return data
    data = data.copy()
    for column in text:
        data[column] = pd.to_datetime(data[column])
    return data


_REFRESHED: weakref.WeakKeyDictionary[Any, weakref.WeakSet] = weakref.WeakKeyDictionary()


def _watch_refresh(editor: Any, node: Any) -> None:
    """
    Run ``node`` again on ``editor``'s refresh. One watcher per editor holds
    its nodes weakly, so removed nodes stop running.
    """
    if "refresh" not in editor.param:
        return
    if editor not in _REFRESHED:
        _REFRESHED[editor] = weakref.WeakSet()
        ref = weakref.ref(editor)
        editor.param.watch(lambda *events: _refresh_dashboard(ref()), "refresh")
    _REFRESHED[editor].add(node)


def _refresh_dashboard(editor: Any) -> None:
    for node in list(_REFRESHED.get(editor, ())):
        node._on_dashboard_refresh()


def forget(pipeline: Any) -> None:
    """Drop the rows ``fetch`` holds for ``pipeline``, so the next fetch runs its query again."""
    if isinstance(pipeline, Pipeline):
        _FETCHED.pop(pipeline, None)
        _PREVIEW_FAILED.pop(pipeline, None)
        forget_distinct(pipeline)


def _compute(pipeline: Pipeline, timeout: float) -> Any:
    # Pipeline offers no public way to compute its rows that raises on failure.
    pool = ThreadPoolExecutor(1)
    future = pool.submit(pipeline._compute_data)
    try:
        return future.result(timeout)
    except FutureTimeout:
        raise TimeoutError(f"Fetching the rows did not finish within {timeout} seconds.") from None
    finally:
        pool.shutdown(wait=False)


def default_table(sql: str) -> str:
    return f"query_{hashlib.sha1(sql.encode()).hexdigest()[:8]}"


# Column names an overview row lists before saying how many more there are.
_OVERVIEW_NAMES = 6
# Tables of a source whose schema a source's overview reads; the others are listed by name only.
_OVERVIEW_TABLES = 20
# Tables whose schema FlowDash's preview worker reads straight away, while the inspector waits for it.
_OVERVIEW_TABLES_INLINE = 5
# Seconds a schema read for a source's overview is remembered, so a changed table shows up again.
_SCHEMA_TTL = 120
# FlowDash's preview worker threads, which run the inspector's and code panel's preview hooks.
_FLOWDASH_PREVIEW_THREAD = "flowdash-preview"
# The schema of each table read for a source's overview, or None if it failed, with when it was
# read, so it is read once in a while even when the source does not store schemas itself.
_SCHEMAS_READ: weakref.WeakKeyDictionary[Source, dict[str, tuple[float, dict | None]]] = weakref.WeakKeyDictionary()
# The schema reads queued or running, and the source nodes to refresh once each is done.
_SCHEMA_PENDING: weakref.WeakKeyDictionary[Source, tuple[Future, weakref.WeakSet]] = weakref.WeakKeyDictionary()


def _table_metadata(source: Source, table: str, cached: Mapping[str, Any]) -> tuple[int | None, list[str] | None]:
    """The row count and column names of ``table`` that are known without a query."""
    schema = cached.get(table)
    if not isinstance(schema, dict):
        schema = _schemas_read(source).get(table)
    if isinstance(schema, dict):
        length = schema.get("__len__")
        rows = length if isinstance(length, int) and not isinstance(length, bool) else None
        return rows, [str(key) for key in schema if key != "__len__"]
    in_memory = source.tables if isinstance(getattr(source, "tables", None), Mapping) else {}
    frame = in_memory.get(table)
    shape, columns = getattr(frame, "shape", None), getattr(frame, "columns", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or columns is None:
        return None, None
    # A lazy frame, e.g. a Dask one, has no row count until it is computed.
    rows = shape[0] if isinstance(shape[0], int) and not isinstance(shape[0], bool) else None
    return rows, [str(column) for column in columns]


def _column_names(names: list[str]) -> str:
    if len(names) <= _OVERVIEW_NAMES:
        return ", ".join(names)
    return f"{', '.join(names[:_OVERVIEW_NAMES])} +{len(names) - _OVERVIEW_NAMES} more"


def _source_overview(source: Source) -> tuple[pd.DataFrame | str, list[str]]:
    """
    An overview of ``source``'s tables from the schemas it stored or the
    frames it holds, without a query, and the tables whose schema it does
    not know yet. Without a known schema for any table the overview is a
    line naming the tables: a one-column frame would be captioned as its
    own shape, e.g. "1 row, 1 column".
    """
    tables = list(listed_tables(source))
    cached = source._get_schema_cache() or {}
    entries = [(table, *_table_metadata(source, table, cached)) for table in tables]
    read = _schemas_read(source)
    missing = [table for table, _, names in entries[:_OVERVIEW_TABLES] if names is None and table not in read]
    if all(names is None for _, _, names in entries):
        count = len(entries)
        listed = ", ".join(tables)
        return (f"{count:,} table{'' if count == 1 else 's'}: {listed}" if entries else "No tables"), missing
    overview = pd.DataFrame({
        "table": [table for table, _, _ in entries],
        "rows": [rows if rows is not None else "" for _, rows, _ in entries],
        "columns": [len(names) if names is not None else "" for _, _, names in entries],
        "column names": [_column_names(names) if names is not None else "" for _, _, names in entries],
    })
    return overview, missing


def _schemas_read(source: Source) -> dict[str, dict | None]:
    """The schemas read for ``source``'s overview that have not expired, None for those that failed."""
    now = time.monotonic()
    with _PREVIEW_LOCK:
        read = _SCHEMAS_READ.get(source, {})
        for table in [table for table, (at, _) in read.items() if now - at > _SCHEMA_TTL]:
            del read[table]
        return {table: schema for table, (_, schema) in read.items()}


def _read_schemas(source: Source, tables: list[str]) -> None:
    """Read the schema of each of ``tables`` and remember it, or that it failed."""
    for table in tables:
        try:
            schema = source.get_schema(table)
        except Exception:
            schema = None
        with _PREVIEW_LOCK:
            _SCHEMAS_READ.setdefault(source, {})[table] = (
                time.monotonic(), schema if isinstance(schema, dict) else None
            )


def _source_summary(node: LumenNode, source: Source) -> pd.DataFrame | str:
    """
    The overview a source node previews: its tables with their row and
    column counts and column names. Reading a schema counts the table's
    rows, so on the event loop the schemas not known yet are read by a
    worker and the node's previews refresh once the batch is done. On
    FlowDash's preview worker, which runs the inspector's Data tab off the
    loop, the first few are read straight away. Anywhere else, e.g. headless
    or an agent's tool thread, only what is known is shown, so nothing is
    queried.
    """
    overview, missing = _source_overview(source)
    if not missing:
        return overview
    if not _blocks_event_loop():
        if not threading.current_thread().name.startswith(_FLOWDASH_PREVIEW_THREAD):
            return overview
        _read_schemas(source, missing[:_OVERVIEW_TABLES_INLINE])
        return _source_overview(source)[0]
    with _PREVIEW_LOCK:
        if (pending := _SCHEMA_PENDING.get(source)) is not None:
            pending[1].add(node)
            return overview
        waiters = weakref.WeakSet([node])
        doc = pn.state.curdoc if getattr(pn.state.curdoc, "session_context", None) else None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def refresh():
            for waiter in list(waiters):
                if waiter._object is source:
                    waiter._preview_rev += 1

        def done(future: Future) -> None:
            with _PREVIEW_LOCK:
                _SCHEMA_PENDING.pop(source, None)
            if not future.cancelled():
                _schedule(refresh, doc, loop)

        future = _preview_pool().submit(_read_schemas, source, missing)
        _SCHEMA_PENDING[source] = (future, waiters)
    future.add_done_callback(done)
    return overview


# The key naming the kind of a ``__flowdash_preview__`` dict, e.g. a chart.
PREVIEW_KIND_KEY = "__flowdash_preview_kind__"
# A card preview fetches one row more than it shows, to tell whether the pipeline has more.
_PREVIEW_FETCH = PREVIEW_ROWS + 1

_PREVIEW_LOCK = threading.RLock()
_PREVIEW_POOL: list[ThreadPoolExecutor] = []
# Why a pipeline's preview could not be fetched, so a failing query is not run on every refresh.
_PREVIEW_FAILED: weakref.WeakKeyDictionary[Pipeline, str] = weakref.WeakKeyDictionary()
# The preview fetches queued or running, and the nodes to refresh once each is done.
_PREVIEW_PENDING: dict[Pipeline, tuple[Future, weakref.WeakSet]] = {}


class _AtLeast(int):
    """A row count only known to be at least this many, written like ``50+``."""

    def __format__(self, spec: str) -> str:
        return f"{int(self):{spec}}+"


class _PreviewFrame(pd.DataFrame):
    """
    The first rows of a pipeline whose ``shape`` gives the pipeline's row
    count, so the caption of a card's table says how many rows it has
    rather than how many were fetched. Anything derived from it is a plain
    DataFrame.
    """

    _metadata = ["_row_count"]

    @property
    def _constructor(self):
        return pd.DataFrame

    @property
    def shape(self) -> tuple[int, int]:
        return (self._row_count, len(self.columns))


def _preview_frame(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if not isinstance(count, _AtLeast) and count == len(frame):
        return frame
    preview = _PreviewFrame(frame)
    preview._row_count = count
    return preview


def _fetched_preview(pipeline: Pipeline) -> pd.DataFrame | None:
    """The first rows of ``pipeline`` and its row count, from the rows fetched the most of."""
    fetched = list(_FETCHED.get(pipeline, {}).items())
    if not fetched:
        return None
    limit, data = max(fetched, key=lambda item: item[0])
    if len(data) < limit:
        count = len(data)
    else:
        count = _AtLeast(PREVIEW_ROWS if limit == _PREVIEW_FETCH else len(data))
    return _preview_frame(data.head(PREVIEW_ROWS), count)


def _blocks_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        doc = pn.state.curdoc
        return doc is not None and doc.session_context is not None
    return True


def _in_served_session() -> bool:
    doc = pn.state.curdoc
    return doc is not None and getattr(doc, "session_context", None) is not None


def _preview_pool() -> ThreadPoolExecutor:
    with _PREVIEW_LOCK:
        if not _PREVIEW_POOL:
            _PREVIEW_POOL.append(ThreadPoolExecutor(2, thread_name_prefix="lumen-preview"))
        return _PREVIEW_POOL[0]


def _fetch_preview(pipeline: Pipeline) -> None:
    try:
        fetch(pipeline, _PREVIEW_FETCH)
    except Exception as exc:
        message = str(exc).strip().splitlines()
        _PREVIEW_FAILED[pipeline] = message[0] if message else type(exc).__name__


def _schedule(callback: Any, doc: Any, loop: asyncio.AbstractEventLoop | None) -> None:
    """Run ``callback`` on the event loop that asked for the preview, if it is still there."""
    if doc is not None:
        try:
            doc.add_next_tick_callback(callback)
            return
        except Exception:
            pass
    if loop is not None and not loop.is_closed():
        try:
            loop.call_soon_threadsafe(callback)
            return
        except RuntimeError:
            pass
    callback()


def _preview_future(pipeline: Pipeline) -> Future | None:
    """The background fetch of ``pipeline``'s preview rows, if one is queued or running."""
    with _PREVIEW_LOCK:
        pending = _PREVIEW_PENDING.get(pipeline)
    return pending[0] if pending is not None else None


def _pipeline_preview(node: LumenNode, pipeline: Pipeline) -> Any:
    """
    The rows a card previews for ``pipeline``, without querying on the
    event loop. Rows already fetched are shown as they are. Otherwise, on
    the event loop, a worker fetches them while the card says so, and the
    node's card refreshes once they are there; elsewhere they are fetched
    straight away. A failure is kept, so the query is not run again until
    the pipeline changes or is forgotten.
    """
    if (rows := _fetched_preview(pipeline)) is not None:
        return rows
    if (failed := _PREVIEW_FAILED.get(pipeline)) is not None:
        return f"No preview: {failed}"
    if not _blocks_event_loop():
        _fetch_preview(pipeline)
        if (failed := _PREVIEW_FAILED.get(pipeline)) is not None:
            return f"No preview: {failed}"
        return _fetched_preview(pipeline)
    with _PREVIEW_LOCK:
        previous = node._preview_of() if node._preview_of is not None else None
        if previous is not None and previous is not pipeline and previous in _PREVIEW_PENDING:
            future, waiters = _PREVIEW_PENDING[previous]
            waiters.discard(node)
            # A slider dragged over a filter builds a pipeline per value; only the last one is wanted.
            if not waiters:
                future.cancel()
        node._preview_of = weakref.ref(pipeline)
        if pipeline in _PREVIEW_PENDING:
            _PREVIEW_PENDING[pipeline][1].add(node)
            return "Fetching rows..."
        waiters = weakref.WeakSet([node])
        pool = _preview_pool()
        doc = pn.state.curdoc if getattr(pn.state.curdoc, "session_context", None) else None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def refresh():
            for waiter in list(waiters):
                if waiter._object is pipeline:
                    waiter._preview_rev += 1

        def done(future: Future) -> None:
            with _PREVIEW_LOCK:
                _PREVIEW_PENDING.pop(pipeline, None)
            if not future.cancelled():
                _schedule(refresh, doc, loop)

        future = pool.submit(_fetch_preview, pipeline)
        _PREVIEW_PENDING[pipeline] = (future, waiters)
    future.add_done_callback(done)
    return "Fetching rows..."


def _preview_kinds() -> set[str]:
    """The preview kinds the installed panel-flowdash draws from a ``__flowdash_preview__`` dict."""
    try:
        from panel_flowdash.ui import node_preview
    except ImportError:
        return set()
    if getattr(node_preview, "PREVIEW_KIND_KEY", None) != PREVIEW_KIND_KEY:
        return set()
    return set(getattr(node_preview, "HEIGHTS", {}))


def _sample_rows(data: Any, limit: int) -> Any:
    """At most ``limit`` evenly spaced rows of ``data``, keeping the first and the last."""
    if data is None or not hasattr(data, "iloc") or len(data) <= limit:
        return data
    step = (len(data) - 1) / (limit - 1)
    return data.iloc[[round(i * step) for i in range(limit)]]


def _chart_summary(spec: dict[str, Any], data: Any) -> str:
    """A line saying what a chart shows, e.g. ``Bar chart of category and total, 4 rows``."""
    mark = spec.get("mark")
    mark = mark.get("type") if isinstance(mark, dict) else mark
    title = f"{str(mark).capitalize()} chart" if isinstance(mark, str) and mark else "Chart"
    encoding = spec.get("encoding") if isinstance(spec.get("encoding"), dict) else {}
    fields = []
    for channel in ("x", "y", "color", "theta"):
        field = encoding.get(channel)
        field = field.get("field") if isinstance(field, dict) else None
        if isinstance(field, str) and field not in fields:
            fields.append(field)
    text = f"{title} of {' and '.join(fields)}" if fields else title
    if data is not None and hasattr(data, "__len__"):
        text += f", {len(data):,} row{'' if len(data) == 1 else 's'}"
    return text


class LumenNode(Viewer):

    _object = param.Parameter(default=None, precedence=-1)

    _error = param.String(default=None, allow_None=True, precedence=-1)

    # Bumped when the rows a card previews arrive, so the card refreshes.
    _preview_rev = param.Integer(default=0, precedence=-1)

    # The dashboard's global filters that reach the node as a view, by column, see GlobalFilterTarget.
    _global_filters = param.Dict(default={}, precedence=-1)

    __abstract = True

    _preview_of: weakref.ref | None = None

    def __init__(self, **params):
        editor = current_dashboard()
        self._editor = weakref.ref(editor) if editor is not None else None
        self._node_id = current_node_id()
        # Taken now, while the editor creating the node is known; it follows later bindings.
        self._context = dashboard_context()
        super().__init__(**params)
        if editor is not None and self._reads_source:
            _watch_refresh(editor, self)

    @property
    def editor(self) -> Any:
        return self._editor() if self._editor is not None else None

    def node_id(self) -> str | None:
        """The id of the node the component was created for."""
        if self._node_id is None and self.editor is not None:
            self._node_id = self.editor.node_id(self)
        return self._node_id

    # Whether the node reads a source directly, and so runs again on the dashboard's refresh.
    _reads_source: ClassVar[bool] = False

    def _read_source(self) -> Any:
        """The source the node reads, if it reads one directly."""
        return getattr(self, "source", None)

    def _on_dashboard_refresh(self):
        """
        Run the node again on the dashboard's refresh with its source's cache
        cleared. Only nodes that read a source do this: each builds a new
        pipeline, which flows downstream and is fetched afresh, so one
        refresh runs each query once.
        """
        source = self._read_source()
        inputs = source if isinstance(source, (list, tuple)) else [source]
        # A SQL node's result changes when that node runs again, which runs this node then.
        if any(is_derived(source) for source in inputs):
            return
        for source in inputs:
            if isinstance(source, Source):
                source.clear_cache()
            if isinstance(source, BaseSQLSource):
                forget_probes(source)
        self._refresh()

    def _resolve(self) -> Any:
        raise NotImplementedError

    def _refresh(self, *events):
        try:
            self._error = None
            self._object = self._resolve()
        except Exception as exc:
            self._object = None
            self._error = self._error_text(exc)

    def _error_text(self, exc: Exception) -> str:
        return str(exc)

    def _value(self):
        if self._error is not None:
            return param.ReactiveError(ValueError(self._error))
        return self._object

    @param.depends("_object", "_error", "_global_filters")
    def _render(self):
        if self._error:
            return pn.pane.Alert(self._error, alert_type="danger")
        return self._body()

    def _body(self):
        return pn.pane.Str(str(self._object))

    def __flowdash_preview__(self, output: str | None = None) -> Any:
        if self._error is not None or self._object is None:
            return None
        if isinstance(self._object, Pipeline):
            return _pipeline_preview(self, self._object)
        return self._object

    def __panel__(self):
        return pn.panel(self._render)


class GlobalFilterTarget:
    """
    A node that applies FlowDash's global filters to what it shows, by
    adding them to the query of its pipeline (panel-flowdash's
    ``FilterTarget`` protocol). Only what the node shows is filtered, never
    its outputs, so nodes downstream still see every row, as for any
    component FlowDash filters. An older panel-flowdash never calls these
    methods, and the node shows every row.
    """

    # The parameters whose change can change the columns of ``_filter_pipeline``.
    _filter_inputs: ClassVar[tuple[str, ...]] = ("_object",)

    def __init__(self, **params):
        super().__init__(**params)
        self._declared = self.__flowdash_columns__()
        self.param.watch(self._columns_changed, list(self._filter_inputs))

    def _filter_pipeline(self) -> Any:
        """The pipeline whose rows the node shows, before the global filters."""
        return self._object

    def _shown(self, pipeline: Any) -> Any:
        """``pipeline`` with the global filters that reach the node."""
        return globally_filtered(pipeline, self._global_filters, row_ceiling(self.editor))

    def __flowdash_columns__(self) -> list[str] | None:
        pipeline = self._filter_pipeline()
        if not isinstance(pipeline, Pipeline):
            return None
        return pipeline_columns(pipeline)

    def __flowdash_apply_filters__(self, filters: Mapping[str, Any]) -> dict[str, str]:
        """Apply ``filters`` to what the node shows; return why it skips the others, by column."""
        filters = dict(filters)
        if filters != self._global_filters:
            self._global_filters = filters
        pipeline = self._filter_pipeline()
        if not filters or not isinstance(pipeline, Pipeline):
            return {}
        return global_conditions(pipeline, filters)[1]

    def __flowdash_column_values__(self, column: str) -> list | tuple | None:
        span = _filter_widget(self.editor, column) == "range"
        return column_values(self._filter_pipeline(), column, FILTER_OPTIONS, span=span)

    def _columns_changed(self, *events):
        """
        Have FlowDash hand the node its filters again when its columns
        change without an input changing, e.g. when a query run off the
        event loop returns.
        """
        columns = self.__flowdash_columns__()
        if columns == self._declared:
            return
        self._declared = columns
        if request_filters is not None:
            request_filters(self)


def _filter_widget(editor: Any, column: str) -> str | None:
    """The widget the editor's global filter on ``column`` names, if it names one."""
    try:
        filters = editor.filters
    except Exception:
        return None
    entry = next((f for f in filters or () if isinstance(f, dict) and f.get("column") == column), None)
    return entry.get("widget") if entry else None


def _track_ref(node: LumenNode) -> None:
    """Refresh ``node``, which names a host source by ref, when ``bind_source`` binds that ref."""
    if node.editor is not None:
        _SOURCE_NODES.setdefault(node.editor, weakref.WeakSet()).add(node)


def _ref_source(ref: str, context: Mapping[str, Any]) -> Source | None:
    """The host source ``ref`` names, None for no ref; an unknown ref raises with the refs there are."""
    if not ref:
        return None
    source = lookup_source(ref, context)
    if source is None:
        raise ValueError(missing_source(ref, host_sources(context)))
    return source


@register(
    component=True, page=False, title="Lumen Source", icon="storage",
    description=(
        "An app data source, named by its ref, for SQL nodes or tables sharing one connection; a table "
        "node can name the ref itself."
    ),
    config=["ref"],
    check_config=check_source_config,
    contract=SourceContract(),
    config_choices=source_choices,
    provides=[{"key": "source", "type": SOURCE_TYPE}],
    **palette_presets(source_presets),
)
class SourceNode(LumenNode):

    ref = param.String(default="", doc="Name of a source the host provides.")

    def __init__(self, **params):
        super().__init__(**params)
        _track_ref(self)

    @param.depends("ref", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> Source | None:
        return _ref_source(self.ref, self._context)

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def source(self):
        return self._value()

    def _body(self):
        source = self._object
        if source is None:
            return pn.pane.Markdown("No source")
        return pn.pane.Markdown(f"{self.ref}: {', '.join(listed_tables(source))}")

    def __flowdash_preview__(self, output: str | None = None) -> pd.DataFrame | str | None:
        if self._error is not None or self._object is None:
            return None
        return _source_summary(self, self._object)

    def __flowdash_definition__(self) -> tuple[str, str] | None:
        """The source's spec as YAML, read only in FlowDash's code panel, with its credentials hidden."""
        if self._error is not None or self._object is None:
            return None
        spec = self._object.to_spec()
        spec = redact_spec({"type": spec.pop("type", None), **spec} if "type" in spec else spec)
        return "yaml", yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)


def single_number(data: Any) -> float | int | None:
    """
    The one cell of a result of one row and one column as a Python number,
    None if it is NULL, raising ``ValueError`` for a result of another shape
    or a cell that is not a number.
    """
    if data is None:
        return None
    rows, columns = data.shape
    if (rows, columns) != (1, 1):
        raise ValueError(
            f"The value output needs a result of one row and one column, e.g. SELECT SUM(x) FROM t; "
            f"this query returns {rows:,} row{'' if rows == 1 else 's'} and "
            f"{columns:,} column{'' if columns == 1 else 's'}."
        )
    value = python_value(data.iat[0, 0])
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"The value output needs a number; the query's {data.columns[0]!r} is "
            f"{type(value).__name__} {value!r}."
        )
    return value


def sql_completions(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any] | None:
    """
    The table and column names of the wired source, completed in the SQL
    field while typing. Columns come from the schemas the source stored, or
    are read once per table otherwise.
    """
    try:
        source = merge(inputs.get("source"))
    except ValueError:
        return None
    if Choices is None or not isinstance(source, Source) or is_failure(source):
        return None
    tables = listed_tables(source)
    stored = source._get_schema_cache() or {}
    types: dict[str, str] = dict.fromkeys(tables, "table")
    words = list(tables)
    for table in tables:
        schema = stored.get(table)
        if not isinstance(schema, dict):
            try:
                schema = source.get_schema(table)
            except Exception:
                continue
        for column, spec in schema.items():
            if column == "__len__" or column in types:
                continue
            words.append(column)
            types[column] = spec.get("type", "") if isinstance(spec, dict) else ""
    return {"sql": Choices(words, restrict=False, types=types)}


@register(
    component=True, page=False, title="Lumen SQL", icon="code",
    description=(
        "A read-only SQL query run by the source. Outputs the result as a pipeline, as a DataFrame, as "
        "`result` (the source plus the result as table `table`, for another SQL node) and, for one row "
        "of one number such as SELECT SUM(x) FROM t, as `value` for a KPI. Connect several results of "
        "one source to `source` to join them."
    ),
    config=["sql", "table", "limit"],
    check_config=check_sql_config,
    requires=[{"key": "source", "type": SOURCE_TYPE, "required": True, "multiple": True}],
    provides=[
        {"key": "pipeline", "type": PIPELINE_TYPE},
        {"key": "data", "type": "DataFrame"},
        {"key": "value", "type": "Number"},
        {"key": "result", "type": SOURCE_TYPE},
    ],
    config_completions=sql_completions,
)
class SQLNode(GlobalFilterTarget, LumenNode):

    _reads_source = True

    source = param.Parameter(doc="""
        The upstream source, or several sources or earlier results of one
        source, whose result tables the query can join.""")

    sql = param.String(
        default="",
        label="SQL",
        doc="One read-only SELECT or WITH query.",
        metadata={"flowdash": {"widget": "code", "language": "sql"}},
    )

    table = param.String(default="", doc="Name of the result table; derived from the SQL if empty.")

    limit = param.Integer(default=None, bounds=(1, None), allow_None=True, doc="""
        Rows of the result kept in memory, at most the host's row ceiling,
        which is also the default.""")

    # Bumped per run, so the result of a query superseded while it ran is dropped.
    _run = 0

    @param.depends("source", "sql", "table", "limit", watch=True, on_init=True)
    def _refresh(self, *events):
        self._run += 1
        # Headless sessions read a node's output right after changing it, so only a served one waits.
        if self._input is None or not self.sql.strip() or not _in_served_session() or self._primed():
            super()._refresh()
            return
        # The previous error describes a query that is being replaced.
        self._error = None
        run, doc = self._run, pn.state.curdoc
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def work():
            try:
                return self._resolve(), None
            except Exception as exc:
                return None, self._error_text(exc)

        def apply(result):
            if run != self._run:
                return
            self._object, self._error = result if result[1] is None else (None, result[1])

        future = _preview_pool().submit(work)
        future.add_done_callback(lambda f: f.cancelled() or _schedule(lambda: apply(f.result()), doc, loop))

    @property
    def _input(self) -> Any:
        """The node's inputs merged into one source, or None if they cannot be (``_resolve`` says why)."""
        inputs = list(self.source) if isinstance(self.source, (list, tuple)) else [self.source]
        cached, merged = getattr(self, "_merged", (None, None))
        if cached is None or len(cached) != len(inputs) or any(a is not b for a, b in zip(cached, inputs, strict=True)):
            try:
                merged = merge(self.source)
            except ValueError:
                merged = None
            self._merged = (inputs, merged)
        return merged

    def _source_edge(self, index: int) -> str | None:
        """The output, as ``node.port``, connected to the node's ``index``-th source input, if the editor tells."""
        try:
            from panel_flowdash.agent.yaml_format import own_name

            node = self.node_id()
            edges = [e for e in self.editor._dataflow_graph.edges if e["target"] == node and e["target_port"] == "source"]
            return f"{own_name(edges[index]['source'])}.{edges[index]['source_port']}"
        except Exception:
            return None

    def _primed(self) -> bool:
        """Whether the caller already ran the query, so resolving it is quick and its result is read at once."""
        if not isinstance(self._input, BaseSQLSource):
            return True
        limit = effective_limit(self.editor, self.limit)
        try:
            root, statement = expand(self.sql, self._input)
            return (id(root), result_sql(root, statement, limit)) in _RESULTS
        except Exception:
            return True

    def _resolve(self) -> Pipeline | None:
        if not self.sql.strip():
            return None
        inputs = self.source if isinstance(self.source, (list, tuple)) else [self.source]
        for index, value in enumerate(inputs):
            if value is not None and not is_failure(value) and not isinstance(value, BaseSQLSource):
                raise ValueError(not_a_source(value, self._source_edge(index)))
        merge(self.source)
        if self._input is None:
            return None
        limit = effective_limit(self.editor, self.limit)
        return compile_sql(self._input, self.sql, self.table or default_table(self.sql), limit)

    def _error_text(self, exc: Exception) -> str:
        """The error as one readable message, with the line and column of the query it concerns."""
        diagnostic = getattr(exc, "diagnostic", None)
        if diagnostic is None and isinstance(self._input, BaseSQLSource) and not isinstance(
            exc, (ValueError, QueryTimeoutError)
        ):
            limit = effective_limit(self.editor, self.limit)
            try:
                root, statement = expand(self.sql, self._input)
            except Exception:
                root, statement = self._input, self.sql
            diagnostic = _engine_error(
                exc, result_sql(root, statement, limit), self.sql, dialect=_read(self._input.dialect)
            )
        text = ANSI_ESCAPE.sub("", str(diagnostic if diagnostic is not None else exc)).strip()
        line, column = getattr(diagnostic, "line", None), getattr(diagnostic, "column", None)
        if line and column:
            return f"Line {line}, column {column}: {text}"
        return f"Line {line}: {text}" if line else text

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def pipeline(self):
        return self._value()

    @param.output(param.DataFrame)
    @param.depends("_object", "_error")
    def data(self):
        pipeline = self._value()
        return pipeline if pipeline is None or is_failure(pipeline) else pipeline.data

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def value(self):
        pipeline = self._value()
        if pipeline is None or is_failure(pipeline):
            return pipeline
        # Every SQL node has this output, so a result it does not fit leaves it empty rather than failing the node.
        try:
            return single_number(pipeline.data)
        except ValueError:
            return None

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def result(self):
        pipeline = self._value()
        return pipeline if pipeline is None or is_failure(pipeline) else result_source(pipeline)

    def _body(self):
        pipeline = self._object
        if pipeline is None:
            return pn.pane.Markdown("Waiting for a source and a query")
        data = fetch(self._shown(pipeline), row_ceiling(self.editor)) if self._global_filters else pipeline.data
        return pn.widgets.Tabulator(data, disabled=True, sizing_mode="stretch_both")

    def __flowdash_preview__(self, output: str | None = None) -> Any:
        if self._error is not None or self._object is None:
            return None
        data = self._object.data
        if output == "value":
            try:
                return single_number(data)
            except ValueError as exc:
                return f"No value: {exc}"
        if output == "pipeline" or data is None:
            return _pipeline_preview(self, self._object)
        return data

    def __flowdash_provenance__(self, via: Sequence[Mapping[str, Any]] = ()) -> dict[str, str] | None:
        """
        The "Ad hoc" label of the views reading the node, its number or
        its rows, saying how its query computes them and that no metric was
        used; through an edge that aggregates the rows, how it does so too.
        """
        if not self.sql.strip() or self._error is not None or self._object is None:
            return None
        from .save_metric import (
            aggregating_step, describe_adhoc_reading, describe_query,
        )

        detail = describe_query(self.sql, _read(getattr(self._input, "dialect", None)))
        if any(step.get("op") == "aggregate" for edge in via for step in edge.get("transform") or []):
            data = self._object.data
            step = aggregating_step(via, len(data) if isinstance(data, pd.DataFrame) else None)
            if step is not None:
                detail = describe_adhoc_reading(*step, detail)
        return {"label": "Ad hoc", "detail": detail, "kind": "ad hoc"}


class PipelineNode(GlobalFilterTarget, LumenNode):
    """A node whose output is a lazy pipeline, previewed only when its view is shown."""

    __abstract = True

    _waiting = ""

    def _body(self):
        pipeline = self._object
        if pipeline is None:
            return pn.pane.Markdown(self._waiting)
        preview = fetch(self._shown(pipeline), PREVIEW_ROWS)
        return pn.widgets.Tabulator(preview, disabled=True, sizing_mode="stretch_both")

    def __panel__(self):
        return pn.param.ParamMethod(self._render, lazy=True, sizing_mode="stretch_both")


@register(
    component=True, page=False, title="Lumen Table", icon="table_chart",
    description=(
        "A table of an app source, picked by the source's ref and the table's name. Its pipeline "
        "fetches nothing itself: filters and the limit join the source's query. A connected source node "
        "is read instead of ref, to share one."
    ),
    config=["ref", "table", "limit"],
    check_config=check_table_config,
    contract=SourceContract(),
    config_choices=table_choices,
    requires=[{"key": "source", "type": SOURCE_TYPE, "required": False}],
    provides=[{"key": "pipeline", "type": PIPELINE_TYPE}],
    **palette_presets(table_presets),
)
class TableNode(PipelineNode):

    _reads_source = True

    source = param.Parameter(doc="A source node's output, read instead of ref while connected.")

    ref = param.String(default="", doc="Name of a source the host provides, read when no source is connected.")

    table = param.String(default="", doc="Name of a table of the source.")

    limit = param.Integer(default=None, bounds=(1, None), allow_None=True, doc="""
        Rows the source returns at most, at most the host's row ceiling,
        which is also the default. It caps the rows after the filters of
        filter nodes downstream, which become part of the same query, so
        it is not a sample the filters then narrow.""")

    _waiting = "Pick a source and a table"

    def __init__(self, **params):
        super().__init__(**params)
        _track_ref(self)

    @param.depends("source", "ref", "table", "limit", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _read_source(self) -> Any:
        if self.source is not None:
            return self.source
        try:
            return _ref_source(self.ref, self._context)
        except ValueError:
            return None

    def _resolve(self) -> Pipeline | None:
        source = self.source if self.source is not None else _ref_source(self.ref, self._context)
        if source is None or not self.table:
            return None
        return table_pipeline(source, self.table, effective_limit(self.editor, self.limit))

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def pipeline(self):
        return self._value()


@register(
    component=True, page=False, title="Lumen Filter", icon="filter_alt",
    description=(
        "Keeps a pipeline's rows whose field matches value: one value, a list or a (start, end) range; "
        "empty keeps all. Or the selection input takes a chart's selection, field optional. A SQL "
        "source filters in its query, before the limit."
    ),
    config=["field"],
    check_config=check_filter_config,
    config_choices=filter_choices,
    requires=[
        {"key": "pipeline", "type": PIPELINE_TYPE, "required": True},
        {"key": "value", "required": False, "blocking": False},
        {"key": "selection", "type": SELECTION_TYPE, "required": False, "blocking": False},
    ],
    provides=[{"key": "filtered", "type": PIPELINE_TYPE}],
)
class FilterNode(PipelineNode):

    pipeline = param.Parameter(doc="The upstream pipeline.")

    value = param.Parameter(doc="The value to keep, e.g. a widget's selection.")

    selection = param.ClassSelector(class_=Selection, default=None, allow_None=True, doc="""
        A plot selection; each of its predicates becomes a condition.""")

    field = param.String(
        default="", doc="The column to filter on.",
        metadata={"flowdash": {"choices": "columns", "of": "pipeline"}},
    )

    _waiting = "Waiting for a pipeline and a field"

    @param.depends("pipeline", "value", "selection", "field", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> Pipeline | None:
        if self.pipeline is None or not (self.field or self.selection is not None):
            return None
        limit = row_ceiling(self.editor)
        pipeline = self.pipeline
        if self.field:
            pipeline = filtered_pipeline(pipeline, self.field, self.value, limit)
        if self.selection is not None and not self.selection.empty:
            pipeline = conditioned_pipeline(pipeline, selection_conditions(self.selection), limit)
        return pipeline

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def filtered(self):
        return self._value()


def _read_tables(sql: str, source: BaseSQLSource) -> dict[str, list[str]]:
    """
    The columns of each of ``source``'s tables in the FROM and JOINs of
    ``sql``'s outermost query, which a filter on a SQL metric may filter
    inside it; a table a subquery or a CTE reads, or a CTE of the table's
    name, is left alone, so a filter never changes, say, a total a share
    is taken of.
    """
    parsed = sqlglot.parse_one(sql, read=None if source.dialect == "any" else source.dialect)
    ctes = {cte.alias_or_name.lower() for cte in parsed.find_all(sqlglot.exp.CTE)}
    tables = {table.lower(): table for table in source.get_tables()}
    read: dict[str, list[str]] = {}
    for node in MetricPreFilter.top_level_tables(parsed):
        name = node.name
        if not isinstance(node.this, sqlglot.exp.Identifier) or name.lower() in ctes or name.lower() not in tables:
            continue
        table = tables[name.lower()]
        if table not in read:
            read[name] = [str(c) for c in source.get_schema(table) if c != "__len__"]
    return read


def _metric_schema(metric: Metric, query: Query, pipeline: Pipeline, timeout: float) -> dict[str, Any]:
    """
    The schema of a metric pipeline's columns, from the engine's answer to
    its query without rows, with each measure's and dimension's ``d3_format``
    and ``description``.
    """
    sql = pipeline_sql(pipeline)
    frame = execute(pipeline.source, SQLLimit(limit=0, write=pipeline.source.dialect).apply(sql), timeout)
    schema = get_dataframe_schema(_parse_temporal(pipeline, frame))["items"]["properties"]
    for name in schema:
        field = metric.measures.get(name) or metric.dimensions.get(name)
        if field is None:
            continue
        if field.format:
            schema[name]["d3_format"] = field.format
        if field.description:
            schema[name]["description"] = field.description
    return schema


def metric_pipeline(metric: Metric, source: Source, query: Query, limit: int = RESULT_LIMIT,
                    timeout: float = NODE_TIMEOUT) -> Pipeline:
    """
    A pipeline of ``metric`` for ``query`` on ``source``, capped at
    ``limit`` rows, that fetches nothing until its data is read; only the
    engine's answer to its query without rows is asked for, for the
    columns. The compiled query passes ``check_query``. A filter node
    downstream filters a structured metric's rows before they are
    aggregated, and a SQL metric's tables inside its query, see
    ``conditioned_pipeline``. Raises ``MetricError`` or ``ValueError``.
    """
    if is_failure(source):
        raise ValueError(f"The source failed: {failure_text(source)}")
    if not isinstance(source, BaseSQLSource):
        raise MetricError(f"Metric {metric.name!r} needs a SQL source; {type(source).__name__} does not run SQL.")
    query, values = prepare(metric, query)
    dialect = source.dialect
    read = None if dialect == "any" else dialect
    if metric.form == "structured":
        if metric.table not in source.get_tables():
            raise MetricError(f"The source {metric.source!r} has no table {metric.table!r}.")
        columns = {str(c): spec for c, spec in source.get_schema(metric.table).items() if c != "__len__"}
        values = coerce_dimension_values(metric, values, columns)
        if query.limit is not None:
            query.limit = min(query.limit, limit)
        transform = aggregate(metric, query, values, dialect)
        transform.columns = list(columns)
        check_query(transform.apply(table_sql(metric.table, read)), source)
        pipeline = Pipeline(
            source=source, table=metric.table, schema={}, name=metric.name,
            sql_transforms=[transform, SQLLimit(limit=limit, write=dialect)],
        )
    else:
        sql = bind_sql(metric, values, dialect)
        check_query(sql, source)
        table = metric.name
        while table in source.get_tables():
            table = f"{table}_result"
        kwargs = {"materialize": False} if isinstance(source, DuckDBSource) else {}
        query_source = source.create_sql_expr_source({table: sql}, **kwargs)
        prefilter = MetricPreFilter(
            conditions=[], tables=_read_tables(sql, source), write=dialect,
            boolean_literals=dialect not in NO_BOOLEAN_LITERALS,
        )
        cap = limit if query.limit is None else min(query.limit, limit)
        pipeline = Pipeline(
            source=query_source, table=table, schema={}, name=metric.name,
            sql_transforms=[prefilter, SQLLimit(limit=cap, write=dialect)],
        )
    pipeline.schema = _metric_schema(metric, query, pipeline, timeout)
    return pipeline


def metric_query(config: Mapping[str, Any], parameters: Any = None) -> Query:
    """The ``Query`` a ``Lumen/metric`` node's config and ``parameters`` input ask for."""
    return Query(
        measures=list(dict.fromkeys(config.get("measures") or [])),
        dimensions=list(dict.fromkeys(config.get("dimensions") or [])),
        grain=config.get("grain") or None, order_by=list(dict.fromkeys(config.get("order_by") or [])),
        limit=config.get("limit"), parameters=merged_parameters(parameters),
    )


def merged_parameters(parameters: Any) -> dict[str, Any]:
    """
    The values a ``Lumen/metric`` node's ``parameters`` input holds: one
    mapping, or the mappings of the nodes wired to it, later ones winning.
    """
    if parameters is None:
        return {}
    items = [parameters] if isinstance(parameters, Mapping) else parameters
    if not isinstance(items, (list, tuple)) or not all(isinstance(item, Mapping) for item in items if item is not None):
        raise MetricError(
            "The parameters input takes a mapping of parameter names to values, e.g. {\"n\": 5}, or the "
            "outputs of Lumen Parameter nodes; wire a widget to a Lumen Parameter node first.", "parameters",
        )
    merged: dict[str, Any] = {}
    for item in items:
        merged.update(item or {})
    return merged


def check_metric_config(config: dict[str, Any], inputs: dict[str, Any]) -> Any:
    """
    Check a ``Lumen/metric`` config without running its query: the metric
    must exist, the picked measures, dimensions, grain and order belong to
    it and the values of the parameters input match their types, bounds and
    options.
    """
    name = config.get("metric")
    if not name:
        return None
    metrics = host_metrics()
    metric = metrics.get(name)
    if metric is None:
        return CheckDiagnostic(missing_metric(name, metrics), field="metric")
    try:
        prepare(metric, metric_query(config, inputs.get("parameters")))
    except MetricError as exc:
        return CheckDiagnostic(str(exc), field=exc.field if exc.field in config else None)
    wired = inputs.get("parameters")
    if isinstance(wired, (list, tuple)):
        names = [name for item in wired if isinstance(item, Mapping) for name in item]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            return _unverified(
                f"More than one Lumen Parameter node sets {', '.join(map(repr, twice))}; the one wired last wins."
            )
    return None


@register(
    component=True, page=False, title="Lumen Metric", icon="functions", tags=["transform"],
    description=(
        "An app metric, a named query over a source: pick measures, dimensions and a time grain; "
        "parameters takes parameter values and dimension filters, e.g. {\"n\": 5}. A filter downstream "
        "applies before aggregation."
    ),
    config=["metric", "measures", "dimensions", "grain", "order_by", "limit"],
    check_config=check_metric_config,
    config_choices=metric_choices,
    requires=[{"key": "parameters", "required": False, "blocking": False, "multiple": True}],
    provides=[{"key": "pipeline", "type": PIPELINE_TYPE}, {"key": "data", "type": "DataFrame"}],
    **palette_presets(metric_presets),
)
class MetricNode(PipelineNode):

    _reads_source = True

    metric = param.String(default="", doc="Name of a metric the app defines.")

    measures = param.List(default=[], item_type=str, doc="The measures to compute.")

    dimensions = param.List(default=[], item_type=str, doc="The dimensions to group by.")

    grain = param.String(default="", doc="""
        What to truncate the time dimensions among the dimensions to, e.g.
        ``year``; empty keeps their values.""")

    order_by = param.List(default=[], item_type=str, doc="""
        Columns to sort by; a leading ``-`` sorts one descending, e.g.
        ``["-capacity_mw"]``.""")

    limit = param.Integer(default=None, bounds=(1, None), allow_None=True, doc="""
        Rows the metric returns at most, e.g. the top 10 with ``order_by``,
        at most the host's row ceiling, which is also the default.""")

    parameters = param.Parameter(default=None, doc="""
        Values of the metric's parameters by name, over their defaults, and
        of dimensions to filter by, e.g. ``{"n": 5, "state": ["TX", "IA"]}``:
        a mapping, or the outputs of the Lumen Parameter nodes wired to it.
        A value of None takes the default.""")

    _waiting = "Pick a metric"

    _source: Source | None = None

    @param.depends("metric", "measures", "dimensions", "grain", "order_by", "limit", "parameters",
                   watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    @property
    def source(self) -> Source | None:
        return self._source

    def _config(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in ("measures", "dimensions", "grain", "order_by", "limit")}

    def _resolve(self) -> Pipeline | None:
        self._source = None
        if not self.metric:
            return None
        metrics = host_metrics(self._context)
        metric = metrics.get(self.metric)
        if metric is None:
            raise ValueError(missing_metric(self.metric, metrics))
        source = lookup_source(metric.source, self._context)
        if source is None:
            raise ValueError(missing_source(metric.source, host_sources(self._context)))
        self._source = source
        query = metric_query(self._config(), self.parameters)
        return metric_pipeline(metric, source, query, row_ceiling(self.editor))

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def pipeline(self):
        return self._value()

    @param.output(param.DataFrame)
    @param.depends("_object", "_error")
    def data(self):
        pipeline = self._value()
        return pipeline if pipeline is None or is_failure(pipeline) else fetch(pipeline, row_ceiling(self.editor))

    def __flowdash_provenance__(self, via: Sequence[Mapping[str, Any]] = ()) -> dict[str, str] | None:
        """
        The "Metric: <name>" label of the views reading the node, with the
        metric's description. A view reading it through an edge that
        aggregates its rows into another number, e.g. the mean of per-state
        sums or the last of several groups, is labelled "Derived" with the
        aggregation instead, and a metric the project no longer defines
        says so.
        """
        if not self.metric:
            return None
        metric = host_metrics(self._context).get(self.metric)
        if metric is None:
            return {"label": f"Missing metric: {self.metric}", "kind": "missing", "icon": "error_outline",
                    "detail": f"The project no longer defines the {self.metric} metric, so this number cannot "
                              "be computed. Pick another metric or define it again in Admin, Metrics."}
        from .save_metric import (
            aggregating_step, derived_label, describe_reading,
        )

        step = aggregating_step(via, self._rows(metric))
        if step is not None:
            how, column = step
            return {"label": derived_label(how, self.metric), "kind": "derived",
                    "detail": describe_reading(how, column, self.metric)}
        detail = metric.description or f"The {self.metric} metric the project defines."
        return {"label": f"Metric: {self.metric}", "detail": detail, "kind": "metric"}

    def _rows(self, metric: Metric) -> int | None:
        """How many rows the node gives: as fetched, else one for a structured metric without dimensions."""
        fetched = _FETCHED.get(self._object) if isinstance(self._object, Pipeline) else None
        if fetched:
            return len(next(iter(fetched.values())))
        if metric.form == "structured" and not self.dimensions:
            return 1
        return None


def check_parameter_config(config: dict[str, Any], inputs: dict[str, Any]) -> Any:
    """Check a ``Lumen/parameter`` config: the name must be one some metric the host defines takes."""
    name = config.get("parameter")
    names = metric_parameter_names(host_metrics())
    if not name or not names or name in names:
        return None
    return CheckDiagnostic(
        f"No metric takes a parameter {name!r}; the metrics take: {', '.join(names)}.", field="parameter",
    )


def metric_parameter_names(metrics: Mapping[str, Metric]) -> list[str]:
    """What the metrics' ``parameters`` inputs take, by name: parameters, column dimensions, ``limit`` and ``order_by``."""
    return list(dict.fromkeys(
        name for metric in metrics.values() for name in [*metric.parameters, *metric._filterable(), *metric.knobs()]
    ))


@register(
    component=True, page=False, title="Lumen Parameter", icon="tune", tags=["transform"],
    description=(
        "Names a value as a metric parameter, e.g. a slider's as n, for Lumen Metric parameters inputs; "
        "empty takes the default."
    ),
    config=["parameter"],
    check_config=check_parameter_config,
    config_choices=parameter_choices,
    requires=[{"key": "value", "required": False, "blocking": False}],
    provides=[{"key": "parameters"}],
)
class ParameterNode(LumenNode):

    parameter = param.String(default="", doc="The name of the metric parameter or dimension the value is for.")

    value = param.Parameter(default=None, doc="The value, e.g. a widget's.")

    @param.depends("parameter", "value", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> dict[str, Any] | None:
        if not self.parameter:
            return None
        value = self.value
        if is_failure(value):
            raise ValueError(f"The input failed: {failure_text(value)}")
        return {self.parameter: python_value(value)}

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def parameters(self):
        return self._value()

    def _body(self):
        if self._object is None:
            return pn.pane.Markdown("Pick a parameter")
        return pn.pane.Markdown(", ".join(f"`{k}` = {v!r}" for k, v in self._object.items()))


@register(
    component=True, page=False, title="Lumen Data", icon="table_rows",
    description=(
        "A pipeline's rows as a DataFrame, up to the host's row ceiling."
    ),
    requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
    provides=[{"key": "data", "type": "DataFrame"}],
)
class DataNode(GlobalFilterTarget, LumenNode):

    pipeline = param.Parameter(doc="The upstream pipeline.")

    _filter_inputs = ("pipeline",)

    def _filter_pipeline(self) -> Any:
        return self.pipeline

    @param.depends("pipeline", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self):
        if self.pipeline is None:
            return None
        if is_failure(self.pipeline):
            raise ValueError(f"The input failed: {failure_text(self.pipeline)}")
        return fetch(self.pipeline, row_ceiling(self.editor))

    @param.output(param.DataFrame)
    @param.depends("_object", "_error")
    def data(self):
        return self._value()

    def _body(self):
        if self._object is None:
            return pn.pane.Markdown("Waiting for a pipeline")
        data = fetch(self._shown(self.pipeline), row_ceiling(self.editor)) if self._global_filters else self._object
        return pn.widgets.Tabulator(data, disabled=True, sizing_mode="stretch_both")


_FIXED_AUTOSIZE = ("pad", "none")

_VIEW_SIZES = {
    "width": ("width", "continuousWidth", "discreteWidth"),
    "height": ("height", "continuousHeight", "discreteHeight"),
}


def responsive_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    ``spec`` with ``"container"`` for the width and height it leaves out, so
    the chart fills its tile and follows resizes. A size the spec sets, at
    the top level or in ``config.view``, is kept, as is a spec whose
    ``autosize`` is ``"pad"`` or ``"none"``. A compound chart, which
    Vega-Lite cannot size to a container, is returned as it is.
    """
    from ..ai.utils import is_compound_vegalite

    autosize = spec.get("autosize")
    if isinstance(autosize, dict):
        autosize = autosize.get("type")
    if autosize in _FIXED_AUTOSIZE or is_compound_vegalite(spec):
        return spec
    config = spec.get("config")
    view = config.get("view") if isinstance(config, dict) else None
    view = view if isinstance(view, dict) else {}
    defaults = {
        dimension: "container" for dimension, keys in _VIEW_SIZES.items()
        if not any(key in view for key in keys)
    }
    return {**defaults, **spec}


def spec_param(doc: str, settings: list | None = None, **params: Any) -> param.Dict:
    """
    The ``spec`` param of a view node: a dict FlowDash edits as JSON, with
    the curated ``settings`` (``panel_flowdash.SpecSetting``) its form shows
    as another view of the same spec. Without settings the spec is edited as
    JSON alone. It is synced back from the node, so a default spec the node
    starts from is saved and undone like any other edit.
    """
    metadata = {"flowdash": {"widget": "spec", "settings": list(settings or []), "two_way": True}}
    return param.Dict(default={}, doc=doc, metadata=metadata, **params)


def register_view(
    *, config: list[str] | None = None, provides: list[dict[str, Any]] | None = None,
    check_config: Any = None, tags: list[str] | None = None, **options: Any,
):
    """
    Register a ``LumenViewNode`` subclass as a FlowDash component that reads
    a required ``pipeline`` input and is configured by its ``spec`` and the
    other ``config`` params. It is tagged a view, so its tile shows even
    when it has outputs, e.g. a selection. Without ``check_config`` the spec is checked by
    the class's ``check_spec`` and ``spec_warnings`` and its declared
    settings, see ``check_view_spec``.
    """
    def decorator(cls: type[LumenViewNode]) -> type[LumenViewNode]:
        check = check_config if check_config is not None else _view_check(cls)
        return register(
            component=True, page=False, tags=["view", *(tags or [])], config=["spec", *(config or [])],
            check_config=check,
            requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
            provides=list(provides or []), **options,
        )(cls)
    return decorator


def _view_check(cls: type[LumenViewNode]) -> Any:
    settings = (cls.param.spec.metadata or {}).get("flowdash", {}).get("settings") or []

    def check_view(config: dict[str, Any], inputs: dict[str, Any]) -> Any:
        return check_view_spec(
            config, inputs, settings=settings, check=cls.check_spec, warnings=cls.spec_warnings,
            requires_spec=cls.requires_spec,
        )

    check_view.__doc__ = f"Check a {cls.__name__} config against its input's columns."
    return check_view


def _next_tick(callback: Any) -> None:
    """Call ``callback`` once the session's current event is handled; outside a session, never."""
    doc = pn.state.curdoc
    if doc is not None and doc.session_context is not None:
        doc.add_next_tick_callback(callback)


def _legacy_new_view(editor: Any) -> bool:
    """
    Whether a view created now is one someone added, for a panel-flowdash
    without ``placement_context``. A dashboard is loaded, or a version
    restored, with the canvas muted; without the flag, in a FlowDash that
    renamed it, no view can tell a new node from a loaded one, so none
    takes a default.
    """
    return editor is not None and not getattr(editor, "_muted", True)


def _legacy_store_default(node: LumenViewNode, default: dict[str, Any]) -> None:
    """
    Set the spec of ``node`` to ``default`` for a panel-flowdash without
    ``FlowDash.set_default``, as part of the edit that gave the node its
    pipeline, which FlowDash records right after the pipeline arrives. A
    pipeline that arrives on its own, e.g. a query run off the event loop,
    has no such edit, so the default is then recorded as a config edit on
    the next tick.
    """
    editor = node.editor
    history = getattr(editor, "_history", None)
    if history is None or not hasattr(editor, "_hydrating"):
        node.spec = default
        return
    head = getattr(history, "_head", None)
    # FlowDash records no version while it hydrates; the spec is still synced back and marks the dashboard unsaved.
    hydrating, editor._hydrating = editor._hydrating, True
    try:
        node.spec = default
    finally:
        editor._hydrating = hydrating
    node_ref, editor_ref = weakref.ref(node), weakref.ref(editor)

    def record():
        owner, view = editor_ref(), node_ref()
        if owner is None or view is None or getattr(getattr(owner, "_history", None), "_head", None) != head:
            return
        graph = getattr(owner, "_dataflow_graph", None)
        name_of, record_edit = getattr(owner, "_node_name", None), getattr(owner, "_record_edit", None)
        if graph is None or not callable(name_of) or not callable(record_edit):
            return
        try:
            node_id = view.node_id()
            if node_id is None or graph.get_config_state(node_id) is None:
                return
            name = name_of(node_id)
            record_edit(f"Changed the config of {name}", op="set_config", args={"name": name, "keys": ["spec"]})
        except Exception:
            logger.warning("The default spec of a Lumen view could not be recorded", exc_info=True)

    _next_tick(record)


def _legacy_code_panel_node(editor: Any) -> Any:
    """The id of the node the code panel shows, for a panel-flowdash without ``FlowDash.code_panel_node``."""
    try:
        return editor.code_panel.param.instance_id
    except Exception:
        return None


@functools.cache
def _view_class(view_type: str) -> type:
    """The Lumen View class of ``view_type``, e.g. ``"vegalite"``, imported on first use."""
    from ..views.base import View

    return View._get_type(view_type)


class LumenViewNode(GlobalFilterTarget, LumenNode):
    """
    A view of a Lumen pipeline drawn from a spec, e.g. a chart. The spec
    names the pipeline's columns and brings no data: the view gets the
    pipeline's rows, at most the host's row ceiling, through ``fetch``, so
    it shares one query with the pipeline's other readers.

    A subclass names the Lumen ``view_type`` it draws (e.g. ``"vegalite"``),
    declares its ``spec`` with ``spec_param`` and is registered with
    ``register_view``. It can override ``check_spec`` (refuse a spec before
    it is drawn), ``spec_warnings``, ``spec_fields`` (the columns the
    dashboard's schema contract records), ``default_spec`` (the spec a new
    view starts from), ``_view_params`` (how the spec becomes the Lumen
    view's params, filling the tile by default), ``_built``, ``_summary``
    and ``_preview`` (the card and its pop-out).

    A view the editor creates with an empty spec takes ``default_spec`` of
    its pipeline's schema the first time it has one, as an edit of its
    config that is saved and undone like any other. A view loaded with a
    dashboard keeps the spec it was saved with, empty or not. Without a
    spec, a view that needs one shows ``_empty`` and offers to edit it.
    """

    __abstract = True

    pipeline = param.Parameter(doc="The upstream pipeline.")

    spec = spec_param("The view's spec, without data.")

    _filter_inputs = ("pipeline",)

    # Whether a pipeline is connected, so the body says what the view waits for when nothing is drawn.
    _connected = param.Boolean(default=False, precedence=-1)

    # The type of the Lumen View the node draws, e.g. "vegalite".
    view_type: ClassVar[str] = ""

    # Whether the view needs a spec to draw; a table, say, draws its pipeline with an empty one.
    requires_spec: ClassVar[bool] = True

    # Whether the view takes the dashboard's colour scheme, so it is drawn again when that changes.
    themed: ClassVar[bool] = False

    _waiting: ClassVar[str] = "Waiting for a pipeline"

    _empty: ClassVar[str] = "No chart yet"

    def __init__(self, **params):
        if placement_context is None:
            self._may_default = _legacy_new_view(current_dashboard())
        else:
            # A view a dashboard load, an undo or a restored version brings back keeps its spec.
            self._may_default = placement_context() == "new"
        super().__init__(**params)
        settings = getattr(self.editor, "settings", None)
        if self.themed and settings is not None and "color_scheme" in settings.param:
            node = weakref.ref(self)

            def follow(*_events):
                if (view := node()) is not None:
                    view._refresh()

            settings.param.watch(follow, "color_scheme")

    @staticmethod
    def check_spec(spec: dict[str, Any], columns: list[str]) -> dict[str, Any]:
        """The spec to draw for a pipeline with ``columns``, or ``ValueError`` saying what to fix."""
        return spec

    @staticmethod
    def spec_warnings(spec: dict[str, Any]) -> list[str]:
        """Problems of ``spec`` shown with the config without refusing it."""
        return []

    @staticmethod
    def spec_fields(spec: Any) -> set[str]:
        """The columns of its pipeline ``spec`` reads."""
        return contract_spec_fields(spec)

    @staticmethod
    def default_spec(schema: dict[str, Any]) -> dict[str, Any] | None:
        """The spec a new view of a pipeline with ``schema`` starts from, or None to start empty."""
        return None

    @param.depends("pipeline", "spec", "_global_filters", watch=True, on_init=True)
    def _refresh(self, *events):
        if self._may_default and self._apply_default():
            return
        with param.parameterized.batch_call_watchers(self):
            super()._refresh()
            self._connected = self.pipeline is not None

    @param.depends("_object", "_error", "_connected")
    def _render(self):
        return super()._render()

    def _apply_default(self) -> bool:
        """Start from ``default_spec`` once the pipeline's schema is known; True if the spec changed."""
        if self.spec:
            self._may_default = False
            return False
        editor = self.editor
        if editor is not None and (editor.preview or editor.read_only or not editor.editable):
            return False
        pipeline = self.pipeline
        if pipeline is None or is_failure(pipeline):
            return False
        try:
            schema = pipeline.schema
        except Exception:
            return False
        if not schema:
            return False
        self._may_default = False
        try:
            default = self.default_spec(schema)
        except Exception:
            return False
        if not default:
            return False
        return self._store_default(default)

    def _store_default(self, default: dict[str, Any]) -> bool:
        """
        Give the node ``default`` as its spec through the editor's
        ``set_default``, which records it as part of the edit that gave the
        node its pipeline, e.g. connecting it, so one undo takes back both.
        True if the spec changed; one set off the event loop changes later.
        """
        if placement_context is None:
            _legacy_store_default(self, default)
            return True
        editor, node_id = self.editor, self.node_id()
        if editor is None or node_id is None:
            return False
        editor.set_default(node_id, {"spec": default})
        return bool(self.spec)

    def _color_scheme(self) -> str | None:
        scheme = getattr(getattr(self.editor, "settings", None), "color_scheme", None)
        return scheme if isinstance(scheme, str) and scheme else None

    def _view_params(self, spec: dict[str, Any]) -> dict[str, Any]:
        """
        The params of the Lumen view besides its pipeline, from the checked
        spec. The view fills its tile, which a view whose spec cannot size it
        to its container does through its pane's sizing.
        """
        return {"spec": spec, "sizing_mode": "stretch_both"}

    def _resolve(self):
        if self.pipeline is None or (self.requires_spec and not self.spec):
            return None
        if is_failure(self.pipeline):
            raise ValueError(f"The input failed: {failure_text(self.pipeline)}")
        spec = self.check_spec(self.spec or {}, pipeline_columns(self.pipeline))
        # The global filters become part of the query, and the rows come through fetch, so
        # the view shares one query with the pipeline's other readers.
        pipeline = self._shown(self.pipeline)
        rows = self._view_rows(fetch(pipeline, row_ceiling(self.editor)), spec)
        view_type = _view_class(self.view_type)
        view = view_type(pipeline=_clone(pipeline, data=rows), **self._view_params(spec))
        view.update()
        self._built(view)
        return view

    def _filter_pipeline(self) -> Any:
        return self.pipeline

    def _view_rows(self, rows: Any, spec: dict[str, Any]) -> Any:
        """The rows the view is drawn with, e.g. only the columns its spec reads."""
        return rows

    def _built(self, view: Any) -> None:
        """Called with each view the node draws, before it is shown."""

    def _body(self):
        view = self._object
        if view is not None:
            return view.panel
        if self.pipeline is None:
            return pn.pane.Markdown(self._waiting)
        return self._empty_state()

    def _empty_state(self) -> pn.viewable.Viewable:
        """What a view without a spec shows: ``_empty``, and a button opening the spec while it can be edited."""
        message = pn.pane.Markdown(self._empty, margin=(0, 5))
        editor = self.editor
        if editor is None or not callable(getattr(editor, "edit_spec", None)):
            return message
        node_id = self.node_id()
        shown = _legacy_code_panel_node(editor) if placement_context is None else editor.param.code_panel_node
        # Hidden while the code panel shows this node, which may also be showing the view itself.
        button = pmui.Button(
            label="Edit spec", icon="data_object", variant="outlined", size="small",
            css_classes=["lumen-edit-spec"],
            visible=pn.bind(
                lambda editable, read_only, preview, shown: (
                    editable and not read_only and not preview and (shown is None or shown != node_id)
                ),
                editor.param.editable, editor.param.read_only, editor.param.preview,
                shown,
            ),
        )
        button.on_click(lambda _event: self.edit_spec())
        return pn.Column(
            message, button, sizing_mode="stretch_both",
            styles={"align-items": "center", "justify-content": "center"},
        )

    def edit_spec(self) -> bool:
        """Open the view's spec in the editor's code panel; False where it cannot be shown."""
        editor, node_id = self.editor, self.node_id()
        if editor is None or node_id is None:
            return False
        return bool(editor.edit_spec(node_id, "spec"))

    def _summary(self, spec: dict[str, Any], data: Any) -> str:
        """A line saying what the view shows, e.g. for hosts that draw no chart previews."""
        title = getattr(getattr(type(self), "__panel_app_metadata__", None), "title", None) or "View"
        if data is None or not hasattr(data, "__len__"):
            return title
        return f"{title}, {len(data):,} row{'' if len(data) == 1 else 's'}"

    def _preview(self, spec: dict[str, Any], data: Any) -> Any:
        """
        What the node's card shows and pops out: by default the first rows
        of the view's data and how many the pipeline has.
        """
        if isinstance(self.pipeline, Pipeline) and (rows := _fetched_preview(self.pipeline)) is not None:
            return rows
        if data is None or not hasattr(data, "head"):
            return self._summary(spec, data)
        return _preview_frame(data.head(PREVIEW_ROWS), len(data))

    def __flowdash_preview__(self, output: str | None = None) -> Any:
        if self._error is None and self._object is None and self.pipeline is not None:
            return self._empty
        if self._error is not None or self._object is None:
            return None
        spec = getattr(self._object, "spec", None)
        spec = spec if isinstance(spec, dict) else self.spec
        data = getattr(getattr(self._object, "pipeline", None), "data", None)
        return self._preview(spec, data)


@register_view(
    title="Lumen Vega-Lite view", icon="bar_chart",
    description=(
        "A Vega-Lite chart of a pipeline, its spec naming columns, no data or URLs. A selection param "
        "feeds the selection output: wire it to a filter on the same unfiltered pipeline and feed other "
        "views from that."
    ),
    config=["debounce"],
    provides=[{"key": "selection", "type": SELECTION_TYPE}],
)
class ViewNode(LumenViewNode):

    spec = spec_param("""
        The Vega-Lite spec, without data. A width or height it leaves out
        follows the tile's, and colours it leaves out the dashboard's
        ``color_scheme`` setting, if it has one. A new chart starts from
        one its pipeline's columns suggest: a line of a number over a date
        or a year, the largest categories' sums as bars, a scatter of two
        numbers or a histogram of one.""", settings=VEGA_LITE_SETTINGS)

    debounce = param.Integer(default=DEBOUNCE, bounds=(0, None), doc="""
        Milliseconds the chart waits after the last change of a selection
        before reporting it, so a drag reports once rather than on every
        mouse move.""")

    selection = param.ClassSelector(class_=Selection, default=EMPTY, doc="""
        What is selected on the chart, by the spec's selection params.""")

    view_type = "vegalite"

    themed = True

    check_spec = staticmethod(check_chart)

    spec_warnings = staticmethod(selection_warnings)

    default_spec = staticmethod(vega_lite_default)

    def __init__(self, **params):
        self._link = None
        super().__init__(**params)

    def _view_params(self, spec: dict[str, Any]) -> dict[str, Any]:
        # The spec fills the tile, so the pane keeps the sizing Vega gives a container size.
        return {"spec": themed_spec(responsive_spec(spec), self._color_scheme())}

    def _built(self, view: Any) -> None:
        if isinstance(view._panel, pn.pane.Vega):
            # A new chart starts with nothing selected; the link reports that.
            self._link = link_selection(
                self, view._panel, debounce=self.debounce, link=self._link, source=self.node_id,
            )

    @param.depends("debounce", watch=True)
    def _update_debounce(self):
        if self._link is not None:
            self._link.debounce = self.debounce
            pane = getattr(self._object, "_panel", None)
            if isinstance(pane, pn.pane.Vega):
                pane.debounce = self.debounce

    def _summary(self, spec: dict[str, Any], data: Any) -> str:
        return _chart_summary(spec, data)

    def _preview(self, spec: dict[str, Any], data: Any) -> Any:
        # A host that does not draw charts would list the dict's keys, so it gets a line of text.
        if "chart" not in _preview_kinds():
            return self._summary(spec, data)
        return {
            PREVIEW_KIND_KEY: "chart",
            "format": "vega-lite",
            "spec": copy.deepcopy(spec),
            "data": _sample_rows(data, PREVIEW_ROWS),
        }


COMPONENTS = {
    SOURCE_ID: SourceNode, TABLE_ID: TableNode, FILTER_ID: FilterNode, SQL_ID: SQLNode,
    DATA_ID: DataNode, VIEW_ID: ViewNode, METRIC_ID: MetricNode, PARAMETER_ID: ParameterNode,
}
