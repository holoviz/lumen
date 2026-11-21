"""
Lumen sources, tables, filters, SQL queries and Vega-Lite views as FlowDash
nodes.

Edges between these nodes carry a Lumen ``Source`` or ``Pipeline`` rather
than a DataFrame, so a query placed on a SQL node runs in the source's
engine, and the filters and row limit of table and filter nodes become part
of the query the source runs rather than being applied after fetching. Table
and filter nodes fetch nothing themselves: rows are fetched where they are
needed, by a data node (the boundary to components that take a DataFrame),
a view node or a tile preview. Each node resolves its Lumen object once per
change and every output reads that object.

Source nodes never build a source from their config: the host hands the
editor the sources a dashboard may use, by ref, in the editor's context under
``"lumen.sources"`` (``FlowDash(context={"lumen.sources": {...}})``, or
``bind_source`` on an editor that exists), and a source node only names one
of them by ``ref``. Whoever edits a dashboard can therefore only point it at
sources the host chose, never open a new connection. A source node whose ref
is bound later picks the source up then.

The host also sets how many rows a query of these nodes may return
(``set_row_ceiling``); a node's ``limit`` can lower that, never raise it.
"""
from __future__ import annotations

import asyncio
import copy
import datetime as dt
import hashlib
import inspect
import math
import threading
import weakref

from collections.abc import Iterator, Mapping
from concurrent.futures import (
    Future, ThreadPoolExecutor, TimeoutError as FutureTimeout,
)
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import panel as pn
import param

from panel.viewable import Viewer
from panel_flowdash import (
    current_dashboard, dashboard_context, register, register_port_type,
)
from panel_flowdash.dataflow_engine import failure_text
from panel_flowdash.util import is_failure

from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource, Source
from ..sources.duckdb import DuckDBSource
from ..transforms.base import Filter as FilterTransform, Iloc
from ..transforms.sql import NO_BOOLEAN_LITERALS, SQLFilter, SQLLimit
from ..util import get_dataframe_schema
from .checks import (
    check_chart, check_filter_config, check_query, check_source_config,
    check_sql_config, check_table_config, check_view_config, forget_probes,
    pipeline_columns,
)
from .view_settings import VEGA_LITE_SETTINGS, themed_spec

SOURCE_COLOR = "#6D4C41"
PIPELINE_COLOR = "#0072B5"


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
    "lumen.Pipeline", color=PIPELINE_COLOR, columns=pipeline_columns
)

SOURCE_ID = "Lumen/source"
TABLE_ID = "Lumen/table"
FILTER_ID = "Lumen/filter"
SQL_ID = "Lumen/sql"
DATA_ID = "Lumen/data"
VIEW_ID = "Lumen/view"

# The key of the editor context the host puts its sources in, by ref.
SOURCES_KEY = "lumen.sources"

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


def bind_source(editor: Any, source: Source, ref: str = "data") -> str:
    """
    Let source nodes of ``editor`` whose ``ref`` is ``ref`` use ``source``,
    including nodes already placed, by adding it to the editor's context.
    Returns the ref.
    """
    editor.context.setdefault(SOURCES_KEY, {})[ref] = source
    for node in list(_SOURCE_NODES.get(editor, ())):
        if node.ref == ref:
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
    or else ``dashboard_context()``. It is the only place a node looks up a
    host source.
    """
    context = dashboard_context() if context is None else context
    return dict(context.get(SOURCES_KEY) or {})


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
    would run it, so it runs once.
    """
    key = (id(source), sql)
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

    Filters downstream wrap the capped statement in a subquery, so they
    see exactly the rows the node outputs. SQL does not promise that the
    outer query keeps the subquery's ORDER BY; DuckDB keeps it in practice.

    Raises ``ValueError`` unless ``sql`` passes ``check_query``.
    """
    if is_failure(source):
        raise ValueError(f"The source failed: {failure_text(source)}")
    if not isinstance(source, BaseSQLSource):
        raise ValueError(f"{type(source).__name__} does not run SQL.")
    check_query(sql, source)
    frame = execute(source, result_sql(source, sql, limit), timeout)
    # The result table joins the source's own tables, so it must not shadow one the query reads.
    while table in source.get_tables():
        table = f"{table}_result"
    kwargs = {"materialize": False} if isinstance(source, DuckDBSource) else {}
    # The capped statement, so a filter downstream sees the rows the node outputs and no more.
    query_source = source.create_sql_expr_source({table: result_sql(source, sql, limit)}, **kwargs)
    schema = get_dataframe_schema(frame)["items"]["properties"]
    pipeline = Pipeline(source=query_source, table=table, schema=schema, data=frame, name=table)
    _FETCHED[pipeline] = {math.inf: frame}
    return pipeline


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
    tables = source.get_tables()
    if table not in tables:
        raise ValueError(f"The source has no table {table!r}. Its tables are: {', '.join(sorted(tables))}.")
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
    raise ValueError(
        f"Cannot filter on {value!r}: give a single value, a list of values, or a (start, end) "
        "range of numbers or dates."
    )


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
    if is_failure(pipeline):
        raise ValueError(f"The input failed: {failure_text(pipeline)}")
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Expected a Lumen pipeline, got {type(pipeline).__name__}.")
    columns = pipeline_columns(pipeline)
    if field not in columns:
        raise ValueError(f"The data has no column {field!r}. Its columns are: {', '.join(columns)}.")
    if not filter_value(value):
        return pipeline
    condition = [(field, filter_condition(value))]
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


def pipeline_sql(pipeline: Pipeline) -> str | None:
    """
    The statement a SQL source runs for ``pipeline``, filters and limit
    included, or None when the pipeline is not read straight from a SQL
    source.
    """
    source = pipeline.source
    if pipeline.pipeline is not None or not isinstance(source, BaseSQLSource):
        return None
    sql = source.get_sql_expr(pipeline.table)
    conditions = list(pipeline._filter_query().items())
    if conditions and getattr(source, "filter_in_sql", True):
        sql = SQLFilter(conditions=conditions, write=source.dialect).apply(sql)
    for transform in pipeline.sql_transforms:
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
    data = data if len(data) <= limit else data.head(limit)
    fetched[limit] = data
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


def _source_summary(source: Source) -> pd.DataFrame:
    """Compact table summary for a SourceNode preview without triggering queries."""
    tables = source.get_tables()
    cached = source._get_schema_cache() or {}
    in_memory = source.tables if isinstance(getattr(source, "tables", None), Mapping) else {}
    entries: list[tuple[str, int | None, int | None]] = []
    for table in tables:
        rows: int | None = None
        cols: int | None = None
        schema = cached.get(table)
        if isinstance(schema, dict):
            cols = sum(1 for key in schema if key != "__len__")
            length = schema.get("__len__")
            if isinstance(length, int) and not isinstance(length, bool):
                rows = length
        elif (frame := in_memory.get(table)) is not None:
            shape = getattr(frame, "shape", None)
            if isinstance(shape, tuple) and len(shape) == 2 and hasattr(frame, "columns"):
                if isinstance(shape[0], int) and not isinstance(shape[0], bool):
                    rows = shape[0]
                if isinstance(shape[1], int) and not isinstance(shape[1], bool):
                    cols = shape[1]
        entries.append((table, rows, cols))
    data: dict[str, list[Any]] = {"table": [table for table, _, _ in entries]}
    if any(rows is not None for _, rows, _ in entries):
        data["rows"] = [rows if rows is not None else "" for _, rows, _ in entries]
    if any(cols is not None for _, _, cols in entries):
        data["cols"] = [cols if cols is not None else "" for _, _, cols in entries]
    return pd.DataFrame(data)


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

    __abstract = True

    _preview_of: weakref.ref | None = None

    def __init__(self, **params):
        editor = current_dashboard()
        self._editor = weakref.ref(editor) if editor is not None else None
        # Taken now, while the editor creating the node is known; it follows later bindings.
        self._context = dashboard_context()
        super().__init__(**params)
        if editor is not None and self._reads_source:
            _watch_refresh(editor, self)

    @property
    def editor(self) -> Any:
        return self._editor() if self._editor is not None else None

    # Whether the node reads a source directly, and so runs again on the dashboard's refresh.
    _reads_source: ClassVar[bool] = False

    def _on_dashboard_refresh(self):
        """
        Run the node again on the dashboard's refresh with its source's cache
        cleared. Only nodes that read a source do this: each builds a new
        pipeline, which flows downstream and is fetched afresh, so one
        refresh runs each query once.
        """
        source = getattr(self, "source", None)
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
            self._error = str(exc)

    def _value(self):
        if self._error is not None:
            return param.ReactiveError(ValueError(self._error))
        return self._object

    @param.depends("_object", "_error")
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


@register(
    component=True, page=False, title="Lumen Source", icon="storage",
    description="A Lumen data source the app provides, named by its ref.",
    config=["ref"],
    check_config=check_source_config,
    provides=[{"key": "source", "type": SOURCE_TYPE}],
)
class SourceNode(LumenNode):

    ref = param.String(default="", doc="Name of a source the host provides.")

    def __init__(self, **params):
        super().__init__(**params)
        if self.editor is not None:
            _SOURCE_NODES.setdefault(self.editor, weakref.WeakSet()).add(self)

    @param.depends("ref", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> Source | None:
        if not self.ref:
            return None
        sources = host_sources(self._context)
        if self.ref not in sources:
            raise ValueError(missing_source(self.ref, sources))
        return sources[self.ref]

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def source(self):
        return self._value()

    def _body(self):
        source = self._object
        if source is None:
            return pn.pane.Markdown("No source")
        return pn.pane.Markdown(f"{type(source).__name__}: {', '.join(source.get_tables())}")

    def __flowdash_preview__(self, output: str | None = None) -> pd.DataFrame | None:
        if self._error is not None or self._object is None:
            return None
        return _source_summary(self._object)


@register(
    component=True, page=False, title="Lumen SQL", icon="code",
    description=(
        "A read-only SQL query run by the source's engine. Outputs a Lumen pipeline "
        "over the result and the result as a DataFrame."
    ),
    config=["sql", "table", "limit"],
    check_config=check_sql_config,
    requires=[{"key": "source", "type": SOURCE_TYPE, "required": True}],
    provides=[{"key": "pipeline", "type": PIPELINE_TYPE}, {"key": "data", "type": "DataFrame"}],
)
class SQLNode(LumenNode):

    _reads_source = True

    source = param.Parameter(doc="The upstream source.")

    sql = param.String(
        default="",
        doc="One read-only SELECT or WITH query.",
        metadata={"flowdash": {"widget": "code", "language": "sql"}},
    )

    table = param.String(default="", doc="Name of the result table; derived from the SQL if empty.")

    limit = param.Integer(default=None, bounds=(1, None), allow_None=True, doc="""
        Rows of the result kept in memory, at most the host's row ceiling,
        which is also the default.""")

    @param.depends("source", "sql", "table", "limit", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> Pipeline | None:
        if self.source is None or not self.sql.strip():
            return None
        limit = effective_limit(self.editor, self.limit)
        return compile_sql(self.source, self.sql, self.table or default_table(self.sql), limit)

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def pipeline(self):
        return self._value()

    @param.output(param.DataFrame)
    @param.depends("_object", "_error")
    def data(self):
        pipeline = self._value()
        return pipeline if pipeline is None or is_failure(pipeline) else pipeline.data

    def _body(self):
        pipeline = self._object
        if pipeline is None:
            return pn.pane.Markdown("Waiting for a source and a query")
        return pn.widgets.Tabulator(pipeline.data, disabled=True, sizing_mode="stretch_both")

    def __flowdash_preview__(self, output: str | None = None) -> Any:
        if self._error is not None or self._object is None:
            return None
        data = self._object.data
        return data if data is not None else _pipeline_preview(self, self._object)


class PipelineNode(LumenNode):
    """A node whose output is a lazy pipeline, previewed only when its view is shown."""

    __abstract = True

    _waiting = ""

    def _body(self):
        pipeline = self._object
        if pipeline is None:
            return pn.pane.Markdown(self._waiting)
        preview = fetch(pipeline, PREVIEW_ROWS)
        return pn.widgets.Tabulator(preview, disabled=True, sizing_mode="stretch_both")

    def __panel__(self):
        return pn.param.ParamMethod(self._render, lazy=True, sizing_mode="stretch_both")


@register(
    component=True, page=False, title="Lumen Table", icon="table_chart",
    description=(
        "A table of a Lumen source. Outputs a Lumen pipeline that fetches nothing by itself: "
        "filter nodes and the row limit become part of the source's query."
    ),
    config=["table", "limit"],
    check_config=check_table_config,
    requires=[{"key": "source", "type": SOURCE_TYPE, "required": True}],
    provides=[{"key": "pipeline", "type": PIPELINE_TYPE}],
)
class TableNode(PipelineNode):

    _reads_source = True

    source = param.Parameter(doc="The upstream source.")

    table = param.String(default="", doc="Name of a table of the source.")

    limit = param.Integer(default=None, bounds=(1, None), allow_None=True, doc="""
        Rows the source returns at most, at most the host's row ceiling,
        which is also the default. It caps the rows after the filters of
        filter nodes downstream, which become part of the same query, so
        it is not a sample the filters then narrow.""")

    _waiting = "Waiting for a source and a table"

    @param.depends("source", "table", "limit", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> Pipeline | None:
        if self.source is None or not self.table:
            return None
        return table_pipeline(self.source, self.table, effective_limit(self.editor, self.limit))

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def pipeline(self):
        return self._value()


@register(
    component=True, page=False, title="Lumen Filter", icon="filter_alt",
    description=(
        "Keeps the rows of a Lumen pipeline whose field matches the value input: equal to a "
        "single value, one of a list, or within a (start, end) range of numbers or dates. An "
        "empty value keeps every row, and a value of another shape is refused. On a table read "
        "from a SQL source the source applies the filter in its query, before the table's limit."
    ),
    config=["field"],
    check_config=check_filter_config,
    requires=[
        {"key": "pipeline", "type": PIPELINE_TYPE, "required": True},
        {"key": "value", "required": False, "blocking": False},
    ],
    provides=[{"key": "filtered", "type": PIPELINE_TYPE}],
)
class FilterNode(PipelineNode):

    pipeline = param.Parameter(doc="The upstream pipeline.")

    value = param.Parameter(doc="The value to keep, e.g. a widget's selection.")

    field = param.String(default="", doc="The column to filter on.")

    _waiting = "Waiting for a pipeline and a field"

    @param.depends("pipeline", "value", "field", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> Pipeline | None:
        if self.pipeline is None or not self.field:
            return None
        return filtered_pipeline(self.pipeline, self.field, self.value, row_ceiling(self.editor))

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def filtered(self):
        return self._value()


@register(
    component=True, page=False, title="Lumen Data", icon="table_rows",
    description=(
        "Fetches the rows of a Lumen pipeline, at most the host's row ceiling, and outputs "
        "them as a DataFrame for components that take one."
    ),
    requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
    provides=[{"key": "data", "type": "DataFrame"}],
)
class DataNode(LumenNode):

    pipeline = param.Parameter(doc="The upstream pipeline.")

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
        return pn.widgets.Tabulator(self._object, disabled=True, sizing_mode="stretch_both")


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


@register(
    component=True, page=False, title="Lumen Vega-Lite view", icon="bar_chart",
    description="A Vega-Lite chart of a Lumen pipeline; the spec names the pipeline's columns and brings no data or URLs.",
    config=["spec"],
    check_config=check_view_config,
    requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
)
class ViewNode(LumenNode):

    pipeline = param.Parameter(doc="The upstream pipeline.")

    spec = param.Dict(default={}, doc="""
        The Vega-Lite spec, without data. A width or height it leaves out
        follows the tile's, and colours it leaves out the dashboard's
        ``color_scheme`` setting, if it has one.""",
        metadata={"flowdash": {"widget": "spec", "settings": VEGA_LITE_SETTINGS}})

    def __init__(self, **params):
        super().__init__(**params)
        settings = getattr(self.editor, "settings", None)
        if settings is not None and "color_scheme" in settings.param:
            node = weakref.ref(self)

            def follow(*_events):
                if (view := node()) is not None:
                    view._refresh()

            settings.param.watch(follow, "color_scheme")

    @param.depends("pipeline", "spec", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _color_scheme(self) -> str | None:
        scheme = getattr(getattr(self.editor, "settings", None), "color_scheme", None)
        return scheme if isinstance(scheme, str) and scheme else None

    def _resolve(self):
        if self.pipeline is None or not self.spec:
            return None
        from ..views.base import VegaLiteView

        if is_failure(self.pipeline):
            raise ValueError(f"The input failed: {failure_text(self.pipeline)}")
        spec = check_chart(self.spec, pipeline_columns(self.pipeline))
        # The rows come through fetch, so the chart shares one query with the pipeline's other readers.
        rows = fetch(self.pipeline, row_ceiling(self.editor))
        spec = themed_spec(responsive_spec(spec), self._color_scheme())
        return VegaLiteView(pipeline=_clone(self.pipeline, data=rows), spec=spec)

    def _body(self):
        view = self._object
        return pn.pane.Markdown("Waiting for a pipeline") if view is None else view.panel

    def __flowdash_preview__(self, output: str | None = None) -> Any:
        if self._error is not None or self._object is None:
            return None
        spec = self._object.spec
        data = getattr(getattr(self._object, "pipeline", None), "data", None)
        # A host that does not draw charts would list the dict's keys, so it gets a line of text.
        if "chart" not in _preview_kinds():
            return _chart_summary(spec, data)
        return {
            PREVIEW_KIND_KEY: "chart",
            "format": "vega-lite",
            "spec": copy.deepcopy(spec),
            "data": _sample_rows(data, PREVIEW_ROWS),
        }


COMPONENTS = {
    SOURCE_ID: SourceNode, TABLE_ID: TableNode, FILTER_ID: FilterNode, SQL_ID: SQLNode,
    DATA_ID: DataNode, VIEW_ID: ViewNode,
}
