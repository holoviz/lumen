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

Source nodes never build a source from their config: the host binds the
sources an editor may use with ``bind_source``, and a source node only
names one of them by ``ref``. Whoever edits a dashboard can therefore only
point it at sources the host chose, never open a new connection. To load a
saved dashboard, create the editor, bind its sources, then import the
dashboard; a source node whose ref is bound later picks the source up then.

The host also sets how many rows a query of these nodes may return
(``set_row_ceiling``); a node's ``limit`` can lower that, never raise it.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import math
import weakref

from collections.abc import Iterator
from concurrent.futures import (
    ThreadPoolExecutor, TimeoutError as FutureTimeout,
)
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import panel as pn
import param

from panel.viewable import Viewer
from panel_flowdash import current_dashboard, register, register_port_type
from panel_flowdash.dataflow_engine import failure_text
from panel_flowdash.util import is_failure

from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource, Source
from ..sources.duckdb import DuckDBSource
from ..transforms.base import Filter as FilterTransform, Iloc
from ..transforms.sql import NO_BOOLEAN_LITERALS, SQLFilter, SQLLimit
from ..util import get_dataframe_schema
from .checks import check_chart, check_query, pipeline_columns

SOURCE_TYPE = register_port_type("lumen.Source")
PIPELINE_TYPE = register_port_type("lumen.Pipeline")

SOURCE_ID = "Lumen/source"
TABLE_ID = "Lumen/table"
FILTER_ID = "Lumen/filter"
SQL_ID = "Lumen/sql"
DATA_ID = "Lumen/data"
VIEW_ID = "Lumen/view"

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

_BOUND: weakref.WeakKeyDictionary[Any, dict[str, Source]] = weakref.WeakKeyDictionary()
_CEILINGS: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
_SOURCE_NODES: weakref.WeakKeyDictionary[Any, weakref.WeakSet] = weakref.WeakKeyDictionary()
_RESULTS: dict[tuple[int, str], Any] = {}
# Rows fetched per pipeline, by the limit they were fetched under, so the nodes that
# read one pipeline (a data node and a chart, say) run its query once.
_FETCHED: weakref.WeakKeyDictionary[Pipeline, dict[float, Any]] = weakref.WeakKeyDictionary()


def bind_source(editor: Any, source: Source, ref: str = "data") -> str:
    """
    Let source nodes of ``editor`` whose ``ref`` is ``ref`` use ``source``,
    including nodes already placed. Returns the ref.
    """
    _BOUND.setdefault(editor, {})[ref] = source
    for node in list(_SOURCE_NODES.get(editor, ())):
        if node.ref == ref:
            node._refresh()
    return ref


def unbind_source(editor: Any, ref: str) -> None:
    """Forget a source bound with ``bind_source``."""
    _BOUND.get(editor, {}).pop(ref, None)


def bound_sources(editor: Any) -> dict[str, Source]:
    return dict(_BOUND.get(editor, {}))


def host_sources(editor: Any) -> dict[str, Source]:
    """
    The sources the host lets the dashboard of ``editor`` use, by ref.

    SEAM(panel-flowdash#166): once ``FlowDash.context`` and
    ``panel_flowdash.dashboard_context()`` are released, hosts put their
    sources in ``editor.context["lumen.sources"]`` and this reads
    ``dashboard_context().get("lumen.sources", {})`` (falling back to
    ``bind_source`` bindings during the transition). It is the only place
    a node looks up a host source.
    """
    return bound_sources(editor) if editor is not None else {}


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
    for fetched_limit, data in fetched.items():
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


class LumenNode(Viewer):

    _object = param.Parameter(default=None, precedence=-1)

    _error = param.String(default=None, allow_None=True, precedence=-1)

    __abstract = True

    def __init__(self, **params):
        editor = current_dashboard()
        self._editor = weakref.ref(editor) if editor is not None else None
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

    def __panel__(self):
        return pn.panel(self._render)


@register(
    component=True, page=False, title="Lumen Source", icon="storage",
    description="A Lumen data source the app provides, named by its ref.",
    config=["ref"],
    provides=[{"key": "source", "type": SOURCE_TYPE}],
)
class SourceNode(LumenNode):

    ref = param.String(default="", doc="Name of a source the host bound with bind_source.")

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
        bound = host_sources(self.editor)
        if self.ref not in bound:
            raise ValueError(f"No source is bound as {self.ref!r}; the app binds the sources a dashboard may use.")
        return bound[self.ref]

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def source(self):
        return self._value()

    def _body(self):
        source = self._object
        if source is None:
            return pn.pane.Markdown("No source")
        return pn.pane.Markdown(f"{type(source).__name__}: {', '.join(source.get_tables())}")


@register(
    component=True, page=False, title="Lumen SQL", icon="code",
    description=(
        "A read-only SQL query run by the source's engine. Outputs a Lumen pipeline "
        "over the result and the result as a DataFrame."
    ),
    config=["sql", "table", "limit"],
    requires=[{"key": "source", "type": SOURCE_TYPE, "required": True}],
    provides=[{"key": "pipeline", "type": PIPELINE_TYPE}, {"key": "data", "type": "DataFrame"}],
)
class SQLNode(LumenNode):

    _reads_source = True

    source = param.Parameter(doc="The upstream source.")

    sql = param.String(default="", doc="One read-only SELECT or WITH query.")

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


@register(
    component=True, page=False, title="Lumen Vega-Lite view", icon="bar_chart",
    description="A Vega-Lite chart of a Lumen pipeline; the spec names the pipeline's columns and brings no data or URLs.",
    config=["spec"],
    requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
)
class ViewNode(LumenNode):

    pipeline = param.Parameter(doc="The upstream pipeline.")

    spec = param.Dict(default={}, doc="The Vega-Lite spec, without data.")

    @param.depends("pipeline", "spec", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self):
        if self.pipeline is None or not self.spec:
            return None
        from ..views.base import VegaLiteView

        if is_failure(self.pipeline):
            raise ValueError(f"The input failed: {failure_text(self.pipeline)}")
        spec = check_chart(self.spec, pipeline_columns(self.pipeline))
        # The rows come through fetch, so the chart shares one query with the pipeline's other readers.
        rows = fetch(self.pipeline, row_ceiling(self.editor))
        return VegaLiteView(pipeline=_clone(self.pipeline, data=rows), spec=spec)

    def _body(self):
        view = self._object
        return pn.pane.Markdown("Waiting for a pipeline") if view is None else view.panel


COMPONENTS = {
    SOURCE_ID: SourceNode, TABLE_ID: TableNode, FILTER_ID: FilterNode, SQL_ID: SQLNode,
    DATA_ID: DataNode, VIEW_ID: ViewNode,
}
