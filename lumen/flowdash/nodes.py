"""
Lumen sources, SQL queries and Vega-Lite views as FlowDash nodes.

Edges between these nodes carry a Lumen ``Source`` or ``Pipeline`` rather
than a DataFrame, so a query placed on a SQL node runs in the source's
engine. Each node resolves its Lumen object once per change and every
output reads that object.

Source nodes never build a source from their config: the host binds the
sources an editor may use with ``bind_source``, and a source node only
names one of them by ``ref``. Whoever edits a dashboard can therefore only
point it at sources the host chose, never open a new connection.
"""
from __future__ import annotations

import asyncio
import hashlib
import weakref

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import panel as pn
import param

from panel.viewable import Viewer
from panel_flowdash import current_dashboard, register, register_port_type
from panel_flowdash.util import is_failure

from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource, Source
from ..sources.duckdb import DuckDBSource
from ..transforms.sql import SQLLimit
from ..util import get_dataframe_schema
from .checks import check_chart, check_query

SOURCE_TYPE = register_port_type("lumen.Source")
PIPELINE_TYPE = register_port_type("lumen.Pipeline")

SOURCE_ID = "Lumen/source"
SQL_ID = "Lumen/sql"
VIEW_ID = "Lumen/view"

# Rows a SQL node keeps in memory unless its config sets another limit.
RESULT_LIMIT = 100_000
QUERY_TIMEOUT = 30

_BOUND: weakref.WeakKeyDictionary[Any, dict[str, Source]] = weakref.WeakKeyDictionary()
_RESULTS: dict[tuple[int, str], Any] = {}


def bind_source(editor: Any, source: Source, ref: str = "data") -> str:
    """Let source nodes of ``editor`` whose ``ref`` is ``ref`` use ``source``. Returns the ref."""
    _BOUND.setdefault(editor, {})[ref] = source
    return ref


def unbind_source(editor: Any, ref: str) -> None:
    """Forget a source bound with ``bind_source``."""
    _BOUND.get(editor, {}).pop(ref, None)


def bound_sources(editor: Any) -> dict[str, Source]:
    return dict(_BOUND.get(editor, {}))


def result_sql(source: BaseSQLSource, sql: str, limit: int = RESULT_LIMIT) -> str:
    """The statement a SQL node runs for ``sql``, with its row cap."""
    return SQLLimit(limit=limit, write=source.dialect, pretty=True, identify=False).apply(sql)


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
                timeout: float = QUERY_TIMEOUT) -> Pipeline:
    """
    A pipeline whose table is ``sql`` evaluated by ``source``'s engine, with
    its result fetched once under ``limit`` rows and ``timeout`` seconds.

    Raises ``ValueError`` unless ``sql`` passes ``check_query``.
    """
    if not isinstance(source, BaseSQLSource):
        raise ValueError(f"{type(source).__name__} does not run SQL.")
    check_query(sql, source)
    frame = execute(source, result_sql(source, sql, limit), timeout)
    # The result table joins the source's own tables, so it must not shadow one the query reads.
    while table in source.get_tables():
        table = f"{table}_result"
    kwargs = {"materialize": False} if isinstance(source, DuckDBSource) else {}
    query_source = source.create_sql_expr_source({table: sql}, **kwargs)
    schema = get_dataframe_schema(frame)["items"]["properties"]
    return Pipeline(source=query_source, table=table, schema=schema, data=frame, name=table)


def default_table(sql: str) -> str:
    return f"query_{hashlib.sha1(sql.encode()).hexdigest()[:8]}"


class LumenNode(Viewer):

    _object = param.Parameter(default=None, precedence=-1)

    _error = param.String(default=None, allow_None=True, precedence=-1)

    __abstract = True

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
        editor = current_dashboard()
        self._editor = weakref.ref(editor) if editor is not None else None
        super().__init__(**params)

    @param.depends("ref", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> Source | None:
        if not self.ref:
            return None
        editor = self._editor() if self._editor is not None else None
        bound = bound_sources(editor) if editor is not None else {}
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

    source = param.Parameter(doc="The upstream source.")

    sql = param.String(default="", doc="One read-only SELECT or WITH query.")

    table = param.String(default="", doc="Name of the result table; derived from the SQL if empty.")

    limit = param.Integer(default=RESULT_LIMIT, bounds=(1, None), doc="Rows of the result kept in memory.")

    @param.depends("source", "sql", "table", "limit", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> Pipeline | None:
        if self.source is None or not self.sql.strip():
            return None
        return compile_sql(self.source, self.sql, self.table or default_table(self.sql), self.limit)

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

        spec = check_chart(self.spec, [str(c) for c in self.pipeline.data.columns])
        return VegaLiteView(pipeline=self.pipeline, spec=spec)

    def _body(self):
        view = self._object
        return pn.pane.Markdown("Waiting for a pipeline") if view is None else view.panel


COMPONENTS = {SOURCE_ID: SourceNode, SQL_ID: SQLNode, VIEW_ID: ViewNode}
