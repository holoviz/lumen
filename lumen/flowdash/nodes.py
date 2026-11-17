"""
Lumen sources, SQL queries and Vega-Lite views as FlowDash nodes.

Edges between these nodes carry a Lumen ``Source`` or ``Pipeline`` rather
than a DataFrame, so a query placed on a SQL node runs in the source's
engine. Each node resolves its Lumen object once per change and every
output reads that object.
"""
from __future__ import annotations

import hashlib

from typing import Any

import panel as pn
import param

from panel.viewable import Viewer
from panel_flowdash import register, register_port_type
from panel_flowdash.util import is_failure

from ..ai.agents.sql import validate_read_only_sql
from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource, Source
from ..sources.duckdb import DuckDBSource
from ..transforms.sql import SQLLimit

SOURCE_TYPE = register_port_type("lumen.Source")
PIPELINE_TYPE = register_port_type("lumen.Pipeline")

SOURCE_ID = "Lumen/source"
SQL_ID = "Lumen/sql"
VIEW_ID = "Lumen/view"

# The same cap SQLAgent puts on a final query.
RESULT_LIMIT = 1_000_000

_BOUND: dict[str, Source] = {}


def bind_source(source: Source, ref: str | None = None) -> str:
    """
    Make a live ``Source`` available to source nodes whose ``ref`` is
    ``ref`` (default: the source's name), e.g. an in-memory fixture that
    cannot be rebuilt from its spec. Returns the ref.
    """
    ref = ref or source.name
    _BOUND[ref] = source
    return ref


def unbind_source(ref: str) -> None:
    """Forget a source bound with ``bind_source``."""
    _BOUND.pop(ref, None)


def source_config(source: Source, ref: str | None = None) -> dict[str, Any]:
    """Config for a source node bound to ``source``, with its spec when it serializes."""
    config: dict[str, Any] = {"ref": ref or source.name}
    try:
        config["spec"] = source.to_spec(context={})
    except Exception:
        pass
    return config


def compile_sql(source: Source, sql: str, table: str) -> Pipeline:
    """
    A pipeline whose table is ``sql`` evaluated by ``source``'s engine.

    Raises ``ValueError`` unless ``sql`` is a single read-only query.
    """
    if not isinstance(source, BaseSQLSource):
        raise ValueError(f"{type(source).__name__} does not run SQL.")
    validate_read_only_sql(sql, source.dialect)
    # The result table joins the source's own tables, so it must not shadow one the query reads.
    while table in source.get_tables():
        table = f"{table}_result"
    kwargs = {"materialize": False} if isinstance(source, DuckDBSource) else {}
    query_source = source.create_sql_expr_source({table: sql}, **kwargs)
    limit = SQLLimit(limit=RESULT_LIMIT, write=source.dialect, pretty=True, identify=False)
    return Pipeline(source=query_source, table=table, sql_transforms=[limit], name=table)


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
    description="A Lumen data source, from its spec or bound by the host by ref.",
    config=["spec", "ref"],
    provides=[{"key": "source", "type": SOURCE_TYPE}],
)
class SourceNode(LumenNode):

    spec = param.Dict(default={}, doc="The source's Lumen spec.")

    ref = param.String(default="", doc="Name of a source the host bound with bind_source.")

    @param.depends("spec", "ref", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> Source | None:
        if self.ref and self.ref in _BOUND:
            return _BOUND[self.ref]
        if self.spec:
            return Source.from_spec(dict(self.spec))
        if self.ref:
            raise ValueError(f"No source is bound as {self.ref!r} and the node has no spec.")
        return None

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
    config=["sql", "table"],
    requires=[{"key": "source", "type": SOURCE_TYPE, "required": True}],
    provides=[{"key": "pipeline", "type": PIPELINE_TYPE}, {"key": "data", "type": "DataFrame"}],
)
class SQLNode(LumenNode):

    source = param.Parameter(doc="The upstream source.")

    sql = param.String(default="", doc="One read-only SELECT or WITH query.")

    table = param.String(default="", doc="Name of the result table; derived from the SQL if empty.")

    @param.depends("source", "sql", "table", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> Pipeline | None:
        if self.source is None or not self.sql.strip():
            return None
        return compile_sql(self.source, self.sql, self.table or default_table(self.sql))

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
    description="A Vega-Lite chart of a Lumen pipeline; the spec names the pipeline's columns.",
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

        return VegaLiteView(pipeline=self.pipeline, spec=dict(self.spec))

    def _body(self):
        view = self._object
        return pn.pane.Markdown("Waiting for a pipeline") if view is None else view.panel


COMPONENTS = {SOURCE_ID: SourceNode, SQL_ID: SQLNode, VIEW_ID: ViewNode}
