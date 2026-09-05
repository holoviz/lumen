"""
A native Panel pane for Mosaic (vgplot) specifications.

Mosaic renders declarative specs by pushing their SQL down to DuckDB and
shipping only aggregated results to the browser, so it scales to datasets far
beyond what an inline-data chart spec can carry.

This is deliberately written as a self-contained Panel component with no
Lumen imports, so it can be contributed upstream to Panel, which has an open
request for exactly this pane (holoviz/panel#7358).

Why this is a `JSComponent` rather than an `AnyWidgetComponent`
---------------------------------------------------------------
Mosaic ships its own anywidget (the ``mosaic-widget`` package), and Panel's
`AnyWidgetComponent` is meant to run anywidget ESM as-is. That does not work
here: Mosaic's widget drives all of its data loading over anywidget's custom
message channel, and Panel's anywidget shim implements neither direction of
that channel. `AnyWidgetModelAdapter` exposes no ``send``, so the browser
cannot ask for data, and its ``msg:custom`` callback is invoked with the
message alone, so the Arrow buffers Mosaic replies with never arrive. Panel's
own `JSComponent` model does support both directions, via ``send_msg`` and
``msg:custom``, which is why the ESM module here is a small Panel-native
reimplementation of Mosaic's render loop rather than a wrapper around theirs.

The one thing `JSComponent` still cannot do is send binary buffers, so Arrow
IPC results are base64-encoded into the JSON message. Mosaic only transfers
query *results* (typically aggregates), not raw tables, so these payloads stay
small; if Panel grows a binary message channel this can switch to it without
any change to the spec format or the Python API.
"""
from __future__ import annotations

import base64
import logging

from typing import TYPE_CHECKING, Any

import duckdb
import param  # type: ignore
import pyarrow as pa

from panel.custom import JSComponent

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame

logger = logging.getLogger(__name__)

# Pinned so the spec grammar the LLM/user writes and the runtime that parses it
# cannot drift apart across releases.
MOSAIC_VERSION = "0.31.0"
FLECHETTE_VERSION = "2.5.0"

CSS = """
.mosaic-pane .input {
  margin-right: 1em;
}
.mosaic-pane .input > * {
  vertical-align: middle;
}
.mosaic-pane .mosaic-pane-error {
  white-space: pre-wrap;
  margin: 0;
  padding: 0.5em;
  color: #b00020;
  font-size: 12px;
}
.mosaic-pane table {
  position: relative;
  table-layout: fixed;
  border-collapse: separate;
  border-spacing: 0;
  font-variant-numeric: tabular-nums;
  box-sizing: border-box;
  max-width: initial;
  min-height: 33px;
  margin: 0;
  width: 100%;
  font-size: 13px;
  line-height: 15.6px;
}
.mosaic-pane thead tr th {
  position: sticky;
  top: 0;
  background: #fff;
  cursor: ns-resize;
  border-bottom: solid 1px #ccc;
}
.mosaic-pane tbody tr:hover {
  background: #eef;
}
.mosaic-pane th {
  color: #111;
  text-align: left;
  vertical-align: bottom;
}
.mosaic-pane td,
.mosaic-pane th {
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
  padding: 3px 6.5px 3px 0;
}
.mosaic-pane tbody tr:first-child td {
  padding-top: 4px;
}
.mosaic-pane td,
.mosaic-pane tr:not(:last-child) th {
  border-bottom: solid 1px #eee;
}
.mosaic-pane td {
  color: #444;
  vertical-align: top;
}
"""


class Mosaic(JSComponent):
    """
    The `Mosaic` pane renders a declarative Mosaic (vgplot) specification.

    Data stays in DuckDB; the browser issues SQL through this pane and receives
    only the results, which is what lets Mosaic cross-filter and brush over
    datasets too large to embed in a chart spec.

    Reference: https://idl.uw.edu/mosaic/

    :Example:

    >>> Mosaic(
    ...     {"plot": [{"mark": "dot", "data": {"from": "penguins"},
    ...                "x": "bill_length", "y": "bill_depth"}]},
    ...     data={"penguins": df},
    ... )
    """

    params = param.Dict(default={}, doc="""
        The chart's live Param/Selection state, keyed by name. Each entry has
        the Param's `value`, plus a SQL `predicate` when it is a Selection.
        Updated by the browser as the user brushes and filters.""")

    preagg_schema = param.String(default="", doc="""
        Schema in which Mosaic may materialize pre-aggregated views to speed up
        interaction. Empty leaves Mosaic's own default in place.""")

    spec = param.Dict(default={}, doc="""
        The declarative mosaic-spec. Marks reference tables by name via
        `data: {from: <table>}`; the tables come from `con`/`data`.""")

    _esm = "mosaic.js"

    _importmap = {
        "imports": {
            # A single entry point, so mosaic-spec, vgplot and mosaic-core all
            # resolve within one module graph. Importing mosaic-core separately
            # would give the pane a different Coordinator than the one the spec
            # instantiates its plots against, and the connector set below would
            # never be reached.
            "@uwdata/mosaic-spec": f"https://esm.sh/@uwdata/mosaic-spec@{MOSAIC_VERSION}",
            "@uwdata/flechette": f"https://esm.sh/@uwdata/flechette@{FLECHETTE_VERSION}",
        }
    }

    _stylesheets = [CSS]

    def __init__(
        self,
        spec: dict[str, Any] | None = None,
        con: duckdb.DuckDBPyConnection | None = None,
        data: dict[str, IntoFrame] | None = None,
        **params,
    ):
        """
        Arguments
        ---------
        spec: dict
            The mosaic-spec to render.
        con: duckdb.DuckDBPyConnection
            Connection to query. Defaults to a fresh in-memory connection.
        data: dict
            Frames to register as virtual tables on the connection, keyed by
            the table name the spec refers to.
        """
        if spec is not None:
            params["spec"] = spec
        super().__init__(**params)
        self._con = duckdb.connect() if con is None else con
        for name, frame in (data or {}).items():
            self._con.register(name, frame)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The DuckDB connection this pane queries."""
        return self._con

    def _handle_msg(self, msg: Any) -> None:
        """
        Answer a query from Mosaic's browser runtime.

        Every reply carries the request's `uuid` so the browser can settle the
        right promise, including on failure -- an unanswered query leaves the
        chart waiting forever rather than reporting anything.
        """
        uuid = msg.get("uuid")
        command = msg.get("type")
        sql = msg.get("sql")
        try:
            if command == "arrow":
                self._send_msg({"type": "arrow", "uuid": uuid, "data": self._query_arrow(sql)})
            elif command == "exec":
                self._con.execute(sql)
                self._send_msg({"type": "exec", "uuid": uuid})
            elif command == "json":
                result = self._con.query(sql).df()
                self._send_msg({
                    "type": "json", "uuid": uuid, "result": result.to_dict(orient="records")
                })
            else:
                raise ValueError(f"Unknown Mosaic query type {command!r}.")
        except Exception as e:
            logger.exception("Mosaic query failed: %s", sql)
            self._send_msg({"error": str(e), "uuid": uuid})

    def _query_arrow(self, sql: str) -> str:
        """Run `sql` and return the result as a base64-encoded Arrow IPC stream."""
        # Depending on the DuckDB version `.arrow()` hands back either a Table
        # or a RecordBatchReader; writing batch by batch covers both and keeps
        # a large result from being materialized twice.
        result = self._con.query(sql).arrow()
        batches = result.to_batches() if isinstance(result, pa.Table) else result
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, result.schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
        return base64.b64encode(sink.getvalue().to_pybytes()).decode("ascii")
