"""Panel application served by ``panel serve`` for a live Lumen dashboard.

It reads a saved lumen-mcp session (path in the ``LUMEN_MCP_DASHBOARD_SESSION`` env var),
reconstructs each chart as an interactive Vega pane and each workspace table as an interactive
Tabulator, and serves them. Running in a subprocess keeps the live server off the MCP server's event
loop while still using the session's real Lumen views and DuckDB workspace.
"""

from __future__ import annotations

import json
import os

import panel as pn

pn.extension("vega", "tabulator")


def _build() -> pn.viewable.Viewable:
    from lumen.pipeline import Pipeline
    from lumen.views import VegaLiteView

    from lumen_mcp import sources
    from lumen_mcp.session import session

    path = os.environ["LUMEN_MCP_DASHBOARD_SESSION"]
    sources.connect_source(path)
    source = session.source

    items: list = [
        pn.pane.Markdown("# Lumen dashboard"),
        pn.pane.Markdown("Interactive charts (pan / zoom / tooltip) and tables (sort / filter), live."),
    ]

    charts_path = f"{path}.charts.json"
    if os.path.exists(charts_path):
        with open(charts_path) as fh:
            charts = json.load(fh)
        for cid, meta in charts.items():
            # Give charts concrete dimensions; "container" sizing collapses to 0px in a Column.
            spec = dict(meta["spec"], width=700, height=350)
            view = VegaLiteView(pipeline=Pipeline(source=source, table=meta["table"]), spec=spec)
            items.append(pn.pane.Markdown(f"### chart `{cid}`  (table `{meta['table']}`)"))
            items.append(view.get_panel())

    for table in source.get_tables():
        items.append(pn.pane.Markdown(f"### table `{table}`"))
        items.append(pn.widgets.Tabulator(
            source.get(table), header_filters=True, pagination="local", page_size=8,
            disabled=True, sizing_mode="stretch_width", height=260,
        ))

    return pn.Column(*items, sizing_mode="stretch_width")


try:
    _layout = _build()
except Exception:  # surface the error in the page rather than serving a blank
    import traceback
    _layout = pn.pane.Markdown(f"### Dashboard error\n```\n{traceback.format_exc()}\n```")

_layout.servable(title="Lumen dashboard")
