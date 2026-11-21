"""
An answer card as an anywidget, for notebooks Panel cannot draw in, such as
marimo. Its chart stays interactive: Vega-Lite through vega-embed, with a
click selecting rows as on the Panel card, and hvPlot through standalone
Bokeh, which pans, zooms and shows tooltips without a server.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anywidget
import holoviews as hv
import traitlets

from bokeh import __version__ as bokeh_version
from bokeh.embed import json_item

from ..util import as_pandas
from ..views import VegaLiteView, hvPlotView
from .selection import CLICK, _matching, with_click

if TYPE_CHECKING:
    from .answer import Answer

#: Rows of an answer without a chart the card shows
TABLE_ROWS = 20

_ESM = """
import embed from "https://esm.sh/vega-embed@6";

const BOKEH = ["", "-gl", "-widgets", "-tables"];

function loadBokeh(version) {
  if (window.Bokeh && window.Bokeh.version === version) return Promise.resolve();
  // One after another: the extensions need the core library loaded first
  return BOKEH.reduce((loaded, part) => loaded.then(() => new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `https://cdn.bokeh.org/bokeh/release/bokeh${part}-${version}.min.js`;
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  })), Promise.resolve());
}

async function render({ model, el }) {
  const title = document.createElement("h3");
  title.textContent = model.get("question");
  // Notebook stylesheets can reset headings, so the card sets its own
  title.style.cssText = "font-size: 1.15em; font-weight: 600; margin: 0 0 0.5em";
  const chart = document.createElement("div");
  const body = document.createElement("div");
  body.innerHTML = model.get("card");
  el.append(title, chart, body);
  const vega = model.get("vega");
  const bokeh = model.get("bokeh");
  if (vega) {
    const { view } = await embed(chart, vega, { actions: false });
    if (model.get("clickable")) {
      view.addSignalListener(model.get("click_signal"), (name, value) => {
        model.set("clicked", value || {});
        model.save_changes();
      });
    }
  } else if (bokeh) {
    await loadBokeh(model.get("bokeh_version"));
    window.Bokeh.embed.embed_item(bokeh, chart);
  }
}

export default { render };
"""


class AnswerWidget(anywidget.AnyWidget):
    """The card of one Answer: its question, chart, text, SQL and steps."""

    _esm = _ESM

    question = traitlets.Unicode().tag(sync=True)

    card = traitlets.Unicode(help="The card below the chart, as HTML").tag(sync=True)

    vega = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)

    bokeh = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)

    bokeh_version = traitlets.Unicode(bokeh_version).tag(sync=True)

    clickable = traitlets.Bool(False).tag(sync=True)

    click_signal = traitlets.Unicode(CLICK).tag(sync=True)

    clicked = traitlets.Dict(help="The keys of the mark last clicked, empty when none").tag(sync=True)


def _keys(value: dict[str, Any]) -> dict[str, Any]:
    """A Vega-Lite point selection's value as one key per field; a click selects one mark."""
    return {field: v[0] if isinstance(v, list) and len(v) == 1 else v for field, v in value.items() if not field.startswith("vlPoint")}


def answer_widget(answer: Answer) -> AnswerWidget:
    """The answer as an AnswerWidget, whose clicks set answer.selection."""
    card = answer._static_bundle(None)["text/html"]
    vega = bokeh_item = None
    if isinstance(answer.view, VegaLiteView):
        vega = answer._vega_lite_spec()
    elif isinstance(answer.view, hvPlotView):
        plot = answer.view.get_panel().object
        bokeh_item = json_item(hv.render(plot, backend="bokeh"))
    elif answer.gdf is not None:
        vega = answer._geoshape_spec()
    if vega is None and bokeh_item is None and answer.pipeline is not None:
        # No chart, so the rows themselves
        rows = as_pandas(answer.pipeline.data)
        note = f"<p><small>The first {TABLE_ROWS} of {len(rows):,} rows.</small></p>" if len(rows) > TABLE_ROWS else ""
        card = rows.head(TABLE_ROWS).to_html(index=False, border=0) + note + card
    clicked = vega is not None and isinstance(answer.view, VegaLiteView)
    widget = AnswerWidget(
        question=answer.question, card=card, vega=with_click(vega) if clicked else vega,
        bokeh=bokeh_item, clickable=clicked,
    )

    pipeline = answer.view.pipeline if answer.view is not None else answer.pipeline

    def on_click(change):
        keys = _keys(change.new)
        answer.selection = _matching(as_pandas(pipeline.data), keys) if keys else None

    widget.observe(on_click, "clicked")
    return widget
