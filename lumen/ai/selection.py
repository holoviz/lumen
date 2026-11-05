"""
Clicks and box selections on a chart as rows of its data, and a box to ask
about them. Answer cards in a notebook and the explorations of the chat UI
share it.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import holoviews as hv
import pandas as pd
import panel as pn
import param

from panel.pane import Markdown, Vega
from panel_material_ui import (
    Alert, Button, Column, Row,
)

from ..util import as_pandas
from ..views import View
from .sharing import shares_values, sharing
from .utils import format_error

#: The Vega-Lite selection a click on a chart reports
CLICK = "lumen_click"

#: Rows of a selection a follow-up prompt shows
ABOUT_ROWS = 20


def with_click(spec: dict) -> dict:
    """
    The spec with a point selection on the fields that tell its marks apart,
    so a click reports the clicked mark's keys. Aggregated, binned and
    time-unit fields are left out, since no row of the data holds them as
    drawn. A faceted or concatenated spec is returned as it is.
    """
    if any(key in spec for key in ("facet", "repeat", "concat", "hconcat", "vconcat")):
        return spec
    target = spec["layer"][0] if "layer" in spec else spec
    encoding = target.get("encoding") or spec.get("encoding") or {}
    fields = sorted({
        channel["field"] for channel in encoding.values()
        if isinstance(channel, dict) and isinstance(channel.get("field"), str)
        and not any(key in channel for key in ("aggregate", "bin", "timeUnit"))
    })
    if not fields or any(p.get("name") == CLICK for p in target.get("params", [])):
        return spec
    target = {**target, "params": [*target.get("params", []), {"name": CLICK, "select": {"type": "point", "fields": fields}}]}
    if "opacity" not in encoding:
        # The clicked mark stays solid and the others fade; with nothing
        # clicked the selection is empty and every mark is solid
        target["encoding"] = {**encoding, "opacity": {"condition": {"param": CLICK, "value": 1}, "value": 0.35}}
    return {**spec, "layer": [target, *spec["layer"][1:]]} if "layer" in spec else target


def about(question: str, rows: pd.DataFrame) -> str:
    """The follow-up message: the question, the rows it is about, and how to use them."""
    if sharing.share != "sample":
        raise ValueError(
            f"Asking about rows sends them to the LLM, which share={sharing.share!r} does not allow. "
            "Use lumen.ai.configure(share='sample') to ask about them."
        )
    columns = [c for c in rows.columns if shares_values(str(c), rows[c].head(ABOUT_ROWS).tolist())]
    shown = "" if len(rows) <= ABOUT_ROWS else f" (the first {ABOUT_ROWS} shown)"
    return (
        f"{question}\n\nThis is about {len(rows)} row{'s' if len(rows) != 1 else ''} of the previous answer's data"
        f"{shown}:\n{rows[columns].head(ABOUT_ROWS).to_csv(index=False)}\n"
        "If these rows hold what is asked, answer from them without a new query. "
        "Otherwise query the data, filtering on these rows' values."
    )


def _matching(df: pd.DataFrame, keys: dict[str, Any]) -> pd.DataFrame | None:
    """The rows whose values match the keys a click on a Vega-Lite chart reported."""
    keys = {key: value for key, value in keys.items() if key in df.columns}
    if not keys:
        return None
    mask = pd.Series(True, index=df.index)
    for key, value in keys.items():
        column = df[key]
        if pd.api.types.is_datetime64_any_dtype(column) and isinstance(value, (int, float)):
            # Vega-Lite reports a date as milliseconds since the epoch
            mask &= column == pd.to_datetime(value, unit="ms")
        elif pd.api.types.is_numeric_dtype(column) and isinstance(value, (int, float)):
            # As numbers: 58 from the JSON is 58.0 in the column, though not as text
            mask &= column == value
        else:
            mask &= column.astype(str) == str(value)
    return df[mask]


def watch_clicks(view: View, on_rows: Callable[[pd.DataFrame | None], None]) -> list:
    """
    Report clicks and box selections on the chart a view shows as rows of
    its data: Vega-Lite through a point selection added to the drawn spec,
    hvPlot through Selection1D for points, bars and shapes, and Tap for
    grids, where the nearest cell is the row. Returns the streams, which
    have to be kept alive.
    """
    view.panel  # noqa: B018  builds the pane the view shows
    pane = view._panel
    if isinstance(pane, Vega):
        def add_click(*events):
            # Drawn spec only, so the view's spec stays what the user edits;
            # added again whenever the view redraws, e.g. on a filter
            if (spec := with_click(pane.object)) is not pane.object:
                pane.object = spec

        def on_click(event):
            # One click reports each field as its own {field: value}
            keys = {k: v for item in event.new or [] if isinstance(item, dict) for k, v in item.items()}
            on_rows(_matching(as_pandas(view.pipeline.data), keys) if keys else None)

        def watch_selection(*events):
            if CLICK in pane._selections:
                pane.selection.param.watch(on_click, CLICK)

        add_click()
        watch_selection()
        pane.param.watch(add_click, "object")
        pane.param.watch(watch_selection, "selection")
        return []
    if not isinstance(pane, pn.pane.HoloViews):
        return []
    streams: list = []

    def attach(obj):
        # The view draws its plot again on render, so each new plot gets its own stream
        if (stream := _hv_stream(obj, view, on_rows)) is not None:
            streams.append(stream)

    attach(pane.object)
    pane.param.watch(lambda event: attach(event.new), "object")
    return streams


def _hv_stream(element: Any, view: View, on_rows: Callable[[pd.DataFrame | None], None]) -> Any:
    """A stream reporting clicks on a HoloViews plot as rows, or None for a plot it cannot read."""
    if isinstance(element, hv.Overlay):
        # The data layer, not the basemap tiles under it
        layers = [layer for layer in element if not isinstance(layer, hv.Tiles)]
        element = layers[-1] if layers else element
    if isinstance(element, (hv.Points, hv.Scatter, hv.Bars, hv.Polygons, hv.Path)):
        element.opts(tools=["tap", "box_select"])
        stream = hv.streams.Selection1D(source=element)
        data = element.dframe()
        stream.add_subscriber(lambda index: on_rows(data.iloc[index] if index else None))
        return stream
    if isinstance(element, (hv.QuadMesh, hv.Image, hv.DynamicMap)):
        stream = hv.streams.Tap(source=element)
        xcol, ycol = (dim.name for dim in element.kdims[:2]) if len(element.kdims) >= 2 else (view.x, view.y)

        def on_tap(x=None, y=None):
            df = as_pandas(view.pipeline.data)
            if x is None or y is None or xcol not in df or ycol not in df:
                return
            nearest = ((df[xcol] - x) ** 2 + (df[ycol] - y) ** 2).idxmin()
            on_rows(df.loc[[nearest]])
        stream.add_subscriber(on_tap)
        return stream
    return None


def ask_box(owner: param.Parameterized, ask: Callable[[str], Awaitable[Any]]) -> Column:
    """
    What is selected on owner.selection and a box to ask about it, hidden
    until something is. ask gets the question and returns what to show
    under the box, or None when it shows the answer elsewhere.
    """
    selected = Markdown(sizing_mode="stretch_width")
    question = pn.widgets.TextInput(placeholder="Ask about this, e.g. what is this?", sizing_mode="stretch_width")
    button = Button(label="Ask", variant="contained")
    answers = Column(sizing_mode="stretch_width")
    box = Column(selected, Row(question, button, sizing_mode="stretch_width"), answers, visible=False, sizing_mode="stretch_width")

    def on_selection(event):
        rows = event.new
        box.visible = rows is not None and not rows.empty
        if box.visible:
            first = ", ".join(f"{k}: {v}" for k, v in rows.iloc[0].items())
            selected.object = f"**Selected:** {first}" if len(rows) == 1 else f"**Selected:** {len(rows)} rows"

    async def on_ask(event):
        if not question.value_input.strip():
            return
        button.loading = True
        try:
            shown = await ask(question.value_input.strip())
        except Exception as e:
            answers.append(Alert(object=format_error(e), title="Lumen could not answer", severity="error"))
        else:
            if shown is not None:
                answers.append(shown)
            question.value = question.value_input = ""
        finally:
            button.loading = False

    owner.param.watch(on_selection, "selection")
    button.on_click(on_ask)
    return box


class Selected(param.Parameterized):
    """The rows selected on one chart, for charts that are not an Answer's."""

    selection = param.DataFrame(default=None, allow_None=True)
