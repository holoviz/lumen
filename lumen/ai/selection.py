"""
Clicks, box selections and drawn areas on a chart as rows of its data, and
a box to ask about them. Answer cards in a notebook and the explorations of
the chat UI share it.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import shapely

from panel.pane import Markdown, Vega
from panel_material_ui import (
    Alert, Button, Column, Row,
)

from ..util import as_pandas, geometry_columns, is_geodataframe
from ..views import View
from ..views.base import hvPlotBaseView
from .sharing import shares_values, sharing
from .utils import format_error

#: The Vega-Lite selection a click on a chart reports
CLICK = "lumen_click"

#: Rows of a selection a follow-up prompt shows
ABOUT_ROWS = 20

#: Characters of one cell a follow-up prompt shows
ABOUT_CELL = 200

#: HoloViews elements whose glyphs a click or box selects
_SELECTABLE = (hv.Points, hv.Scatter, hv.Bars, hv.Polygons, hv.Path)

#: The group of the empty layer an area is drawn on, which holds no data
DRAW_GROUP = "LumenDraw"

#: How to draw an area, which Bokeh's polygon tool does not show
DRAW_HINT = "Draw an area to ask about it: pick the polygon tool, press and hold to start, click to add corners, press and hold to finish."

#: Column names read as longitude and latitude
_LON, _LAT = ("lon", "longitude", "lng"), ("lat", "latitude")


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


def _shareable(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows as a prompt may show them: no geometry, no withheld columns, long cells cut."""
    if sharing.share != "sample":
        raise ValueError(
            f"Asking about rows sends them to the LLM, which share={sharing.share!r} does not allow. "
            "Use lumen.ai.configure(share='sample') to ask about them."
        )
    # A shape's coordinates say little to the LLM and can run to megabytes
    geometry = set(geometry_columns(rows))
    columns = [
        c for c in rows.columns
        if c not in geometry and shares_values(str(c), rows[c].head(ABOUT_ROWS).tolist())
    ]
    return rows[columns]


def _csv(rows: pd.DataFrame) -> str:
    shown = rows.head(ABOUT_ROWS).map(
        lambda v: v[:ABOUT_CELL] + "..." if isinstance(v, str) and len(v) > ABOUT_CELL else v
    )
    # Same text on every platform, so a recorded prompt replays everywhere
    return shown.to_csv(index=False, lineterminator="\n")


def about(question: str, rows: pd.DataFrame) -> str:
    """The follow-up message: the question, the rows it is about, and how to use them."""
    shown = _shareable(rows)
    note = "" if len(rows) <= ABOUT_ROWS else f" (the first {ABOUT_ROWS} shown)"
    return (
        f"{question}\n\nThis is about {len(rows)} row{'s' if len(rows) != 1 else ''} of the previous answer's data"
        f"{note}:\n{_csv(shown)}\n"
        "If these rows hold what is asked, answer from them without a new query. "
        "Otherwise query the data, filtering on these rows' values."
    )


def about_region(question: str, region: Any, inside: pd.DataFrame, outside: pd.DataFrame) -> str:
    """
    The follow-up message for an area drawn on a map: the question, the
    area, and its rows beside the rest, summarised so the LLM can compare
    them without a query, and the predicate to query with when it cannot.
    """
    inside, outside = _shareable(inside), _shareable(outside)
    numbers = inside.select_dtypes("number").columns.intersection(outside.columns)
    stats = ""
    if len(numbers):
        summary = pd.concat(
            {side: rows[numbers].agg(["mean", "min", "max"]).T for side, rows in (("inside", inside), ("outside", outside))},
            axis=1,
        ).round(4)
        summary.columns = [f"{side} {stat}" for side, stat in summary.columns]
        stats = f"\nThe numeric columns inside and outside it:\n{summary.to_csv(lineterminator=chr(10))}"
    note = "" if len(inside) <= ABOUT_ROWS else f" (the first {ABOUT_ROWS} shown)"
    return (
        f"{question}\n\nThis is about an area drawn on the previous answer's map, in longitude and latitude:\n"
        f"{shapely.to_wkt(region, rounding_precision=4)}\n"
        f"{len(inside)} of the data's {len(inside) + len(outside)} rows fall inside it{note}:\n{_csv(inside)}{stats}\n"
        "If this holds what is asked, answer from it without a new query. Otherwise query the data, keeping "
        f"the rows inside the area with ST_Within(ST_Point(lon, lat), ST_GeomFromText('{shapely.to_wkt(region, rounding_precision=6)}')) "
        "or ST_Intersects on a geometry column."
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


def watch_clicks(
    view: View, on_rows: Callable[[pd.DataFrame | None], None],
    on_region: Callable[[Any], None] | None = None,
) -> list:
    """
    Report clicks and box selections on the chart a view shows as rows of
    its data: Vega-Lite through a point selection added to the drawn spec,
    hvPlot through Selection1D for points, bars and shapes, and Tap for
    grids, where the nearest cell is the row. With on_region, a map of
    longitude and latitude also gets a polygon tool, and on_region gets
    the drawn area as a shapely Polygon, or None once it is cleared.
    Returns the streams, which have to be kept alive.
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
        if on_region is not None and _drawable(view, obj) and _draw_layer(obj) is None:
            # Showing the plot with the drawing layer on top attaches again
            pane.object = obj * _empty_area()
            return
        # The view draws its plot again on render, so each new plot gets its own stream
        streams.extend(_hv_streams(obj, view, on_rows))
        if on_region is not None and (layer := _draw_layer(obj)) is not None:
            streams.append(_draw_stream(layer, on_region))

    # Watched first, so a plot shown with the drawing layer added attaches its streams
    pane.param.watch(lambda event: attach(event.new), "object")
    attach(pane.object)
    return streams


def _layers(element: Any) -> list[tuple[Any, pd.DataFrame]]:
    """The selectable data layers of a plot, each with its rows, the basemap tiles left out."""
    if isinstance(element, hv.NdOverlay):
        # A by= chart: one layer per category, whose value is the overlay's key
        names = [dim.name for dim in element.kdims]
        return [
            (layer, _rows(layer).assign(**dict(zip(names, key if isinstance(key, tuple) else (key,), strict=True))))
            for key, layer in element.items() if isinstance(layer, _SELECTABLE)
        ]
    if isinstance(element, hv.Overlay):
        return [layer for item in element if not isinstance(item, hv.Tiles) for layer in _layers(item)]
    return [(element, _rows(element))] if isinstance(element, _SELECTABLE) and element.group != DRAW_GROUP else []


def _rows(element: Any) -> pd.DataFrame:
    """One row per glyph, as a selection's indices count them."""
    if element.interface.datatype == "geodataframe":
        # One row per shape already; dframe() expands shapes into vertices
        # and fails on a MultiPolygon, whose parts have different lengths
        return pd.DataFrame(element.data)
    return element.dframe()


def _drawable(view: View, element: Any) -> bool:
    """Whether a plot is a map of longitude and latitude an area can be drawn on."""
    return isinstance(view, hvPlotBaseView) and view.geo and isinstance(element, (hv.Element, hv.Overlay))


def _empty_area() -> hv.Polygons:
    """The layer a polygon is drawn on, in longitude and latitude however the map is projected."""
    try:
        # Deferred: geoviews is optional; with it the drawn vertices come back
        # in the layer's CRS even on a Web Mercator basemap
        import cartopy.crs as ccrs
        import geoviews as gv
    except ImportError:
        return hv.Polygons([], group=DRAW_GROUP)
    return gv.Polygons([], crs=ccrs.PlateCarree(), group=DRAW_GROUP)


def _draw_layer(element: Any) -> hv.Polygons | None:
    items = element.values() if isinstance(element, hv.Overlay) else [element]
    return next((item for item in items if isinstance(item, hv.Polygons) and item.group == DRAW_GROUP), None)


def region(data: dict) -> Any:
    """The last polygon a PolyDraw stream holds, longitudes in -180 to 180, or None for fewer than three vertices."""
    xs, ys = (data or {}).get("xs") or [], (data or {}).get("ys") or []
    if not xs or len(xs[-1]) < 3:
        return None
    lons = (np.asarray(xs[-1], dtype=float) + 180) % 360 - 180
    return shapely.Polygon(zip(lons, ys[-1], strict=True))


def split_by_region(df: pd.DataFrame, area: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rows inside an area and the rest: shapes that touch it, or points whose longitude and latitude fall in it."""
    if is_geodataframe(df):
        shapes = df.geometry
        if shapes.crs is not None and not shapes.crs.is_geographic:
            # The area is in longitude and latitude; a projected frame is compared in them too
            shapes = shapes.to_crs(4326)
        mask = np.asarray(shapes.intersects(area))
    else:
        lon = next((c for c in df.columns if str(c).lower() in _LON), None)
        lat = next((c for c in df.columns if str(c).lower() in _LAT), None)
        if lon is None or lat is None:
            raise ValueError("The data has no longitude and latitude columns or geometry to place in the drawn area.")
        lons = (df[lon].astype(float) + 180) % 360 - 180
        mask = shapely.contains_xy(area, lons.to_numpy(), df[lat].astype(float).to_numpy())
    return df[mask], df[~mask]


def _draw_stream(layer: hv.Polygons, on_region: Callable[[Any], None]) -> hv.streams.PolyDraw:
    layer.opts(fill_alpha=0.15, line_width=2)
    # Bokeh 3.4+ starts and ends a polygon on a press and hold, not a double tap
    stream = hv.streams.PolyDraw(
        source=layer, num_objects=1, drag=False,
        tooltip=DRAW_HINT,
    )
    stream.add_subscriber(lambda data: on_region(region(data)))
    return stream


def _numeric(column: pd.Series, value: Any) -> tuple[pd.Series, float]:
    """A column and a value on it as plain numbers, so distances can be taken on dates too."""
    if pd.api.types.is_datetime64_any_dtype(column):
        # Bokeh reports a date axis in milliseconds since the epoch
        stamp = pd.to_datetime(value, unit="ms") if isinstance(value, (int, float)) else pd.Timestamp(value)
        return column.astype("int64").astype(float), float(stamp.value)
    return column.astype(float), float(value)


def _nearest(df: pd.DataFrame, xcol: str, ycol: str, x: Any, y: Any) -> Any:
    """The index of the row nearest a point, with each axis scaled by its range."""
    distance = 0
    for col, value in ((xcol, x), (ycol, y)):
        column, value = _numeric(df[col], value)
        span = (column.max() - column.min()) or 1.0
        distance = distance + ((column - value) / span) ** 2
    return distance.idxmin()


_GRIDS = (hv.QuadMesh, hv.Image, hv.DynamicMap)


def _grid_element(element: Any) -> Any:
    """The grid, image or density map a plot draws, which a click reads by position, or None."""
    if isinstance(element, _GRIDS):
        return element
    if isinstance(element, (hv.Overlay, hv.NdOverlay)):
        # The last one, as a legend's points are drawn after the image they describe
        grids = [item for item in element.values() if isinstance(item, _GRIDS)]
        return grids[-1] if grids else None
    return None


def _tap_stream(element: Any, view: View, on_rows: Callable[[pd.DataFrame | None], None]) -> hv.streams.Tap:
    stream = hv.streams.Tap(source=element)
    # A DynamicMap's kdims are its slider dimensions, not its axes
    if view.x and view.y:
        xcol, ycol = view.x, view.y
    else:
        xcol, ycol = (dim.name for dim in element.kdims[:2])

    def on_tap(x=None, y=None):
        df = as_pandas(view.pipeline.data)
        if x is None or y is None or xcol not in df or ycol not in df:
            return
        # The plot on screen, which the slider steps; the stream's source can be a copy of it
        shown = view._panel.object
        if isinstance(shown, hv.DynamicMap) and shown.kdims and shown.current_key is not None:
            # Only the step the slider shows, not every step's cells
            key = shown.current_key if isinstance(shown.current_key, tuple) else (shown.current_key,)
            for dim, value in zip(shown.kdims, key, strict=False):
                if dim.name in df:
                    df = df[df[dim.name] == value]
        on_rows(df.loc[[_nearest(df, xcol, ycol, x, y)]])
    stream.add_subscriber(on_tap)
    return stream


def _hv_streams(element: Any, view: View, on_rows: Callable[[pd.DataFrame | None], None]) -> list:
    """The streams reporting clicks on a HoloViews plot as rows, none for a plot it cannot read."""
    # A grid or density image first: a legend's points beside it are not the data
    if (grid := _grid_element(element)) is not None:
        return [_tap_stream(grid, view, on_rows)]
    if layers := _layers(element):
        # A box can cover several layers, so each layer's pick is kept and all are reported
        picked: dict[int, pd.DataFrame] = {}

        def report(i, data, index):
            if index:
                picked[i] = data.iloc[index]
            else:
                picked.pop(i, None)
            on_rows(pd.concat(picked.values()) if picked else None)

        streams = []
        for i, (layer, data) in enumerate(layers):
            layer.opts(tools=["tap", "box_select"])
            stream = hv.streams.Selection1D(source=layer)
            stream.add_subscriber(lambda index, i=i, data=data: report(i, data, index))
            streams.append(stream)
        return streams
    return []


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
            # Floats to a readable precision; the question gets the full values
            first = ", ".join(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}" for k, v in rows.iloc[0].items())
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
