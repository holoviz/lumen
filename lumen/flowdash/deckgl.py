"""
A deck.gl map of a Lumen pipeline as a FlowDash node.

The spec is deck.gl's JSON format (as pydeck writes it) without data: every
layer draws the pipeline's rows, and column-based props are deck.gl
expressions such as ``"getPosition": "@@=[lon, lat]"``. ``check_deckgl``
refuses a spec with unknown layer types, layer data or columns the pipeline
does not have before it is drawn. A spec that leaves out the basemap gets a
light or dark one following the page, one that leaves out the initial view
state is fitted to the rows, and layers without a colour take the
dashboard's colour scheme.
"""
from __future__ import annotations

import copy
import difflib
import json
import math
import re
import weakref

from typing import Any

import panel as pn

from .nodes import (
    COMPONENTS, LumenViewNode, fetch, register_view, row_ceiling, spec_param,
)

DECKGL_ID = "Lumen/deckgl"

# The layers that draw one datum per row, which a view of a pipeline can feed.
# Tile, bitmap and mesh layers load their own data from URLs and are left out.
LAYER_TYPES = frozenset({
    "ArcLayer", "ColumnLayer", "ContourLayer", "GeoJsonLayer", "GeohashLayer",
    "GreatCircleLayer", "GridCellLayer", "GridLayer", "H3ClusterLayer",
    "H3HexagonLayer", "HeatmapLayer", "HexagonLayer", "LineLayer", "PathLayer",
    "PointCloudLayer", "PolygonLayer", "QuadkeyLayer", "S2Layer",
    "ScatterplotLayer", "ScreenGridLayer", "SolidPolygonLayer", "TextLayer",
    "TripsLayer",
})

# Accessors that take positions, which a list of column names cannot give.
POSITION_PROPS = ("getPosition", "getSourcePosition", "getTargetPosition")

# The colour accessors a layer draws black or grey without, given the dashboard's colour instead.
COLOR_PROPS = {
    "ArcLayer": ("getSourceColor", "getTargetColor"),
    "GreatCircleLayer": ("getSourceColor", "getTargetColor"),
    "LineLayer": ("getColor",),
    "PathLayer": ("getColor",),
    "PointCloudLayer": ("getColor",),
    "TripsLayer": ("getColor",),
    **dict.fromkeys((
        "ColumnLayer", "GeoJsonLayer", "GeohashLayer", "GridCellLayer", "H3HexagonLayer",
        "H3ClusterLayer", "PolygonLayer", "QuadkeyLayer", "S2Layer", "ScatterplotLayer",
        "SolidPolygonLayer",
    ), ("getFillColor",)),
}

# A colour of each of the dashboard's Vega colour schemes, dark enough for points and
# arcs on either basemap; the first one is Vega's default, used without a scheme.
SCHEME_COLORS = {
    "tableau10": [76, 120, 168], "category10": [31, 119, 180], "set2": [102, 194, 165],
    "dark2": [27, 158, 119], "paired": [31, 120, 180], "observable10": [66, 105, 208],
    "blues": [49, 130, 189], "greens": [49, 163, 84], "teals": [42, 139, 150],
    "oranges": [230, 85, 13], "purples": [117, 107, 177], "viridis": [33, 145, 140],
    "magma": [183, 55, 121],
}

# CARTO's keyless basemaps, which a spec without a mapStyle gets by the page's theme.
BASEMAPS = {
    "light": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
}

EXAMPLE = {
    "layers": [{
        "@@type": "ScatterplotLayer",
        "getPosition": "@@=[longitude, latitude]",
        "getRadius": "@@=capacity * 10",
        "radiusMinPixels": 2,
        "pickable": True,
    }],
}

_EXPRESSION = "@@="
_STRINGS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
# A name not read off another value, with the member read off it and whether it is called.
_NAMES = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)(?:\s*\.\s*([A-Za-z_$][\w$]*))?(\s*\()?")
_GLOBALS = frozenset({
    "true", "false", "null", "undefined", "NaN", "Infinity", "Math", "Number",
    "String", "Boolean", "Date", "JSON", "this", "in", "typeof",
})
# What a GeoJsonLayer accessor may read off its datum, a GeoJSON Feature.
_FEATURE_KEYS = frozenset({"properties", "geometry", "type", "id"})


def _layer_type(layer: dict[str, Any]) -> str | None:
    kind = layer.get("@@type")
    return kind if isinstance(kind, str) else None


def _layer_name(index: int, layer: dict[str, Any]) -> str:
    return f"Layer {layer['id']!r}" if isinstance(layer.get("id"), str) else f"Layer {index + 1}"


def _expressions(value: Any) -> list[str]:
    """The deck.gl expressions (``"@@=..."`` strings) anywhere in ``value``."""
    if isinstance(value, str):
        return [value[len(_EXPRESSION):]] if value.startswith(_EXPRESSION) else []
    if isinstance(value, dict):
        return [e for item in value.values() for e in _expressions(item)]
    if isinstance(value, list):
        return [e for item in value for e in _expressions(item)]
    return []


def expression_names(expression: str, feature: bool = False) -> tuple[set[str], set[str]]:
    """
    The columns a deck.gl expression reads, and the names it reads that are
    no column of a ``feature`` (a GeoJSON Feature, whose columns are its
    ``properties``). ``"-"`` reads the whole row.
    """
    expression = expression.strip()
    if expression == "-":
        return set(), set()
    columns, stray = set(), set()
    for match in _NAMES.finditer(_STRINGS.sub("''", expression)):
        name, member, call = match.groups()
        if (call and not member) or name in _GLOBALS:
            continue
        if not feature:
            columns.add(name)
        elif name == "properties" and member:
            columns.add(member)
        elif name not in _FEATURE_KEYS:
            stray.add(name)
    return columns, stray


def layer_fields(layer: dict[str, Any]) -> set[str]:
    """The columns a deck.gl layer's expressions read."""
    feature = _layer_type(layer) == "GeoJsonLayer"
    fields: set[str] = set()
    for expression in _expressions(layer):
        fields |= expression_names(expression, feature)[0]
    return fields


def deckgl_fields(spec: Any) -> set[str]:
    """The columns a deck.gl spec's layers read."""
    layers = spec.get("layers") if isinstance(spec, dict) else None
    if not isinstance(layers, list):
        return set()
    return {f for layer in layers if isinstance(layer, dict) for f in layer_fields(layer)}


def _suggest(name: str, options: list[str]) -> str:
    close = difflib.get_close_matches(name, options, n=1)
    return f" Did you mean {close[0]!r}?" if close else ""


def check_deckgl(spec: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """
    The deck.gl spec to draw for a pipeline with ``columns``, or ``ValueError``
    saying what to fix: the spec is plain JSON, each layer has a known
    ``@@type`` and no ``data`` of its own, and its expressions read columns
    the pipeline has.
    """
    if not isinstance(spec, dict):
        raise ValueError("The spec must be a deck.gl JSON object.")
    try:
        json.dumps(spec, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The spec must be plain JSON: {exc}.") from None
    if "map_style" in spec:
        raise ValueError("Name the basemap with `mapStyle`, not `map_style`, or leave it out to follow the page's theme.")
    style = spec.get("mapStyle")
    if isinstance(style, str) and style.startswith("mapbox://"):
        raise ValueError("Mapbox styles need an API key; leave `mapStyle` out to use a CARTO basemap that follows the page's theme.")
    if not isinstance(spec.get("initialViewState", {}), dict):
        raise ValueError("`initialViewState` must be an object, e.g. {\"latitude\": 40, \"longitude\": -100, \"zoom\": 3}.")
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("The spec needs a `layers` list with at least one layer.")
    known = sorted(LAYER_TYPES)
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"Layer {index + 1} must be an object.")
        name = _layer_name(index, layer)
        kind = _layer_type(layer)
        if kind is None:
            hint = " (`type` is pydeck's name for it)" if "type" in layer else ""
            raise ValueError(f"{name} needs an `@@type`{hint}, e.g. \"ScatterplotLayer\".")
        if kind not in LAYER_TYPES:
            raise ValueError(
                f"{name} has an unknown type {kind!r}.{_suggest(kind, known)} "
                f"The layer types that draw the data's rows are: {', '.join(known)}."
            )
        if "data" in layer:
            raise ValueError(f"{name} draws the rows of the node's input; remove its `data`.")
        for prop in POSITION_PROPS:
            value = layer.get(prop)
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                raise ValueError(
                    f"{name}'s `{prop}` lists column names; write it as an expression, "
                    f"\"@@=[{', '.join(value)}]\"."
                )
        feature = kind == "GeoJsonLayer"
        for expression in _expressions(layer):
            fields, stray = expression_names(expression, feature)
            if stray:
                raise ValueError(
                    f"{name} is a GeoJsonLayer, whose accessors get a GeoJSON Feature: read columns "
                    f"as properties.<column>, e.g. properties.{sorted(stray)[0]}."
                )
            if unknown := sorted(fields - set(columns)):
                hint = _suggest(unknown[0], columns)
                raise ValueError(
                    f"{name} reads {', '.join(unknown)}, which the data does not have.{hint} "
                    f"Its columns are: {', '.join(columns)}."
                )
    return copy.deepcopy(spec)


def _positions(spec: dict[str, Any]) -> tuple[str, str] | None:
    """The longitude and latitude columns of the first layer positioned by two columns."""
    for layer in spec.get("layers") or []:
        value = layer.get("getPosition") if isinstance(layer, dict) else None
        if not isinstance(value, str) or not value.startswith(_EXPRESSION):
            continue
        match = re.fullmatch(r"\s*\[\s*([A-Za-z_$][\w$]*)\s*,\s*([A-Za-z_$][\w$]*)\s*\]\s*", value[len(_EXPRESSION):])
        if match:
            return match.group(1), match.group(2)
    return None


def fit_view_state(spec: dict[str, Any], data: Any) -> dict[str, Any] | None:
    """
    An ``initialViewState`` showing the rows' positions, from the first
    layer positioned by longitude and latitude columns, leaving out the
    outer 1% at each side; None without such a layer or valid positions.
    """
    columns = _positions(spec)
    if columns is None or data is None or not all(c in getattr(data, "columns", ()) for c in columns):
        return None
    lon, lat = (data[c] for c in columns)
    try:
        valid = lon.between(-180, 180) & lat.between(-90, 90)
        lon, lat = lon[valid].astype(float), lat[valid].astype(float)
    except (TypeError, ValueError):
        return None
    if lon.empty:
        return None
    west, east = lon.quantile([0.01, 0.99])
    south, north = lat.quantile([0.01, 0.99])
    # Latitudes stretch away from the equator on the map, so they count for more.
    span = max(east - west, (north - south) * 1.5, 1e-3)
    zoom = min(max(math.log2(400 / span), 0), 15)
    return {
        "longitude": round((west + east) / 2, 4), "latitude": round((south + north) / 2, 4),
        "zoom": round(zoom, 1), "pitch": 0, "bearing": 0,
    }


def _summary(spec: dict[str, Any], data: Any) -> str:
    """A line saying what a map shows, e.g. ``Map of 2 layers (Hexagon, Scatterplot) of xlong and ylat, 4 rows``."""
    layers = [layer for layer in spec.get("layers") or [] if isinstance(layer, dict)]
    kinds = list(dict.fromkeys(
        kind.removesuffix("Layer") for layer in layers if (kind := _layer_type(layer))
    ))
    if len(layers) == 1 and kinds:
        text = f"{kinds[0]} map"
    elif layers:
        text = f"Map of {len(layers)} layers ({', '.join(kinds)})"
    else:
        text = "Map"
    position = _positions(spec)
    fields = list(position) if position else sorted(deckgl_fields(spec))[:3]
    if fields:
        text += f" of {' and '.join(fields)}"
    if data is not None and hasattr(data, "__len__"):
        text += f", {len(data):,} row{'' if len(data) == 1 else 's'}"
    return text


@register_view(
    title="Lumen DeckGL map", icon="map", extensions=["deckgl"],
    default_size={"height": 420}, min_size={"height": 240},
    description=(
        "A deck.gl map of a Lumen pipeline's rows, e.g. points, hexagon bins or arcs on a "
        "basemap; the spec's layers read the pipeline's columns in @@= expressions and bring no data."
    ),
)
class DeckGLNode(LumenViewNode):

    spec = spec_param(f"""
        The deck.gl JSON spec, without data: each layer draws the pipeline's
        rows and has an `@@type` (e.g. ScatterplotLayer, HexagonLayer,
        ColumnLayer, ArcLayer, HeatmapLayer), and reads columns in `@@=`
        expressions, e.g. "getPosition": "@@=[lon, lat]" or "getElevation":
        "@@=value * 100"; a GeoJsonLayer reads them as properties.<column>.
        Leave out `mapStyle` for a basemap following the page's light or
        dark theme, `initialViewState` to fit the map to the rows, and layer
        colours to take the dashboard's colour scheme. Example:
        {json.dumps(EXAMPLE)}""")

    view_type = "deckgl"

    themed = True

    check_spec = staticmethod(check_deckgl)

    spec_fields = staticmethod(deckgl_fields)

    def __init__(self, **params):
        # The page's theme once a drawn map has seen it, and the theme of the basemap drawn.
        self._dark: bool | None = None
        self._drawn_dark: bool | None = None
        super().__init__(**params)

    def _view_params(self, spec: dict[str, Any]) -> dict[str, Any]:
        spec = copy.deepcopy(spec)
        self._drawn_dark = None
        if spec.get("mapStyle", BASEMAPS["light"]) in BASEMAPS.values():
            self._drawn_dark = self._dark if self._dark is not None else pn.config.theme == "dark"
            spec["mapStyle"] = BASEMAPS["dark" if self._drawn_dark else "light"]
        if "initialViewState" not in spec:
            # fetch keeps the rows the view is drawn with, so this runs no query.
            fitted = fit_view_state(spec, fetch(self.pipeline, row_ceiling(self.editor)))
            if fitted is not None:
                spec["initialViewState"] = fitted
        color = SCHEME_COLORS.get(self._color_scheme() or "", SCHEME_COLORS["tableau10"])
        for layer in spec["layers"]:
            for prop in COLOR_PROPS.get(_layer_type(layer) or "", ()):
                layer.setdefault(prop, color)
        return {"spec": spec, "sizing_mode": "stretch_both"}

    def _follow_theme(self, dark: bool) -> bool:
        """Note the page's theme; whether the basemap drawn has the other one."""
        self._dark = dark
        return self._drawn_dark is not None and self._drawn_dark != dark

    def _body(self):
        body = super()._body()
        if self._object is None:
            return body
        # A pmui component follows the page's theme switch, so the map is drawn again with the other basemap.
        from panel_material_ui import Column

        frame = Column(body, sizing_mode="stretch_both", margin=0)
        node = weakref.ref(self)

        def follow(event):
            if (view := node()) is not None and view._follow_theme(event.new):
                view._refresh()

        frame.param.watch(follow, "dark_theme")
        if self._follow_theme(frame.dark_theme) and pn.state.curdoc is not None:
            # Drawn before the session's theme was known, e.g. while the dashboard loaded.
            pn.state.curdoc.add_next_tick_callback(self._refresh)
        return frame

    def _summary(self, spec: dict[str, Any], data: Any) -> str:
        return _summary(spec, data)


# Offered with the other Lumen nodes; nodes.py cannot import this module, which builds on it.
COMPONENTS[DECKGL_ID] = DeckGLNode
