"""
A deck.gl map of a Lumen pipeline as a FlowDash node.

The spec is deck.gl's JSON format (as pydeck writes it) without data: every
layer draws the pipeline's rows, and column-based props are deck.gl
expressions such as ``"getPosition": "@@=[lon, lat]"``. A spec is written by
whoever edits the dashboard, its agent included, and is drawn in the browser
of everyone who views it, so ``check_deckgl`` takes only what it knows to
draw without fetching anything: the top-level keys, layer types and string
props it lists, CARTO's basemaps and expressions that read columns the
pipeline has. A spec that leaves out the basemap gets a light or dark one
following the page, one that leaves out the initial view state is fitted to
the rows, and layers without a colour take the dashboard's colour scheme.
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
    LumenViewNode, fetch, register_view, row_ceiling, spec_param,
)

DECKGL_ID = "Lumen/deckgl"

# The layers that draw one datum per row from the data they are given.
LAYER_TYPES = frozenset({
    "ArcLayer", "ColumnLayer", "ContourLayer", "GeoJsonLayer", "GeohashLayer",
    "GreatCircleLayer", "GridCellLayer", "GridLayer", "H3ClusterLayer",
    "H3HexagonLayer", "HeatmapLayer", "HexagonLayer", "LineLayer", "PathLayer",
    "PointCloudLayer", "PolygonLayer", "QuadkeyLayer", "S2Layer",
    "ScatterplotLayer", "ScreenGridLayer", "SolidPolygonLayer", "TextLayer",
    "TripsLayer",
})

# The keys a spec may have at the top level.
SPEC_KEYS = ("layers", "initialViewState", "views", "mapStyle", "controller", "tooltip")

# The view classes a spec's views may name.
VIEW_TYPES = frozenset({"MapView"})

# Layer props that take a plain string, a name or keyword rather than anything
# deck.gl would load; any other string must be an expression.
STRING_PROPS = frozenset({
    "id", "fontFamily", "fontWeight", "textAnchor", "alignmentBaseline", "wordBreak",
    "characterSet", "colorScaleType", "elevationScaleType", "aggregation",
    "colorAggregation", "elevationAggregation", "positionFormat", "radiusUnits",
    "widthUnits", "sizeUnits", "lineWidthUnits", "getTextAnchor", "getAlignmentBaseline",
    "getText",
})
_PLAIN = re.compile(r"[A-Za-z0-9_][\w ,'\-]*")
# deck.gl's enumerations Panel registers, e.g. "@@#COORDINATE_SYSTEM.LNGLAT".
_CONSTANT = re.compile(r"@@#(?:COORDINATE_SYSTEM|GL)\.[A-Z0-9_]+")

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

# The basemaps a spec may name; null draws none.
MAP_STYLES = frozenset({
    *BASEMAPS.values(),
    *(f"https://basemaps.cartocdn.com/gl/{name}-gl-style/style.json" for name in (
        "positron-nolabels", "dark-matter-nolabels", "voyager", "voyager-nolabels",
    )),
})

# Where a spec without positions starts: the whole world.
WORLD_VIEW = {"longitude": 0, "latitude": 20, "zoom": 1, "pitch": 0, "bearing": 0}

# The attribution box of a dark basemap, which MapLibre draws white.
DARK_ATTRIBUTION = """
.maplibregl-ctrl.maplibregl-ctrl-attrib { background-color: rgba(30, 30, 30, 0.7); color: #ddd; }
.maplibregl-ctrl-attrib a { color: #ddd; }
"""

EXAMPLE = {
    "layers": [{
        "@@type": "ScatterplotLayer",
        "getPosition": "@@=[longitude, latitude]",
        "getRadius": "@@=capacity * 10",
        "radiusMinPixels": 2,
        "pickable": True,
    }],
    "tooltip": {"text": "{name}: {capacity}"},
}

_EXPRESSION = "@@="
_STRINGS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
# A datum's column read by name in brackets, e.g. this['total sales'].
_BRACKETS = re.compile(r"(?<![\w$.])(this|properties)\s*\[\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s*\]")
# A name not read off another value, with the member read off it.
_NAMES = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)(?:\s*\.\s*([A-Za-z_$][\w$]*))?")
_CALL = re.compile(r"[\w$)\]]\s*\(")
_LITERALS = frozenset({"true", "false", "null"})
# Object internals deck.gl refuses to read, as a member or in brackets.
_HIDDEN = re.compile(r"(?<![\w$])(constructor|prototype|__proto__)(?![\w$])")
# Names deck.gl does not resolve on the datum, so an expression reading them draws nothing.
_GLOBALS = frozenset({
    "Math", "Number", "String", "Boolean", "Date", "JSON", "Object", "Array", "NaN",
    "Infinity", "undefined", "window", "document", "globalThis", "constructor",
    "__proto__", "prototype",
})
# What a GeoJsonLayer accessor may read off its datum, a GeoJSON Feature.
_FEATURE_KEYS = frozenset({"properties", "geometry", "type", "id"})
_TEMPLATE = re.compile(r"\{([^{}]+)\}")


def layer_type(layer: dict[str, Any]) -> str | None:
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


def _unquote(literal: str) -> str:
    return re.sub(r"\\(.)", r"\1", literal[1:-1])


def expression_names(expression: str, feature: bool = False) -> tuple[set[str], set[str]]:
    """
    The columns a deck.gl expression reads, and the other names it reads:
    globals, and for a ``feature`` (a GeoJSON Feature, whose columns are its
    ``properties``) names that are no column. A column may be read bare,
    off ``this`` or in brackets (``this['a b']``); ``"-"`` reads the whole row.
    """
    expression = expression.strip()
    if expression == "-":
        return set(), set()
    columns, stray = set(), set()
    for match in _BRACKETS.finditer(expression):
        if match.group(1) == "properties" or not feature:
            columns.add(_unquote(match.group(2)))
    expression = _STRINGS.sub("''", _BRACKETS.sub("0", expression))
    for name, member in _NAMES.findall(expression):
        if name == "this" and member:
            name, member = member, ""
        if name in _LITERALS or name == "this":
            continue
        if name in _GLOBALS:
            stray.add(name)
        elif not feature:
            columns.add(name)
        elif name == "properties" and member:
            columns.add(member)
        elif name not in _FEATURE_KEYS:
            stray.add(name)
    return columns, stray


def tooltip_fields(tooltip: Any) -> set[str]:
    """The columns a ``{"text": "{column} ..."}`` tooltip names."""
    text = tooltip.get("text") if isinstance(tooltip, dict) else None
    return {name.strip() for name in _TEMPLATE.findall(text)} if isinstance(text, str) else set()


def layer_fields(layer: dict[str, Any]) -> set[str]:
    """The columns a deck.gl layer's expressions read."""
    feature = layer_type(layer) == "GeoJsonLayer"
    fields: set[str] = set()
    for expression in _expressions(layer):
        fields |= expression_names(expression, feature)[0]
    return fields


def deckgl_fields(spec: Any) -> set[str]:
    """The columns a deck.gl spec's layers and tooltip read."""
    if not isinstance(spec, dict):
        return set()
    layers = spec.get("layers")
    fields = tooltip_fields(spec.get("tooltip"))
    if isinstance(layers, list):
        fields |= {f for layer in layers if isinstance(layer, dict) for f in layer_fields(layer)}
    return fields


def _suggest(name: str, options: list[str]) -> str:
    close = difflib.get_close_matches(name, options, n=1)
    return f" Did you mean {close[0]!r}?" if close else ""


def _check_value(where: str, key: str, value: Any, *, top: bool = False) -> None:
    """
    Refuse what deck.gl's JSON converter would build or load from ``value``:
    any ``@@function``, any ``@@type`` other than a layer's own, and strings
    other than expressions, enumerations and plain names under ``STRING_PROPS``.
    """
    if isinstance(value, dict):
        if "@@function" in value:
            raise ValueError(f"{where} calls a function with `@@function`, which a map spec may not do.")
        if "@@type" in value and not top:
            raise ValueError(
                f"{where} builds a {value['@@type']!r} in `{key}`; a map spec may only name its layers' "
                "types, not extensions, loaders or other deck.gl classes."
            )
        for item_key, item in value.items():
            if item_key in ("loadOptions", "_subLayerProps", "fetch"):
                raise ValueError(f"{where} sets `{item_key}`, which a map spec may not set.")
            if top and item_key == "@@type":
                continue
            _check_value(where, item_key, item)
    elif isinstance(value, list):
        for item in value:
            _check_value(where, key, item)
    elif isinstance(value, str):
        if value.startswith(_EXPRESSION) or _CONSTANT.fullmatch(value):
            return
        if key in STRING_PROPS and _PLAIN.fullmatch(value):
            return
        raise ValueError(
            f"{where} sets `{key}` to the string {value!r}; a map spec takes strings only as "
            "@@= expressions and as plain names for "
            f"{', '.join(sorted(STRING_PROPS))}, so it never loads URLs or files."
        )


def _check_expression(name: str, expression: str, feature: bool, columns: list[str]) -> None:
    if any(re.search(r"[:/]", literal) for literal in _STRINGS.findall(expression)):
        raise ValueError(
            f"{name} has a string with ':' or '/' in \"@@={expression}\"; a map spec may not name URLs or paths."
        )
    if _CALL.search(_STRINGS.sub("''", expression)):
        raise ValueError(
            f"{name} calls a function in \"@@={expression}\"; deck.gl expressions take no function "
            "calls, so use arithmetic, comparisons and ?: instead, e.g. \"@@=value * 10\"."
        )
    if hidden := sorted(set(_HIDDEN.findall(expression))):
        raise ValueError(
            f"{name} reads {', '.join(hidden)} in \"@@={expression}\"; deck.gl expressions can only "
            "read the row's columns and their values, not object internals."
        )
    fields, stray = expression_names(expression, feature)
    if globals_read := sorted(stray & _GLOBALS):
        raise ValueError(
            f"{name} reads {', '.join(globals_read)} in \"@@={expression}\"; deck.gl expressions can "
            "only read the row's columns, so use arithmetic, comparisons and ?: instead."
        )
    if stray:
        raise ValueError(
            f"{name} is a GeoJsonLayer, whose accessors get a GeoJSON Feature: read columns "
            f"as properties.<column>, e.g. properties.{sorted(stray)[0]}."
        )
    _check_columns(name, fields, columns)


def _check_columns(name: str, fields: set[str], columns: list[str]) -> None:
    if unknown := sorted(fields - set(columns)):
        raise ValueError(
            f"{name} reads {', '.join(unknown)}, which the data does not have.{_suggest(unknown[0], columns)} "
            f"Its columns are: {', '.join(columns)}."
        )


def _check_tooltip(tooltip: Any, columns: list[str]) -> None:
    if isinstance(tooltip, bool):
        return
    if not isinstance(tooltip, dict) or set(tooltip) != {"text"} or not isinstance(tooltip["text"], str):
        raise ValueError(
            "`tooltip` is true, false or {\"text\": \"...\"} naming columns in braces, "
            "e.g. {\"text\": \"{name}: {value}\"}."
        )
    _check_columns("The tooltip", tooltip_fields(tooltip), columns)


def check_deckgl(spec: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """
    The deck.gl spec to draw for a pipeline with ``columns``, or ``ValueError``
    saying what to fix. Only what is known to draw without loading anything
    passes: the top-level keys in ``SPEC_KEYS``, layers of ``LAYER_TYPES``
    without data, ``MapView`` views, a basemap of ``MAP_STYLES`` (or null),
    strings that are expressions, enumerations or plain names of
    ``STRING_PROPS``, and expressions that read columns the pipeline has
    without calling functions.
    """
    if not isinstance(spec, dict):
        raise ValueError("The spec must be a deck.gl JSON object.")
    try:
        json.dumps(spec, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The spec must be plain JSON: {exc}.") from None
    if "map_style" in spec:
        raise ValueError("Name the basemap with `mapStyle`, not `map_style`, or leave it out to follow the page's theme.")
    if unknown := [key for key in spec if key not in SPEC_KEYS]:
        raise ValueError(f"The spec may not set {', '.join(map(repr, unknown))}; its keys are {', '.join(SPEC_KEYS)}.")
    style = spec.get("mapStyle")
    if style is not None and (not isinstance(style, str) or style not in MAP_STYLES):
        raise ValueError(
            "`mapStyle` must be one of CARTO's basemaps or null for none; leave it out for one "
            f"that follows the page's theme. The basemaps are: {', '.join(sorted(MAP_STYLES))}."
        )
    state = spec.get("initialViewState", {})
    if not isinstance(state, dict) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in state.values()
    ):
        raise ValueError("`initialViewState` must be an object of numbers, e.g. {\"latitude\": 40, \"longitude\": -100, \"zoom\": 3}.")
    if not isinstance(spec.get("controller", True), bool):
        raise ValueError("`controller` must be true or false.")
    views = spec.get("views", [])
    if not isinstance(views, list) or not all(isinstance(v, dict) and v.get("@@type") in VIEW_TYPES for v in views):
        raise ValueError(f"`views` must be a list of {', '.join(sorted(VIEW_TYPES))} objects, e.g. [{{\"@@type\": \"MapView\", \"controller\": true}}].")
    for view in views:
        _check_value("A view", "views", view, top=True)
    if "tooltip" in spec:
        _check_tooltip(spec["tooltip"], columns)
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("The spec needs a `layers` list with at least one layer.")
    known = sorted(LAYER_TYPES)
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"Layer {index + 1} must be an object.")
        name = _layer_name(index, layer)
        kind = layer_type(layer)
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
        _check_value(name, "layers", layer, top=True)
        for expression in _expressions(layer):
            _check_expression(name, expression, kind == "GeoJsonLayer", columns)
    return copy.deepcopy(spec)


def position_columns(spec: dict[str, Any]) -> tuple[str, str] | None:
    """The longitude and latitude columns of the first layer positioned by two columns."""
    for layer in spec.get("layers") or []:
        value = layer.get("getPosition") if isinstance(layer, dict) else None
        if not isinstance(value, str) or not value.startswith(_EXPRESSION):
            continue
        match = re.fullmatch(r"\s*\[\s*([A-Za-z_$][\w$]*)\s*,\s*([A-Za-z_$][\w$]*)\s*\]\s*", value[len(_EXPRESSION):])
        if match:
            return match.group(1), match.group(2)
    return None


LONGITUDE_NAMES = ("lon", "lng", "long", "longitude", "xlong", "x_long")

LATITUDE_NAMES = ("lat", "latitude", "ylat", "y_lat")

_IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")


# Prefixes of the second point of a pair, e.g. a trip's end, which a map of the first point skips.
_SECOND_POINT = ("end", "dest", "destination", "dropoff", "to", "target", "arrival")


def _coordinates(schema: dict[str, Any], names: tuple[str, ...], bound: float) -> dict[str, str]:
    """
    The number columns named like ``names`` by their prefix, e.g. ``""`` for
    ``lon`` and ``"pickup"`` for ``pickup_lon``, whose bounds are numbers
    within +/- ``bound``.
    """
    found: dict[str, str] = {}
    for column, spec in schema.items():
        if column == "__len__" or not isinstance(spec, dict) or spec.get("type") not in ("number", "integer"):
            continue
        name = str(column).lower()
        if name in names:
            # xlong and ylat pair with each other, never with a bare lon or lat.
            prefix = "xy" if name[0] in "xy" and name[1:].lstrip("_") in names else ""
        elif "_" in name and name.rsplit("_", 1)[1] in names:
            prefix = name.rsplit("_", 1)[0]
        else:
            continue
        if not _IDENTIFIER.fullmatch(str(column)) or prefix in found:
            continue
        bounds = [spec[key] for key in ("inclusiveMinimum", "inclusiveMaximum") if key in spec]
        if any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or abs(v) > bound
            for v in bounds
        ):
            continue
        found[prefix] = str(column)
    return found


def deckgl_default(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    A deck.gl spec for a pipeline with ``schema``: a scatter of its rows,
    sized in pixels, at a longitude and a latitude column of one pair, e.g.
    ``lon`` and ``lat`` or ``origin_lng`` and ``origin_lat``, whose values
    fit. Unprefixed names come first, then a pair's first point (``start``
    before ``end``), then the order of the columns. None without a pair.
    """
    schema = schema or {}
    lons = _coordinates(schema, LONGITUDE_NAMES, 180)
    lats = _coordinates(schema, LATITUDE_NAMES, 90)
    pairs = [prefix for prefix in lons if prefix in lats and lons[prefix] != lats[prefix]]
    if not pairs:
        return None
    prefix = min(pairs, key=lambda p: (p not in ("", "xy"), p != "", p.split("_")[-1] in _SECOND_POINT, pairs.index(p)))
    lon, lat = lons[prefix], lats[prefix]
    return {
        "layers": [{
            "@@type": "ScatterplotLayer",
            "getPosition": f"@@=[{lon}, {lat}]",
            "radiusUnits": "pixels",
            "getRadius": 3,
            "pickable": True,
        }],
    }


def fit_view_state(spec: dict[str, Any], data: Any) -> dict[str, Any] | None:
    """
    An ``initialViewState`` showing the rows' positions, from the first
    layer positioned by longitude and latitude columns, leaving out the
    outer 1% at each side; None without such a layer or valid positions.
    """
    columns = position_columns(spec)
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


def map_summary(spec: dict[str, Any], data: Any) -> str:
    """A line saying what a map shows, e.g. ``Map of 2 layers (Hexagon, Scatterplot) of xlong and ylat, 4 rows``."""
    layers = [layer for layer in spec.get("layers") or [] if isinstance(layer, dict)]
    kinds = list(dict.fromkeys(
        kind.removesuffix("Layer") for layer in layers if (kind := layer_type(layer))
    ))
    if len(layers) == 1 and kinds:
        text = f"{kinds[0]} map"
    elif layers:
        text = f"Map of {len(layers)} layers ({', '.join(kinds)})"
    else:
        text = "Map"
    position = position_columns(spec)
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
        colours to take the dashboard's colour scheme. A new map of rows
        with longitude and latitude columns starts as a scatter of them.
        Example:
        {json.dumps(EXAMPLE)}""")

    view_type = "deckgl"

    themed = True

    check_spec = staticmethod(check_deckgl)

    spec_fields = staticmethod(deckgl_fields)

    default_spec = staticmethod(deckgl_default)

    _empty = "No map yet"

    def __init__(self, **params):
        # The page's theme once a drawn map has seen it, and the theme of the basemap drawn.
        self._dark: bool | None = None
        self._drawn_dark: bool | None = None
        super().__init__(**params)

    def _view_rows(self, rows: Any, spec: dict[str, Any]) -> Any:
        """Only the columns the spec reads, and a GeoDataFrame's geometry, go to the browser."""
        if not hasattr(rows, "columns") or any(e.strip() == "-" for e in _expressions(spec.get("layers"))):
            return rows
        keep = deckgl_fields(spec)
        geometry = getattr(getattr(rows, "geometry", None), "name", None)
        return rows[[c for c in rows.columns if c in keep or c == geometry]]

    def _style(self, dark: bool) -> dict[str, Any]:
        """
        The pane params that change with the page's theme. A changed
        ``configuration`` makes Panel's deck.gl model draw the map again
        with the new basemap from the data it already has.
        """
        return {
            "configuration": "{}" if dark else "",
            "stylesheets": [DARK_ATTRIBUTION] if dark else [],
        }

    def _view_params(self, spec: dict[str, Any]) -> dict[str, Any]:
        spec = copy.deepcopy(spec)
        tooltip = spec.pop("tooltip", True)
        self._drawn_dark = None
        dark = self._dark if self._dark is not None else pn.config.theme == "dark"
        if spec.get("mapStyle", BASEMAPS["light"]) in BASEMAPS.values():
            self._drawn_dark = dark
            spec["mapStyle"] = BASEMAPS["dark" if dark else "light"]
        if "initialViewState" not in spec:
            # fetch keeps the rows the view is drawn with, so this runs no query.
            fitted = fit_view_state(spec, fetch(self.pipeline, row_ceiling(self.editor)))
            spec["initialViewState"] = fitted or dict(WORLD_VIEW)
        color = SCHEME_COLORS.get(self._color_scheme() or "", SCHEME_COLORS["tableau10"])
        for layer in spec["layers"]:
            for prop in COLOR_PROPS.get(layer_type(layer) or "", ()):
                layer.setdefault(prop, color)
        return {
            "spec": spec, "tooltips": tooltip, "sizing_mode": "stretch_both",
            **self._style(bool(self._drawn_dark)),
        }

    def _follow_theme(self, dark: bool) -> None:
        """Give the drawn map the basemap of the page's theme, without sending its rows again."""
        self._dark = dark
        view = self._object
        if self._drawn_dark is None or self._drawn_dark == dark or view is None:
            return
        self._drawn_dark = dark
        view.spec["mapStyle"] = BASEMAPS["dark" if dark else "light"]
        pane = getattr(view, "_panel", None)
        if isinstance(pane, pn.pane.DeckGL) and isinstance(pane.object, dict):
            # The basemap goes first, so the map drawn again for the new configuration has it.
            style = self._style(dark)
            pane.param.update(object={**pane.object, "mapStyle": view.spec["mapStyle"]}, stylesheets=style["stylesheets"])
            pane.configuration = style["configuration"]

    def _body(self):
        body = super()._body()
        if self._object is None:
            return body
        # A pmui component follows the page's theme switch, which the basemap follows too.
        from panel_material_ui import Column

        frame = Column(body, sizing_mode="stretch_both", margin=0)
        node = weakref.ref(self)

        def follow(event):
            if (view := node()) is not None:
                view._follow_theme(event.new)

        frame.param.watch(follow, "dark_theme")
        # Drawn before the session's theme was known, e.g. while the dashboard loaded.
        self._follow_theme(frame.dark_theme)
        return frame

    def _summary(self, spec: dict[str, Any], data: Any) -> str:
        return map_summary(spec, data)
