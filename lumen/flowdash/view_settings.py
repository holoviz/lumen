"""
The settings of a Vega-Lite view that FlowDash's inspector shows as a form
beside the spec, and the theme a view follows when it is drawn.

Each setting is a path into the spec (see ``panel_flowdash.SpecSetting``):
the form writes it and reads it back when the spec is edited, and keys no
setting covers are kept as written. A setting the spec leaves out is not
written into it, so a new view fills its tile (``responsive_spec``) and takes
its colours and light or dark styling from the dashboard (``themed_spec``)
when it is drawn, never from values stored in the spec.
"""
from __future__ import annotations

import copy

from typing import Any

try:
    from panel_flowdash import SpecSetting
except ImportError:  # panel-flowdash without spec settings
    SpecSetting = None

# The colour schemes offered, Vega's names; "Follow dashboard" leaves it to the dashboard.
COLOR_SCHEMES = [
    "tableau10", "category10", "set2", "dark2", "paired", "observable10",
    "blues", "greens", "teals", "oranges", "purples", "viridis", "magma",
]

MARKS = {"Bar": "bar", "Line": "line", "Area": "area", "Point": "point"}

FIELD_TYPES = ["nominal", "ordinal", "quantitative", "temporal"]

AGGREGATES = {"None": None, "Sum": "sum", "Mean": "mean", "Median": "median",
              "Min": "min", "Max": "max", "Count": "count"}

LEGEND = {"Auto": "auto", "Top": "top", "Bottom": "bottom", "Left": "left",
          "Right": "right", "Off": "off"}

SIZE = {"Fill tile": "fill", "Fixed": "fixed"}

# The size a fixed chart starts from, in pixels.
FIXED_SIZE = {"width": 400, "height": 300}


def _mark(spec: dict) -> dict:
    """The spec's mark as an object, made one in place if it was given as a type."""
    mark = spec.get("mark")
    if not isinstance(mark, dict):
        mark = spec["mark"] = {} if mark is None else {"type": mark}
    return mark


def _drop_from_mark(spec: dict, key: str) -> None:
    """Remove ``key`` from an object mark, back to its type alone if nothing else is left."""
    mark = spec.get("mark")
    if not isinstance(mark, dict):
        return
    mark.pop(key, None)
    if set(mark) == {"type"}:
        spec["mark"] = mark["type"]


def mark_type(spec: dict) -> Any:
    mark = spec.get("mark")
    return mark.get("type") if isinstance(mark, dict) else mark


def set_mark_type(spec: dict, value: Any) -> None:
    if isinstance(spec.get("mark"), dict):
        spec["mark"]["type"] = value
    else:
        spec["mark"] = value


def points(spec: dict) -> bool:
    mark = spec.get("mark")
    return isinstance(mark, dict) and bool(mark.get("point"))


def set_points(spec: dict, value: bool) -> None:
    if value:
        _mark(spec)["point"] = True
    else:
        _drop_from_mark(spec, "point")


def tooltip(spec: dict) -> bool:
    mark = spec.get("mark")
    encoding = spec.get("encoding")
    in_mark = isinstance(mark, dict) and bool(mark.get("tooltip"))
    return in_mark or (isinstance(encoding, dict) and bool(encoding.get("tooltip")))


def set_tooltip(spec: dict, value: bool) -> None:
    """Turn tooltips on in the mark, or off in the mark and the encoding."""
    if value:
        _mark(spec)["tooltip"] = True
        return
    _drop_from_mark(spec, "tooltip")
    encoding = spec.get("encoding")
    if isinstance(encoding, dict):
        encoding.pop("tooltip", None)


def size(spec: dict) -> str:
    fixed = any(isinstance(spec.get(key), (int, float)) for key in FIXED_SIZE)
    return "fixed" if fixed else "fill"


def set_size(spec: dict, value: str) -> None:
    """Fill the tile by leaving the size out, or fix it, starting from ``FIXED_SIZE``."""
    for key, default in FIXED_SIZE.items():
        if value == "fill":
            spec.pop(key, None)
        elif not isinstance(spec.get(key), (int, float)):
            spec[key] = default


def _pixels(key: str):
    def get(spec: dict) -> Any:
        value = spec.get(key)
        return value if isinstance(value, (int, float)) else None

    def set(spec: dict, value: Any) -> None:
        if value is None:
            spec.pop(key, None)
        else:
            spec[key] = value

    return get, set


def _fixed(spec: dict) -> bool:
    return size(spec) == "fixed"


def title(spec: dict) -> Any:
    value = spec.get("title")
    return value.get("text") if isinstance(value, dict) else value


def set_title(spec: dict, value: Any) -> None:
    current = spec.get("title")
    if isinstance(current, dict):
        if value is None:
            current.pop("text", None)
        else:
            current["text"] = value
    elif value is None:
        spec.pop("title", None)
    else:
        spec["title"] = value


def color_scheme(spec: dict) -> Any:
    category = spec.get("config", {}).get("range", {}).get("category")
    return category.get("scheme") if isinstance(category, dict) else None


def set_color_scheme(spec: dict, value: Any) -> None:
    """Set the scheme of categorical and continuous colours, or leave both to the dashboard."""
    config = spec.setdefault("config", {})
    ranges = config.setdefault("range", {})
    for key in ("category", "ramp"):
        if value is None:
            ranges.pop(key, None)
        else:
            ranges[key] = {"scheme": value}
    if not ranges:
        config.pop("range")
    if not config:
        spec.pop("config")


def legend(spec: dict) -> str:
    config = spec.get("config", {}).get("legend", {})
    if config.get("disable"):
        return "off"
    return config.get("orient", "auto")


def set_legend(spec: dict, value: str) -> None:
    config = spec.setdefault("config", {})
    legend = config.setdefault("legend", {})
    legend.pop("disable", None)
    legend.pop("orient", None)
    if value == "off":
        legend["disable"] = True
    elif value != "auto":
        legend["orient"] = value
    if not legend:
        config.pop("legend")
    if not config:
        spec.pop("config")


def _settings() -> list:
    if SpecSetting is None:
        return []
    width_get, width_set = _pixels("width")
    height_get, height_set = _pixels("height")
    return [
        SpecSetting("mark.type", label="Mark", widget="radio", options=MARKS, group="Chart",
                    get=mark_type, set=set_mark_type, restrict=False),
        SpecSetting("mark.point", label="Points", widget="switch", group="Chart",
                    get=points, set=set_points, default=False,
                    enabled=lambda spec: mark_type(spec) in ("line", "area")),
        SpecSetting("encoding.x.field", label="X", widget="select", group="Chart",
                    choices="columns", of="pipeline", restrict=False),
        SpecSetting("encoding.x.type", label="X type", widget="select", group="Chart",
                    options=FIELD_TYPES),
        SpecSetting("encoding.y.field", label="Y", widget="select", group="Chart",
                    choices="columns", of="pipeline", restrict=False),
        SpecSetting("encoding.y.type", label="Y type", widget="select", group="Chart",
                    options=FIELD_TYPES),
        SpecSetting("encoding.y.aggregate", label="Aggregate", widget="select", group="Chart",
                    options=AGGREGATES),
        SpecSetting("encoding.color.field", label="Colour by", widget="select", group="Chart",
                    choices="columns", of="pipeline", restrict=False),
        SpecSetting("size", label="Size", widget="radio", options=SIZE, default="fill",
                    group="Size and style", get=size, set=set_size, covers=("width", "height"),
                    doc="Fill the tile, or a fixed width and height in pixels."),
        SpecSetting("width", label="Width", widget="integer", group="Size and style",
                    get=width_get, set=width_set, enabled=_fixed),
        SpecSetting("height", label="Height", widget="integer", group="Size and style",
                    get=height_get, set=height_set, enabled=_fixed),
        SpecSetting("config.range.category", label="Colours", widget="select",
                    options={"Follow dashboard": None, **{s: s for s in COLOR_SCHEMES}},
                    group="Size and style", get=color_scheme, set=set_color_scheme,
                    covers=("config.range.ramp",), restrict=False),
        SpecSetting("config.legend.orient", label="Legend", widget="radio", options=LEGEND,
                    default="auto", group="Size and style", get=legend, set=set_legend,
                    covers=("config.legend.disable",)),
        SpecSetting("mark.tooltip", label="Tooltips", widget="switch", default=False,
                    group="Size and style", get=tooltip, set=set_tooltip,
                    covers=("encoding.tooltip",)),
        SpecSetting("title", label="Title", group="Labels", get=title, set=set_title),
        SpecSetting("encoding.x.title", label="X title", group="Labels"),
        SpecSetting("encoding.y.title", label="Y title", group="Labels"),
    ]


#: The settings of a Vega-Lite view's spec, as the inspector's form shows them.
VEGA_LITE_SETTINGS = _settings()


def themed_spec(spec: dict[str, Any], scheme: str | None = None) -> dict[str, Any]:
    """
    ``spec`` as it is drawn on a dashboard: with a transparent background,
    so the chart sits on its tile in light and dark mode (the page styles
    the text and axes, see panel-material-ui), and with the dashboard's
    colour ``scheme`` unless the spec sets colours of its own. The spec is
    not changed.
    """
    config = copy.deepcopy(spec.get("config")) if isinstance(spec.get("config"), dict) else {}
    config.setdefault("background", "transparent")
    if scheme:
        ranges = config.setdefault("range", {})
        for key in ("category", "ramp"):
            ranges.setdefault(key, {"scheme": scheme})
    return {**spec, "config": config}
