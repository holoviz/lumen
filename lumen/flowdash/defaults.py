"""
The spec a new view node starts from, built from its pipeline's schema, see
``LumenViewNode.default_spec``. Each returns None when no chart is obvious,
and the view then shows its empty state.

A default leaves out sizes and colours, so the view fills its tile and
follows the dashboard's theme when it is drawn (``responsive_spec`` and
``themed_spec``).
"""
from __future__ import annotations

import math
import re

from typing import Any

# The most categories a default bar chart shows, the largest first.
TOP_CATEGORIES = 20

# The integer range a column of years falls in.
YEARS = (1800, 2100)

_TEMPORAL_FORMATS = ("date", "datetime", "date-time")

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?")


def _columns(schema: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(name): spec for name, spec in (schema or {}).items()
        if name != "__len__" and isinstance(spec, dict)
    }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _is_empty(spec: dict[str, Any]) -> bool:
    """Whether Lumen's schema says the column holds only nulls: listed values all null, or number bounds that are not numbers."""
    enum = spec.get("enum")
    if isinstance(enum, list) and enum and all(_is_null(value) for value in enum):
        return True
    bounds = [spec[key] for key in ("inclusiveMinimum", "inclusiveMaximum") if key in spec]
    return spec.get("type") in ("number", "integer") and any(not _finite(value) for value in bounds)


# Columns of coordinates, which locate a row rather than measure it.
_COORDINATES = ("lon", "lng", "long", "longitude", "xlong", "lat", "latitude", "ylat")


def _is_id(name: str) -> bool:
    """Whether a number column names or locates rows rather than measuring them, e.g. an id or a longitude."""
    lowered = name.lower()
    if lowered in ("id", "index") or lowered.endswith("_id"):
        return True
    return lowered in _COORDINATES or lowered.rsplit("_", 1)[-1] in _COORDINATES


def _named_year(name: str, spec: dict[str, Any]) -> bool:
    lowered = name.lower()
    if not (lowered in ("year", "yr") or lowered.endswith(("_year", "_yr")) or lowered.startswith("year_")):
        return False
    return spec.get("type") == "integer" or not _finite(spec.get("inclusiveMaximum")) or spec["inclusiveMaximum"] <= YEARS[1]


def _year_range(spec: dict[str, Any]) -> bool:
    low, high = spec.get("inclusiveMinimum"), spec.get("inclusiveMaximum")
    return spec.get("type") == "integer" and _finite(low) and _finite(high) and YEARS[0] <= low <= high <= YEARS[1]


def _is_date_text(spec: dict[str, Any]) -> bool:
    values = [value for value in spec.get("enum") or [] if not _is_null(value)]
    return bool(values) and all(isinstance(value, str) and _ISO_DATE.fullmatch(value.strip()) for value in values)


def schema_kinds(schema: dict[str, Any] | None) -> dict[str, list[str]]:
    """
    The columns of a Lumen ``schema`` as ``temporal``, ``year``, ``numeric``
    (fractional numbers before integers) and ``categorical``, leaving out
    columns of nulls and id columns.
    """
    kinds: dict[str, list[str]] = {"temporal": [], "year": [], "numeric": [], "categorical": []}
    integers, in_range = [], []
    for name, spec in _columns(schema).items():
        if _is_empty(spec):
            continue
        kind = spec.get("type")
        if spec.get("format") in _TEMPORAL_FORMATS or (kind == "string" and _is_date_text(spec)):
            kinds["temporal"].append(name)
        elif kind in ("number", "integer") and _named_year(name, spec):
            kinds["year"].append(name)
        elif _year_range(spec) and not _is_id(name):
            in_range.append(name)
        elif kind == "number" and not _is_id(name):
            kinds["numeric"].append(name)
        elif kind == "integer" and not _is_id(name):
            integers.append(name)
        elif kind == "boolean" or (kind == "string" and not spec.get("format")):
            kinds["categorical"].append(name)
    kinds["numeric"] += integers
    # Integers whose values all look like years are years when something else is measured, else the measure.
    kinds["year" if kinds["numeric"] else "numeric"] += in_range
    return kinds


def vega_field(name: str) -> str:
    """``name`` as a Vega-Lite field, which would read a dot or a bracket as a nested property."""
    return re.sub(r"([.\[\]\\])", r"\\\1", name)


def field_column(field: str) -> str:
    """The column a Vega-Lite ``field`` reads, without the escapes ``vega_field`` adds."""
    return re.sub(r"\\(.)", r"\1", field)


def _categorical(schema: dict[str, Any], names: list[str]) -> str:
    # A column Lumen lists the values of has few of them, so it makes a readable bar chart.
    columns = _columns(schema)
    return next((name for name in names if columns[name].get("enum")), names[0])


def _line(x: str, x_type: str, y: str) -> dict[str, Any]:
    return {
        "mark": "line",
        "encoding": {
            "x": {"field": vega_field(x), "type": x_type},
            "y": {"field": vega_field(y), "aggregate": "sum", "type": "quantitative"},
        },
    }


def vega_lite_default(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    A Vega-Lite spec for a pipeline with ``schema``: a line of the first
    number over a date or a year, else the sum of the first number for each
    category as bars, the largest first and at most ``TOP_CATEGORIES`` of
    them, else a scatter of two numbers, else a histogram of one. None when
    no column is a number to measure.
    """
    kinds = schema_kinds(schema)
    numeric = kinds["numeric"]
    if not numeric:
        return None
    y = numeric[0]
    if kinds["temporal"]:
        return _line(kinds["temporal"][0], "temporal", y)
    if kinds["year"]:
        return _line(kinds["year"][0], "ordinal", y)
    if kinds["categorical"]:
        category = _categorical(schema, kinds["categorical"])
        spec: dict[str, Any] = {
            "mark": "bar",
            "encoding": {
                "x": {"field": vega_field(category), "type": "nominal", "sort": "-y"},
                "y": {"field": vega_field(y), "aggregate": "sum", "type": "quantitative"},
            },
        }
        column = _columns(schema)[category]
        enum = column.get("enum")
        if column.get("type") != "boolean" and not (isinstance(enum, list) and len(enum) <= TOP_CATEGORIES):
            # Summed first, so the rank is of the sums the bars show and the bars sum one value each.
            spec["transform"] = [
                {"aggregate": [{"op": "sum", "field": vega_field(y), "as": y}], "groupby": [vega_field(category)]},
                {"window": [{"op": "row_number", "as": "_rank"}], "sort": [{"field": vega_field(y), "order": "descending"}]},
                {"filter": f"datum._rank <= {TOP_CATEGORIES}"},
            ]
        return spec
    if len(numeric) > 1:
        return {
            "mark": "point",
            "encoding": {
                "x": {"field": vega_field(numeric[0]), "type": "quantitative"},
                "y": {"field": vega_field(numeric[1]), "type": "quantitative"},
            },
        }
    return {
        "mark": "bar",
        "encoding": {
            "x": {"field": vega_field(y), "bin": True, "type": "quantitative"},
            "y": {"aggregate": "count", "type": "quantitative"},
        },
    }
