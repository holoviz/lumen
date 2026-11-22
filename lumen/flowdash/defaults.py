"""
The spec a new view node starts from, built from its pipeline's schema, see
``LumenViewNode.default_spec``. Each returns None when no chart is obvious,
and the view then shows its empty state.

A default leaves out sizes and colours, so the view fills its tile and
follows the dashboard's theme when it is drawn (``responsive_spec`` and
``themed_spec``).
"""
from __future__ import annotations

from typing import Any

# The most categories a default bar chart shows, the largest first.
TOP_CATEGORIES = 20

_TEMPORAL_FORMATS = ("date", "datetime", "date-time")


def _columns(schema: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(name): spec for name, spec in (schema or {}).items()
        if name != "__len__" and isinstance(spec, dict)
    }


def _is_id(name: str) -> bool:
    lowered = name.lower()
    return lowered in ("id", "index") or lowered.endswith("_id")


def schema_kinds(schema: dict[str, Any] | None) -> dict[str, list[str]]:
    """The columns of a Lumen ``schema`` as ``temporal``, ``numeric`` and ``categorical``, in order."""
    kinds: dict[str, list[str]] = {"temporal": [], "numeric": [], "categorical": []}
    for name, spec in _columns(schema).items():
        kind = spec.get("type")
        if spec.get("format") in _TEMPORAL_FORMATS:
            kinds["temporal"].append(name)
        elif kind in ("number", "integer") and not _is_id(name):
            kinds["numeric"].append(name)
        elif kind == "boolean" or (kind == "string" and not spec.get("format")):
            kinds["categorical"].append(name)
    return kinds


def _categorical(schema: dict[str, Any], names: list[str]) -> str:
    # A column Lumen lists the values of has few of them, so it makes a readable bar chart.
    columns = _columns(schema)
    return next((name for name in names if columns[name].get("enum")), names[0])


def vega_lite_default(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    A Vega-Lite spec for a pipeline with ``schema``: a line of the first
    number over a date, else the sum of the first number for each of the
    largest categories as bars, else a scatter of two numbers, else a
    histogram of one. None when no column is a number.
    """
    kinds = schema_kinds(schema)
    numeric, temporal, categorical = kinds["numeric"], kinds["temporal"], kinds["categorical"]
    if not numeric:
        return None
    y = numeric[0]
    if temporal:
        return {
            "mark": "line",
            "encoding": {
                "x": {"field": temporal[0], "type": "temporal"},
                "y": {"field": y, "aggregate": "sum", "type": "quantitative"},
            },
        }
    if categorical:
        category = _categorical(schema, categorical)
        return {
            "transform": [
                {"aggregate": [{"op": "sum", "field": y, "as": y}], "groupby": [category]},
                {"window": [{"op": "row_number", "as": "_rank"}], "sort": [{"field": y, "order": "descending"}]},
                {"filter": f"datum._rank <= {TOP_CATEGORIES}"},
            ],
            "mark": "bar",
            "encoding": {
                "y": {"field": category, "type": "nominal", "sort": "-x"},
                "x": {"field": y, "type": "quantitative", "title": f"Sum of {y}"},
            },
        }
    if len(numeric) > 1:
        return {
            "mark": "point",
            "encoding": {
                "x": {"field": numeric[0], "type": "quantitative"},
                "y": {"field": numeric[1], "type": "quantitative"},
            },
        }
    return {
        "mark": "bar",
        "encoding": {
            "x": {"field": y, "bin": True, "type": "quantitative"},
            "y": {"aggregate": "count", "type": "quantitative"},
        },
    }
