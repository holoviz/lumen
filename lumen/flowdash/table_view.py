"""
A table of a Lumen pipeline's rows as a FlowDash view node, drawn by Lumen's
``Table`` view (Panel's ``Tabulator``).

The node draws with an empty spec. The spec holds a few Tabulator options,
each checked against the pipeline's columns before the table is drawn, and
the inspector's form shows the common ones (``TABLE_SETTINGS``) beside the
JSON. Formatters are limited to Tabulator's built-in ones that draw plain
values (``FORMATTERS``), with parameters that hold no HTML, since a spec may
come from the agent.
"""
from __future__ import annotations

import copy

from typing import Any

import pandas as pd

from .nodes import (
    LumenViewNode, _preview_frame, register_view, spec_param,
)

try:
    from panel_flowdash import SpecSetting
except ImportError:  # panel-flowdash without spec settings
    SpecSetting = None

TABLE_VIEW_ID = "Lumen/table_view"

# The column layouts the form offers; the spec may name any of Tabulator's (``TABULATOR_LAYOUTS``).
LAYOUTS = {
    "Fit data": "fit_data_fill",
    "Fill tile": "fit_columns",
    "Stretch last column": "fit_data_stretch",
}

TABULATOR_LAYOUTS = ("fit_data", "fit_data_fill", "fit_data_stretch", "fit_data_table", "fit_columns")

# Tabulator's formatters that draw a cell's value without interpreting it as HTML or a URL.
FORMATTERS = ("plaintext", "textarea", "money", "progress", "star", "tickCross", "traffic", "color")

DIRECTIONS = ("asc", "desc")

# Spec options and what each holds; the spec's other keys are refused.
OPTIONS = {
    "page_size": "rows per page, or leave it out to scroll",
    "show_index": "whether to show the index column (default false)",
    "layout": f"how columns are sized: {', '.join(TABULATOR_LAYOUTS)}",
    "header_filters": "whether each column header has a filter",
    "frozen_columns": "columns kept in view while scrolling sideways",
    "hidden_columns": "columns not shown",
    "sorters": 'the initial sort, e.g. [{"field": "amount", "dir": "desc"}]',
    "formatters": 'a formatter per column, a name or {"type": name, ...params}',
    "titles": "a header per column",
    "widths": "a width in pixels per column",
}

DEFAULTS = {"show_index": False, "layout": "fit_data_fill"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _plain(value: Any) -> bool:
    """Whether a value Tabulator writes into the page as HTML holds none."""
    return not isinstance(value, str) or not any(c in value for c in "<>&")


def _columns_of(spec: dict[str, Any], key: str) -> list[str]:
    value = spec.get(key)
    if key in ("frozen_columns", "hidden_columns"):
        return list(value) if isinstance(value, list) else []
    if key == "sorters":
        return [s["field"] for s in value if isinstance(s, dict) and "field" in s] if isinstance(value, list) else []
    return list(value) if isinstance(value, dict) else []


_COLUMN_KEYS = ("frozen_columns", "hidden_columns", "sorters", "formatters", "titles", "widths")


def table_fields(spec: Any) -> set[str]:
    """The columns a table spec names; the table shows its pipeline's other columns as they are."""
    if not isinstance(spec, dict):
        return set()
    return {str(c) for key in _COLUMN_KEYS for c in _columns_of(spec, key)}


def _check_formatter(column: str, formatter: Any) -> None:
    name = formatter.get("type") if isinstance(formatter, dict) else formatter
    if name not in FORMATTERS:
        raise ValueError(
            f"The formatter of {column!r} is {name!r}; use one of {', '.join(FORMATTERS)}, "
            'as a name or as {"type": name, ...params}.'
        )
    if isinstance(formatter, dict):
        for key, value in formatter.items():
            if not isinstance(value, (str, int, float, bool)) or not _plain(value):
                raise ValueError(f"The formatter of {column!r} has {key}={value!r}; params are numbers, true/false or text without HTML.")


def check_table(spec: dict[str, Any] | None, columns: list[str]) -> dict[str, Any] | None:
    """
    ``spec`` if it is a table spec naming only ``columns``, or ``ValueError``
    saying what to fix. No spec is an empty one.
    """
    if spec is None:
        return spec
    if not isinstance(spec, dict):
        raise ValueError("A table spec is an object of Tabulator options.")
    unknown = [key for key in spec if key not in OPTIONS]
    if unknown:
        listed = "; ".join(f"{key}: {text}" for key, text in OPTIONS.items())
        raise ValueError(f"Unknown table option(s) {', '.join(map(repr, unknown))}. The options are {listed}.")
    page_size = spec.get("page_size")
    if page_size is not None and (not _is_int(page_size) or page_size < 1):
        raise ValueError(f"page_size is {page_size!r}; it must be a whole number of rows, at least 1.")
    for key in ("show_index", "header_filters"):
        if key in spec and not isinstance(spec[key], bool):
            raise ValueError(f"{key} is {spec[key]!r}; it must be true or false.")
    if "layout" in spec and spec["layout"] not in TABULATOR_LAYOUTS:
        raise ValueError(f"layout is {spec['layout']!r}; it must be one of {', '.join(TABULATOR_LAYOUTS)}.")
    for key in ("frozen_columns", "hidden_columns"):
        if key in spec and not (isinstance(spec[key], list) and all(isinstance(c, str) for c in spec[key])):
            raise ValueError(f"{key} must be a list of column names.")
    for key in ("formatters", "titles", "widths"):
        if key in spec and not isinstance(spec[key], dict):
            raise ValueError(f"{key} must map column names to values.")
    sorters = spec.get("sorters", [])
    if not isinstance(sorters, list) or not all(
        isinstance(s, dict) and isinstance(s.get("field"), str) and s.get("dir", "asc") in DIRECTIONS
        and set(s) <= {"field", "dir"} for s in sorters
    ):
        raise ValueError('sorters must be a list like [{"field": "amount", "dir": "desc"}], dir "asc" or "desc".')
    for column, formatter in (spec.get("formatters") or {}).items():
        _check_formatter(column, formatter)
    for column, title in (spec.get("titles") or {}).items():
        if not isinstance(title, str) or not _plain(title):
            raise ValueError(f"The title of {column!r} must be text without HTML.")
    for column, width in (spec.get("widths") or {}).items():
        if not _is_int(width) or width < 1:
            raise ValueError(f"The width of {column!r} is {width!r}; it must be a whole number of pixels.")
    if columns:
        missing = sorted(table_fields(spec) - set(columns))
        if missing:
            raise ValueError(
                f"No column(s) {', '.join(map(repr, missing))} in the input; its columns are {', '.join(columns)}."
            )
    return spec


def tabulator_options(spec: dict[str, Any]) -> dict[str, Any]:
    """The ``Tabulator`` params of a checked table spec: pages only when it sets a page size."""
    options = {**DEFAULTS, **copy.deepcopy(spec)}
    if options.get("page_size"):
        options["pagination"] = "remote"
    else:
        options.pop("page_size", None)
    return options


def frozen_column(spec: dict) -> Any:
    """The one frozen column, or the list, which the select cannot show."""
    frozen = spec.get("frozen_columns")
    if isinstance(frozen, list) and len(frozen) == 1:
        return frozen[0]
    return frozen or None


def set_frozen_column(spec: dict, value: Any) -> None:
    if value is None:
        spec.pop("frozen_columns", None)
    else:
        spec["frozen_columns"] = [value]


def _settings() -> list:
    if SpecSetting is None:
        return []
    return [
        SpecSetting("page_size", label="Rows per page", widget="integer", group="Table",
                    doc="Show the rows in pages of this many; leave it empty to scroll."),
        SpecSetting("layout", label="Column widths", widget="select", options=LAYOUTS,
                    default=DEFAULTS["layout"], group="Table", restrict=False),
        SpecSetting("frozen_columns", label="Freeze column", widget="select", group="Table",
                    choices="columns", of="pipeline", restrict=False,
                    get=frozen_column, set=set_frozen_column,
                    doc="A column kept in view while the table scrolls sideways."),
        SpecSetting("show_index", label="Index column", widget="switch",
                    default=DEFAULTS["show_index"], group="Table"),
        SpecSetting("header_filters", label="Column filters", widget="switch", default=False,
                    group="Table", doc="A filter under each column header."),
    ]


#: The settings of a table's spec, as the inspector's form shows them.
TABLE_SETTINGS = _settings()


@register_view(
    title="Lumen table view", icon="table_view",
    description=(
        "A table of a Lumen pipeline's rows. It draws with an empty spec; the spec can set "
        "Tabulator options: " + "; ".join(f"{key}: {text}" for key, text in OPTIONS.items()) + "."
    ),
    tags=["table"], default_size={"height": 400}, min_size={"height": 200},
)
class TableViewNode(LumenViewNode):

    spec = spec_param("""
        Tabulator options for the table, e.g. a page size, frozen columns or
        column formatters. An empty spec shows every column.""", settings=TABLE_SETTINGS)

    view_type = "table"

    requires_spec = False

    check_spec = staticmethod(check_table)

    spec_fields = staticmethod(table_fields)

    def _view_params(self, spec: dict[str, Any]) -> dict[str, Any]:
        return {**tabulator_options(spec), "sizing_mode": "stretch_both"}

    def _summary(self, spec: dict[str, Any], data: Any) -> str:
        if not isinstance(data, pd.DataFrame):
            return "Table"
        shown = [c for c in data.columns if c not in set(spec.get("hidden_columns") or ())]
        rows = len(data)
        return (
            f"Table, {rows:,} row{'' if rows == 1 else 's'}, "
            f"{len(shown)} column{'' if len(shown) == 1 else 's'}"
        )

    def _preview(self, spec: dict[str, Any], data: Any) -> Any:
        """The view's first rows as the table shows them: without hidden columns, under their titles."""
        preview = super()._preview(spec, data)
        if not isinstance(preview, pd.DataFrame):
            return preview
        count = preview.shape[0]
        hidden = [c for c in spec.get("hidden_columns") or () if c in preview.columns]
        frame = pd.DataFrame(preview).drop(columns=hidden).rename(columns=spec.get("titles") or {})
        return _preview_frame(frame, count)


__all__ = [
    "FORMATTERS",
    "LAYOUTS",
    "TABLE_SETTINGS",
    "TABLE_VIEW_ID",
    "TABULATOR_LAYOUTS",
    "TableViewNode",
    "check_table",
    "table_fields",
    "tabulator_options",
]
