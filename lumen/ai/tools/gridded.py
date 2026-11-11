"""
SQL, for the engine gridded tables run on, for the operations questions about grids keep asking for: a time slice,
an area-weighted mean, a climatology, anomalies from it, and unit
conversions. Each tool writes the query for a table's real dimension
names, calendar and units, which an LLM gets wrong more often than the
arithmetic: latitude named ``latitude`` rather than ``lat``, cftime dates,
Kelvin read as Celsius.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ...sources.xarray_sql import XArraySQLSource
from .base import FunctionTool, define_tool

#: Factor and offset from a unit to another, value_to = value * factor + offset
_CONVERSIONS: dict[tuple[str, str], tuple[float, float]] = {
    ("K", "degC"): (1, -273.15),
    ("degC", "K"): (1, 273.15),
    ("Pa", "hPa"): (0.01, 0),
    ("hPa", "Pa"): (100, 0),
    ("m", "mm"): (1000, 0),
    ("kg m-2 s-1", "mm day-1"): (86400, 0),
    ("m s-1", "km h-1"): (3.6, 0),
}

#: Spellings CF files use for the units above
_ALIASES = {
    "k": "K", "degk": "K", "kelvin": "K", "degc": "degC", "celsius": "degC", "deg_c": "degC",
    "°c": "degC", "pa": "Pa", "hpa": "hPa", "mbar": "hPa", "m": "m", "mm": "mm",
    "kg m-2 s-1": "kg m-2 s-1", "kg m**-2 s**-1": "kg m-2 s-1", "mm/day": "mm day-1", "mm day-1": "mm day-1",
    "m s-1": "m s-1", "m s**-1": "m s-1", "m/s": "m s-1", "km/h": "km h-1", "km h-1": "km h-1",
}

# Calendar quarters (JFM, AMJ, ...), not meteorological seasons (DJF, MAM, ...)
_PERIODS = {"month": "MONTH", "quarter": "QUARTER", "dayofyear": "DOY", "hour": "HOUR"}


class _Grid:
    """A gridded table's dimension and variable names, calendar and units."""

    def __init__(self, source: XArraySQLSource, table: str):
        ds = source._dataset
        var = ds[table]
        self.table, self.variable = table, table
        self.units = var.attrs.get("units", "")

        def find(units: set[str], names: set[str]) -> str | None:
            for dim in var.dims:
                attrs = ds.coords[dim].attrs if dim in ds.coords else {}
                if attrs.get("units") in units or attrs.get("standard_name") in names or dim in names:
                    return dim
            return None

        self.lat = find({"degrees_north", "degree_north", "degrees_N"}, {"latitude", "lat"})
        self.lon = find({"degrees_east", "degree_east", "degrees_E"}, {"longitude", "lon"})
        self.time = next((
            dim for dim in var.dims if dim in ds.coords
            and (np.issubdtype(ds[dim].dtype, np.datetime64) or ds[dim].dtype == object)
        ), None)
        self.calendar = None
        if self.time is not None:
            try:
                self.calendar = ds[self.time].dt.calendar
            except (AttributeError, TypeError):
                pass
        self.dims = list(var.dims)

    @property
    def dated(self) -> bool:
        """Whether SQL sees the time column as timestamps, so dates and EXTRACT work."""
        # Only these: xarray-sql stores other calendars' dates as time elapsed
        # in that calendar, so a noleap 2050-01-01 reads as 2049-12-12 in SQL
        return self.time is not None and self.calendar in (None, "standard", "gregorian", "proleptic_gregorian")

    def need(self, *dims: str) -> str | None:
        """A message naming what the grid lacks for an operation, or None."""
        missing = [name for name in dims if getattr(self, name) is None]
        if missing:
            return f"{self.table} has no {' or '.join(missing)} dimension (its dimensions: {', '.join(self.dims)})."
        if "time" in dims and not self.dated:
            return (
                f"{self.table}'s {self.time} uses the {self.calendar} calendar, which SQL sees as numbers, "
                "not dates, so periods cannot be taken from it."
            )
        return None


def _quote(name: str) -> str:
    return f'"{name}"'


def gridded_tools(sources: list[XArraySQLSource]) -> list[FunctionTool]:
    """The gridded SQL tools for these sources' tables."""
    # Only data variables: tables derived by a query have no grid attributes to read
    tables = {
        table: source for source in sources for table in source.get_tables() if table in source._dataset.data_vars
    }

    def grid(table: str) -> _Grid | str:
        if table not in tables:
            return f"No gridded table {table!r}; the gridded tables are {', '.join(sorted(tables))}."
        return _Grid(tables[table], table)

    def time_filter(g: _Grid, start: str | None, end: str | None) -> str:
        bounds = []
        if start:
            bounds.append(f"{_quote(g.time)} >= TIMESTAMP '{start}'")
        if end:
            bounds.append(f"{_quote(g.time)} < TIMESTAMP '{end}'")
        return f" WHERE {' AND '.join(bounds)}" if bounds else ""

    @define_tool(purpose="SQL selecting a gridded table between two dates, every grid cell, end excluded.")
    def time_slice(table: str, start: str, end: str) -> str:
        """SQL selecting a gridded table's cells between two dates (ISO, e.g. 2014-02-24), end excluded."""
        g = grid(table)
        if isinstance(g, str):
            return g
        if problem := g.need("time"):
            return problem
        columns = ", ".join(_quote(d) for d in [*g.dims, g.variable])
        return f"SELECT {columns} FROM {_quote(g.table)}{time_filter(g, start, end)} ORDER BY {_quote(g.time)}"

    @define_tool(purpose="SQL for the area-weighted mean of a gridded variable, per time step or overall.")
    def area_weighted_mean(table: str, start: str | None = None, end: str | None = None, per_time: bool = True) -> str:
        """
        SQL for the cos(latitude)-weighted mean of a gridded variable over its
        grid, per time step or over the whole period, optionally between two dates.
        """
        g = grid(table)
        if isinstance(g, str):
            return g
        if problem := g.need("lat") or ((start or end or per_time) and g.need("time")):
            return problem
        weight = f"COS(RADIANS({_quote(g.lat)}))"
        mean = f"SUM({_quote(g.variable)} * {weight}) / SUM({weight}) AS {_quote(g.variable + '_weighted_mean')}"
        where = time_filter(g, start, end) if g.time else ""
        if per_time:
            return (
                f"SELECT {_quote(g.time)}, {mean} FROM {_quote(g.table)}{where} "
                f"GROUP BY {_quote(g.time)} ORDER BY {_quote(g.time)}"
            )
        return f"SELECT {mean} FROM {_quote(g.table)}{where}"

    @define_tool(purpose="SQL for the climatology of a gridded variable: its mean per month, quarter, day of year or hour, per cell.")
    def climatology(table: str, period: Literal["month", "quarter", "dayofyear", "hour"] = "month") -> str:
        """SQL for a gridded variable's mean per month (or calendar quarter, day of year, hour) at each grid cell."""
        g = grid(table)
        if isinstance(g, str):
            return g
        if problem := g.need("time"):
            return problem
        part = f"EXTRACT({_PERIODS[period]} FROM {_quote(g.time)})"
        cells = [_quote(d) for d in g.dims if d != g.time]
        # Keys written out: gridded tables run on DataFusion, which has no GROUP BY ALL
        keys = ", ".join([part, *cells])
        return (
            f"SELECT {part} AS {period}, {', '.join(cells)}, AVG({_quote(g.variable)}) "
            f"AS {_quote(g.variable + '_climatology')} FROM {_quote(g.table)} GROUP BY {keys} ORDER BY {keys}"
        )

    @define_tool(purpose="SQL for anomalies of a gridded variable: each value minus its cell's climatology for that period.")
    def anomaly(table: str, period: Literal["month", "quarter", "dayofyear", "hour"] = "month") -> str:
        """SQL for each value of a gridded variable minus the mean of its cell for the same month (or other period)."""
        g = grid(table)
        if isinstance(g, str):
            return g
        if problem := g.need("time"):
            return problem
        cells = [_quote(d) for d in g.dims if d != g.time]
        partition = ", ".join([*cells, f"EXTRACT({_PERIODS[period]} FROM {_quote(g.time)})"])
        columns = ", ".join(_quote(d) for d in g.dims)
        return (
            f"SELECT {columns}, {_quote(g.variable)} - AVG({_quote(g.variable)}) OVER (PARTITION BY {partition}) "
            f"AS {_quote(g.variable + '_anomaly')} FROM {_quote(g.table)} ORDER BY {columns}"
        )

    @define_tool(purpose="SQL converting a gridded variable to other units, e.g. Kelvin to degC, Pa to hPa, kg m-2 s-1 to mm day-1.")
    def convert_units(table: str, to: str) -> str:
        """SQL selecting a gridded variable converted from its CF units to the units asked for, e.g. degC."""
        g = grid(table)
        if isinstance(g, str):
            return g
        source_units = _ALIASES.get(g.units.lower(), g.units)
        target = _ALIASES.get(to.lower(), to)
        if (source_units, target) not in _CONVERSIONS:
            known = ", ".join(f"{a} to {b}" for a, b in _CONVERSIONS)
            return f"{table} is in {g.units or 'unknown units'}; converting to {to} is not one of: {known}."
        factor, offset = _CONVERSIONS[source_units, target]
        expr = _quote(g.variable)
        if factor != 1:
            expr = f"{expr} * {factor:g}"
        if offset:
            expr = f"{expr} {'+' if offset > 0 else '-'} {abs(offset):g}"
        columns = ", ".join(_quote(d) for d in g.dims)
        name = f"{g.variable}_{target.replace(' ', '_').replace('-', '')}"
        return f"SELECT {columns}, {expr} AS {_quote(name)} FROM {_quote(g.table)}"

    return [FunctionTool(fn) for fn in (time_slice, area_weighted_mean, climatology, anomaly, convert_units)]


def gridded_sql_tools(context: dict[str, Any]) -> list[FunctionTool] | None:
    """An ``llm_tools`` entry: the gridded SQL tools, when a source in the context is gridded."""
    sources = [s for s in context.get("sources", []) if isinstance(s, XArraySQLSource)]
    return gridded_tools(sources) if sources else None
