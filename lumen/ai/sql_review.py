"""
A review of the SQL an LLM writes against maps and grids, for the mistakes
that run without error and give a wrong answer: distances and areas in
degrees, spatial joins without a spatial predicate, and temperatures
compared in the wrong unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot

from sqlglot import exp

#: Functions whose result is in the units of the geometry's CRS
_MEASURES = {"ST_AREA", "ST_LENGTH", "ST_PERIMETER", "ST_DISTANCE"}

#: Functions whose distance argument is in the units of the geometry's CRS
_DISTANCES = {"ST_BUFFER", "ST_DWITHIN"}

#: Functions that read points as (lat, lon), so lon/lat geometry has to be flipped first
_LAT_LON = {
    "ST_AREA_SPHEROID", "ST_LENGTH_SPHEROID", "ST_PERIMETER_SPHEROID", "ST_DISTANCE_SPHEROID",
    "ST_DISTANCE_SPHERE", "ST_DWITHIN_SPHEROID",
}

#: Predicates that relate two geometries in a join
_PREDICATES = {
    "ST_INTERSECTS", "ST_CONTAINS", "ST_WITHIN", "ST_DWITHIN", "ST_TOUCHES", "ST_OVERLAPS",
    "ST_COVERS", "ST_COVEREDBY", "ST_CROSSES", "ST_DISJOINT", "ST_EQUALS", "ST_DWITHIN_SPHEROID",
}

#: How the functions above are usually written, for the findings
_SPELLING = {
    "ST_AREA": "ST_Area", "ST_LENGTH": "ST_Length", "ST_PERIMETER": "ST_Perimeter",
    "ST_DISTANCE": "ST_Distance", "ST_BUFFER": "ST_Buffer", "ST_DWITHIN": "ST_DWithin",
}

_KELVIN = {"k", "degk", "kelvin", "kelvins", "deg_k"}
_CELSIUS = {"degc", "celsius", "deg_c", "c", "°c", "degree_celsius", "degrees_celsius"}


@dataclass
class TableFacts:
    """What the review needs to know about one table's columns."""

    #: Geometry columns and whether their CRS is geographic (degrees)
    geometry: dict[str, bool] = field(default_factory=dict)
    #: CF units of the columns that declare them
    units: dict[str, str] = field(default_factory=dict)


def geographic_crs(column_type: str) -> bool:
    """
    Whether a geometry column type such as GEOMETRY('EPSG:4326') has a
    geographic CRS, in degrees. False when it names no CRS or one that is
    not geographic.
    """
    start, end = column_type.find("("), column_type.rfind(")")
    if start == -1 or end <= start:
        return False
    try:
        # Deferred: pyproj comes with geopandas, which plain tables do without
        from pyproj import CRS
        return CRS.from_user_input(column_type[start + 1:end].strip("'\" ")).is_geographic
    except Exception:
        return False


def _name(node: exp.Expression) -> str:
    if isinstance(node, exp.StDistance):
        return "ST_DISTANCE"
    return node.name.upper() if isinstance(node, exp.Anonymous) else ""


def _args(node: exp.Expression) -> list[exp.Expression]:
    if isinstance(node, exp.StDistance):
        return [node.this, node.expression]
    return list(node.expressions)


def _geographic_column(arg: exp.Expression, geographic: set[str]) -> str | None:
    """The geographic geometry column an argument is, as it is, not transformed."""
    return arg.name if isinstance(arg, exp.Column) and arg.name in geographic else None


def _number(node: exp.Expression | None) -> float | None:
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal) and node.this.is_number:
        return -float(node.this.this)
    if isinstance(node, exp.Literal) and node.is_number:
        return float(node.this)
    return None


def review_sql(sql: str, tables: dict[str, TableFacts], dialect: str | None = "duckdb") -> list[str]:
    """
    The problems found in a query, each as a sentence the LLM can act on:
    a buffer or distance in degrees on longitude and latitude, an area or
    length that comes out in degrees, a join of two geometry tables with no
    spatial predicate, and a Kelvin column compared with what looks like a
    Celsius value or the other way round. Empty when nothing is found or the
    query does not parse.
    """
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError:
        return []
    geographic = {col for facts in tables.values() for col, is_geo in facts.geometry.items() if is_geo}
    findings: list[str] = []
    for node in tree.find_all(exp.Anonymous, exp.StDistance):
        name = _name(node)
        args = _args(node)
        if name in _DISTANCES and len(args) >= 2 and (col := _geographic_column(args[0], geographic)):
            # The distance is the last argument: ST_Buffer(g, d), ST_DWithin(g1, g2, d)
            distance = _number(args[-1])
            if distance is not None and distance > 1:
                findings.append(
                    f"{_SPELLING[name]}({col}, ..., {args[-1].sql()}) works in degrees: {col} is in longitude and latitude, so "
                    f"{args[-1].sql()} means {args[-1].sql()} degrees, not metres. Buffer in a projected CRS in "
                    f"metres, e.g. ST_Transform({col}, 'EPSG:4326', 'EPSG:6933', always_xy := true) or a local UTM "
                    "zone (not EPSG:3857, whose metres shrink away from the equator), or filter on "
                    f"ST_Distance_Sphere(ST_FlipCoordinates({col}), ST_FlipCoordinates(other)) <= metres."
                )
        elif name in _MEASURES and args and (col := _geographic_column(args[0], geographic)):
            unit = "square degrees" if name == "ST_AREA" else "degrees"
            # The sphere and spheroid functions read (lat, lon), so lon/lat
            # geometry is flipped first; unflipped the result looks right but is not
            fix = {
                "ST_AREA": f"ST_Area(ST_Transform({col}, 'EPSG:4326', 'EPSG:6933', always_xy := true))",
                "ST_LENGTH": f"ST_Length_Spheroid(ST_FlipCoordinates({col}))",
                "ST_PERIMETER": f"ST_Perimeter_Spheroid(ST_FlipCoordinates({col}))",
                "ST_DISTANCE": f"ST_Distance_Sphere(ST_FlipCoordinates({col}), ST_FlipCoordinates(other))",
            }[name]
            findings.append(
                f"{_SPELLING[name]}({col}) returns {unit}, since {col} is in longitude and latitude. Use "
                f"{fix} for metres."
            )
        elif name in _LAT_LON and args and (col := _geographic_column(args[0], geographic)):
            findings.append(
                f"{node.name}({col}) reads points as (latitude, longitude), but {col} is longitude first, "
                f"so the result is wrong though it looks plausible. Flip it: {node.name}(ST_FlipCoordinates({col}), ...)."
            )
    findings += _unpredicated_joins(tree, tables)
    findings += _unit_mismatches(tree, tables)
    return findings


def _spatial(condition: exp.Expression | None) -> bool:
    """Whether a join condition relates geometries: a spatial predicate, or a distance compared with a threshold."""
    if condition is None:
        return False
    if any(_name(f) in _PREDICATES for f in condition.find_all(exp.Anonymous)):
        return True
    # e.g. ST_Distance_Sphere(a, b) <= 750000, the within-distance join
    return any(
        _name(f).startswith(("ST_DISTANCE", "ST_DWITHIN"))
        for comparison in condition.find_all(exp.LT, exp.LTE, exp.GT, exp.GTE)
        for f in comparison.find_all(exp.Anonymous, exp.StDistance)
    )


def _unpredicated_joins(tree: exp.Expression, tables: dict[str, TableFacts]) -> list[str]:
    """Joins of two tables that both have geometry, with neither a key nor a spatial predicate."""
    with_geometry = {name for name, facts in tables.items() if facts.geometry}
    findings = []
    for select in tree.find_all(exp.Select):
        joins = select.args.get("joins") or []
        # sqlglot 26 and later call it from_
        from_ = select.args.get("from_") or select.args.get("from")
        if not joins or from_ is None:
            continue
        first = from_.this
        if not isinstance(first, exp.Table) or first.name not in with_geometry:
            continue
        where = select.args.get("where")
        spatial_where = _spatial(where)
        for join in joins:
            other = join.this
            if not isinstance(other, exp.Table) or other.name not in with_geometry:
                continue
            on, using = join.args.get("on"), join.args.get("using")
            spatial_on = _spatial(on)
            if spatial_on or spatial_where or using or (on is not None and not spatial_on and on.find(exp.EQ)):
                continue
            findings.append(
                f"{first.name} and {other.name} both have geometry but are joined without a spatial "
                "predicate, which pairs every row with every row. Join ON ST_Intersects, ST_Within or "
                "ST_Contains of their geometries."
            )
    return findings


def _unit_mismatches(tree: exp.Expression, tables: dict[str, TableFacts]) -> list[str]:
    """A Kelvin column compared with a value that looks like Celsius, or the other way round."""
    # str: a units attribute need not be a string
    units = {col: str(unit).lower() for facts in tables.values() for col, unit in facts.units.items()}
    findings = []
    seen = set()
    for comparison in tree.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.NEQ):
        column, value = comparison.this, _number(comparison.expression)
        if not isinstance(column, exp.Column):
            # Written the other way round, e.g. 25 < air
            column, value = comparison.expression, _number(comparison.this)
        if not isinstance(column, exp.Column) or value is None or column.name not in units:
            continue
        unit = units[column.name]
        if unit in _KELVIN and -90 <= value <= 60 and column.name not in seen:
            seen.add(column.name)
            findings.append(
                f"{column.name} is in {unit}, so {value:g} looks like degrees Celsius ({value + 273.15:g} K). "
                f"Compare with {value + 273.15:g}, or convert the column with {column.name} - 273.15."
            )
        elif unit in _CELSIUS and value > 150 and column.name not in seen:
            seen.add(column.name)
            findings.append(
                f"{column.name} is in {unit}, so {value:g} looks like Kelvin ({value - 273.15:g} °C). "
                f"Compare with {value - 273.15:g} instead."
            )
    return findings
