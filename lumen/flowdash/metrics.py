"""
Metrics: named, described queries over one host source, defined once in the
``metrics:`` section of the project file and shared by dashboards and the
agent.

A metric takes one of two forms. The structured form names a table and the
measures and dimensions a dashboard may pick; a ``Lumen/metric`` node
compiles the picked ones into ``SELECT dims, measures FROM table WHERE ...
GROUP BY dims``::

    metrics:
      capacity:
        description: Installed wind capacity.
        source: turbines            # a host source ref, as Lumen/source names it
        table: turbines_csv
        measures:
          capacity_mw: {expr: "SUM(t_cap) / 1000", format: ",.1f", description: Rated capacity in MW}
          turbines: "COUNT(*)"
        dimensions:
          state: {column: t_state, description: Two-letter state code}
          online: {column: online_date, type: time, grains: [year, month]}
        parameters:
          since: {type: integer, default: 2000, column: p_year, op: ">="}

The SQL form is one fixed read-only query with ``:name`` placeholders, for
what the structured form cannot say, such as rankings, windows and joins::

      top_states:
        source: turbines
        sql: SELECT t_state, SUM(t_cap) AS kw FROM turbines_csv GROUP BY t_state ORDER BY kw DESC LIMIT :n
        parameters:
          n: {type: integer, default: 10, bounds: [1, 50]}

Parameters have a ``type`` (``string``, ``number``, ``integer``,
``boolean``, ``date``, ``datetime``, ``list[<type>]`` or ``range[<type>]``),
and optionally a ``default``, ``required``, ``bounds`` and ``options``. In
the structured form a parameter filters ``column`` with ``op`` before
aggregating, and every column dimension can be filtered by its name too; in
the SQL form a parameter is a value bound to its placeholder. Values are
coerced to the declared type and bound as sqlglot literals, never formatted
into the SQL text. A value of None takes the parameter's default. A
structured metric's ``limit`` and ``order_by`` are parameters as well, see
``Metric.knobs``; a SQL metric's parameters each need a default or
``required: true``. ``format`` is a d3-format string.

A time dimension (``type: time``) is truncated to the grain a node picks;
period comparisons will be keys of the time dimension, so the format leaves
room for them.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import difflib
import functools
import logging
import math
import re
import threading
import time
import weakref

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import param
import sqlglot

from sqlglot import exp

from ..sources.base import BaseSQLSource, Source
from ..transforms.sql import NO_BOOLEAN_LITERALS, MetricAggregate, SQLPreFilter
from .project import (
    Problem, Project, ProjectError, ProjectSection, Section, register_section,
)

SECTION = "metrics"

_UNREAD = object()

logger = logging.getLogger(__name__)

# The key of the editor context the host puts its ``MetricSet`` in.
METRICS_KEY = "lumen.metrics"

SCALAR_TYPES = ("string", "number", "integer", "boolean", "date", "datetime")
RANGE_TYPES = ("number", "integer", "date", "datetime")
GRAINS = ("year", "quarter", "month", "week", "day", "hour")

# The parameters that shape a structured metric's result, see ``Metric.knobs``.
KNOBS = ("limit", "order_by")

OPS = {"=", "!=", ">", ">=", "<", "<=", "in", "between"}
_COMPARISONS = {"=": exp.EQ, "!=": exp.NEQ, ">": exp.GT, ">=": exp.GTE, "<": exp.LT, "<=": exp.LTE}

# d3-format's specifier: [[fill]align][sign][symbol][0][width][,][.precision][~][type]
_D3_FORMAT = re.compile(r"(?:.?[<>=^])?[-+( ]?[$#]?0?\d*,?(?:\.\d+)?~?[eEfgGrsopxXbcdn%]?")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TYPE = re.compile(r"(list|range)\[(\w+)\]|(\w+)")

_METRIC_KEYS = {"description", "source", "table", "sql", "measures", "dimensions", "parameters"}
_MEASURE_KEYS = {"expr", "format", "description"}
_DIMENSION_KEYS = {"column", "expr", "type", "grains", "format", "description"}
_PARAMETER_KEYS = {"type", "default", "required", "bounds", "options", "description", "column", "op"}


class MetricError(ValueError):
    """A problem with a metric or the values given for it, with the config field it concerns."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


# ---------------------------------------------------------------------------
# Parameters

def _plain(value: Any) -> Any:
    """``value`` with numpy and pandas scalars as Python ones."""
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if hasattr(value, "item") and type(value).__module__ == "numpy":
        return value.item()
    return value


def coerce_scalar(kind: str, value: Any) -> Any:
    """``value`` as a Python value of parameter type ``kind``, or ``ValueError``."""
    value = _plain(value)
    if kind == "string":
        if isinstance(value, str):
            return value
    elif kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
    elif kind == "integer":
        if isinstance(value, str) and re.fullmatch(r"\s*[+-]?\d+\s*", value):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif kind == "number":
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return value
    elif kind == "date":
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        if isinstance(value, str):
            try:
                return dt.date.fromisoformat(value.strip())
            except ValueError:
                pass
    elif kind == "datetime":
        if isinstance(value, dt.datetime):
            return value
        if isinstance(value, dt.date):
            return dt.datetime(value.year, value.month, value.day)
        if isinstance(value, str):
            try:
                return dt.datetime.fromisoformat(value.strip())
            except ValueError:
                pass
    raise ValueError(f"expected {'an' if kind[0] in 'ai' else 'a'} {kind}, got {value!r}")


@dataclass(frozen=True)
class Parameter:
    """A typed value a metric takes, e.g. a year to start from or how many rows to rank."""

    name: str
    kind: str
    item: str | None = None
    default: Any = None
    required: bool = False
    bounds: tuple[Any, Any] | None = None
    options: tuple[Any, ...] | None = None
    description: str = ""
    column: str | None = None
    op: str | None = None

    @property
    def type(self) -> str:
        return f"{self.kind}[{self.item}]" if self.item else self.kind

    def _check_item(self, value: Any) -> Any:
        value = coerce_scalar(self.item or self.kind, value)
        if self.options is not None and value not in self.options:
            raise ValueError(f"{value!r} is not one of {', '.join(map(repr, self.options))}")
        if self.bounds is not None:
            low, high = self.bounds
            if (low is not None and value < low) or (high is not None and value > high):
                raise ValueError(f"{value!r} is outside the bounds [{low}, {high}]")
        return value

    def coerce(self, value: Any) -> Any:
        """``value`` as this parameter's type, within its bounds and options, or ``ValueError``."""
        if value is None:
            return None
        if self.kind == "list":
            values = list(value) if isinstance(value, (list, tuple, set)) else [value]
            return [self._check_item(v) for v in values]
        if self.kind == "range":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"expected a [start, end] range, got {value!r}")
            low, high = (None if v is None else self._check_item(v) for v in value)
            if low is not None and high is not None and low > high:
                raise ValueError(f"the range starts after it ends: {value!r}")
            return (low, high)
        return self._check_item(value)

    def describe(self) -> str:
        text = self.type
        if self.default is not None:
            text += f", default {json_value(self.default)}"
        if self.required:
            text += ", required"
        if self.bounds is not None:
            text += f", from {json_value(self.bounds[0])} to {json_value(self.bounds[1])}"
        if self.options is not None:
            text += f", one of {', '.join(map(json_value, self.options))}"
        if self.column:
            text += f", filters {self.column} {self.op}"
        return text


def json_value(value: Any) -> str:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, tuple):
        return f"[{', '.join(json_value(v) for v in value)}]"
    return repr(value) if isinstance(value, str) else str(value)


def literal(kind: str, value: Any, dialect: str | None) -> exp.Expression:
    """A sqlglot literal of ``value``, already coerced to the scalar type ``kind``."""
    if kind in ("integer", "number"):
        return exp.Literal.number(repr(value) if isinstance(value, float) else str(value))
    if kind == "boolean":
        return exp.Boolean(this=value) if dialect not in NO_BOOLEAN_LITERALS else exp.Literal.number(int(value))
    if kind == "date":
        text = exp.Literal.string(value.isoformat())
        return text if dialect == "sqlite" else exp.Cast(this=text, to=exp.DataType.build("DATE"))
    if kind == "datetime":
        text = exp.Literal.string(value.isoformat(sep=" "))
        return text if dialect == "sqlite" else exp.Cast(this=text, to=exp.DataType.build("TIMESTAMP"))
    return exp.Literal.string(value)


def _column(name: str) -> exp.Column:
    return exp.Column(this=exp.Identifier(this=name, quoted=True))


def condition(target: exp.Expression, parameter: Parameter, value: Any, dialect: str | None) -> exp.Expression | None:
    """The condition ``parameter`` puts on ``target`` for its coerced ``value``; None filters nothing."""
    if value is None:
        return None
    item = parameter.item or parameter.kind
    if parameter.op == "in":
        if not value:
            return None
        return target.isin(*(literal(item, v, dialect) for v in value))
    if parameter.op == "between":
        low, high = value
        parts = [
            cls(this=target.copy(), expression=literal(item, bound, dialect))
            for cls, bound in ((exp.GTE, low), (exp.LTE, high)) if bound is not None
        ]
        return exp.and_(*parts) if parts else None
    return _COMPARISONS[parameter.op](this=target, expression=literal(item, value, dialect))


# ---------------------------------------------------------------------------
# Definitions

@dataclass(frozen=True)
class Measure:
    name: str
    expr: str
    format: str | None = None
    description: str = ""


@dataclass(frozen=True)
class Dimension:
    name: str
    expr: str
    column: str | None = None
    time: bool = False
    grains: tuple[str, ...] = ()
    format: str | None = None
    description: str = ""


@dataclass(frozen=True)
class Metric:
    """One metric of the project file, in the structured form (``table``) or the SQL form (``sql``)."""

    name: str
    source: str
    description: str = ""
    table: str | None = None
    sql: str | None = None
    measures: dict[str, Measure] = field(default_factory=dict)
    dimensions: dict[str, Dimension] = field(default_factory=dict)
    parameters: dict[str, Parameter] = field(default_factory=dict)

    @property
    def form(self) -> str:
        return "sql" if self.sql is not None else "structured"

    def values(self, given: Mapping[str, Any] | None) -> dict[str, Any]:
        """
        The parameter values of a query: the defaults updated with ``given``,
        coerced and checked. A value of None, or one left out, takes the
        parameter's default; a required parameter without either is refused.
        In the structured form ``given`` may also name a column dimension,
        whose value keeps the rows equal to it (or to one of a list), and
        ``limit`` and ``order_by``, which ``knobs`` reads. Raises
        ``MetricError``.
        """
        given = {key: value for key, value in (given or {}).items() if key not in self.knobs()}
        unknown = [key for key in given if key not in self.parameters and key not in self._filterable()]
        if unknown:
            known = [*self.parameters, *self._filterable(), *self.knobs()]
            raise MetricError(
                f"Metric {self.name!r} has no parameter {unknown[0]!r}; its parameters are: "
                f"{', '.join(known) or 'none'}.", "parameters",
            )
        values: dict[str, Any] = {}
        for name, parameter in self.parameters.items():
            if name in self.knobs():
                continue
            raw = given.get(name)
            raw = parameter.default if raw is None else raw
            try:
                values[name] = parameter.coerce(raw)
            except ValueError as exc:
                raise MetricError(f"Parameter {name!r} of metric {self.name!r}: {exc}.", "parameters") from None
            if values[name] is None and parameter.required:
                raise MetricError(f"Metric {self.name!r} needs a value for {name!r}.", "parameters")
        for name in given:
            if name not in self.parameters and given[name] is not None:
                values[name] = _dimension_value(self.name, name, given[name])
        return values

    def knobs(self) -> dict[str, Parameter]:
        """
        The parameters of a structured metric that shape its result rather
        than filter its rows: ``limit``, the rows it returns at most, and
        ``order_by``, the columns it is sorted by (``-`` first for
        descending). The metric may declare either to give it a default,
        bounds or options; a SQL metric has none.
        """
        if self.form == "sql":
            return {}
        return {
            "limit": self.parameters.get("limit") or Parameter(
                "limit", "integer", bounds=(1, None), description="Rows returned at most, e.g. for a top 10"),
            "order_by": self.parameters.get("order_by") or Parameter(
                "order_by", "list", "string", description="Columns to sort by, - first for descending"),
        }

    def _filterable(self) -> list[str]:
        if self.form == "sql":
            return []
        return [name for name, d in self.dimensions.items() if d.column is not None]


def _dimension_value(metric: str, name: str, value: Any) -> Any:
    value = _plain(value)
    scalars = (str, int, float, bool, dt.date)
    if value is None or isinstance(value, scalars):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(_plain(v), scalars) for v in value):
        return [_plain(v) for v in value]
    raise MetricError(f"Filter {name!r} of metric {metric!r} takes a value or a list of values, got {value!r}.",
                      "parameters")


@dataclass(frozen=True)
class MetricFailure:
    """Why a metric of the project file is left out: what checking it against its source found, and where."""

    metric: Metric
    message: str
    path: tuple = ()
    """The keys of the metric's definition at fault, e.g. ``("table",)``."""

    @property
    def kind(self) -> str:
        """What is wrong: ``source`` or ``table`` missing, a ``column`` missing, or the ``query`` refused."""
        if self.path[:1] == ("source",):
            return "source"
        if self.path[:1] == ("table",):
            return "table"
        if self.message.startswith("Unknown column"):
            return "column"
        return "query"


class MetricSet(Mapping[str, Metric]):
    """
    The metrics of a project file, by name, and where each is written;
    ``failures`` holds the ones left out because they do not work on the
    app's sources, by name.
    """

    def __init__(
        self, metrics: Mapping[str, Metric] | None = None, section: Section | None = None,
        failures: Mapping[str, MetricFailure] | None = None,
    ):
        self._metrics = dict(metrics or {})
        self.section = section
        self.failures = dict(failures or {})

    def __getitem__(self, name: str) -> Metric:
        return self._metrics[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        return f"MetricSet({list(self._metrics)})"

    def snapshot(self) -> MetricSet:
        """These metrics as one set that does not change, to read several of them together."""
        return self

    def problem(self, path: tuple, message: str) -> Problem:
        return Problem(message) if self.section is None else self.section.problem(path, message)


@functools.lru_cache(maxsize=512)
def _parse(text: str, dialect: str | None) -> exp.Expression:
    return sqlglot.parse_one(text, read=dialect)


def parse_expression(text: str, dialect: str | None = None) -> exp.Expression:
    """``text`` parsed as one SQL expression in ``dialect``, a copy the caller may change."""
    return _parse(text, dialect).copy()


def _denied() -> frozenset[str]:
    from .checks import DENIED_FUNCTIONS

    return DENIED_FUNCTIONS


def _function_name(func: exp.Func) -> str:
    return (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()


def check_expression(text: str, aggregate: bool, dialect: str | None = None) -> exp.Expression:
    """
    ``text`` parsed as one scalar expression over the table's columns, or
    ``ValueError``: no query, subquery, table, placeholder or ``*`` outside
    ``COUNT(*)``, none of the functions ``check_query`` denies, and an
    aggregate for a measure (``aggregate``) or none for a dimension.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("expected a SQL expression")
    try:
        parsed = parse_expression(text, dialect)
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError(f"does not parse: {str(exc).splitlines()[0].rstrip('.')}") from None
    if isinstance(parsed, (exp.Query, exp.Command, exp.DDL, exp.DML)) or parsed.find(exp.Query, exp.Subquery, exp.Table):
        raise ValueError("must be one expression over the table's columns, not a query")
    if parsed.find(exp.Placeholder, exp.Parameter):
        raise ValueError("may not use placeholders; give a parameter a column to filter by instead")
    for star in parsed.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):
            raise ValueError("may only use * inside COUNT(*)")
    for func in parsed.find_all(exp.Func):
        if _function_name(func) in _denied():
            raise ValueError(f"calls {_function_name(func)}(), which reads the engine's settings or environment")
    for column in parsed.find_all(exp.Column):
        if column.table:
            raise ValueError(f"names {column.sql()}; name the table's columns without a table")
    has_aggregate = parsed.find(exp.AggFunc) is not None
    if aggregate and not has_aggregate:
        raise ValueError("must aggregate the rows, e.g. SUM(column) or COUNT(*)")
    if not aggregate and (has_aggregate or parsed.find(exp.Window)):
        raise ValueError("must not aggregate; a dimension groups the rows")
    return parsed


def expression_columns(text: str, dialect: str | None = None) -> list[str]:
    return list(dict.fromkeys(c.name for c in parse_expression(text, dialect).find_all(exp.Column)))


def placeholders(sql: str, dialect: str | None = None) -> list[str]:
    return list(dict.fromkeys(p.name for p in parse_expression(sql, dialect).find_all(exp.Placeholder) if p.name))


class _Reader:
    """Reads the ``metrics:`` section into ``Metric``s, collecting every problem with its position."""

    def __init__(self, section: Section):
        self.section = section
        self.problems: list[Problem] = []

    def fail(self, path: tuple, message: str, key: bool = False) -> None:
        self.problems.append(self.section.problem(path, message, key=key))

    def mapping(self, value: Any, path: tuple, what: str, keys: set[str] | None = None) -> dict | None:
        if not isinstance(value, dict):
            self.fail(path, f"{what} must be a mapping.")
            return None
        if keys is not None:
            for key in value:
                if key not in keys:
                    close = difflib.get_close_matches(str(key), sorted(keys), n=1)
                    hint = f" Did you mean {close[0]!r}?" if close else f" Its keys are: {', '.join(sorted(keys))}."
                    self.fail((*path, key), f"Unknown key {key!r} in {what}.{hint}", key=True)
        return value

    def text(self, value: Any, path: tuple, what: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            self.fail(path, f"{what} must be text.")
            return None
        return value

    def name(self, name: Any, path: tuple, what: str) -> bool:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            self.fail(path, f"{what} {name!r} must be a name of letters, digits and underscores.")
            return False
        return True

    def format(self, value: Any, path: tuple) -> str | None:
        value = self.text(value, path, "format")
        if value is not None and not _D3_FORMAT.fullmatch(value):
            self.fail(path, f"format {value!r} is not a d3-format string, e.g. ',.1f' or '.0%'.")
            return None
        return value

    def read(self) -> MetricSet:
        section = self.section.value
        if section is None:
            return MetricSet(section=self.section)
        if not isinstance(section, dict):
            self.fail((), "`metrics` must be a mapping of metric names to definitions.")
            raise ProjectError(self.problems)
        metrics = {}
        for name, spec in section.items():
            if self.name(name, (name,), "Metric name"):
                metric = self.metric(name, spec)
                if metric is not None:
                    metrics[name] = metric
        if self.problems:
            raise ProjectError(self.problems)
        return MetricSet(metrics, self.section)

    def metric(self, name: str, spec: Any) -> Metric | None:
        path = (name,)
        before = len(self.problems)
        spec = self.mapping(spec, path, f"Metric {name!r}", _METRIC_KEYS)
        if spec is None:
            return None
        source = self.text(spec.get("source"), (*path, "source"), "source")
        if not source:
            self.fail(path, f"Metric {name!r} needs a `source`: the ref of a source the app serves.")
        description = self.text(spec.get("description"), (*path, "description"), "description") or ""
        table = self.text(spec.get("table"), (*path, "table"), "table")
        sql = self.text(spec.get("sql"), (*path, "sql"), "sql")
        if (table is None) == (sql is None):
            self.fail(path, f"Metric {name!r} needs either a `table` with measures and dimensions, or a `sql` query.")
            return None
        measures: dict[str, Measure] = {}
        dimensions: dict[str, Dimension] = {}
        if sql is not None:
            for key in ("measures", "dimensions"):
                if key in spec:
                    self.fail((*path, key), f"A SQL metric has no {key}; its columns are the query's.")
        else:
            measures = self.measures(spec.get("measures"), (*path, "measures"))
            dimensions = self.dimensions(spec.get("dimensions"), (*path, "dimensions"))
            if not measures and not spec.get("measures"):
                self.fail(path, f"Metric {name!r} needs at least one measure.")
            for clash in set(measures) & set(dimensions):
                self.fail((*path, "dimensions", clash), f"{clash!r} is both a measure and a dimension.")
        parameters = self.parameters(spec.get("parameters"), (*path, "parameters"), sql is not None)
        for clash in set(parameters) & set(dimensions):
            self.fail((*path, "parameters", clash),
                      f"Parameter {clash!r} has the name of a dimension, which filters by that name already.")
        if sql is not None:
            self.check_sql(sql, parameters, (*path, "sql"), spec.get("parameters") or {})
        if len(self.problems) > before:
            return None
        return Metric(name, source, description, table, sql, measures, dimensions, parameters)

    def measures(self, spec: Any, path: tuple) -> dict[str, Measure]:
        spec = self.mapping(spec or {}, path, "measures")
        measures = {}
        for name, value in (spec or {}).items():
            if not self.name(name, (*path, name), "Measure"):
                continue
            item = {"expr": value} if isinstance(value, str) else self.mapping(
                value, (*path, name), f"Measure {name!r}", _MEASURE_KEYS)
            if item is None:
                continue
            expr = item.get("expr")
            try:
                check_expression(expr, aggregate=True)
            except ValueError as exc:
                self.fail((*path, name, "expr") if "expr" in item else (*path, name), f"Measure {name!r} {exc}.")
                continue
            measures[name] = Measure(
                name, expr, self.format(item.get("format"), (*path, name, "format")),
                self.text(item.get("description"), (*path, name, "description"), "description") or "",
            )
        return measures

    def dimensions(self, spec: Any, path: tuple) -> dict[str, Dimension]:
        spec = self.mapping(spec or {}, path, "dimensions")
        dimensions = {}
        for name, value in (spec or {}).items():
            if not self.name(name, (*path, name), "Dimension"):
                continue
            item = {"column": value} if isinstance(value, str) else self.mapping(
                value, (*path, name), f"Dimension {name!r}", _DIMENSION_KEYS)
            if item is None:
                continue
            column, expr = item.get("column"), item.get("expr")
            if (column is None) == (expr is None):
                self.fail((*path, name), f"Dimension {name!r} needs either a `column` or an `expr`.")
                continue
            if column is not None:
                if not isinstance(column, str) or not column:
                    self.fail((*path, name, "column"), f"Dimension {name!r}: column must be a column name.")
                    continue
                expr = exp.column(column, quoted=True).sql()
            else:
                try:
                    check_expression(expr, aggregate=False)
                except ValueError as exc:
                    self.fail((*path, name, "expr"), f"Dimension {name!r} {exc}.")
                    continue
            kind = item.get("type")
            if kind not in (None, "time"):
                self.fail((*path, name, "type"), f"Dimension {name!r}: type must be `time` or left out.")
                continue
            grains: tuple[str, ...] = ()
            if kind == "time":
                given = item.get("grains", list(GRAINS))
                if not isinstance(given, list) or not given or any(g not in GRAINS for g in given):
                    self.fail((*path, name, "grains"), f"Dimension {name!r}: grains must be a list of {', '.join(GRAINS)}.")
                    continue
                grains = tuple(given)
            elif "grains" in item:
                self.fail((*path, name, "grains"), f"Dimension {name!r} has grains but is not `type: time`.")
                continue
            dimensions[name] = Dimension(
                name, expr, column, kind == "time", grains, self.format(item.get("format"), (*path, name, "format")),
                self.text(item.get("description"), (*path, name, "description"), "description") or "",
            )
        return dimensions

    def parameters(self, spec: Any, path: tuple, sql: bool) -> dict[str, Parameter]:
        spec = self.mapping(spec or {}, path, "parameters")
        parameters = {}
        for name, value in (spec or {}).items():
            if not self.name(name, (*path, name), "Parameter"):
                continue
            item = self.mapping(value, (*path, name), f"Parameter {name!r}", _PARAMETER_KEYS)
            if item is not None and (parameter := self.parameter(name, item, (*path, name), sql)) is not None:
                parameters[name] = parameter
        return parameters

    def parameter(self, name: str, item: dict, path: tuple, sql: bool) -> Parameter | None:
        before = len(self.problems)
        given = item.get("type")
        match = _TYPE.fullmatch(given) if isinstance(given, str) else None
        kind, inner = (match.group(1), match.group(2)) if match and match.group(1) else (match and match.group(3), None)
        if not match or (inner is None and kind not in SCALAR_TYPES) or (inner is not None and inner not in SCALAR_TYPES):
            self.fail((*path, "type"), f"Parameter {name!r}: type must be one of {', '.join(SCALAR_TYPES)}, "
                      "or list[<type>] or range[<type>] of them.")
            return None
        if kind == "range" and inner not in RANGE_TYPES:
            self.fail((*path, "type"), f"Parameter {name!r}: a range is of {', '.join(RANGE_TYPES)}.")
            return None
        if kind == "range" and sql:
            self.fail((*path, "type"), f"Parameter {name!r}: a SQL metric takes no range; use two parameters.")
            return None
        scalar = inner or kind
        bounds = item.get("bounds")
        if bounds is not None:
            if scalar not in RANGE_TYPES or not isinstance(bounds, list) or len(bounds) != 2:
                self.fail((*path, "bounds"), f"Parameter {name!r}: bounds are [low, high] of a number, integer, date or datetime.")
            else:
                try:
                    bounds = tuple(None if b is None else coerce_scalar(scalar, b) for b in bounds)
                except ValueError as exc:
                    self.fail((*path, "bounds"), f"Parameter {name!r}: bounds {exc}.")
        options = item.get("options")
        if options is not None:
            if not isinstance(options, list) or not options:
                self.fail((*path, "options"), f"Parameter {name!r}: options must be a list of values.")
            else:
                try:
                    options = tuple(coerce_scalar(scalar, o) for o in options)
                except ValueError as exc:
                    self.fail((*path, "options"), f"Parameter {name!r}: an option is wrong: {exc}.")
        required = item.get("required", False)
        if not isinstance(required, bool):
            self.fail((*path, "required"), f"Parameter {name!r}: required must be true or false.")
        column = self.text(item.get("column"), (*path, "column"), "column")
        op = item.get("op")
        default_op = {"list": "in", "range": "between"}.get(kind, "=")
        if sql:
            for key in ("column", "op"):
                if key in item:
                    self.fail((*path, key), f"Parameter {name!r}: a SQL metric binds :{name} in its query and takes no {key}.")
        elif name in KNOBS:
            expected = {"limit": "integer", "order_by": "list[string]"}[name]
            if given != expected or column is not None or op is not None:
                self.fail(path, f"Parameter {name!r} sets the metric's {name}: it is of type {expected} "
                          "and has no column or op.")
        elif column is None:
            self.fail(path, f"Parameter {name!r} needs a `column` to filter by; only `limit`, `order_by` "
                      "and a SQL metric's parameters take none.")
        elif op is None:
            op = default_op
        elif op not in OPS:
            self.fail((*path, "op"), f"Parameter {name!r}: op must be one of {', '.join(sorted(OPS))}.")
        elif (op == "in") != (kind == "list") or (op == "between") != (kind == "range"):
            self.fail((*path, "op"), f"Parameter {name!r}: op {op!r} does not fit type {given!r}; "
                      f"use op {default_op!r} for it.")
        if sql and item.get("default") is None and required is not True:
            self.fail(path, f"Parameter {name!r} of a SQL metric needs a `default` or `required: true`.")
        if len(self.problems) > before:
            return None
        parameter = Parameter(name, kind, inner, None, required, bounds, options,
                              self.text(item.get("description"), (*path, "description"), "description") or "",
                              column, op if not (sql or name in KNOBS) else None)
        if item.get("default") is not None:
            try:
                parameter = dataclasses.replace(parameter, default=parameter.coerce(item["default"]))
            except ValueError as exc:
                self.fail((*path, "default"), f"Parameter {name!r}: the default is wrong: {exc}.")
                return None
        return parameter

    def check_sql(self, sql: str, parameters: dict[str, Parameter], path: tuple, declared: Any = ()) -> None:
        try:
            parsed = parse_expression(sql)
        except sqlglot.errors.SqlglotError as exc:
            self.fail(path, f"The query does not parse: {str(exc).splitlines()[0]}")
            return
        if not isinstance(parsed, exp.Query):
            self.fail(path, "The query must be one SELECT or WITH query.")
            return
        used = placeholders(sql)
        for name in used:
            if name not in parameters and name not in declared:
                self.fail(path, f"The query uses :{name}, which is not a parameter of the metric.")
        for name, parameter in parameters.items():
            if name not in used:
                self.fail((*path[:-1], "parameters", name), f"Parameter {name!r} is not used as :{name} in the query.")
            elif parameter.kind == "list":
                for placeholder in parsed.find_all(exp.Placeholder):
                    if placeholder.name == name and not isinstance(placeholder.parent, exp.In):
                        self.fail(path, f"List parameter :{name} may only be used in IN (:{name}).")
            for placeholder in parsed.find_all(exp.Placeholder):
                if (placeholder.name == name and isinstance(placeholder.parent, (exp.Limit, exp.Offset))
                        and parameter.kind != "integer"):
                    self.fail(path, f"Parameter :{name} in LIMIT or OFFSET must be an integer.")


def load_section(section: Section) -> MetricSet:
    """
    The loader of the ``metrics:`` section: its metrics, checked against
    the Lumen sources of the project's context when it has them (see
    ``check_metrics``), with a metric over a source the app does not serve
    left out and reported by ``Section.warn``. Raises ``ProjectError`` with
    every problem found.
    """
    from .nodes import SOURCES_KEY

    metrics = _Reader(section).read()
    if SOURCES_KEY not in section.context:
        return metrics
    return check_metrics(metrics, section.context)[0]


def read_metrics(project: Project) -> MetricSet:
    """The metrics of a loaded project file, or none if it has no ``metrics:`` section."""
    metrics = project.get(SECTION)
    return metrics if isinstance(metrics, MetricSet) else MetricSet()


# ---------------------------------------------------------------------------
# Compiling

def truncate(target: exp.Expression, grain: str, dialect: str | None) -> exp.Expression:
    """``target`` truncated to the start of its ``grain``, e.g. the first day of its year."""
    if dialect != "sqlite":
        return exp.TimestampTrunc(this=target, unit=exp.Var(this=grain.upper()))

    def date(*modifiers: exp.Expression) -> exp.Expression:
        return exp.Anonymous(this="DATE", expressions=[target.copy(), *modifiers])

    text = exp.Literal.string
    if grain == "hour":
        return exp.Anonymous(this="STRFTIME", expressions=[text("%Y-%m-%d %H:00:00"), target.copy()])
    if grain == "day":
        return date()
    if grain == "week":
        return date(text("weekday 0"), text("-6 days"))
    if grain == "quarter":
        month = exp.Cast(this=exp.Anonymous(this="STRFTIME", expressions=[text("%m"), target.copy()]),
                         to=exp.DataType.build("INTEGER"))
        back = exp.Mod(this=exp.Paren(this=exp.Sub(this=month, expression=exp.Literal.number(1))),
                       expression=exp.Literal.number(3))
        shift = exp.DPipe(this=exp.DPipe(this=text("-"), expression=exp.Paren(this=back)), expression=text(" months"))
        return date(text("start of month"), shift)
    return date(text(f"start of {grain}"))


class MetricPreFilter(SQLPreFilter):
    """
    The filters on the tables a SQL metric's query reads, applied inside the
    query: a filter on a column of one of them that the result does not
    have, e.g. a state when the query ranks projects, filters the rows the
    query reads rather than its result.
    """

    tables = param.Dict(default={}, doc="The columns of each table the query reads.")

    transform_type: ClassVar[str] = "sql_metric_prefilter"

    @staticmethod
    def top_level_tables(parsed: exp.Expression) -> list[exp.Table]:
        """The tables in the FROM and JOINs of the SELECTs whose rows are the result, see ``outermost_selects``."""
        found = []
        for select in outermost_selects(parsed):
            sources = [select.args.get("from_", select.args.get("from")), *(select.args.get("joins") or [])]
            found += [s.this for s in sources if s is not None and isinstance(s.this, exp.Table)]
        return found

    def apply(self, sql_in: str) -> str:
        conditions = {table: conds for table, conds in self.conditions if conds}
        if not conditions:
            return sql_in
        expression = self.parse_sql(sql_in)
        ctes = {cte.alias_or_name.lower() for cte in expression.find_all(exp.CTE)}
        for table in self.top_level_tables(expression):
            if table.name in conditions and table.name.lower() not in ctes and isinstance(table.this, exp.Identifier):
                table.replace(self._create_filtered_subquery(table.name, conditions[table.name], table.alias or None))
        return self.to_sql(expression)

    def input_rows(self, field: str, sql_in: str) -> str | None:
        """
        The rows, with their filters, of the first table the query reads
        that has ``field``, e.g. to read the values a filter on that table
        can match; None when no table has it.
        """
        table = next((name for name, columns in self.tables.items() if field in columns), None)
        if table is None:
            return None
        conditions = dict(self.conditions).get(table) or []
        return self.to_sql(exp.select("*").from_(self._create_filtered_subquery(table, conditions)))


@dataclass
class Query:
    """What a metric node asks of a metric: the picked fields and the parameter values."""

    measures: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    grain: str | None = None
    order_by: list[str] = field(default_factory=list)
    limit: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


def check_query_fields(metric: Metric, query: Query) -> None:
    """Raise ``MetricError`` unless the picked fields belong to ``metric``."""
    if metric.form == "sql":
        for key in ("measures", "dimensions"):
            if getattr(query, key):
                raise MetricError(f"Metric {metric.name!r} is a SQL query; it has no {key} to pick.", key)
        if query.grain:
            raise MetricError(f"Metric {metric.name!r} is a SQL query; it has no time dimension.", "grain")
        return
    for key, known in (("measures", metric.measures), ("dimensions", metric.dimensions)):
        for name in getattr(query, key):
            if name not in known:
                raise MetricError(
                    f"Metric {metric.name!r} has no {key[:-1]} {name!r}; its {key} are: {', '.join(known)}.", key,
                )
    if not (query.measures or query.dimensions):
        raise MetricError(f"Pick at least one measure or dimension of metric {metric.name!r}.", "measures")
    if query.grain:
        times = [d for d in query.dimensions if metric.dimensions[d].time]
        if not times:
            raise MetricError("A grain needs a time dimension among the dimensions.", "grain")
        for name in times:
            if query.grain not in metric.dimensions[name].grains:
                raise MetricError(
                    f"Dimension {name!r} has no grain {query.grain!r}; its grains are: "
                    f"{', '.join(metric.dimensions[name].grains)}.", "grain",
                )
    outputs = [*query.dimensions, *query.measures]
    for key in query.order_by:
        name = key[1:] if isinstance(key, str) and key.startswith("-") else key
        if name not in outputs:
            raise MetricError(f"Cannot order by {key!r}; order by one of {', '.join(outputs)}, "
                              "with - first for descending.", "order_by")


def resolve_knobs(metric: Metric, query: Query) -> Query:
    """
    ``query`` with the ``limit`` and ``order_by`` of its parameters, which
    win over the node's config, which wins over the metric's defaults;
    each is checked like a parameter. Raises ``MetricError``.
    """
    knobs = metric.knobs()
    if not knobs:
        return query
    given = query.parameters
    resolved = {}
    for name, configured in (("limit", query.limit), ("order_by", query.order_by)):
        value = given.get(name)
        value = configured if value is None or value == [] else value
        value = knobs[name].default if value is None or value == [] else value
        try:
            resolved[name] = knobs[name].coerce(value)
        except ValueError as exc:
            raise MetricError(f"Parameter {name!r} of metric {metric.name!r}: {exc}.", "parameters") from None
    return dataclasses.replace(
        query, limit=resolved["limit"], order_by=resolved["order_by"] or [],
        parameters={k: v for k, v in given.items() if k not in knobs},
    )


def prepare(metric: Metric, query: Query) -> tuple[Query, dict[str, Any]]:
    """
    ``query`` with its ``limit`` and ``order_by`` resolved, checked against
    ``metric``, and the coerced values of its parameters. Raises ``MetricError``.
    """
    query = resolve_knobs(metric, query)
    check_query_fields(metric, query)
    return query, metric.values(query.parameters)


def aggregate(metric: Metric, query: Query, values: Mapping[str, Any], dialect: str | None) -> MetricAggregate:
    """The ``MetricAggregate`` of a structured metric for ``query`` with coerced parameter ``values``."""
    read = None if dialect == "any" else dialect

    def sql(expression: exp.Expression) -> str:
        return expression.sql(dialect=read)

    dimensions, temporal = [], []
    for name in query.dimensions:
        dimension = metric.dimensions[name]
        target = parse_expression(dimension.expr, read)
        if dimension.time and query.grain:
            target = truncate(target, query.grain, read)
            temporal.append(name)
        dimensions.append([name, sql(target)])
    measures = [[name, sql(parse_expression(metric.measures[name].expr, read))] for name in query.measures]
    where = []
    for name, value in values.items():
        if name in metric.knobs():
            continue
        if name in metric.parameters:
            parameter = metric.parameters[name]
            cond = condition(_column(parameter.column), parameter, value, read)
        else:
            cond = _dimension_condition(metric.dimensions[name].column, value, read)
        if cond is not None:
            where.append(sql(cond))
    return MetricAggregate(
        dimensions=dimensions, measures=measures, where=where,
        order_by=[[key.lstrip("-"), key.startswith("-")] for key in query.order_by],
        limit=query.limit, temporal=temporal, read=read, write=read,
    )


def column_kind(spec: Mapping[str, Any] | None) -> str | None:
    """The parameter type of a column from its schema entry, e.g. ``integer`` or ``date``."""
    spec = spec or {}
    if spec.get("format") in ("date", "datetime", "date-time"):
        return "date" if spec["format"] == "date" else "datetime"
    return {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean"}.get(spec.get("type"))


def coerce_dimension_values(metric: Metric, values: Mapping[str, Any], columns: Mapping[str, Any]) -> dict[str, Any]:
    """
    ``values`` with each dimension filter's value coerced to the type of its
    column in ``columns``, the table's schema. Raises ``MetricError``.
    """
    coerced = dict(values)
    for name, value in values.items():
        if name in metric.parameters or value is None:
            continue
        kind = column_kind(columns.get(metric.dimensions[name].column))
        if kind is None:
            continue
        try:
            coerced[name] = [coerce_scalar(kind, v) for v in value] if isinstance(value, list) else coerce_scalar(kind, value)
        except ValueError as exc:
            raise MetricError(f"Filter {name!r} of metric {metric.name!r}: {exc}.", "parameters") from None
    return coerced


def _dimension_condition(column: str, value: Any, dialect: str | None) -> exp.Expression | None:
    def item(v: Any) -> exp.Expression:
        if isinstance(v, bool):
            return literal("boolean", v, dialect)
        if isinstance(v, (int, float)):
            return literal("number", v, dialect)
        if isinstance(v, dt.datetime):
            return literal("datetime", v, dialect)
        if isinstance(v, dt.date):
            return literal("date", v, dialect)
        return literal("string", v, dialect)

    target = _column(column)
    if value is None:
        return None
    if isinstance(value, list):
        return target.isin(*map(item, value)) if value else None
    return target.eq(item(value))


def outermost_selects(parsed: exp.Expression) -> list[exp.Select]:
    """
    The SELECTs whose rows are the result of ``parsed``: the query itself,
    each branch of a UNION, and the left side of an EXCEPT or INTERSECT,
    where a filter narrows the result as it would on its own. A subquery,
    a CTE and the right side of EXCEPT are never among them.
    """
    if isinstance(parsed, exp.Select):
        return [parsed]
    if isinstance(parsed, exp.Union):
        return [*outermost_selects(parsed.this), *outermost_selects(parsed.expression)]
    if isinstance(parsed, (exp.Except, exp.Intersect)):
        return outermost_selects(parsed.this)
    if isinstance(parsed, exp.Subquery):
        return outermost_selects(parsed.this)
    return []


def _empty_list(node: exp.In, values: Mapping[str, Any], metric: Metric) -> str | None:
    """
    The list parameter of ``node`` given an empty list, when every item of
    the IN is such a list and so nothing is left to compare with. With
    other items the empty lists only drop out of the IN.
    """
    names = []
    for item in node.expressions:
        parameter = metric.parameters.get(item.name) if isinstance(item, exp.Placeholder) else None
        if parameter is None or parameter.kind != "list" or values.get(item.name):
            return None
        names.append(item.name)
    return names[0] if names else None


def _drop_condition(node: exp.In, name: str, metric: Metric, selects: list[exp.Select],
                    dialect: str | None) -> None:
    """
    Replace ``x IN (:name)``, or ``x NOT IN (:name)``, whose list is empty
    with TRUE, so it filters nothing, like an empty filter value. That only
    holds for a condition ANDed into the WHERE or HAVING of one of the
    ``selects`` whose rows are the result; anywhere else, e.g. under OR, in
    a subquery, a CTE, NOT EXISTS, a join or an aggregate's FILTER, TRUE
    could filter everything or change a measure, so the empty list is refused.
    """
    target = node
    while isinstance(target.parent, exp.Paren):
        target = target.parent
    if isinstance(target.parent, exp.Not):
        target = target.parent
    while isinstance(target.parent, exp.Paren):
        target = target.parent
    clause = target.parent
    while isinstance(clause, (exp.And, exp.Paren)):
        clause = clause.parent
    if not (isinstance(clause, (exp.Where, exp.Having)) and any(clause.parent is select for select in selects)):
        raise MetricError(
            f"Parameter {name!r} of metric {metric.name!r} is an empty list, which only drops a condition ANDed "
            "into the WHERE or HAVING of the query's outermost SELECT; give it at least one value.", "parameters",
        )
    target.replace(exp.true() if dialect not in NO_BOOLEAN_LITERALS else exp.EQ(
        this=exp.Literal.number(1), expression=exp.Literal.number(1)))


def bind_sql(metric: Metric, values: Mapping[str, Any], dialect: str | None) -> str:
    """
    The query of a SQL metric with each placeholder replaced by a typed
    literal of its coerced value. A list parameter given an empty list drops
    the condition it is in, see ``_drop_condition``.
    """
    read = None if dialect == "any" else dialect
    parsed = parse_expression(metric.sql, read)
    selects = outermost_selects(parsed)
    for node in list(parsed.find_all(exp.In)):
        if (name := _empty_list(node, values, metric)) is not None:
            _drop_condition(node, name, metric, selects, read)

    def replace(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.In):
            items = []
            for item in node.expressions:
                parameter = metric.parameters.get(item.name) if isinstance(item, exp.Placeholder) else None
                if parameter is not None and parameter.kind == "list":
                    items.extend(literal(parameter.item, v, read) for v in values.get(item.name) or [])
                else:
                    items.append(item)
            node.set("expressions", items)
            return node
        if isinstance(node, exp.Placeholder) and node.name in metric.parameters:
            value = values.get(node.name)
            if value is None:
                return exp.Null()
            if isinstance(node.parent, (exp.Limit, exp.Offset)) and value < 0:
                raise MetricError(f"Parameter {node.name!r} of metric {metric.name!r} is a row count; "
                                  f"it cannot be {value}.", "parameters")
            return literal(metric.parameters[node.name].kind, value, read)
        return node

    return parsed.transform(replace).sql(dialect=read)


def _sample(parameter: Parameter) -> Any:
    """A value of ``parameter``'s type within its bounds and options, to check a query without a default."""
    if parameter.options:
        return [parameter.options[0]] if parameter.kind == "list" else parameter.options[0]
    scalar = parameter.item or parameter.kind
    value: Any = {"string": "", "boolean": False, "integer": 1, "number": 1, "date": dt.date(2000, 1, 1),
                  "datetime": dt.datetime(2000, 1, 1)}[scalar]
    if parameter.bounds is not None and parameter.bounds[0] is not None:
        value = parameter.bounds[0]
    return [value] if parameter.kind == "list" else (value, value) if parameter.kind == "range" else value


def check_metrics(metrics: MetricSet, context: Mapping[str, Any]) -> tuple[MetricSet, list[Problem]]:
    """
    Check each metric against the host sources in ``context``, with every
    measure and dimension picked and the defaults bound: its source must
    exist and run SQL, its table and columns exist, and the compiled query
    pass ``check_query`` and the engine's planning. A metric that fails is
    left out, so the others keep working, and reported as a warning; the
    returned set keeps why in its ``failures``.
    """
    from panel_flowdash.value_text import redact_secrets

    from .nodes import host_sources, lookup_source

    warnings: list[Problem] = []
    kept: dict[str, Metric] = {}
    failures: dict[str, MetricFailure] = {}
    # Section.warn logs the warning and keeps it in the project's warnings.
    report = getattr(metrics.section, "warn", None)

    def fail(name: str, path: tuple, message: str, warning: str | None = None) -> None:
        # An engine's error may quote its connection URL, which is shown and logged.
        message = redact_secrets(message)
        failures[name] = MetricFailure(metrics[name], message, path)
        text = warning or f"Metric {name!r} is left out: {message}"
        warnings.append(report((name, *path), text) if report is not None else Problem(text))

    for name, metric in metrics.items():
        source = lookup_source(metric.source, context)
        if source is None:
            refs = ", ".join(sorted(host_sources(context))) or "none"
            fail(name, ("source",), f"The app serves no source {metric.source!r}; its sources are: {refs}.",
                 f"Metric {name!r} is left out: the app serves no source {metric.source!r}; its sources are: {refs}.")
            continue
        try:
            check_metric(metric, source)
        except MetricCheckError as exc:
            fail(name, exc.path, str(exc))
            continue
        except Exception as exc:
            fail(name, (), f"Metric {name!r}: {exc}")
            continue
        kept[name] = metric
    return MetricSet(kept, metrics.section, failures), warnings


class MetricCheckError(ValueError):
    def __init__(self, message: str, path: tuple = ()):
        super().__init__(message)
        self.path = path


def _columns(source: Source, table: str) -> dict[str, Any]:
    schema = source.get_schema(table)
    return {column: spec for column, spec in schema.items() if column != "__len__"}


def check_metric(metric: Metric, source: Source) -> None:
    """Raise ``MetricCheckError``, with the path of the key at fault, unless ``metric`` works on ``source``."""
    from .checks import check_query

    if not isinstance(source, BaseSQLSource):
        raise MetricCheckError(f"Metric {metric.name!r}: {type(source).__name__} does not run SQL.", ("source",))
    dialect = None if source.dialect == "any" else source.dialect
    values = {name: p.default if p.default is not None else _sample(p) for name, p in metric.parameters.items()}
    if metric.form == "sql":
        try:
            sql = bind_sql(metric, values, dialect)
            check_query(sql, source)
        except (ValueError, sqlglot.errors.SqlglotError) as exc:
            raise MetricCheckError(f"Metric {metric.name!r}: {exc}", ("sql",)) from None
        _probe(source, sql, metric, ("sql",))
    else:
        tables = source.get_tables()
        if metric.table not in tables:
            close = difflib.get_close_matches(metric.table, tables, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else f" Its tables are: {', '.join(sorted(tables))}."
            raise MetricCheckError(f"The source {metric.source!r} has no table {metric.table!r}.{hint}", ("table",))
        columns = _columns(source, metric.table)
        lowered = {c.lower(): c for c in columns}
        fields: list[tuple[tuple, str, str]] = [
            *((("measures", n, "expr"), n, m.expr) for n, m in metric.measures.items()),
            *((("dimensions", n), n, d.expr) for n, d in metric.dimensions.items()),
        ]
        for path, name, expr in fields:
            try:
                check_expression(expr, aggregate=path[0] == "measures", dialect=dialect)
            except ValueError as exc:
                raise MetricCheckError(f"{path[0][:-1].capitalize()} {name!r} {exc}.", path) from None
            for column in expression_columns(expr, dialect):
                if column not in columns and column.lower() not in lowered:
                    close = difflib.get_close_matches(column, list(columns), n=1)
                    hint = f" Did you mean {close[0]!r}?" if close else ""
                    raise MetricCheckError(f"Unknown column {column!r} in {path[0][:-1]} {name!r}.{hint}", path)
        for name, parameter in metric.parameters.items():
            if parameter.column is None:
                continue
            if parameter.column not in columns and parameter.column.lower() not in lowered:
                close = difflib.get_close_matches(parameter.column, list(columns), n=1)
                hint = f" Did you mean {close[0]!r}?" if close else ""
                raise MetricCheckError(f"Unknown column {parameter.column!r} in parameter {name!r}.{hint}",
                                       ("parameters", name, "column"))
        times = [d for d in metric.dimensions.values() if d.time]
        query = Query(list(metric.measures), [d for d in metric.dimensions if not metric.dimensions[d].time])
        transforms = [aggregate(metric, query, values, dialect)]
        transforms += [aggregate(metric, Query(list(metric.measures), [d.name], grain), {}, dialect)
                       for d in times for grain in d.grains]
        base = table_sql(metric.table, dialect)
        for transform in transforms:
            try:
                sql = transform.apply(base)
                check_query(sql, source)
            except (ValueError, sqlglot.errors.SqlglotError) as exc:
                raise MetricCheckError(f"Metric {metric.name!r}: {exc}") from None
            _probe(source, sql, metric)


def table_sql(table: str, dialect: str | None) -> str:
    """``SELECT * FROM <table>``, the query a metric's compiled SQL is checked over, as a SQL node's would be."""
    return exp.select("*").from_(exp.Table(this=exp.Identifier(this=table, quoted=True))).sql(dialect=dialect)


def _probe(source: BaseSQLSource, sql: str, metric: Metric, path: tuple = ()) -> None:
    from ..transforms.sql import SQLLimit
    from .nodes import NODE_TIMEOUT, execute

    try:
        execute(source, SQLLimit(limit=0, write=source.dialect).apply(sql), NODE_TIMEOUT)
    except Exception as exc:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise MetricCheckError(f"Metric {metric.name!r}: the engine refused its query: {message}", path) from None


class ProjectMetrics(MetricSet):
    """
    The metrics of ``project`` as it is now, for an app's context: a metric
    an admin saves with ``Project.set`` is used from then on, by every session.

    Each change of the project's data is read once. A ``metrics:`` section
    that fails to load is logged once, with credentials masked, and the
    metrics read before it are kept, with its ``problems``; a failure is
    tried again at most every ``retry`` seconds, e.g. for a source that was
    down. A metric that does not work on its source is left out on its own
    (see ``check_metrics``) and checked again as often. Read ``snapshot()``
    to use several metrics together: the mapping methods each read the
    current snapshot, which an admin's save may replace in between.
    """

    def __init__(self, project: Project, retry: float = 60.0):
        self.project = project
        self.retry = retry
        self._lock = threading.Lock()
        self._data: Any = _UNREAD
        self._raw: Any = _UNREAD
        self._snapshot = MetricSet()
        self._failed_at: float | None = None
        self._checked_at = time.monotonic()
        self.problems: list[Problem] = []
        """Why the section as the file has it now is not used, if it is not."""

    def _due(self) -> bool:
        now = time.monotonic()
        if self._failed_at is not None:
            return now - self._failed_at >= self.retry
        return bool(self._snapshot.failures) and now - self._checked_at >= self.retry

    def snapshot(self) -> MetricSet:
        data = self.project.data
        if data is self._data and not self._due():
            return self._snapshot
        # One thread reads at a time; while it does, e.g. waiting on a slow source, the others get the
        # metrics read before. Only the very first read waits, as there is nothing to give before it.
        if not self._lock.acquire(blocking=self._data is _UNREAD):
            return self._snapshot
        try:
            data = self.project.data
            if data is not self._data:
                raw = data.get(SECTION)
                changed = self._raw is _UNREAD or raw != self._raw
                self._data, self._raw = data, copy.deepcopy(raw)
                if changed:
                    return self._read(log=True)
            if self._failed_at is not None and time.monotonic() - self._failed_at >= self.retry:
                return self._read(log=False)
            if self._due():
                return self._recheck()
            return self._snapshot
        finally:
            self._lock.release()

    def _read(self, log: bool) -> MetricSet:
        try:
            metrics = read_metrics(self.project)
        except ProjectError as exc:
            self._failed_at = time.monotonic()
            self.problems = list(exc.problems)
            if log:
                from panel_flowdash.value_text import error_text

                kept = len(self._snapshot)
                logger.warning(
                    "Not using the project file's metrics: %s%s", error_text(exc),
                    f"\nKeeping the {kept} metric{'' if kept == 1 else 's'} read before." if kept else "",
                )
            return self._snapshot
        self._failed_at = None
        self._checked_at = time.monotonic()
        self.problems = []
        self._snapshot = metrics
        return metrics

    def _recheck(self) -> MetricSet:
        """The snapshot with the metrics left out checked again, e.g. once their source is back."""
        current = self._snapshot
        retried, _ = check_metrics(
            MetricSet({name: failure.metric for name, failure in current.failures.items()}), self.project.context,
        )
        self._checked_at = time.monotonic()
        if not retried:
            return current
        for name in retried:
            logger.info("Metric %r works again and is used from now on.", name)
        metrics = {**current._metrics, **retried._metrics}
        order = [name for name in (self._raw if isinstance(self._raw, dict) else {}) if name in metrics]
        order += [name for name in metrics if name not in order]
        self._snapshot = MetricSet({name: metrics[name] for name in order}, current.section, retried.failures)
        return self._snapshot

    @property
    def failures(self) -> dict[str, MetricFailure]:  # type: ignore[override]
        return self.snapshot().failures

    @property
    def _metrics(self) -> dict[str, Metric]:  # type: ignore[override]
        return self.snapshot()._metrics

    @property
    def section(self) -> Section | None:  # type: ignore[override]
        return self.snapshot().section

    def __iter__(self) -> Iterator[str]:
        return iter(list(self.snapshot()._metrics))

    def keys(self):
        return self.snapshot().keys()

    def items(self):
        return self.snapshot().items()

    def values(self):
        return self.snapshot().values()

    def __repr__(self) -> str:
        return f"ProjectMetrics({list(self.snapshot()._metrics)})"


def _settings_view(project: Project, editable: bool) -> Any:
    from .metrics_admin import metrics_settings

    return metrics_settings(project, editable)


def _wizard_view(project: Project) -> Any:
    from .metrics_admin import metrics_wizard_step

    return metrics_wizard_step(project)


# The Bokeh documents of the sessions of apps that serve Lumen sources, see ``serve_lumen``.
_LUMEN_DOCS: weakref.WeakSet = weakref.WeakSet()
# Those whose app shows the Data section, see ``serve_data``.
_DATA_DOCS: weakref.WeakSet = weakref.WeakSet()


def serve_lumen(doc: Any) -> None:
    """Mark ``doc``, a session's document, as one of an app that serves Lumen sources."""
    if doc is not None:
        _LUMEN_DOCS.add(doc)


def serves_lumen() -> bool:
    """Whether the current session is one of an app that serves Lumen sources."""
    import panel as pn

    doc = pn.state.curdoc
    return doc is not None and doc in _LUMEN_DOCS


def serve_data(doc: Any) -> None:
    """Mark ``doc``, a session's document, as one of an app whose sidebar has the Data section."""
    if doc is not None:
        _DATA_DOCS.add(doc)


def shows_data() -> bool:
    """Whether the current session's app has the Data section, which shows the catalog and the metrics."""
    import panel as pn

    doc = pn.state.curdoc
    return doc is not None and doc in _DATA_DOCS


def metrics_page() -> str:
    """Where the current session's people find the metrics, to name in a message."""
    return "Data, Metrics" if shows_data() else "Admin, Metrics"


class _LumenSection(ProjectSection):
    """
    A Lumen section of the project file, whose admin view and setup step
    exist only in the sessions of an app serving Lumen sources: registering
    a section is process-wide, and another FlowDash app in the process keeps
    its own Data step and has no Metrics section.
    """

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if name in ("settings_view", "wizard_view") and value is not None and not serves_lumen():
            return None
        return value


class _DataSection(_LumenSection):
    """
    A Lumen section whose page is in the Data section where the app has
    one (``shows_data``), so the admin view leaves it out; on a
    panel-flowdash without sidebar sections it stays in the admin view.
    """

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name == "settings_view" and value is not None and shows_data():
            return None
        return value


# The admin view and the setup wizard, on a panel-flowdash whose sections have them.
_ADMIN = {
    key: value for key, value in {
        "settings_view": _settings_view, "wizard_view": _wizard_view, "group": "Data",
    }.items() if key in getattr(ProjectSection, "__dataclass_fields__", {})
}

register_section((_DataSection if _ADMIN else ProjectSection)(
    SECTION, load_section, title="Metrics", icon="functions",
    description="Numbers defined here are the agreed definitions; the assistant uses them instead of "
    "writing its own SQL.",
    **_ADMIN,
))
