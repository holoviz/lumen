"""A stand-in dbt Semantic Layer client returning ``dbtsl`` model objects (or field-for-field mirrors)."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import enum

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class _Measure:
    name: str
    agg_time_dimension: str | None
    agg: Any
    expr: str


@dataclass
class _Dimension:
    name: str
    qualified_name: str
    description: str | None
    type: Any
    label: str | None
    is_partition: bool
    expr: str | None
    queryable_granularities: list = field(default_factory=list)


@dataclass
class _Metric:
    name: str
    description: str | None
    type: Any
    label: str
    requires_metric_time: bool
    queryable_granularities: list = field(default_factory=list)
    dimensions: list = field(default_factory=list)
    measures: list = field(default_factory=list)
    entities: list = field(default_factory=list)


# Used when dbtsl is not installed: the fields every dbtsl version has, and nothing more.
MIRRORS = {"Measure": _Measure, "Dimension": _Dimension, "Metric": _Metric}

try:
    from dbtsl import models
except ImportError:
    models = SimpleNamespace(
        **MIRRORS,
        AggregationType=enum.Enum("AggregationType", {k: k for k in (
            "SUM", "MIN", "MAX", "COUNT_DISTINCT", "SUM_BOOLEAN", "AVERAGE", "PERCENTILE", "MEDIAN", "COUNT")}),
        DimensionType=enum.Enum("DimensionType", {k: k for k in ("CATEGORICAL", "TIME")}),
        MetricType=enum.Enum("MetricType", {k: k for k in ("SIMPLE", "RATIO", "CUMULATIVE", "DERIVED", "CONVERSION")}),
        TimeGranularity=enum.Enum("TimeGranularity", {k: k for k in (
            "NANOSECOND", "MICROSECOND", "MILLISECOND", "SECOND", "MINUTE", "HOUR", "DAY", "WEEK", "MONTH", "QUARTER", "YEAR")}),
    )

AggregationType = models.AggregationType
DimensionType = models.DimensionType
MetricType = models.MetricType
TimeGranularity = models.TimeGranularity

_EMPTY: dict[str, Any] = {"description": None, "label": "", "expr": None, "is_partition": False,
                          "requires_metric_time": False, "agg_time_dimension": None}


def _build(cls: type, **given: Any) -> Any:
    """``cls`` with ``given``, every other field empty; fields the installed ``dbtsl`` lacks are refused."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(given) - set(fields)
    assert not unknown, f"dbtsl {cls.__name__} has no {sorted(unknown)}"
    for name, f in fields.items():
        if name in given:
            continue
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue
        given[name] = _EMPTY.get(name, [] if "List" in str(f.type) or "list" in str(f.type) else None)
    return cls(**given)


def measure(name: str, agg: str, expr: str, agg_time_dimension: str | None = None) -> Any:
    return _build(models.Measure, name=name, agg=AggregationType(agg), expr=expr, agg_time_dimension=agg_time_dimension)


def dimension(qualified_name: str, kind: str = "CATEGORICAL", expr: str | None = None, grains=(), description: str | None = None) -> Any:
    return _build(
        models.Dimension, name=qualified_name.rsplit("__", 1)[-1], qualified_name=qualified_name,
        type=DimensionType(kind), expr=expr, description=description,
        queryable_granularities=[TimeGranularity(g) for g in grains],
    )


def metric(name: str, kind: str = "SIMPLE", measures=(), dimensions=(), description: str | None = None, **extra: Any) -> Any:
    return _build(
        models.Metric, name=name, type=MetricType(kind), measures=list(measures), dimensions=list(dimensions),
        description=description, **extra,
    )


def semantic_sql(table: str, select: str, own: dict[str, str] | None = None, where: str | None = None):
    """
    What dbt compiles for a metric over ``table``: ``own`` maps the group-bys
    on the metric's own semantic model to their column; any other group-by
    joins another semantic model, and ``where`` is a metric filter.
    """
    own = {"metric_time": "DATE_TRUNC('day', order_date)", **(own or {})}

    def build(group_by: list[str]) -> str:
        cols, joins = [], []
        for g in group_by:
            if g in own:
                cols.append(f"{own[g]} AS {g}")
            else:
                other = g.split("__", 1)[0]
                joins.append(f"LEFT OUTER JOIN analytics.{other}s {other}_src ON src.{other}_id = {other}_src.{other}_id")
                cols.append(f"{other}_src.{g.rsplit('__', 1)[-1]} AS {g}")
        inner = ", ".join([*cols, select])
        sql = f"SELECT {inner} FROM (SELECT * FROM {table} src {' '.join(joins)}"
        sql += f" WHERE {where}" if where else ""
        sql += ") subq"
        return sql + (f" GROUP BY {', '.join(group_by)}" if group_by else "")

    return build


class FakeSemanticLayer:
    """Answers like ``AsyncSemanticLayerClient``: ``session()``, ``metrics()`` and ``compile_sql()``."""

    def __init__(self, metrics: list[Any], sql: dict[str, Any] | None = None, fail: set[str] = frozenset(), delay: float = 0):
        self._metrics = metrics
        self._sql = sql or {}
        self._fail = set(fail)
        self._delay = delay
        self.compile_calls: list[dict[str, Any]] = []
        self.metric_calls = 0

    @contextlib.asynccontextmanager
    async def session(self):
        yield self

    async def metrics(self) -> list[Any]:
        self.metric_calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return list(self._metrics)

    async def compile_sql(self, metrics: list[str], group_by: list[str] | None = None, **_kwargs: Any) -> str:
        self.compile_calls.append({"metrics": metrics, "group_by": group_by})
        name = metrics[0]
        if name in self._fail:
            raise RuntimeError(f"compile_sql failed for {name}")
        sql = self._sql[name]
        return sql(list(group_by or [])) if callable(sql) else sql
