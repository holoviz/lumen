"""
Whether a numeric column is a property of an entity its rows belong to, e.g.
a project's capacity on each of its turbines' rows, so that a sum over the
rows counts each entity's value once per row.

The grain is decided once per source table and column, on the table's rows
rather than on a filtered slice of them, so every slice of the table gets the
same answer, and kept for ``GRAIN_TTL`` seconds.
"""
from __future__ import annotations

import dataclasses
import time

from typing import Any

import pandas as pd
import sqlglot

from sqlglot import exp

from ..flowdash.nodes import execute
from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource
from ..sources.duckdb import DuckDBSource

# A column is a key's property when, of the rows whose key value has several rows, at least
# GRAIN_SINGLE hold one value of the column per key value; the key has at least GRAIN_GROUPS
# values, each on GRAIN_REPEAT rows on average.
GRAIN_SINGLE = 0.9
GRAIN_GROUPS = 10
GRAIN_REPEAT = 1.5
# Single key columns probed, and the low-cardinality columns paired into composite keys. A pair
# only counts when neither of its columns alone holds the value for more than GRAIN_PAIR_ALONE of
# the rows: a value nearly fixed by one column, e.g. a turbine's capacity by its model, is a
# per-row value that a second column, e.g. the state, only appears to complete.
GRAIN_KEYS = 8
GRAIN_PAIRS_OF = 5
GRAIN_PAIR_ALONE = 0.5
# The most distinct values a column paired into a composite key has.
GRAIN_PAIR_CARDINALITY = 1000
# Rows probed on a source other than DuckDB, whose queries a timeout may not stop.
GRAIN_SAMPLE = 200_000
# Seconds per probe query, and seconds a decision is kept.
GRAIN_TIMEOUT = 5
GRAIN_TTL = 600

_KEY_KINDS = ("object", "string", "bool", "int", "uint", "datetime")


@dataclasses.dataclass(frozen=True)
class Grain:
    """How ``column`` of ``table`` depends on ``key``: the share of rows of repeated key values holding one value."""

    table: str
    column: str
    key: tuple[str, ...]
    groups: int
    rows: int
    share: float

    @property
    def is_property(self) -> bool:
        return (self.groups >= GRAIN_GROUPS and self.rows >= GRAIN_REPEAT * self.groups
                and self.share >= GRAIN_SINGLE)

    @property
    def key_text(self) -> str:
        return " and ".join(self.key)

    def describe(self) -> str:
        return (f"{self.share:.0%} of the rows of {self.table} hold one {self.column} per {self.key_text}, which has "
                f"{self.groups:,} values on about {self.rows / max(self.groups, 1):,.0f} rows each")


class Unchecked(Exception):
    """The grain could not be checked, e.g. a probe failed or timed out."""


# Decisions and column types by connection, table query and column, with the time they were made.
_CACHE: dict[tuple, tuple[float, Any]] = {}


def _cached(source: BaseSQLSource, key: tuple, make):
    """``make()``, kept per connection of ``source`` (SQL nodes' result sources share their source's) and ``key``."""
    entry = (id(getattr(source, "_connection", source)), *key)
    now = time.monotonic()
    if (hit := _CACHE.get(entry)) is not None and now - hit[0] < GRAIN_TTL:
        return hit[1]
    value = make()
    for old in [k for k, (t, _) in _CACHE.items() if now - t >= GRAIN_TTL]:
        del _CACHE[old]
    _CACHE[entry] = (now, value)
    return value


def _quote(name: str, dialect: str | None) -> str:
    return exp.to_identifier(name, quoted=True).sql(dialect=dialect)


def _dialect(source: BaseSQLSource) -> str | None:
    return None if source.dialect == "any" else source.dialect


def _run(source: BaseSQLSource, sql: str) -> pd.DataFrame:
    try:
        return execute(source, sql, GRAIN_TIMEOUT)
    except Exception as exc:
        raise Unchecked(str(exc)) from exc


def _derives(parsed: exp.Expression) -> bool:
    """Whether a query aggregates or deduplicates its rows, so its columns are not its tables' columns."""
    return any(True for _ in parsed.find_all(exp.Group, exp.Distinct, exp.AggFunc, exp.Window))


def base_table(pipeline: Pipeline, column: str) -> tuple[BaseSQLSource, str] | None:
    """
    The source table whose rows ``column`` of ``pipeline`` comes from
    unchanged, following the queries of SQL nodes down to it; None when a
    query on the way aggregates or deduplicates the rows, computes the
    column, or the pipeline is not read from a SQL source.
    """
    source = pipeline.source
    if pipeline.pipeline is not None or not isinstance(source, BaseSQLSource):
        return None
    if any("Metric" in type(t).__name__ or "Aggregate" in type(t).__name__ for t in pipeline.sql_transforms):
        return None
    tables = set(source.get_tables())
    table = pipeline.table
    for _ in range(8):
        sql = source.get_sql_expr(table)
        try:
            parsed = sqlglot.parse_one(sql, read=_dialect(source))
        except sqlglot.errors.SqlglotError:
            return None
        ctes = {cte.alias_or_name for cte in parsed.find_all(exp.CTE)}
        read = [t.name for t in parsed.find_all(exp.Table) if t.name in tables and t.name not in ctes and t.name != table]
        if not read:
            return source, table
        if _derives(parsed):
            return None
        holding = [t for t in dict.fromkeys(read) if column in _columns(source, t)]
        if not holding:
            return None
        table = holding[0]
    return None


def _columns(source: BaseSQLSource, table: str) -> pd.Series:
    """The dtypes of ``table``'s columns, by name, from its query without rows."""
    sql = source.get_sql_expr(table)
    return _cached(source, ("columns", sql), lambda: _run(source, f"SELECT * FROM ({sql}) AS t LIMIT 0").dtypes)


def _rows_sql(source: BaseSQLSource, table: str) -> str:
    sql = source.get_sql_expr(table)
    # DuckDB stops a query at the timeout; other engines may not, so they are probed on a sample.
    return sql if isinstance(source, DuckDBSource) else f"SELECT * FROM ({sql}) AS s LIMIT {GRAIN_SAMPLE}"


def dependency(source: BaseSQLSource, table: str, column: str, key: tuple[str, ...]) -> Grain:
    """How ``column`` of ``table`` depends on ``key``, measured on the table's rows."""
    dialect = _dialect(source)
    value = _quote(column, dialect)
    keys = ", ".join(_quote(k, dialect) for k in key)
    frame = _run(source, (
        "SELECT COUNT(*), SUM(n), SUM(CASE WHEN n > 1 THEN n ELSE 0 END), "
        "SUM(CASE WHEN n > 1 AND d = 1 THEN n ELSE 0 END) FROM ("
        f"SELECT {keys}, COUNT(*) AS n, COUNT(DISTINCT {value}) AS d FROM ({_rows_sql(source, table)}) AS r "
        f"WHERE {value} IS NOT NULL GROUP BY {keys}) AS g"
    ))
    groups, rows, multi, single = (0 if pd.isna(v) else int(v) for v in frame.iloc[0])
    return Grain(table, column, key, groups, rows, single / multi if multi else 0.0)


def _candidates(source: BaseSQLSource, table: str, column: str) -> tuple[list[str], list[str]]:
    """The single key columns to probe, entity-like names first, and the low-cardinality ones to pair."""
    dtypes = _columns(source, table)
    names = [str(c) for c, d in dtypes.items() if str(c) != column and str(d).lower().startswith(_KEY_KINDS)][:30]
    if not names:
        return [], []
    dialect = _dialect(source)
    counts = _run(source, f"SELECT COUNT(*), COUNT(DISTINCT {_quote(column, dialect)}), " + ", ".join(
        f"COUNT(DISTINCT {_quote(n, dialect)})" for n in names) + f" FROM ({_rows_sql(source, table)}) AS r").iloc[0]
    rows, values = int(counts.iloc[0]), int(counts.iloc[1])
    # A constant is the same everywhere, not a property of one entity.
    if values < 2:
        return [], []
    distinct = dict(zip(names, (int(v) for v in counts.iloc[2:]), strict=True))
    single = [n for n in names if GRAIN_GROUPS <= distinct[n] and rows >= GRAIN_REPEAT * distinct[n]]
    entity = ("id", "name", "key", "code")
    single.sort(key=lambda n: (not n.lower().endswith(entity), -distinct[n]))
    low = sorted((n for n in names if 2 <= distinct[n] <= GRAIN_PAIR_CARDINALITY), key=lambda n: distinct[n])
    return single[:GRAIN_KEYS], low[:GRAIN_PAIRS_OF]


def column_grain(source: BaseSQLSource, table: str, column: str) -> Grain | None:
    """
    The key ``column`` of ``table`` is a property of, a single column or a
    pair of low-cardinality ones, or None; decided once per table, query and
    column for ``GRAIN_TTL`` seconds. Raises ``Unchecked`` when a probe fails.
    """
    return _cached(source, ("grain", source.get_sql_expr(table), column), lambda: _decide(source, table, column))


def _decide(source: BaseSQLSource, table: str, column: str) -> Grain | None:
    singles, low = _candidates(source, table, column)
    found = [g for g in (dependency(source, table, column, (k,)) for k in singles) if g.is_property]
    if not found:
        alone = {k: dependency(source, table, column, (k,)).share for k in low}
        low = [k for k in low if alone[k] <= GRAIN_PAIR_ALONE]
        pairs = [(a, b) for i, a in enumerate(low) for b in low[i + 1:]]
        found = [g for g in (dependency(source, table, column, p) for p in pairs) if g.is_property]
    # The coarsest entity the value belongs to, e.g. a customer rather than each of its orders.
    return min(found, key=lambda g: (g.groups, -g.share)) if found else None
