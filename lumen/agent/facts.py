"""
Facts the data tools state next to what they return, so the agent judges a
number with the shape of the data in view: what one row of a table is and
which columns hold one value per group of rows, the span of its dates, and
for a query's result, when it holds no rows or only zeros, which text values
its filters miss, and the latest date that "last month" and the like are read
from. They describe the data and never refuse a query: whether a sum over
repeated values is wrong is the agent's call.
"""
from __future__ import annotations

import asyncio
import difflib
import math
import re
import weakref

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
import sqlglot

from sqlglot import exp

from ..data.sharing import shares_values, sharing
from ..data.stats import (
    EXACT, ColumnStats, TableStats, format_name, format_value, get_adapter,
    quote_identifier,
)
from ..sources.base import Source
from ..util import as_pandas

# Seconds each query behind these facts may run.
FACTS_TIMEOUT = 10
# Columns tested as the group a table's rows repeat in, most distinct values first.
GROUPS_TESTED = 2
# Pairs of columns tested as a key when no single column is unique.
PAIRS_TESTED = 3
# Columns tested for holding one value per group, and number columns whose per-group sums
# the numbers among them are compared with.
DEPENDENTS_TESTED = 40
SUMS_TESTED = 12
# Column names a note lists before summarising the rest.
NAMES_SHOWN = 10
# Stored values read to check a filter's text against, and the closest of them shown.
VALUES_READ = 1000
CLOSEST_SHOWN = 4
# Rows of values a missed-values note gives as examples.
EXAMPLES_SHOWN = 2
# Dialects whose LIKE ignores case.
CASELESS_LIKE = frozenset({"sqlite", "mysql", "tsql", "mssql"})

_ID_NAME = re.compile(r"(?:^|_)(?:id|key|code|uuid)$", re.IGNORECASE)
_CAMEL_ID = re.compile(r"[a-z0-9]I[dD]$")
_YEAR_NAME = re.compile(r"year|(?:^|_)yr(?:$|_)", re.IGNORECASE)
_NOW_TYPES = tuple(getattr(exp, name) for name in (
    "CurrentDate", "CurrentTimestamp", "CurrentDatetime", "CurrentTime", "Localtimestamp", "Localtime",
    "CurrentTimestampLTZ",
) if hasattr(exp, name))
_NOW_FUNCTIONS = frozenset({
    "NOW", "TODAY", "GETDATE", "GETUTCDATE", "GET_CURRENT_TIMESTAMP", "SYSDATETIME", "CURDATE", "CURTIME",
    "SYSDATE", "SYSTIMESTAMP", "UTC_DATE", "UTC_TIMESTAMP", "CURRENT_DATE", "CURRENT_TIMESTAMP",
})

_GRAINS: weakref.WeakKeyDictionary[Source, dict[tuple[str, Any], str | None]] = weakref.WeakKeyDictionary()


def today() -> date:
    """The date relative periods would be read from, as a function so tests can fix it."""
    return date.today()


def _join(names: list[str]) -> str:
    names = [format_name(name) for name in names]
    if len(names) > NAMES_SHOWN:
        names = [*names[:NAMES_SHOWN - 1], f"{len(names) - NAMES_SHOWN + 1} more"]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _integer(col: ColumnStats) -> bool:
    return col.kind == "numeric" and "INT" in (col.type or "").upper()


def _unique(stats: TableStats, col: ColumnStats) -> bool:
    return col.kind != "other" and not col.nulls and col.distinct == stats.rows


def _id_like(name: str) -> bool:
    return bool(_ID_NAME.search(name) or _CAMEL_ID.search(name))


def _key_rank(col: ColumnStats) -> tuple:
    return (not col.primary_key, not _id_like(col.name), col.kind == "temporal")


# -- what one row of a table is ---------------------------------------------------------------------


@dataclass
class Group:
    """A column whose values repeat over rows, and the columns holding one value per value of it."""
    column: str
    values: int
    low: int
    high: int
    fixed: list[str]
    totals: list[str] = field(default_factory=list)
    """Of ``fixed``, the number columns that total the group, as ``"<column> is its row count"`` or ``"<column> is its SUM of <other>"``."""


@dataclass
class Grain:
    key: list[str] = field(default_factory=list)
    unique: list[str] = field(default_factory=list)
    """Other columns unique on their own."""
    groups: list[Group] = field(default_factory=list)


def _plan(stats: TableStats) -> tuple[list[ColumnStats], list[tuple[ColumnStats, ColumnStats]],
                                     list[tuple[ColumnStats, list[ColumnStats]]], list[ColumnStats]]:
    """
    The unique columns of a table with exact statistics, best key first, the
    pairs of columns to test as a key when none is unique, the columns to
    test as the group rows repeat in, each with the columns to test for one
    value per group: no more distinct values than the group has, and the
    number columns whose sums per group a number holding one value per
    group may total.
    """
    rows = stats.rows or 0
    columns = [c for c in stats.columns if c.kind != "other" and c.distinct is not None]
    # A unique measure, e.g. a float that never repeats, says nothing about what a row is.
    unique = sorted((c for c in columns if _unique(stats, c) and (c.kind in ("string", "temporal") or _integer(c))),
                    key=_key_rank)
    pairs = []
    if not unique:
        keyable = [c for c in columns if not c.nulls and c.distinct >= 2
                   and (c.kind in ("string", "boolean", "temporal") or _integer(c))]
        pairs = sorted(
            ((a, b) for i, a in enumerate(keyable) for b in keyable[i + 1:] if a.distinct * b.distinct >= rows),
            key=lambda pair: pair[0].distinct * pair[1].distinct,
        )[:PAIRS_TESTED]
    candidates = sorted(
        (c for c in columns if not _unique(stats, c) and 2 <= c.distinct
         and 2 * c.distinct <= rows * (1 - (c.nulls or 0)) and (c.kind == "string" or (_integer(c) and _id_like(c.name)))),
        key=lambda c: -c.distinct,
    )[:GROUPS_TESTED]
    groups = []
    for group in candidates:
        dependents = [c for c in columns if c is not group and not _unique(stats, c) and 2 <= c.distinct <= group.distinct]
        if dependents:
            groups.append((group, dependents[:DEPENDENTS_TESTED]))
    summed = [c for c in columns if c.kind == "numeric" and not _unique(stats, c)][:SUMS_TESTED]
    return unique, pairs, groups, summed


def _grain_sql(adapter, relation: str, pairs, groups, summed) -> str | None:
    """
    One statement answering every test of ``_plan``: one single-row
    subquery per pair and group, cross joined. Per group, a number column
    totals it when its value is the same multiple of the group's row count,
    or of another column's sum over the group, in every group.
    """
    def q(name: str) -> str:
        return quote_identifier(name, adapter.dialect)

    parts = []
    for i, (a, b) in enumerate(pairs):
        inner = f"SELECT COUNT(*) AS lumen_n FROM {relation} GROUP BY {q(a.name)}, {q(b.name)}"
        parts.append(f"SELECT MAX(lumen_n) AS lumen_p{i}_hi FROM ({inner}) {adapter.alias('lumen_g')}")
    for i, (group, dependents) in enumerate(groups):
        p = f"lumen_g{i}_"
        inner = ["COUNT(*) AS lumen_n"]
        outer = [f"COUNT(*) AS {p}n", f"MIN(lumen_n) AS {p}lo", f"MAX(lumen_n) AS {p}hi"]
        inner += [f"SUM({q(col.name)}) AS lumen_s{k}" for k, col in enumerate(summed)]
        for j, col in enumerate(dependents):
            inner.append(f"COUNT(DISTINCT {q(col.name)}) AS lumen_d{j}")
            outer.append(f"SUM(CASE WHEN lumen_d{j} > 1 THEN 1 ELSE 0 END) AS {p}v{j}")
            if col.kind != "numeric":
                continue
            inner.append(f"MIN({q(col.name)}) AS lumen_m{j}")
            if _integer(col):
                outer.append(f"SUM(CASE WHEN lumen_m{j} = lumen_n THEN 1 ELSE 0 END) AS {p}r{j}")
            for k, other in enumerate(summed):
                if other is not col:
                    ratio = f"lumen_m{j} * 1.0 / NULLIF(lumen_s{k}, 0)"
                    outer += [f"MIN({ratio}) AS {p}a{j}_{k}", f"MAX({ratio}) AS {p}b{j}_{k}"]
        sub = (f"SELECT {', '.join(inner)} FROM {relation} WHERE {q(group.name)} IS NOT NULL "
               f"GROUP BY {q(group.name)}")
        parts.append(f"SELECT {', '.join(outer)} FROM ({sub}) {adapter.alias('lumen_g')}")
    if not parts:
        return None
    return "SELECT * FROM " + " CROSS JOIN ".join(
        f"({part}) {adapter.alias(f'lumen_q{i}')}" for i, part in enumerate(parts)
    )


async def _run(adapter, sql: str, params) -> pd.DataFrame:
    if adapter.thread_bound:
        return adapter.execute(sql, params, FACTS_TIMEOUT)
    return await asyncio.to_thread(adapter.execute, sql, params, FACTS_TIMEOUT)


async def table_grain(source: Source, table: str, stats: TableStats) -> Grain | None:
    """
    What one row of ``table`` is, from its exact statistics and one query:
    its key, a single unique column or a unique pair, and the columns its
    rows repeat in with the columns holding one value per group. None when
    the query fails or the statistics are not exact, i.e. the table was too
    large for the profile to read in full.
    """
    if stats.method != EXACT or not stats.rows or stats.rows < 2:
        return None
    unique, pairs, groups, summed = _plan(stats)
    grain = Grain(key=[unique[0].name] if unique else [], unique=[c.name for c in unique[1:]])
    adapter = get_adapter(source)
    sql = _grain_sql(adapter, adapter.relation(table), pairs, groups, summed)
    if sql is None:
        return grain
    try:
        frame = await _run(adapter, sql, adapter.params(table))
        row = {str(k).lower(): frame[k].iloc[0] for k in frame.columns}
    except Exception:
        return None

    def number(name: str) -> int:
        value = row.get(name)
        return 0 if value is None or pd.isna(value) else int(value)

    if not grain.key:
        grain.key = next(([a.name, b.name] for i, (a, b) in enumerate(pairs) if number(f"lumen_p{i}_hi") == 1), [])
    for i, (group, dependents) in enumerate(groups):
        p = f"lumen_g{i}_"
        values = number(f"{p}n")
        fixed = [c for j, c in enumerate(dependents) if number(f"{p}v{j}") == 0]
        if not values or not fixed:
            continue
        low, high = number(f"{p}lo"), number(f"{p}hi")
        totals = []
        for j, col in enumerate(dependents):
            if col not in fixed or col.kind != "numeric":
                continue
            if f"{p}r{j}" in row and number(f"{p}r{j}") == values:
                totals.append(f"{format_name(col.name)} is its row count")
            # With groups of one size, any multiple of another column holding one value per group would pass.
            elif values >= 3 and low != high:
                total = next((f"{format_name(col.name)} is its SUM of {format_name(other.name)}{scale}"
                              for k, other in enumerate(summed) if (scale := _scale(row, f"{p}a{j}_{k}", f"{p}b{j}_{k}"))
                              is not None), None)
                if total:
                    totals.append(total)
        grain.groups.append(Group(group.name, values, low, high, [c.name for c in fixed], totals))
    return grain


def _scale(row: dict[str, Any], low: str, high: str) -> str | None:
    """How a ratio found in every group scales a sum, e.g. " / 1000", when it is one power of ten throughout."""
    a, b = row.get(low), row.get(high)
    if a is None or b is None or pd.isna(a) or pd.isna(b) or a <= 0 or abs(b - a) > 1e-9 * a:
        return None
    power = round(math.log10(a))
    if abs(power) > 9 or abs(a / 10 ** power - 1) > 1e-6:
        return None
    return "" if power == 0 else f" * {10 ** power}" if power > 0 else f" / {10 ** -power}"


def grain_note(grain: Grain) -> str | None:
    """A grain as a line under its table's header, or None when there is nothing to say."""
    sentences = []
    if grain.key:
        key = format_name(grain.key[0]) if len(grain.key) == 1 else f"({', '.join(map(format_name, grain.key))})"
        text = f"one row per {key}"
        if grain.unique:
            text += f"; {_join(grain.unique)} {'is' if len(grain.unique) == 1 else 'are'} unique too"
        sentences.append(text)
    for group in grain.groups:
        name = format_name(group.column)
        verb = "holds" if len(group.fixed) == 1 else "hold"
        sentences.append(
            f"{name} repeats, {group.values} values on {group.low} to {group.high} rows each; "
            f"{_join(group.fixed)} {verb} one value per {name}, repeated on each of its rows"
        )
        if group.totals:
            totals = group.totals if len(group.totals) == 1 else [*group.totals[:-1], f"and {group.totals[-1]}"]
            sentences.append(
                f"Per {name}, {', '.join(totals) if len(totals) > 2 else ' '.join(totals)}: totals that a SUM over "
                f"rows multiplies by the {name}'s row count"
            )
    if not sentences:
        return None
    return "grain: " + ". ".join(sentences) + "."


# -- the span of a table's dates --------------------------------------------------------------------


def _year_like(col: ColumnStats) -> bool:
    return (
        _integer(col) and bool(_YEAR_NAME.search(col.name)) and isinstance(col.min, int | float)
        and isinstance(col.max, int | float) and 1000 <= col.min <= col.max <= 2200
    )


def _dated(stats: TableStats) -> list[ColumnStats]:
    """The date, time and year columns of ``stats`` whose exact range may be shown."""
    if not stats.ranges_exact:
        return []
    return [
        col for col in stats.columns
        if col.min is not None and col.max is not None and (col.kind == "temporal" or _year_like(col))
        and shares_values(col.name)
    ]


def _span(col: ColumnStats) -> tuple[str, str]:
    if col.kind != "temporal":
        return str(int(col.min)), str(int(col.max))
    return format_value(col.min, "temporal", sql_type=col.type), format_value(col.max, "temporal", sql_type=col.type)


def _ends_before(cols: list[ColumnStats], day: date) -> bool:
    """Whether the latest of ``cols``, its dates if it has any, else its years, is before ``day``."""
    temporal = [c for c in cols if c.kind == "temporal"]
    try:
        if temporal:
            return max(pd.Timestamp(c.max).date() for c in temporal) < day
        return bool(cols) and max(int(c.max) for c in cols) < day.year
    except Exception:
        return False


def _anchor(latest: str) -> str:
    return f"read last month, this year and the like from {latest}, not from today, and say which dates you used"


def dates_note(stats: TableStats, day: date | None = None) -> str | None:
    """The span of a table's date, time and year columns as a line under its header, with what relative periods are read from."""
    cols = _dated(stats)
    if not cols:
        return None
    spans = ", ".join(f"{format_name(c.name)} {lo} to {hi}" for c in cols for lo, hi in [_span(c)])
    if _ends_before(cols, day or today()):
        return f"dates: {spans}. The data ends before today: {_anchor('its latest date')}."
    return f"dates: {spans}."


async def table_notes(source: Source, table: str, stats: TableStats | None) -> list[str]:
    """
    The lines list_tables and describe_table show under a table's header:
    its grain, computed once per profile of the table, and its dates.
    Nothing without exact statistics, or when the sharing level shows names
    only.
    """
    if stats is None or stats.method != EXACT or sharing.share == "none":
        return []
    cache = _GRAINS.setdefault(source, {})
    key = (table, stats.fingerprint or stats.computed_at)
    if key not in cache:
        grain = await table_grain(source, table, stats)
        cache[key] = grain_note(grain) if grain is not None else None
    return [note for note in (cache[key], dates_note(stats)) if note]


# -- what a query's result answers ------------------------------------------------------------------


def _number(value: Any) -> str:
    return "NULL" if pd.isna(value) else format_value(value, "numeric")


def answer_notes(frame: Any) -> list[str]:
    """
    A plain statement when a result answers nothing as it stands: it has no
    rows, its one row holds only zeros or NULLs, or a number column of
    several rows is all zero or NULL, or every number column holds one value.
    """
    df = as_pandas(frame)
    if df.empty:
        return ["The query returned no rows, so this is not an answer yet: check its filters, dates and joins "
                "against the data before presenting it."]
    numbers = df.select_dtypes(include="number")
    if numbers.empty:
        return []
    empty = [str(c) for c in numbers.columns if (numbers[c].fillna(0) == 0).all()]
    if len(df) == 1:
        if len(empty) < len(numbers.columns):
            return []
        values = ", ".join(f"{c} = {_number(numbers[c].iloc[0])}" for c in empty)
        return [f"The result's only numbers are zero or NULL ({values}), which usually means its filters or dates "
                "matched no rows: check them before presenting this as the answer."]
    if empty:
        facts = f"{_join(empty)} {'is' if len(empty) == 1 else 'are'} zero or NULL in all {len(df)} rows"
    elif all(numbers[c].nunique(dropna=False) == 1 for c in numbers.columns):
        values = ", ".join(f"{c} = {_number(numbers[c].iloc[0])}" for c in numbers.columns)
        facts = f"all {len(df)} rows hold the same numbers ({values})"
    else:
        return []
    return [f"In this result, {facts}, which usually means a filter, date range or join matched less than "
            "meant: check them before presenting this as the answer."]


def _read_tables(tree: exp.Expression, source: Source) -> dict[str, str]:
    """The source tables ``tree`` reads, by the name or alias it reads them under, lowercased."""
    try:
        names = {table.lower(): table for table in source.get_tables()}
    except Exception:
        return {}
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    found = {}
    for table in tree.find_all(exp.Table):
        if table.name.lower() in ctes or (name := names.get(table.name.lower())) is None:
            continue
        found[table.alias_or_name.lower()] = name
    return found


def _resolve(column: exp.Column, tables: dict[str, str], stats: dict[str, TableStats]) -> tuple[str, ColumnStats] | None:
    """The table and column statistics a column reference reads, if a table read by the query has it."""
    qualifier = column.table.lower()
    candidates = [tables[qualifier]] if qualifier in tables else [] if qualifier else list(dict.fromkeys(tables.values()))
    for table in candidates:
        if (found := stats.get(table)) is None:
            continue
        if col := next((c for c in found.columns if c.name.lower() == column.name.lower()), None):
            return table, col
    return None


def _reads_today(tree: exp.Expression) -> bool:
    for node in tree.walk():
        if isinstance(node, _NOW_TYPES):
            return True
        if isinstance(node, exp.Anonymous) and str(node.name).upper() in _NOW_FUNCTIONS:
            return True
        if (isinstance(node, exp.Literal) and node.is_string and node.this.strip().lower() in ("now", "today")
                and isinstance(node.parent, exp.Func | exp.Cast)):
            return True
    return False


def relative_date_notes(tree: exp.Expression, tables: dict[str, str], stats: dict[str, TableStats],
                        day: date | None = None) -> list[str]:
    """
    When a query reads today's date and the dates it reads end before
    today, where they end, and that relative periods are read from there.
    """
    if not _reads_today(tree):
        return []
    dated = {table: _dated(found) for table, found in stats.items()}
    used = []
    for column in tree.find_all(exp.Column):
        resolved = _resolve(column, tables, stats)
        if resolved is not None and resolved[1] in dated.get(resolved[0], []) and resolved not in used:
            used.append(resolved)
    if not used:
        used = [(table, col) for table, cols in dated.items() for col in cols if col.kind == "temporal"]
    day = day or today()
    ended = [(table, col) for table, col in used if _ends_before([col], day)]
    if not ended:
        return []
    ends = ", ".join(f"{table}.{format_name(col.name)} ends at {_span(col)[1]}" for table, col in ended)
    return [f"The query reads today's date, but {ends}, before today. Unless the user asked about calendar "
            f"dates, {_anchor('that latest date')}."]


def span_notes(tree: exp.Expression, tables: dict[str, str], stats: dict[str, TableStats]) -> list[str]:
    """The span of each date, time or year column a query compares with a value, for a result that answers nothing."""
    dated = {table: _dated(found) for table, found in stats.items()}
    spans = []
    for node in tree.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.Between):
        for column in node.find_all(exp.Column):
            resolved = _resolve(column, tables, stats)
            if resolved is None or resolved[1] not in dated.get(resolved[0], []):
                continue
            table, col = resolved
            lo, hi = _span(col)
            span = f"{table}.{format_name(col.name)} runs from {lo} to {hi}"
            if span not in spans:
                spans.append(span)
    return [f"The data it filters on: {'; '.join(spans)}."] if spans else []


@dataclass
class TextFilter:
    table: str
    column: ColumnStats
    op: str
    """``=`` (also for IN), ``LIKE`` or ``ILIKE``."""
    text: str
    caseless: bool


def _filtered_column(node: exp.Expression) -> tuple[exp.Column, bool] | None:
    """The column a filter compares, through LOWER, UPPER and TRIM, and whether case is ignored."""
    caseless = False
    while isinstance(node, exp.Lower | exp.Upper | exp.Trim):
        caseless |= not isinstance(node, exp.Trim)
        node = node.this
    return (node, caseless) if isinstance(node, exp.Column) else None


def _string(node: exp.Expression) -> str | None:
    return node.this if isinstance(node, exp.Literal) and node.is_string else None


def text_filters(tree: exp.Expression, tables: dict[str, str], stats: dict[str, TableStats], dialect: str | None) -> list[TextFilter]:
    """The comparisons of a text column with a text value in ``tree``: =, IN, LIKE and ILIKE, not negated."""
    found = []

    def add(column_node, op, values):
        if (target := _filtered_column(column_node)) is None:
            return
        column, caseless = target
        if (resolved := _resolve(column, tables, stats)) is None or resolved[1].kind != "string":
            return
        if op == "LIKE" and dialect in CASELESS_LIKE:
            op = "ILIKE"
        for value in values:
            found.append(TextFilter(resolved[0], resolved[1], op, value, caseless or op == "ILIKE"))

    for node in tree.find_all(exp.EQ, exp.In, exp.Like, exp.ILike):
        if isinstance(node.parent, exp.Not):
            continue
        if isinstance(node, exp.EQ):
            for side, other in ((node.this, node.expression), (node.expression, node.this)):
                if (value := _string(other)) is not None:
                    add(side, "=", [value])
        elif isinstance(node, exp.In):
            values = [_string(v) for v in node.expressions]
            if values and None not in values:
                add(node.this, "=", values)
        elif (pattern := _string(node.expression)) is not None:
            add(node.this, "ILIKE" if isinstance(node, exp.ILike) else "LIKE", [pattern])
    return found


def _like_regex(pattern: str, caseless: bool) -> re.Pattern:
    body = "".join(".*" if ch == "%" else "." if ch == "_" else re.escape(ch) for ch in pattern)
    return re.compile(body, re.IGNORECASE | re.DOTALL if caseless else re.DOTALL)


def _abbreviates(short: str, long: str) -> bool:
    """
    Whether ``short`` could abbreviate ``long``: the initials of its words,
    as NY does New York, or of one word, its letters in order from the
    first, as TX does Texas.
    """
    short, long = short.lower(), long.lower()
    if not (2 <= len(short) <= 4 and short.isalpha() and len(long) > len(short) and short[0] == long[0]):
        return False
    words = [word for word in re.findall(r"[a-z]+", long) if word not in ("of", "the", "and")]
    if len(words) > 1:
        return short == "".join(word[0] for word in words)
    rest = iter(long)
    return all(ch in rest for ch in short)


def closest(text: str, values: list[str], n: int = CLOSEST_SHOWN) -> list[str]:
    """The stored values ``text`` most likely means: the same ignoring case, an abbreviation either way, a part of it or a near spelling."""
    text = text.strip()
    low = text.lower()
    if not low:
        return []
    picked: list[str] = []
    stripped = [(v, v.strip(), v.strip().lower()) for v in values]
    tiers = (
        [v for v, _, s in stripped if s == low],
        [v for v, s, _ in stripped if _abbreviates(s, text) or _abbreviates(text, s)],
        [v for v, _, s in stripped if low in s or (len(s) > 2 and s in low)],
        [lowered for lowered in difflib.get_close_matches(low, [s for _, _, s in stripped], n=n, cutoff=0.75)],
    )
    by_lower = {s: v for v, _, s in reversed(stripped)}
    for i, tier in enumerate(tiers):
        for value in tier:
            value = by_lower[value] if i == 3 else value
            if value not in picked:
                picked.append(value)
    return picked[:n]


def _quoted(value: str) -> str:
    return format_value(value, "string", quote=True)


def _counted(values: list[tuple[str, int]]) -> str:
    return ", ".join(f"{_quoted(v)} ({n} rows)" for v, n in values)


def _matches(found: TextFilter) -> Callable[[str], bool]:
    if found.op != "=":
        return _like_regex(found.text, found.caseless).fullmatch
    text = found.text.lower() if found.caseless else found.text
    return (lambda value: value.lower() == text) if found.caseless else (lambda value: value == text)


def _any_of(matchers: list[Callable[[str], bool]]) -> Callable[[str], bool]:
    return lambda value: any(match(value) for match in matchers)


def filter_notes(found: TextFilter, stored: list[tuple[str, int]], complete: bool,
                 covered: Callable[[str], bool] | None = None) -> list[str]:
    """
    What a text filter misses, checked against the column's stored values
    and their row counts (all of them when ``complete``): a value no row
    holds, with the closest stored ones, or for a pattern, the rows it
    misses that hold its text in another case or an abbreviation of it.
    ``covered`` tells the values the query's other filters on the column
    match, which a pattern does not miss.
    """
    name = format_name(found.column.name)
    shown = shares_values(found.column.name, [v for v, _ in stored])
    counts = dict(stored)
    if found.op == "=":
        if found.text in counts or (found.caseless and any(v.lower() == found.text.lower() for v in counts)):
            return []
        if not complete:
            return []
        note = f"No row of {found.table} has {name} = {_quoted(found.text)}{', ignoring case' if found.caseless else ''}."
        if not shown:
            return [note]
        if near := closest(found.text, list(counts)):
            return [f"{note} The closest stored values: {_counted([(v, counts[v]) for v in near])}."]
        top = sorted(stored, key=lambda item: -item[1])[:CLOSEST_SHOWN]
        return [f"{note} Its most frequent values: {_counted(top)}; distinct_values lists more."]
    regex = _like_regex(found.text, found.caseless)
    matched = sum(n for v, n in stored if regex.fullmatch(v))
    missed = [(v, n) for v, n in stored if not regex.fullmatch(v) and not (covered and covered(v))]
    clause = f"{name} {found.op} {_quoted(found.text)}"
    core = found.text.strip("%_")
    kinds = []
    if not found.caseless:
        loose = _like_regex(found.text, True)
        if cased := [(v, n) for v, n in missed if loose.fullmatch(v)]:
            kinds.append(("that match it ignoring case", cased))
    if core.isalpha() and len(core) > 4:
        tokens: dict[str, list[tuple[str, int]]] = {}
        for v, n in missed:
            for token in dict.fromkeys(re.findall(r"\b[A-Z]{2,3}\b", v)):
                if _abbreviates(token, core):
                    tokens.setdefault(token, []).append((v, n))
        for token, rows in tokens.items():
            kinds.append((f"with {_quoted(token)} as a word" if shown else "with an abbreviation of it", rows))
    if not kinds:
        if matched or not complete:
            return []
        note = f"No row of {found.table} has {clause}."
        if shown and (near := closest(core, list(counts))):
            note += f" The closest stored values: {_counted([(v, counts[v]) for v in near])}."
        return [note]
    misses = []
    for what, rows in kinds:
        examples = sorted(rows, key=lambda item: -item[1])[:EXAMPLES_SHOWN]
        example = f", e.g. {', '.join(_quoted(v) for v, _ in examples)}" if shown else ""
        misses.append(f"{sum(n for _, n in rows)}{'' if complete else ' or more'} rows {what}{example}")
    return [f"{clause} matches {matched} rows of {found.table}, not the {'; nor the '.join(misses)}. "
            "Check whether they belong in the answer."]


async def stored_values(source: Source, table: str, column: ColumnStats) -> tuple[list[tuple[str, int]], bool] | None:
    """A text column's stored values with their row counts, most frequent first, and whether that is all of them."""
    if column.values and column.values_complete and all(isinstance(n, int) for _, n in column.values):
        return [(str(v), int(n)) for v, n in column.values], True
    adapter = get_adapter(source)
    q = quote_identifier(column.name, adapter.dialect)
    sql = adapter.limit(
        f"SELECT {q} AS lumen_v, COUNT(*) AS lumen_n FROM {adapter.relation(table)} WHERE {q} IS NOT NULL "
        f"GROUP BY {q} ORDER BY lumen_n DESC, lumen_v", VALUES_READ + 1,
    )
    try:
        frame = await _run(adapter, sql, adapter.params(table))
    except Exception:
        return None
    values = [(str(v), int(n)) for v, n in zip(frame.iloc[:, 0], frame.iloc[:, 1], strict=False)]
    return values[:VALUES_READ], len(values) <= VALUES_READ


async def query_notes(source: Source, sql: str, frame: Any,
                      stats_of: Callable[[str], TableStats | None]) -> list[str]:
    """
    The facts a query's result is reported with, beyond ``answer_notes``:
    where the dates it reads end, when it reads today's date, and which of
    its text filters miss stored values. Text filters are checked on tables
    small enough to have exact statistics, with at most one query per
    filtered column.
    """
    dialect = getattr(source, "dialect", None)
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return []
    if tree is None:
        return []
    tables = _read_tables(tree, source)
    stats = {table: found for table in set(tables.values()) if (found := stats_of(table)) is not None}
    notes = relative_date_notes(tree, tables, stats)
    if not notes and answer_notes(frame):
        notes += span_notes(tree, tables, stats)
    by_column: dict[tuple[str, str], list[TextFilter]] = {}
    for found in text_filters(tree, tables, stats, dialect):
        if stats[found.table].method == EXACT:
            by_column.setdefault((found.table, found.column.name), []).append(found)
    for (table, _), filters in by_column.items():
        values = await stored_values(source, table, filters[0].column)
        if values is None:
            continue
        covered = _any_of([_matches(found) for found in filters])
        for found in filters:
            notes += filter_notes(found, *values, covered=covered)
    return list(dict.fromkeys(notes))
