"""
Column guards mask sensitive columns inside a Source, before a table, a schema
or a query result leaves it.

A guard is on when a Source has ``guardrails`` or ``mask_columns``. The first
time a table is read, a sample of it is checked by each guardrail and the
flagged columns, plus ``mask_columns`` and minus ``unmask_columns``, are masked
from then on:

- SQL sources rewrite every query they run, so each table a query reads is
  replaced by a subquery that masks those columns. Aliases, joins, schema
  statistics and LLM-written SQL all read the masked values, and none of the
  raw values reach a prompt.
- Other sources mask the DataFrame ``get`` returns, before it is cached.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import inspect
import threading

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any

import pandas as pd
import sqlglot

from sqlglot import exp

from ..util import (
    as_narwhals, as_pandas, is_narwhals, to_backend,
)

if TYPE_CHECKING:
    from .base import Source

#: Rows read from a table to decide which of its columns to mask.
SAMPLE_ROWS = 100

#: Alias given to the raw relation inside a masking subquery. Tables under it
#: are never masked again, which keeps the rewrite idempotent.
RAW_ALIAS = "__lumen_raw"

#: Schemas holding the database's own catalog, which are never masked.
SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "sys"}

# Set while a guarded call runs, so the calls it makes (fetch calling execute,
# execute_async calling execute in a thread) and the sampling that builds a
# plan run on the SQL as given.
_BYPASS: contextvars.ContextVar[bool] = contextvars.ContextVar("lumen_guard_bypass", default=False)


@contextmanager
def bypass():
    token = _BYPASS.set(True)
    try:
        yield
    finally:
        _BYPASS.reset(token)


def run_sync(coro):
    """Run a coroutine to completion from sync code, even inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


def hash_value(value: Any) -> str | None:
    """The 8-character hash a value is masked with; equal values hash equally."""
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    return hashlib.md5(str(value).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def build_plan(source: Source, sample) -> GuardPlan:
    """Decide which columns of ``sample`` to mask."""
    # lumen.ai is optional; a source only reaches it when given guardrails.
    from ..ai.guardrails import flagged_columns

    sample = as_pandas(sample)
    columns = tuple(str(column) for column in sample.columns)
    masked: dict[str, list[str]] = {}
    for guardrail in source.guardrails:
        for column, rules in run_sync(flagged_columns(sample, guardrail)).items():
            masked.setdefault(str(column), []).extend(rules)
    always = {column.lower() for column in source.mask_columns}
    never = {column.lower() for column in source.unmask_columns}
    for column in columns:
        if column.lower() in always:
            masked.setdefault(column, []).append("mask_columns")
    masked = {column: rules for column, rules in masked.items() if column.lower() not in never}
    return GuardPlan(columns=columns, masked=masked)


def mask_frame(df, plan: GuardPlan, mode: str):
    """Return ``df`` with the columns ``plan`` masks hashed, nulled or dropped."""
    columns = [column for column in plan.masked if column in df.columns]
    if not columns:
        return df
    is_dask = hasattr(df, "map_partitions")
    if not (is_dask or isinstance(df, pd.DataFrame)):
        # polars, pyarrow and other narwhals frames: mask as pandas and convert
        # back, which costs a copy but only for guarded tables.
        narwhals_df = as_narwhals(df)
        backend = narwhals_df.implementation.name.lower() if is_narwhals(narwhals_df) else None
        return to_backend(mask_frame(as_pandas(df), plan, mode), backend)
    if mode == "drop":
        return df.drop(columns=columns)
    mask = hash_value if mode == "hash" else (lambda _: None)
    df = df.copy()
    for column in columns:
        df[column] = df[column].map(mask, meta=(column, "object")) if is_dask else df[column].map(mask)
    return df


def _is_raw(node: exp.Expression) -> bool:
    """Whether ``node`` sits inside a relation a masking subquery reads."""
    while node is not None:
        if isinstance(node, (exp.Table, exp.Subquery)) and node.alias == RAW_ALIAS:
            return True
        node = node.parent
    return False


def _defines(table: exp.Table, name: str) -> bool:
    """Whether ``table`` sits in the body of the CTE ``name``, where it names the real table."""
    node = table.parent
    while node is not None:
        if isinstance(node, exp.CTE) and node.alias_or_name.lower() == name:
            return True
        node = node.parent
    return False


def relation_key(table: exp.Table, dialect: str | None) -> str | None:
    """Name a relation by what it reads, or None if it is never masked."""
    if isinstance(table.this, exp.Identifier):
        if table.db and table.db.lower() in SYSTEM_SCHEMAS:
            return None
        return ".".join(part.name for part in table.parts).lower()
    if isinstance(table.this, exp.Func):  # read_csv(...), read_parquet(...)
        return table.this.sql(dialect=dialect)
    return None


def masked_select(relation: exp.Expression, plan: GuardPlan, mode: str) -> exp.Select:
    """``SELECT <masked columns> FROM <relation> AS __lumen_raw``."""
    projections: list[exp.Expression] = []
    for column in plan.columns:
        ident = exp.to_identifier(column, quoted=True)
        if column not in plan.masked:
            projections.append(exp.Column(this=ident))
            continue
        if mode == "drop":
            continue
        text = exp.Cast(this=exp.Column(this=ident.copy()), to=exp.DataType.build("VARCHAR"))
        if mode == "hash":
            masked = exp.Substring(this=exp.MD5(this=text), start=exp.Literal.number(1), length=exp.Literal.number(8))
        else:
            masked = exp.Cast(this=exp.Null(), to=exp.DataType.build("VARCHAR"))
        projections.append(exp.alias_(masked, ident.copy()))
    relation = relation.copy()
    relation.set("alias", exp.TableAlias(this=exp.to_identifier(RAW_ALIAS)))
    return exp.select(*projections).from_(relation)


def guard_query(sql_query: str, dialect: str | None, plan_for: Callable[[str, exp.Table], GuardPlan | None], mode: str) -> str:
    """
    Replace each table ``sql_query`` reads with a subquery masking its columns.

    ``plan_for(key, table)`` returns the plan of a relation, or None to leave
    it as is. The query is returned unchanged if nothing needs masking.
    """
    statements = [s for s in sqlglot.parse(sql_query, read=dialect) if s is not None]
    changed = False
    for statement in statements:
        ctes = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
        for table in list(statement.find_all(exp.Table)):
            # Only relations a SELECT reads; not CREATE/INSERT/DROP targets.
            if not isinstance(table.parent, (exp.From, exp.Join)) or _is_raw(table):
                continue
            key = relation_key(table, dialect)
            if key is None or (not table.db and key in ctes and not _defines(table, key)):
                continue
            plan = plan_for(key, table)
            if plan is None or not plan.masked:
                continue
            alias = table.args.get("alias")
            alias = alias.copy() if alias else exp.TableAlias(this=exp.to_identifier(table.name or "__lumen_guarded"))
            subquery = masked_select(table, plan, mode).subquery()
            subquery.set("alias", alias)
            table.replace(subquery)
            changed = True
    if not changed:
        return sql_query
    return ";\n".join(statement.sql(dialect=dialect) for statement in statements)


def guard_expr(sql_expr: str, dialect: str | None, plan: GuardPlan, mode: str) -> str:
    """Wrap a table's SQL expression in a subquery masking its columns."""
    if not plan.masked:
        return sql_expr
    parsed = sqlglot.parse_one(sql_expr, read=dialect)
    # A source may name a table instead of giving a query for it.
    relation = parsed.subquery() if isinstance(parsed, exp.Query) else exp.to_table(sql_expr, dialect=dialect)
    return masked_select(relation, plan, mode).sql(dialect=dialect)


def guarded_sql(method):
    """
    Mask the tables the SQL query passed to ``method`` reads.

    Applied to every method of a SQL source that runs a query (``execute``,
    ``execute_async``, ``fetch``, ``execute_with_timeout``). Calls made from
    inside a guarded call run their SQL as given.
    """
    if inspect.iscoroutinefunction(method):
        @wraps(method)
        async def wrapped_async(self, sql_query, *args, **kwargs):
            if _BYPASS.get() or not self._guard_active:
                return await method(self, sql_query, *args, **kwargs)
            with bypass():
                # Building a plan may sample the table and call a guardrail.
                sql_query = await asyncio.to_thread(self._guard_sql, sql_query)
                return await method(self, sql_query, *args, **kwargs)
        wrapped_async.__lumen_guarded__ = True
        return wrapped_async

    @wraps(method)
    def wrapped(self, sql_query, *args, **kwargs):
        if _BYPASS.get() or not self._guard_active:
            return method(self, sql_query, *args, **kwargs)
        with bypass():
            sql_query = self._guard_sql(sql_query)
            return method(self, sql_query, *args, **kwargs)
    wrapped.__lumen_guarded__ = True
    return wrapped


def guarded_expr(method):
    """Mask the columns of the SQL expression ``get_sql_expr`` returns for a table."""
    @wraps(method)
    def wrapped(self, table, *args, **kwargs):
        sql_expr = method(self, table, *args, **kwargs)
        if _BYPASS.get() or not self._guard_active or not isinstance(sql_expr, str):
            return sql_expr
        with bypass():
            plan = self._table_plan(table, sql_expr)
        if plan is None:
            return sql_expr
        return guard_expr(sql_expr, self.dialect, plan, self.mask)
    wrapped.__lumen_guarded__ = True
    return wrapped


@dataclass(frozen=True)
class GuardPlan:
    """The columns of one table, and the reasons each masked one is masked."""

    columns: tuple[str, ...] = ()
    masked: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class GuardState:
    """
    Plans shared by a Source and the sources derived from it, so a table is
    sampled once and a derived table, built from already masked SQL, is not
    masked twice.
    """

    plans: dict[str, GuardPlan] = field(default_factory=dict)
    derived: set[str] = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)
