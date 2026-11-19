"""
Column contracts: which tables and columns a query reads, and whether a
source still has them with the types it had.

``query_columns`` resolves the columns a SQL query reads to the tables they
come from, with sqlglot and no engine: a column is assigned by its
qualifier, by the one table its query level reads, or by the table a
schema lists it in. ``source_schema`` asks a source for the columns of its
tables, fetching no rows, and ``schema_problems`` compares a recorded
contract (tables, columns and their type families) with a source's schema,
so a query can be refused before it runs on data it cannot use.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import sqlglot

from sqlglot import exp
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from ..sources.base import BaseSQLSource, Source

# Column types are compared by family. Integer and number columns are interchangeable,
# since new data often turns one into the other (a missing value, a decimal).
FAMILIES = ("integer", "number", "string", "boolean", "datetime", "other")
_NUMERIC = {"integer", "number"}

# How completely ``query_columns`` knows what a query reads.
FULL, PARTIAL, NONE = "full", "partial", "none"

Schema = Mapping[str, Mapping[str, Any]]


def family(dtype: Any) -> str:
    """The family of a pandas dtype, a dtype name or a SQL type name."""
    if isinstance(dtype, str) and not _is_dtype_name(dtype):
        return _sql_family(dtype)
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_numeric_dtype(dtype):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "string"
    return "other"


def _is_dtype_name(name: str) -> bool:
    try:
        pd.api.types.pandas_dtype(name)
    except TypeError:
        return False
    return True


def _sql_family(name: str) -> str:
    name = name.upper()
    if "BOOL" in name:
        return "boolean"
    if "INT" in name:
        return "integer"
    if any(key in name for key in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC")):
        return "number"
    if "DATE" in name or "TIME" in name:
        return "datetime"
    if any(key in name for key in ("CHAR", "TEXT", "STRING", "CLOB")):
        return "string"
    return "other"


def compatible(expected: str, actual: str) -> bool:
    """Whether a column of family ``actual`` can stand in for one recorded as ``expected``."""
    return expected in (actual, "other") or {expected, actual} <= _NUMERIC


def source_dialect(source: Source) -> str | None:
    """The sqlglot dialect of a source's SQL, None when it has none."""
    dialect = getattr(source, "dialect", None)
    return None if dialect in (None, "any") else dialect


def table_schema(source: Source, table: str) -> dict[str, str]:
    """The columns of ``table`` and their families, fetching no rows."""
    if isinstance(source, BaseSQLSource):
        name = exp.to_identifier(table, quoted=True).sql(dialect=source_dialect(source))
        frame = source.execute(f"SELECT * FROM {name} LIMIT 0")
        return {str(column): family(dtype) for column, dtype in frame.dtypes.items()}
    schema = source.get_schema(table, limit=1)
    return {str(column): _json_family(spec) for column, spec in schema.items() if column != "__len__"}


def _json_family(spec: Mapping[str, Any]) -> str:
    kind = spec.get("type")
    if kind == "string" and spec.get("format") in ("date", "date-time", "datetime"):
        return "datetime"
    return kind if kind in ("integer", "number", "boolean", "string") else "other"


def source_schema(source: Source, tables: Iterable[str] | None = None) -> dict[str, dict[str, str]]:
    """
    The schemas of ``tables`` of ``source``, by default all of them, keyed as
    given. Names match case-insensitively; a table the source lacks is left out.
    """
    names = list(source.get_tables())
    lowered = {name.lower(): name for name in names}
    found = {}
    for table in names if tables is None else tables:
        actual = table if table in names else lowered.get(table.lower())
        if actual is not None:
            found[table] = table_schema(source, actual)
    return found


@dataclass
class QueryColumns:
    """
    The tables a query reads and, for each, the columns it reads.

    ``coverage`` is ``full`` when every column is known, ``partial`` when
    the query selects a table with ``*``, so it may rely on more columns
    than listed, and ``none`` when the query does not parse or reads
    columns no table could be found for, which are in ``unresolved``.
    """

    tables: dict[str, set[str]] = field(default_factory=dict)
    coverage: str = FULL
    unresolved: set[str] = field(default_factory=set)
    reason: str | None = None

    def add(self, table: str, column: str | None = None):
        columns = self.tables.setdefault(table, set())
        if column is not None:
            columns.add(column)


def _resolver(schema: Schema):
    """Map table and column names case-insensitively onto the schema's own spelling."""
    tables = {name.lower(): name for name in schema}

    def table(name: str) -> str:
        return tables.get(name.lower(), name)

    def column(table_name: str, name: str) -> str:
        lowered = {c.lower(): c for c in schema.get(table(table_name)) or {}}
        return lowered.get(name.lower(), name)

    return table, column


def _projected_star(scope: Scope) -> list[str]:
    """The aliases of the tables ``scope`` selects with ``*`` or ``t.*``; ``COUNT(*)`` is not one."""
    select = scope.expression
    if not isinstance(select, exp.Select):
        return []
    stars = []
    for projection in select.expressions:
        if isinstance(projection, exp.Star):
            stars += [alias for alias, source in scope.sources.items() if isinstance(source, exp.Table)]
        elif isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
            if isinstance(scope.sources.get(projection.table), exp.Table):
                stars.append(projection.table)
    return stars


def _is_table(source: Any) -> bool:
    return isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier)


def query_columns(sql: str, schema: Schema | None = None, dialect: str | None = None) -> QueryColumns:
    """
    The tables and columns ``sql`` reads, resolved against ``schema`` (table
    to columns), which is only needed for unqualified columns of a query
    level that reads several tables. Names take the schema's spelling.
    CTEs, subqueries and output aliases are not tables or columns.
    """
    schema = schema or {}
    found = QueryColumns()
    table_name, column_name = _resolver(schema)
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except SqlglotError as exc:
        found.coverage, found.reason = NONE, f"The SQL does not parse: {str(exc).splitlines()[0]}"
        return found
    ctes = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    known = {
        table_name(t.name): {str(c): "TEXT" for c in schema[table_name(t.name)]}
        for t in parsed.find_all(exp.Table)
        if _is_table(t) and t.name.lower() not in ctes and schema.get(table_name(t.name))
    }
    scopes, error = None, None
    # A qualified column the schema lacks makes qualify refuse the query; read
    # without the schema it is kept as written, so a check can say it is missing.
    for attempt in ([known, None] if known else [None]):
        try:
            qualified = qualify(parsed.copy(), schema=attempt, dialect=dialect, expand_stars=False,
                                validate_qualify_columns=False, quote_identifiers=False)
            scopes = traverse_scope(qualified)
            break
        except (OptimizeError, SqlglotError) as exc:
            error = exc
    if scopes is None:
        found.coverage = NONE
        found.reason = f"The query could not be analysed: {str(error).split('. Line:')[0]}"
        return found
    for scope in scopes:
        tables = {alias: source for alias, source in scope.sources.items() if _is_table(source)}
        for source in tables.values():
            found.add(table_name(source.name))
        outputs = {s.alias_or_name.lower() for s in getattr(scope.expression, "selects", [])}
        for column in scope.columns:
            if column.table:
                source = scope.sources.get(column.table)
                if _is_table(source):
                    found.add(table_name(source.name), column_name(source.name, column.name))
            elif len(scope.sources) == 1 and len(tables) == 1:
                source = next(iter(tables.values()))
                found.add(table_name(source.name), column_name(source.name, column.name))
            elif not (column.find_ancestor(exp.Order) and column.name.lower() in outputs):
                found.unresolved.add(column.name)
        for alias in _projected_star(scope):
            found.add(table_name(scope.sources[alias].name))
            if found.coverage == FULL:
                found.coverage, found.reason = PARTIAL, f"It reads {scope.sources[alias].name} with *."
    if found.unresolved:
        found.coverage = NONE
        found.reason = f"Columns with no known table: {', '.join(sorted(found.unresolved))}."
    return found


@dataclass(frozen=True)
class ColumnProblem:
    """A table or column a contract records that a schema lacks or holds with another type."""

    table: str
    column: str | None
    message: str


def schema_problems(expected: Schema, schema: Schema) -> list[ColumnProblem]:
    """
    What ``schema`` (table to column to family) lacks of ``expected``, the
    same shape as recorded: missing tables, missing columns and columns
    whose family is not ``compatible``. Names match case-insensitively.
    """
    lowered = {t.lower(): t for t in schema}
    problems = []
    for table, columns in expected.items():
        actual_table = table if table in schema else lowered.get(table.lower())
        if actual_table is None:
            problems.append(ColumnProblem(table, None, f"The data has no table {table!r}."))
            continue
        actual = {c.lower(): kind for c, kind in schema[actual_table].items()}
        for column, kind in columns.items():
            if column.lower() not in actual:
                problems.append(ColumnProblem(table, column, f"Table {table!r} has no column {column!r}."))
            elif not compatible(kind, actual[column.lower()]):
                problems.append(ColumnProblem(table, column, (
                    f"Column {table}.{column} is {actual[column.lower()]} now; the contract expects {kind}."
                )))
    return problems


__all__ = [
    "FAMILIES",
    "FULL",
    "NONE",
    "PARTIAL",
    "ColumnProblem",
    "QueryColumns",
    "compatible",
    "family",
    "query_columns",
    "schema_problems",
    "source_dialect",
    "source_schema",
    "table_schema",
]
