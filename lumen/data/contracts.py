"""
Column contracts: which tables and columns a query reads, and whether a
source still has them with the types it had.

``query_columns`` resolves the columns a SQL query reads to the tables they
come from, with sqlglot and no engine: a column is assigned by its
qualifier, by the one source its query level reads, through CTEs and
subqueries that select ``*``, or by the table a schema lists it in.
``source_schema`` asks a source for the columns of its tables and their
types, and ``schema_problems`` compares a recorded contract (tables,
columns and their type families) with a source's schema, so a query can be
refused before it runs on data it cannot use.
"""
from __future__ import annotations

import logging
import re

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
from ..transforms.sql import SQLLimit

logger = logging.getLogger(__name__)

# Column types are compared by family. A recorded ``number`` column may become
# ``integer`` and a ``datetime`` one ``date``, never the other way round.
FAMILIES = ("integer", "number", "string", "boolean", "date", "datetime", "other")
_WIDER = {"number": {"integer"}, "datetime": {"date"}}

# How completely ``query_columns`` knows what a query reads.
FULL, PARTIAL, NONE = "full", "partial", "none"

Schema = Mapping[str, Mapping[str, Any]]

_INTEGER = {
    "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT", "HUGEINT", "UHUGEINT",
    "UBIGINT", "UINTEGER", "USMALLINT", "UTINYINT", "INT1", "INT2", "INT4", "INT8", "INT16",
    "INT32", "INT64", "INT128", "UINT8", "UINT16", "UINT32", "UINT64", "SERIAL", "BIGSERIAL",
    "SMALLSERIAL", "LONG", "BYTEINT",
}
_NUMBER = {"DOUBLE", "FLOAT", "FLOAT4", "FLOAT8", "FLOAT32", "FLOAT64", "REAL", "DECIMAL",
           "NUMERIC", "NUMBER", "MONEY", "BIGNUMERIC"}
_STRING = {"CHAR", "VARCHAR", "NVARCHAR", "NCHAR", "TEXT", "STRING", "CLOB", "NTEXT", "UUID",
           "CHARACTER", "BPCHAR", "ENUM", "CITEXT"}
_BOOLEAN = {"BOOL", "BOOLEAN", "BIT"}
_DATETIME = {"TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ", "TIMESTAMP_TZ",
             "DATETIME", "DATETIME2", "SMALLDATETIME", "DATETIMEOFFSET", "TIME", "TIMETZ"}


def family(dtype: Any) -> str:
    """The family of a pandas dtype, a dtype name or a SQL type name, e.g. ``NUMBER(38,0)``."""
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
    if pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype):
        return "string"
    return "other"


def _is_dtype_name(name: str) -> bool:
    try:
        pd.api.types.pandas_dtype(name)
    except TypeError:
        return False
    return True


_SQL_TYPE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)[^(]*(?:\((.*)\))?")


def _sql_family(name: str) -> str:
    match = _SQL_TYPE.match(name)
    if match is None:
        return "other"
    base = match[1].upper()
    args = [a.strip() for a in (match[2] or "").split(",")]
    if base in _BOOLEAN:
        return "boolean"
    if base in _INTEGER:
        return "integer"
    if base in _NUMBER:
        # NUMBER(38,0) and DECIMAL(10) have no fractional digits.
        if base in ("DECIMAL", "NUMERIC", "NUMBER") and match[2] and (len(args) == 1 or args[1] == "0"):
            return "integer"
        return "number"
    if base == "DATE":
        return "date"
    if base in _DATETIME:
        return "datetime"
    if base in _STRING:
        return "string"
    return "other"


def compatible(expected: str, actual: str) -> bool:
    """
    Whether a column of family ``actual`` can stand in for one recorded as
    ``expected``. A type that is unknown on either side (``other``) passes;
    ``schema_problems`` warns about an unknown actual type.
    """
    return "other" in (expected, actual) or expected == actual or actual in _WIDER.get(expected, ())


def source_dialect(source: Source) -> str | None:
    """The sqlglot dialect of a source's SQL, None when it has none."""
    dialect = getattr(source, "dialect", None)
    return None if dialect in (None, "any") else dialect


def _probe(source: BaseSQLSource, table: str) -> str:
    """A query returning no rows of ``table``, in the source's dialect."""
    return SQLLimit(limit=0, write=source.dialect, identify=False).apply(source.get_sql_expr(table))


def _reflected(source: Source, table: str) -> dict[str, str] | None:
    """The column types SQLAlchemy reflects for a stored table, None for any other table."""
    inspector = getattr(source, "_inspector", None)
    tables = getattr(source, "tables", None)
    if inspector is None or (isinstance(tables, dict) and table in tables):
        return None
    schema, _, name = table.rpartition(".")
    try:
        columns = inspector.get_columns(name, schema=schema or getattr(source, "schema", None))
    except Exception:
        return None
    return {str(col["name"]): family(str(col["type"])) for col in columns} or None


def table_schema(source: Source, table: str) -> dict[str, str]:
    """
    The columns of ``table`` and their families. A SQL source answers a
    query that returns no rows: DuckDB describes it, SQLAlchemy reflects a
    stored table, and other engines give an empty result whose types may be
    unknown (``other``). Any other source computes its schema, which reads
    data. Raises whatever the source raises when it cannot read the table.
    """
    if isinstance(source, BaseSQLSource):
        if (reflected := _reflected(source, table)) is not None:
            return reflected
        probe = _probe(source, table)
        if source.dialect == "duckdb":
            described = source.execute(f"DESCRIBE {probe}")
            return {
                str(row["column_name"]): family(str(row["column_type"]))
                for _, row in described.iterrows()
            }
        frame = source.execute(probe)
        return {str(column): family(dtype) for column, dtype in frame.dtypes.items()}
    schema = source.get_schema(table, limit=1)
    return {str(column): _json_family(spec) for column, spec in schema.items() if column != "__len__"}


def _json_family(spec: Mapping[str, Any]) -> str:
    kind = spec.get("type")
    if kind == "string" and spec.get("format") == "date":
        return "date"
    if kind == "string" and spec.get("format") in ("date-time", "datetime"):
        return "datetime"
    return kind if kind in ("integer", "number", "boolean", "string") else "other"


def _find_table(name: str, names: Iterable[str]) -> str | None:
    """The table of ``names`` that ``name`` refers to: the same name in any case, or a unique last part."""
    names = list(names)
    lowered = {n.lower(): n for n in names}
    if name in names:
        return name
    if name.lower() in lowered:
        return lowered[name.lower()]
    parts = name.lower().split(".")
    matches = [n for n in names if n.lower().split(".")[-len(parts):] == parts]
    return matches[0] if len(matches) == 1 else None


class SourceSchema(dict):
    """Table to column to family, and in ``unreadable`` the tables that exist but could not be read, with why."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unreadable: dict[str, str] = {}


def source_schema(source: Source, tables: Iterable[str] | None = None) -> SourceSchema:
    """
    The schemas of ``tables`` of ``source``, by default all of them, keyed as
    given. A name matches a table whose trailing name parts it equals, so
    ``orders`` finds ``public.orders`` and ``analytics.orders`` finds
    ``DB.ANALYTICS.ORDERS`` when no other table matches. A table the source
    lacks is left out, and one it cannot describe is listed in ``unreadable``.
    """
    names = list(source.get_tables())
    found = SourceSchema()
    for table in names if tables is None else tables:
        actual = _find_table(table, names)
        if actual is None:
            continue
        try:
            found[table] = table_schema(source, actual)
        except Exception as exc:
            logger.warning("Could not read the columns of table %r: %s", actual, exc)
            text = str(exc).strip()
            found.unreadable[table] = text.splitlines()[0] if text else type(exc).__name__
    return found


@dataclass
class QueryColumns:
    """
    The tables a query reads and, for each, the columns it reads.

    ``coverage`` is ``full`` when every column is known; ``partial`` when the
    query selects a table with ``*``, or reads a table function, a pivot or
    another source whose columns are not recorded, so it may rely on more
    than is listed; and ``none`` when the query does not parse or reads
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

    def partly(self, reason: str):
        if self.coverage == FULL:
            self.coverage, self.reason = PARTIAL, reason


def _is_table(source: Any) -> bool:
    return isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier)


def _table_key(table: exp.Table) -> str:
    return ".".join(part.name for part in table.parts)


def _projected_star(scope: Scope, any_source: bool = False) -> list[str]:
    """
    The aliases of the tables ``scope`` selects with ``*`` or ``t.*``
    (``COUNT(*)`` is not one); with ``any_source`` also derived tables and
    other sources.
    """
    select = scope.expression
    if not isinstance(select, exp.Select):
        return []

    def wanted(source):
        return any_source or _is_table(source)

    stars = []
    for projection in select.expressions:
        if isinstance(projection, exp.Star):
            stars += [alias for alias, source in scope.sources.items() if wanted(source)]
        elif isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
            if wanted(scope.sources.get(projection.table)):
                stars.append(projection.table)
    return stars


def _alias_ref(column: exp.Column, scope: Scope, outputs: set[str]) -> bool:
    """
    Whether an unqualified column names an output alias: in GROUP BY,
    HAVING, ORDER BY or QUALIFY, or in another projection of the same
    SELECT, as DuckDB's lateral aliases allow.
    """
    name = column.name.lower()
    if name not in outputs:
        return False
    if column.find_ancestor(exp.Group, exp.Having, exp.Order, exp.Qualify) is not None:
        return True
    inside = column.find_ancestor(exp.Alias)
    for projection in getattr(scope.expression, "selects", []):
        if not isinstance(projection, exp.Alias) or projection.alias.lower() != name:
            continue
        renames = not (isinstance(projection.this, exp.Column) and projection.this.name.lower() == name)
        return renames and inside is not None and inside is not projection
    return False


class _Resolver:
    """Resolves the columns of each query level onto tables, following ``*`` through derived tables."""

    def __init__(self, found: QueryColumns, schema: Schema):
        self.found = found
        self.schema = schema

    def table(self, source: exp.Table) -> str:
        key = _table_key(source)
        return _find_table(key, self.schema) or key

    def add(self, source: exp.Table, name: str):
        table = self.table(source)
        lowered = {c.lower(): c for c in self.schema.get(table) or {}}
        self.found.add(table, lowered.get(name.lower(), name))

    def opaque(self, source: Any):
        self.found.partly(f"It reads {source.sql()[:60]}, whose columns are not recorded.")

    def through(self, scope: Scope, name: str, seen: frozenset = frozenset()) -> bool:
        """Record column ``name`` of a derived table on the table it selects ``*`` from."""
        if id(scope) in seen or not isinstance(scope.expression, exp.Select):
            return False
        if name.lower() in {s.alias_or_name.lower() for s in scope.expression.selects}:
            return True
        for alias in _projected_star(scope, any_source=True):
            source = scope.sources.get(alias)
            if _is_table(source):
                self.add(source, name)
                return True
            if isinstance(source, Scope) and self.through(source, name, seen | {id(scope)}):
                return True
            if source is not None and not isinstance(source, Scope):
                self.opaque(source)
                return True
        return False

    def resolve(self, scope: Scope, column: exp.Column, outputs: set[str]):
        sources = scope.sources
        if column.table:
            source = sources.get(column.table)
            if _is_table(source):
                self.add(source, column.name)
            elif isinstance(source, Scope):
                self.through(source, column.name)
            elif source is not None:
                self.opaque(source)
            return
        if _alias_ref(column, scope, outputs):
            return
        if len(sources) == 1:
            source = next(iter(sources.values()))
            if _is_table(source):
                self.add(source, column.name)
            elif not isinstance(source, Scope):
                self.opaque(source)
            elif not self.through(source, column.name):
                self.found.unresolved.add(column.name)
            return
        self.found.unresolved.add(column.name)


def _scopes(parsed: exp.Expression, schema: dict, dialect: str | None) -> tuple[list[Scope], bool]:
    """
    The query levels of ``parsed``, qualified as far as the schema allows,
    and whether its columns are left as written because qualify gave up.
    """
    # A qualified column the schema lacks makes qualify refuse the query; read
    # without the schema it is kept as written, so a check can say it is missing.
    # Without a schema, output aliases in HAVING or GROUP BY make qualify give up,
    # and the columns are then resolved as written.
    attempts = [{"schema": schema}] if schema else []
    attempts += [{"schema": None}, {"schema": None, "qualify_columns": False}]
    error: Exception | None = None
    for options in attempts:
        try:
            qualified = qualify(parsed.copy(), dialect=dialect, expand_stars=False,
                                validate_qualify_columns=False, quote_identifiers=False, **options)
            return traverse_scope(qualified), options.get("qualify_columns") is False
        except (OptimizeError, SqlglotError) as exc:
            error = exc
    raise OptimizeError(str(error))


def query_columns(sql: str, schema: Schema | None = None, dialect: str | None = None) -> QueryColumns:
    """
    The tables and columns ``sql`` reads, resolved against ``schema`` (table
    to columns), which is only needed for unqualified columns of a query
    level that reads several tables. Names take the schema's spelling;
    tables keep their schema prefix, so ``s1.t`` and ``s2.t`` stay apart.
    CTEs, subqueries and output aliases are not tables or columns.
    """
    schema = schema or {}
    found = QueryColumns()
    resolver = _Resolver(found, schema)
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except SqlglotError as exc:
        found.coverage, found.reason = NONE, f"The SQL does not parse: {str(exc).splitlines()[0]}"
        return found
    ctes = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    known = {}
    for table in parsed.find_all(exp.Table):
        name = resolver.table(table)
        if _is_table(table) and table.name.lower() not in ctes and schema.get(name):
            known[name] = {str(c): "TEXT" for c in schema[name]}
    try:
        scopes, as_written = _scopes(parsed, known, dialect)
    except OptimizeError as exc:
        found.coverage = NONE
        found.reason = f"The query could not be analysed: {str(exc).split('. Line:')[0]}"
        return found
    if as_written and (len(scopes) != 1 or len(scopes[0].sources) != 1):
        # Unqualified columns across several query levels or sources, e.g. a
        # correlated subquery, cannot be told apart without the tables' columns.
        found.coverage = NONE
        found.reason = "The query reads several tables or levels; their columns need a schema."
        for table in parsed.find_all(exp.Table):
            if _is_table(table) and table.name.lower() not in ctes:
                found.add(resolver.table(table))
        return found
    for pivot in parsed.find_all(exp.Pivot):
        if not _is_table(pivot.this):
            continue
        for part in (*pivot.expressions, *(pivot.args.get("using") or [])):
            for column in part.find_all(exp.Column):
                resolver.add(pivot.this, column.name)
        found.add(resolver.table(pivot.this))
        found.partly(f"It pivots {pivot.this.name}, whose output columns are not recorded.")
    for scope in scopes:
        for source in scope.sources.values():
            if _is_table(source):
                found.add(resolver.table(source))
                if source.args.get("pivots"):
                    resolver.opaque(source)
            elif not isinstance(source, Scope):
                resolver.opaque(source)
        outputs = {s.alias_or_name.lower() for s in getattr(scope.expression, "selects", [])}
        columns = list(scope.columns)
        # sqlglot leaves the columns of a QUALIFY window out of the scope.
        if isinstance(scope.expression, exp.Select) and (window := scope.expression.args.get("qualify")):
            seen = {id(c) for c in columns}
            columns += [c for c in window.find_all(exp.Column) if id(c) not in seen]
        for column in columns:
            resolver.resolve(scope, column, outputs)
        for alias in _projected_star(scope):
            source = scope.sources[alias]
            found.add(resolver.table(source))
            found.partly(f"It reads {source.name} with *.")
    if found.unresolved:
        found.coverage = NONE
        found.reason = f"Columns with no known table: {', '.join(sorted(found.unresolved))}."
    return found


@dataclass(frozen=True)
class ColumnProblem:
    """
    A table or column a contract records that a schema lacks or holds with
    another type. A ``warning`` does not break the contract, e.g. a column
    whose type the source cannot tell.
    """

    table: str
    column: str | None
    message: str
    warning: bool = False


def schema_problems(expected: Schema, schema: Schema) -> list[ColumnProblem]:
    """
    What ``schema`` (table to column to family) lacks of ``expected``, the
    same shape as recorded: missing tables, missing columns and columns
    whose family is not ``compatible``. Names match case-insensitively, and
    a table also by its last name part.
    """
    problems = []
    unreadable = getattr(schema, "unreadable", {})
    for table, columns in expected.items():
        actual_table = _find_table(table, schema)
        if actual_table is None:
            message = (
                f"Table {table!r} could not be read: {unreadable[table]}" if table in unreadable
                else f"The data has no table {table!r}."
            )
            problems.append(ColumnProblem(table, None, message))
            continue
        actual = {c.lower(): kind for c, kind in schema[actual_table].items()}
        for column, kind in columns.items():
            if column.lower() not in actual:
                problems.append(ColumnProblem(table, column, f"Table {table!r} has no column {column!r}."))
            elif actual[column.lower()] == "other" and kind != "other":
                problems.append(ColumnProblem(table, column, (
                    f"The type of {table}.{column} cannot be told; the contract expects {kind}."
                ), warning=True))
            elif not compatible(kind, actual[column.lower()]):
                problems.append(ColumnProblem(table, column, (
                    f"Column {table}.{column} is {actual[column.lower()]} now; the contract expects {kind}."
                )))
    return problems


__all__ = [
    "ColumnProblem",
    "QueryColumns",
    "query_columns",
    "schema_problems",
    "source_schema",
]
