"""
Column statistics for the table context shown to the LLM.

`Source.get_schema` describes a table for widgets and filters. The LLM needs
something different: SQL types, keys, true value ranges and literal values,
and it needs them to be byte-identical across runs so providers can cache the
prompt. Computing that must also stay affordable on warehouse tables, so each
table is profiled with the cheapest strategy that is still accurate:

1. Metadata the engine already maintains (``pg_stats``, Parquet footers,
   Snowflake micro-partition metadata, Postgres and SQLite row estimates),
   which is free.
2. One exact aggregate pass when the table has at most ``exact_row_limit`` rows.
3. A seeded engine-side sample of ``sample_rows`` rows above that.

Low-cardinality columns found in step 2 or 3 get exact value counts when the
engine can afford a ``GROUP BY``. Every result records how it was computed so
the prompt can say ``stats from a 10000-row sample`` rather than implying
that a sampled range is exact.

Results are cached in memory and, for sources with a stable identity, on disk,
keyed by the table definition and validated against its column types and a
modification token where the engine exposes one. Persisted entries contain
real data values (ranges, frequent values, examples); set
``LUMEN_TABLE_STATS_CACHE`` to an empty string to keep them in memory only.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import decimal
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
import weakref

from collections import OrderedDict
from concurrent.futures import (
    ThreadPoolExecutor, TimeoutError as FutureTimeout,
)
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import numpy as np
import pandas as pd
import param
import sqlglot

from platformdirs import user_cache_dir
from sqlglot import exp
from sqlglot.dialects import Dialect

from .utils import log_debug

if TYPE_CHECKING:
    from ..sources.base import Source

#: Bumped whenever the computed statistics change meaning, which invalidates
#: every persisted entry.
STATS_VERSION = 2

EXACT = "exact"
ESTIMATED = "estimated"
SAMPLED = "sampled"
TYPES_ONLY = "types"
#: Adapted from a `Source.get_schema` dict whose provenance is unknown.
SCHEMA = "schema"

_NUMERIC_RE = re.compile(
    r"INT|DEC|NUMERIC|NUMBER|REAL|FLOAT|DOUBLE|MONEY|SERIAL|BIGNUM", re.IGNORECASE
)
_TEMPORAL_RE = re.compile(r"DATE|TIME", re.IGNORECASE)
_BOOLEAN_RE = re.compile(r"^BOOL", re.IGNORECASE)
_STRING_RE = re.compile(r"CHAR|TEXT|STRING|CLOB|UUID|ENUM|NAME|VARCHAR", re.IGNORECASE)
_OTHER_RE = re.compile(
    r"BLOB|BINARY|BYTEA|BYTES|STRUCT|MAP|LIST|ARRAY|\[\]|JSON|GEOMETRY|GEOGRAPHY|VARIANT|OBJECT|INTERVAL",
    re.IGNORECASE,
)
# Hyphens are allowed because BigQuery project IDs contain them.
_BARE_NAME_RE = re.compile(r'^\s*(?:"[^"]+"|`[^`]+`|[A-Za-z_][\w$-]*)(?:\s*\.\s*(?:"[^"]+"|`[^`]+`|[A-Za-z_][\w$-]*)){0,2}\s*$')
_PLAIN_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_SELECT_STAR_RE = re.compile(r'^\s*SELECT\s+\*\s+FROM\s+(.+?)\s*;?\s*$', re.IGNORECASE | re.DOTALL)
_WHOLE_PARQUET_RE = re.compile(
    r"""^\s*(?:SELECT\s+\*\s+FROM\s+)?read_parquet\s*\(\s*'([^']+)'\s*\)\s*;?\s*$""", re.IGNORECASE
)
_FILE_ARG_RE = re.compile(r"""read_\w+\s*\(\s*\[?\s*['"]([^'"]+)['"]""", re.IGNORECASE)


@dataclass
class ColumnStats:
    """Statistics for one column, JSON-serializable."""

    name: str
    type: str | None = None
    kind: str = "other"
    nulls: float | None = None
    min: Any = None
    max: Any = None
    distinct: int | None = None
    values: list[list[Any]] | None = None
    values_complete: bool = False
    examples: list[Any] | None = None
    primary_key: bool = False
    references: str | None = None


@dataclass
class TableStats:
    """Statistics for one table and how they were obtained."""

    table: str
    columns: list[ColumnStats] = field(default_factory=list)
    rows: int | None = None
    rows_exact: bool = False
    #: Lower bound when the row count is unknown but exceeds a limit.
    rows_at_least: int | None = None
    method: str = TYPES_ONLY
    sample_size: int | None = None
    #: False when the "sample" is the first rows, which may be sorted.
    sample_random: bool = True
    ranges_exact: bool = False
    #: Hash of the column names and types as the engine described them.
    signature: str | None = None
    fingerprint: str | None = None
    computed_at: float = 0.0
    version: int = STATS_VERSION

    def column(self, name: str) -> ColumnStats | None:
        return next((col for col in self.columns if col.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableStats:
        data = dict(data)
        data["columns"] = [ColumnStats(**col) for col in data.get("columns", [])]
        return cls(**data)

    @classmethod
    def from_json_schema(cls, table: str, schema: dict[str, Any]) -> TableStats:
        """
        Adapt a `Source.get_schema`-style dict, as older callers still set on
        `Metaset.schemas`, so every table renders through the same path.
        """
        columns = []
        for name, spec in schema.items():
            if name == "__len__":
                continue
            col = ColumnStats(name=name)
            if not isinstance(spec, dict):
                col.nulls = 1.0 if spec == "<null>" else None
                columns.append(col)
                continue
            json_type = spec.get("type")
            if json_type in ("integer", "int"):
                col.type, col.kind = "INTEGER", "numeric"
            elif json_type in ("number", "num"):
                col.type, col.kind = "DOUBLE", "numeric"
            elif json_type in ("boolean", "bool"):
                col.type, col.kind = "BOOLEAN", "boolean"
            elif spec.get("format") in ("datetime", "date", "date-time"):
                col.type, col.kind = "TIMESTAMP", "temporal"
            elif json_type in ("string", "str", "enum") or "enum" in spec:
                col.type, col.kind = "VARCHAR", "string"
            lo = spec.get("min", spec.get("inclusiveMinimum"))
            hi = spec.get("max", spec.get("inclusiveMaximum"))
            if lo is not None or hi is not None:
                col.min, col.max = _to_python(lo), _to_python(hi)
                if col.kind == "other":
                    col.kind = "numeric"
            if enum := spec.get("enum"):
                values = [_to_python(v) for v in enum if v != "..."]
                col.values = [[v, None] for v in values if v is not None]
                col.values_complete = "..." not in enum
            columns.append(col)
        rows = schema.get("__len__")
        return cls(
            table=table, columns=columns, rows=None if rows is None else int(rows),
            rows_exact=rows is not None, method=SCHEMA,
        )


class StatsBudgetExceeded(Exception):
    """Raised when a source has spent its statistics budget."""


class StatementTimeout(Exception):
    """Raised when one statistics query exceeds its statement timeout."""


def column_kind(sql_type: str | None) -> str:
    """Classify a SQL type into the families the renderer distinguishes."""
    if not sql_type:
        return "other"
    if _OTHER_RE.search(sql_type):
        return "other"
    if _BOOLEAN_RE.search(sql_type):
        return "boolean"
    if _TEMPORAL_RE.search(sql_type) and not _STRING_RE.search(sql_type):
        return "temporal"
    if _NUMERIC_RE.search(sql_type):
        return "numeric"
    if _STRING_RE.search(sql_type):
        return "string"
    return "other"


def _kind_from_series(series: pd.Series) -> tuple[str, str]:
    """A SQL type guessed from fetched values, for engines that did not report one."""
    series = series.infer_objects()
    kind = getattr(series.dtype, "kind", "O")
    if kind == "O":
        # Drivers return DATE and DECIMAL as Python objects, which would
        # otherwise read as text and lose their ranges.
        sample = series.dropna()
        first = sample.iloc[0] if len(sample) else None
        if isinstance(first, dt.datetime):
            return "TIMESTAMP", "temporal"
        if isinstance(first, dt.date):
            return "DATE", "temporal"
        if isinstance(first, decimal.Decimal):
            return "DECIMAL", "numeric"
    if kind == "b":
        return "BOOLEAN", "boolean"
    if kind in "iu":
        return "INTEGER", "numeric"
    if kind == "f":
        return "DOUBLE", "numeric"
    if kind == "M":
        return "TIMESTAMP", "temporal"
    return "VARCHAR", "string"


def _to_python(value: Any) -> Any:
    """Convert a scalar to something JSON can round-trip without loss of meaning."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, decimal.Decimal):
        if value.is_nan():
            return None
        as_float = float(value)
        return int(value) if value == value.to_integral_value() else as_float
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        value = value.to_pydatetime()
    if isinstance(value, dt.datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, bytes | bytearray | memoryview):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def quote_identifier(name: str, dialect: str | None) -> str:
    if dialect in ("mysql", "mariadb", "bigquery", "databricks", "spark", "hive"):
        return "`" + name.replace("`", "``") + "`"
    if dialect in ("mssql", "tsql"):
        return "[" + name.replace("]", "]]") + "]"
    return '"' + name.replace('"', '""') + '"'


def _reserved_words() -> frozenset[str]:
    # sqlglot only lists reserved words for some dialects; their union is a
    # superset, and quoting a non-reserved name is harmless once case-folded.
    words: set[str] = set()
    for name in ("duckdb", "bigquery", "mysql", "postgres", "snowflake", "tsql", "oracle"):
        try:
            words |= {w.lower() for w in Dialect.get_or_raise(name).generator_class.RESERVED_KEYWORDS}
        except Exception:
            continue
    return frozenset(words | {"order", "group", "user", "table", "select", "from", "where", "limit"})


_RESERVED = _reserved_words()


def _quote_part(part: str, dialect: str | None) -> str:
    """
    Quote one name part only when it needs it. Quoting makes a name
    case-sensitive, so plain names stay unquoted to keep the engine's own
    case folding, and reserved words are folded the way the engine would
    have folded them unquoted.
    """
    if part[0] in '"`[':
        return part
    if _PLAIN_IDENT_RE.match(part) and part.lower() not in _RESERVED:
        return part
    if _PLAIN_IDENT_RE.match(part):
        if dialect in ("postgresql", "postgres", "redshift"):
            part = part.lower()
        elif dialect in ("snowflake", "oracle"):
            part = part.upper()
    return quote_identifier(part, dialect)


def _quote_qualified(name: str, dialect: str | None) -> str:
    parts = re.findall(r'"[^"]+"|`[^`]+`|\[[^\]]+\]|[^.\s]+', name)
    return ".".join(_quote_part(part, dialect) for part in parts)


def _stat_token(paths: list[str], journals: tuple[str, ...] = ()) -> str | None:
    """
    Size and mtime of `paths`, plus each path's write-ahead log (suffixes in
    `journals`) when it exists: in WAL mode a commit, UPDATE included, only
    touches the log until the next checkpoint.
    """
    tokens = []
    for path in paths:
        try:
            stat = os.stat(path)
        except OSError:
            return None
        tokens.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        for suffix in journals:
            try:
                stat = os.stat(path + suffix)
            except OSError:
                continue
            tokens.append(f"{suffix}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(tokens)


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------

class StatsAdapter:
    """
    Engine-specific access used by the profiler. The base class only relies
    on `Source.execute` and portable SQL, so any SQL source is supported;
    subclasses add free metadata, keys, seeded sampling and cancellation.
    """

    #: Whether an unfiltered COUNT(*) is answered from metadata.
    metadata_count = False

    #: Whether exact GROUP BY value counts are affordable on large tables.
    local = False

    #: Whether reading the first rows bills a full scan (BigQuery), so it is
    #: never used as a fallback for a failed or empty sample.
    limit_scans_table = False

    #: Whether only the thread that created the data can see it, so the
    #: table cannot be profiled on a worker thread.
    thread_bound = False

    #: Whether min/max from `metadata_stats` are exact rather than estimates.
    exact_metadata_ranges = False

    def __init__(self, source: Source):
        self.source = source
        self.dialect = getattr(source, "dialect", None)

    # Identity ---------------------------------------------------------------

    def identity(self) -> str | None:
        """A string identifying the database, or None when not stable across processes."""
        return None

    def definition(self, table: str) -> str:
        tables = getattr(self.source, "tables", None)
        # Non-SQL sources may map names to DataFrames, whose repr is no definition.
        if isinstance(tables, dict) and isinstance(tables.get(table), str):
            return tables[table]
        try:
            return self.source.get_sql_expr(table)
        except Exception:
            return table

    def modified(self, table: str) -> str | None:
        """A token that changes when the table's data changes, if cheaply known."""
        paths = self._file_paths(table)
        return _stat_token(paths) if paths else None

    def _file_paths(self, table: str) -> list[str]:
        file_tables = getattr(self.source, "_file_based_tables", {}) or {}
        candidates = [str(file_tables[table])] if table in file_tables else []
        candidates += _FILE_ARG_RE.findall(self.definition(table))
        paths = []
        for candidate in candidates:
            if "://" in candidate or any(ch in candidate for ch in "*?["):
                continue
            path = Path(candidate).expanduser()
            if path.exists():
                paths.append(str(path.resolve()))
        return sorted(set(paths))

    # Relation ----------------------------------------------------------------

    def bare_name(self, table: str) -> str | None:
        """The physical table name if `table` is not defined by an expression."""
        tables = getattr(self.source, "tables", None)
        definition = tables[table] if isinstance(tables, dict) and table in tables else table
        if not isinstance(definition, str):
            return None
        if match := _SELECT_STAR_RE.match(definition):
            definition = match.group(1)
        if _BARE_NAME_RE.match(definition):
            return definition.strip()
        return None

    def relation(self, table: str) -> str:
        """A FROM-clause item for `table`."""
        if name := self.bare_name(table):
            return _quote_qualified(name, self.dialect)
        return f"({self.source.get_sql_expr(table)}) {self.alias('lumen_t')}"

    def params(self, table: str):
        return (getattr(self.source, "table_params", None) or {}).get(table) or None

    # Dialect -------------------------------------------------------------------

    def limit(self, sql: str, rows: int) -> str:
        """`sql` (a plain SELECT) restricted to its first `rows` rows."""
        if self.dialect in ("mssql", "tsql"):
            return re.sub(r"^\s*SELECT\s", f"SELECT TOP {rows} ", sql, count=1, flags=re.IGNORECASE)
        if self.dialect == "oracle":
            return f"{sql} FETCH FIRST {rows} ROWS ONLY"
        return f"{sql} LIMIT {rows}"

    def alias(self, name: str) -> str:
        """A subquery alias; Oracle rejects ``AS`` there."""
        return name if self.dialect == "oracle" else f"AS {name}"

    # Execution -----------------------------------------------------------------

    def execute(self, sql: str, params=None, timeout: float | None = None) -> pd.DataFrame:
        return _run_abandonable(lambda: self.source.execute(sql, params), timeout)

    # Metadata ------------------------------------------------------------------

    def columns(self, table: str, relation: str, timeout: float | None) -> list[tuple[str, str | None]]:
        df = self.execute(self.limit(f"SELECT * FROM {relation}", 0), self.params(table), timeout)
        return [(str(name), None) for name in df.columns] if df is not None else []

    def keys(self, table: str) -> tuple[set[str], dict[str, str]]:
        return set(), {}

    def exact_row_count(self, table: str) -> int | None:
        """An exact row count from metadata, e.g. Parquet footers."""
        return None

    def row_estimate(self, table: str) -> int | None:
        """A row count the engine already knows without scanning."""
        return None

    def metadata_stats(self, table: str, columns: list[ColumnStats], rows: int | None, timeout: float | None) -> bool:
        """Fill min/max/nulls (and more) from engine metadata; return True on success."""
        return False

    def sample_sql(self, relation: str, rows: int, total: int | None, seed: int) -> tuple[str | None, bool]:
        """
        A query returning roughly `rows` rows and whether it is a random
        sample, or None when no affordable sample exists.
        """
        return self.limit(f"SELECT * FROM {relation}", rows), False



def _percent(rows: int, total: int | None) -> float:
    if not total:
        return 100.0
    return min(100.0, max(0.001, 100.0 * rows * 1.5 / total))


class DuckDBStatsAdapter(StatsAdapter):

    local = True

    # Parquet footers bound every row group exactly.
    exact_metadata_ranges = True

    def identity(self) -> str | None:
        uri = getattr(self.source, "uri", None)
        if uri and uri != ":memory:" and "://" not in uri:
            return f"duckdb:{Path(uri).expanduser().resolve()}"
        return None

    def modified(self, table: str) -> str | None:
        # A table or view's oid changes on CREATE OR REPLACE and its
        # estimated size on inserts; neither changes on UPDATE, which only
        # the database file and its WAL reveal. In-memory databases have no
        # file, so an UPDATE there is not seen while the connection lives.
        tokens = [token] if (token := super().modified(table)) else []
        if identity := self.identity():
            if (db := _stat_token([identity.removeprefix("duckdb:")], (".wal",))) is None:
                return None
            tokens.append(db)
        if names := self._referenced_tables(table):
            placeholders = ", ".join("?" for _ in names)
            try:
                df = self.execute(
                    "SELECT lower(table_name) AS name, table_oid AS oid, estimated_size AS size "
                    f"FROM duckdb_tables() WHERE lower(table_name) IN ({placeholders}) "
                    "UNION ALL SELECT lower(view_name), view_oid, NULL FROM duckdb_views() "
                    f"WHERE NOT internal AND lower(view_name) IN ({placeholders}) ORDER BY 1, 2",
                    [*names, *names], 5,
                )
            except Exception:
                return None
            tokens.append(";".join(f"{name}:{oid}:{size}" for name, oid, size in df.itertuples(index=False)))
        return "|".join(tokens) or None

    def _referenced_tables(self, table: str) -> list[str]:
        if name := self.bare_name(table):
            return [name.split(".")[-1].strip('"`').lower()]
        try:
            tree = sqlglot.parse_one(self.definition(table), read="duckdb")
        except Exception:
            return []
        ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        names = {t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes}
        return sorted(names)

    def execute(self, sql: str, params=None, timeout: float | None = None) -> pd.DataFrame:
        # A cursor per statement so interrupt() only stops this one.
        cursor = self.source._connection.cursor()

        def run():
            return (cursor.execute(sql, params) if params else cursor.execute(sql)).fetch_df()

        try:
            return _run_interruptible(run, timeout, cursor.interrupt)
        finally:
            cursor.close()

    def columns(self, table, relation, timeout):
        df = self.execute(f"DESCRIBE SELECT * FROM {relation}", self.params(table), timeout)
        return list(zip(df["column_name"].astype(str), df["column_type"].astype(str), strict=False))

    def keys(self, table):
        name = self.bare_name(table)
        if not name:
            return set(), {}
        table_name = name.split(".")[-1].strip('"')
        try:
            df = self.execute(
                "SELECT constraint_type, constraint_column_names, referenced_table, "
                "referenced_column_names FROM duckdb_constraints() WHERE table_name = ?",
                [table_name], 5,
            )
        except Exception:
            return set(), {}
        pk, fks = set(), {}
        for row in df.itertuples(index=False):
            cols = list(row.constraint_column_names)
            if row.constraint_type == "PRIMARY KEY" and len(cols) == 1:
                pk.add(cols[0])
            elif row.constraint_type == "FOREIGN KEY":
                refs = list(row.referenced_column_names)
                for col, ref in zip(cols, refs, strict=False):
                    fks[col] = f"{row.referenced_table}.{ref}"
        return pk, fks

    def exact_row_count(self, table):
        if path := self._parquet_path(table):
            try:
                df = self.execute(f"SELECT SUM(num_rows) AS n FROM parquet_file_metadata('{path}')", None, 10)
                return int(df["n"].iloc[0])
            except Exception:
                return None
        return None

    def _parquet_path(self, table) -> str | None:
        """
        The Parquet file `table` reads in full. Footer counts and ranges
        describe the whole file, so any filter, join or projection on top of
        it disqualifies the table.
        """
        file_tables = getattr(self.source, "_file_based_tables", {}) or {}
        if table in file_tables:
            candidate = str(file_tables[table])
        elif match := _WHOLE_PARQUET_RE.match(self.definition(table)):
            candidate = match.group(1)
        else:
            return None
        if not candidate.lower().endswith((".parquet", ".parq")) or "://" in candidate or any(ch in candidate for ch in "*?["):
            return None
        path = Path(candidate).expanduser()
        return str(path.resolve()).replace("'", "''") if path.exists() else None

    def metadata_stats(self, table, columns, rows, timeout):
        path = self._parquet_path(table)
        if not path:
            return False
        try:
            df = self.execute(
                "SELECT path_in_schema AS col, SUM(stats_null_count) AS nulls, "
                "LIST(stats_min_value) AS mins, LIST(stats_max_value) AS maxs "
                f"FROM parquet_metadata('{path}') GROUP BY path_in_schema",
                None, timeout,
            )
        except Exception:
            return False
        by_name = {str(row.col): row for row in df.itertuples(index=False)}
        found = False
        for col in columns:
            row = by_name.get(col.name)
            if row is None:
                continue
            found = True
            if rows and row.nulls is not None and not pd.isna(row.nulls):
                col.nulls = float(row.nulls) / rows
            if col.kind in ("numeric", "temporal"):
                caster = _caster(col)
                mins = [caster(v) for v in row.mins if v is not None]
                maxs = [caster(v) for v in row.maxs if v is not None]
                mins = [v for v in mins if v is not None]
                maxs = [v for v in maxs if v is not None]
                if mins and maxs:
                    col.min, col.max = _to_python(min(mins)), _to_python(max(maxs))
        return found

    #: Below this many rows a reservoir sample (one full scan) is affordable;
    #: system sampling picks whole 2048-row vectors and on smaller tables
    #: often returns nothing at all.
    reservoir_row_limit = 50_000_000

    def sample_sql(self, relation, rows, total, seed):
        if total is None or total <= rows:
            return super().sample_sql(relation, rows, total, seed)
        if total <= self.reservoir_row_limit:
            return (
                f"SELECT * FROM {relation} USING SAMPLE reservoir({rows} ROWS) REPEATABLE ({seed})"
            ), True
        pct = _percent(rows, total)
        return (
            f"SELECT * FROM (SELECT * FROM {relation} USING SAMPLE {pct:.4f} PERCENT (system, {seed})) "
            f"AS lumen_s LIMIT {rows}"
        ), True



def _caster(col: ColumnStats):
    if col.kind == "numeric":
        def cast(v):
            try:
                return float(v) if not re.fullmatch(r"-?\d+", str(v)) else int(v)
            except (TypeError, ValueError):
                return None
        return cast
    def cast_time(v):
        try:
            return pd.Timestamp(v)
        except (TypeError, ValueError):
            return None
    return cast_time


class SQLAlchemyStatsAdapter(StatsAdapter):

    def __init__(self, source):
        super().__init__(source)
        self.local = self.dialect == "sqlite"
        # SingletonThreadPool, SQLAlchemy's default for SQLite :memory:,
        # gives every thread its own empty database.
        pool = getattr(getattr(source, "_engine", None), "pool", None)
        self.thread_bound = (
            self.dialect == "sqlite" and type(pool).__name__ == "SingletonThreadPool"
            and (getattr(getattr(source, "_url", None), "database", None) or ":memory:") == ":memory:"
        )

    def identity(self) -> str | None:
        url = getattr(self.source, "_url", None)
        if url is None:
            return None
        if url.get_backend_name() == "sqlite":
            path = self._sqlite_path()
            return f"sqlite:{path}" if path else None
        return url.render_as_string(hide_password=True)

    def _sqlite_path(self) -> str | None:
        database = self.source._url.database or ""
        if database in ("", ":memory:"):
            return None
        database = database.split("?", 1)[0].removeprefix("file:")
        path = Path(unquote(database)).expanduser()
        return str(path.resolve()) if path.exists() else None

    def modified(self, table):
        if self.dialect == "sqlite" and (path := self._sqlite_path()):
            return _stat_token([path], ("-wal",))
        return super().modified(table)

    def _split(self, table: str) -> tuple[str | None, str] | None:
        name = self.bare_name(table)
        if not name:
            return None
        parts = [p.strip('"`') for p in re.findall(r'"[^"]+"|`[^`]+`|[^.\s]+', name)]
        if len(parts) == 1:
            return getattr(self.source, "schema", None), parts[0]
        return parts[-2], parts[-1]

    def columns(self, table, relation, timeout):
        split = self._split(table)
        inspector = getattr(self.source, "_inspector", None)
        if split and inspector is not None:
            try:
                cols = inspector.get_columns(split[1], schema=split[0])
                if cols:
                    return [(col["name"], str(col["type"]) or None) for col in cols]
            except Exception as e:
                log_debug(f"[table_stats] inspector failed for {table!r}: {e}")
        return super().columns(table, relation, timeout)

    def keys(self, table):
        split = self._split(table)
        inspector = getattr(self.source, "_inspector", None)
        if not split or inspector is None:
            return set(), {}
        pk, fks = set(), {}
        try:
            constraint = inspector.get_pk_constraint(split[1], schema=split[0]) or {}
            cols = constraint.get("constrained_columns") or []
            if len(cols) == 1:
                pk.add(cols[0])
            for fk in inspector.get_foreign_keys(split[1], schema=split[0]) or []:
                for col, ref in zip(fk.get("constrained_columns") or [], fk.get("referred_columns") or [], strict=False):
                    fks[col] = f"{fk.get('referred_table')}.{ref}"
        except Exception as e:
            log_debug(f"[table_stats] key inspection failed for {table!r}: {e}")
        return pk, fks

    def execute(self, sql, params=None, timeout=None):
        if self.dialect == "sqlite" and not getattr(self.source, "_driver_is_async", False):
            return self._execute_sqlite(sql, params, timeout)
        if self.dialect in ("postgresql", "postgres") and timeout and not getattr(self.source, "_driver_is_async", False):
            return self._execute_postgres(sql, params, timeout)
        return super().execute(sql, params, timeout)

    def _execute_sqlite(self, sql, params, timeout):
        # SQLite has no statement timeout, but a progress handler can abort
        # the statement from inside the engine rather than abandoning a thread.
        raw = self.source._engine.raw_connection()
        try:
            conn = getattr(raw, "driver_connection", None) or raw.connection
            deadline = time.monotonic() + timeout if timeout else None
            if deadline is not None:
                conn.set_progress_handler(lambda: time.monotonic() > deadline, 10000)
            try:
                cursor = conn.execute(sql, params or [])
                names = [d[0] for d in cursor.description or []]
                return pd.DataFrame(cursor.fetchall(), columns=names)
            except Exception as e:
                if deadline is not None and time.monotonic() > deadline:
                    raise StatementTimeout(f"statement exceeded {timeout:g}s") from e
                raise
            finally:
                if deadline is not None:
                    conn.set_progress_handler(None, 0)
        finally:
            raw.close()

    def _execute_postgres(self, sql, params, timeout):
        from sqlalchemy import text

        # Without bind parameters every colon is literal, e.g. inside a
        # quoted identifier, and must not be parsed as one.
        stmt = text(sql) if params else text(sql.replace(":", "\\:"))
        start = time.monotonic()
        try:
            # SET LOCAL ends with the transaction, so nothing has to run
            # after a timeout has aborted it.
            with self.source._engine.connect() as conn, conn.begin():
                conn.exec_driver_sql(f"SET LOCAL statement_timeout = {max(1, int(timeout * 1000))}")
                result = conn.execute(stmt, params or {})
                return pd.DataFrame(result.fetchall(), columns=list(result.keys()))
        except Exception as e:
            if _is_pg_timeout(e) or time.monotonic() - start >= timeout:
                raise StatementTimeout(f"statement exceeded {timeout:g}s") from e
            raise

    def row_estimate(self, table):
        split = self._split(table)
        if not split:
            return None
        schema, name = split
        try:
            if self.dialect == "sqlite":
                df = self.execute("SELECT stat FROM sqlite_stat1 WHERE tbl = ? LIMIT 1", [name], 5)
                if len(df):
                    return int(str(df["stat"].iloc[0]).split()[0])
            elif self.dialect in ("postgresql", "postgres"):
                qualified = f"{schema}.{name}" if schema else name
                df = self.execute(
                    "SELECT reltuples::bigint AS n FROM pg_class WHERE oid = to_regclass(:name)",
                    {"name": qualified}, 5,
                )
                if len(df) and df["n"].iloc[0] is not None and int(df["n"].iloc[0]) >= 0:
                    return int(df["n"].iloc[0])
        except Exception:
            return None
        return None

    def metadata_stats(self, table, columns, rows, timeout):
        if self.dialect not in ("postgresql", "postgres"):
            return False
        split = self._split(table)
        if not split:
            return False
        schema, name = split
        try:
            df = self.execute(
                "SELECT attname, null_frac, n_distinct, most_common_vals::text AS mcv, "
                "most_common_freqs::text AS mcf, histogram_bounds::text AS hist "
                "FROM pg_stats WHERE tablename = :name AND schemaname = COALESCE(:schema, current_schema())",
                {"name": name, "schema": schema}, timeout,
            )
        except Exception:
            return False
        if df.empty:
            return False
        by_name = {row.attname: row for row in df.itertuples(index=False)}
        for col in columns:
            row = by_name.get(col.name)
            if row is None:
                continue
            col.nulls = float(row.null_frac)
            n_distinct = float(row.n_distinct)
            if n_distinct < 0 and rows:
                col.distinct = round(-n_distinct * rows)
            elif n_distinct > 0:
                col.distinct = int(n_distinct)
            mcv = _parse_pg_array(row.mcv)
            freqs = [float(f) for f in _parse_pg_array(row.mcf)]
            caster = _caster(col) if col.kind in ("numeric", "temporal") else (lambda v: v)
            if mcv and len(mcv) == len(freqs):
                col.values = [[_to_python(caster(v)), f] for v, f in zip(mcv, freqs, strict=False)]
                col.values_complete = bool(col.distinct and len(mcv) >= col.distinct)
            hist = _parse_pg_array(row.hist)
            if col.kind in ("numeric", "temporal"):
                bounds = [caster(v) for v in hist + mcv]
                bounds = [b for b in bounds if b is not None]
                if bounds:
                    col.min, col.max = _to_python(min(bounds)), _to_python(max(bounds))
        return True

    def sample_sql(self, relation, rows, total, seed):
        if total is None or total <= rows:
            return super().sample_sql(relation, rows, total, seed)
        if self.dialect in ("postgresql", "postgres") and not relation.startswith("("):
            pct = _percent(rows, total)
            return (
                f"SELECT * FROM {relation} TABLESAMPLE SYSTEM ({pct:.4f}) REPEATABLE ({seed}) LIMIT {rows}"
            ), True
        if self.dialect == "sqlite" and not relation.startswith("("):
            # Deterministic hash of rowid rather than ORDER BY RANDOM(), which
            # sorts the whole table and differs on every run.
            modulus = max(1, math.ceil(total / rows))
            return (
                f"SELECT * FROM {relation} WHERE ((rowid * 2654435761 + {seed}) % {modulus * 7919}) < 7919 "
                f"LIMIT {rows}"
            ), True
        return super().sample_sql(relation, rows, total, seed)


def _parse_pg_array(text: str | None) -> list[str]:
    """Parse the text form of a Postgres array, e.g. ``{a,"b c",NULL}``."""
    if not isinstance(text, str) or text in ("", "{}"):
        return []
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    values, current, quoted, escape, was_quoted = [], [], False, False, False
    for ch in text:
        if escape:
            current.append(ch)
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            quoted = not quoted
            was_quoted = True
        elif ch == "," and not quoted:
            value = "".join(current)
            values.append(None if value == "NULL" and not was_quoted else value)
            current, was_quoted = [], False
        else:
            current.append(ch)
    value = "".join(current)
    values.append(None if value == "NULL" and not was_quoted else value)
    return [v for v in values if v is not None]


#: Snowflake's internal type names, as cursor metadata reports them, mapped
#: to SQL type names the profiler classifies.
_SNOWFLAKE_TYPES = {"FIXED": "NUMBER", "REAL": "FLOAT", "TEXT": "VARCHAR"}


class SnowflakeStatsAdapter(StatsAdapter):

    # Snowflake answers COUNT(*), COUNT(col), MIN and MAX on base tables from
    # micro-partition metadata, so these never consume warehouse time.
    metadata_count = True

    exact_metadata_ranges = True

    def columns(self, table, relation, timeout):
        # A LIMIT 0 result carries names only, and without types no column
        # would get a range. describe() compiles the query without running it.
        from snowflake.connector.constants import (  # type: ignore[import-not-found]
            FIELD_ID_TO_NAME,
        )
        cursor = self.source._conn.cursor()
        try:
            meta = cursor.describe(f"SELECT * FROM {relation}")
        except Exception as e:
            log_debug(f"[table_stats] describe failed for {table!r}: {e}")
            return super().columns(table, relation, timeout)
        finally:
            cursor.close()
        columns = []
        for col in meta:
            name = FIELD_ID_TO_NAME.get(col.type_code)
            columns.append((col.name, _SNOWFLAKE_TYPES.get(name, name)))
        return columns

    def identity(self):
        src = self.source
        if not getattr(src, "account", None):
            return None
        # Stats computed under one role must not be served to another.
        conn = getattr(src, "_conn", None)
        conn_kwargs = getattr(src, "conn_kwargs", None) or {}
        user = getattr(src, "user", None) or getattr(conn, "user", None)
        role = getattr(conn, "role", None) or conn_kwargs.get("role")
        parts = [src.account, user, role, getattr(src, "database", None), getattr(src, "schema", None)]
        return "snowflake:" + "/".join(str(p) for p in parts)

    def execute(self, sql, params=None, timeout=None):
        # The source's shared cursor would interleave with user queries, and
        # a cursor-level timeout cancels the query server-side.
        cursor = self.source._conn.cursor()
        kwargs = {"timeout": max(1, math.ceil(timeout))} if timeout else {}
        start = time.monotonic()
        try:
            if params:
                cursor.execute(sql, params, **kwargs)
            else:
                cursor.execute(sql, **kwargs)
            return cursor.fetch_pandas_all()
        except Exception as e:
            if timeout and time.monotonic() - start >= timeout:
                raise StatementTimeout(f"statement exceeded {timeout:g}s") from e
            raise
        finally:
            cursor.close()

    def metadata_stats(self, table, columns, rows, timeout):
        if not self.bare_name(table):
            return False
        relation = self.relation(table)
        exprs = []
        for i, col in enumerate(columns):
            q = quote_identifier(col.name, self.dialect)
            exprs.append(f"COUNT({q}) AS N{i}")
            if col.kind in ("numeric", "temporal"):
                exprs += [f"MIN({q}) AS LO{i}", f"MAX({q}) AS HI{i}"]
        try:
            df = self.execute(f"SELECT COUNT(*) AS N, {', '.join(exprs)} FROM {relation}", None, timeout)
        except Exception:
            return False
        row = {str(k).upper(): df[k].iloc[0] for k in df.columns}
        total = int(row["N"])
        for i, col in enumerate(columns):
            if total:
                col.nulls = 1 - int(row[f"N{i}"]) / total
            if col.kind in ("numeric", "temporal"):
                col.min, col.max = _to_python(row.get(f"LO{i}")), _to_python(row.get(f"HI{i}"))
        return True

    def sample_sql(self, relation, rows, total, seed):
        if total is None or total <= rows or relation.startswith("("):
            return super().sample_sql(relation, rows, total, seed)
        pct = _percent(rows, total)
        return f"SELECT * FROM {relation} SAMPLE SYSTEM ({pct:.4f}) SEED ({seed}) LIMIT {rows}", True



class BigQueryStatsAdapter(StatsAdapter):

    # An unfiltered COUNT(*) on a BigQuery table bills zero bytes.
    metadata_count = True

    def __init__(self, source, max_bytes_billed: int | None = None):
        super().__init__(source)
        self.max_bytes_billed = max_bytes_billed

    # LIMIT does not reduce the bytes BigQuery bills.
    limit_scans_table = True

    def identity(self):
        project = getattr(self.source, "project_id", None)
        if not project:
            return None
        principal = getattr(getattr(self.source, "_credentials", None), "service_account_email", None)
        return f"bigquery:{project}" + (f"/{principal}" if principal else "")

    def _table(self, table: str):
        """
        The BigQuery table object for a bare table name. `get_metadata` would
        list every dataset and table to answer the same question.
        """
        if not (name := self.bare_name(table)):
            return None
        try:
            return self.source._get_table(name.replace("`", ""))
        except Exception:
            return None

    def modified(self, table):
        bq_table = self._table(table)
        return str(bq_table.modified) if bq_table is not None and bq_table.modified else None

    def columns(self, table, relation, timeout):
        from google.cloud import bigquery  # type: ignore[import-not-found]

        # A LIMIT 0 result carries no SQL types. Tables have a schema; any
        # other expression gets one from a dry run, which bills nothing.
        if (bq_table := self._table(table)) is not None and bq_table.schema:
            fields = bq_table.schema
        else:
            try:
                job = self.source._sql_client.query(
                    f"SELECT * FROM {relation}",
                    job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
                )
                fields = job.schema
            except Exception as e:
                log_debug(f"[table_stats] dry run failed for {table!r}: {e}")
                return super().columns(table, relation, timeout)
        return [(field.name, _bigquery_type(field)) for field in fields]

    def execute(self, sql, params=None, timeout=None):
        from google.cloud import bigquery  # type: ignore[import-not-found]

        # Built here rather than by source.execute, which replaces any
        # job_config when it binds parameters and would drop the caps.
        config = self.source._build_query_config(params) if params else bigquery.QueryJobConfig()
        if self.max_bytes_billed:
            config.maximum_bytes_billed = int(self.max_bytes_billed)
        if timeout:
            config.job_timeout_ms = max(1, int(timeout * 1000))
        start = time.monotonic()
        try:
            return self.source.execute(sql, None, job_config=config)
        except Exception as e:
            if timeout and time.monotonic() - start >= timeout:
                raise StatementTimeout(f"statement exceeded {timeout:g}s") from e
            raise

    def row_estimate(self, table):
        bq_table = self._table(table)
        return int(bq_table.num_rows) if bq_table is not None and bq_table.num_rows is not None else None

    def sample_sql(self, relation, rows, total, seed):
        if total is None or total <= rows:
            return super().sample_sql(relation, rows, total, seed)
        if relation.startswith("("):
            # TABLESAMPLE only applies to tables, and anything else bills a
            # scan of the whole expression.
            return None, False
        pct = _percent(rows, total)
        # BigQuery cannot seed TABLESAMPLE; the persisted cache is what keeps
        # the rendered context stable across runs.
        return f"SELECT * FROM {relation} TABLESAMPLE SYSTEM ({pct:.4f} PERCENT) LIMIT {rows}", True



def _bigquery_type(field) -> str:
    if field.field_type in ("RECORD", "STRUCT"):
        sql_type = "STRUCT"
    else:
        sql_type = field.field_type
    return f"ARRAY<{sql_type}>" if field.mode == "REPEATED" else sql_type


def get_adapter(source: Source, max_bytes_billed: int | None = None) -> StatsAdapter:
    from ..sources.duckdb import DuckDBSource
    if isinstance(source, DuckDBSource):
        return DuckDBStatsAdapter(source)
    name = type(source).__name__
    if name == "SnowflakeSource":
        return SnowflakeStatsAdapter(source)
    if name == "BigQuerySource":
        return BigQueryStatsAdapter(source, max_bytes_billed)
    if hasattr(source, "_engine") and hasattr(source, "_url"):
        return SQLAlchemyStatsAdapter(source)
    return StatsAdapter(source)


#: Statements of engines that cannot be interrupted from inside run here so
#: the profiler can give up on them.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="lumen-table-stats-sql")


def _run_abandonable(fn, timeout: float | None):
    """
    Run `fn` in a worker and stop waiting after `timeout` seconds. The deadline
    covers time spent queued behind statements abandoned earlier, and a
    statement that never started is cancelled rather than run late.
    """
    if not timeout:
        return fn()
    deadline = time.monotonic() + timeout
    started = threading.Event()

    def run():
        started.set()
        return fn()

    future = _EXECUTOR.submit(run)
    try:
        if not started.wait(timeout):
            raise FutureTimeout
        return future.result(timeout=max(0.0, deadline - time.monotonic()))
    except FutureTimeout as e:
        future.cancel()
        raise StatementTimeout(f"statement exceeded {timeout:g}s") from e


def _run_interruptible(fn, timeout: float | None, interrupt):
    """Run `fn` in this thread and call `interrupt` from a timer once `timeout` passes."""
    if not timeout:
        return fn()
    fired = threading.Event()

    def fire():
        fired.set()
        try:
            interrupt()
        except Exception:
            pass

    timer = threading.Timer(timeout, fire)
    timer.daemon = True
    timer.start()
    try:
        return fn()
    except Exception as e:
        if fired.is_set():
            raise StatementTimeout(f"statement exceeded {timeout:g}s") from e
        raise
    finally:
        timer.cancel()


def _is_pg_timeout(error: Exception) -> bool:
    orig = getattr(error, "orig", error)
    return (getattr(orig, "pgcode", None) or getattr(orig, "sqlstate", None)) == "57014"


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class _Budget:
    """Wall time spent profiling one database, accumulated across tables."""

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.spent = 0.0
        self.created = time.monotonic()
        self._local = threading.local()

    def __enter__(self):
        self._local.start = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.spent += time.monotonic() - self._local.start
        self._local.start = None

    def remaining(self) -> float | None:
        if self.seconds is None:
            return None
        start = getattr(self._local, "start", None)
        running = 0 if start is None else time.monotonic() - start
        return self.seconds - self.spent - running

    def timeout(self, statement_timeout: float | None) -> float | None:
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            raise StatsBudgetExceeded("statistics budget exhausted")
        if remaining is None:
            return statement_timeout
        return remaining if statement_timeout is None else min(remaining, statement_timeout)


def _signature(columns: list[tuple[str, str | None]]) -> str:
    return hashlib.sha1(json.dumps(columns, default=str).encode()).hexdigest()


class TableProfiler(param.Parameterized):
    """
    Computes `TableStats` for one table with the tiered strategy described in
    the module docstring.
    """

    exact_row_limit = param.Integer(default=1_000_000, bounds=(0, None), doc="""
        Tables with at most this many rows get exact statistics from one
        aggregate pass; larger tables are sampled.""")

    sample_rows = param.Integer(default=10_000, bounds=(1, None), doc="""
        Rows drawn by the engine-side sample for tables above
        `exact_row_limit`, and the source of example values.""")

    group_by_row_limit = param.Integer(default=10_000_000, bounds=(0, None), doc="""
        On local engines, largest table for which exact value counts of
        low-cardinality columns are computed after sampling.""")

    enum_limit = param.Integer(default=24, bounds=(1, None), doc="""
        Columns with at most this many distinct values list their values.""")

    top_values = param.Integer(default=12, bounds=(1, None), doc="""
        Values listed for an enumerated column.""")

    statement_timeout = param.Number(default=30, allow_None=True, doc="""
        Seconds any single statistics query may run.""")

    seed = param.Integer(default=42, doc="Seed for engine-side sampling.")

    max_bytes_billed = param.Integer(default=10 * 1024**3, allow_None=True, doc="""
        Per-query cap on bytes billed for BigQuery statistics queries.""")

    def profile(self, source: Source, table: str, budget: _Budget | None = None) -> TableStats:
        budget = budget or _Budget(None)
        with budget:
            return self._profile(source, table, budget)

    def _profile(self, source: Source, table: str, budget: _Budget) -> TableStats:
        adapter = get_adapter(source, self.max_bytes_billed)
        stats = TableStats(table=table, computed_at=time.time())
        relation = adapter.relation(table)
        params = adapter.params(table)
        try:
            # Names and types are cheap and still rendered once the budget is spent.
            columns = adapter.columns(table, relation, self.statement_timeout)
        except Exception as e:
            log_debug(f"[table_stats] could not describe {table!r}: {e}")
            return stats
        stats.signature = _signature(columns)
        pk, fks = adapter.keys(table)
        stats.columns = [
            ColumnStats(
                name=name, type=sql_type, kind=column_kind(sql_type),
                primary_key=name in pk, references=fks.get(name),
            ) for name, sql_type in columns
        ]
        try:
            self._compute(adapter, stats, relation, params, budget)
        except (StatsBudgetExceeded, StatementTimeout) as e:
            log_debug(f"[table_stats] {table!r} limited to {stats.method}: {e}")
        except Exception as e:
            log_debug(f"[table_stats] statistics failed for {table!r}: {type(e).__name__}: {e}")
        return stats

    def _compute(self, adapter, stats, relation, params, budget):
        rows = self._row_count(adapter, stats, relation, params, budget)
        if stats.rows_exact and rows <= self.exact_row_limit:
            try:
                # Only for example values. Where LIMIT bills a full scan, a
                # real sample is cheaper than the first rows.
                sample = self._sample(adapter, stats, relation, params, budget, rows if adapter.limit_scans_table else None)
            except (StatsBudgetExceeded, StatementTimeout):
                raise
            except Exception as e:
                log_debug(f"[table_stats] no example rows for {stats.table!r}: {e}")
                sample = None
            self._exact(adapter, stats, relation, params, budget, sample)
            return
        if adapter.metadata_stats(stats.table, stats.columns, stats.rows, budget.timeout(self.statement_timeout)):
            stats.method = ESTIMATED
            # Only a claim about the ranges metadata actually supplied;
            # anything filled from the sample later is not exact.
            ranged = [c for c in stats.columns if c.kind in ("numeric", "temporal")]
            stats.ranges_exact = adapter.exact_metadata_ranges and bool(ranged) and all(
                c.min is not None and c.max is not None for c in ranged
            )
        sample = self._sample(adapter, stats, relation, params, budget, rows)
        if sample is None:
            return
        self._from_sample(stats, sample)
        if stats.method != ESTIMATED:
            stats.method = SAMPLED
        if adapter.local and stats.rows is not None and stats.rows <= self.group_by_row_limit:
            self._value_counts(adapter, stats, relation, params, budget)

    def _row_count(self, adapter, stats, relation, params, budget) -> int:
        """
        Set `stats.rows` and return it, or a lower bound when the table is
        larger than `exact_row_limit` and the engine offers no estimate.
        """
        if (exact := adapter.exact_row_count(stats.table)) is not None:
            stats.rows, stats.rows_exact = exact, True
            return exact
        if adapter.metadata_count and adapter.bare_name(stats.table):
            df = adapter.execute(f"SELECT COUNT(*) AS n FROM {relation}", params, budget.timeout(self.statement_timeout))
            stats.rows, stats.rows_exact = int(df.iloc[0, 0]), True
            return stats.rows
        estimate = adapter.row_estimate(stats.table)
        if estimate is not None and estimate > self.exact_row_limit:
            stats.rows, stats.rows_exact = estimate, False
            return estimate
        # Bounded so a huge table costs at most exact_row_limit rows of scan.
        limit = self.exact_row_limit + 1
        bounded = adapter.limit(f"SELECT 1 AS one FROM {relation}", limit)
        df = adapter.execute(
            f"SELECT COUNT(*) AS n FROM ({bounded}) {adapter.alias('lumen_c')}",
            params, budget.timeout(self.statement_timeout),
        )
        count = int(df.iloc[0, 0])
        if count < limit:
            stats.rows, stats.rows_exact = count, True
            return count
        if estimate is None and adapter.local:
            try:
                df = adapter.execute(f"SELECT COUNT(*) AS n FROM {relation}", params, budget.timeout(self.statement_timeout))
                stats.rows, stats.rows_exact = int(df.iloc[0, 0]), True
                return stats.rows
            except StatementTimeout:
                pass
        stats.rows, stats.rows_exact = estimate, False
        if estimate is None:
            stats.rows_at_least = self.exact_row_limit
            return limit
        return estimate

    def _sample(self, adapter, stats, relation, params, budget, total: int | None) -> pd.DataFrame | None:
        sql, random = adapter.sample_sql(relation, self.sample_rows, total, self.seed)
        if sql is None:
            return None
        first_rows = adapter.limit(f"SELECT * FROM {relation}", self.sample_rows)
        try:
            df = adapter.execute(sql, params, budget.timeout(self.statement_timeout))
        except (StatsBudgetExceeded, StatementTimeout):
            raise
        except Exception as e:
            if not random or adapter.limit_scans_table:
                raise
            log_debug(f"[table_stats] sampling failed for {stats.table!r}, reading first rows: {e}")
            df, random = adapter.execute(first_rows, params, budget.timeout(self.statement_timeout)), False
        if random and df is not None and df.empty and (total or 0) > 0 and not adapter.limit_scans_table:
            # Block sampling can miss every block of a small or skewed table.
            df, random = adapter.execute(first_rows, params, budget.timeout(self.statement_timeout)), False
        if df is None:
            return None
        stats.sample_size = len(df)
        # Fewer rows than requested means the read covered the whole table.
        stats.sample_random = random or len(df) < self.sample_rows
        for col in stats.columns:
            # SQLite columns may be declared without a type, and generic
            # adapters report none at all.
            if col.type in (None, "", "NULL") and col.name in df.columns:
                col.type, col.kind = _kind_from_series(df[col.name])
        return df

    def _exact(self, adapter, stats, relation, params, budget, sample):
        # Aliases must be valid unquoted identifiers everywhere, so no
        # leading underscores (Oracle).
        groups = []
        for i, col in enumerate(stats.columns):
            q = quote_identifier(col.name, adapter.dialect)
            exprs = [f"COUNT({q}) AS lumen_c{i}"]
            if col.kind != "other":
                exprs.append(f"COUNT(DISTINCT {q}) AS lumen_d{i}")
            if col.kind in ("numeric", "temporal"):
                exprs += [f"MIN({q}) AS lumen_lo{i}", f"MAX({q}) AS lumen_hi{i}"]
            groups.append(exprs)

        def aggregate(exprs):
            df = adapter.execute(
                f"SELECT {', '.join(['COUNT(*) AS lumen_n', *exprs])} FROM {relation}",
                params, budget.timeout(self.statement_timeout),
            )
            # Per column, since a row-wise iloc upcasts mixed dtypes to float.
            return {str(k).lower(): df[k].iloc[0] for k in df.columns}

        # Chunked so very wide tables stay under engine limits on select-list size.
        row: dict[str, Any] = {}
        chunk: list[list[str]] = []
        for group in [*groups, None]:
            if group is not None and sum(map(len, chunk)) + len(group) <= 240:
                chunk.append(group)
                continue
            if chunk:
                try:
                    row.update(aggregate([e for g in chunk for e in g]))
                except (StatsBudgetExceeded, StatementTimeout):
                    raise
                except Exception as e:
                    # One column the engine cannot aggregate, such as a CLOB
                    # or a Postgres range, fails the statement. Retrying per
                    # column limits the loss to that column, keeping its plain
                    # COUNT where DISTINCT or MIN/MAX are what failed.
                    log_debug(f"[table_stats] aggregates failed for {stats.table!r}, retrying per column: {e}")
                    for g in chunk:
                        for attempt in (g, g[:1]) if len(g) > 1 else (g,):
                            try:
                                row.update(aggregate(attempt))
                                break
                            except (StatsBudgetExceeded, StatementTimeout):
                                raise
                            except Exception:
                                continue
            chunk = [group] if group is not None else []
        if "lumen_n" not in row:
            row.update(aggregate([]))
        total = int(row["lumen_n"])
        stats.rows, stats.rows_exact, stats.rows_at_least = total, True, None
        for i, col in enumerate(stats.columns):
            if f"lumen_c{i}" not in row:
                continue
            nonnull = int(row[f"lumen_c{i}"])
            col.nulls = (1 - nonnull / total) if total else 0.0
            if f"lumen_d{i}" in row and row[f"lumen_d{i}"] is not None:
                col.distinct = int(row[f"lumen_d{i}"])
            if col.kind in ("numeric", "temporal"):
                col.min, col.max = _to_python(row.get(f"lumen_lo{i}")), _to_python(row.get(f"lumen_hi{i}"))
        stats.method, stats.ranges_exact, stats.sample_size = EXACT, True, None
        if sample is not None:
            self._examples(stats, sample)
        self._value_counts(adapter, stats, relation, params, budget)

    def _is_candidate(self, col: ColumnStats) -> bool:
        return (
            col.kind in ("string", "boolean")
            and col.distinct is not None
            and 0 < col.distinct <= self.enum_limit
        )

    def _value_counts(self, adapter, stats, relation, params, budget):
        for col in stats.columns:
            if not self._is_candidate(col):
                continue
            q = quote_identifier(col.name, adapter.dialect)
            sql = adapter.limit(
                f"SELECT {q} AS v, COUNT(*) AS n FROM {relation} WHERE {q} IS NOT NULL "
                f"GROUP BY {q} ORDER BY n DESC, v", self.top_values,
            )
            try:
                df = adapter.execute(sql, params, budget.timeout(self.statement_timeout))
            except (StatsBudgetExceeded, StatementTimeout):
                raise
            except Exception as e:
                log_debug(f"[table_stats] value counts failed for {col.name!r}: {e}")
                continue
            col.values = [[_to_python(v), int(n)] for v, n in zip(df.iloc[:, 0], df.iloc[:, 1], strict=False)]
            # Fewer rows than the LIMIT means GROUP BY returned every value.
            col.values_complete = len(col.values) < self.top_values or (
                stats.method == EXACT and len(col.values) >= (col.distinct or 0)
            )
            if col.values_complete:
                col.distinct = len(col.values)
            col.examples = None

    def _examples(self, stats: TableStats, sample: pd.DataFrame):
        for col in stats.columns:
            if col.kind != "string" or col.name not in sample.columns or self._is_candidate(col):
                continue
            series = sample[col.name].dropna()
            if series.empty:
                continue
            try:
                counts = series.astype(str).value_counts()
            except Exception:
                continue
            # Most frequent first, ties broken by value, so the pick is stable.
            ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            col.examples = [value for value, _ in ordered[:3]]

    def _from_sample(self, stats: TableStats, sample: pd.DataFrame):
        n = len(sample)
        for col in stats.columns:
            if col.name not in sample.columns:
                continue
            series = sample[col.name]
            nonnull = series.dropna()
            if col.nulls is None and n:
                col.nulls = 1 - len(nonnull) / n
            if col.kind == "other" or nonnull.empty:
                continue
            try:
                distinct = int(nonnull.astype(str).nunique()) if nonnull.dtype == object else int(nonnull.nunique())
            except Exception:
                continue
            if col.distinct is None:
                col.distinct = distinct
            if col.kind in ("numeric", "temporal") and col.min is None:
                try:
                    col.min, col.max = _to_python(nonnull.min()), _to_python(nonnull.max())
                except Exception:
                    pass
            if col.values is None and col.kind in ("string", "boolean") and distinct <= self.enum_limit:
                counts = nonnull.value_counts()
                ordered = sorted(
                    ((_to_python(v), int(c)) for v, c in counts.items()),
                    key=lambda kv: (-kv[1], str(kv[0])),
                )[:self.top_values]
                col.values = [[v, round(c / n, 4)] for v, c in ordered]
                col.values_complete = False
        self._examples(stats, sample)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PLAIN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNSAFE_VALUE_RE = re.compile(r"[,:{}'\"]|^\s|\s$|^$|^NULL$")
MAX_VALUE_CHARS = 60
# Characters of listed values per column before the rest are summarised as
# "+N more"; long values (descriptions, JSON) otherwise dominate the prompt.
MAX_VALUES_CHARS = 180
MAX_EXAMPLE_CHARS = 40
MAX_EXAMPLES_CHARS = 60


def format_name(name: str) -> str:
    """Column names are rendered as SQL would need them."""
    return name if _PLAIN_NAME_RE.match(name) else '"' + name.replace('"', '""') + '"'


def format_number(value: float | int) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) < 1e15):
        return str(int(value))
    if value != 0 and (abs(value) >= 1e15 or abs(value) < 1e-4):
        return f"{value:.6g}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    # Keep six significant digits for small magnitudes, e.g. 0.00176056.
    if abs(value) < 1:
        text = f"{value:.6g}"
    return text


def format_value(
    value: Any, kind: str = "string", quote: bool | None = None, sql_type: str | None = None,
    max_chars: int = MAX_VALUE_CHARS,
) -> str:
    """Render a literal compactly without losing what a SQL filter needs."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float) and kind != "string":
        return format_number(value)
    text = str(value)
    if kind == "temporal":
        if sql_type and re.fullmatch(r"DATE", sql_type.strip(), re.IGNORECASE) and text.endswith(" 00:00:00"):
            text = text[:-9]
        return text
    if len(text) > max_chars:
        text = text[:max_chars - 1] + "…"
    if quote is None:
        quote = bool(_UNSAFE_VALUE_RE.search(text))
    if quote:
        return "'" + text.replace("'", "''") + "'"
    return text


def _fit(items: list[str], max_chars: int, minimum: int) -> list[str]:
    """The leading items whose combined length stays within `max_chars`."""
    kept, used = [], 0
    for item in items:
        if len(kept) >= minimum and used + len(item) > max_chars:
            break
        kept.append(item)
        used += len(item) + 2
    return kept


def _format_count(count: Any) -> str:
    if count is None:
        return ""
    # Floats are shares of a sample or of pg_stats; counts are always ints.
    if isinstance(count, float):
        pct = count * 100
        return ":<1%" if 0 < pct < 1 else f":{pct:.0f}%"
    return f":{int(count)}"


def _format_nulls(nulls: float | None) -> str | None:
    if not nulls:
        return None
    if nulls >= 1:
        return None
    pct = nulls * 100
    if pct < 1:
        return "nulls <1%"
    if pct > 99:
        return "nulls >99%"
    return f"nulls {pct:.0f}%"


def render_column(
    col: ColumnStats, stats: TableStats | None = None, description: str | None = None,
    detail: str = "full", name: str | None = None,
) -> str:
    """
    One line per column, e.g. ``AvgScrMath INTEGER 289..699 nulls 26%``.

    `detail` is ``"full"`` for every statistic, ``"types"`` for name, type and
    keys only.
    """
    parts = [name or format_name(col.name)]
    if col.type:
        parts.append(col.type)
    if col.primary_key:
        parts.append("PK")
    if col.references:
        parts.append(f"-> {col.references}")
    if detail == "full" and stats is not None and stats.method != TYPES_ONLY:
        parts.extend(_column_facts(col, stats))
    line = " ".join(parts)
    if description:
        line += f" -- {' '.join(description.split())}"
    return line


def _column_facts(col: ColumnStats, stats: TableStats) -> list[str]:
    facts = []
    if col.nulls is not None and col.nulls >= 1:
        return ["all NULL"]
    exact = stats.method == EXACT
    nonnull_rows = None
    if exact and stats.rows is not None and col.nulls is not None:
        nonnull_rows = round(stats.rows * (1 - col.nulls))
    unique = bool(exact and col.distinct and nonnull_rows and nonnull_rows > 1 and col.distinct == nonnull_rows)
    if col.values:
        items = _fit([format_value(v, col.kind, sql_type=col.type) + _format_count(c) for v, c in col.values],
                     MAX_VALUES_CHARS, minimum=3)
        shown = ", ".join(items)
        hidden = None
        if col.distinct and col.distinct > len(items) and (exact or col.values_complete):
            hidden = col.distinct - len(items)
        elif len(items) < len(col.values):
            hidden = len(col.values) - len(items) if col.values_complete else None
        if hidden:
            facts.append("{" + shown + f", +{hidden} more" + "}")
        elif col.values_complete:
            facts.append("{" + shown + "}")
        else:
            facts.append("{" + shown + ", ...}")
    elif col.kind in ("numeric", "temporal") and col.min is not None and col.max is not None:
        lo = format_value(col.min, col.kind, sql_type=col.type)
        hi = format_value(col.max, col.kind, sql_type=col.type)
        facts.append(lo if lo == hi else f"{lo}..{hi}")
        if unique and not col.primary_key:
            facts.append("unique")
    elif col.kind == "string":
        if unique and not col.primary_key:
            facts.append("unique")
        elif exact and col.distinct and not unique:
            facts.append(f"{col.distinct} distinct")
        if col.examples:
            examples = [format_value(v, "string", quote=True, max_chars=MAX_EXAMPLE_CHARS) for v in col.examples]
            facts.append("e.g. " + ", ".join(_fit(examples, MAX_EXAMPLES_CHARS, minimum=1)))
    if nulls := _format_nulls(col.nulls):
        facts.append(nulls)
    return facts


def render_stats_header(stats: TableStats | None) -> str:
    """Row count and how the column statistics were obtained."""
    if stats is None:
        return ""
    if stats.rows is None:
        rows = f"over {stats.rows_at_least} rows" if stats.rows_at_least else None
    elif stats.rows_exact:
        rows = f"{stats.rows} rows"
    else:
        rows = f"~{stats.rows} rows"
    notes = [rows] if rows else []
    if stats.method == SAMPLED:
        if stats.sample_random:
            notes.append(f"stats from a {stats.sample_size}-row sample")
        else:
            notes.append(f"stats from the first {stats.sample_size} rows")
    elif stats.method == ESTIMATED:
        notes.append("engine-estimated stats" + ("" if not stats.ranges_exact else ", exact ranges"))
    return "; ".join(notes)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _default_cache_dir() -> Path | None:
    env = os.environ.get("LUMEN_TABLE_STATS_CACHE")
    if env is not None:
        return Path(env).expanduser() if env.strip() else None
    return Path(user_cache_dir("lumen")) / "table_stats"


@dataclass
class _Entry:
    stats: TableStats
    token: str | None
    checked_at: float


#: Profiling runs here rather than in the event loop's default executor,
#: which also serves LLM tool calls and query execution.
_PROFILE_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="lumen-table-stats")


def _forget(store_ref: weakref.ref, identity: str) -> None:
    if (store := store_ref()) is not None:
        store._forget(identity)


class TableStatsStore(param.Parameterized):
    """
    Computes table statistics in the background and caches them in memory
    and, for sources with a stable identity, on disk.
    """

    budget_seconds = param.Number(default=300, allow_None=True, doc="""
        Wall time one database may spend on statistics within
        `budget_window`. Once spent, further tables render names and types
        only.""")

    budget_window = param.Number(default=3600, allow_None=True, doc="""
        Seconds after which a database's spent budget is reset, so a
        long-running server keeps profiling new and changed tables. None
        never resets it.""")

    cache_dir = param.Parameter(default=None, doc="""
        Directory for persisted statistics; None disables persistence.
        Defaults to the user cache directory, or LUMEN_TABLE_STATS_CACHE.
        Entries contain data values and are written readable by the
        current user only on POSIX systems; on Windows the user cache
        directory's ACL applies.""")

    max_age = param.Number(default=7 * 24 * 3600, allow_None=True, doc="""
        Seconds cached statistics stay valid when the engine exposes no
        modification token for the table.""")

    retry_after = param.Number(default=300, doc="""
        Seconds before a table whose statistics could not be computed, and
        which therefore renders names and types only, is profiled again.""")

    revalidate_interval = param.Number(default=60, doc="""
        Seconds between checks that a cached table on a remote engine is
        unchanged. Tables on local engines are checked on every lookup.""")

    max_entries = param.Integer(default=5000, bounds=(1, None), doc="""
        Tables kept in memory; the least recently used are dropped first.""")

    concurrency = param.Integer(default=2, bounds=(1, None), doc="""
        Tables of one source profiled concurrently.""")

    def __init__(self, profiler: TableProfiler | None = None, **params):
        if "cache_dir" not in params:
            params["cache_dir"] = _default_cache_dir()
        super().__init__(**params)
        self.profiler = profiler or TableProfiler()
        self._memory: OrderedDict[tuple, _Entry] = OrderedDict()
        self._tasks: dict[tuple, asyncio.Future] = {}
        # Keyed by database identity so sources derived from one another
        # share a single budget and concurrency limit.
        self._budgets: dict[str, _Budget] = {}
        self._slots: dict[str, threading.BoundedSemaphore] = {}
        self._async_slots: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._ephemeral: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()

    def clear(self):
        with self._lock:
            self._memory.clear()
            self._tasks.clear()
            self._budgets.clear()
            self._slots.clear()

    # Keys ----------------------------------------------------------------------

    def _adapter(self, source: Source) -> StatsAdapter:
        return get_adapter(source, self.profiler.max_bytes_billed)

    def _identity(self, source: Source, adapter: StatsAdapter) -> str:
        if (identity := adapter.identity()) is not None:
            return identity
        # Sources derived with create_sql_expr_source share a DuckDB
        # connection, so key by it and every derivation reuses the stats.
        # The identity is minted per object because id() is reused once an
        # object is freed, which would serve one session's data to another.
        owner = getattr(source, "_connection", None)
        if owner is None:
            owner = source
        with self._lock:
            identity = self._ephemeral.get(owner)
            if identity is None:
                identity = f"memory:{uuid.uuid4().hex}"
                self._ephemeral[owner] = identity
                weakref.finalize(owner, _forget, weakref.ref(self), identity)
        return identity

    def _forget(self, identity: str) -> None:
        with self._lock:
            for key in [k for k in self._memory if k[0] == identity]:
                del self._memory[key]
            for key in [k for k in self._tasks if k[0] == identity]:
                del self._tasks[key]
            self._budgets.pop(identity, None)
            self._slots.pop(identity, None)
            for slots in list(self._async_slots.values()):
                slots.pop(identity, None)

    def _key(self, source: Source, table: str) -> tuple[tuple, StatsAdapter]:
        adapter = self._adapter(source)
        return (self._identity(source, adapter), table, adapter.definition(table)), adapter

    def _path(self, key: tuple) -> Path | None:
        if self.cache_dir is None or str(key[0]).startswith("memory:"):
            return None
        digest = hashlib.sha1(json.dumps(key, default=str).encode()).hexdigest()
        return Path(self.cache_dir) / f"{digest}.json"

    @staticmethod
    def _fingerprint(signature: str | None, token: str | None) -> str:
        return hashlib.sha1(json.dumps([STATS_VERSION, signature, token], default=str).encode()).hexdigest()

    @staticmethod
    def _token(adapter: StatsAdapter, table: str) -> str | None:
        try:
            return adapter.modified(table)
        except Exception:
            return None

    # Memory --------------------------------------------------------------------

    def _lookup(self, key: tuple, adapter: StatsAdapter, table: str, validate: str = "local") -> TableStats | None:
        """
        A cached entry if it is still valid. `validate` is ``"local"`` to
        check the modification token on local engines only, since remote
        ones may need a network round trip, ``"all"`` to also check remote
        engines once `revalidate_interval` has passed, or ``"none"`` for an
        entry already validated by the caller.
        """
        with self._lock:
            entry = self._memory.get(key)
        if entry is None:
            return None
        now = time.time()
        stats = entry.stats
        valid = True
        if stats.method == TYPES_ONLY and now - stats.computed_at > self.retry_after:
            valid = False
        elif entry.token is None and self.max_age is not None and now - stats.computed_at > self.max_age:
            valid = False
        elif validate != "none" and (
            adapter.local or (validate == "all" and now - entry.checked_at > self.revalidate_interval)
        ):
            valid = self._token(adapter, table) == entry.token
            entry.checked_at = now
        with self._lock:
            if not valid:
                if self._memory.get(key) is entry:
                    del self._memory[key]
                return None
            if key in self._memory:
                self._memory.move_to_end(key)
        return stats

    def _needs_check(self, key: tuple, adapter: StatsAdapter) -> bool:
        entry = self._memory.get(key)
        return (
            entry is not None and not adapter.local
            and time.time() - entry.checked_at > self.revalidate_interval
        )

    def _remember(self, key: tuple, stats: TableStats, token: str | None) -> None:
        with self._lock:
            self._memory[key] = _Entry(stats, token, time.time())
            self._memory.move_to_end(key)
            while len(self._memory) > self.max_entries:
                self._memory.popitem(last=False)

    # Disk ----------------------------------------------------------------------

    def _load(self, key: tuple, adapter: StatsAdapter, table: str) -> TableStats | None:
        path = self._path(key)
        if path is None or not path.exists():
            return None
        try:
            stats = TableStats.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None
        if stats.version != STATS_VERSION:
            return None
        token = self._token(adapter, table)
        if token is None and self.max_age is not None and time.time() - stats.computed_at > self.max_age:
            return None
        try:
            columns = adapter.columns(table, adapter.relation(table), self.profiler.statement_timeout)
        except Exception:
            return None
        if stats.fingerprint != self._fingerprint(_signature(columns), token):
            return None
        self._remember(key, stats, token)
        return stats

    def _persist(self, path: Path, stats: TableStats) -> None:
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(stats.to_dict(), default=str))
            os.replace(tmp, path)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            log_debug(f"[table_stats] could not persist stats for {stats.table!r}: {e}")

    # Computation ---------------------------------------------------------------

    def _budget(self, identity: str) -> _Budget:
        with self._lock:
            budget = self._budgets.get(identity)
            if budget is None or (
                self.budget_window is not None and time.monotonic() - budget.created > self.budget_window
            ):
                budget = self._budgets[identity] = _Budget(self.budget_seconds)
            return budget

    def _from_schema(self, source: Source, table: str) -> TableStats:
        # Sources without SQL describe themselves through get_schema; the
        # whole table, unshuffled, so the result is deterministic.
        try:
            schema = source.get_schema(table)
        except Exception as e:
            log_debug(f"[table_stats] get_schema failed for {table!r}: {e}")
            return TableStats(table=table, computed_at=time.time())
        stats = TableStats.from_json_schema(table, schema)
        stats.computed_at = time.time()
        stats.signature = _signature([(c.name, c.type) for c in stats.columns])
        return stats

    def get(self, source: Source, table: str) -> TableStats | None:
        """
        Cached statistics, if any, without computing anything or touching
        remote engines, so it is safe to call from the event loop.
        """
        try:
            key, adapter = self._key(source, table)
        except Exception:
            return None
        return self._lookup(key, adapter, table)

    def compute(self, source: Source, table: str) -> TableStats:
        """Compute (or load) statistics synchronously."""
        key, adapter = self._key(source, table)
        if (cached := self._lookup(key, adapter, table, validate="all")) is not None:
            return cached
        with self._lock:
            slot = self._slots.setdefault(key[0], threading.BoundedSemaphore(self.concurrency))
        with slot:
            # Invalid entries were dropped above, so anything here was just
            # computed by another thread.
            if (cached := self._lookup(key, adapter, table, validate="none")) is not None:
                return cached
            if (loaded := self._load(key, adapter, table)) is not None:
                return loaded
            if hasattr(source, "execute"):
                stats = self.profiler.profile(source, table, self._budget(key[0]))
            else:
                stats = self._from_schema(source, table)
        if not stats.columns:
            # Not even described; caching that would hide the table until
            # retry_after, though the cause (e.g. a transient error) may pass.
            return stats
        token = self._token(adapter, table)
        stats.fingerprint = self._fingerprint(stats.signature, token)
        self._remember(key, stats, token)
        path = self._path(key)
        if path is not None and stats.method != TYPES_ONLY:
            self._persist(path, stats)
        return stats

    def _async_slot(self, identity: str) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._lock:
            slots = self._async_slots.get(loop)
            if slots is None:
                slots = self._async_slots[loop] = {}
            return slots.setdefault(identity, asyncio.Semaphore(self.concurrency))

    def schedule(self, source: Source, tables: list[str]) -> list[asyncio.Future]:
        """Start computing statistics for `tables` in the background."""
        return list(self._schedule(source, tables)[1].values())

    def _schedule(
        self, source: Source, tables: list[str]
    ) -> tuple[dict[str, TableStats], dict[str, asyncio.Future]]:
        """Validated cached statistics, and tasks for the tables that need computing."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return {}, {}
        ready, futures = {}, {}
        for table in tables:
            try:
                key, adapter = self._key(source, table)
            except Exception:
                continue
            cached = self._lookup(key, adapter, table)
            if cached is not None:
                ready[table] = cached
                if not self._needs_check(key, adapter):
                    continue
            with self._lock:
                task = self._tasks.get(key)
                if task is None or task.done() or task.get_loop() is not loop:
                    task = loop.create_task(self._run(source, table, key))
                    self._tasks[key] = task
            futures[table] = task
        return ready, futures

    async def _run(self, source, table, key):
        try:
            # Tables wait for a slot here rather than in a worker thread, so
            # a large catalog holds no threads while queued.
            async with self._async_slot(key[0]):
                if self._adapter(source).thread_bound:
                    # Only the thread that created the data sees it, which
                    # is normally the loop's; the statement timeout still
                    # bounds how long the loop is blocked.
                    return self.compute(source, table)
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(_PROFILE_EXECUTOR, self.compute, source, table)
        except Exception as e:
            log_debug(f"[table_stats] failed for {table!r}: {e}")
            return None
        finally:
            with self._lock:
                if self._tasks.get(key) is asyncio.current_task():
                    self._tasks.pop(key, None)

    async def ensure(
        self, source: Source, tables: list[str], timeout: float | None = None
    ) -> dict[str, TableStats]:
        """
        Return statistics for `tables`, waiting at most `timeout` seconds for
        any still being computed; those keep computing in the background.
        """
        ready, futures = self._schedule(source, tables)
        pending = [f for f in futures.values() if not f.done()]
        if pending:
            await asyncio.wait(pending, timeout=timeout)
        # Results come from the tasks and the lookups above, so no table is
        # validated twice.
        for table, future in futures.items():
            if future.done() and not future.cancelled() and (stats := future.result()) is not None:
                ready[table] = stats
        return {table: ready[table] for table in tables if table in ready}


_DEFAULT_STORE: dict[str, TableStatsStore] = {}


def get_stats_store() -> TableStatsStore:
    """The process-wide store agents and metasets share."""
    if "store" not in _DEFAULT_STORE:
        _DEFAULT_STORE["store"] = TableStatsStore()
    return _DEFAULT_STORE["store"]


def set_stats_store(store: TableStatsStore | None) -> None:
    """Replace the shared store, or reset it to a fresh default with None."""
    if store is None:
        _DEFAULT_STORE.pop("store", None)
    else:
        _DEFAULT_STORE["store"] = store
