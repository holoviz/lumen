"""
Column statistics for the table context shown to the LLM.

`Source.get_schema` describes a table for widgets and filters. The LLM needs
something different: SQL types, keys, true value ranges and literal values,
and it needs them to be byte-identical across runs so providers can cache the
prompt. Computing that must also stay affordable on warehouse tables, so each
table is profiled with the cheapest strategy that is still accurate:

1. Metadata the engine already maintains (``pg_stats``, Parquet footers,
   Snowflake micro-partition metadata, ``duckdb_tables()``), which is free.
2. One exact aggregate pass when the table has at most ``exact_row_limit`` rows.
3. A seeded engine-side sample of ``sample_rows`` rows above that.

Low-cardinality columns found in step 2 or 3 get exact value counts when the
engine can afford a ``GROUP BY``. Every result records how it was computed so
the prompt can say ``sampled 10000 of 2300000000 rows`` rather than implying
that a sampled range is exact.

Results are cached in memory and, for sources with a stable identity, on disk,
keyed by the table definition and validated against its column types and
modification time where the engine exposes one.
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

from concurrent.futures import (
    ThreadPoolExecutor, TimeoutError as FutureTimeout,
)
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import param

from .utils import log_debug

if TYPE_CHECKING:
    from ..sources.base import Source

#: Bumped whenever the computed statistics change meaning, which invalidates
#: every persisted entry.
STATS_VERSION = 1

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
_BARE_NAME_RE = re.compile(r'^\s*(?:"[^"]+"|`[^`]+`|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:"[^"]+"|`[^`]+`|[A-Za-z_][\w$]*)){0,2}\s*$')
_SELECT_STAR_RE = re.compile(r'^\s*SELECT\s+\*\s+FROM\s+(.+?)\s*;?\s*$', re.IGNORECASE | re.DOTALL)
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
    method: str = TYPES_ONLY
    sample_size: int | None = None
    ranges_exact: bool = False
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


def _kind_from_dtype(dtype) -> tuple[str, str]:
    kind = getattr(dtype, "kind", "O")
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


def _quote_qualified(name: str, dialect: str | None) -> str:
    parts = re.findall(r'"[^"]+"|`[^`]+`|[^.\s]+', name)
    return ".".join(
        part if part[0] in '"`' else quote_identifier(part, dialect) for part in parts
    )


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

    def __init__(self, source: Source):
        self.source = source
        self.dialect = getattr(source, "dialect", None)

    # Identity ---------------------------------------------------------------

    def identity(self) -> str | None:
        """A string identifying the database, or None when not stable across processes."""
        return None

    def definition(self, table: str) -> str:
        tables = getattr(self.source, "tables", None)
        if isinstance(tables, dict) and table in tables:
            return str(tables[table])
        try:
            return self.source.get_sql_expr(table)
        except Exception:
            return table

    def modified(self, table: str) -> str | None:
        """A token that changes when the table's data changes, if cheaply known."""
        paths = self._file_paths(table)
        if not paths:
            return None
        tokens = []
        for path in paths:
            try:
                stat = os.stat(path)
            except OSError:
                return None
            tokens.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        return "|".join(tokens)

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
        return f"({self.source.get_sql_expr(table)}) AS {quote_identifier('__lumen_t', self.dialect)}"

    def params(self, table: str):
        return (getattr(self.source, "table_params", None) or {}).get(table) or None

    # Execution -----------------------------------------------------------------

    def execute(self, sql: str, params=None, timeout: float | None = None) -> pd.DataFrame:
        return _run_with_timeout(lambda: self.source.execute(sql, params), timeout, self.cancel)

    def cancel(self) -> None:
        """Interrupt the statement currently running, where the engine allows it."""

    # Metadata ------------------------------------------------------------------

    def columns(self, table: str, relation: str, timeout: float | None) -> list[tuple[str, str | None]]:
        sql = f"SELECT * FROM {relation} LIMIT 0"
        df = self.execute(sql, self.params(table), timeout)
        return [(str(name), None) for name in df.columns] if df is not None else []

    def keys(self, table: str) -> tuple[set[str], dict[str, str]]:
        return set(), {}

    def row_estimate(self, table: str) -> int | None:
        """A row count the engine already knows without scanning."""
        return None

    def metadata_stats(self, table: str, columns: list[ColumnStats], rows: int | None, timeout: float | None) -> bool:
        """Fill min/max/nulls (and more) from engine metadata; return True on success."""
        return False

    def sample_sql(self, relation: str, rows: int, total: int | None, seed: int) -> tuple[str, bool]:
        """A query returning roughly `rows` rows, and whether it is a random sample."""
        return f"SELECT * FROM {relation} LIMIT {rows}", False



def _percent(rows: int, total: int | None) -> float:
    if not total:
        return 100.0
    return min(100.0, max(0.001, 100.0 * rows * 1.5 / total))


class DuckDBStatsAdapter(StatsAdapter):

    local = True

    def __init__(self, source):
        super().__init__(source)
        self._cursor = None

    def identity(self) -> str | None:
        uri = getattr(self.source, "uri", None)
        if uri and uri != ":memory:" and "://" not in uri:
            return f"duckdb:{Path(uri).expanduser().resolve()}"
        return None

    def modified(self, table: str) -> str | None:
        if self.identity() and self.bare_name(table):
            uri = Path(self.source.uri).expanduser()
            try:
                stat = uri.stat()
            except OSError:
                return None
            return f"{stat.st_size}:{stat.st_mtime_ns}"
        return super().modified(table)

    def relation(self, table: str) -> str:
        # DuckDB resolves views and table functions itself; the only thing to
        # guard against is a reserved word used as a table name.
        if name := self.bare_name(table):
            return _quote_qualified(name, self.dialect)
        expr = self.source.get_sql_expr(table)
        return f"({expr}) AS __lumen_t"

    def execute(self, sql: str, params=None, timeout: float | None = None) -> pd.DataFrame:
        def run():
            cursor = self.source._connection.cursor()
            self._cursor = cursor
            try:
                rel = cursor.execute(sql, params) if params else cursor.execute(sql)
                return rel.fetch_df()
            finally:
                self._cursor = None
                cursor.close()
        return _run_with_timeout(run, timeout, self.cancel)

    def cancel(self) -> None:
        if (cursor := self._cursor) is not None:
            try:
                cursor.interrupt()
            except Exception:
                pass

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

    def row_estimate(self, table):
        if path := self._parquet_path(table):
            try:
                df = self.execute(f"SELECT SUM(num_rows) AS n FROM parquet_file_metadata('{path}')", None, 10)
                return int(df["n"].iloc[0])
            except Exception:
                return None
        return None

    def _parquet_path(self, table) -> str | None:
        paths = [p for p in self._file_paths(table) if p.lower().endswith((".parquet", ".parq"))]
        return paths[0].replace("'", "''") if len(paths) == 1 else None

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
            return f"SELECT * FROM {relation} LIMIT {rows}", False
        if total <= self.reservoir_row_limit:
            return (
                f"SELECT * FROM {relation} USING SAMPLE reservoir({rows} ROWS) REPEATABLE ({seed})"
            ), True
        pct = _percent(rows, total)
        return (
            f"SELECT * FROM (SELECT * FROM {relation} USING SAMPLE {pct:.4f} PERCENT (system, {seed})) "
            f"AS __lumen_s LIMIT {rows}"
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
        from urllib.parse import unquote
        path = Path(unquote(database)).expanduser()
        return str(path.resolve()) if path.exists() else None

    def modified(self, table):
        if self.dialect == "sqlite" and (path := self._sqlite_path()):
            stat = os.stat(path)
            return f"{stat.st_size}:{stat.st_mtime_ns}"
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
        if self.dialect in ("postgresql", "postgres") and timeout:
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
        with self.source._engine.connect() as conn:
            conn.exec_driver_sql(f"SET statement_timeout = {int(timeout * 1000)}")
            try:
                result = conn.execute(text(sql), params or {})
                return pd.DataFrame(result.fetchall(), columns=list(result.keys()))
            finally:
                conn.exec_driver_sql("RESET statement_timeout")

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
            return f"SELECT * FROM {relation} LIMIT {rows}", False
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


class SnowflakeStatsAdapter(StatsAdapter):

    # Snowflake answers COUNT(*), COUNT(col), MIN and MAX on base tables from
    # micro-partition metadata, so these never consume warehouse time.
    metadata_count = True

    def identity(self):
        src = self.source
        parts = [getattr(src, attr, None) for attr in ("account", "database", "schema")]
        return "snowflake:" + "/".join(str(p) for p in parts) if parts[0] else None

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

    def identity(self):
        project = getattr(self.source, "project_id", None)
        return f"bigquery:{project}" if project else None

    def modified(self, table):
        try:
            meta = self.source.get_metadata(table) or {}
        except Exception:
            return None
        return str(meta.get("modified")) if meta.get("modified") else None

    def execute(self, sql, params=None, timeout=None):
        from google.cloud import bigquery  # type: ignore[import-not-found]
        config = bigquery.QueryJobConfig()
        if self.max_bytes_billed:
            config.maximum_bytes_billed = int(self.max_bytes_billed)
        if timeout:
            config.job_timeout_ms = int(timeout * 1000)
        return _run_with_timeout(
            lambda: self.source.execute(sql, params, job_config=config), timeout, self.cancel
        )

    def row_estimate(self, table):
        try:
            meta = self.source.get_metadata(table) or {}
        except Exception:
            return None
        rows = meta.get("num_rows", meta.get("rows"))
        return int(rows) if rows is not None else None

    def sample_sql(self, relation, rows, total, seed):
        if total is None or total <= rows or relation.startswith("("):
            return super().sample_sql(relation, rows, total, seed)
        pct = _percent(rows, total)
        # BigQuery cannot seed TABLESAMPLE; the persisted cache is what keeps
        # the rendered context stable across runs.
        return f"SELECT * FROM {relation} TABLESAMPLE SYSTEM ({pct:.4f} PERCENT) LIMIT {rows}", True



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


_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="lumen-table-stats")


def _run_with_timeout(fn, timeout: float | None, cancel=None):
    if not timeout:
        return fn()
    future = _EXECUTOR.submit(fn)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout as e:
        if cancel is not None:
            cancel()
        raise StatementTimeout(f"statement exceeded {timeout:g}s") from e


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class _Budget:
    """Wall time spent profiling one database, accumulated across tables."""

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.spent = 0.0
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
            sample = self._sample(adapter, stats, relation, params, budget, None)
            self._exact(adapter, stats, relation, params, budget, sample)
            return
        if adapter.metadata_stats(stats.table, stats.columns, stats.rows, budget.timeout(self.statement_timeout)):
            stats.method = ESTIMATED
            stats.ranges_exact = isinstance(adapter, SnowflakeStatsAdapter | DuckDBStatsAdapter)
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
        df = adapter.execute(
            f"SELECT COUNT(*) AS n FROM (SELECT 1 AS one FROM {relation} LIMIT {limit}) AS __lumen_c",
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
        return estimate if estimate is not None else limit

    def _sample(self, adapter, stats, relation, params, budget, total: int | None) -> pd.DataFrame | None:
        sql, random = adapter.sample_sql(relation, self.sample_rows, total, self.seed)
        try:
            df = adapter.execute(sql, params, budget.timeout(self.statement_timeout))
        except (StatsBudgetExceeded, StatementTimeout):
            raise
        except Exception as e:
            if not random:
                raise
            log_debug(f"[table_stats] sampling failed for {stats.table!r}, reading first rows: {e}")
            df = adapter.execute(f"SELECT * FROM {relation} LIMIT {self.sample_rows}", params, budget.timeout(self.statement_timeout))
        if random and df is not None and df.empty and (total or 0) > 0:
            # Block sampling can miss every block of a small or skewed table.
            df = adapter.execute(f"SELECT * FROM {relation} LIMIT {self.sample_rows}", params, budget.timeout(self.statement_timeout))
        if df is None:
            return None
        if random or total is not None:
            stats.sample_size = len(df)
        for col in stats.columns:
            # SQLite columns may be declared without a type.
            if col.type in (None, "", "NULL") and col.name in df.columns:
                col.type, col.kind = _kind_from_dtype(df[col.name].infer_objects().dtype)
        return df

    def _exact(self, adapter, stats, relation, params, budget, sample):
        exprs = ["COUNT(*) AS __n"]
        for i, col in enumerate(stats.columns):
            q = quote_identifier(col.name, adapter.dialect)
            exprs.append(f"COUNT({q}) AS __c{i}")
            if col.kind != "other":
                exprs.append(f"COUNT(DISTINCT {q}) AS __d{i}")
            if col.kind in ("numeric", "temporal"):
                exprs += [f"MIN({q}) AS __lo{i}", f"MAX({q}) AS __hi{i}"]
        # Chunked so very wide tables stay under engine limits on select-list size.
        row: dict[str, Any] = {}
        for start in range(0, len(exprs), 240):
            chunk = exprs[start:start + 240]
            if start:
                chunk = ["COUNT(*) AS __n", *chunk]
            df = adapter.execute(f"SELECT {', '.join(chunk)} FROM {relation}", params, budget.timeout(self.statement_timeout))
            # Per column, since a row-wise iloc upcasts mixed dtypes to float.
            row.update({str(k).lower(): df[k].iloc[0] for k in df.columns})
        total = int(row["__n"])
        stats.rows, stats.rows_exact = total, True
        for i, col in enumerate(stats.columns):
            nonnull = int(row[f"__c{i}"])
            col.nulls = (1 - nonnull / total) if total else 0.0
            if f"__d{i}" in row and row[f"__d{i}"] is not None:
                col.distinct = int(row[f"__d{i}"])
            if col.kind in ("numeric", "temporal"):
                col.min, col.max = _to_python(row.get(f"__lo{i}")), _to_python(row.get(f"__hi{i}"))
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
            sql = (
                f"SELECT {q} AS v, COUNT(*) AS n FROM {relation} WHERE {q} IS NOT NULL "
                f"GROUP BY {q} ORDER BY n DESC, v LIMIT {self.top_values}"
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
    if isinstance(count, float) and count < 1:
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
        rows = f"over {TableProfiler.param.exact_row_limit.default} rows" if stats.method == SAMPLED else None
    elif stats.rows_exact:
        rows = f"{stats.rows} rows"
    else:
        rows = f"~{stats.rows} rows"
    notes = [rows] if rows else []
    if stats.method == SAMPLED:
        notes.append(f"stats from a {stats.sample_size}-row sample")
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
    try:
        from platformdirs import user_cache_dir
    except ImportError:
        return None
    return Path(user_cache_dir("lumen")) / "table_stats"


class TableStatsStore(param.Parameterized):
    """
    Computes table statistics in the background and caches them in memory
    and, for sources with a stable identity, on disk.
    """

    budget_seconds = param.Number(default=300, allow_None=True, doc="""
        Wall time one source may spend on statistics per session. Once spent,
        further tables render names and types only.""")

    cache_dir = param.Parameter(default=None, doc="""
        Directory for persisted statistics; None disables persistence.
        Defaults to the user cache directory, or LUMEN_TABLE_STATS_CACHE.""")

    max_age = param.Number(default=7 * 24 * 3600, doc="""
        Seconds a persisted entry stays valid when the engine exposes no
        modification time for the table.""")

    concurrency = param.Integer(default=2, bounds=(1, None), doc="""
        Tables of one source profiled concurrently.""")

    def __init__(self, profiler: TableProfiler | None = None, **params):
        if "cache_dir" not in params:
            params["cache_dir"] = _default_cache_dir()
        super().__init__(**params)
        self.profiler = profiler or TableProfiler()
        self._memory: dict[tuple, TableStats] = {}
        self._tasks: dict[tuple, asyncio.Future] = {}
        # Keyed by database identity so sources derived from one another
        # share a single budget and concurrency limit.
        self._budgets: dict[str, _Budget] = {}
        self._slots: dict[str, threading.BoundedSemaphore] = {}
        self._lock = threading.Lock()

    def clear(self):
        self._memory.clear()
        self._tasks.clear()
        self._budgets.clear()

    def _key(self, source: Source, table: str) -> tuple:
        adapter = get_adapter(source)
        identity = adapter.identity()
        if identity is None:
            # Sources derived with create_sql_expr_source share a DuckDB
            # connection, so key by it and every derivation reuses the stats.
            conn = getattr(source, "_connection", None)
            identity = f"memory:{id(conn) if conn is not None else id(source)}"
        return (identity, table, adapter.definition(table))

    def _path(self, key: tuple) -> Path | None:
        if self.cache_dir is None or str(key[0]).startswith("memory:"):
            return None
        digest = hashlib.sha1(json.dumps(key, default=str).encode()).hexdigest()
        return Path(self.cache_dir) / f"{digest}.json"

    def _fingerprint(self, source: Source, table: str, stats: TableStats) -> str:
        adapter = get_adapter(source)
        token = adapter.modified(table)
        columns = [(c.name, c.type) for c in stats.columns]
        return hashlib.sha1(json.dumps([STATS_VERSION, columns, token], default=str).encode()).hexdigest()

    def get(self, source: Source, table: str) -> TableStats | None:
        """Cached statistics, if any, without computing anything."""
        key = self._key(source, table)
        if key in self._memory:
            return self._memory[key]
        path = self._path(key)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            stats = TableStats.from_dict(data)
        except Exception:
            return None
        if stats.version != STATS_VERSION:
            return None
        token = get_adapter(source).modified(table)
        if token is None and time.time() - stats.computed_at > self.max_age:
            return None
        if stats.fingerprint != self._fingerprint(source, table, stats):
            return None
        self._memory[key] = stats
        return stats

    def compute(self, source: Source, table: str) -> TableStats:
        """Compute (or load) statistics synchronously."""
        if (cached := self.get(source, table)) is not None:
            return cached
        key = self._key(source, table)
        with self._lock:
            budget = self._budgets.setdefault(key[0], _Budget(self.budget_seconds))
            slot = self._slots.setdefault(key[0], threading.BoundedSemaphore(self.concurrency))
        with slot:
            if key in self._memory:
                return self._memory[key]
            stats = self.profiler.profile(source, table, budget)
        stats.fingerprint = self._fingerprint(source, table, stats)
        self._memory[key] = stats
        path = self._path(key)
        if path is not None and stats.method != TYPES_ONLY:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(stats.to_dict(), default=str), encoding="utf-8")
                tmp.replace(path)
            except OSError as e:
                log_debug(f"[table_stats] could not persist stats for {table!r}: {e}")
        return stats

    def schedule(self, source: Source, tables: list[str]) -> list[asyncio.Future]:
        """Start computing statistics for `tables` in the background."""
        if not hasattr(source, "execute"):
            return []
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return []
        futures = []
        for table in tables:
            try:
                key = self._key(source, table)
            except Exception:
                continue
            if key in self._memory:
                continue
            task = self._tasks.get(key)
            if task is None or task.done() or task.get_loop() is not loop:
                task = loop.create_task(self._run(source, table, key))
                self._tasks[key] = task
            futures.append(task)
        return futures

    async def _run(self, source, table, key):
        try:
            return await asyncio.to_thread(self.compute, source, table)
        except Exception as e:
            log_debug(f"[table_stats] failed for {table!r}: {e}")
            return None
        finally:
            if self._tasks.get(key) is asyncio.current_task():
                self._tasks.pop(key, None)

    async def ensure(
        self, source: Source, tables: list[str], timeout: float | None = None
    ) -> dict[str, TableStats]:
        """
        Return statistics for `tables`, waiting at most `timeout` seconds for
        any still being computed; those keep computing in the background.
        """
        futures = self.schedule(source, tables)
        pending = [f for f in futures if not f.done()]
        if pending:
            await asyncio.wait(pending, timeout=timeout)
        result = {}
        for table in tables:
            try:
                key = self._key(source, table)
            except Exception:
                continue
            if key in self._memory:
                result[table] = self._memory[key]
        return result


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
