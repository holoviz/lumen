"""Tests for lumen.ai.table_stats: tiered statistics, caching and rendering."""
import asyncio
import gc
import json
import sqlite3
import sys
import threading
import time
import types
import warnings

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.schemas import Metaset, TableCatalogEntry, get_metaset
from lumen.ai.table_stats import (
    ESTIMATED, EXACT, SAMPLED, SCHEMA, TYPES_ONLY, BigQueryStatsAdapter,
    ColumnStats, DuckDBStatsAdapter, SnowflakeStatsAdapter,
    SQLAlchemyStatsAdapter, StatementTimeout, StatsAdapter, TableProfiler,
    TableStats, TableStatsStore, _format_count, _parse_pg_array,
    _quote_qualified, _run_abandonable, column_kind, format_number,
    format_value, render_column, render_stats_header,
)
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.sources.duckdb import DuckDBSource

try:
    from lumen.sources.sqlalchemy import SQLAlchemySource
except ImportError:
    SQLAlchemySource = None

SEP = SOURCE_TABLE_SEPARATOR
needs_sqlalchemy = pytest.mark.skipif(SQLAlchemySource is None, reason="sqlalchemy not installed")


@pytest.fixture
def duckdb_source():
    source = DuckDBSource(uri=":memory:")
    conn = source._connection
    conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name VARCHAR, region VARCHAR)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), amount DOUBLE, status VARCHAR, note VARCHAR)")
    conn.execute("INSERT INTO customers VALUES (1, 'Ada', 'east'), (2, 'Bo', 'east'), (3, 'Cy', 'west')")
    conn.execute("""
        INSERT INTO orders VALUES
        (10, 1, 12.5, 'paid', NULL), (11, 1, 7.25, 'paid', 'gift'),
        (12, 2, 100.0, 'cancelled', NULL), (13, 3, 0.001234, 'paid', 'late')
    """)
    source.tables = {"customers": "SELECT * FROM customers", "orders": "SELECT * FROM orders"}
    yield source
    source.close()


@pytest.fixture
def sqlite_path(tmp_path):
    path = tmp_path / "shop.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE "order" (order_id INTEGER PRIMARY KEY, account_id INTEGER REFERENCES account(account_id), k_symbol TEXT, day DATE)')
        conn.execute("CREATE TABLE account (account_id INTEGER PRIMARY KEY, frequency TEXT)")
        conn.executemany('INSERT INTO "order" VALUES (?, ?, ?, ?)', [
            (i, i % 5, ["SIPO", "", "UVER", None][i % 4], f"2020-01-{1 + i % 28:02d}") for i in range(1, 2001)
        ])
        conn.executemany("INSERT INTO account VALUES (?, ?)", [(i, "MONTHLY") for i in range(5)])
    return path


def _sqlite_source(path):
    return SQLAlchemySource(url=f"sqlite:///{path}", tables=["order", "account"])


# ---------------------------------------------------------------
# Profiling tiers
# ---------------------------------------------------------------

class TestExactTier:

    def test_duckdb_types_keys_and_ranges(self, duckdb_source):
        stats = TableProfiler().profile(duckdb_source, "orders")
        assert stats.method == EXACT
        assert (stats.rows, stats.rows_exact) == (4, True)
        by_name = {c.name: c for c in stats.columns}
        assert by_name["id"].type == "INTEGER" and by_name["id"].primary_key
        assert by_name["customer_id"].references == "customers.id"
        assert (by_name["amount"].min, by_name["amount"].max) == (0.001234, 100.0)
        assert by_name["status"].values == [["paid", 3], ["cancelled", 1]]
        assert by_name["status"].values_complete
        assert by_name["note"].nulls == 0.5

    def test_sqlite_reserved_table_name_and_inspector_keys(self, sqlite_path):
        pytest.importorskip("sqlalchemy")
        stats = TableProfiler().profile(_sqlite_source(sqlite_path), "order")
        assert stats.method == EXACT and stats.rows == 2000
        by_name = {c.name: c for c in stats.columns}
        assert by_name["order_id"].primary_key
        assert by_name["account_id"].references == "account.account_id"
        assert by_name["day"].kind == "temporal"
        assert (by_name["day"].min, by_name["day"].max) == ("2020-01-01", "2020-01-28")
        # Empty strings are real values and must survive as such.
        assert ["", 500] in by_name["k_symbol"].values
        assert by_name["k_symbol"].nulls == 0.25

    def test_integer_ranges_stay_integers(self, duckdb_source):
        stats = TableProfiler().profile(duckdb_source, "orders")
        col = stats.column("customer_id")
        assert (col.min, col.max) == (1, 3)
        assert isinstance(col.min, int)


class TestSampledTier:

    @pytest.fixture
    def big_source(self):
        source = DuckDBSource(uri=":memory:")
        source._connection.execute(
            "CREATE TABLE big AS SELECT i AS id, CASE WHEN i % 10 = 0 THEN 'rare' ELSE 'common' END AS kind "
            "FROM range(50000) t(i)"
        )
        source.tables = {"big": "SELECT * FROM big"}
        yield source
        source.close()

    def test_large_table_is_sampled_and_labelled(self, big_source):
        profiler = TableProfiler(exact_row_limit=1000, sample_rows=500)
        stats = profiler.profile(big_source, "big")
        assert stats.method == SAMPLED
        assert (stats.rows, stats.rows_exact) == (50000, True)
        assert 0 < stats.sample_size <= 500
        assert "stats from a" in render_stats_header(stats)
        # Local engines recount low-cardinality columns exactly after sampling.
        assert stats.column("kind").values == [["common", 45000], ["rare", 5000]]

    def test_sample_is_deterministic(self, big_source):
        profiler = TableProfiler(exact_row_limit=1000, sample_rows=500)
        first = profiler.profile(big_source, "big")
        second = profiler.profile(big_source, "big")
        assert first.column("id").min == second.column("id").min
        assert first.sample_size == second.sample_size

    def test_sqlite_hash_sample_is_deterministic(self, sqlite_path):
        pytest.importorskip("sqlalchemy")
        source = _sqlite_source(sqlite_path)
        profiler = TableProfiler(exact_row_limit=100, sample_rows=50)
        first = profiler.profile(source, "order")
        second = profiler.profile(source, "order")
        assert first.method == SAMPLED
        assert first.to_dict() | {"computed_at": 0} == second.to_dict() | {"computed_at": 0}


class TestLimits:

    def test_exhausted_budget_keeps_names_and_types(self, duckdb_source):
        store = TableStatsStore(cache_dir=None, budget_seconds=0)
        stats = store.compute(duckdb_source, "orders")
        assert stats.method == TYPES_ONLY
        assert [c.type for c in stats.columns] == ["INTEGER", "INTEGER", "DOUBLE", "VARCHAR", "VARCHAR"]
        assert stats.column("id").primary_key

    def test_sqlite_statement_timeout_aborts_query(self, sqlite_path):
        pytest.importorskip("sqlalchemy")
        adapter = SQLAlchemyStatsAdapter(_sqlite_source(sqlite_path))
        slow = (
            "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r) "
            "SELECT COUNT(*) FROM r"
        )
        start = time.monotonic()
        with pytest.raises(StatementTimeout):
            adapter.execute(slow, None, timeout=0.2)
        assert time.monotonic() - start < 5


# ---------------------------------------------------------------
# Engine metadata (mocked engines)
# ---------------------------------------------------------------

def test_parse_pg_array():
    assert _parse_pg_array('{a,"b c",NULL,"NULL"}') == ["a", "b c", "NULL"]
    assert _parse_pg_array('{"x \\"y\\""}') == ['x "y"']
    assert _parse_pg_array(None) == []


class _FakePostgres:
    dialect = "postgresql"
    tables = ["public.events"]
    schema = None
    table_params = {}
    _inspector = None


def test_postgres_pg_stats_fill_estimates():
    adapter = SQLAlchemyStatsAdapter.__new__(SQLAlchemyStatsAdapter)
    adapter.source, adapter.dialect, adapter.local = _FakePostgres(), "postgresql", False
    pg_stats = pd.DataFrame([
        {"attname": "kind", "null_frac": 0.1, "n_distinct": 3.0, "mcv": "{click,view,buy}", "mcf": "{0.6,0.3,0.1}", "hist": None},
        {"attname": "value", "null_frac": 0.0, "n_distinct": -0.5, "mcv": None, "mcf": None, "hist": "{1,5,90}"},
    ])
    columns = [ColumnStats("kind", "text", "string"), ColumnStats("value", "integer", "numeric")]
    with patch.object(SQLAlchemyStatsAdapter, "execute", return_value=pg_stats):
        assert adapter.metadata_stats("public.events", columns, rows=1000, timeout=5)
    kind, value = columns
    assert kind.nulls == 0.1 and kind.values_complete
    assert kind.values == [["click", 0.6], ["view", 0.3], ["buy", 0.1]]
    assert (value.min, value.max, value.distinct) == (1, 90, 500)


class _FakeSnowflake:
    dialect = "snowflake"
    tables = ["EVENTS"]
    table_params = {}
    account = "acct"


def test_snowflake_metadata_served_aggregates():
    adapter = SnowflakeStatsAdapter(_FakeSnowflake())
    result = pd.DataFrame([{"N": 200, "N0": 150, "LO0": 3, "HI0": 9}])
    columns = [ColumnStats("VALUE", "NUMBER", "numeric")]
    with patch.object(SnowflakeStatsAdapter, "execute", return_value=result) as execute:
        assert adapter.metadata_stats("EVENTS", columns, rows=200, timeout=5)
    sql = execute.call_args[0][0]
    # Only aggregates Snowflake answers from micro-partition metadata.
    assert "DISTINCT" not in sql and "GROUP BY" not in sql
    assert (columns[0].nulls, columns[0].min, columns[0].max) == (0.25, 3, 9)
    sample, random = adapter.sample_sql('"EVENTS"', 100, 10_000_000, 42)
    assert "SAMPLE SYSTEM" in sample and "SEED (42)" in sample and random


# ---------------------------------------------------------------
# Store: caching, persistence, background computation
# ---------------------------------------------------------------

class TestStore:

    def test_file_backed_stats_persist_and_invalidate(self, sqlite_path, tmp_path):
        pytest.importorskip("sqlalchemy")
        cache = tmp_path / "cache"
        TableStatsStore(cache_dir=cache).compute(_sqlite_source(sqlite_path), "account")
        [entry] = list(cache.glob("*.json"))
        assert json.loads(entry.read_text())["rows"] == 5

        fresh = TableStatsStore(cache_dir=cache)
        # Disk is only read off the event loop, by compute.
        assert fresh.get(_sqlite_source(sqlite_path), "account") is None
        with patch.object(TableProfiler, "profile", side_effect=AssertionError("recomputed")):
            assert fresh.compute(_sqlite_source(sqlite_path), "account").rows == 5

        with sqlite3.connect(sqlite_path) as conn:
            conn.execute("INSERT INTO account VALUES (99, 'WEEKLY')")
        assert TableStatsStore(cache_dir=cache).compute(_sqlite_source(sqlite_path), "account").rows == 6
        # The in-memory entry is revalidated too, since SQLite is local.
        assert fresh.get(_sqlite_source(sqlite_path), "account") is None

    def test_persisted_entry_rejected_when_columns_change(self, sqlite_path, tmp_path):
        pytest.importorskip("sqlalchemy")
        cache = tmp_path / "cache"
        TableStatsStore(cache_dir=cache).compute(_sqlite_source(sqlite_path), "account")
        store = TableStatsStore(cache_dir=cache)
        with (
            patch.object(SQLAlchemyStatsAdapter, "modified", return_value="unchanged"),
            patch.object(SQLAlchemyStatsAdapter, "columns", return_value=[("account_id", "INTEGER")]),
            patch.object(TableProfiler, "profile", side_effect=AssertionError("recomputed")),
            pytest.raises(AssertionError, match="recomputed"),
        ):
            store.compute(_sqlite_source(sqlite_path), "account")

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows ignores POSIX mode bits; the profile ACL applies")
    def test_persisted_files_are_private(self, sqlite_path, tmp_path):
        pytest.importorskip("sqlalchemy")
        cache = tmp_path / "cache"
        TableStatsStore(cache_dir=cache).compute(_sqlite_source(sqlite_path), "account")
        [entry] = list(cache.glob("*.json"))
        assert entry.stat().st_mode & 0o077 == 0
        assert cache.stat().st_mode & 0o077 == 0

    def test_in_memory_sources_are_not_persisted(self, duckdb_source, tmp_path):
        TableStatsStore(cache_dir=tmp_path).compute(duckdb_source, "orders")
        assert not list(tmp_path.glob("*.json"))

    def test_derived_sources_share_cached_stats(self, duckdb_source):
        store = TableStatsStore(cache_dir=None)
        stats = store.compute(duckdb_source, "orders")
        derived = duckdb_source.create_sql_expr_source({"paid": "SELECT * FROM orders WHERE status = 'paid'"})
        assert store.get(derived, "orders") is stats

    async def test_ensure_returns_ready_and_keeps_computing(self, duckdb_source, table_stats_store):
        slow = TableProfiler()
        original = slow.profile

        def delayed(*args, **kwargs):
            time.sleep(0.5)
            return original(*args, **kwargs)

        table_stats_store.profiler = slow
        with patch.object(slow, "profile", side_effect=delayed):
            assert await table_stats_store.ensure(duckdb_source, ["orders"], timeout=0.01) == {}
            result = await table_stats_store.ensure(duckdb_source, ["orders"], timeout=5)
        assert result["orders"].rows == 4

    async def test_schedule_without_loop_is_noop(self, duckdb_source):
        assert await asyncio.to_thread(TableStatsStore(cache_dir=None).schedule, duckdb_source, ["orders"]) == []


# ---------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------

class TestRendering:

    @pytest.mark.parametrize("value, expected", [
        (3, "3"), (3.0, "3"), (0.001234, "0.001234"), (0.00176056338, "0.00176056"),
        (1234567.891, "1234567.891"), (1e-7, "1e-07"), (-124.28481, "-124.28481"),
    ])
    def test_format_number(self, value, expected):
        assert format_number(value) == expected

    @pytest.mark.parametrize("value, expected", [
        ("N", "N"), ("Unified School District", "Unified School District"),
        ("", "''"), ("a,b", "'a,b'"), ("O'Hare", "'O''Hare'"), (" x", "' x'"),
        ("NULL", "'NULL'"), (None, "NULL"),
    ])
    def test_format_value_quotes_only_when_ambiguous(self, value, expected):
        assert format_value(value) == expected

    def test_render_column_variants(self):
        stats = TableStats("t", rows=100, rows_exact=True, method=EXACT)
        enum = ColumnStats("Virtual", "TEXT", "string", nulls=0.39,
                           values=[["N", 50], ["P", 10], ["F", 1]], values_complete=True, distinct=3)
        assert render_column(enum, stats) == "Virtual TEXT {N:50, P:10, F:1} nulls 39%"
        ranged = ColumnStats("Avg Math", "INTEGER", "numeric", nulls=0.0, min=289, max=699, distinct=40)
        assert render_column(ranged, stats) == '"Avg Math" INTEGER 289..699'
        key = ColumnStats("id", "INTEGER", "numeric", nulls=0.0, min=1, max=100, distinct=100, primary_key=True)
        assert render_column(key, stats) == "id INTEGER PK 1..100"
        text = ColumnStats("code", "TEXT", "string", nulls=0.0, distinct=100, examples=["0110017"])
        assert render_column(text, stats, "School code") == "code TEXT unique e.g. '0110017' -- School code"
        empty = ColumnStats("gone", "TEXT", "string", nulls=1.0)
        assert render_column(empty, stats) == "gone TEXT all NULL"
        assert render_column(enum, stats, detail="types") == "Virtual TEXT"

    def test_sampled_values_render_as_shares(self):
        stats = TableStats("t", rows=None, rows_at_least=1_000_000, method=SAMPLED, sample_size=10000)
        col = ColumnStats("type", "TEXT", "string", values=[["VYDAJ", 0.6254], ["VYBER", 0.004]])
        assert render_column(col, stats) == "type TEXT {VYDAJ:63%, VYBER:<1%, ...}"
        assert render_stats_header(stats) == "over 1000000 rows; stats from a 10000-row sample"

    def test_full_sampled_share_renders_as_percent(self):
        assert _format_count(1.0) == ":100%"
        assert _format_count(5) == ":5"

    def test_first_rows_and_lower_bound_headers(self):
        stats = TableStats("t", rows=None, rows_at_least=500, method=SAMPLED, sample_size=50, sample_random=False)
        assert render_stats_header(stats) == "over 500 rows; stats from the first 50 rows"

    def test_estimated_header(self):
        assert render_stats_header(TableStats("t", rows=5_000_000, method=ESTIMATED)) == "~5000000 rows; engine-estimated stats"

    def test_column_kind(self):
        assert [column_kind(t) for t in ("VARCHAR(20)", "DECIMAL(10,2)", "TIMESTAMP WITH TIME ZONE", "BOOLEAN", "BLOB", "INTEGER[]", None)] == [
            "string", "numeric", "temporal", "boolean", "other", "other", "other"
        ]

    def test_legacy_schema_adapter(self):
        stats = TableStats.from_json_schema("t", {
            "__len__": 3, "x": {"type": "integer", "inclusiveMinimum": 1, "inclusiveMaximum": 3},
            "c": {"type": "string", "enum": ["a", "b", "..."]},
        })
        assert stats.method == SCHEMA and stats.rows == 3
        assert render_column(stats.column("c"), stats) == "c VARCHAR {a, b, ...}"


# ---------------------------------------------------------------
# Metaset integration
# ---------------------------------------------------------------

class TestMetasetIntegration:

    async def test_get_metaset_renders_stats_deterministically(self, duckdb_source):
        first = (await get_metaset([duckdb_source], ["customers", "orders"])).compact_context()
        second = (await get_metaset([duckdb_source], ["customers", "orders"])).compact_context()
        assert first == second
        assert "orders (4 rows)\n  id INTEGER PK 10..13\n" in first
        assert "  customer_id INTEGER -> customers.id 1..3" in first
        assert "  status VARCHAR {paid:3, cancelled:1}" in first
        assert ".nan" not in first

    async def test_token_budget_degrades_to_types_then_names(self, duckdb_source):
        metaset = await get_metaset([duckdb_source], ["customers", "orders"])
        full = metaset.compact_context()
        tight = metaset.compact_context(max_tokens=60)
        assert len(tight) < len(full)
        assert "load_table_schemas" in tight
        # Every table still appears in some form.
        assert "customers" in tight and "orders" in tight

    async def test_ensure_stats_fills_catalog_metaset(self, duckdb_source):
        slug = f"{duckdb_source.name}{SEP}orders"
        metaset = Metaset(query=None, catalog={slug: TableCatalogEntry(slug, 1, [], source=duckdb_source)})
        assert metaset.compact_context() == "orders"
        await metaset.ensure_stats()
        assert "status VARCHAR {paid:3, cancelled:1}" in metaset.compact_context()


# ---------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------

class TestEphemeralIdentity:

    def test_freed_in_memory_sources_never_share_stats(self):
        store = TableStatsStore(cache_dir=None)
        seen = set()
        for i in range(8):
            source = DuckDBSource.from_df(tables={"data": pd.DataFrame({"secret": [f"user{i}"] * 3})})
            values = store.compute(source, "data").column("secret").values
            assert values == [[f"user{i}", 3]]
            seen.add(store._identity(source, store._adapter(source)))
            source.close()
            del source
            gc.collect()
        assert len(seen) == 8
        # Entries of collected connections are dropped with them.
        assert not store._memory and not store._budgets

    def test_replaced_table_is_profiled_again(self, duckdb_source):
        store = TableStatsStore(cache_dir=None)
        assert store.compute(duckdb_source, "orders").rows == 4
        duckdb_source._connection.execute("CREATE OR REPLACE TABLE orders AS SELECT 1 AS id, 'paid' AS status")
        assert store.get(duckdb_source, "orders") is None
        assert store.compute(duckdb_source, "orders").rows == 1

    def test_derived_expression_tracks_parent_table(self, duckdb_source):
        store = TableStatsStore(cache_dir=None)
        derived = duckdb_source.create_sql_expr_source({"paid": "SELECT * FROM orders WHERE status = 'paid'"})
        assert store.compute(derived, "paid").rows == 3
        duckdb_source._connection.execute("INSERT INTO orders VALUES (14, 2, 1.0, 'paid', NULL)")
        assert store.get(derived, "paid") is None


class TestBackgroundScheduling:

    async def test_queued_tables_do_not_hold_default_executor_threads(self):
        source = DuckDBSource(uri=":memory:")
        tables = [f"t{i}" for i in range(40)]
        for t in tables:
            source._connection.execute(f"CREATE TABLE {t} AS SELECT 1 AS x")
        source.tables = tables
        store = TableStatsStore(cache_dir=None, concurrency=2)
        running, peak = 0, 0
        lock = threading.Lock()
        original = store.profiler.profile

        def slow(*args, **kwargs):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.2)
            with lock:
                running -= 1
            return original(*args, **kwargs)

        with patch.object(store.profiler, "profile", side_effect=slow):
            futures = store.schedule(source, tables)
            start = time.monotonic()
            await asyncio.to_thread(lambda: None)
            assert time.monotonic() - start < 0.5
            await asyncio.wait(futures, timeout=30)
        assert peak <= 2
        source.close()


class TestStatementTimeouts:

    def test_queued_statement_is_cancelled_not_run_late(self):
        ran = threading.Event()
        release = threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)
        pool.submit(release.wait)
        try:
            with patch("lumen.ai.table_stats._EXECUTOR", pool), pytest.raises(StatementTimeout):
                _run_abandonable(ran.set, 0.1)
        finally:
            release.set()
            pool.shutdown(wait=True)
        assert not ran.is_set()

    def test_duckdb_statement_is_interrupted(self, duckdb_source):
        adapter = DuckDBStatsAdapter(duckdb_source)
        start = time.monotonic()
        with pytest.raises(StatementTimeout):
            adapter.execute(
                "SELECT COUNT(*) FROM range(100000000) a, range(100000) b WHERE a.range + b.range < 0",
                None, timeout=0.2,
            )
        assert time.monotonic() - start < 5
        # The connection stays usable.
        assert adapter.execute("SELECT 1 AS one", None, 1)["one"].iloc[0] == 1

    def test_postgres_timeout_uses_set_local_and_is_recognised(self):
        executed = []

        class QueryCanceled(Exception):
            pgcode = "57014"

        conn = MagicMock()
        conn.exec_driver_sql.side_effect = executed.append

        def run(stmt, params):
            executed.append(str(stmt))
            error = Exception("canceling statement due to statement timeout")
            error.orig = QueryCanceled()
            raise error

        conn.execute.side_effect = run
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        adapter = SQLAlchemyStatsAdapter.__new__(SQLAlchemyStatsAdapter)
        adapter.source = types.SimpleNamespace(_engine=engine, _driver_is_async=False)
        adapter.dialect, adapter.local = "postgresql", False
        with pytest.raises(StatementTimeout):
            adapter.execute('SELECT COUNT("a:b") AS n FROM t', None, timeout=5)
        assert executed[0] == "SET LOCAL statement_timeout = 5000"
        assert not any("RESET" in sql for sql in executed)
        # The colon in a quoted identifier is not a bind parameter.
        assert executed[1] == 'SELECT COUNT("a:b") AS n FROM t'


class _FakeSnowflakeConn:

    user = "ANALYST"
    role = "REPORTING"

    def __init__(self):
        self.cursors = []

    def cursor(self):
        cursor = MagicMock()
        cursor.fetch_pandas_all.return_value = pd.DataFrame({"N": [1]})
        self.cursors.append(cursor)
        return cursor


def test_snowflake_uses_own_cursor_with_server_timeout():
    source = _FakeSnowflake()
    source._conn = _FakeSnowflakeConn()
    source._cursor = MagicMock()
    adapter = SnowflakeStatsAdapter(source)
    assert adapter.execute("SELECT 1", None, timeout=4.5)["N"].iloc[0] == 1
    [cursor] = source._conn.cursors
    cursor.execute.assert_called_once_with("SELECT 1", timeout=5)
    cursor.close.assert_called_once()
    source._cursor.execute.assert_not_called()


def test_snowflake_identity_includes_user_and_role():
    source = _FakeSnowflake()
    source._conn = _FakeSnowflakeConn()
    identity = SnowflakeStatsAdapter(source).identity()
    assert "ANALYST" in identity and "REPORTING" in identity


class _QueryJobConfig:
    maximum_bytes_billed = None
    job_timeout_ms = None

    def __init__(self, query_parameters=None, **kwargs):
        self.query_parameters = query_parameters
        self.__dict__.update(kwargs)


class _FakeBigQuery:
    dialect = "bigquery"
    project_id = "my-proj"
    table_params = {}

    def __init__(self, tables):
        self.tables = tables
        self.calls = []

    def _build_query_config(self, params):
        return _QueryJobConfig(query_parameters=params)

    def execute(self, sql, params=None, **kwargs):
        self.calls.append((sql, params, kwargs))
        return pd.DataFrame()

    def get_sql_expr(self, table):
        return self.tables[table] if isinstance(self.tables, dict) else f"SELECT * FROM {table}"


@pytest.fixture
def fake_bigquery_module():
    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    bigquery = types.ModuleType("google.cloud.bigquery")
    bigquery.QueryJobConfig = _QueryJobConfig
    google.cloud, cloud.bigquery = cloud, bigquery
    with patch.dict(sys.modules, {"google": google, "google.cloud": cloud, "google.cloud.bigquery": bigquery}):
        yield


class TestBigQuery:

    def test_caps_survive_bound_parameters(self, fake_bigquery_module):
        source = _FakeBigQuery(["my-proj.ds.events"])
        adapter = BigQueryStatsAdapter(source, max_bytes_billed=1000)
        adapter.execute("SELECT @x", {"x": 1}, timeout=2)
        _, params, kwargs = source.calls[0]
        config = kwargs["job_config"]
        assert params is None and config.query_parameters == {"x": 1}
        assert (config.maximum_bytes_billed, config.job_timeout_ms) == (1000, 2000)

    def test_hyphenated_project_is_a_bare_table(self):
        adapter = BigQueryStatsAdapter(_FakeBigQuery(["my-proj.ds.events"]))
        assert adapter.bare_name("my-proj.ds.events") == "my-proj.ds.events"
        relation = adapter.relation("my-proj.ds.events")
        assert relation == "`my-proj`.ds.events"
        sql, random = adapter.sample_sql(relation, 100, 10_000_000, 42)
        assert "TABLESAMPLE SYSTEM" in sql and random

    def test_expression_tables_are_never_scanned_for_a_sample(self):
        source = _FakeBigQuery({"recent": "SELECT * FROM `my-proj.ds.events` WHERE day > '2024-01-01'"})
        adapter = BigQueryStatsAdapter(source)
        assert adapter.sample_sql(adapter.relation("recent"), 100, 10_000_000, 42) == (None, False)

    def test_empty_sample_does_not_fall_back_to_first_rows(self, fake_bigquery_module):
        source = _FakeBigQuery(["my-proj.ds.events"])
        adapter = BigQueryStatsAdapter(source)
        stats = TableStats("my-proj.ds.events", columns=[ColumnStats("x", "INT64", "numeric")])
        TableProfiler(sample_rows=100)._sample(
            adapter, stats, adapter.relation("my-proj.ds.events"), None, MagicMock(timeout=lambda t: t), 10_000_000,
        )
        assert [sql for sql, *_ in source.calls] == [
            "SELECT * FROM `my-proj`.ds.events TABLESAMPLE SYSTEM (0.0015 PERCENT) LIMIT 100"
        ]


class TestQuoting:

    @pytest.mark.parametrize("name, dialect, expected", [
        ("mydb.public.orders", "snowflake", "mydb.public.orders"),
        ("Users", "postgresql", "Users"),
        ("order", "postgresql", '"order"'),
        ("Order", "postgresql", '"order"'),
        ("order", "snowflake", '"ORDER"'),
        ("my-proj.ds.order", "bigquery", "`my-proj`.ds.`order`"),
        ('"Mixed Case".t', "duckdb", '"Mixed Case".t'),
    ])
    def test_quotes_only_what_needs_it(self, name, dialect, expected):
        assert _quote_qualified(name, dialect) == expected

    @pytest.mark.parametrize("dialect, expected", [
        ("mssql", "SELECT TOP 5 * FROM t"),
        ("oracle", "SELECT * FROM t FETCH FIRST 5 ROWS ONLY"),
        ("duckdb", "SELECT * FROM t LIMIT 5"),
    ])
    def test_row_limit_per_dialect(self, dialect, expected):
        adapter = StatsAdapter(types.SimpleNamespace(dialect=dialect))
        assert adapter.limit("SELECT * FROM t", 5) == expected


class TestSampleLabels:

    def test_first_rows_of_an_expression_are_labelled(self, sqlite_path):
        pytest.importorskip("sqlalchemy")
        source = SQLAlchemySource(url=f"sqlite:///{sqlite_path}", tables={"o": 'SELECT order_id, k_symbol FROM "order"'})
        stats = TableProfiler(exact_row_limit=100, sample_rows=50).profile(source, "o")
        assert stats.method == SAMPLED and not stats.sample_random
        assert render_stats_header(stats).endswith("stats from the first 50 rows")

    def test_parquet_row_count_is_exact(self, tmp_path):
        path = tmp_path / "data.parquet"
        pd.DataFrame({"x": range(10)}).to_parquet(path)
        source = DuckDBSource(uri=":memory:", tables={"data": f"SELECT * FROM read_parquet('{path}')"})
        stats = TableProfiler().profile(source, "data")
        assert (stats.rows, stats.rows_exact) == (10, True)
        assert render_stats_header(stats) == "10 rows"
        source.close()


class TestStoreLifetime:

    def test_budget_resets_after_window(self, duckdb_source):
        store = TableStatsStore(cache_dir=None, budget_seconds=0, budget_window=0.05)
        assert store.compute(duckdb_source, "orders").method == TYPES_ONLY
        store.budget_seconds = 300
        time.sleep(0.1)
        # Types-only results are retried, and the new window has budget.
        store.retry_after = 0
        assert store.compute(duckdb_source, "orders").method == EXACT

    def test_types_only_results_are_retried(self, duckdb_source):
        store = TableStatsStore(cache_dir=None, budget_seconds=0, retry_after=3600)
        assert store.compute(duckdb_source, "orders").method == TYPES_ONLY
        store.budget_seconds = None
        store._budgets.clear()
        assert store.compute(duckdb_source, "orders").method == TYPES_ONLY
        store.retry_after = 0
        assert store.compute(duckdb_source, "orders").method == EXACT

    def test_memory_is_bounded(self, duckdb_source):
        store = TableStatsStore(cache_dir=None, max_entries=1)
        store.compute(duckdb_source, "orders")
        store.compute(duckdb_source, "customers")
        assert len(store._memory) == 1
        assert store.get(duckdb_source, "customers") is not None


class TestMetasetFixes:

    async def test_get_metaset_wait_is_bounded(self, duckdb_source, table_stats_store):
        original = table_stats_store.profiler.profile

        def slow(*args, **kwargs):
            time.sleep(1)
            return original(*args, **kwargs)

        with patch.object(table_stats_store.profiler, "profile", side_effect=slow):
            start = time.monotonic()
            metaset = await get_metaset([duckdb_source], ["orders"], stats_timeout=0.05)
            assert time.monotonic() - start < 0.9
        assert not metaset.stats

    async def test_non_sql_sources_get_schema_stats(self):
        from lumen.sources.base import InMemorySource
        source = InMemorySource(tables={"t": pd.DataFrame({"x": [1, 2, 3], "c": ["a", "b", "a"]})})
        slug = f"{source.name}{SEP}t"
        metaset = Metaset(query=None, catalog={slug: TableCatalogEntry(slug, 1, [], source=source)})
        await metaset.ensure_stats()
        context = metaset.compact_context()
        assert "x INTEGER 1..3" in context
        assert "load_table_schemas" not in context

    async def test_legacy_schema_methods_warn(self, duckdb_source):
        slug = f"{duckdb_source.name}{SEP}orders"
        metaset = Metaset(query=None, catalog={slug: TableCatalogEntry(slug, 1, [], source=duckdb_source)})
        with pytest.warns(DeprecationWarning, match="ensure_schemas"):
            await metaset.ensure_schemas()
        assert slug in metaset.stats
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(DeprecationWarning, match="get_schema"):
                await metaset.get_schema(slug)

    async def test_less_relevant_tables_never_get_more_detail(self, duckdb_source):
        metaset = await get_metaset([duckdb_source], ["orders", "customers"])
        full_orders = metaset._render_table(
            f"{duckdb_source.name}{SEP}orders", "orders", metaset.catalog[f"{duckdb_source.name}{SEP}orders"],
            "full", True, False, True, True, 0,
        )
        from lumen.ai.utils import count_tokens
        tight = metaset.compact_context(max_tokens=count_tokens(full_orders) - 1, schema_tables=[
            f"{duckdb_source.name}{SEP}orders", f"{duckdb_source.name}{SEP}customers",
        ])
        customers = tight.split("customers", 1)[1]
        assert "{" not in customers.split("\n\n", 1)[0]


# ---------------------------------------------------------------
# Second review
# ---------------------------------------------------------------

class TestChangeTokens:

    def test_update_in_file_backed_duckdb_invalidates(self, tmp_path):
        source = DuckDBSource(uri=str(tmp_path / "db.duckdb"), read_only=False)
        source._connection.execute("CREATE TABLE t AS SELECT range AS x FROM range(10)")
        source._connection.execute("CHECKPOINT")
        source.tables = ["t"]
        store = TableStatsStore(cache_dir=tmp_path / "cache")
        assert store.compute(source, "t").column("x").max == 9
        source._connection.execute("UPDATE t SET x = 100 WHERE x = 9")
        assert store.get(source, "t") is None
        assert store.compute(source, "t").column("x").max == 100
        source.close()

    def test_update_in_sqlite_wal_mode_invalidates(self, sqlite_path):
        pytest.importorskip("sqlalchemy")
        with sqlite3.connect(sqlite_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
        # An open connection keeps the WAL from being checkpointed away.
        writer = sqlite3.connect(sqlite_path)
        try:
            store = TableStatsStore(cache_dir=None)
            source = _sqlite_source(sqlite_path)
            assert store.compute(source, "account").column("frequency").values == [["MONTHLY", 5]]
            writer.execute("UPDATE account SET frequency = 'WEEKLY'")
            writer.commit()
            assert store.get(source, "account") is None
        finally:
            writer.close()


def test_parquet_footer_stats_only_for_whole_file(tmp_path):
    path = tmp_path / "data.parquet"
    pd.DataFrame({"x": range(5000)}).to_parquet(path)
    source = DuckDBSource(uri=":memory:", tables={
        "whole": f"SELECT * FROM read_parquet('{path}')",
        "small": f"SELECT * FROM read_parquet('{path}') WHERE x < 100",
    })
    assert DuckDBStatsAdapter(source)._parquet_path("small") is None
    stats = TableProfiler().profile(source, "small")
    assert (stats.rows, stats.column("x").min, stats.column("x").max) == (100, 0, 99)
    assert DuckDBStatsAdapter(source).exact_row_count("whole") == 5000
    source.close()


async def test_in_memory_sqlite_is_profiled_on_its_own_thread():
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import text
    source = SQLAlchemySource(url="sqlite:///:memory:")
    with source._engine.connect() as conn:
        conn.execute(text("CREATE TABLE t (x INTEGER)"))
        conn.execute(text("INSERT INTO t VALUES (1), (2), (3)"))
        conn.commit()
    source.tables = ["t"]
    assert SQLAlchemyStatsAdapter(source).thread_bound
    store = TableStatsStore(cache_dir=None)
    result = await store.ensure(source, ["t"], timeout=5)
    assert (result["t"].rows, result["t"].column("x").max) == (3, 3)


def test_undescribed_tables_are_not_cached(duckdb_source):
    store = TableStatsStore(cache_dir=None)
    with patch.object(DuckDBStatsAdapter, "columns", side_effect=RuntimeError("no such table")):
        assert store.compute(duckdb_source, "orders").columns == []
    assert store.compute(duckdb_source, "orders").rows == 4


class _Column:

    def __init__(self, name, type_code):
        self.name, self.type_code = name, type_code


def test_snowflake_columns_carry_types():
    constants = types.ModuleType("snowflake.connector.constants")
    constants.FIELD_ID_TO_NAME = {0: "FIXED", 2: "TEXT", 8: "TIMESTAMP_NTZ"}
    connector = types.ModuleType("snowflake.connector")
    snowflake = types.ModuleType("snowflake")
    snowflake.connector, connector.constants = connector, constants
    source = _FakeSnowflake()
    source._conn = _FakeSnowflakeConn()
    modules = {"snowflake": snowflake, "snowflake.connector": connector, "snowflake.connector.constants": constants}
    with patch.dict(sys.modules, modules), patch.object(
        _FakeSnowflakeConn, "cursor", lambda self: MagicMock(describe=lambda sql: [_Column("V", 0), _Column("S", 2), _Column("T", 8)])
    ):
        columns = SnowflakeStatsAdapter(source).columns("EVENTS", '"EVENTS"', 5)
    assert columns == [("V", "NUMBER"), ("S", "VARCHAR"), ("T", "TIMESTAMP_NTZ")]
    assert [column_kind(t) for _, t in columns] == ["numeric", "string", "temporal"]


class _UntypedSnowflake(SnowflakeStatsAdapter):
    """Snowflake whose columns came back without types, so metadata supplies no ranges."""

    metadata_count = False

    def metadata_stats(self, table, columns, rows, timeout):
        return True

    def sample_sql(self, relation, rows, total, seed):
        return "sample", True

    def execute(self, sql, params=None, timeout=None):
        if "COUNT" in sql:
            return pd.DataFrame({"n": [10_000_000]})
        return pd.DataFrame({"v": [2175, 8126]})


def test_sample_ranges_are_never_labelled_exact():
    from lumen.ai.table_stats import _Budget
    adapter = _UntypedSnowflake(_FakeSnowflake())
    stats = TableStats("t", columns=[ColumnStats("v")])
    TableProfiler()._compute(adapter, stats, "t", None, _Budget(None))
    assert stats.method == ESTIMATED and stats.column("v").min == 2175
    assert not stats.ranges_exact
    assert "exact ranges" not in render_stats_header(stats)


def test_unaggregatable_column_keeps_the_rest(duckdb_source):
    adapter = DuckDBStatsAdapter(duckdb_source)
    original = DuckDBStatsAdapter.execute

    def execute(self, sql, params=None, timeout=None):
        if 'MIN(amount)' in sql or 'MIN("amount")' in sql:
            raise RuntimeError("cannot aggregate")
        return original(self, sql, params, timeout)

    with patch.object(DuckDBStatsAdapter, "execute", execute):
        stats = TableProfiler().profile(duckdb_source, "orders")
    assert stats.method == EXACT and stats.rows == 4
    assert (stats.column("customer_id").min, stats.column("customer_id").max) == (1, 3)
    amount = stats.column("amount")
    assert amount.min is None and amount.nulls == 0.0
    assert adapter.dialect == duckdb_source.dialect


# ---------------------------------------------------------------
# Third review
# ---------------------------------------------------------------

def _field(name, field_type, mode="NULLABLE"):
    return types.SimpleNamespace(name=name, field_type=field_type, mode=mode)


class TestBigQueryMetadata:

    @pytest.fixture
    def source(self):
        source = _FakeBigQuery({"events": "my-proj.ds.events", "recent": "SELECT * FROM `my-proj.ds.events` WHERE day > '2024-01-01'"})
        table = types.SimpleNamespace(
            modified="2026-10-01 12:00:00", num_rows=2_000_000,
            schema=[_field("day", "DATE"), _field("amount", "NUMERIC"), _field("tags", "STRING", "REPEATED")],
        )
        source._get_table = MagicMock(return_value=table)
        source._sql_client = MagicMock()
        source._sql_client.query.return_value = types.SimpleNamespace(schema=[_field("day", "DATE")])
        return source

    def test_table_object_supplies_token_rows_and_types(self, source, fake_bigquery_module):
        adapter = BigQueryStatsAdapter(source)
        assert adapter.modified("events") == "2026-10-01 12:00:00"
        assert adapter.row_estimate("events") == 2_000_000
        columns = adapter.columns("events", adapter.relation("events"), 5)
        assert columns == [("day", "DATE"), ("amount", "NUMERIC"), ("tags", "ARRAY<STRING>")]
        assert [column_kind(t) for _, t in columns] == ["temporal", "numeric", "other"]
        source._get_table.assert_called_with("my-proj.ds.events")

    def test_expression_types_come_from_a_dry_run(self, source, fake_bigquery_module):
        adapter = BigQueryStatsAdapter(source)
        assert adapter.columns("recent", adapter.relation("recent"), 5) == [("day", "DATE")]
        config = source._sql_client.query.call_args.kwargs["job_config"]
        assert config.dry_run and not source.calls
        assert adapter.modified("recent") is None


def test_untyped_dates_and_decimals_keep_their_ranges():
    import datetime as dt
    import decimal

    from lumen.ai.table_stats import _Budget

    class Untyped(StatsAdapter):
        def sample_sql(self, relation, rows, total, seed):
            return "sample", True

        def execute(self, sql, params=None, timeout=None):
            # Object dtype, as drivers return DATE and DECIMAL.
            return pd.DataFrame({
                "day": [dt.date(2024, 1, 1), dt.date(2024, 2, 1)],
                "amount": [decimal.Decimal("1.5"), decimal.Decimal("9.25")],
            })

    stats = TableStats("t", columns=[ColumnStats("day"), ColumnStats("amount")])
    profiler = TableProfiler()
    sample = profiler._sample(Untyped(types.SimpleNamespace(dialect=None)), stats, "t", None, _Budget(None), 10_000_000)
    profiler._from_sample(stats, sample)
    day, amount = stats.column("day"), stats.column("amount")
    assert (day.type, day.kind, day.min, day.max) == ("DATE", "temporal", "2024-01-01", "2024-02-01")
    assert (amount.type, amount.kind, amount.min, amount.max) == ("DECIMAL", "numeric", 1.5, 9.25)
