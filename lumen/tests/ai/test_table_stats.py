"""Tests for lumen.ai.table_stats: tiered statistics, caching and rendering."""
import asyncio
import json
import sqlite3
import time

from unittest.mock import patch

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.schemas import Metaset, TableCatalogEntry, get_metaset
from lumen.ai.table_stats import (
    ESTIMATED, EXACT, SAMPLED, SCHEMA, TYPES_ONLY, ColumnStats,
    SnowflakeStatsAdapter, SQLAlchemyStatsAdapter, StatementTimeout,
    TableProfiler, TableStats, TableStatsStore, _parse_pg_array, column_kind,
    format_number, format_value, render_column, render_stats_header,
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
        with patch.object(TableProfiler, "profile", side_effect=AssertionError("recomputed")):
            assert fresh.get(_sqlite_source(sqlite_path), "account").rows == 5

        with sqlite3.connect(sqlite_path) as conn:
            conn.execute("INSERT INTO account VALUES (99, 'WEEKLY')")
        assert TableStatsStore(cache_dir=cache).get(_sqlite_source(sqlite_path), "account") is None

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
        stats = TableStats("t", rows=None, method=SAMPLED, sample_size=10000)
        col = ColumnStats("type", "TEXT", "string", values=[["VYDAJ", 0.6254], ["VYBER", 0.004]])
        assert render_column(col, stats) == "type TEXT {VYDAJ:63%, VYBER:<1%, ...}"
        assert render_stats_header(stats) == "over 1000000 rows; stats from a 10000-row sample"

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
