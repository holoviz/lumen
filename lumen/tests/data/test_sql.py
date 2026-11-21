import pandas as pd
import pytest
import sqlglot

from lumen.data.sql import (
    EXPLORATION_MAX_ROWS, build_distinct_values_sql, execute_distinct_values,
    execute_exploration_sql, format_exploration_result, inline_tables,
    validate_read_only_sql,
)


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "WITH a AS (SELECT 1 AS x) SELECT x FROM a",
    "SELECT 1; -- trailing comment",
    "SELECT * FROM t UNION ALL SELECT * FROM u",
])
def test_validate_read_only_sql_accepts_one_query(sql):
    validate_read_only_sql(sql, "duckdb")


@pytest.mark.parametrize("sql", [
    "SELECT 1; SELECT 2",
    "DELETE FROM t",
    "DROP TABLE t",
    "INSERT INTO t SELECT 1",
    "CREATE TABLE u AS SELECT 1",
    "SELECT * INTO u FROM t",
    "UPDATE t SET a = 1",
])
def test_validate_read_only_sql_rejects_writes_and_several_statements(sql):
    with pytest.raises(ValueError, match="Only one read-only SELECT query is allowed"):
        validate_read_only_sql(sql, "duckdb")


def test_build_distinct_values_sql_fetches_one_extra_row():
    sql = build_distinct_values_sql('SELECT * FROM "t"', "name", "duckdb", limit=3)
    assert "LIMIT 4" in sql
    assert 'GROUP BY "name"' in sql


def test_format_exploration_result_reports_shape_and_sample():
    pytest.importorskip("tabulate")
    result = format_exploration_result(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert result.startswith("2 rows x 2 columns")
    assert "a: int64, b: str" in result
    assert "Sample (all rows):" in result


@pytest.fixture
def duckdb_sources():
    pytest.importorskip("duckdb")
    from lumen.sources.duckdb import DuckDBSource

    source = DuckDBSource(tables={
        "people": """
            SELECT * FROM (VALUES (1, 'Female'), (2, 'Male'), (3, 'Female')) AS t(id, gender)
        """,
        "numbers": f"SELECT range AS n FROM range({EXPLORATION_MAX_ROWS + 5})",
    })
    yield {(source.name, "people"): source, (source.name, "numbers"): source}
    source.close()


async def test_execute_exploration_sql_runs_a_read_only_query(duckdb_sources):
    pytest.importorskip("tabulate")
    result = await execute_exploration_sql(None, "SELECT gender FROM people", sources=duckdb_sources)
    assert result.startswith("3 rows x 1 columns")


async def test_execute_exploration_sql_caps_the_rows_it_reads(duckdb_sources):
    pytest.importorskip("tabulate")
    result = await execute_exploration_sql(None, "SELECT n FROM numbers", sources=duckdb_sources)
    assert result.startswith(f"at least {EXPLORATION_MAX_ROWS} rows x 1 columns")


async def test_execute_exploration_sql_refuses_writes(duckdb_sources):
    result = await execute_exploration_sql(None, "DROP TABLE people", sources=duckdb_sources)
    assert result.startswith("SQL parse/clean error")


async def test_execute_distinct_values_counts_values(duckdb_sources):
    result = await execute_distinct_values("people", "gender", sources=duckdb_sources)
    assert result.splitlines() == [
        "Values of 'gender' in 'people', most frequent first (value: rows):",
        "- 'Female': 2",
        "- 'Male': 1",
    ]


ORDERS = [("Ada", "east", 50, "paid"), ("Bo", "east", 15, "paid"), ("Cy", "west", 40, "paid"), ("Ada", "east", 7, "open")]

RESULTS = {
    "paid": "SELECT customer, region, amount FROM orders WHERE status = 'paid'",
    "by_region": "SELECT region, SUM(amount) AS total FROM paid GROUP BY region",
    "top": (
        "WITH ranked AS (SELECT region, total, RANK() OVER (ORDER BY total DESC) AS r FROM by_region) "
        "SELECT region, total FROM ranked WHERE r = 1"
    ),
}


@pytest.fixture(params=["duckdb", "sqlite"])
def engine(request):
    if request.param == "duckdb":
        duckdb = pytest.importorskip("duckdb")
        conn = duckdb.connect()
    else:
        import sqlite3
        conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE orders (customer TEXT, region TEXT, amount INTEGER, status TEXT)")
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", ORDERS)
    conn.execute("CREATE TABLE ranked (x INTEGER)")
    conn.execute("INSERT INTO ranked VALUES (1)")
    yield request.param, lambda sql: [tuple(row) for row in conn.execute(sql).fetchall()]
    conn.close()


def test_inline_tables_leaves_a_query_reading_none_of_them_as_written():
    sql = "SELECT  customer FROM orders -- a comment"
    assert inline_tables(sql, RESULTS, "duckdb") == sql


def test_inline_tables_runs_a_chain_on_the_base_tables(engine):
    dialect, run = engine
    sql = inline_tables("SELECT region, total FROM by_region ORDER BY region", RESULTS, dialect)
    assert sql.startswith("WITH paid AS")
    assert sql.index("paid AS") < sql.index("by_region AS")
    assert run(sql) == [("east", 65), ("west", 40)]


def test_inline_tables_hoists_a_definitions_with_and_renames_its_ctes_clashing_with_a_table(engine):
    dialect, run = engine
    sql = inline_tables("SELECT top.region, ranked.x FROM top, ranked", RESULTS, dialect)
    assert sql.count("WITH") == 1
    assert "ranked_1 AS" in sql
    assert run(sql) == [("east", 1)]


def test_inline_tables_joins_two_definitions_sharing_an_upstream_once(engine):
    dialect, run = engine
    tables = {**RESULTS, "customers": "SELECT region, COUNT(DISTINCT customer) AS n FROM paid GROUP BY region"}
    sql = inline_tables(
        "SELECT c.region, c.n, b.total FROM customers AS c JOIN by_region AS b ON c.region = b.region ORDER BY 1",
        tables, dialect,
    )
    assert sql.count("paid AS") == 1
    assert run(sql) == [("east", 2, 65), ("west", 1, 40)]


def test_inline_tables_keeps_the_querys_own_ctes_and_renames_a_definition_they_shadow(engine):
    dialect, run = engine
    sql = inline_tables(
        "WITH paid AS (SELECT * FROM paid WHERE amount > 20) SELECT paid.customer FROM paid ORDER BY 1",
        RESULTS, dialect,
    )
    assert "paid_1 AS" in sql
    assert "FROM paid_1 AS paid" in sql
    assert run(sql) == [("Ada",), ("Cy",)]


def test_inline_tables_does_not_inline_a_name_bound_to_a_cte():
    sql = "WITH by_region AS (SELECT 1 AS x FROM orders) SELECT * FROM by_region"
    assert inline_tables(sql, RESULTS, "duckdb") == sql


def test_inline_tables_reads_definitions_inside_subqueries(engine):
    dialect, run = engine
    sql = inline_tables(
        "SELECT customer FROM orders WHERE region IN (SELECT region FROM top) GROUP BY customer ORDER BY 1",
        RESULTS, dialect,
    )
    assert run(sql) == [("Ada",), ("Bo",)]


def test_inline_tables_keeps_a_recursive_with_recursive():
    tables = {"seq": "WITH RECURSIVE n AS (SELECT 1 AS i UNION ALL SELECT i + 1 FROM n WHERE i < 3) SELECT i FROM n"}
    duckdb = pytest.importorskip("duckdb")
    sql = inline_tables("SELECT SUM(i) FROM seq", tables, "duckdb")
    assert sql.startswith("WITH RECURSIVE")
    assert duckdb.sql(sql).fetchall() == [(6,)]


def test_inline_tables_raises_on_a_cycle():
    with pytest.raises(ValueError, match="cycle: a -> b -> a"):
        inline_tables("SELECT * FROM a", {"a": "SELECT * FROM b", "b": "SELECT * FROM a"})


@pytest.mark.parametrize("dialect", ["postgres", "snowflake", "bigquery", "tsql"])
def test_inline_tables_writes_one_with_clause_in_other_dialects(dialect):
    sql = inline_tables("SELECT region, total FROM top", RESULTS, dialect)
    parsed = sqlglot.parse_one(sql, read=dialect)
    assert len(list(parsed.find_all(sqlglot.exp.With))) == 1
    assert [cte.alias for cte in parsed.find_all(sqlglot.exp.CTE)] == ["paid", "by_region", "ranked", "top"]
