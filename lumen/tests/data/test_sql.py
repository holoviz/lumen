import pandas as pd
import pytest

from lumen.data.sql import (
    EXPLORATION_MAX_ROWS, build_distinct_values_sql, execute_distinct_values,
    execute_exploration_sql, format_exploration_result, validate_read_only_sql,
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
