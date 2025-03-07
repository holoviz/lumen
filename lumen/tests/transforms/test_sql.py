import datetime as dt

from lumen.transforms.sql import (
    SQLColumns, SQLCount, SQLDistinct, SQLFilter, SQLFormat, SQLGroupBy,
    SQLLimit, SQLMinMax, SQLOverride, SQLSample, SQLSelectFrom,
)


def test_sql_group_by_single_column():
    result = SQLGroupBy.apply_to(
        "SELECT * FROM TABLE", by=["A"], aggregates={"AVG": "B"}
    )
    expected = "SELECT A, AVG(B) AS B FROM (SELECT * FROM TABLE) GROUP BY A"
    assert result == expected


def test_sql_group_by_multi_columns():
    result = SQLGroupBy.apply_to(
        "SELECT * FROM TABLE", by=["A"], aggregates={"AVG": ["B", "C"]}
    )
    expected = (
        "SELECT A, AVG(B) AS B, AVG(C) AS C FROM (SELECT * FROM TABLE) GROUP BY A"
    )
    assert result == expected


def test_sql_limit():
    result = SQLLimit.apply_to("SELECT * FROM TABLE", limit=10)
    expected = "SELECT * FROM (SELECT * FROM TABLE) LIMIT 10"
    assert result == expected


def test_sql_columns():
    result = SQLColumns.apply_to("SELECT * FROM TABLE", columns=["A", "B"])
    expected = "SELECT A, B FROM (SELECT * FROM TABLE)"
    assert result == expected


def test_sql_distinct():
    result = SQLDistinct.apply_to("SELECT * FROM TABLE", columns=["A", "B"])
    expected = "SELECT DISTINCT A, B FROM (SELECT * FROM TABLE)"
    assert result == expected


def test_sql_min_max():
    result = SQLMinMax.apply_to("SELECT * FROM TABLE", columns=["A", "B"])
    expected = "SELECT MIN(A) AS A_min, MAX(A) AS A_max, MIN(B) AS B_min, MAX(B) AS B_max FROM (SELECT * FROM TABLE)"
    assert result == expected


def test_sql_filter_none():
    result = SQLFilter.apply_to("SELECT * FROM TABLE", conditions=[("A", None)])
    expected = "SELECT * FROM (SELECT * FROM TABLE) WHERE A IS NULL"
    assert result == expected


def test_sql_filter_scalar():
    result = SQLFilter.apply_to("SELECT * FROM TABLE", conditions=[("A", 1)])
    expected = "SELECT * FROM (SELECT * FROM TABLE) WHERE A = 1"
    assert result == expected


def test_sql_filter_isin():
    result = SQLFilter.apply_to(
        "SELECT * FROM TABLE", conditions=[("A", ["A", "B", "C"])]
    )
    expected = "SELECT * FROM (SELECT * FROM TABLE) WHERE A IN ('A', 'B', 'C')"
    assert result == expected


def test_sql_filter_datetime():
    result = SQLFilter.apply_to(
        "SELECT * FROM TABLE", conditions=[("A", dt.datetime(2017, 4, 14))]
    )
    expected = "SELECT * FROM (SELECT * FROM TABLE) WHERE A = '2017-04-14 00:00:00'"
    assert result == expected


def test_sql_filter_date():
    result = SQLFilter.apply_to(
        "SELECT * FROM TABLE", conditions=[("A", dt.date(2017, 4, 14))]
    )
    expected = "SELECT * FROM (SELECT * FROM TABLE) WHERE A BETWEEN '2017-04-14 00:00:00' AND '2017-04-14 23:59:59'"
    assert result == expected


def test_sql_filter_date_range():
    result = SQLFilter.apply_to(
        "SELECT * FROM TABLE",
        conditions=[("A", (dt.date(2017, 2, 22), dt.date(2017, 4, 14)))],
    )
    expected = "SELECT * FROM (SELECT * FROM TABLE) WHERE A BETWEEN '2017-02-22 00:00:00' AND '2017-04-14 23:59:59'"
    assert result == expected


def test_sql_filter_datetime_range():
    result = SQLFilter.apply_to(
        "SELECT * FROM TABLE",
        conditions=[("A", (dt.datetime(2017, 2, 22), dt.datetime(2017, 4, 14)))],
    )
    expected = "SELECT * FROM (SELECT * FROM TABLE) WHERE A BETWEEN '2017-02-22 00:00:00' AND '2017-04-14 00:00:00'"
    assert result == expected


def test_sql_count():
    result = SQLCount.apply_to("SELECT * FROM TABLE")
    expected = "SELECT COUNT(*) AS count FROM (SELECT * FROM TABLE)"
    assert result == expected


def test_sql_override():
    result = SQLOverride.apply_to(
        "SELECT * FROM TABLE", override="SELECT A, B FROM TABLE"
    )
    expected = "SELECT A, B FROM TABLE"
    assert result == expected


def test_sql_format():
    result = SQLFormat.apply_to(
        "SELECT * FROM {table} WHERE {column} > :value",
        parameters={"table": "my_table", "column": "col1", "value": 10},
    )
    expected = "SELECT * FROM my_table WHERE col1 > 10"
    assert result == expected


def test_sql_select_from_basic():
    result = SQLSelectFrom.apply_to("my_table")
    expected = "SELECT * FROM my_table"
    assert result == expected


def test_sql_select_from_replace_table():
    result = SQLSelectFrom.apply_to("SELECT * FROM old_table", tables=["new_table"])
    expected = "SELECT * FROM new_table"
    assert result == expected


def test_sql_select_from_custom_expr():
    result = SQLSelectFrom.apply_to("my_table", sql_expr="SELECT id, name FROM {table}")
    expected = "SELECT id, name FROM my_table"
    assert result == expected


def test_sql_sample_tablesample_percent():
    """Test percent-based sampling with TABLESAMPLE (PostgreSQL, etc.)"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", percent=20, write="postgres")
    expected = "SELECT * FROM (SELECT * FROM TABLE) AS subquery TABLESAMPLE (20)"
    assert result == expected


def test_sql_sample_tablesample_size():
    """Test size-based sampling with TABLESAMPLE (PostgreSQL, etc.)"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100, write="postgres")
    expected = "SELECT * FROM (SELECT * FROM TABLE) AS subquery TABLESAMPLE (100)"
    assert result == expected


def test_sql_sample_tablesample_with_seed():
    """Test sampling with seed using TABLESAMPLE (PostgreSQL, etc.)"""
    result = SQLSample.apply_to(
        "SELECT * FROM TABLE", percent=20, seed=42, write="postgres"
    )
    expected = "SELECT * FROM (SELECT * FROM TABLE) AS subquery TABLESAMPLE (20) REPEATABLE (42)"
    assert result == expected


def test_sql_sample_tablesample_with_method():
    """Test sampling with method using TABLESAMPLE (PostgreSQL, etc.)"""
    result = SQLSample.apply_to(
        "SELECT * FROM TABLE",
        percent=20,
        write="postgres",
        sample_kwargs=dict(method="BERNOULLI"),
    )
    expected = (
        "SELECT * FROM (SELECT * FROM TABLE) AS subquery TABLESAMPLE 'BERNOULLI' (20)"
    )
    assert result == expected


def test_sql_sample_mysql_percent():
    """Test percent-based sampling in MySQL"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", percent=20, write="mysql")
    expected = "SELECT * FROM (SELECT * FROM `TABLE`) AS subquery ORDER BY RAND() LIMIT FLOOR(COUNT(*) * 0.2)"
    assert result == expected


def test_sql_sample_mysql_size():
    """Test size-based sampling in MySQL"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100, write="mysql")
    expected = "SELECT * FROM (SELECT * FROM `TABLE`) AS subquery WHERE FROZEN_RAND() BETWEEN RAND() AND RAND() + 0.0003 ORDER BY RAND() LIMIT 100"
    assert result == expected


def test_sql_sample_sqlite_percent():
    """Test percent-based sampling in SQLite"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", percent=20, write="sqlite")
    expected = "SELECT * FROM (SELECT * FROM TABLE) AS main_query WHERE RANDOM() < 0.2"
    assert result == expected


def test_sql_sample_sqlite_size():
    """Test size-based sampling in SQLite"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100, write="sqlite")
    expected = "SELECT * FROM (SELECT * FROM TABLE) AS main_query WHERE rowid IN (SELECT rowid FROM (SELECT * FROM TABLE) AS sampling_query ORDER BY RANDOM() LIMIT 100)"
    assert result == expected


def test_sql_sample_sqlite_with_seed():
    """Test sampling with seed in SQLite"""
    result = SQLSample.apply_to(
        "SELECT * FROM TABLE", size=100, seed=42, write="sqlite"
    )
    expected = "SELECT * FROM (SELECT * FROM TABLE) AS main_query WHERE rowid IN (SELECT rowid FROM (SELECT * FROM TABLE) AS sampling_query ORDER BY RANDOM(42) LIMIT 100)"
    assert result == expected


def test_sql_sample_generic_percent():
    """Test percent-based sampling with generic dialect"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", percent=20)
    expected = "SELECT * FROM (SELECT * FROM TABLE) AS subquery WHERE RAND() < 0.2"
    assert result == expected


def test_sql_sample_generic_size():
    """Test size-based sampling with generic dialect"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100)
    expected = (
        "SELECT * FROM (SELECT * FROM TABLE) AS subquery ORDER BY RAND() LIMIT 100"
    )
    assert result == expected


def test_sql_sample_generic_with_seed():
    """Test sampling with seed using generic dialect"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100, seed=42)
    expected = (
        "SELECT * FROM (SELECT * FROM TABLE) AS subquery ORDER BY RAND(42) LIMIT 100"
    )
    assert result == expected


class TestSQLFormat:
    def test_basic_formatting(self):
        """Test basic string formatting with parameters."""
        sql_in = "SELECT * FROM {table} WHERE {column} > :value"
        result = SQLFormat.apply_to(
            sql_in, parameters={"table": "my_table", "column": "col1", "value": 10}
        )
        expected = "SELECT * FROM my_table WHERE col1 > 10"
        assert result == expected

    def test_py_format_conversion(self):
        """Test conversion of Python-style formatting to SQLGlot placeholders."""
        sql_in = "SELECT * FROM {table} WHERE {column} > {value}"
        result = SQLFormat.apply_to(
            sql_in, parameters={"table": "my_table", "column": "col1", "value": 10}
        )
        expected = "SELECT * FROM my_table WHERE col1 > 10"
        assert result == expected

    def test_sqlglot_placeholders(self):
        """Test using SQLGlot-style placeholders."""
        sql_in = "SELECT * FROM :table WHERE :column > :value"
        result = SQLFormat.apply_to(
            sql_in, parameters={"table": "my_table", "column": "col1", "value": 10}
        )
        expected = "SELECT * FROM my_table WHERE col1 > 10"
        assert result == expected


class TestSQLSelectFrom:
    def test_replace_multiple_tables_with_list(self):
        """Test replacing multiple table names with a list of new table names."""
        sql_in = "SELECT * FROM table1 JOIN table2"
        result = SQLSelectFrom.apply_to(
            sql_in, tables=["new_table1", "new_table2"], identify=True
        )
        expected = 'SELECT * FROM "new_table1" AS "", "new_table2" AS ""'
        assert result == expected

    def test_dialect_specifics(self):
        """Test dialect-specific behavior."""
        sql_in = "my_table"
        result = SQLSelectFrom.apply_to(sql_in, identify=True, write="postgres")
        expected = "SELECT * FROM my_table"
        assert result == expected

    def test_complex_query_table_replacement(self):
        """Test replacing tables in a complex query with JOIN, WHERE, etc."""
        sql_in = """
            SELECT t1.col1, t2.col2
            FROM table1 t1
            JOIN table2 t2 ON t1.id = t2.id
            WHERE t1.col1 > 10
            GROUP BY t1.col1
        """
        result = SQLSelectFrom.apply_to(
            sql_in, tables={"table1": "new_table1", "table2": "new_table2"}
        )
        normalized_result = " ".join(result.split())
        expected = "SELECT t1.col1, t2.col2 FROM new_table1 AS t1 JOIN new_table2 AS t2 ON t1.id = t2.id WHERE t1.col1 > 10 GROUP BY t1.col1"
        assert normalized_result == expected
