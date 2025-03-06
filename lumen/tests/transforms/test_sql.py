import datetime as dt

from lumen.transforms.sql import (
    SQLColumns, SQLCount, SQLDistinct, SQLFilter, SQLFormat, SQLGroupBy,
    SQLLimit, SQLMinMax, SQLOverride, SQLSample, SQLSelectFrom,
)


def test_sql_group_by_single_column():
    result = SQLGroupBy.apply_to(
        "SELECT * FROM TABLE", by=["A"], aggregates={"AVG": "B"}
    )
    assert "AVG(B) AS B" in result
    assert "GROUP BY A" in result


def test_sql_group_by_multi_columns():
    result = SQLGroupBy.apply_to(
        "SELECT * FROM TABLE", by=["A"], aggregates={"AVG": ["B", "C"]}
    )
    assert "AVG(B) AS B" in result
    assert "AVG(C) AS C" in result
    assert "GROUP BY A" in result


def test_sql_limit():
    assert (
        SQLLimit.apply_to("SELECT * FROM TABLE", limit=10)
        == "SELECT * FROM (SELECT * FROM TABLE) LIMIT 10"
    )


def test_sql_columns():
    result = SQLColumns.apply_to("SELECT * FROM TABLE", columns=["A", "B"])
    assert "A" in result
    assert "B" in result


def test_sql_distinct():
    result = SQLDistinct.apply_to("SELECT * FROM TABLE", columns=["A", "B"])
    assert "DISTINCT" in result
    assert "A" in result
    assert "B" in result


def test_sql_min_max():
    result = SQLMinMax.apply_to("SELECT * FROM TABLE", columns=["A", "B"])
    assert "MIN(A)" in result.replace('"', "")
    assert "MAX(A)" in result.replace('"', "")
    assert "MIN(B)" in result.replace('"', "")
    assert "MAX(B)" in result.replace('"', "")


def test_sql_filter_none():
    result = SQLFilter.apply_to("SELECT * FROM TABLE", conditions=[("A", None)])
    assert "WHERE" in result
    assert "A IS NULL" in result


def test_sql_filter_scalar():
    result = SQLFilter.apply_to("SELECT * FROM TABLE", conditions=[("A", 1)])
    assert "WHERE" in result
    assert "A = 1" in result


def test_sql_filter_isin():
    result = SQLFilter.apply_to(
        "SELECT * FROM TABLE", conditions=[("A", ["A", "B", "C"])]
    )
    assert "WHERE" in result
    assert "A IN ('A', 'B', 'C')" in result


def test_sql_filter_datetime():
    result = SQLFilter.apply_to(
        "SELECT * FROM TABLE", conditions=[("A", dt.datetime(2017, 4, 14))]
    )
    assert "WHERE" in result
    assert "A = '2017-04-14 00:00:00'" in result


def test_sql_filter_date():
    result = SQLFilter.apply_to(
        "SELECT * FROM TABLE", conditions=[("A", dt.date(2017, 4, 14))]
    )
    assert "WHERE" in result
    assert "A BETWEEN '2017-04-14 00:00:00' AND '2017-04-14 23:59:59'" in result


def test_sql_filter_date_range():
    result = SQLFilter.apply_to(
        "SELECT * FROM TABLE",
        conditions=[("A", (dt.date(2017, 2, 22), dt.date(2017, 4, 14)))],
    )
    assert "WHERE" in result
    assert "A BETWEEN '2017-02-22 00:00:00' AND '2017-04-14 23:59:59'" in result


def test_sql_filter_datetime_range():
    result = SQLFilter.apply_to(
        "SELECT * FROM TABLE",
        conditions=[("A", (dt.datetime(2017, 2, 22), dt.datetime(2017, 4, 14)))],
    )
    assert "WHERE" in result
    assert "A BETWEEN '2017-02-22 00:00:00' AND '2017-04-14 00:00:00'" in result


def test_sql_count():
    result = SQLCount.apply_to("SELECT * FROM TABLE")
    assert "COUNT(*)" in result
    assert "count" in result


def test_sql_override():
    assert (
        SQLOverride.apply_to("SELECT * FROM TABLE", override="SELECT A, B FROM TABLE")
        == "SELECT A, B FROM TABLE"
    )


def test_sql_format():
    result = SQLFormat.apply_to(
        "SELECT * FROM {table} WHERE {column} > :value",
        parameters={"table": "my_table", "column": "col1", "value": 10},
    )
    assert "SELECT * FROM my_table WHERE col1 > 10" in result


def test_sql_select_from_basic():
    result = SQLSelectFrom.apply_to("my_table")
    assert result == "SELECT * FROM my_table"


def test_sql_select_from_replace_table():
    result = SQLSelectFrom.apply_to("SELECT * FROM old_table", tables=["new_table"])
    assert "FROM new_table" in result


def test_sql_select_from_custom_expr():
    result = SQLSelectFrom.apply_to("my_table", sql_expr="SELECT id, name FROM {table}")
    assert result == "SELECT id, name FROM my_table"


def test_sql_sample_tablesample_percent():
    """Test percent-based sampling with TABLESAMPLE (PostgreSQL, etc.)"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", percent=20, write="postgres")
    assert "TABLESAMPLE" in result
    assert "20" in result


def test_sql_sample_tablesample_size():
    """Test size-based sampling with TABLESAMPLE (PostgreSQL, etc.)"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100, write="postgres")
    assert "TABLESAMPLE" in result
    assert "100" in result


def test_sql_sample_tablesample_with_seed():
    """Test sampling with seed using TABLESAMPLE (PostgreSQL, etc.)"""
    result = SQLSample.apply_to(
        "SELECT * FROM TABLE", percent=20, seed=42, write="postgres"
    )
    assert "TABLESAMPLE" in result
    assert "20" in result
    assert "REPEATABLE" in result
    assert "42" in result


def test_sql_sample_tablesample_with_method():
    """Test sampling with method using TABLESAMPLE (PostgreSQL, etc.)"""
    result = SQLSample.apply_to(
        "SELECT * FROM TABLE",
        percent=20,
        write="postgres",
        sample_kwargs=dict(method="BERNOULLI"),
    )
    assert "TABLESAMPLE" in result
    assert "20" in result
    assert "BERNOULLI" in result


def test_sql_sample_mysql_percent():
    """Test percent-based sampling in MySQL"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", percent=20, write="mysql")
    assert "ORDER BY RAND()" in result
    assert "LIMIT" in result
    assert "COUNT" in result
    assert "* 0.2" in result


def test_sql_sample_mysql_size():
    """Test size-based sampling in MySQL"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100, write="mysql")
    assert "ORDER BY RAND()" in result
    assert "LIMIT 100" in result


def test_sql_sample_sqlite_percent():
    """Test percent-based sampling in SQLite"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", percent=20, write="sqlite")
    assert "RANDOM() < 0.2" in result
    assert "main_query" in result


def test_sql_sample_sqlite_size():
    """Test size-based sampling in SQLite"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100, write="sqlite")
    assert "rowid IN" in result
    assert "ORDER BY RANDOM()" in result
    assert "LIMIT 100" in result
    assert "main_query" in result
    assert "sampling_query" in result


def test_sql_sample_sqlite_with_seed():
    """Test sampling with seed in SQLite"""
    result = SQLSample.apply_to(
        "SELECT * FROM TABLE", size=100, seed=42, write="sqlite"
    )
    assert "ORDER BY RANDOM(42)" in result
    assert "LIMIT 100" in result


def test_sql_sample_generic_percent():
    """Test percent-based sampling with generic dialect"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", percent=20)
    assert "RAND() < 0.2" in result


def test_sql_sample_generic_size():
    """Test size-based sampling with generic dialect"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100)
    assert "ORDER BY RAND()" in result
    assert "LIMIT 100" in result


def test_sql_sample_generic_with_seed():
    """Test sampling with seed using generic dialect"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100, seed=42)
    assert "ORDER BY RAND(42)" in result
    assert "LIMIT 100" in result


class TestSQLFormat:
    def test_basic_formatting(self):
        """Test basic string formatting with parameters."""
        sql_in = "SELECT * FROM {table} WHERE {column} > :value"
        result = SQLFormat.apply_to(
            sql_in, parameters={"table": "my_table", "column": "col1", "value": 10}
        )
        assert "SELECT * FROM my_table WHERE col1 > 10" in result

    def test_py_format_conversion(self):
        """Test conversion of Python-style formatting to SQLGlot placeholders."""
        sql_in = "SELECT * FROM {table} WHERE {column} > {value}"
        result = SQLFormat.apply_to(
            sql_in, parameters={"table": "my_table", "column": "col1", "value": 10}
        )
        assert "SELECT * FROM my_table WHERE col1 > 10" in result

    def test_sqlglot_placeholders(self):
        """Test using SQLGlot-style placeholders."""
        sql_in = "SELECT * FROM :table WHERE :column > :value"
        result = SQLFormat.apply_to(
            sql_in, parameters={"table": "my_table", "column": "col1", "value": 10}
        )
        assert "SELECT * FROM my_table WHERE col1 > 10" in result

class TestSQLSelectFrom:
    def test_replace_multiple_tables_with_list(self):
        """Test replacing multiple table names with a list of new table names."""
        sql_in = "SELECT * FROM table1 JOIN table2"
        result = SQLSelectFrom.apply_to(
            sql_in,
            tables=["new_table1", "new_table2"],
            identify=True
        )
        assert "new_table1" in result
        assert "new_table2" in result

    def test_dialect_specifics(self):
        """Test dialect-specific behavior."""
        sql_in = "my_table"
        result = SQLSelectFrom.apply_to(
            sql_in,
            identify=True,
            write="postgres"
        )
        assert "FROM" in result
        assert "my_table" in result

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
            sql_in,
            tables={"table1": "new_table1", "table2": "new_table2"}
        )
        assert "new_table1" in result
        assert "new_table2" in result
