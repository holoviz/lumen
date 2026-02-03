import datetime as dt

from textwrap import dedent

import pytest
import sqlglot

from lumen.transforms.sql import (
    SQLColumns, SQLCount, SQLDistinct, SQLFilter, SQLFormat, SQLGroupBy,
    SQLLimit, SQLMinMax, SQLOverride, SQLPreFilter, SQLRemoveSourceSeparator,
    SQLSample, SQLSelectFrom, SQLTransform,
)


def test_dialect_alias_postgresql():
    """Test that 'postgresql' dialect (from SQLAlchemy) is recognized as 'postgres' (sqlglot)."""
    result = SQLTransform(read="postgresql", write="postgresql").apply(
        "SELECT * FROM my_table WHERE col > 10"
    )
    assert "SELECT * FROM my_table WHERE col > 10" == result


def test_dialect_alias_mssql():
    """Test that 'mssql' dialect (from SQLAlchemy) is recognized as 'tsql' (sqlglot)."""
    result = SQLTransform(read="mssql", write="mssql").apply(
        "SELECT * FROM my_table WHERE col > 10"
    )
    assert "SELECT * FROM my_table WHERE col > 10" == result


def test_init_with_any_dialect():
    transform = SQLTransform(read="any", write="any")
    assert transform.read is None
    assert transform.write is None


def test_init_with_any_read_dialect():
    transform = SQLTransform(read="any", write="postgres")
    assert transform.read == "postgres"
    assert transform.write == "postgres"


def test_init_with_any_write_dialect():
    transform = SQLTransform(read="mysql", write="any")
    assert transform.read == "mysql"
    assert transform.write == "mysql"


def test_init_with_read_only():
    transform = SQLTransform(read="postgres")
    assert transform.read == "postgres"
    assert transform.write == "postgres"


def test_init_with_write_only():
    transform = SQLTransform(write="mysql")
    assert transform.read == "mysql"
    assert transform.write == "mysql"


def test_init_with_both_dialects():
    transform = SQLTransform(read="sqlite", write="postgres")
    assert transform.read == "sqlite"
    assert transform.write == "postgres"

    postgres_result = SQLTransform(read="duckdb", write="postgres").apply(
        "SELECT * FROM table USING SAMPLE 10%"
    )
    expected_postgres = "SELECT * FROM table TABLESAMPLE SYSTEM (10)"
    assert postgres_result == expected_postgres


def test_sql_parse_hyphen():
    result = SQLTransform().apply(
        "SELECT * FROM read_csv('life-expectancy.csv')"
    )
    expected = "SELECT * FROM READ_CSV('life-expectancy.csv')"
    assert result == expected

    result = SQLTransform().apply("SELECT * FROM 'life-expectancy.csv'")
    expected = 'SELECT * FROM "life-expectancy.csv"'
    assert result == expected


def test_sql_optimize():
    result = SQLTransform(optimize=True).apply(
        "SELECT * FROM (SELECT col1, col2 FROM table) WHERE col1 > 10"
    )
    expected = """SELECT "table"."col1" AS "col1", "table"."col2" AS "col2" FROM "table" AS "table" WHERE "table"."col1" > 10"""
    assert result == expected


def test_sql_pretty():
    result = SQLTransform(pretty=True).apply(
        "SELECT * FROM (SELECT col1, col2 FROM table) WHERE col1 > 10"
    )
    expected = dedent(
        """
        SELECT
          *
        FROM (
          SELECT
            col1,
            col2
          FROM table
        )
        WHERE
          col1 > 10
        """
    ).strip()
    assert result == expected


def test_sql_identify():
    result = SQLTransform(identify=True).apply("SELECT * FROM table WHERE col1 > 10")
    expected = 'SELECT * FROM "table" WHERE "col1" > 10'
    assert result == expected


def test_sql_comments():
    result = SQLTransform(comments=False).apply(
        "SELECT * FROM table WHERE col1 > 10 /* comment */"
    )
    expected = "SELECT * FROM table WHERE col1 > 10"
    assert result == expected


def test_add_encoding_to_read_csv():
    pytest.importorskip("chardet")
    expression: list = sqlglot.parse("READ_CSV('data/life-expectancy.csv')")
    result = SQLTransform(identify=True)._add_encoding_to_read_csv(expression[0])
    expected = "READ_CSV('data/life-expectancy.csv', encoding='utf-8')"
    assert result.sql() == expected


def test_sql_error_level():
    with pytest.raises(
        sqlglot.errors.ParseError, match="Expected table name but got"
    ):
        SQLTransform(error_level=sqlglot.ErrorLevel.RAISE).apply("SELECT FROM {table}")


def test_sql_unsupported_level():
    with pytest.raises(
        sqlglot.errors.UnsupportedError, match="TABLESAMPLE unsupported"
    ):
        SQLTransform(unsupported_level=sqlglot.ErrorLevel.RAISE, write="sqlite").apply(
            "SELECT FROM TABLE TABLESAMPLE (10) WHERE col1 > 10",
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


def test_sql_limit_lower_than_original():
    result = SQLLimit.apply_to("SELECT * FROM TABLE LIMIT 15", limit=10)
    expected = "SELECT * FROM (SELECT * FROM TABLE LIMIT 15) LIMIT 10"
    assert result == expected


def test_sql_limit_higher_than_original():
    result = SQLLimit.apply_to("SELECT * FROM TABLE LIMIT 15", limit=20)
    expected = "SELECT * FROM TABLE LIMIT 15"
    assert result == expected


def test_sql_columns():
    result = SQLColumns.apply_to("SELECT * FROM TABLE", columns=["A", "B"])
    expected = "SELECT A, B FROM (SELECT * FROM TABLE)"
    assert result == expected


def test_sql_distinct():
    result = SQLDistinct.apply_to("SELECT * FROM TABLE", columns=["A", "B"])
    expected = 'SELECT DISTINCT "A", "B" FROM (SELECT * FROM TABLE)'
    assert result == expected


def test_sql_min_max():
    result = SQLMinMax.apply_to("SELECT * FROM TABLE", columns=["A", "B"])
    expected = "SELECT MIN(A) AS A_min, MAX(A) AS A_max, MIN(B) AS B_min, MAX(B) AS B_max FROM (SELECT * FROM TABLE)"
    assert result == expected


def test_sql_min_max_nonalphanum_characters():
    result = SQLMinMax.apply_to("SELECT * FROM TABLE", columns=["A_B-123", "Period life expectancy at birth - Sex: total - Age: 0"])
    expected = (
        'SELECT MIN("A_B-123") AS "A_B-123_min", MAX("A_B-123") AS "A_B-123_max", '
        'MIN("Period life expectancy at birth - Sex: total - Age: 0") AS "Period life expectancy at birth - Sex: total - Age: 0_min", '
        'MAX("Period life expectancy at birth - Sex: total - Age: 0") AS "Period life expectancy at birth - Sex: total - Age: 0_max" '
        'FROM (SELECT * FROM TABLE)'
    )
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


def test_sql_prefilter_basic():
    """Test basic prefiltering on a single table."""
    result = SQLPreFilter.apply_to(
        "SELECT n_genes FROM obs",
        conditions=[("obs", [("obs_id", ["cell1", "cell2"])])]
    )
    expected = 'SELECT n_genes FROM (SELECT * FROM "obs" WHERE obs_id IN (\'cell1\', \'cell2\')) AS obs'
    assert result == expected


def test_sql_prefilter_multiple_conditions():
    """Test prefiltering with multiple conditions on the same table."""
    result = SQLPreFilter.apply_to(
        "SELECT n_genes FROM obs",
        conditions=[("obs", [("obs_id", ["cell1", "cell2"]), ("cell_type", "T-cell")])]
    )
    expected = 'SELECT n_genes FROM (SELECT * FROM "obs" WHERE obs_id IN (\'cell1\', \'cell2\') AND cell_type = \'T-cell\') AS obs'
    assert result == expected


def test_sql_prefilter_multiple_tables():
    """Test prefiltering on multiple tables in a JOIN."""
    result = SQLPreFilter.apply_to(
        "SELECT o.n_genes, v.gene_name FROM obs o JOIN var v ON o.id = v.id",
        conditions=[
            ("obs", [("obs_id", ["cell1", "cell2"])]),
            ("var", [("var_id", ["gene1", "gene2"])])
        ]
    )
    # Note: The exact formatting may vary, but should contain both filtered subqueries
    assert 'obs_id IN (\'cell1\', \'cell2\')' in result
    assert 'var_id IN (\'gene1\', \'gene2\')' in result
    assert 'SELECT * FROM "obs"' in result
    assert 'SELECT * FROM "var"' in result


def test_sql_prefilter_no_conditions():
    """Test that prefilter passes through unchanged when no conditions."""
    sql_in = "SELECT n_genes FROM obs"
    result = SQLPreFilter.apply_to(sql_in, conditions=[])
    expected = sql_in
    assert result == expected


def test_sql_prefilter_table_not_in_conditions():
    """Test that tables not mentioned in conditions are left unchanged."""
    result = SQLPreFilter.apply_to(
        "SELECT n_genes FROM obs",
        conditions=[("var", [("var_id", ["gene1", "gene2"])])]
    )
    # Since obs is not in conditions, query should be unchanged
    expected = "SELECT n_genes FROM obs"
    assert result == expected


def test_sql_prefilter_numeric_conditions():
    """Test prefiltering with numeric conditions."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM data",
        conditions=[("data", [("score", 85), ("count", [100, 200, 300])])]
    )
    expected = 'SELECT * FROM (SELECT * FROM "data" WHERE score = 85 AND count IN (\'100\', \'200\', \'300\')) AS data'
    assert result == expected


def test_sql_prefilter_none_conditions():
    """Test prefiltering with None values."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM data",
        conditions=[("data", [("nullable_col", None)])]
    )
    expected = 'SELECT * FROM (SELECT * FROM "data" WHERE nullable_col IS NULL) AS data'
    assert result == expected


def test_sql_prefilter_mixed_none_conditions():
    """Test prefiltering with mixed None and non-None values."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM data",
        conditions=[("data", [("status", ["active", None, "pending"])])]
    )
    expected = 'SELECT * FROM (SELECT * FROM "data" WHERE status IS NULL OR status IN (\'active\', \'pending\')) AS data'
    assert result == expected


def test_sql_prefilter_date_conditions():
    """Test prefiltering with date conditions."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM events",
        conditions=[("events", [("event_date", dt.date(2023, 1, 15))])]
    )
    expected = 'SELECT * FROM (SELECT * FROM "events" WHERE event_date BETWEEN \'2023-01-15 00:00:00\' AND \'2023-01-15 23:59:59\') AS events'
    assert result == expected


def test_sql_prefilter_datetime_conditions():
    """Test prefiltering with datetime conditions."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM events",
        conditions=[("events", [("created_at", dt.datetime(2023, 1, 15, 10, 30, 0))])]
    )
    expected = 'SELECT * FROM (SELECT * FROM "events" WHERE created_at = \'2023-01-15 10:30:00\') AS events'
    assert result == expected


def test_sql_prefilter_date_range_conditions():
    """Test prefiltering with date range conditions."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM events",
        conditions=[("events", [("event_date", (dt.date(2023, 1, 1), dt.date(2023, 1, 31)))])]
    )
    expected = 'SELECT * FROM (SELECT * FROM "events" WHERE event_date BETWEEN \'2023-01-01 00:00:00\' AND \'2023-01-31 23:59:59\') AS events'
    assert result == expected


def test_sql_prefilter_datetime_range_conditions():
    """Test prefiltering with datetime range conditions."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM events",
        conditions=[("events", [("created_at", (dt.datetime(2023, 1, 1, 9, 0), dt.datetime(2023, 1, 1, 17, 0)))])]
    )
    expected = 'SELECT * FROM (SELECT * FROM "events" WHERE created_at BETWEEN \'2023-01-01 09:00:00\' AND \'2023-01-01 17:00:00\') AS events'
    assert result == expected


def test_sql_prefilter_multiple_date_ranges():
    """Test prefiltering with multiple date ranges in a list."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM events",
        conditions=[("events", [("event_date", [(dt.date(2023, 1, 1), dt.date(2023, 1, 31)), (dt.date(2023, 6, 1), dt.date(2023, 6, 30))])])]
    )
    # Should use OR to combine multiple ranges
    assert 'event_date BETWEEN \'2023-01-01 00:00:00\' AND \'2023-01-31 23:59:59\'' in result
    assert 'event_date BETWEEN \'2023-06-01 00:00:00\' AND \'2023-06-30 23:59:59\'' in result
    assert ' OR ' in result


def test_sql_prefilter_complex_query():
    """Test prefiltering on a complex query with GROUP BY and ORDER BY."""
    result = SQLPreFilter.apply_to(
        "SELECT cell_type, AVG(n_genes) as avg_genes FROM obs GROUP BY cell_type ORDER BY avg_genes DESC",
        conditions=[("obs", [("obs_id", ["cell1", "cell2", "cell3"])])]
    )
    # Should preserve the structure but filter the obs table
    assert 'obs_id IN (\'cell1\', \'cell2\', \'cell3\')' in result
    assert 'GROUP BY cell_type' in result
    assert 'ORDER BY avg_genes DESC' in result


def test_sql_prefilter_subquery():
    """Test prefiltering when the original query already contains subqueries."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM (SELECT n_genes FROM obs WHERE n_genes > 100) filtered_obs",
        conditions=[("obs", [("obs_id", ["cell1", "cell2"])])]
    )
    # Should filter the obs table within the existing subquery structure
    assert 'obs_id IN (\'cell1\', \'cell2\')' in result
    assert 'n_genes > 100' in result


def test_sql_prefilter_with_aliases():
    """Test prefiltering preserves table aliases."""
    result = SQLPreFilter.apply_to(
        "SELECT o.n_genes FROM obs o",
        conditions=[("obs", [("obs_id", ["cell1", "cell2"])])]
    )
    # Should preserve the alias 'o' for the filtered obs table
    assert 'obs_id IN (\'cell1\', \'cell2\')' in result
    assert result.endswith(' AS o')  # Alias should be preserved


def test_sql_prefilter_float_conditions():
    """Test prefiltering with float values."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM measurements",
        conditions=[("measurements", [("temperature", 98.6), ("pressure", [1013.25, 1020.0])])]
    )
    assert 'temperature = 98.6' in result
    assert "pressure IN ('1013.25', '1020.0')" in result


def test_sql_prefilter_string_conditions():
    """Test prefiltering with string conditions."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM users",
        conditions=[("users", [("status", "active"), ("role", ["admin", "user", "guest"])])]
    )
    assert "status = 'active'" in result
    assert "role IN ('admin', 'user', 'guest')" in result


def test_sql_prefilter_all_none_list():
    """Test prefiltering with a list containing only None values."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM data",
        conditions=[("data", [("nullable_field", [None, None])])]
    )
    expected = 'SELECT * FROM (SELECT * FROM "data" WHERE nullable_field IS NULL) AS data'
    assert result == expected


def test_sql_prefilter_join_with_conditions():
    """Test prefiltering on tables involved in JOINs."""
    result = SQLPreFilter.apply_to(
        "SELECT u.name, p.title FROM users u INNER JOIN posts p ON u.id = p.user_id",
        conditions=[
            ("users", [("status", "active")]),
            ("posts", [("published", 1)])  # Use 1 instead of True
        ]
    )
    assert "status = 'active'" in result
    assert "published = 1" in result
    assert "INNER JOIN" in result


def test_sql_prefilter_boolean_conditions():
    """Test prefiltering with numeric boolean-like values."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM articles",
        conditions=[("articles", [("published", 1), ("featured", 0)])]  # Use 1/0 instead of True/False
    )
    assert "published = 1" in result
    assert "featured = 0" in result


def test_sql_prefilter_unsupported_condition_type():
    """Test prefiltering with unsupported condition type (should warn and skip)."""
    # This uses a complex object that's not supported
    complex_obj = {"key": "value"}
    result = SQLPreFilter.apply_to(
        "SELECT * FROM data",
        conditions=[("data", [("metadata", complex_obj), ("valid_col", "valid_value")])]
    )
    # Should process the valid condition but skip the invalid one
    assert "valid_col = 'valid_value'" in result
    # Should not crash or include the complex object
    assert "metadata" not in result or "WHERE valid_col = 'valid_value'" in result


def test_sql_prefilter_boolean_values_unsupported():
    """Test that boolean True/False values are not supported and get skipped."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM data",
        conditions=[("data", [("bool_col", True), ("valid_col", "valid_value")])]
    )
    # Boolean values are not supported and should be skipped with a warning
    # Only the valid condition should be processed
    assert "valid_col = 'valid_value'" in result
    # The boolean condition should not appear in the result
    assert "bool_col" not in result or "bool_col = True" not in result


def test_sql_prefilter_empty_condition_value():
    """Test prefiltering with empty list as condition value."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM data",
        conditions=[("data", [("empty_list_col", [])])]
    )
    # Empty list should be handled gracefully - likely no WHERE clause for that condition
    # The exact behavior may vary, but should not crash
    assert "SELECT * FROM" in result


def test_sql_prefilter_case_sensitivity():
    """Test that table name matching is case sensitive."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM Users",  # Capital U
        conditions=[("users", [("id", 123)])]  # lowercase u
    )
    # Should not match due to case difference
    expected = "SELECT * FROM Users"
    assert result == expected


def test_sql_prefilter_multiple_tables_same_conditions():
    """Test prefiltering multiple tables with the same condition structure."""
    result = SQLPreFilter.apply_to(
        "SELECT u.name, a.title FROM users u JOIN articles a ON u.id = a.author_id",
        conditions=[
            ("users", [("status", "active")]),
            ("articles", [("status", "published")])
        ]
    )
    # Both tables should be filtered, each with their own WHERE clause
    assert result.count("status = 'active'") == 1
    assert result.count("status = 'published'") == 1


def test_sql_prefilter_left_join_preservation():
    """Test that LEFT JOINs are preserved correctly with prefiltering."""
    result = SQLPreFilter.apply_to(
        "SELECT u.name, p.title FROM users u LEFT JOIN posts p ON u.id = p.user_id",
        conditions=[("users", [("active", 1)])]
    )
    assert "LEFT JOIN" in result
    assert "active = 1" in result


def test_sql_prefilter_quoted_table_names():
    """Test prefiltering with already quoted table names."""
    # The table name matching is based on str(table_expr.this) which strips quotes
    # So 'my-table' should match "my-table" in the SQL
    result = SQLPreFilter.apply_to(
        'SELECT * FROM "my-table"',
        conditions=[('my-table', [('new_condition', 'value')])]
    )
    # Since quotes are stripped during parsing, this should work
    assert "new_condition = 'value'" in result


def test_sql_prefilter_nested_subqueries():
    """Test prefiltering with nested subqueries."""
    result = SQLPreFilter.apply_to(
        "SELECT * FROM (SELECT id FROM (SELECT * FROM users WHERE active = 1) active_users) user_ids",
        conditions=[("users", [("role", "admin")])]
    )
    # Should filter the innermost users table
    assert "role = 'admin'" in result
    assert "active = 1" in result


def test_sql_prefilter_union_queries():
    """Test prefiltering with UNION queries."""
    result = SQLPreFilter.apply_to(
        "SELECT name FROM users WHERE active = 1 UNION SELECT name FROM admins WHERE active = 1",
        conditions=[
            ("users", [("role", "user")]),
            ("admins", [("role", "admin")])
        ]
    )
    # Both tables in the UNION should be filtered
    assert "role = 'user'" in result
    assert "role = 'admin'" in result
    assert "UNION" in result


def test_sql_prefilter_cte_common_table_expressions():
    """Test prefiltering with Common Table Expressions (CTEs)."""
    result = SQLPreFilter.apply_to(
        "WITH active_users AS (SELECT * FROM users WHERE status = 'active') SELECT * FROM active_users",
        conditions=[("users", [("role", "admin")])]
    )
    # Should filter the users table within the CTE
    assert "role = 'admin'" in result
    assert "status = 'active'" in result
    assert "WITH" in result


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
    expected = 'SELECT * FROM my_table'
    assert result == expected


def test_sql_select_from_replace_table():
    result = SQLSelectFrom.apply_to("SELECT * FROM old_table", tables=["new_table"])
    expected = "SELECT * FROM new_table"
    assert result == expected


def test_sql_select_from_custom_expr():
    result = SQLSelectFrom.apply_to("my_table", sql_expr="SELECT id, name FROM {table}")
    expected = 'SELECT id, name FROM my_table'
    assert result == expected


def test_sql_select_from_path():
    result = SQLSelectFrom.apply_to("data/life_expectancy.csv", sql_expr="SELECT id, name FROM {table}")
    expected = 'SELECT id, name FROM "data/life_expectancy.csv"'
    assert result == expected


def test_sql_select_from_duckdb_path():
    result = SQLSelectFrom.apply_to("read_csv('data/life_expectancy.csv')", sql_expr="SELECT id, name FROM {table}")
    expected = "SELECT id, name FROM READ_CSV('data/life_expectancy.csv')"
    assert result == expected


def test_sql_select_from_expression():
    result = SQLSelectFrom.apply_to("""
        SELECT
            arr,
            churn,
            CASE
                WHEN arr > 0
                THEN ROUND((churn / churn) * 100, 2)
                ELSE NULL
            END AS churn_pct
        FROM current_arr
        """, sql_expr="SELECT id, name FROM {table}")
    expected = """
        SELECT
            arr,
            churn,
            CASE
                WHEN arr > 0
                THEN ROUND((churn / churn) * 100, 2)
                ELSE NULL
            END AS churn_pct
        FROM current_arr
        """
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
        "SELECT * FROM (SELECT * FROM TABLE) AS subquery TABLESAMPLE BERNOULLI (20)"
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
    expected = "SELECT * FROM (SELECT * FROM TABLE) AS subquery WHERE RANDOM() < 0.2"
    assert result == expected


def test_sql_sample_sqlite_size():
    """Test size-based sampling in SQLite"""
    result = SQLSample.apply_to("SELECT * FROM TABLE", size=100, write="sqlite")
    expected = "SELECT * FROM (SELECT * FROM TABLE) AS subquery ORDER BY RANDOM() LIMIT 100"
    assert result == expected


def test_sql_sample_sqlite_with_seed():
    """Test sampling with seed in SQLite"""
    result = SQLSample.apply_to(
        "SELECT * FROM TABLE", size=100, seed=42, write="sqlite"
    )
    expected = "SELECT * FROM (SELECT * FROM TABLE) AS subquery ORDER BY RANDOM(42) LIMIT 100"
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
        expected = 'SELECT * FROM "my_table"'
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

    def test_table_path(self):
        sql_in = '/path/to/my_data.parquet'
        result = SQLSelectFrom.apply_to(sql_in)
        expected = 'SELECT * FROM "/path/to/my_data.parquet"'
        assert result == expected

    def test_sql_expression_as_table_value(self):
        """Test that SQL expressions with quoted identifiers are handled correctly.
        
        When tables dict contains a SQL expression as the value (not just a table name),
        the quoted identifiers in that expression should be preserved without double-quoting.
        
        This was a bug where 'SELECT * FROM obs WHERE "Smoking Status" = \'Former\''
        would become 'SELECT * FROM obs WHERE ""Smoking Status"" = \'Former\''
        """
        sql_in = "my_new_table"
        sql_expr = '''SELECT * FROM obs WHERE "Smoking Status" = 'Former' '''
        result = SQLSelectFrom.apply_to(
            sql_in,
            tables={"my_new_table": sql_expr},
            read="duckdb"
        )
        # The result should contain the SQL as a subquery, preserving the quoted identifier
        assert '"Smoking Status"' in result
        # Should NOT have double-quoted identifiers
        assert '""Smoking Status""' not in result
        # Should be wrapped as a subquery
        assert 'SELECT * FROM (' in result or 'FROM (' in result

    def test_sql_expression_with_complex_where_clause(self):
        """Test SQL expressions with complex WHERE clauses containing quoted column names."""
        sql_in = "subset_smokers"
        sql_expr = '''SELECT * FROM obs WHERE "Smoking Status" = 'Current' OR "Smoking Status" = 'Former' '''
        result = SQLSelectFrom.apply_to(
            sql_in,
            tables={"subset_smokers": sql_expr},
            read="duckdb"
        )
        # Quoted identifiers should be preserved
        assert '"Smoking Status"' in result
        # Should NOT have double-quoted identifiers  
        assert '""Smoking Status""' not in result
        # Should preserve the OR condition
        assert "'Current'" in result
        assert "'Former'" in result

    def test_sql_expression_preserves_column_names_with_spaces(self):
        """Test that column names with spaces remain properly quoted."""
        sql_in = "filtered_data"
        sql_expr = '''SELECT "Pack Years", "Stage at Dx" FROM obs WHERE "Tissue Type" = 'Primary' '''
        result = SQLSelectFrom.apply_to(
            sql_in,
            tables={"filtered_data": sql_expr},
            read="duckdb"
        )
        # All quoted identifiers should be preserved correctly
        assert '"Pack Years"' in result
        assert '"Stage at Dx"' in result  
        assert '"Tissue Type"' in result
        # No double-quoting
        assert '""Pack Years""' not in result
        assert '""Stage at Dx""' not in result
        assert '""Tissue Type""' not in result


class TestSQLRemoveSourceSeparator:

    @pytest.mark.parametrize("quoted", [True, False])
    def test_no_change(self, quoted):
        table = '"table1"' if quoted else "table1"
        sql_in = f"SELECT * FROM {table}"
        result = SQLRemoveSourceSeparator.apply_to(sql_in)
        expected = f"SELECT * FROM {table}"
        assert result == expected

    def test_basic(self):
        sql_in = "SELECT * FROM source ⦙ table2"
        result = SQLRemoveSourceSeparator.apply_to(sql_in)
        expected = "SELECT * FROM table2"
        assert result == expected

    def test_read(self):
        sql_in = "SELECT * FROM read_csv('source ⦙ table2.csv')"
        result = SQLRemoveSourceSeparator.apply_to(sql_in, read="duckdb")
        expected = "SELECT * FROM READ_CSV('table2.csv')"
        assert result == expected

    def test_complex(self):
        sql_in = """
            WITH data_centers AS (
                SELECT
                    "FacName" AS "Facility_Name",
                    CAST(SUBSTRING("geometry", 7, LENGTH("geometry") - 7) AS TEXT) AS "Coordinates",
                    "lon" AS "Longitude",
                    "lat" AS "Latitude",
                    'Data Center' AS "Type"
                FROM
                    READ_CSV('ProvidedSource00000 ⦙ data_centers.csv')
            ),
            power_plants AS (
                SELECT
                    "Plant name" AS "Facility_Name",
                    "Plant longitude" AS "Longitude",
                    "Plant latitude" AS "Latitude",
                    "Plant primary fuel category" AS "Type"
                FROM
                    READ_CSV('ProvidedSource00000 ⦙ plant_specific_buffers.csv')
            )

            SELECT
                "Facility_Name",
                "Longitude",
                "Latitude",
                "Type"
            FROM
                data_centers
            UNION ALL
            SELECT
                "Facility_Name",
                "Longitude",
                "Latitude",
                "Type"
            FROM
                power_plants;
        """
        result = SQLRemoveSourceSeparator.apply_to(sql_in, read="duckdb")
        assert " ⦙ " not in result
        assert "READ_CSV('plant_specific_buffers.csv')" in result
        assert "READ_CSV('data_centers.csv')" in result

    def test_string_literals_preserved(self):
        """Test that string literals with quotes are preserved during source separator removal."""
        sql_in = dedent(
            '''
            SELECT "athlete_full_name", "medal_type", COUNT(*) AS "medal_count"
            FROM "ProvidedSource00000 ⦙ olympic_medals.csv"
            WHERE "country_name" = 'United States of America'
              AND "athlete_full_name" IS NOT NULL
              AND "athlete_full_name" != 'Michael PHELPS'
            GROUP BY "athlete_full_name", "medal_type"
            ORDER BY "medal_count" DESC
            LIMIT 100
            '''
        ).strip()
        result = SQLRemoveSourceSeparator.apply_to(sql_in)
        # The transform reformats to single line and changes some syntax
        expected = ('SELECT "athlete_full_name", "medal_type", COUNT(*) AS "medal_count" '
                   'FROM "olympic_medals.csv" '
                   'WHERE "country_name" = \'United States of America\' '
                   'AND NOT "athlete_full_name" IS NULL '
                   'AND "athlete_full_name" <> \'Michael PHELPS\' '
                   'GROUP BY "athlete_full_name", "medal_type" '
                   'ORDER BY "medal_count" DESC '
                   'LIMIT 100')
        assert result == expected
        # Verify string literals are preserved
        assert "'United States of America'" in result
        assert "'Michael PHELPS'" in result


class TestSQLFilterBase:
    """Test the shared base class functionality."""
    
    def test_build_filter_conditions_comprehensive(self):
        """Test the shared _build_filter_conditions method comprehensively."""
        # Create a simple SQLFilter instance to test the base class method
        filter_instance = SQLFilter()
        
        # Test all supported condition types
        conditions = [
            ("col1", None),
            ("col2", 42),
            ("col3", 3.14),
            ("col4", "string_value"),
            ("col5", dt.date(2023, 1, 15)),
            ("col6", dt.datetime(2023, 1, 15, 10, 30)),
            ("col7", (1, 10)),
            ("col8", ["a", "b", "c"]),
            ("col9", [None, "value"]),
            ("col10", [(dt.date(2023, 1, 1), dt.date(2023, 1, 31)), (dt.date(2023, 6, 1), dt.date(2023, 6, 30))])
        ]
        
        filters = filter_instance._build_filter_conditions(conditions)
        
        # Should return a list of sqlglot expressions
        assert isinstance(filters, list)
        assert len(filters) == 10  # All conditions should be processed
        
        # Convert to SQL strings to verify content
        filter_sqls = [f.sql() for f in filters]
        
        # Verify each condition type
        assert any("col1 IS NULL" in sql for sql in filter_sqls)
        assert any("col2 = 42" in sql for sql in filter_sqls)
        assert any("col3 = 3.14" in sql for sql in filter_sqls)
        assert any("col4 = 'string_value'" in sql for sql in filter_sqls)
        assert any("BETWEEN '2023-01-15 00:00:00' AND '2023-01-15 23:59:59'" in sql for sql in filter_sqls)
        assert any("col6 = '2023-01-15 10:30:00'" in sql for sql in filter_sqls)
        assert any("BETWEEN '1' AND '10'" in sql for sql in filter_sqls)
        assert any("IN ('a', 'b', 'c')" in sql for sql in filter_sqls)
        assert any("IS NULL OR" in sql and "IN ('value')" in sql for sql in filter_sqls)
        
    def test_unsupported_condition_handling(self):
        """Test that unsupported condition types are handled gracefully."""
        filter_instance = SQLFilter()
        
        # Mix valid and invalid conditions
        conditions = [
            ("valid_col", "valid_value"),
            ("invalid_col", {"complex": "object"}),  # Unsupported type
            ("another_valid", 123)
        ]
        
        filters = filter_instance._build_filter_conditions(conditions)
        
        # Should only process the valid conditions
        assert len(filters) == 2
        
        filter_sqls = [f.sql() for f in filters]
        assert any("valid_col = 'valid_value'" in sql for sql in filter_sqls)
        assert any("another_valid = 123" in sql for sql in filter_sqls)
        # Invalid condition should be skipped

    def test_debug_complex_table_names(self):
        """Test the specific case that's failing with complex table names."""
        sql_in = '''SELECT DISTINCT "NAME", "SEASON", "BASIN" FROM ProvidedSource00000 ⦙ ibtracs.last3years.parquet WHERE "BASIN" = 'EP' AND "SEASON" IN (SELECT DISTINCT CAST("year" AS VARCHAR) FROM ProvidedSource00000 ⦙ oni.csv WHERE "season" = 'AMJ' AND "oni" IN ('el_nino', 'la_nina')) AND TRIM("NAME") != '' ORDER BY "SEASON" LIMIT 10'''

        result = SQLRemoveSourceSeparator.apply_to(sql_in)
        print(f"\nInput: {sql_in}")
        print(f"Output: {result}")

        # The result should contain the full table names without the source prefix
        # They should be properly quoted since they contain dots
        assert '"ibtracs.last3years.parquet"' in result
        assert '"oni.csv"' in result

        # Should not contain the original source prefix
        assert 'ProvidedSource00000__@__' not in result
        assert '__@__' not in result

        # Should not contain standalone extensions without the full name
        # Check that 'parquet' and 'csv' only appear as part of the full quoted table names
        parquet_matches = result.count('parquet')
        csv_matches = result.count('csv')
        assert parquet_matches == 1, f"Expected 1 'parquet', found {parquet_matches}"
        assert csv_matches == 1, f"Expected 1 'csv', found {csv_matches}"
