import pytest

from lumen.data.contracts import (
    FULL, NONE, PARTIAL, ColumnProblem, compatible, family, query_columns,
    schema_problems, source_schema,
)

SCHEMA = {
    "customers": {"customer_id": "integer", "customer": "string", "region": "string"},
    "orders": {"order_id": "integer", "customer_id": "integer", "amount": "integer", "status": "string"},
}

JOIN = """
SELECT region, SUM(amount) AS total
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE status = 'paid' GROUP BY region ORDER BY total
"""


def columns(found):
    return {table: sorted(names) for table, names in found.tables.items()}


def test_join_columns_resolve_to_their_tables_with_the_schema():
    found = query_columns(JOIN, SCHEMA, "duckdb")
    assert found.coverage == FULL
    assert columns(found) == {
        "orders": ["amount", "customer_id", "status"], "customers": ["customer_id", "region"],
    }


def test_unqualified_join_columns_without_a_schema_are_unresolved():
    found = query_columns(JOIN, None, "duckdb")
    assert found.coverage == NONE
    assert found.unresolved == {"region", "amount", "status"}
    assert set(found.tables) == {"orders", "customers"}


def test_a_single_table_needs_no_schema_and_ctes_and_aliases_are_not_columns():
    sql = """
    WITH paid AS (SELECT customer_id, amount FROM orders WHERE status = 'paid')
    SELECT customer_id, SUM(amount) AS total FROM paid GROUP BY customer_id ORDER BY total DESC
    """
    found = query_columns(sql, dialect="duckdb")
    assert found.coverage == FULL
    assert columns(found) == {"orders": ["amount", "customer_id", "status"]}


def test_subqueries_are_followed_to_their_tables():
    sql = "SELECT n FROM (SELECT COUNT(order_id) AS n FROM orders WHERE amount > 10) AS t"
    assert columns(query_columns(sql, SCHEMA, "duckdb")) == {"orders": ["amount", "order_id"]}


def test_names_take_the_schemas_spelling():
    found = query_columns('SELECT "Region" FROM Customers', {"customers": {"region": "string"}}, "duckdb")
    assert columns(found) == {"customers": ["region"]}


def test_a_qualified_column_the_schema_lacks_is_kept():
    found = query_columns("SELECT c.area FROM customers c", SCHEMA, "duckdb")
    assert columns(found) == {"customers": ["area"]}


def test_select_star_is_partly_covered_and_count_star_is_not():
    assert query_columns("SELECT * FROM orders", SCHEMA, "duckdb").coverage == PARTIAL
    assert query_columns("SELECT o.*, c.region FROM orders o JOIN customers c USING (customer_id)",
                         SCHEMA, "duckdb").coverage == PARTIAL
    assert query_columns("SELECT COUNT(*) AS n FROM orders", SCHEMA, "duckdb").coverage == FULL


def test_sql_that_does_not_parse_is_not_covered():
    found = query_columns("SELEC region FROM", SCHEMA, "duckdb")
    assert found.coverage == NONE and "does not parse" in found.reason


def test_families():
    assert family("int64") == "integer" and family("float64") == "number"
    assert family("object") == "string" and family("datetime64[ns]") == "datetime"
    assert family("bool") == "boolean"
    assert family("VARCHAR") == "string" and family("DECIMAL(10,2)") == "number"
    assert family("TIMESTAMP") == "datetime" and family("BLOB") == "other"
    assert compatible("integer", "number") and compatible("number", "integer")
    assert compatible("other", "string") and not compatible("integer", "string")


def test_schema_problems():
    expected = {"orders": {"amount": "integer", "status": "string"}, "customers": {"region": "string"}}
    schema = {"Orders": {"AMOUNT": "string", "status": "string"}}
    assert schema_problems(expected, schema) == [
        ColumnProblem("orders", "amount", "Column orders.amount is string now; the contract expects integer."),
        ColumnProblem("customers", None, "The data has no table 'customers'."),
    ]
    assert schema_problems(expected, {"orders": {"amount": "number", "status": "string"},
                                      "customers": {"region": "string", "extra": "integer"}}) == []
    assert schema_problems({"orders": {"gone": "string"}}, schema) == [
        ColumnProblem("orders", "gone", "Table 'orders' has no column 'gone'."),
    ]


def test_source_schema_fetches_no_rows(monkeypatch):
    duckdb_source = pytest.importorskip("lumen.sources.duckdb")
    tables = {
        "orders": "SELECT * FROM (VALUES (101, 1, 30, 'paid')) AS t(order_id, customer_id, amount, status)",
    }
    source = duckdb_source.DuckDBSource(uri=":memory:", tables=tables)
    executed = []
    execute = source.execute
    monkeypatch.setattr(source, "execute", lambda sql, *a, **k: executed.append(sql) or execute(sql, *a, **k))
    assert source_schema(source, ["ORDERS", "missing"]) == {"ORDERS": SCHEMA["orders"]}
    assert executed and all(sql.endswith("LIMIT 0") for sql in executed)
