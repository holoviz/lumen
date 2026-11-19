import pytest

from lumen.data.contracts import (
    FULL, NONE, PARTIAL, ColumnProblem, compatible, family, query_columns,
    schema_problems, source_schema, table_schema,
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
    assert family("datetime64[ns]") == "datetime" and family("bool") == "boolean"
    assert family("object") == "other" and family("string") == "string"
    assert family("VARCHAR") == "string" and family("DECIMAL(10,2)") == "number"
    assert family("NUMBER(38,0)") == "integer" and family("NUMBER(38, 2)") == "number"
    assert family("DATE") == "date" and family("TIMESTAMP WITH TIME ZONE") == "datetime"
    assert family("POINT") == "other" and family("BLOB") == "other" and family("INTERVAL") == "other"
    assert family("INTEGER") == "integer" and family("BIGINT") == "integer"


def test_compatibility_is_one_way():
    assert compatible("number", "integer") and not compatible("integer", "number")
    assert compatible("datetime", "date") and not compatible("date", "datetime")
    assert compatible("other", "string") and not compatible("integer", "string")


def test_table_functions_and_pivots_are_partly_covered():
    found = query_columns("SELECT a, b FROM read_parquet('x.parquet')", dialect="duckdb")
    assert (found.coverage, found.tables) == (PARTIAL, {})
    found = query_columns(
        "SELECT o.amount, p.b FROM orders o JOIN read_parquet('x') p ON o.order_id = p.id", SCHEMA, "duckdb"
    )
    assert found.coverage == PARTIAL and columns(found) == {"orders": ["amount", "order_id"]}
    found = query_columns("SELECT * FROM read_parquet('x')", dialect="duckdb")
    assert found.coverage == PARTIAL and "" not in found.tables
    found = query_columns("SELECT x FROM generate_series(1, 3) AS g(x)", dialect="duckdb")
    assert found.coverage == PARTIAL
    found = query_columns("SELECT * FROM orders PIVOT (SUM(amount) FOR status IN ('paid', 'x'))",
                          dialect="duckdb")
    assert found.coverage == PARTIAL and columns(found) == {"orders": ["amount", "status"]}


def test_qualify_windows_are_read():
    sql = "SELECT order_id FROM orders QUALIFY ROW_NUMBER() OVER (PARTITION BY status ORDER BY amount) = 1"
    found = query_columns(sql, dialect="duckdb")
    assert found.coverage == FULL
    assert columns(found) == {"orders": ["amount", "order_id", "status"]}


@pytest.mark.parametrize("sql, expected", [
    ("SELECT status AS s, COUNT(*) AS n FROM orders GROUP BY s HAVING n > 1", {"orders": ["status"]}),
    ("SELECT amount FROM orders WHERE order_id IN (SELECT order_id FROM refunds)",
     {"orders": ["amount", "order_id"], "refunds": ["order_id"]}),
    ("WITH t AS (SELECT * FROM orders) SELECT amount FROM t WHERE status = 'paid'",
     {"orders": ["amount", "status"]}),
    ("SELECT customer_id AS amount FROM orders WHERE amount > 0", {"orders": ["amount", "customer_id"]}),
])
def test_single_tables_need_no_schema(sql, expected):
    found = query_columns(sql, dialect="duckdb")
    assert found.coverage in (FULL, PARTIAL) and not found.unresolved
    assert columns(found) == expected


def test_schema_prefixes_keep_tables_apart():
    found = query_columns("SELECT t.a, u.b FROM s1.t JOIN s2.t AS u ON t.x = u.x", dialect="duckdb")
    assert columns(found) == {"s1.t": ["a", "x"], "s2.t": ["b", "x"]}


def test_schema_problems():
    expected = {"orders": {"amount": "integer", "status": "string"}, "customers": {"region": "string"}}
    schema = {"Orders": {"AMOUNT": "string", "status": "string"}}
    assert schema_problems(expected, schema) == [
        ColumnProblem("orders", "amount", "Column orders.amount is string now; the contract expects integer."),
        ColumnProblem("customers", None, "The data has no table 'customers'."),
    ]
    assert schema_problems(expected, {"orders": {"amount": "integer", "status": "string"},
                                      "customers": {"region": "string", "extra": "integer"}}) == []
    assert schema_problems({"orders": {"gone": "string"}}, schema) == [
        ColumnProblem("orders", "gone", "Table 'orders' has no column 'gone'."),
    ]


def test_duckdb_tables_are_described_without_fetching_rows(monkeypatch):
    duckdb_source = pytest.importorskip("lumen.sources.duckdb")
    tables = {
        "orders": "SELECT * FROM (VALUES (101, 1, 30, 'paid')) AS t(order_id, customer_id, amount, status)",
        "days": "SELECT DATE '2026-01-01' AS day, TIMESTAMP '2026-01-01 10:00' AS at, 2.5 AS x",
    }
    source = duckdb_source.DuckDBSource(uri=":memory:", tables=tables)
    executed = []
    execute = source.execute
    monkeypatch.setattr(source, "execute", lambda sql, *a, **k: executed.append(sql) or execute(sql, *a, **k))
    assert source_schema(source, ["ORDERS", "days", "missing"]) == {
        "ORDERS": SCHEMA["orders"], "days": {"day": "date", "at": "datetime", "x": "number"},
    }
    assert executed and all("LIMIT 0" in sql for sql in executed)


def test_sqlalchemy_tables_are_reflected(tmp_path):
    pytest.importorskip("sqlalchemy")
    import sqlite3

    from lumen.sources.sqlalchemy import SQLAlchemySource

    path = tmp_path / "shop.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE orders (id INTEGER, amount REAL, status TEXT, day DATE)")
    source = SQLAlchemySource(url=f"sqlite:///{path}")
    assert source_schema(source) == {
        "orders": {"id": "integer", "amount": "number", "status": "string", "day": "date"},
    }
    view = SQLAlchemySource(url=f"sqlite:///{path}", tables={"paid": "SELECT id FROM orders"})
    assert source_schema(view) == {"paid": {"id": "other"}}


def test_a_table_that_cannot_be_read_is_left_out(caplog):
    duckdb_source = pytest.importorskip("lumen.sources.duckdb")
    source = duckdb_source.DuckDBSource(uri=":memory:", tables={
        "ok": "SELECT 1 AS a", "broken": "SELECT * FROM nowhere",
    })
    assert source_schema(source) == {"ok": {"a": "integer"}}
    assert "Could not read the columns of table 'broken'" in caplog.text
    with pytest.raises(Exception):
        table_schema(source, "broken")


def test_a_bare_name_finds_a_table_with_a_schema_prefix():
    class Prefixed:
        def get_tables(self):
            return ["public.orders", "public.customers", "audit.customers"]

        def get_schema(self, table, limit=None):
            return {"id": {"type": "integer"}}

    assert source_schema(Prefixed(), ["orders", "customers"]) == {"orders": {"id": "integer"}}
