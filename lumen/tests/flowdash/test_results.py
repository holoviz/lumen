import sqlite3

import pytest

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps

from lumen.flowdash import (
    COMPONENTS, DATA_ID, SOURCE_ID, SQL_ID, TABLE_ID, bind_source,
    check_sql_config, set_row_ceiling,
)
from lumen.flowdash.nodes import sql_completions
from lumen.flowdash.results import expand, is_derived, root_of
from lumen.sources.duckdb import DuckDBSource
from lumen.sources.sqlalchemy import SQLAlchemySource

ORDERS = [
    (1, "Ada", "east", 50, "paid"), (2, "Bo", "east", 15, "paid"),
    (3, "Cy", "west", 40, "paid"), (4, "Ada", "east", 7, "open"),
]
PAID = "SELECT customer, region, amount FROM orders WHERE status = 'paid'"
BY_REGION = "SELECT region, SUM(amount) AS total FROM paid GROUP BY region ORDER BY region"


def output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


def rows(ops, name):
    return [tuple(row) for row in output(ops, name, "data").values.tolist()]


def duckdb_source():
    values = ", ".join(f"({i}, '{c}', '{r}', {a}, '{s}')" for i, c, r, a, s in ORDERS)
    return DuckDBSource(uri=":memory:", tables={
        "orders": f"SELECT * FROM (VALUES {values}) AS t(id, customer, region, amount, status)"
    })


def sqlite_source(tmp_path):
    path = tmp_path / "orders.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE orders (id INTEGER, customer TEXT, region TEXT, amount INTEGER, status TEXT)")
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", ORDERS)
    return SQLAlchemySource(url=f"sqlite:///{path}")


@pytest.fixture(params=["duckdb", "sqlite"])
def orders(request, tmp_path):
    return duckdb_source() if request.param == "duckdb" else sqlite_source(tmp_path)


def chain(source, *steps):
    """A source node and SQL nodes, each (name, sql) reading the result of the one before it."""
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    upstream = "data.source"
    for name, sql in steps:
        ops.add_node(SQL_ID, {"sql": sql, "table": name}, name=name)
        ops.connect(upstream, f"{name}.source")
        upstream = f"{name}.result"
    return ops


def test_a_sql_node_queries_an_earlier_result_by_table_name(orders):
    ops = chain(orders, ("paid", PAID), ("by_region", BY_REGION))

    assert rows(ops, "by_region") == [("east", 65), ("west", 40)]
    result = output(ops, "paid", "result")
    assert is_derived(result) and root_of(result) is orders
    assert "paid" in result.get_tables() and "orders" in result.get_tables()


def test_a_follower_runs_one_statement_on_the_root_source(executed):
    source = duckdb_source()
    ops = chain(source, ("paid", PAID), ("by_region", BY_REGION))
    executed.clear()

    ops.set_config("by_region", {"sql": BY_REGION + " LIMIT 1"})

    assert rows(ops, "by_region") == [("east", 65)]
    statement = next(q for q in executed if "by_region" not in q and "SUM(amount)" in q)
    assert statement.startswith("WITH paid AS (SELECT customer, region, amount FROM orders")


def test_a_chain_of_three_runs_and_a_table_node_reads_a_result(orders):
    top = "SELECT region FROM by_region WHERE total = (SELECT MAX(total) FROM by_region)"
    ops = chain(orders, ("paid", PAID), ("by_region", BY_REGION), ("top", top))
    ops.add_node(TABLE_ID, {"table": "by_region"}, name="table")
    ops.connect("by_region.result", "table.source")
    ops.add_node(DATA_ID, {}, name="table_rows")
    ops.connect("table.pipeline", "table_rows.pipeline")

    assert rows(ops, "top") == [("east",)]
    assert sorted(rows(ops, "table_rows")) == [("east", 65), ("west", 40)]


def test_a_follower_reads_the_full_query_not_the_capped_statement(orders):
    ops = chain(orders, ("paid", PAID), ("n", "SELECT COUNT(*) AS n FROM paid"))
    set_row_ceiling(ops.editor, 2)
    ops.set_config("paid", {"sql": PAID + " "})

    assert len(output(ops, "paid", "data")) == 2
    assert rows(ops, "n") == [(3,)]


def test_editing_the_upstream_updates_the_follower(orders):
    ops = chain(orders, ("paid", PAID), ("by_region", BY_REGION))

    ops.set_config("paid", {"sql": PAID + " AND amount > 20"})

    assert rows(ops, "by_region") == [("east", 50), ("west", 40)]


def test_a_follower_uses_the_renamed_result_table():
    source = duckdb_source()
    ops = chain(source, ("orders", PAID), ("n", "SELECT COUNT(*) AS n FROM orders_result"))

    assert output(ops, "orders", "pipeline").table == "orders_result"
    assert rows(ops, "n") == [(3,)]


def test_a_follower_cannot_read_a_result_it_is_not_connected_to(orders):
    ops = chain(orders, ("paid", PAID))
    ops.add_node(SQL_ID, {"sql": "SELECT * FROM paid", "table": "other"}, name="other")
    ops.connect("data.source", "other.source")

    error = getattr(output(ops, "other", "data"), "exception", None)
    assert "not a table of this source" in str(error)


async def test_check_sql_config_probes_a_follower_on_the_root_source():
    source = duckdb_source()
    ops = chain(source, ("paid", PAID))
    result = output(ops, "paid", "result")

    assert await check_sql_config({"sql": BY_REGION}, {"source": result}) is None
    problem = await check_sql_config({"sql": "SELECT nope FROM paid"}, {"source": result})
    assert "nope" in str(problem)


def test_expand_leaves_a_query_on_a_root_source_alone():
    source = duckdb_source()
    assert expand(PAID, source) == (source, PAID)


def test_a_refresh_runs_each_query_of_a_chain_once(executed):
    source = duckdb_source()
    ops = chain(source, ("paid", PAID), ("by_region", BY_REGION))
    executed.clear()

    ops.editor.param.trigger("refresh")

    assert len([q for q in executed if not q.endswith("LIMIT 0")]) == 2
    assert rows(ops, "by_region") == [("east", 65), ("west", 40)]


def test_completions_offer_a_results_columns_without_running_it(executed):
    ops = chain(duckdb_source(), ("by_region", "SELECT region, SUM(amount) AS total FROM orders GROUP BY region"))
    result = output(ops, "by_region", "result")
    executed.clear()

    words = sql_completions({"sql": ""}, {"source": result})["sql"]

    assert set(words.values()) >= {"orders", "by_region", "total"}
    assert words.types["by_region"] == "table"
    assert not [q for q in executed if "SUM(amount)" in q]
