import sqlite3

import pytest
import yaml

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, OperationError

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


def test_a_refresh_runs_a_table_node_on_a_result_once(executed):
    source = duckdb_source()
    ops = chain(source, ("paid", PAID))
    ops.add_node(TABLE_ID, {"table": "paid"}, name="table")
    ops.connect("paid.result", "table.source")
    ops.add_node(DATA_ID, {}, name="table_rows")
    ops.connect("table.pipeline", "table_rows.pipeline")
    assert len(output(ops, "table_rows", "data")) == 3
    executed.clear()

    ops.editor.param.trigger("refresh")

    assert len(output(ops, "table_rows", "data")) == 3
    assert len([q for q in executed if not q.endswith("LIMIT 0")]) == 2


def test_a_result_table_is_renamed_from_a_table_differing_only_in_case(tmp_path):
    path = tmp_path / "cased.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE Orders (id INTEGER, customer TEXT, region TEXT, amount INTEGER, status TEXT)")
        conn.executemany("INSERT INTO Orders VALUES (?, ?, ?, ?, ?)", ORDERS)
    source = SQLAlchemySource(url=f"sqlite:///{path}")
    ops = chain(source, ("orders", PAID.replace("FROM orders", "FROM Orders")), ("n", "SELECT COUNT(*) AS n FROM orders_result"))

    assert output(ops, "orders", "pipeline").table == "orders_result"
    assert rows(ops, "n") == [(3,)]


def test_the_stored_result_schema_keeps_types_but_no_values():
    ops = chain(duckdb_source(), ("paid", PAID))
    stored = output(ops, "paid", "result")._get_schema_cache()["paid"]
    assert set(stored) == {"customer", "region", "amount"}
    assert all(set(spec) <= {"type", "format"} for spec in stored.values())


CUSTOMERS = "SELECT region, COUNT(DISTINCT customer) AS customers FROM paid GROUP BY region"
JOIN = (
    "SELECT r.region, c.customers, r.total FROM by_region AS r JOIN customers_per_region AS c "
    "ON r.region = c.region ORDER BY r.region"
)


def joined(source):
    """paid, then by_region and customers_per_region on it, then a node joining those two."""
    ops = chain(source, ("paid", PAID), ("by_region", BY_REGION))
    ops.add_node(SQL_ID, {"sql": CUSTOMERS, "table": "customers_per_region"}, name="customers_per_region")
    ops.connect("paid.result", "customers_per_region.source")
    ops.add_node(SQL_ID, {"sql": JOIN, "table": "joined"}, name="joined")
    ops.connect("by_region.result", "joined.source")
    ops.connect("customers_per_region.result", "joined.source")
    return ops


def test_a_sql_node_joins_two_earlier_results(orders):
    ops = joined(orders)

    assert rows(ops, "joined") == [("east", 2, 65), ("west", 1, 40)]
    assert {"by_region", "customers_per_region", "paid", "orders"} <= set(output(ops, "joined", "result").get_tables())


def test_a_refresh_runs_a_join_once_per_input_and_the_rest_once(executed):
    ops = joined(duckdb_source())
    executed.clear()

    ops.editor.param.trigger("refresh")

    # The join runs at most once per input: once when FlowDash hands it its new inputs together,
    # otherwise once as each arrives. Every other node runs once.
    queries = [q for q in executed if not q.endswith("LIMIT 0")]
    joins = sum("JOIN customers_per_region" in q for q in queries)
    assert joins in (1, 2) and len(queries) == 3 + joins
    assert rows(ops, "joined") == [("east", 2, 65), ("west", 1, 40)]


def test_joining_results_of_different_sources_is_refused():
    other = duckdb_source()
    ops = chain(duckdb_source(), ("paid", PAID))
    bind_source(ops.editor, other, "other")
    ops.add_node(SOURCE_ID, {"ref": "other"}, name="other")
    ops.add_node(SQL_ID, {"sql": "SELECT region FROM orders", "table": "regions"}, name="regions")
    ops.connect("other.source", "regions.source")
    ops.add_node(SQL_ID, {"sql": "SELECT * FROM paid, regions", "table": "both"}, name="both")
    ops.connect("paid.result", "both.source")
    ops.connect("regions.result", "both.source")

    error = str(getattr(output(ops, "both", "data"), "exception", ""))
    assert "Joining results of different sources is not supported" in error


def test_joining_two_results_with_one_table_name_is_refused():
    ops = chain(duckdb_source(), ("paid", PAID))
    for name, sql in (("big", "SELECT * FROM paid WHERE amount > 20"), ("small", "SELECT * FROM paid WHERE amount < 20")):
        ops.add_node(SQL_ID, {"sql": sql, "table": "picked"}, name=name)
        ops.connect("paid.result", f"{name}.source")
    ops.add_node(SQL_ID, {"sql": "SELECT * FROM picked", "table": "both"}, name="both")
    ops.connect("big.result", "both.source")
    ops.connect("small.result", "both.source")

    error = str(getattr(output(ops, "both", "data"), "exception", ""))
    assert "Two inputs hold different results named 'picked'" in error


async def test_the_config_check_and_completions_see_every_joined_result():
    ops = joined(duckdb_source())
    inputs = {"source": [output(ops, "by_region", "result"), output(ops, "customers_per_region", "result")]}

    assert await check_sql_config({"sql": JOIN}, inputs) is None
    assert "nope" in str(await check_sql_config({"sql": "SELECT nope FROM by_region"}, inputs))
    words = sql_completions({"sql": ""}, inputs)["sql"]
    assert {"by_region", "customers_per_region", "customers", "total"} <= set(words.values())


def test_a_pipeline_on_the_source_input_is_refused_naming_its_edge():
    ops = chain(duckdb_source(), ("paid", PAID))
    ops.add_node(SQL_ID, {"sql": "SELECT * FROM paid", "table": "both"}, name="both")
    ops.connect("paid.result", "both.source")
    try:
        ops.connect("paid.pipeline", "both.source")
    except OperationError as exc:
        # panel-flowdash type-checks the members of a multiple input itself.
        assert "Type mismatch" in str(exc)
        return

    error = str(getattr(output(ops, "both", "data"), "exception", ""))
    assert "source takes a Lumen SQL source or a SQL node's result; paid.pipeline gives a Lumen pipeline." in error


@pytest.mark.parametrize("table", ["Sales Data", "paid-orders", "_paid"])
async def test_a_new_result_table_name_must_be_one_duckdb_keeps(table):
    problem = await check_sql_config({"sql": PAID, "table": table}, {})
    assert "should be letters, digits and underscores" in str(problem)
    assert await check_sql_config({"sql": PAID, "table": "PaidOrders"}, {}) is None


def saved(source, nodes, edges):
    """A dashboard loaded as saved, with configs no check has seen."""
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    doc = {"flowdash": 4, "nodes": {"data": {"component": SOURCE_ID, "config": {"ref": "data"}}}, "edges": []}
    for name, sql, table in nodes:
        doc["nodes"][name] = {"component": SQL_ID, "config": {"sql": sql, "table": table}}
    doc["edges"] = [{"from": a, "to": b} for a, b in edges]
    ops.replace_yaml(yaml.safe_dump(doc), revision=ops.revision())
    return ops


def test_a_saved_node_with_a_mixed_case_table_and_a_follower_stays_editable():
    ops = saved(duckdb_source(), [("paid", PAID, "PaidOrders"), ("n", "SELECT COUNT(*) AS n FROM PaidOrders", "n")],
                [("data.source", "paid.source"), ("paid.result", "n.source")])
    assert rows(ops, "n") == [(3,)]

    ops.set_config("paid", {"sql": PAID + " AND amount > 20"})

    assert rows(ops, "n") == [(2,)]


def test_a_saved_node_with_a_table_name_duckdb_changes_stays_editable_with_a_warning():
    ops = saved(duckdb_source(), [("sales", PAID, "Sales Data")], [("data.source", "sales.source")])

    ops.set_config("sales", {"sql": PAID + " AND amount > 20"})

    assert len(output(ops, "sales", "data")) == 2
    with pytest.raises(OperationError, match="should be letters, digits and underscores"):
        ops.set_config("sales", {"table": "More Sales"})


async def test_a_stored_table_name_duckdb_changes_is_a_warning(monkeypatch):
    from panel_flowdash import CheckWarning

    from lumen.flowdash import checks

    monkeypatch.setattr(checks, "stored_config", lambda: {"sql": PAID, "table": "Sales Data"})
    problem = await check_sql_config({"sql": PAID + " AND amount > 20", "table": "Sales Data"}, {})
    assert isinstance(problem, CheckWarning) and "should be letters, digits and underscores" in problem
