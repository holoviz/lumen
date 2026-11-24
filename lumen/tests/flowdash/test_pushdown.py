import asyncio
import datetime as dt
import gc
import time

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import (
    DashboardOps, OperationError, dashboard_from_yaml, dashboard_to_yaml,
)

from lumen.flowdash import (
    COMPONENTS, DATA_ID, FILTER_ID, PREVIEW_ROWS, RESULT_LIMIT, SOURCE_ID,
    SQL_ID, TABLE_ID, VIEW_ID, bind_source, check_filter_config,
    check_sql_config, check_table_config, check_view_config, compile_sql,
    fetch, filtered_pipeline, pipeline_sql, python_value, set_row_ceiling,
    table_pipeline,
)
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.sources.duckdb import DuckDBSource

SPEC = {"mark": "bar", "encoding": {"x": {"field": "category", "type": "nominal"}, "y": {"field": "amount", "type": "quantitative"}}}


def output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


def failure(ops, name, port):
    value = output(ops, name, port)
    assert not isinstance(value, (Pipeline, pd.DataFrame))
    return str(getattr(value, "exception", value))


def instance(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def data_queries(executed):
    """The statements that fetched rows, leaving out the schema and stats queries a pipeline makes once."""
    return [q for q in executed if "__count" not in q and not (q.endswith(" LIMIT 1") and "subquery" not in q)]


def filtered_dashboard(source, widget="Widgets/Select", widget_config=None, limit=None):
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales", "limit": limit}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node(widget, widget_config or {"label": "Category", "default_options": ["A", "B"]}, name="pick")
    ops.add_node(FILTER_ID, {"field": "category"}, name="by_category")
    ops.connect("sales.pipeline", "by_category.pipeline")
    ops.connect("pick.selected", "by_category.value")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("by_category.filtered", "rows.pipeline")
    return ops


def rows(ops, name="rows"):
    return sorted(map(tuple, output(ops, name, "data").values.tolist()))


def test_table_and_filter_nodes_fetch_nothing_themselves(source, executed):
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node(FILTER_ID, {"field": "category"}, name="by_category")
    ops.connect("sales.pipeline", "by_category.pipeline")

    assert isinstance(output(ops, "by_category", "filtered"), Pipeline)
    assert data_queries(executed) == []


def test_filter_is_pushed_into_the_source_query(source, executed):
    ops = filtered_dashboard(source)
    instance(ops, "pick")._widget.value = "B"
    executed.clear()
    source.clear_cache()

    instance(ops, "pick")._widget.value = "A"

    assert rows(ops) == [("A", 5), ("A", 10)]
    statement = pipeline_sql(output(ops, "by_category", "filtered"))
    assert data_queries(executed) == [" ".join(statement.split())]
    assert "WHERE \"category\" = 'A'" in statement
    assert f"LIMIT {RESULT_LIMIT}" in statement


def test_list_and_range_values_are_pushed_down(source, executed):
    ops = filtered_dashboard(source, "Widgets/MultiChoice")
    instance(ops, "pick")._widget.value = ["A", "B"]
    assert rows(ops) == [("A", 5), ("A", 10), ("B", 20)]
    assert any("IN ('A', 'B')" in q for q in data_queries(executed))

    ops.set_config("by_category", {"field": "amount"})
    instance(ops, "pick")._widget.value = []
    executed.clear()
    pipeline = output(ops, "sales", "pipeline")
    ranged = filtered_pipeline(pipeline, "amount", (6, 20))
    assert sorted(map(tuple, fetch(ranged).values.tolist())) == [("A", 10), ("B", 20)]
    assert "\"amount\" BETWEEN '6' AND '20'" in data_queries(executed)[-1]


def test_empty_value_keeps_every_row(source, executed):
    ops = filtered_dashboard(source, "Widgets/MultiChoice")
    instance(ops, "pick")._widget.value = []

    assert output(ops, "by_category", "filtered") is output(ops, "sales", "pipeline")
    assert len(output(ops, "rows", "data")) == 3
    assert all("WHERE" not in q for q in data_queries(executed))


def test_table_limit_is_pushed_down_after_the_filter(source, executed):
    ops = filtered_dashboard(source, limit=1)
    instance(ops, "pick")._widget.value = "B"
    executed.clear()
    source.clear_cache()
    instance(ops, "pick")._widget.value = "A"

    assert len(output(ops, "rows", "data")) == 1
    (query,) = data_queries(executed)
    assert query.endswith("WHERE \"category\" = 'A' LIMIT 1")


def test_table_limit_cannot_exceed_host_ceiling(executed):
    big = DuckDBSource(uri=":memory:", tables={"big": "SELECT * FROM range(300000) AS t(i)"})
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, big)
    set_row_ceiling(editor, 1000)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "big", "limit": 10**9}, name="big")
    ops.connect("data.source", "big.source")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("big.pipeline", "rows.pipeline")

    assert len(output(ops, "rows", "data")) == 1000
    assert data_queries(executed)[-1].endswith("LIMIT 1000")
    ops.set_config("big", {"limit": 7})
    assert len(output(ops, "rows", "data")) == 7


def test_filter_on_a_sql_node_wraps_its_query_and_keeps_the_ceiling(source, executed):
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(SQL_ID, {"sql": "SELECT category, SUM(amount) AS total FROM sales GROUP BY category", "table": "totals"}, name="totals")
    ops.connect("data.source", "totals.source")
    ops.add_node("Widgets/Select", {"default_options": ["A", "B"]}, name="pick")
    ops.add_node(FILTER_ID, {"field": "category"}, name="by_category")
    ops.connect("totals.pipeline", "by_category.pipeline")
    ops.connect("pick.selected", "by_category.value")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("by_category.filtered", "rows.pipeline")
    instance(ops, "pick")._widget.value = "B"
    executed.clear()
    output(ops, "totals", "pipeline").source.clear_cache()

    instance(ops, "pick")._widget.value = "A"

    assert rows(ops) == [("A", 15)]
    (query,) = data_queries(executed)
    assert "GROUP BY category" in query
    assert query.endswith(f"WHERE \"category\" = 'A' LIMIT {RESULT_LIMIT}")


def test_filter_on_a_derived_pipeline_filters_its_rows():
    source = InMemorySource(tables={"sales": pd.DataFrame({"category": ["A", "B", "A"], "amount": [10, 20, 5]})})
    parent = table_pipeline(source, "sales")
    child = parent.chain()

    filtered = filtered_pipeline(child, "category", "B")

    assert filtered.pipeline is child
    assert fetch(filtered).values.tolist() == [["B", 20]]


def test_table_node_on_a_source_without_sql_cuts_rows_after_fetching():
    source = InMemorySource(tables={"sales": pd.DataFrame({"category": ["A", "B", "A"], "amount": [10, 20, 5]})})
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales", "limit": 2}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node("Widgets/Select", {"default_options": ["A", "B"]}, name="pick")
    ops.add_node(FILTER_ID, {"field": "category"}, name="by_category")
    ops.connect("sales.pipeline", "by_category.pipeline")
    ops.connect("pick.selected", "by_category.value")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("by_category.filtered", "rows.pipeline")
    instance(ops, "pick")._widget.value = "A"

    assert pipeline_sql(output(ops, "sales", "pipeline")) is None
    assert rows(ops) == [("A", 5), ("A", 10)]


def test_view_node_charts_the_filtered_rows(source, executed):
    ops = filtered_dashboard(source)
    ops.add_node(VIEW_ID, {"spec": SPEC}, name="chart")
    ops.connect("by_category.filtered", "chart.pipeline")
    instance(ops, "pick")._widget.value = "B"

    view = instance(ops, "chart")._object
    assert view.get_data()["category"].tolist() == ["B"]
    assert all("WHERE" in q for q in data_queries(executed)[-1:])


@pytest.mark.parametrize("node, config, port, error", [
    (TABLE_ID, {"table": "nope"}, "pipeline", "The source has no table 'nope'. Its tables are: sales."),
    (FILTER_ID, {"field": "nope"}, "filtered", "The data has no column 'nope'. Its columns are: category, amount."),
])
def test_nodes_report_unknown_tables_and_fields(source, node, config, port, error):
    ops = filtered_dashboard(source)
    name = "sales" if node == TABLE_ID else "by_category"
    with pytest.raises(OperationError, match="was not changed") as refused:
        ops.set_config(name, config)
    assert error in str(refused.value)
    assert "failed" not in str(output(ops, "rows", "data"))

    ops.editor._apply_config_patch(ops._instance(name), config)
    assert error in failure(ops, name, port)
    assert "failed" in failure(ops, "rows", "data")


def test_yaml_round_trip_keeps_table_and_filter_nodes(source, executed):
    ops = filtered_dashboard(source, limit=2)
    instance(ops, "pick")._widget.value = "A"
    text = dashboard_to_yaml(ops.editor)

    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    editor.import_dashboard(dashboard_from_yaml(text))
    imported = DashboardOps(editor)
    instance(imported, "pick")._widget.value = "B"

    nodes = yaml.safe_load(text)["nodes"]
    assert (nodes["sales"]["component"], nodes["sales"]["config"]) == (TABLE_ID, {"table": "sales", "limit": 2})
    assert (nodes["by_category"]["component"], nodes["by_category"]["config"]) == (FILTER_ID, {"field": "category"})
    assert rows(imported) == [("B", 20)]


def test_table_node_takes_no_sql(source):
    ops = filtered_dashboard(source)
    with pytest.raises(OperationError, match="no config field 'sql'"):
        ops.set_config("sales", {"sql": "SELECT * FROM sales"})


def test_config_checks_fetch_no_rows(source, executed):
    pipeline = table_pipeline(source, "sales")
    executed.clear()

    assert check_table_config({"table": "sales"}, {"source": source}) is None
    assert "no table 'orders'" in check_table_config({"table": "orders"}, {"source": source})
    assert check_filter_config({"field": "amount"}, {"pipeline": pipeline}) is None
    assert "no column 'price'" in check_filter_config({"field": "price"}, {"pipeline": pipeline})
    assert asyncio.run(check_sql_config({"sql": "SELECT * FROM sales"}, {"source": source})) is None
    assert "read-only" in asyncio.run(check_sql_config({"sql": "DROP TABLE sales"}, {"source": source}))
    assert "nope" in asyncio.run(check_sql_config({"sql": "SELECT nope FROM sales"}, {"source": source}))
    assert check_view_config({"spec": SPEC}, {"pipeline": pipeline}) is None
    assert "revenue" in check_view_config({"spec": {"mark": "bar", "encoding": {"x": {"field": "revenue"}}}}, {"pipeline": pipeline})
    assert asyncio.run(check_sql_config({"sql": "SELECT 1"}, {})) is None
    assert data_queries(executed) == ["SELECT * FROM sales LIMIT 0", "SELECT nope FROM sales LIMIT 0"]


def test_preview_fetches_only_when_shown_and_only_a_few_rows(executed):
    big = DuckDBSource(uri=":memory:", tables={"big": "SELECT * FROM range(1000) AS t(i)"})
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, big)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "big"}, name="big")
    ops.connect("data.source", "big.source")
    view = instance(ops, "big").__panel__()
    assert data_queries(executed) == []

    view.get_root()

    assert data_queries(executed)[-1].endswith(f"LIMIT {PREVIEW_ROWS}")
    assert len(view._inner_layout[0].value) == PREVIEW_ROWS


MIXED = """SELECT * FROM (VALUES
 ('A', 1, 1.5, TRUE, DATE '2024-01-01'),
 ('a', 2, 2.5, FALSE, DATE '2024-01-02'),
 (NULL, 3, NULL, NULL, NULL),
 ('B', 10, 10.0, TRUE, DATE '2024-02-01')
) AS t(s, i, f, b, d)"""


@pytest.mark.parametrize("filters, expected", [
    ([("s", "A")], [1]),
    ([("i", np.int64(2))], [2]),
    ([("f", 2.5)], [2]),
    ([("b", True)], [1, 10]),
    ([("b", np.bool_(False))], [2]),
    ([("b", [True])], [1, 10]),
    ([("s", ["A", "B"])], [1, 10]),
    ([("s", {"A", "B"})], [1, 10]),
    ([("s", ["A", None])], [1, 3]),
    ([("f", [1.5, 10.0])], [1, 10]),
    ([("i", (2, 10))], [2, 3, 10]),
    ([("i", (2, None))], [2, 3, 10]),
    ([("i", (None, 2))], [1, 2]),
    ([("d", (dt.date(2024, 1, 1), dt.date(2024, 1, 2)))], [1, 2]),
    ([("i", [1, 5])], [1]),
    ([("i", (1, 5)), ("i", (2, 20))], [2, 3]),
    ([("s", ["A", "B"]), ("i", (1, 5))], [1]),
])
def test_filters_keep_the_same_rows_pushed_down_in_memory_and_chained(filters, expected):
    duck = DuckDBSource(uri=":memory:", tables={"t": MIXED})
    memory = InMemorySource(tables={"t": duck.execute(MIXED)})
    pipelines = [table_pipeline(duck, "t"), table_pipeline(memory, "t"), table_pipeline(memory, "t").chain()]
    for pipeline in pipelines:
        for field, value in filters:
            pipeline = filtered_pipeline(pipeline, field, value)
        assert sorted(fetch(pipeline)["i"].tolist()) == expected


@pytest.mark.parametrize("value", [(1, 2, 3), ("1", "9"), (None, None), {"a": 1}, [(1, 2)], (True, False)])
def test_filter_refuses_values_it_cannot_apply(source, value):
    pipeline = table_pipeline(source, "sales")
    with pytest.raises(ValueError, match="Cannot filter on"):
        filtered_pipeline(pipeline, "amount", value)


def test_toggle_filters_a_boolean_column(executed):
    flags = DuckDBSource(uri=":memory:", tables={"t": "SELECT * FROM (VALUES (1, TRUE), (2, FALSE), (3, TRUE)) AS t(i, b)"})
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, flags)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "t"}, name="t")
    ops.connect("data.source", "t.source")
    ops.add_node("Widgets/Toggle", {}, name="flag")
    ops.add_node(FILTER_ID, {"field": "b"}, name="by_flag")
    ops.connect("t.pipeline", "by_flag.pipeline")
    ops.connect("flag.checked", "by_flag.value")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("by_flag.filtered", "rows.pipeline")

    instance(ops, "flag")._widget.value = True
    assert sorted(output(ops, "rows", "data")["i"].tolist()) == [1, 3]
    instance(ops, "flag")._widget.value = False
    assert output(ops, "rows", "data")["i"].tolist() == [2]
    assert data_queries(executed)[-1].endswith('WHERE "b" = FALSE LIMIT 100000')


def test_filter_node_reports_a_value_it_cannot_apply(source):
    ops = filtered_dashboard(source)
    instance(ops, "by_category").value = ("A", "B", "C")

    assert "Cannot filter on ('A', 'B', 'C')" in failure(ops, "by_category", "filtered")


def test_filter_on_a_sql_node_keeps_the_node_limit(source):
    pipeline = compile_sql(source, "SELECT * FROM sales ORDER BY amount", "q", limit=1)
    assert pipeline.data.values.tolist() == [["A", 5]]

    assert fetch(filtered_pipeline(pipeline, "category", "A")).values.tolist() == [["A", 5]]
    assert fetch(filtered_pipeline(pipeline, "category", "B")).empty


def test_fetch_stops_a_slow_query_after_the_timeout():
    slow = DuckDBSource(uri=":memory:", tables={
        "endless": "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT i FROM n WHERE i < 0",
    })
    pipeline = Pipeline(source=slow, table="endless", schema={"i": {"type": "integer"}})
    with pytest.raises(Exception, match="did not finish within 0.5 seconds"):
        fetch(pipeline, timeout=0.5)


def test_data_node_and_chart_share_one_query(source, executed):
    ops = filtered_dashboard(source)
    ops.add_node(VIEW_ID, {"spec": SPEC}, name="chart")
    ops.connect("by_category.filtered", "chart.pipeline")
    instance(ops, "pick")._widget.value = "B"
    executed.clear()
    source.clear_cache()

    instance(ops, "pick")._widget.value = "A"

    assert rows(ops) == [("A", 5), ("A", 10)]
    assert instance(ops, "chart")._object.get_data()["category"].tolist() == ["A", "A"]
    assert len(data_queries(executed)) == 1


def test_datetime64_and_timedelta64_values_become_python_values():
    stamp = np.datetime64("2024-01-02T00:00:00.000000000")
    assert python_value(stamp) == dt.datetime(2024, 1, 2)
    assert python_value((stamp, None)) == (dt.datetime(2024, 1, 2), None)
    assert python_value(np.timedelta64(90, "s")) == dt.timedelta(seconds=90)

    events = DuckDBSource(uri=":memory:", tables={"t": "SELECT * FROM (VALUES (1, TIMESTAMP '2024-01-01 12:00'), (2, TIMESTAMP '2024-01-02 00:00')) AS t(i, ts)"})
    pipeline = filtered_pipeline(table_pipeline(events, "t"), "ts", stamp)
    assert fetch(pipeline)["i"].tolist() == [2]


class TSQLSource(DuckDBSource):
    dialect = "tsql"


def test_filters_parse_and_write_the_source_dialect():
    source = TSQLSource(uri=":memory:", tables={"q": "SELECT TOP 10 * FROM t ORDER BY i"})
    pipeline = Pipeline(source=source, table="q", schema={"i": {"type": "integer"}, "b": {"type": "boolean"}})
    pipeline = filtered_pipeline(filtered_pipeline(pipeline, "i", (2, None)), "b", True)

    assert " ".join(pipeline_sql(pipeline).split()) == (
        "SELECT TOP 100000 * FROM (SELECT * FROM (SELECT TOP 10 * FROM t ORDER BY i) AS subquery "
        "WHERE [i] >= '2') AS subquery WHERE [b] = 1"
    )


def test_refresh_runs_the_nodes_again_on_new_rows():
    source = DuckDBSource(uri=":memory:", tables=["t"], initializers=["CREATE TABLE t AS SELECT * FROM (VALUES ('A', 1)) AS v(s, i)"])
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "t"}, name="t")
    ops.connect("data.source", "t.source")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("t.pipeline", "rows.pipeline")
    assert len(output(ops, "rows", "data")) == 1

    source._connection.execute("INSERT INTO t VALUES ('B', 2)")
    editor.param.trigger("refresh")

    assert len(output(ops, "rows", "data")) == 2


def test_fetch_without_sql_gives_up_after_the_timeout(monkeypatch):
    source = InMemorySource(tables={"t": pd.DataFrame({"i": [1]})})
    pipeline = table_pipeline(source, "t")
    monkeypatch.setattr(type(pipeline), "_compute_data", lambda self: time.sleep(2) or pd.DataFrame())

    with pytest.raises(TimeoutError, match="did not finish within 0.2 seconds"):
        fetch(pipeline, timeout=0.2)


def test_refresh_runs_each_data_query_once(executed):
    source = DuckDBSource(uri=":memory:", tables=["t"], initializers=["CREATE TABLE t AS SELECT * FROM (VALUES ('A', 1), ('B', 2)) AS v(s, i)"])
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "t"}, name="t")
    ops.connect("data.source", "t.source")
    ops.add_node("Widgets/Select", {"default_options": ["A", "B"]}, name="pick")
    ops.add_node(FILTER_ID, {"field": "s"}, name="by_s")
    ops.connect("t.pipeline", "by_s.pipeline")
    ops.connect("pick.selected", "by_s.value")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("by_s.filtered", "rows.pipeline")
    ops.add_node(VIEW_ID, {"spec": {"mark": "bar", "encoding": {"x": {"field": "s", "type": "nominal"}}}}, name="chart")
    ops.connect("by_s.filtered", "chart.pipeline")
    executed.clear()

    editor.param.trigger("refresh")

    assert len(data_queries(executed)) == 1
    for name in ("rows", "by_s", "t", "chart"):
        ops.remove_node(name)
    gc.collect()
    executed.clear()
    editor.param.trigger("refresh")
    assert executed == []
