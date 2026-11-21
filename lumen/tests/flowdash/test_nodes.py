import time

from pathlib import Path

import pytest
import yaml

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import (
    DashboardOps, OperationError, dashboard_from_yaml, dashboard_to_yaml,
)

from lumen.flowdash import (
    COMPONENTS, NODE_TIMEOUT, RESULT_LIMIT, SOURCE_ID, SQL_ID, VIEW_ID,
    bind_source, check_chart, check_query, compile_sql, primed_result,
    result_sql, set_row_ceiling,
)
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource

QUERY = "SELECT category, SUM(amount) AS total FROM sales GROUP BY category"
SPEC = {"mark": "bar", "encoding": {"x": {"field": "category", "type": "nominal"}, "y": {"field": "total", "type": "quantitative"}}}


def output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


def failure(ops, name, port="data"):
    value = output(ops, name, port)
    assert not isinstance(value, Pipeline)
    return str(getattr(value, "exception", value))


def without_turns(text):
    data = yaml.safe_load(text)
    data.pop("turns", None)
    return data


def view_error(ops, name):
    return ops.editor._component_instances[ops._instance(name)]._error or ""


def editor_with(source, ref="data"):
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source, ref)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": ref}, name=ref)
    return ops


def dashboard(source, sql=QUERY):
    ops = editor_with(source)
    ops.add_node(SQL_ID, {"sql": sql, "table": "by_category"}, name="by_category")
    ops.connect("data.source", "by_category.source")
    ops.add_node(VIEW_ID, {"spec": SPEC}, name="chart")
    ops.connect("by_category.pipeline", "chart.pipeline")
    return ops


def test_sql_node_runs_query_once_in_source_engine(source, executed):
    ops = dashboard(source)

    pipeline = output(ops, "by_category", "pipeline")
    data = output(ops, "by_category", "data")

    assert isinstance(pipeline, Pipeline)
    assert output(ops, "chart", "pipeline") is pipeline
    assert sorted(map(tuple, data.values.tolist())) == [("A", 15), ("B", 20)]
    assert executed == [" ".join(result_sql(source, QUERY).split())]


def test_primed_result_is_not_executed_again(source, executed):
    statement = result_sql(source, QUERY)
    frame = source.execute(statement)
    executed.clear()
    with primed_result(source, statement, frame):
        pipeline = compile_sql(source, QUERY, "by_category")
    assert pipeline.data is frame
    assert executed == []


def test_sql_node_limit_caps_rows(source):
    ops = dashboard(source, "SELECT * FROM sales")
    ops.set_config("by_category", {"limit": 2})
    assert len(output(ops, "by_category", "data")) == 2


def test_sql_node_limit_cannot_exceed_host_ceiling():
    big = DuckDBSource(uri=":memory:", tables={"big": "SELECT * FROM range(300000) AS t(i)"})
    ops = editor_with(big)
    ops.add_node(SQL_ID, {"sql": "SELECT * FROM big", "limit": 10**12}, name="rows")
    ops.connect("data.source", "rows.source")
    assert len(output(ops, "rows", "data")) == RESULT_LIMIT

    set_row_ceiling(ops.editor, 200_000)
    ops.set_config("rows", {"limit": None})
    assert len(output(ops, "rows", "data")) == 200_000
    ops.set_config("rows", {"limit": 50})
    assert len(output(ops, "rows", "data")) == 50


def test_node_path_stops_slow_query_after_node_timeout(source):
    ops = dashboard(source)
    endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n, sales"
    started = time.monotonic()
    ops.set_config("by_category", {"sql": endless})
    assert "did not finish" in failure(ops, "by_category")
    assert time.monotonic() - started < NODE_TIMEOUT + 5


def test_sql_node_reports_failed_source(source):
    ops = editor_with(source)
    ops.add_node(SOURCE_ID, {"ref": "missing"}, name="other")
    ops.add_node(SQL_ID, {"sql": QUERY}, name="q")
    ops.connect("other.source", "q.source")
    assert "No source is bound as 'missing'" in failure(ops, "q")


def test_sql_node_times_out(source):
    endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n, sales"
    with pytest.raises(Exception, match="did not finish within 0.5 seconds"):
        compile_sql(source, endless, "endless", timeout=0.5)


def test_sql_node_table_does_not_shadow_source_table(source):
    ops = editor_with(source)
    ops.add_node(SQL_ID, {"sql": "SELECT * FROM sales WHERE amount > 5", "table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")

    assert len(output(ops, "sales", "data")) == 2


@pytest.mark.parametrize("sql, error", [
    ("DROP TABLE sales", "read-only"),
    ("SELECT 1; DELETE FROM sales", "read-only"),
    ("CREATE TABLE copy AS SELECT * FROM sales", "read-only"),
    ("INSERT INTO sales VALUES ('C', 1)", "read-only"),
    ("SELECT 42 AS answer", "reads no table"),
    ("SELECT * FROM read_text('/etc/hosts')", "READ_TEXT"),
    ("SELECT * FROM read_blob('/etc/hosts')", "READ_BLOB"),
    ("SELECT * FROM glob('/etc/*')", "GLOB"),
    ("SELECT * FROM read_csv('/etc/hosts')", "READ_CSV"),
    ("SELECT * FROM '/etc/hosts'", "not a table of this source"),
    ("SELECT * FROM 'https://example.invalid/x.parquet'", "not a table of this source"),
    ("SELECT * FROM duckdb_settings()", "duckdb_settings"),
    ("SELECT * FROM information_schema.tables", "information_schema.tables"),
    ("SELECT * FROM other_table", "not a table of this source"),
    ("WITH x AS (SELECT 1 AS x) SELECT * FROM x", "reads no table"),
    ("WITH a AS (SELECT 1 AS x), b AS (SELECT * FROM a) SELECT * FROM b", "reads no table"),
    ("WITH sales AS (SELECT 42 AS total) SELECT * FROM sales", "reads no table"),
    ("WITH unused AS (SELECT * FROM sales) SELECT 42 AS answer FROM (SELECT 1) AS t", "reads no table"),
    ("SELECT current_setting('home_directory') AS h FROM sales", "current_setting"),
    ("SELECT getenv('HOME') AS h FROM sales", "getenv"),
    ("SELECT * FROM other.sales", "not a table of this source"),
])
def test_sql_node_refuses_unsafe_queries_on_every_path(source, executed, sql, error):
    ops = dashboard(source, sql)
    assert error in failure(ops, "by_category")
    assert ops.preview()["nodes"]["by_category"]["outputs"]["data"]["type"] == "failure"

    ops.set_config("by_category", {"sql": QUERY})
    assert len(output(ops, "by_category", "data")) == 2
    ops.set_config("by_category", {"sql": sql})
    assert error in failure(ops, "by_category")

    text = dashboard_to_yaml(ops.editor)
    imported = FlowDash(components=COMPONENTS)
    bind_source(imported, source)
    imported.import_dashboard(dashboard_from_yaml(text))
    assert error in failure(DashboardOps(imported), "by_category")
    assert not any(word in query.upper() for query in executed for word in ("DROP", "INSERT", "CREATE", "DELETE", "/ETC/", "HTTPS:", "DUCKDB_SETTINGS", "INFORMATION_SCHEMA"))


@pytest.mark.parametrize("sql", [
    "SELECT * FROM main.sales",
    "SELECT * FROM memory.main.sales",
    "WITH a AS (SELECT * FROM sales), b AS (SELECT * FROM a) SELECT * FROM b",
    "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 3) SELECT * FROM n, sales",
])
def test_check_query_accepts(source, sql):
    check_query(sql, source)


def test_check_query_allows_ctes_and_declared_table_functions():
    fixtures = Path(__file__).parents[3] / "tests" / "evals" / "fixtures"
    orders = f"read_csv('{fixtures / 'orders.csv'}')"
    source = DuckDBSource(uri=":memory:", tables={"orders_csv": orders})

    check_query("WITH paid AS (SELECT * FROM orders_csv WHERE status = 'paid') SELECT COUNT(*) FROM paid", source)
    check_query(f"SELECT COUNT(*) FROM {orders}", source)
    with pytest.raises(ValueError, match="READ_CSV"):
        check_query(f"SELECT * FROM read_csv('{fixtures / 'refunds.json'}')", source)


def test_source_node_has_no_spec_to_build_a_source_from(source):
    ops = editor_with(source)
    with pytest.raises(OperationError, match="no config field 'spec'"):
        ops.add_node(SOURCE_ID, {"spec": {"type": "duckdb", "uri": ":memory:", "initializers": ["COPY (SELECT 1) TO 'x.csv'"]}}, name="evil")
    with pytest.raises(OperationError, match="no config field 'spec'"):
        ops.set_config("data", {"spec": {"type": "duckdb", "uri": ":memory:"}})


def test_source_node_reports_unbound_ref(source):
    ops = editor_with(source)
    ops.add_node(SOURCE_ID, {"ref": "missing"}, name="other")

    assert "No source is bound as 'missing'" in failure(ops, "other", "source")


def test_bindings_belong_to_their_editor():
    first = DuckDBSource(uri=":memory:", tables={"a_table": "SELECT 1 AS x"})
    second = DuckDBSource(uri=":memory:", tables={"b_table": "SELECT 2 AS y"})
    a, b = editor_with(first), editor_with(second)

    a.replace_yaml(a.get_dashboard())
    assert output(a, "data", "source") is first
    assert output(b, "data", "source") is second

    b.editor.dispose()
    del b
    a.replace_yaml(a.get_dashboard())
    assert output(a, "data", "source") is first

    unbound = FlowDash(components=COMPONENTS)
    unbound.import_dashboard(dashboard_from_yaml(dashboard_to_yaml(a.editor)))
    assert "No source is bound as 'data'" in failure(DashboardOps(unbound), "data", "source")


def test_yaml_round_trip_reruns_pushed_down_query(source, executed):
    text = dashboard_to_yaml(dashboard(source).editor)
    executed.clear()

    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    editor.import_dashboard(dashboard_from_yaml(text))
    ops = DashboardOps(editor)

    assert without_turns(dashboard_to_yaml(editor)) == without_turns(text)
    data_node = yaml.safe_load(text)["nodes"]["data"]
    assert (data_node["component"], data_node["config"]) == (SOURCE_ID, {"ref": "data"})
    assert output(ops, "chart", "pipeline") is output(ops, "by_category", "pipeline")
    assert len(output(ops, "by_category", "data")) == 2
    assert executed == [" ".join(result_sql(source, QUERY).split())]


@pytest.mark.parametrize("spec, error", [
    ({**SPEC, "data": {"url": "https://example.invalid/x.csv"}}, "remove `data`"),
    ({"mark": "image", "encoding": {"url": {"value": "https://example.invalid/i.png"}}}, "URLs"),
    ({**SPEC, "encoding": {**SPEC["encoding"], "href": {"value": "https://example.invalid/"}}}, "URLs"),
    ({"mark": "bar", "encoding": {"x": {"field": "revenue", "type": "nominal"}}}, "revenue"),
])
def test_view_node_checks_spec_on_every_path(source, spec, error):
    ops = dashboard(source)
    ops.set_config("chart", {"spec": spec})
    assert error in view_error(ops, "chart")

    ops.add_node(VIEW_ID, {"spec": spec}, name="other_chart")
    ops.connect("by_category.pipeline", "other_chart.pipeline")
    assert error in view_error(ops, "other_chart")


@pytest.mark.parametrize("spec", [
    {"layer": [SPEC, {"mark": "rule", "data": {"values": [{"total": 1}]}, "encoding": {"y": {"field": "total", "type": "quantitative"}}}]},
    {**SPEC, "data": {"sequence": {"start": 0, "stop": 10}}},
    {**SPEC, "datasets": {"x": [{"total": 1}]}},
    {**SPEC, "transform": [{"lookup": "category", "from": {"data": {"name": "other"}, "key": "category", "fields": ["total"]}}]},
])
def test_check_chart_refuses_data_at_any_depth(spec):
    with pytest.raises(ValueError, match="remove `data` and `datasets`"):
        check_chart(spec, ["category", "total"])


def test_source_node_picks_up_a_binding_made_after_import(source):
    text = dashboard_to_yaml(dashboard(source).editor)
    editor = FlowDash(components=COMPONENTS)
    editor.import_dashboard(dashboard_from_yaml(text))
    ops = DashboardOps(editor)
    assert "No source is bound" in failure(ops, "data", "source")

    bind_source(editor, source)
    assert output(ops, "data", "source") is source
    assert len(output(ops, "by_category", "data")) == 2


def test_view_node_drops_named_data(source):
    ops = dashboard(source)
    ops.set_config("chart", {"spec": {**SPEC, "data": {"name": "by_category"}}})
    view = ops.editor._component_instances[ops._instance("chart")]._object
    assert "data" not in view.spec


def test_pipeline_does_not_connect_to_source_input(source):
    ops = dashboard(source)
    ops.add_node(SQL_ID, {}, name="other")

    with pytest.raises(OperationError, match=r"(?i)type"):
        ops.connect("by_category.pipeline", "other.source")
