import pytest

pytest.importorskip("panel_flowdash")

from panel_flowdash import FlowDash
from panel_flowdash.agent import (
    DashboardOps, dashboard_from_yaml, dashboard_to_yaml,
)

from lumen.flowdash import (
    COMPONENTS, SOURCE_ID, SQL_ID, VIEW_ID, bind_source, source_config,
    unbind_source,
)
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource

TABLES = {"sales": "SELECT * FROM (VALUES ('A', 10), ('B', 20), ('A', 5)) AS t(category, amount)"}
QUERY = "SELECT category, SUM(amount) AS total FROM sales GROUP BY category"
SPEC = {"mark": "bar", "encoding": {"x": {"field": "category", "type": "nominal"}, "y": {"field": "total", "type": "quantitative"}}}


@pytest.fixture
def fetches(monkeypatch):
    queries = []
    fetch = DuckDBSource.fetch

    def logged(self, sql, *args, **kwargs):
        queries.append(" ".join(str(sql).split()))
        return fetch(self, sql, *args, **kwargs)

    monkeypatch.setattr(DuckDBSource, "fetch", logged)
    return queries


@pytest.fixture
def source():
    source = DuckDBSource(uri=":memory:", tables=TABLES)
    ref = bind_source(source, "data")
    yield source
    unbind_source(ref)


def output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


def dashboard(source, sql=QUERY):
    ops = DashboardOps(FlowDash(components=COMPONENTS))
    ops.add_node(SOURCE_ID, source_config(source, "data"), name="data")
    ops.add_node(SQL_ID, {"sql": sql, "table": "by_category"}, name="by_category")
    ops.connect("data.source", "by_category.source")
    ops.add_node(VIEW_ID, {"spec": SPEC}, name="chart")
    ops.connect("by_category.pipeline", "chart.pipeline")
    return ops


def test_sql_node_runs_query_in_source_engine(source, fetches):
    ops = dashboard(source)

    pipeline = output(ops, "by_category", "pipeline")
    data = output(ops, "by_category", "data")

    assert isinstance(pipeline, Pipeline)
    assert sorted(map(tuple, data.values.tolist())) == [("A", 15), ("B", 20)]
    assert fetches == [f"{QUERY} LIMIT 1000000"]


def test_sql_node_table_does_not_shadow_source_table(source):
    ops = DashboardOps(FlowDash(components=COMPONENTS))
    ops.add_node(SOURCE_ID, source_config(source, "data"), name="data")
    ops.add_node(SQL_ID, {"sql": "SELECT * FROM sales WHERE amount > 5", "table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")

    assert len(output(ops, "sales", "data")) == 2


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "SELECT 1; DELETE FROM sales",
    "CREATE TABLE copy AS SELECT * FROM sales",
    "INSERT INTO sales VALUES ('C', 1)",
])
def test_sql_node_refuses_anything_but_one_read_only_query(source, fetches, sql):
    ops = dashboard(source, sql)

    for port in ("pipeline", "data"):
        value = output(ops, "by_category", port)
        assert not isinstance(value, Pipeline)
        assert "read-only" in str(getattr(value, "exception", value))
    assert fetches == []
    assert ops.preview()["nodes"]["by_category"]["outputs"]["data"]["type"] == "failure"


def test_source_node_resolves_spec_without_binding():
    ops = DashboardOps(FlowDash(components=COMPONENTS))
    ops.add_node(SOURCE_ID, {"spec": {"type": "duckdb", "uri": ":memory:", "tables": TABLES}}, name="data")
    ops.add_node(SQL_ID, {"sql": QUERY}, name="by_category")
    ops.connect("data.source", "by_category.source")

    assert len(output(ops, "by_category", "data")) == 2


def test_source_node_reports_unbound_ref():
    ops = DashboardOps(FlowDash(components=COMPONENTS))
    ops.add_node(SOURCE_ID, {"ref": "missing"}, name="data")

    assert "No source is bound as 'missing'" in str(output(ops, "data", "source").exception)


def test_yaml_round_trip_reruns_pushed_down_query(source, fetches):
    text = dashboard_to_yaml(dashboard(source).editor)
    fetches.clear()

    editor = FlowDash(components=COMPONENTS)
    editor.import_dashboard(dashboard_from_yaml(text))
    ops = DashboardOps(editor)

    assert dashboard_to_yaml(editor) == text
    assert output(ops, "chart", "pipeline") is output(ops, "by_category", "pipeline")
    assert len(output(ops, "by_category", "data")) == 2
    assert fetches == [f"{QUERY} LIMIT 1000000"]


def test_pipeline_does_not_connect_to_source_input(source):
    ops = dashboard(source)
    ops.add_node(SQL_ID, {}, name="other")

    with pytest.raises(Exception, match="(?i)type"):
        ops.connect("by_category.pipeline", "other.source")
