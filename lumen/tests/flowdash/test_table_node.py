import pytest
import yaml

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import (
    DashboardOps, OperationError, dashboard_to_yaml,
)

from lumen.flowdash import (
    COMPONENTS, DATA_ID, FILTER_ID, SOURCE_ALIASES_KEY, SOURCE_ID, SOURCES_KEY,
    TABLE_ID, VIEW_ID, bind_source,
)
from lumen.flowdash.contracts import ref_nodes, usages
from lumen.flowdash.run import (
    CONTRACT, OK, USAGE, run_dashboard,
)
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "sales": "SELECT * FROM (VALUES ('A', 10), ('B', 20), ('A', 5)) AS t(category, amount)",
    "regions": "SELECT * FROM (VALUES ('A', 'North')) AS t(category, region)",
}

BAR = {"mark": "bar", "encoding": {"x": {"field": "category", "type": "nominal"}, "y": {"field": "amount", "type": "quantitative"}}}

# A dashboard saved before table nodes took a ref: a source node feeding a table node.
SOURCE_THEN_TABLE = {
    "flowdash": 4,
    "nodes": {
        "data": {"component": "Lumen/source", "config": {"ref": "data"}},
        "sales": {"component": "Lumen/table", "config": {"table": "sales", "limit": 2}},
        "only_a": {"component": "Lumen/filter", "config": {"field": "category"}},
        "rows": {"component": "Lumen/data", "config": {}},
    },
    "edges": [
        {"from": "data.source", "to": "sales.source"},
        {"from": "sales.pipeline", "to": "only_a.pipeline"},
        {"from": "only_a.filtered", "to": "rows.pipeline"},
    ],
}


@pytest.fixture
def source():
    return DuckDBSource(uri=":memory:", tables=TABLES)


def editor(sources, **context):
    return DashboardOps(FlowDash(components=COMPONENTS, context={SOURCES_KEY: sources, **context}, notifications=False))


def instance(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def test_a_table_node_reads_a_table_of_the_source_its_ref_names(source):
    ops = editor({"shop": source})
    ops.add_node(TABLE_ID, {"ref": "shop", "table": "sales"}, name="sales")
    ops.add_node(VIEW_ID, {"spec": BAR}, name="chart")
    ops.connect("sales.pipeline", "chart.pipeline")
    pipeline = ops.node_output("sales.pipeline")
    assert isinstance(pipeline, Pipeline) and pipeline.source is source and pipeline.table == "sales"
    assert sorted(pipeline.data["amount"]) == [5, 10, 20]
    assert instance(ops, "chart")._error is None
    saved = yaml.safe_load(dashboard_to_yaml(ops.editor))
    assert saved["nodes"]["sales"]["config"] == {"ref": "shop", "table": "sales"}
    assert saved["edges"] == [{"from": "sales.pipeline", "to": "chart.pipeline"}]


def test_a_table_node_without_a_ref_or_table_waits(source):
    ops = editor({"shop": source})
    ops.add_node(TABLE_ID, {"ref": "shop"}, name="sales")
    node = instance(ops, "sales")
    assert node._object is None and node._error is None
    assert node._waiting == "Pick a source and a table"


def test_a_table_node_refuses_a_ref_the_app_does_not_provide(source):
    ops = editor({"shop": source})
    with pytest.raises(OperationError, match=r"The app provides no source `nope`; its sources are: `shop`\."):
        ops.add_node(TABLE_ID, {"ref": "nope", "table": "sales"}, name="sales")
    ops.add_node(TABLE_ID, {"ref": "shop"}, name="sales")
    with pytest.raises(OperationError, match=r"The source has no table 'nope'\. Its tables are: regions, sales\."):
        ops.set_config("sales", {"table": "nope"})


def test_a_table_node_names_a_source_by_an_alias(source):
    ops = editor({"shop": source}, **{SOURCE_ALIASES_KEY: {"data": "shop"}})
    ops.add_node(TABLE_ID, {"ref": "data", "table": "regions"}, name="regions")
    assert ops.node_output("regions.pipeline").source is source


def test_a_table_node_picks_up_a_source_bound_later(source):
    ops = editor({})
    ops.add_node(TABLE_ID, {}, name="sales")
    ops.editor._apply_config_patch(ops._instance("sales"), {"ref": "late", "table": "sales"})
    assert "The app provides no Lumen sources" in instance(ops, "sales")._error
    bind_source(ops.editor, source, "late")
    assert instance(ops, "sales")._error is None
    assert ops.node_output("sales.pipeline").table == "sales"


def test_a_connected_source_is_read_instead_of_the_ref(source):
    other = DuckDBSource(uri=":memory:", tables={"sales": "SELECT 'Z' AS category, 1 AS amount"})
    ops = editor({"shop": source, "other": other})
    ops.add_node(SOURCE_ID, {"ref": "other"}, name="other")
    ops.add_node(TABLE_ID, {"ref": "shop", "table": "sales"}, name="sales")
    assert ops.node_output("sales.pipeline").source is source
    ops.connect("other.source", "sales.source")
    assert ops.node_output("sales.pipeline").source is other
    ops.disconnect("other.source", "sales.source")
    assert ops.node_output("sales.pipeline").source is source


def test_a_saved_source_then_table_dashboard_loads_unchanged(source):
    ops = editor({"data": source})
    ops.editor.import_dashboard(yaml.safe_dump(SOURCE_THEN_TABLE))
    instance(ops, "only_a").value = "A"
    assert sorted(ops.node_output("rows.data")["amount"]) == [5, 10]
    assert ops.node_output("sales.pipeline").source is source
    saved = yaml.safe_load(dashboard_to_yaml(ops.editor))
    assert {name: node.get("config", {}) for name, node in saved["nodes"].items()} == {
        name: node["config"] for name, node in SOURCE_THEN_TABLE["nodes"].items()
    }
    assert saved["edges"] == SOURCE_THEN_TABLE["edges"]


def test_the_dashboard_refresh_reads_a_ref_source_again(source):
    ops = editor({"shop": source})
    if "refresh" not in ops.editor.param:
        pytest.skip("panel-flowdash without a dashboard refresh")
    ops.add_node(TABLE_ID, {"ref": "shop", "table": "sales"}, name="sales")
    before = ops.node_output("sales.pipeline")
    cleared = []
    source.clear_cache = lambda: cleared.append(1)
    ops.editor.refresh = True
    assert cleared and ops.node_output("sales.pipeline") is not before


def one_table(**config):
    return {
        "flowdash": 4,
        "nodes": {
            "sales": {"component": "Lumen/table", "config": {"ref": "data", "table": "sales", **config}},
            "only_a": {"component": "Lumen/filter", "config": {"field": "category"}},
            "chart": {"component": "Lumen/view", "config": {"spec": BAR}},
        },
        "edges": [
            {"from": "sales.pipeline", "to": "only_a.pipeline"},
            {"from": "only_a.filtered", "to": "chart.pipeline"},
        ],
    }


def test_contracts_cover_a_table_node_that_names_its_source():
    doc = one_table()
    assert ref_nodes(doc) == {"sales": "data"}
    schema = {"sales": {"category": "string", "amount": "integer"}}
    found = {u.node: {t: sorted(c) for t, c in u.tables.items()} for u in usages(doc, "sales", schema)}
    assert found == {"sales": {"sales": []}, "only_a": {"sales": ["category"]}, "chart": {"sales": ["amount", "category"]}}
    connected = {**doc, "nodes": {**doc["nodes"], "data": {"component": "Lumen/source", "config": {"ref": "data"}}},
                 "edges": [*doc["edges"], {"from": "data.source", "to": "sales.source"}]}
    assert ref_nodes(connected) == {"data": "data"}


def test_a_parameter_on_a_table_nodes_ref_records_and_checks_its_contract(source):
    ops = editor({"data": source})
    ops.editor.import_dashboard(yaml.safe_dump(one_table()))
    ops.declare_parameter("data", "sales", "ref", type="source")
    declared = yaml.safe_load(dashboard_to_yaml(ops.editor))["parameters"]["data"]
    assert declared["expects"] == {"tables": {"sales": {"amount": "integer", "category": "string"}}}
    renamed = DuckDBSource(uri=":memory:", tables={"sales": "SELECT 'A' AS kind, 1 AS amount"})
    bind_source(ops.editor, renamed, "renamed")
    with pytest.raises(OperationError, match="only_a: Table 'sales' has no column 'category'"):
        ops.bind_parameters({"data": "renamed"})


def test_run_binds_a_table_nodes_ref(source, tmp_path):
    text = yaml.safe_dump(one_table())
    assert run_dashboard(text).message == "Bind the data of the dashboard's sources: --source data=<data>."
    path = tmp_path / "sales.csv"
    path.write_text("category,amount\nA,1\nB,2\n")
    report = run_dashboard(text, sources={"data": str(path)})
    assert report.exit_code == OK, report
    path.write_text("kind,amount\nA,1\n")
    report = run_dashboard(text, sources={"data": str(path)})
    assert report.exit_code == CONTRACT
    assert {"node": "only_a", "message": "Table 'sales' has no column 'category'."} in report.problems
    assert run_dashboard("flowdash: 99\n").exit_code == USAGE


def test_table_nodes_list_after_filters_and_data(source):
    ops = editor({"shop": source})
    ops.add_node(TABLE_ID, {"ref": "shop", "table": "sales"}, name="sales")
    ops.add_node(FILTER_ID, {"field": "category"}, name="only_a")
    ops.connect("sales.pipeline", "only_a.pipeline")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("only_a.filtered", "rows.pipeline")
    instance(ops, "only_a").value = "B"
    assert ops.node_output("rows.data")["amount"].tolist() == [20]
