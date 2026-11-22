"""A new view node starts from a spec its pipeline's columns suggest, or shows an empty state."""
import panel as pn
import pytest

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps

from lumen.flowdash import (
    COMPONENTS, SOURCE_ID, TABLE_ID, VIEW_ID, ViewNode, bind_source,
)
from lumen.flowdash.checks import check_chart
from lumen.flowdash.defaults import (
    TOP_CATEGORIES, schema_kinds, vega_lite_default,
)
from lumen.views.base import VegaLiteView

DATE = {"type": "string", "format": "datetime"}
TEXT = {"type": "string", "enum": ["a", "b"]}
NUMBER = {"type": "number", "inclusiveMinimum": 0, "inclusiveMaximum": 10}
INTEGER = {"type": "integer"}

TABLES = {
    "sales": "SELECT * FROM (VALUES ('A', 10), ('B', 20), ('A', 5)) AS t(category, amount)",
    "names": "SELECT * FROM (VALUES ('A', 'x'), ('B', 'y')) AS t(code, label)",
}


@pytest.fixture
def source():
    from lumen.sources.duckdb import DuckDBSource

    return DuckDBSource(uri=":memory:", tables=TABLES)


def editor_with(source, table="sales", **editor_params):
    editor = FlowDash(components=COMPONENTS, **editor_params)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": table}, name="rows")
    ops.connect("data.source", "rows.source")
    return ops


def instance(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def config(ops, name):
    return next(item["config"] for item in ops.editor._tile_items if item["instance_id"] == ops._instance(name))


def test_schema_kinds_sort_columns_and_skip_ids():
    schema = {
        "when": DATE, "day": {"type": "string", "format": "date"}, "region": TEXT,
        "ok": {"type": "boolean"}, "amount": NUMBER, "id": INTEGER, "order_id": INTEGER, "__len__": 3,
    }
    assert schema_kinds(schema) == {
        "temporal": ["when", "day"], "numeric": ["amount"], "categorical": ["region", "ok"],
    }


def test_a_date_and_a_number_make_a_line():
    spec = vega_lite_default({"region": TEXT, "when": DATE, "amount": NUMBER})
    assert spec["mark"] == "line"
    assert spec["encoding"]["x"] == {"field": "when", "type": "temporal"}
    assert spec["encoding"]["y"] == {"field": "amount", "aggregate": "sum", "type": "quantitative"}


def test_a_category_and_a_number_make_bars_of_the_largest_sums():
    spec = vega_lite_default({"name": {"type": "string"}, "region": TEXT, "amount": NUMBER})
    assert spec["mark"] == "bar"
    aggregate, window, top = spec["transform"]
    assert aggregate == {"aggregate": [{"op": "sum", "field": "amount", "as": "amount"}], "groupby": ["region"]}
    assert window["sort"] == [{"field": "amount", "order": "descending"}]
    assert top == {"filter": f"datum._rank <= {TOP_CATEGORIES}"}
    assert spec["encoding"]["y"] == {"field": "region", "type": "nominal", "sort": "-x"}
    assert spec["encoding"]["x"]["field"] == "amount"


def test_a_category_without_listed_values_is_used_when_no_other_is():
    spec = vega_lite_default({"name": {"type": "string"}, "amount": NUMBER})
    assert spec["transform"][0]["groupby"] == ["name"]


def test_two_numbers_make_points_and_one_a_histogram():
    points = vega_lite_default({"x": NUMBER, "y": INTEGER})
    assert points["mark"] == "point"
    assert [points["encoding"][axis]["field"] for axis in ("x", "y")] == ["x", "y"]

    histogram = vega_lite_default({"amount": NUMBER, "id": INTEGER})
    assert histogram["mark"] == "bar"
    assert histogram["encoding"]["x"] == {"field": "amount", "bin": True, "type": "quantitative"}
    assert histogram["encoding"]["y"] == {"aggregate": "count", "type": "quantitative"}


@pytest.mark.parametrize("schema", [{}, None, {"region": TEXT, "when": DATE}, {"id": INTEGER, "__len__": 2}])
def test_no_number_makes_no_default(schema):
    assert vega_lite_default(schema) is None


@pytest.mark.parametrize("schema", [
    {"when": DATE, "amount": NUMBER}, {"region": TEXT, "amount": NUMBER},
    {"x": NUMBER, "y": NUMBER}, {"amount": NUMBER},
])
def test_every_default_passes_the_chart_check(schema):
    spec = vega_lite_default(schema)
    assert check_chart(spec, list(schema)) == spec
    assert "width" not in spec and "height" not in spec and "config" not in spec


def test_a_new_view_starts_from_the_default_when_connected(source):
    ops = editor_with(source)
    ops.add_node(VIEW_ID, {}, name="chart")
    node = instance(ops, "chart")
    assert node.spec == {}
    assert "Waiting for a pipeline" in node._body().object

    ops.connect("rows.pipeline", "chart.pipeline")
    assert node.spec == vega_lite_default(node.pipeline.schema)
    assert node.spec["encoding"]["y"]["field"] == "category"
    assert isinstance(node._object, VegaLiteView) and node._error is None
    assert config(ops, "chart")["spec"] == node.spec
    saved = ops.editor.export_dashboard()
    item = next(item for item in saved["items"] if item["instance_id"] == ops._instance("chart"))
    assert item["config"]["spec"] == node.spec


def test_undoing_the_connection_takes_its_default_with_it(source):
    ops = editor_with(source)
    ops.add_node(VIEW_ID, {}, name="chart")
    history = ops.editor._history
    history.flush()
    ops.connect("rows.pipeline", "chart.pipeline")
    history.flush()
    assert instance(ops, "chart").spec
    assert history.versions[-1].metadata["ops"][0] == {"op": "set_config", "args": {"name": "chart", "keys": ["spec"]}}

    assert ops.editor.undo(whole_runs=False)
    node = instance(ops, "chart")
    assert node.spec == {} and config(ops, "chart").get("spec", {}) == {}
    assert "Waiting for a pipeline" in node._body().object

    assert ops.editor.redo()
    node = instance(ops, "chart")
    assert node.spec == vega_lite_default(node.pipeline.schema)
    assert isinstance(node._object, VegaLiteView)


def test_a_view_given_a_spec_keeps_it(source):
    ops = editor_with(source)
    spec = {"mark": "point", "encoding": {"x": {"field": "amount", "type": "quantitative"}}}
    ops.add_node(VIEW_ID, {"spec": spec}, name="chart")
    ops.connect("rows.pipeline", "chart.pipeline")
    assert instance(ops, "chart").spec == spec


def test_a_view_cleared_later_takes_no_default(source):
    ops = editor_with(source)
    ops.add_node(VIEW_ID, {}, name="chart")
    ops.connect("rows.pipeline", "chart.pipeline")
    ops.set_config("chart", {"spec": {}})
    ops.disconnect("rows.pipeline", "chart.pipeline")
    ops.connect("rows.pipeline", "chart.pipeline")
    assert instance(ops, "chart").spec == {}


def test_a_loaded_view_with_an_empty_spec_stays_empty(source):
    ops = editor_with(source)
    ops.add_node(VIEW_ID, {}, name="chart")
    ops.connect("rows.pipeline", "chart.pipeline")
    ops.set_config("chart", {"spec": {}})
    saved = ops.editor.export_dashboard()

    loaded = FlowDash(components=COMPONENTS)
    bind_source(loaded, source)
    loaded.import_dashboard(saved)
    loaded_ops = DashboardOps(loaded)
    node = instance(loaded_ops, "chart")
    assert node.pipeline is not None
    assert node.spec == {}
    assert "No chart yet" in node._body().objects[0].object

    loaded_ops.disconnect("rows.pipeline", "chart.pipeline")
    loaded_ops.connect("rows.pipeline", "chart.pipeline")
    assert node.spec == {}


def test_a_view_without_a_default_offers_to_edit_its_spec(source, monkeypatch):
    ops = editor_with(source, table="names")
    ops.add_node(VIEW_ID, {}, name="chart")
    ops.connect("rows.pipeline", "chart.pipeline")
    node = instance(ops, "chart")
    assert node.spec == {} and node._object is None and node._error is None
    assert node.__flowdash_preview__() == "No chart yet"

    opened = []
    monkeypatch.setattr(ops.editor, "edit_spec", lambda instance_id, field=None: opened.append((instance_id, field)) or True)
    message, button = node._body().objects
    assert "No chart yet" in message.object
    assert button.label == "Edit spec"
    button.clicks += 1
    assert opened == [(ops._instance("chart"), "spec")]


def test_the_edit_spec_button_opens_the_code_panel(source):
    ops = editor_with(source, table="names")
    ops.add_node(VIEW_ID, {}, name="chart")
    ops.connect("rows.pipeline", "chart.pipeline")
    node = instance(ops, "chart")
    assert node.edit_spec()
    panel = ops.editor.code_panel
    assert panel.instance_id == ops._instance("chart")


def test_viewers_get_no_edit_spec_button(source):
    ops = editor_with(source, table="names")
    ops.add_node(VIEW_ID, {}, name="chart")
    ops.connect("rows.pipeline", "chart.pipeline")
    node = instance(ops, "chart")
    _, button = node._body().objects
    ops.editor.read_only = True
    assert not button.visible


def test_a_view_outside_an_editor_takes_no_default(source):
    from lumen.pipeline import Pipeline

    node = ViewNode(pipeline=Pipeline(source=source, table="sales"))
    assert node.spec == {}
    assert node._body().object == "No chart yet"


def test_the_drawn_tile_follows_a_connection_without_a_chart(source):
    ops = editor_with(source, table="names")
    ops.add_node(VIEW_ID, {}, name="chart")
    node = instance(ops, "chart")
    tile = pn.panel(node)
    assert "Waiting for a pipeline" in tile._inner_layout[0].object

    ops.connect("rows.pipeline", "chart.pipeline")
    message, button = tile._inner_layout[0].objects
    assert message.object == "No chart yet" and button.label == "Edit spec"

    ops.disconnect("rows.pipeline", "chart.pipeline")
    assert "Waiting for a pipeline" in tile._inner_layout[0].object
