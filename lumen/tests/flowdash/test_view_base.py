"""LumenViewNode: what a view node of a Lumen pipeline gets from the shared base."""
import pandas as pd
import param
import pytest

pytest.importorskip("duckdb")

from panel_flowdash import CheckWarning, DashboardSettings, FlowDash
from panel_flowdash.agent import DashboardOps, OperationError
from panel_flowdash.ui.node_preview import hook_preview

from lumen.data.contracts import FULL
from lumen.flowdash import (
    COMPONENTS, PIPELINE_TYPE, SOURCE_ID, TABLE_ID, VIEW_ID, CheckDiagnostic,
    LumenViewNode, ViewNode, bind_source, check_view_config,
    nodes as flowdash_nodes, register_view, spec_param,
)
from lumen.flowdash.contracts import usages
from lumen.views.base import DeckGLView, Table

MAP_ID = "Test/map"
TABLE_VIEW_ID = "Test/table_view"


def _check_layers(spec, columns):
    for layer in spec.get("layers", []):
        if "data" in layer:
            raise ValueError("A layer brings its own data.")
    return spec


@register_view(title="Test map", icon="map", description="A deck.gl map of a Lumen pipeline.")
class MapNode(LumenViewNode):

    spec = spec_param("The deck.gl spec, without data.")

    view_type = "deckgl"

    check_spec = staticmethod(_check_layers)

    @staticmethod
    def spec_warnings(spec):
        return [] if spec.get("initialViewState") else ["No initial view state."]

    @staticmethod
    def spec_fields(spec):
        return {layer["column"] for layer in spec.get("layers", []) if "column" in layer}


@register_view(
    title="Test table", icon="table", description="A table of a Lumen pipeline.",
    config=["page_size"],
)
class TableViewNode(LumenViewNode):

    spec = spec_param("Tabulator options.")

    page_size = param.Integer(default=20)

    view_type = "table"

    requires_spec = False

    @staticmethod
    def spec_warnings(spec):
        return [] if "show_index" in spec else ["Show an index?"]

    def _view_params(self, spec):
        return {**spec, "page_size": self.page_size, "sizing_mode": "stretch_both"}


COMPONENTS_WITH_TESTS = {**COMPONENTS, MAP_ID: MapNode, TABLE_VIEW_ID: TableViewNode}

MAP = {"initialViewState": {"zoom": 3}, "layers": [{"@@type": "ScatterplotLayer", "column": "amount"}]}


@pytest.fixture
def components(monkeypatch):
    monkeypatch.setattr(flowdash_nodes, "COMPONENTS", COMPONENTS_WITH_TESTS)
    return COMPONENTS_WITH_TESTS


class SchemeSettings(DashboardSettings):

    color_scheme = param.Selector(default="teals", objects=["teals", "blues"])


def editor_with(source, components, **editor_params):
    editor = FlowDash(components=components, **editor_params)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")
    return ops


def instance(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def test_register_view_declares_the_pipeline_input_and_spec_config():
    meta = MapNode.__panel_app_metadata__
    assert meta.component and not meta.page
    assert meta.requires == [{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}]
    assert meta.provides == []
    assert meta.config == ["spec"]
    assert TableViewNode.__panel_app_metadata__.config == ["spec", "page_size"]
    assert MapNode.param.spec.metadata == {"flowdash": {"widget": "spec", "settings": [], "two_way": True}}


def test_vega_lite_view_keeps_its_registration():
    meta = ViewNode.__panel_app_metadata__
    assert meta.config == ["spec", "debounce"]
    assert meta.check_config({"spec": {"mark": "bar"}}, {}) == check_view_config({"spec": {"mark": "bar"}}, {})
    assert [port["key"] for port in meta.requires] == ["pipeline"]
    assert [port["key"] for port in meta.provides] == ["selection"]
    assert ViewNode.view_type == "vegalite"
    assert ViewNode.param.spec.metadata["flowdash"]["settings"]


def test_a_view_node_draws_its_view_filling_the_tile(source, components):
    ops = editor_with(source, components)
    ops.add_node(MAP_ID, {"spec": MAP}, name="map")
    ops.connect("sales.pipeline", "map.pipeline")

    node = instance(ops, "map")
    assert node._error is None
    assert isinstance(node._object, DeckGLView)
    assert node._object.spec == MAP
    assert len(node._object.pipeline.data) == 3
    assert node._object.panel.sizing_mode == "stretch_both"


def test_a_view_node_without_a_spec_or_pipeline_waits(source, components):
    ops = editor_with(source, components)
    ops.add_node(MAP_ID, {"spec": MAP}, name="map")
    node = instance(ops, "map")
    assert node._object is None and node._error is None
    assert node.__flowdash_preview__() is None
    assert "Waiting for a pipeline" in node._body().object

    ops.connect("sales.pipeline", "map.pipeline")
    ops.set_config("map", {"spec": {}})
    assert node._object is None and node._error is None
    assert "No chart yet" in node._body().objects[0].object
    assert node.__flowdash_preview__() == "No chart yet"


def test_a_view_node_that_needs_no_spec_draws_with_an_empty_one(source, components):
    ops = editor_with(source, components)
    ops.add_node(TABLE_VIEW_ID, {}, name="rows")
    ops.connect("sales.pipeline", "rows.pipeline")

    node = instance(ops, "rows")
    assert node.spec == {} and node._error is None
    assert isinstance(node._object, Table)
    assert len(node._object.panel.value) == 3
    preview = node.__flowdash_preview__()
    assert isinstance(preview, pd.DataFrame) and len(preview) == 3
    assert not [issue for issue in ops.validate() if issue.severity == "error"]

    check = TableViewNode.__panel_app_metadata__.check_config
    pipeline = instance(ops, "sales")._object
    assert check({"spec": {}}, {"pipeline": pipeline}) == ["Show an index?"]


def test_only_themed_views_draw_again_when_the_colour_scheme_changes(source, components):
    ops = editor_with(source, components, settings_class=SchemeSettings)
    ops.add_node(MAP_ID, {"spec": MAP}, name="map")
    ops.connect("sales.pipeline", "map.pipeline")
    ops.add_node(VIEW_ID, {"spec": {"mark": "bar", "encoding": {"x": {"field": "category", "type": "nominal"}}}}, name="chart")
    ops.connect("sales.pipeline", "chart.pipeline")
    drawn = {name: instance(ops, name)._object for name in ("map", "chart")}
    ops.editor.settings.color_scheme = "blues"
    assert instance(ops, "map")._object is drawn["map"]
    assert instance(ops, "chart")._object is not drawn["chart"]


def test_a_view_node_checks_its_spec_against_the_pipeline(source, components):
    ops = editor_with(source, components)
    ops.add_node(MAP_ID, {"spec": MAP}, name="map")
    ops.connect("sales.pipeline", "map.pipeline")

    with pytest.raises(OperationError, match="brings its own data"):
        ops.set_config("map", {"spec": {"layers": [{"data": []}]}})
    assert instance(ops, "map")._object.spec == MAP

    check = MapNode.__panel_app_metadata__.check_config
    pipeline = instance(ops, "sales")._object
    found = check({"spec": {"layers": [{"data": []}]}}, {"pipeline": pipeline})
    assert isinstance(found, CheckDiagnostic) and found.field == "spec"
    warnings = check({"spec": {"layers": []}}, {"pipeline": pipeline})
    assert [str(w) for w in warnings] == ["No initial view state."]
    assert all(isinstance(w, CheckWarning) for w in warnings)
    assert check({"spec": MAP}, {}) is None


def test_a_failed_input_is_the_view_nodes_error(components):
    node = MapNode(spec=MAP)
    node.pipeline = param.ReactiveError(ValueError("boom"))
    assert "The input failed" in node._error


def test_a_view_node_previews_and_pops_out_its_rows(source, components):
    ops = editor_with(source, components)
    ops.add_node(MAP_ID, {"spec": MAP}, name="map")
    ops.connect("sales.pipeline", "map.pipeline")

    preview = instance(ops, "map").__flowdash_preview__()
    assert isinstance(preview, pd.DataFrame)
    assert sorted(preview["amount"]) == [5, 10, 20]
    assert hook_preview(instance(ops, "map")) is not None

    ops.editor._open_card_popout(ops._instance("map"))


def test_a_view_node_summary_names_the_component(components):
    node = MapNode(spec=MAP)
    assert node._summary(MAP, None) == "Test map"
    assert node._summary(MAP, [1]) == "Test map, 1 row"


def test_a_view_node_can_turn_its_spec_into_other_view_params(source, components):
    ops = editor_with(source, components)
    ops.add_node(TABLE_VIEW_ID, {"spec": {"show_index": False}, "page_size": 5}, name="rows")
    ops.connect("sales.pipeline", "rows.pipeline")

    view = instance(ops, "rows")._object
    assert isinstance(view, Table)
    assert view.kwargs["show_index"] is False and view.page_size == 5
    assert view.panel.sizing_mode == "stretch_both"
    preview = instance(ops, "rows").__flowdash_preview__()
    assert isinstance(preview, pd.DataFrame) and len(preview) == 3


def test_the_schema_contract_reads_a_view_nodes_spec_fields(components):
    doc = {
        "flowdash": 4,
        "nodes": {
            "data": {"component": "Lumen/source", "config": {"ref": "data"}},
            "sales": {"component": TABLE_ID, "config": {"table": "sales"}},
            "map": {"component": MAP_ID, "config": {"spec": MAP}},
            "chart": {"component": VIEW_ID, "config": {"spec": {"encoding": {"x": {"field": "category"}}}}},
            "other": {"component": "Other/thing", "config": {"spec": MAP}},
        },
        "edges": [
            {"from": "data.source", "to": "sales.source"},
            {"from": "sales.pipeline", "to": "map.pipeline"},
            {"from": "sales.pipeline", "to": "chart.pipeline"},
            {"from": "sales.pipeline", "to": "other.pipeline"},
        ],
    }
    schema = {"sales": {"category": "string", "amount": "integer"}}
    found = {u.node: (u.coverage, {t: sorted(c) for t, c in u.tables.items()}) for u in usages(doc, "data", schema, "duckdb")}
    assert found == {
        "sales": (FULL, {"sales": []}),
        "map": (FULL, {"sales": ["amount"]}),
        "chart": (FULL, {"sales": ["category"]}),
    }
