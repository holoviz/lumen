import asyncio
import json

import panel as pn
import param
import pytest

pytest.importorskip("duckdb")
spec_settings = pytest.importorskip("panel_flowdash.spec_settings")

from panel_flowdash import DashboardSettings, FlowDash
from panel_flowdash.agent import DashboardOps
from panel_flowdash.ui.spec_field import SpecEditor
from panel_flowdash.ui.status_bar import select_on_canvas

from lumen.flowdash import (
    COMPONENTS, SOURCE_ID, SQL_ID, VIEW_ID, ViewNode, bind_source,
    check_view_config,
)
from lumen.flowdash.view_settings import VEGA_LITE_SETTINGS, themed_spec

QUERY = "SELECT category, SUM(amount) AS total FROM sales GROUP BY category"

SPEC = {
    "mark": "bar",
    "encoding": {
        "x": {"field": "category", "type": "nominal", "axis": {"labelAngle": -45}},
        "y": {"field": "total", "type": "quantitative"},
    },
    "transform": [{"filter": "datum.total > 0"}],
}

SETTINGS = {setting.key: setting for setting in VEGA_LITE_SETTINGS}


class SchemeSettings(DashboardSettings):

    color_scheme = param.Selector(default="teals", objects=["teals", "blues"])


def dashboard(source, **editor_params):
    editor = FlowDash(components=COMPONENTS, notifications=False, **editor_params)
    bind_source(editor, source, "data")
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(SQL_ID, {"sql": QUERY, "table": "by_category"}, name="by_category")
    ops.connect("data.source", "by_category.source")
    ops.add_node(VIEW_ID, {"spec": SPEC}, name="chart")
    ops.connect("by_category.pipeline", "chart.pipeline")
    return ops


def rendered(ops, name="chart"):
    return ops.editor._component_instances[ops._instance(name)]._object.spec


def write(spec, key, value):
    return SETTINGS[key].write(spec, value)


def read(spec, key):
    return SETTINGS[key].read(spec)


def test_the_spec_param_declares_the_settings():
    assert spec_settings.declared_settings(ViewNode.param.spec) == VEGA_LITE_SETTINGS


@pytest.mark.parametrize("key, value", [
    ("mark.type", "line"),
    ("encoding.x.field", "total"),
    ("encoding.y.type", "ordinal"),
    ("encoding.y.aggregate", "sum"),
    ("encoding.color.field", "category"),
    ("config.range.category", "set2"),
    ("config.legend.orient", "bottom"),
    ("config.legend.orient", "off"),
    ("mark.tooltip", True),
    ("title", "Sales by category"),
    ("encoding.x.title", "Category"),
    ("encoding.y.title", "Total"),
])
def test_each_setting_round_trips_and_keeps_the_rest(key, value):
    before = read(SPEC, key)
    spec = write(SPEC, key, value)
    assert read(spec, key) == value
    assert spec["transform"] == SPEC["transform"]
    assert spec["encoding"]["x"]["axis"] == {"labelAngle": -45}
    assert write(spec, key, before) == SPEC


def test_unknown_keys_are_reported_and_kept():
    unknown = [".".join(path) for path in spec_settings.unknown_paths(SPEC, VEGA_LITE_SETTINGS)]
    assert unknown == ["encoding.x.axis", "transform"]
    spec = write(SPEC, "title", "Sales")
    assert [".".join(p) for p in spec_settings.unknown_paths(spec, VEGA_LITE_SETTINGS)] == unknown


def test_the_mark_keeps_its_shorthand_until_an_option_needs_an_object():
    spec = write(SPEC, "mark.type", "line")
    assert spec["mark"] == "line"
    spec = write(spec, "mark.point", True)
    assert spec["mark"] == {"type": "line", "point": True}
    assert write(spec, "mark.type", "area")["mark"] == {"type": "area", "point": True}
    assert not SETTINGS["mark.point"].is_enabled(SPEC)
    assert SETTINGS["mark.point"].is_enabled(spec)


def test_size_fills_the_tile_or_fixes_width_and_height():
    assert read(SPEC, "size") == "fill"
    assert not SETTINGS["width"].is_enabled(SPEC)
    fixed = write(SPEC, "size", "fixed")
    assert (fixed["width"], fixed["height"]) == (400, 300)
    assert SETTINGS["width"].is_enabled(fixed)
    fixed = write(fixed, "width", 640)
    assert read(fixed, "width") == 640
    assert write(fixed, "size", "fill") == SPEC
    assert read({**SPEC, "width": "container"}, "width") is None


def test_a_tooltip_encoding_is_left_to_the_spec():
    spec = {**SPEC, "encoding": {**SPEC["encoding"], "tooltip": [{"field": "total"}]}}
    assert spec_settings.shown_in_spec(spec, VEGA_LITE_SETTINGS) == [SETTINGS["mark.tooltip"]]
    flagged = [".".join(p) for p in spec_settings.unrepresented_paths(spec, VEGA_LITE_SETTINGS)]
    assert "encoding.tooltip" in flagged
    on = write(SPEC, "mark.tooltip", True)
    assert on["mark"] == {"type": "bar", "tooltip": True}
    assert write(on, "mark.tooltip", False) == SPEC


def test_a_colour_list_is_flagged_and_kept():
    spec = {**SPEC, "config": {"range": {"category": ["#f00", "#0f0"]}}}
    assert SETTINGS["config.range.category"] in spec_settings.shown_in_spec(spec, VEGA_LITE_SETTINGS)
    flagged = [".".join(p) for p in spec_settings.unrepresented_paths(spec, VEGA_LITE_SETTINGS)]
    assert "config.range.category" in flagged


@pytest.mark.parametrize("config", [None, {"range": None, "legend": None}, {"legend": {"orient": None}}])
def test_null_config_reads_as_unset(config):
    spec = {**SPEC, "config": config}
    assert read(spec, "config.range.category") is None
    assert read(spec, "config.legend.orient") in ("auto", None)
    assert read(write(spec, "config.legend.orient", "off"), "config.legend.orient") == "off"
    assert read(write(spec, "config.range.category", "set2"), "config.range.category") == "set2"


@pytest.mark.parametrize("spec, disabled", [
    ({"facet": {"field": "category"}, "spec": SPEC}, {"size", "width", "mark.type", "encoding.x.field", "encoding.x.title"}),
    ({"hconcat": [SPEC, SPEC]}, {"size", "mark.type", "encoding.y.field"}),
    ({"layer": [SPEC]}, {"mark.type", "encoding.y.field", "encoding.color.field", "mark.tooltip"}),
])
def test_compound_and_layered_specs_leave_their_parts_to_the_spec(spec, disabled):
    reasons = {key: s.disabled_reason(spec) for key, s in SETTINGS.items()}
    assert all(reasons[key] for key in disabled)
    assert reasons["title"] is None


def test_colours_follow_the_dashboard_unless_a_scheme_is_set():
    assert read(SPEC, "config.range.category") is None
    spec = write(SPEC, "config.range.category", "set2")
    assert spec["config"]["range"] == {"category": {"scheme": "set2"}, "ramp": {"scheme": "set2"}}
    assert write(spec, "config.range.category", None) == SPEC


def test_a_title_object_keeps_its_style():
    spec = {**SPEC, "title": {"text": "Sales", "fontSize": 20}}
    assert read(spec, "title") == "Sales"
    assert write(spec, "title", "Totals")["title"] == {"text": "Totals", "fontSize": 20}


def test_themed_spec_is_transparent_and_takes_the_scheme_without_storing_it():
    themed = themed_spec(SPEC, "teals")
    assert themed["config"]["background"] == "transparent"
    assert themed["config"]["range"]["category"] == {"scheme": "teals"}
    assert "config" not in SPEC
    own = write(SPEC, "config.range.category", "set2")
    assert themed_spec(own, "teals")["config"]["range"]["category"] == {"scheme": "set2"}
    assert "range" not in themed_spec(SPEC)["config"]


def test_check_refuses_a_value_outside_a_restricted_setting(source):
    spec = write(SPEC, "encoding.y.type", "ordinal")
    spec["encoding"]["y"]["type"] = "time"
    problem = check_view_config({"spec": spec}, {})
    assert 'encoding.y.type is "time"' in problem


def test_the_view_follows_the_dashboard_colour_scheme(source):
    ops = dashboard(source, settings_class=SchemeSettings)
    assert rendered(ops)["config"]["range"]["category"] == {"scheme": "teals"}
    ops.editor.settings.color_scheme = "blues"
    assert rendered(ops)["config"]["range"]["category"] == {"scheme": "blues"}
    assert "config" not in ops.node_config("chart")["spec"]


def test_without_a_dashboard_scheme_the_chart_keeps_vega_colours(source):
    ops = dashboard(source)
    assert "range" not in rendered(ops)["config"]
    assert rendered(ops)["config"]["background"] == "transparent"


async def test_the_inspector_edits_the_spec_through_the_form(source):
    ops = dashboard(source, history_coalesce=0)
    editor = ops.editor
    node = ops._instance("chart")
    select_on_canvas(editor._flow, node_id=node)
    fields = list(pn.Column(*editor._inspector._body).select(SpecEditor))
    assert len(fields) == 1
    field = fields[0]
    assert field.unknown == ["encoding.x.axis", "transform"]
    await editor._inspector.settle()
    for _ in range(50):
        if field.choices.get("encoding.x.field"):
            break
        await asyncio.sleep(0.02)
    assert field.choices["encoding.x.field"].values() == ["category", "total"]
    before = json.dumps(ops.node_config("chart")["spec"])
    field._controls["title"][1].value = "Sales by category"
    await editor._inspector.settle()
    assert ops.node_config("chart")["spec"]["title"] == "Sales by category"
    assert rendered(ops)["title"] == "Sales by category"
    assert editor.undo()
    assert json.dumps(ops.node_config("chart")["spec"]) == before
