"""A new view node starts from a spec its pipeline's columns suggest, or shows an empty state."""
import panel as pn
import pytest

pytest.importorskip("duckdb")
spec_settings = pytest.importorskip("panel_flowdash.spec_settings")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps

from lumen.flowdash import (
    COMPONENTS, SOURCE_ID, TABLE_ID, VIEW_ID, ViewNode, bind_source,
)
from lumen.flowdash.checks import check_chart
from lumen.flowdash.contracts import spec_fields
from lumen.flowdash.defaults import (
    TOP_CATEGORIES, schema_kinds, vega_lite_default,
)
from lumen.flowdash.view_settings import VEGA_LITE_SETTINGS
from lumen.views.base import VegaLiteView

DATE = {"type": "string", "format": "datetime"}
TEXT = {"type": "string", "enum": ["a", "b"]}
NUMBER = {"type": "number", "inclusiveMinimum": 0, "inclusiveMaximum": 10}
INTEGER = {"type": "integer"}
NAN = float("nan")

TABLES = {
    "sales": "SELECT * FROM (VALUES ('A', 10), ('B', 20), ('A', 5)) AS t(category, amount)",
    "names": "SELECT * FROM (VALUES ('A', 'x'), ('B', 'y')) AS t(code, label)",
}


@pytest.fixture
def legacy(monkeypatch):
    """Lumen as it runs on a panel-flowdash without ``placement_context`` and ``set_default``."""
    from lumen.flowdash import nodes

    monkeypatch.setattr(nodes, "placement_context", None)


@pytest.fixture
def public():
    from lumen.flowdash import nodes

    if nodes.placement_context is None:
        pytest.skip("panel-flowdash has no placement_context")


@pytest.fixture
def duckdb_source():
    from lumen.sources.duckdb import DuckDBSource

    return DuckDBSource(uri=":memory:", tables=TABLES)


@pytest.fixture(params=["public", "legacy"])
def source(request, duckdb_source):
    """A source to chart, with each of the ways a view takes its default."""
    request.getfixturevalue(request.param)
    return duckdb_source


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


def test_schema_kinds_sort_columns_and_skip_ids_and_nulls():
    schema = {
        "when": DATE, "day": {"type": "string", "format": "date"}, "region": TEXT,
        "ok": {"type": "boolean"}, "count": INTEGER, "amount": NUMBER, "id": INTEGER, "order_id": INTEGER,
        "empty": {"type": "string", "enum": [None]},
        "blank": {"type": "number", "inclusiveMinimum": NAN, "inclusiveMaximum": NAN},
        "year": {"type": "integer", "inclusiveMinimum": 1990, "inclusiveMaximum": 2024},
        "built": {"type": "integer", "inclusiveMinimum": 1950, "inclusiveMaximum": 2020},
        "__len__": 3,
    }
    assert schema_kinds(schema) == {
        "temporal": ["when", "day"], "year": ["year", "built"],
        "numeric": ["amount", "count"], "categorical": ["region", "ok"],
    }


def test_text_listing_only_dates_is_temporal():
    dates = {"type": "string", "enum": ["2024-01-01", "2024-02-01", None]}
    assert schema_kinds({"month": dates})["temporal"] == ["month"]
    assert schema_kinds({"code": {"type": "string", "enum": ["2024-01-01", "other"]}})["categorical"] == ["code"]
    spec = vega_lite_default({"month": dates, "region": TEXT, "amount": NUMBER})
    assert spec["encoding"]["x"] == {"field": "month", "type": "temporal"}


def test_a_date_and_a_number_make_a_line():
    spec = vega_lite_default({"region": TEXT, "when": DATE, "amount": NUMBER})
    assert spec["mark"] == "line"
    assert spec["encoding"]["x"] == {"field": "when", "type": "temporal"}
    assert spec["encoding"]["y"] == {"field": "amount", "aggregate": "sum", "type": "quantitative"}


def test_a_year_is_the_x_axis_never_the_measure():
    year = {"type": "integer", "inclusiveMinimum": 2001, "inclusiveMaximum": 2020}
    spec = vega_lite_default({"year": year, "region": TEXT, "sales": INTEGER})
    assert spec["mark"] == "line"
    assert spec["encoding"]["x"] == {"field": "year", "type": "ordinal"}
    assert spec["encoding"]["y"]["field"] == "sales"
    assert vega_lite_default({"fiscal_yr": INTEGER, "sales": NUMBER})["encoding"]["x"]["field"] == "fiscal_yr"
    assert vega_lite_default({"year": year}) is None


def test_coordinates_are_not_measured():
    spec = vega_lite_default({"state": TEXT, "xlong": NUMBER, "pickup_lat": NUMBER, "capacity": INTEGER})
    assert spec["encoding"]["y"]["field"] == "capacity"
    assert vega_lite_default({"lon": NUMBER, "lat": NUMBER}) is None


def test_a_fraction_is_measured_before_an_integer():
    spec = vega_lite_default({"region": TEXT, "units": INTEGER, "revenue": NUMBER})
    assert spec["encoding"]["y"]["field"] == "revenue"


def test_columns_of_nulls_are_never_charted():
    spec = vega_lite_default({
        "empty": {"type": "string", "enum": [None]}, "region": TEXT,
        "blank": {"type": "number", "inclusiveMinimum": NAN, "inclusiveMaximum": NAN}, "amount": NUMBER,
    })
    assert spec["encoding"]["x"]["field"] == "region"
    assert spec["encoding"]["y"]["field"] == "amount"
    assert vega_lite_default({"blank": {"type": "number", "inclusiveMinimum": None, "inclusiveMaximum": None}}) is None


def test_a_category_and_a_number_make_bars_largest_first():
    spec = vega_lite_default({"name": {"type": "string"}, "region": TEXT, "amount": NUMBER})
    assert spec == {
        "mark": "bar",
        "encoding": {
            "x": {"field": "region", "type": "nominal", "sort": "-y"},
            "y": {"field": "amount", "aggregate": "sum", "type": "quantitative"},
        },
    }
    assert spec_settings.unknown_paths(spec, VEGA_LITE_SETTINGS) == []


def test_many_categories_keep_the_largest_sums():
    spec = vega_lite_default({"name": {"type": "string"}, "amount": NUMBER})
    aggregate, window, top = spec["transform"]
    assert aggregate == {"aggregate": [{"op": "sum", "field": "amount", "as": "amount"}], "groupby": ["name"]}
    assert window["sort"] == [{"field": "amount", "order": "descending"}]
    assert top == {"filter": f"datum._rank <= {TOP_CATEGORIES}"}
    assert spec["encoding"]["x"]["sort"] == "-y"
    listed = {"type": "string", "enum": [str(i) for i in range(TOP_CATEGORIES + 1)]}
    assert "transform" in vega_lite_default({"name": listed, "amount": NUMBER})


def test_dotted_names_are_escaped_and_still_read_their_columns():
    schema = {"a.b": {"type": "string"}, "v[1]": NUMBER}
    spec = vega_lite_default(schema)
    assert spec["encoding"]["x"]["field"] == "a\\.b"
    assert spec["encoding"]["y"]["field"] == "v\\[1\\]"
    assert check_chart(spec, list(schema)) == spec
    assert spec_fields(spec) == {"a.b", "v[1]"}


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
    {"x": NUMBER, "y": NUMBER}, {"amount": NUMBER}, {"name": {"type": "string"}, "amount": NUMBER},
    {"year": INTEGER, "amount": NUMBER},
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
    assert node.spec["encoding"]["x"]["field"] == "category"
    assert isinstance(node._object, VegaLiteView) and node._error is None
    assert config(ops, "chart")["spec"] == node.spec
    saved = ops.editor.export_dashboard()
    item = next(item for item in saved["items"] if item["instance_id"] == ops._instance("chart"))
    assert item["config"]["spec"] == node.spec


@pytest.mark.parametrize("coalesce", [0, 60])
def test_the_default_is_part_of_the_connection_and_undone_with_it(source, coalesce):
    ops = editor_with(source, history_coalesce=coalesce)
    ops.add_node(VIEW_ID, {}, name="chart")
    history = ops.editor.edit_history
    history.flush()
    before = len(history.versions)
    ops.connect("rows.pipeline", "chart.pipeline")
    history.flush()
    assert instance(ops, "chart").spec
    assert len(history.versions) == before + 1
    assert history.versions[-1].metadata["op"] == "connect"
    assert "transform" in history.versions[-1].value or "mark: bar" in history.versions[-1].value
    assert ops.editor.dirty

    assert ops.editor.undo(whole_runs=False)
    node = instance(ops, "chart")
    assert node.spec == {} and config(ops, "chart").get("spec", {}) == {}
    assert "Waiting for a pipeline" in node._body().object

    assert ops.editor.redo()
    node = instance(ops, "chart")
    assert node.spec == vega_lite_default(node.pipeline.schema)
    assert isinstance(node._object, VegaLiteView)


def test_preview_mode_leaves_the_default_for_later(source):
    ops = editor_with(source)
    ops.add_node(VIEW_ID, {}, name="chart")
    node = instance(ops, "chart")
    ops.editor.preview = True
    node.pipeline = instance(ops, "rows")._object
    assert node.spec == {} and node._may_default
    ops.editor.preview = False
    ops.connect("rows.pipeline", "chart.pipeline")
    assert node.spec


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


def test_the_edit_spec_button_hides_while_the_code_panel_shows_the_view(source):
    ops = editor_with(source, table="names")
    ops.add_node(VIEW_ID, {}, name="chart")
    ops.connect("rows.pipeline", "chart.pipeline")
    node = instance(ops, "chart")
    _, button = node._body().objects
    assert button.visible
    assert node.edit_spec()
    assert not button.visible
    ops.editor.close_code_panel()
    ops.editor.code_panel.instance_id = None
    assert button.visible


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



@pytest.fixture
def ticks(monkeypatch):
    from lumen.flowdash import nodes

    callbacks = []
    monkeypatch.setattr(nodes, "_next_tick", callbacks.append)
    return callbacks


def test_a_pipeline_arriving_without_an_edit_records_the_default_right_away(duckdb_source, public):
    ops = editor_with(duckdb_source, history_coalesce=0)
    ops.add_node(VIEW_ID, {}, name="chart")
    history = ops.editor.edit_history
    history.flush()
    before = len(history.versions)
    node = instance(ops, "chart")
    node.pipeline = instance(ops, "rows")._object
    history.flush()
    assert node.spec and config(ops, "chart")["spec"] == node.spec
    assert len(history.versions) == before + 1
    assert history.versions[-1].metadata["op"] == "set_config"


def test_a_pipeline_arriving_without_an_edit_records_the_default_on_the_next_tick(duckdb_source, legacy, ticks):
    ops = editor_with(duckdb_source, history_coalesce=0)
    ops.add_node(VIEW_ID, {}, name="chart")
    history = ops.editor.edit_history
    history.flush()
    before = len(history.versions)
    node = instance(ops, "chart")
    node.pipeline = instance(ops, "rows")._object
    assert node.spec and len(history.versions) == before
    [record] = ticks
    record()
    history.flush()
    assert len(history.versions) == before + 1
    assert history.versions[-1].metadata["op"] == "set_config"


def test_a_connection_recorded_after_the_default_needs_no_extra_edit(duckdb_source, legacy, ticks):
    ops = editor_with(duckdb_source, history_coalesce=0)
    ops.add_node(VIEW_ID, {}, name="chart")
    history = ops.editor.edit_history
    history.flush()
    ops.connect("rows.pipeline", "chart.pipeline")
    history.flush()
    count = len(history.versions)
    [record] = ticks
    record()
    history.flush()
    assert len(history.versions) == count


def test_year_like_values_are_measured_when_nothing_else_is():
    spread = {"type": "integer", "inclusiveMinimum": 1850, "inclusiveMaximum": 1990}
    spec = vega_lite_default({"region": TEXT, "visitors": spread})
    assert spec["mark"] == "bar" and spec["encoding"]["y"]["field"] == "visitors"
    spec = vega_lite_default({"built": spread, "capacity": NUMBER})
    assert spec["encoding"]["x"] == {"field": "built", "type": "ordinal"}


def test_a_true_false_category_takes_no_top_n():
    spec = vega_lite_default({"active": {"type": "boolean"}, "amount": NUMBER})
    assert spec["encoding"]["x"]["field"] == "active" and "transform" not in spec


class _Without:
    """An editor as a node sees it, without one private attribute, as if FlowDash had renamed it."""

    def __init__(self, editor, missing):
        self._editor, self._missing = editor, missing

    def __getattr__(self, name):
        if name == self._missing:
            raise AttributeError(name)
        return getattr(self._editor, name)

    def __setattr__(self, name, value):
        if name in ("_editor", "_missing"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._editor, name, value)


def test_without_the_mute_flag_no_view_takes_a_default(duckdb_source, legacy, monkeypatch):
    from lumen.flowdash import nodes

    ops = editor_with(duckdb_source)
    monkeypatch.setattr(nodes, "current_dashboard", lambda: _Without(ops.editor, "_muted"))
    assert not ViewNode()._may_default


@pytest.mark.parametrize("missing", ["_history", "_hydrating", "_dataflow_graph", "_node_name", "_record_edit", "code_panel"])
def test_a_flowdash_without_a_private_attribute_still_draws_the_view(duckdb_source, legacy, ticks, missing):
    import weakref

    ops = editor_with(duckdb_source)
    ops.add_node(VIEW_ID, {}, name="chart")
    node = instance(ops, "chart")
    editor = _Without(ops.editor, missing)
    node._editor = weakref.ref(editor)
    node.pipeline = instance(ops, "rows")._object
    assert node.spec and isinstance(node._object, VegaLiteView)
    for record in ticks:
        record()

    empty = editor_with(duckdb_source, table="names")
    empty.add_node(VIEW_ID, {}, name="chart")
    view = instance(empty, "chart")
    stand_in = _Without(empty.editor, missing)
    view._editor = weakref.ref(stand_in)
    empty.connect("rows.pipeline", "chart.pipeline")
    message, button = view._body().objects
    assert message.object == "No chart yet" and button.visible


def test_the_edit_spec_button_follows_the_code_panel_without_creating_it(duckdb_source, public):
    ops = editor_with(duckdb_source, table="names")
    ops.add_node(VIEW_ID, {}, name="chart")
    ops.connect("rows.pipeline", "chart.pipeline")
    node = instance(ops, "chart")
    _, button = node._body().objects
    assert button.visible and ops.editor._code_panel is None
    assert node.edit_spec()
    assert ops.editor.code_panel_node == ops._instance("chart") and not button.visible
