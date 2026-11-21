"""The Lumen deck.gl map node: its spec check, drawing, theme, card and contract."""
import pandas as pd
import panel as pn
import param
import pytest

pytest.importorskip("duckdb")

from panel_flowdash import DashboardSettings, FlowDash
from panel_flowdash.agent import DashboardOps, OperationError

from lumen.data.contracts import FULL
from lumen.flowdash import (
    COMPONENTS, DECKGL_ID, SOURCE_ID, TABLE_ID, CheckDiagnostic, DeckGLNode,
    bind_source, check_deckgl,
)
from lumen.flowdash.contracts import usages
from lumen.flowdash.deckgl import (
    BASEMAPS, SCHEME_COLORS, deckgl_fields, expression_names, fit_view_state,
)
from lumen.views.base import DeckGLView

COLUMNS = ["state", "xlong", "ylat", "capacity"]

TURBINES = (
    "SELECT * FROM (VALUES ('IA', -93.5, 42.0, 2.0), ('TX', -100.0, 32.0, 3.0), "
    "('CA', -118.0, 35.0, 1.5)) AS t(state, xlong, ylat, capacity)"
)

POINTS = {
    "layers": [{
        "@@type": "ScatterplotLayer", "id": "turbines",
        "getPosition": "@@=[xlong, ylat]", "getRadius": "@@=capacity * 1000",
    }],
}


@pytest.fixture
def turbines():
    duckdb_source = pytest.importorskip("lumen.sources.duckdb")
    return duckdb_source.DuckDBSource(uri=":memory:", tables={"turbines": TURBINES})


class SchemeSettings(DashboardSettings):

    color_scheme = param.Selector(default="teals", objects=["teals", "oranges"])


def map_of(source, spec, **editor_params):
    editor = FlowDash(components=COMPONENTS, **editor_params)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "turbines"}, name="turbines")
    ops.connect("data.source", "turbines.source")
    ops.add_node(DECKGL_ID, {"spec": spec}, name="map")
    ops.connect("turbines.pipeline", "map.pipeline")
    return ops


def node(ops, name="map"):
    return ops.editor._component_instances[ops._instance(name)]


def test_the_node_is_a_registered_view_with_a_json_spec():
    meta = DeckGLNode.__panel_app_metadata__
    assert COMPONENTS[DECKGL_ID] is DeckGLNode
    assert [port["key"] for port in meta.requires] == ["pipeline"]
    assert meta.config == ["spec"]
    assert meta.extensions == ["deckgl"]
    assert meta.default_size == {"height": 420} and meta.min_size == {"height": 240}
    assert DeckGLNode.param.spec.metadata == {"flowdash": {"widget": "spec", "settings": []}}
    assert DeckGLNode.requires_spec and DeckGLNode.themed


def test_the_agent_reads_a_description_and_an_example_spec():
    meta = DeckGLNode.__panel_app_metadata__
    assert "deck.gl" in meta.description and "@@=" in meta.description
    doc = DeckGLNode.param.spec.doc
    assert '"getPosition": "@@=[longitude, latitude]"' in doc
    assert "mapStyle" in doc and "initialViewState" in doc


def test_check_accepts_a_spec_reading_the_datas_columns():
    assert check_deckgl(POINTS, COLUMNS) == POINTS
    hexagons = {"layers": [{"@@type": "HexagonLayer", "getPosition": "@@=[xlong, ylat]", "extruded": True}]}
    assert check_deckgl(hexagons, COLUMNS) == hexagons


@pytest.mark.parametrize(("spec", "message"), [
    ([], "must be a deck.gl JSON object"),
    ({"layers": [{"@@type": "ScatterplotLayer", "radiusScale": float("nan")}]}, "plain JSON"),
    ({"layers": []}, "at least one layer"),
    ({"layers": "ScatterplotLayer"}, "at least one layer"),
    ({"layers": ["ScatterplotLayer"]}, "Layer 1 must be an object"),
    ({"layers": [{"type": "ScatterplotLayer"}]}, "`type` is pydeck's name"),
    ({"layers": [{"@@type": "ScaterplotLayer"}]}, "Did you mean 'ScatterplotLayer'"),
    ({"layers": [{"@@type": "TileLayer"}]}, "unknown type 'TileLayer'"),
    ({"layers": [{"@@type": "ScatterplotLayer", "data": "https://example.com/points.json"}]}, "remove its `data`"),
    ({"layers": [{"@@type": "ScatterplotLayer", "getPosition": ["xlong", "ylat"]}]}, '"@@=[xlong, ylat]"'),
    ({"layers": [{"@@type": "ScatterplotLayer", "id": "pts", "getPosition": "@@=[lon, ylat]"}]}, "Layer 'pts' reads lon"),
    ({"layers": [{"@@type": "GeoJsonLayer", "getFillColor": "@@=[capacity, 0, 0]"}]}, "properties.capacity"),
    ({"layers": [{"@@type": "ScatterplotLayer"}], "map_style": "light"}, "`mapStyle`"),
    ({"layers": [{"@@type": "ScatterplotLayer"}], "mapStyle": "mapbox://styles/mapbox/dark-v9"}, "API key"),
    ({"layers": [{"@@type": "ScatterplotLayer"}], "initialViewState": [0, 0]}, "must be an object"),
])
def test_check_refuses_what_it_cannot_draw(spec, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace("[", r"\[")):
        check_deckgl(spec, COLUMNS)


def test_check_suggests_a_close_column():
    spec = {"layers": [{"@@type": "ScatterplotLayer", "getPosition": "@@=[xlon, ylat]"}]}
    with pytest.raises(ValueError, match="Did you mean 'xlong'.*Its columns are: state, xlong, ylat, capacity"):
        check_deckgl(spec, COLUMNS)


def test_expressions_read_columns_but_not_globals_strings_or_calls():
    assert expression_names("[xlong, ylat]") == ({"xlong", "ylat"}, set())
    assert expression_names("capacity > 2 ? [255, 0, 0] : [0, 0, 255]") == ({"capacity"}, set())
    assert expression_names("Math.sqrt(capacity) * 1e3") == ({"capacity"}, set())
    assert expression_names("state === 'TX' ? 1 : 0") == ({"state"}, set())
    assert expression_names("-") == (set(), set())
    assert expression_names("[properties.value, 0, 0]", feature=True) == ({"value"}, set())
    assert expression_names("value * 2", feature=True) == (set(), {"value"})


def test_spec_fields_are_the_columns_its_layers_read():
    spec = {"layers": [
        POINTS["layers"][0],
        {"@@type": "GeoJsonLayer", "getFillColor": "@@=[properties.share * 255, 0, 0]"},
        {"@@type": "ArcLayer", "getSourcePosition": [0, 0], "getWidth": 2},
    ]}
    assert deckgl_fields(spec) == {"xlong", "ylat", "capacity", "share"}
    assert deckgl_fields({"layers": "nope"}) == set()
    assert DeckGLNode.spec_fields is deckgl_fields


def test_the_schema_contract_records_the_columns_the_map_reads():
    doc = {
        "flowdash": 4,
        "nodes": {
            "data": {"component": SOURCE_ID, "config": {"ref": "data"}},
            "turbines": {"component": TABLE_ID, "config": {"table": "turbines"}},
            "map": {"component": DECKGL_ID, "config": {"spec": POINTS}},
        },
        "edges": [
            {"from": "data.source", "to": "turbines.source"},
            {"from": "turbines.pipeline", "to": "map.pipeline"},
        ],
    }
    schema = {"turbines": {c: "number" for c in COLUMNS}}
    found = {u.node: (u.coverage, {t: sorted(c) for t, c in u.tables.items()}) for u in usages(doc, "data", schema, "duckdb")}
    assert found["map"] == (FULL, {"turbines": ["capacity", "xlong", "ylat"]})


def test_the_map_draws_the_rows_fitted_with_a_themed_basemap(turbines):
    ops = map_of(turbines, POINTS)
    view = node(ops)._object
    assert node(ops)._error is None
    assert isinstance(view, DeckGLView)
    assert len(view.pipeline.data) == 3
    assert view.panel.sizing_mode == "stretch_both"
    spec = view.spec
    assert spec["mapStyle"] == BASEMAPS["light"]
    state = spec["initialViewState"]
    assert -118 < state["longitude"] < -93.5 and 32 < state["latitude"] < 42
    assert 2 < state["zoom"] < 6
    assert spec["layers"][0]["getFillColor"] == SCHEME_COLORS["tableau10"]
    assert node(ops).spec == POINTS
    layer = view.get_panel().object["layers"][0]
    assert len(layer["data"]) == 3


def test_a_spec_keeps_its_own_view_state_colours_and_basemap(turbines):
    style = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
    spec = {
        **POINTS, "mapStyle": style, "initialViewState": {"latitude": 0, "longitude": 0, "zoom": 1},
        "layers": [{**POINTS["layers"][0], "getFillColor": [1, 2, 3]}],
    }
    drawn = node(map_of(turbines, spec))._object.spec
    assert drawn["mapStyle"] == style
    assert drawn["initialViewState"] == spec["initialViewState"]
    assert drawn["layers"][0]["getFillColor"] == [1, 2, 3]
    without = node(map_of(turbines, {**POINTS, "mapStyle": None}))._object.spec
    assert without["mapStyle"] is None


def test_the_map_follows_the_dashboards_colour_scheme(turbines):
    ops = map_of(turbines, POINTS, settings_class=SchemeSettings)
    assert node(ops)._object.spec["layers"][0]["getFillColor"] == SCHEME_COLORS["teals"]
    ops.editor.settings.color_scheme = "oranges"
    assert node(ops)._object.spec["layers"][0]["getFillColor"] == SCHEME_COLORS["oranges"]


def test_the_basemap_follows_the_pages_theme(turbines, monkeypatch):
    ops = map_of(turbines, {**POINTS, "mapStyle": BASEMAPS["light"]})
    frame = node(ops)._body()
    assert frame.dark_theme is False
    # A page switches the session's theme before its components.
    monkeypatch.setattr(pn.config, "theme", "dark")
    frame.dark_theme = True
    assert node(ops)._object.spec["mapStyle"] == BASEMAPS["dark"]
    monkeypatch.setattr(pn.config, "theme", "default")
    frame.dark_theme = False
    assert node(ops)._object.spec["mapStyle"] == BASEMAPS["light"]


def test_a_chosen_basemap_is_not_drawn_again_on_a_theme_switch(turbines):
    style = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
    ops = map_of(turbines, {**POINTS, "mapStyle": style})
    drawn = node(ops)._object
    node(ops)._body().dark_theme = True
    assert node(ops)._object is drawn


def test_the_map_is_drawn_dark_in_a_dark_session(turbines, monkeypatch):
    monkeypatch.setattr(pn.config, "theme", "dark")
    ops = map_of(turbines, POINTS)
    assert node(ops)._object.spec["mapStyle"] == BASEMAPS["dark"]


def test_a_spec_the_data_cannot_draw_is_refused(turbines):
    ops = map_of(turbines, POINTS)
    bad = {"layers": [{"@@type": "ScatterplotLayer", "getPosition": "@@=[lon, lat]"}]}
    with pytest.raises(OperationError, match="reads lat, lon"):
        ops.set_config("map", {"spec": bad})
    assert node(ops)._object.spec["layers"][0]["getPosition"] == "@@=[xlong, ylat]"

    check = DeckGLNode.__panel_app_metadata__.check_config
    pipeline = node(ops, "turbines")._object
    found = check({"spec": bad}, {"pipeline": pipeline})
    assert isinstance(found, CheckDiagnostic) and found.field == "spec"
    assert check({"spec": POINTS}, {"pipeline": pipeline}) is None
    assert check({"spec": {}}, {"pipeline": pipeline}) is None


def test_the_card_previews_rows_and_says_what_the_map_shows(turbines):
    ops = map_of(turbines, POINTS)
    preview = node(ops).__flowdash_preview__()
    assert isinstance(preview, pd.DataFrame)
    assert sorted(preview["state"]) == ["CA", "IA", "TX"]
    assert node(ops)._summary(POINTS, preview) == "Scatterplot map of xlong and ylat, 3 rows"
    two = {"layers": [*POINTS["layers"], {"@@type": "HexagonLayer", "getPosition": "@@=[xlong, ylat]"}]}
    assert node(ops)._summary(two, None) == "Map of 2 layers (Scatterplot, Hexagon) of xlong and ylat"
    arcs = {"layers": [{"@@type": "ArcLayer", "getWidth": "@@=capacity"}]}
    assert node(ops)._summary(arcs, [1]) == "Arc map of capacity, 1 row"
    ops.editor._open_card_popout(ops._instance("map"))


def test_fitting_skips_rows_off_the_globe_and_maps_without_positions():
    data = pd.DataFrame({"xlong": [10.0, 11.0, 999.0], "ylat": [50.0, 51.0, 0.0]})
    state = fit_view_state(POINTS, data)
    assert 10 <= state["longitude"] <= 11 and 50 <= state["latitude"] <= 51
    assert state["zoom"] > 6
    assert fit_view_state({"layers": [{"@@type": "HexagonLayer"}]}, data) is None
    assert fit_view_state(POINTS, data.iloc[2:]) is None
    assert fit_view_state(POINTS, pd.DataFrame({"x": [1]})) is None
