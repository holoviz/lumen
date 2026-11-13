"""The hvPlot agent could not produce a renderable spec at all.

Each test here stands for one step of that failure: the schema offered the
model a value its own view rejects, the recovery path raised instead of
recovering, and the one field the agent adds to the schema reached the view.
"""
import holoviews as hv
import pandas as pd
import param
import pytest

from pydantic import BaseModel

from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.translate import param_to_pydantic
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.views.base import hvPlotUIView, hvPlotView


@pytest.fixture
def pipeline():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    return Pipeline(source=InMemorySource(tables={"points": df}), table="points")


# ---- A Selector holding None offered the model the string "None" ----

class _Aggregating(param.Parameterized):

    aggregator = param.Selector(default=None, objects=[None, "count", "mean"], doc="""
        How to reduce the rows landing in one pixel.""")


def test_a_none_option_is_not_offered_as_a_string():
    """param names the None option 'None' for display, and that name was being
    handed to the model as though it were a value."""
    model = param_to_pydantic(_Aggregating, base_model=BaseModel)[_Aggregating.__name__]

    options = model.model_json_schema()["properties"]["aggregator"]["anyOf"]
    enum = next(option["enum"] for option in options if "enum" in option)

    assert "None" not in enum
    assert {"type": "null"} in options
    assert set(enum) >= {"count", "mean"}


def test_an_optional_selector_is_not_required():
    """Requiring it leaves the model no way to say 'no aggregation', which is
    what its default means."""
    model = param_to_pydantic(_Aggregating, base_model=BaseModel)[_Aggregating.__name__]

    schema = model.model_json_schema()

    assert "aggregator" not in schema.get("required", [])
    assert schema["properties"]["aggregator"]["default"] is None


def test_the_view_accepts_every_aggregator_the_schema_offers(pipeline):
    """The schema and the view have to agree, or a valid answer fails to render."""
    model = hvPlotAgent()._get_model("main", {"x": {}, "y": {}})
    options = model.model_json_schema()["properties"]["aggregator"]["anyOf"]
    enum = next(option["enum"] for option in options if "enum" in option)

    for aggregator in enum:
        hvPlotUIView(pipeline=pipeline, kind="scatter", x="x", y="y", aggregator=aggregator)


# ---- The agent's own extra field reached the view ----

async def test_chain_of_thought_does_not_reach_the_view(pipeline):
    """It is added to the schema to make the model reason, not to be plotted;
    hvPlot rejects a keyword it does not know."""
    spec = {"kind": "scatter", "x": "x", "y": "y", "chain_of_thought": "because"}

    extracted = await hvPlotAgent()._extract_spec({"pipeline": pipeline}, spec)

    assert "chain_of_thought" not in extracted
    hvPlotAgent.view_type(pipeline=pipeline, **extracted).get_panel()


# ---- The recovery path raised instead of recovering ----

def test_a_prompt_other_than_main_still_resolves_its_model():
    """_get_model is called for every prompt, so overriding it with a required
    extra argument broke revise, which is what runs when a spec fails."""
    assert hvPlotAgent()._get_model("revise_output") is not None


class _Colormapped(param.Parameterized):

    cmap = param.Selector(default="viridis", objects=["viridis", "fire"], check_on_set=False, doc="""
        The colormap to use.""")


def test_a_value_outside_the_choices_does_not_break_the_model():
    """A Selector that accepts more than it lists (hvPlot reads a dict cmap as a
    color key) puts that value in range, and it is not a Literal member."""
    instance = _Colormapped(cmap={"Irish": "#e41a1c"})

    model = param_to_pydantic(type(instance), base_model=BaseModel)[type(instance).__name__]

    assert set(model.model_json_schema()["properties"]["cmap"]["enum"]) == {"viridis", "fire"}


async def test_an_axis_naming_a_missing_column_is_dropped(pipeline):
    """The model can answer with a column it expected its query to create, and
    hvPlot only finds out while drawing, where it raises a bare KeyError."""
    spec = {"kind": "bar", "x": "elevation_band", "y": "y", "by": ["nope"], "groupby": ["x"]}

    extracted = await hvPlotAgent()._extract_spec({"pipeline": pipeline}, spec)

    assert "x" not in extracted
    assert extracted["y"] == "y"
    assert "by" not in extracted
    assert extracted["groupby"] == ["x"]


# ---- Maps: a basemap under longitude and latitude ----

def _geo_pipeline(crs):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    from lumen.sources.duckdb import DuckDBSource
    gdf = gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=[Point(13.4, 52.5), Point(2.3, 48.9)], crs="EPSG:4326").to_crs(crs)
    return Pipeline(source=DuckDBSource.from_df({"places": gdf}), table="places")


async def test_lon_lat_geometry_gets_a_basemap():
    spec = await hvPlotAgent()._extract_spec({"pipeline": _geo_pipeline("EPSG:4326")}, {"kind": "points", "c": "v"})
    assert spec["geo"] is True and spec["tiles"] == "OSM"


async def test_a_map_gets_a_fixed_height_not_a_responsive_frame():
    spec = await hvPlotAgent()._extract_spec({"pipeline": _geo_pipeline("EPSG:4326")}, {"kind": "points", "c": "v"})
    assert "responsive" not in spec and spec["frame_height"] == 400
    plain = await hvPlotAgent()._extract_spec({"pipeline": _geo_pipeline("EPSG:3857")}, {"kind": "points", "c": "v"})
    assert plain["responsive"] is True and "frame_height" not in plain


async def test_a_chart_has_grid_lines():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    pipe = Pipeline(source=InMemorySource(tables={"t": df}), table="t")
    spec = await hvPlotAgent()._extract_spec({"pipeline": pipe}, {"kind": "line", "x": "x", "y": "y"})
    assert spec["grid"] is True
    view = hvPlotView(pipeline=pipe, **spec)
    fig = hv.render(view.get_plot(view.get_data()))
    assert all(g.grid_line_color is not None for g in fig.grid)


async def test_a_map_keeps_the_column_it_is_coloured_by():
    spec = await hvPlotAgent()._extract_spec({"pipeline": _geo_pipeline("EPSG:4326")}, {"kind": "points", "z": "v"})
    assert spec["z"] == "v"


async def test_projected_geometry_gets_no_basemap():
    spec = await hvPlotAgent()._extract_spec({"pipeline": _geo_pipeline("EPSG:3857")}, {"kind": "points", "c": "v"})
    assert "tiles" not in spec and "geo" not in spec


async def test_lon_lat_columns_get_a_basemap_unless_turned_off():
    df = pd.DataFrame({"lon": [13.4, 2.3], "lat": [52.5, 48.9]})
    pipe = Pipeline(source=InMemorySource(tables={"pts": df}), table="pts")
    spec = await hvPlotAgent()._extract_spec({"pipeline": pipe}, {"kind": "points", "x": "lon", "y": "lat"})
    assert spec["tiles"] == "OSM"
    off = await hvPlotAgent()._extract_spec({"pipeline": pipe}, {"kind": "points", "x": "lon", "y": "lat", "geo": False})
    assert "tiles" not in off


async def test_a_gridded_map_gets_coastlines_not_tiles():
    df = pd.DataFrame({"lon": [0.0, 1.0, 0.0, 1.0], "lat": [0.0, 0.0, 1.0, 1.0], "t": [1.0, 2.0, 3.0, 4.0]})
    pipe = Pipeline(source=InMemorySource(tables={"grid": df}), table="grid")
    spec = await hvPlotAgent()._extract_spec(
        {"pipeline": pipe}, {"kind": "quadmesh", "x": "lon", "y": "lat", "z": "t"}
    )
    assert "tiles" not in spec and spec["coastline"] is True
    image = await hvPlotAgent()._extract_spec({"pipeline": pipe}, {"kind": "image", "x": "lon", "y": "lat", "z": "t"})
    assert image["kind"] == "quadmesh"


def test_a_bar_chart_without_x_takes_its_text_column():
    spec = {"kind": "barh", "y": "density"}
    hvPlotAgent._complete_axes(spec, pd.DataFrame({"region": ["a", "b"], "density": [1.0, 2.0]}))
    assert spec["x"] == "region"


def test_points_covering_a_lon_lat_grid_become_a_quadmesh():
    grid = pd.DataFrame({"lon": [0.0, 1.0, 0.0, 1.0], "lat": [0.0, 0.0, 1.0, 1.0], "t": [1.0, 2.0, 3.0, 4.0]})
    spec = {"kind": "points", "x": "lon", "y": "lat"}
    hvPlotAgent._complete_axes(spec, grid)
    assert spec["kind"] == "quadmesh" and spec["z"] == "t"
    quakes = pd.DataFrame({"lon": [0.0, 1.0, 2.0], "lat": [0.0, 1.0, 2.0], "mag": [4.0, 5.0, 6.0]})
    spec = {"kind": "points", "x": "lon", "y": "lat"}
    hvPlotAgent._complete_axes(spec, quakes)
    assert spec["kind"] == "points" and "z" not in spec


def test_the_schema_offers_size_as_a_column():
    model = hvPlotAgent()._get_model("main", {"x": {}, "y": {}})
    options = model.model_json_schema()["properties"]["size"]["anyOf"]
    assert {"type": "null"} in options


async def test_a_size_naming_no_column_is_dropped(pipeline):
    spec = await hvPlotAgent()._extract_spec({"pipeline": pipeline}, {"kind": "scatter", "x": "x", "y": "y", "size": "y"})
    assert spec["size"] == "y"
    spec = {"kind": "scatter", "x": "x", "y": "y", "size": "population"}
    hvPlotAgent._drop_unknown_columns(spec, {"x", "y"})
    assert "size" not in spec


async def test_points_keep_the_column_they_are_coloured_by(pipeline):
    spec = await hvPlotAgent()._extract_spec({"pipeline": pipeline}, {"kind": "scatter", "x": "x", "y": "y", "z": "y"})
    assert spec["z"] == "y"


async def test_a_large_grid_is_rasterized(monkeypatch):
    pytest.importorskip("datashader")
    import lumen.ai.agents.hvplot as hvplot_agent
    monkeypatch.setattr(hvplot_agent, "GRID_RASTERIZE_CELLS", 3)
    df = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0], "t": [1.0, 2.0, 3.0, 4.0]})
    pipe = Pipeline(source=InMemorySource(tables={"grid": df}), table="grid")
    spec = await hvPlotAgent()._extract_spec({"pipeline": pipe}, {"kind": "quadmesh", "x": "x", "y": "y", "z": "t"})
    assert spec["rasterize"] is True


async def test_without_datashader_a_large_grid_is_not_rasterized(monkeypatch):
    import importlib.util

    import lumen.ai.agents.hvplot as hvplot_agent
    monkeypatch.setattr(hvplot_agent, "GRID_RASTERIZE_CELLS", 3)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "datashader" else real(name, *a))
    df = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0], "t": [1.0, 2.0, 3.0, 4.0]})
    pipe = Pipeline(source=InMemorySource(tables={"grid": df}), table="grid")
    spec = await hvPlotAgent()._extract_spec({"pipeline": pipe}, {"kind": "quadmesh", "x": "x", "y": "y", "z": "t"})
    assert "rasterize" not in spec


# ---- A colour column gets a colormap that fits it ----

@pytest.fixture
def coloured():
    df = pd.DataFrame({
        "x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "anomaly": [-1.5, 0.2, 2.0],
        "wind_dir": [10.0, 200.0, 350.0], "air": [280.0, 290.0, 300.0], "celsius": [-5.0, 2.0, 12.0],
        "aspect_ratio": [1.0, 1.5, 2.0], "kind": ["a", "b", "a"],
    })
    return Pipeline(source=InMemorySource(tables={"t": df}), table="t")


@pytest.mark.parametrize("column, cmap", [
    ("anomaly", "coolwarm"), ("wind_dir", "colorwheel"), ("air", None),
    ("celsius", None), ("aspect_ratio", None), ("kind", None),
])
async def test_the_colour_column_picks_a_colormap(coloured, column, cmap):
    spec = await hvPlotAgent()._extract_spec({"pipeline": coloured}, {"kind": "points", "x": "x", "y": "y", "z": column})
    assert spec.get("cmap") == cmap
    assert spec.get("symmetric") is (True if cmap == "coolwarm" else None)


async def test_a_colormap_the_model_chose_is_kept(coloured):
    spec = await hvPlotAgent()._extract_spec(
        {"pipeline": coloured}, {"kind": "points", "x": "x", "y": "y", "z": "anomaly", "cmap": "fire"}
    )
    assert spec["cmap"] == "fire" and "symmetric" not in spec


async def test_a_signed_colour_is_drawn_centred_on_zero(coloured):
    spec = await hvPlotAgent()._extract_spec({"pipeline": coloured}, {"kind": "points", "x": "x", "y": "y", "z": "anomaly"})
    view = hvPlotView(pipeline=coloured, **spec)
    plot = view.get_plot(view.get_data())
    assert hv.Store.lookup_options("bokeh", plot, "style").kwargs["cmap"] == "coolwarm"
    assert hv.Store.lookup_options("bokeh", plot, "plot").kwargs["symmetric"] is True


def test_the_model_chooses_cmap_from_a_short_list():
    model = hvPlotAgent()._get_model("main", schema={"type": "object", "properties": {"x": {"type": "number"}}})
    schema = model.model_json_schema()["properties"]["cmap"]
    assert schema["anyOf"][0]["enum"] == ["kbc_r", "fire", "coolwarm", "colorwheel"]
    assert "\n" not in schema["description"]
