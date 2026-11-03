"""The hvPlot agent could not produce a renderable spec at all.

Each test here stands for one step of that failure: the schema offered the
model a value its own view rejects, the recovery path raised instead of
recovering, and the one field the agent adds to the schema reached the view.
"""
import pandas as pd
import param
import pytest

from pydantic import BaseModel

from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.translate import param_to_pydantic
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.views.base import hvPlotUIView


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

    enum = model.model_json_schema()["properties"]["aggregator"]["enum"]

    assert "None" not in enum
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
    enum = model.model_json_schema()["properties"]["aggregator"]["enum"]

    for aggregator in enum:
        hvPlotUIView(pipeline=pipeline, kind="scatter", x="x", y="y", aggregator=aggregator)


# ---- The agent's own extra field reached the view ----

async def test_chain_of_thought_does_not_reach_the_view(pipeline):
    """It is added to the schema to make the model reason, not to be plotted;
    hvPlotExplorer rejects any keyword none of its controls claims."""
    spec = {"kind": "scatter", "x": "x", "y": "y", "chain_of_thought": "because"}

    extracted = await hvPlotAgent()._extract_spec({"pipeline": pipeline}, spec)

    assert "chain_of_thought" not in extracted
    hvPlotUIView(pipeline=pipeline, **extracted).get_panel()


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
    assert spec["geo"] is True and spec["tiles"] == "CartoLight"


async def test_projected_geometry_gets_no_basemap():
    spec = await hvPlotAgent()._extract_spec({"pipeline": _geo_pipeline("EPSG:3857")}, {"kind": "points", "c": "v"})
    assert "tiles" not in spec and "geo" not in spec


async def test_lon_lat_columns_get_a_basemap_unless_turned_off():
    df = pd.DataFrame({"lon": [13.4, 2.3], "lat": [52.5, 48.9]})
    pipe = Pipeline(source=InMemorySource(tables={"pts": df}), table="pts")
    spec = await hvPlotAgent()._extract_spec({"pipeline": pipe}, {"kind": "points", "x": "lon", "y": "lat"})
    assert spec["tiles"] == "CartoLight"
    off = await hvPlotAgent()._extract_spec({"pipeline": pipe}, {"kind": "points", "x": "lon", "y": "lat", "geo": False})
    assert "tiles" not in off


async def test_a_large_grid_is_rasterized(monkeypatch):
    import lumen.ai.agents.hvplot as hvplot_agent
    monkeypatch.setattr(hvplot_agent, "GRID_RASTERIZE_CELLS", 3)
    df = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0], "t": [1.0, 2.0, 3.0, 4.0]})
    pipe = Pipeline(source=InMemorySource(tables={"grid": df}), table="grid")
    spec = await hvPlotAgent()._extract_spec({"pipeline": pipe}, {"kind": "quadmesh", "x": "x", "y": "y", "z": "t"})
    assert spec["rasterize"] is True
