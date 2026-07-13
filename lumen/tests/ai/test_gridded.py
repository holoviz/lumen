import datetime

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("xarray_sql")

from lumen.ai.config import PROMPTS_DIR
from lumen.ai.utils import (
    _spec_field_references, gridded_metadata, render_template,
    subset_gridded_to_2d,
)
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.sources.xarray_sql import XArraySQLSource
from lumen.util import try_import_xarray

pytestmark = pytest.mark.skipif(
    try_import_xarray() is None, reason="xarray not installed"
)


# ---- Fixtures ----

@pytest.fixture
def simple_dataset():
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([10.0, 20.0, 30.0, 40.0])
    air = np.arange(12, dtype=float).reshape(3, 4)
    return xr.Dataset(
        {"air": (["lat", "lon"], air)},
        coords={"lat": lats, "lon": lons},
    )


@pytest.fixture
def xarray_source(simple_dataset):
    return XArraySQLSource(_dataset=simple_dataset)


@pytest.fixture
def xarray_pipeline(xarray_source):
    return Pipeline(source=xarray_source, table="air")


@pytest.fixture
def simple_dataset_3d():
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([10.0, 20.0, 30.0, 40.0])
    times = np.array([0, 1], dtype="datetime64[D]")
    air = np.arange(24, dtype=float).reshape(2, 3, 4)
    return xr.Dataset(
        {"air": (["time", "lat", "lon"], air)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


@pytest.fixture
def simple_dataset_3d_pipeline(simple_dataset_3d):
    return Pipeline(source=XArraySQLSource(_dataset=simple_dataset_3d), table="air")


@pytest.fixture
def tabular_pipeline():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    source = InMemorySource(tables={"data": df})
    return Pipeline(source=source, table="data")


# ---- gridded_metadata tests ----

def test_gridded_metadata_xarray_source(xarray_pipeline):
    result = gridded_metadata(xarray_pipeline)
    assert result is not None
    assert result["source_type"] == "xarray"
    assert "lat" in result["dims"]
    assert "lon" in result["dims"]
    assert "air" in result["data_vars"]
    # simple_dataset has evenly-spaced lats [0,1,2] and lons [10,20,30,40]
    assert result["regular"] is True


def test_gridded_metadata_pandas_source(tabular_pipeline):
    result = gridded_metadata(tabular_pipeline)
    assert result is None


def test_gridded_metadata_regular_flag_false_for_irregular():
    """A non-uniformly-spaced coord must produce regular=False."""
    irregular_lats = np.array([0.0, 1.0, 5.0])
    lons = np.array([10.0, 20.0, 30.0, 40.0])
    air = np.arange(12, dtype=float).reshape(3, 4)
    ds = xr.Dataset(
        {"air": (["lat", "lon"], air)},
        coords={"lat": irregular_lats, "lon": lons},
    )
    pipeline = Pipeline(source=XArraySQLSource(_dataset=ds), table="air")
    result = gridded_metadata(pipeline)
    assert result is not None
    assert result["regular"] is False


def test_gridded_metadata_handles_datetime_coord():
    """datetime64 coords must not blow up the regularity check."""
    times = pd.date_range("2026-01-01", periods=4, freq="D").values
    lons = np.array([10.0, 20.0, 30.0])
    temp = np.arange(12, dtype=float).reshape(4, 3)
    ds = xr.Dataset(
        {"temp": (["time", "lon"], temp)},
        coords={"time": times, "lon": lons},
    )
    pipeline = Pipeline(source=XArraySQLSource(_dataset=ds), table="temp")
    result = gridded_metadata(pipeline)
    assert result is not None
    # daily-spaced datetimes are regular
    assert result["regular"] is True


# ---- Prompt rendering tests ----

def _base_context(gridded=None):
    """Minimal context required to render BaseViewAgent/main.jinja2."""
    class _FakeSource:
        dialect = None

    class _FakePipeline:
        table = "air"
        source = _FakeSource()

    class _FakeMemory(dict):
        pass

    memory = _FakeMemory({
        "pipeline": _FakePipeline(),
        "data": "lat,lon,air\n0,10,1.0",
    })
    return dict(
        memory=memory,
        doc="hvPlotUIView description",
        gridded=gridded,
        current_datetime=datetime.datetime(2026, 1, 1),
        actor_name="hvPlotAgent",
    )


def test_baseview_prompt_includes_gridded_block():
    gridded = {
        "source_type": "xarray",
        "dims": ["lat", "lon"],
        "coords": {"lat": [3], "lon": [4]},
        "data_vars": ["air"],
    }
    rendered = render_template(
        PROMPTS_DIR / "BaseViewAgent" / "main.jinja2",
        **_base_context(gridded=gridded),
    )
    assert "xarray dataset" in rendered
    assert "lat, lon" in rendered
    assert "air" in rendered
    # The base prompt must stay agent-agnostic: no hvPlot-specific kind
    # values, since VegaLite and DeckGL inherit this template too.
    assert "kind=" not in rendered
    assert "quadmesh" not in rendered
    assert "heatmap" not in rendered


def test_baseview_prompt_no_gridded_block_for_tabular():
    rendered = render_template(
        PROMPTS_DIR / "BaseViewAgent" / "main.jinja2",
        **_base_context(gridded=None),
    )
    assert "xarray dataset" not in rendered
    assert "gridded plot kinds" not in rendered


def test_hvplot_prompt_recommends_image_for_regular_grid():
    gridded = {
        "source_type": "xarray",
        "dims": ["lat", "lon"],
        "coords": {"lat": [3], "lon": [4]},
        "data_vars": ["air"],
        "regular": True,
    }
    rendered = render_template(
        PROMPTS_DIR / "hvPlotAgent" / "main.jinja2",
        **_base_context(gridded=gridded),
    )
    assert "kind='image'" in rendered
    assert "regularly-spaced coordinates" in rendered
    assert "Do NOT use 'image'" not in rendered


def test_hvplot_prompt_recommends_quadmesh_for_irregular_grid():
    gridded = {
        "source_type": "xarray",
        "dims": ["lat", "lon"],
        "coords": {"lat": [3], "lon": [4]},
        "data_vars": ["air"],
        "regular": False,
    }
    rendered = render_template(
        PROMPTS_DIR / "hvPlotAgent" / "main.jinja2",
        **_base_context(gridded=gridded),
    )
    assert "kind='quadmesh'" in rendered
    assert "irregularly-spaced coordinates" in rendered
    assert "Do NOT use 'image'" in rendered


def test_hvplot_prompt_pages_remaining_dims_with_groupby():
    """The hvPlot prompt lists the raw dims (no lon/lat prescription) and tells
    the LLM to page any remaining dimension with groupby."""
    gridded = {
        "source_type": "xarray",
        "dims": ["time", "lat", "lon"],
        "coords": {"time": [2], "lat": [3], "lon": [4]},
        "data_vars": ["air"],
        "regular": True,
    }
    rendered = render_template(
        PROMPTS_DIR / "hvPlotAgent" / "main.jinja2",
        **_base_context(gridded=gridded),
    )
    assert "time, lat, lon" in rendered
    assert "groupby='time'" in rendered
    # no hard-coded spatial-axis prescription any more
    assert "spatial axes" not in rendered


def test_gridded_metadata_stays_agnostic_about_spatial_dims(simple_dataset_3d_pipeline):
    """gridded_metadata reports the raw dims and does not label which are
    spatial, so the view spec decides the axes (a lon/time Hovmoller is valid)."""
    md = gridded_metadata(simple_dataset_3d_pipeline)
    assert md["dims"] == ["time", "lat", "lon"]
    assert "spatial_dims" not in md
    assert "extra_dims" not in md


def test_spec_field_references_vegalite_and_deckgl():
    """Field references are pulled from Vega-Lite encoding fields and deck.gl
    ``@@=`` accessors, each under its own explicit ``kind``."""
    vega = {"encoding": {"x": {"field": "lon"}, "y": {"field": "time"},
                         "color": {"field": "air"}}}
    assert _spec_field_references(vega, "vega-lite") == {"lon", "time", "air"}
    deck = {"layers": [{"getPosition": "@@=[lon, lat]", "getElevation": "@@=air"}]}
    assert {"lon", "lat", "air"} <= _spec_field_references(deck, "deckgl")


def test_subset_collapses_dims_absent_from_spec(simple_dataset_3d_pipeline):
    """Dims the spec does not reference (here time on a lon/lat map) collapse to
    a single slice; the spec drives the subset, not a name guess."""
    spec = {"encoding": {"x": {"field": "lon"}, "y": {"field": "lat"}}}
    n_full = len(simple_dataset_3d_pipeline.data)
    sub = subset_gridded_to_2d(simple_dataset_3d_pipeline, spec, "vega-lite")
    assert len(sub.data) < n_full
    assert sub.data["time"].nunique() == 1


def test_subset_keeps_dims_the_spec_uses(simple_dataset_3d_pipeline):
    """A lon/time spec (Hovmoller) keeps time as an axis; lat collapses instead."""
    spec = {"encoding": {"x": {"field": "lon"}, "y": {"field": "time"}}}
    sub = subset_gridded_to_2d(simple_dataset_3d_pipeline, spec, "vega-lite")
    assert sub.data["time"].nunique() > 1
    assert sub.data["lat"].nunique() == 1


def test_subset_noop_when_spec_uses_all_dims(xarray_pipeline):
    """When the spec references every dim, the pipeline is returned unchanged."""
    spec = {"encoding": {"x": {"field": "lon"}, "y": {"field": "lat"}}}
    assert subset_gridded_to_2d(xarray_pipeline, spec, "vega-lite") is xarray_pipeline


def test_subset_noop_for_tabular(tabular_pipeline):
    """A non-xarray pipeline is returned unchanged."""
    assert subset_gridded_to_2d(tabular_pipeline, {}, "deckgl") is tabular_pipeline


def test_hvplot_prompt_no_gridded_rules_for_tabular():
    rendered = render_template(
        PROMPTS_DIR / "hvPlotAgent" / "main.jinja2",
        **_base_context(gridded=None),
    )
    assert "For gridded xarray data" not in rendered


# ---- Pipeline.get_dataset + gridded view rendering from to_dataset ----

def test_pipeline_get_dataset_returns_gridded(simple_dataset_3d_pipeline):
    """A plain xarray-backed pipeline yields a compact gridded Dataset."""
    ds = simple_dataset_3d_pipeline.get_dataset()
    assert isinstance(ds, xr.Dataset)
    assert set(ds.dims) >= {"time", "lat", "lon"}
    assert "air" in ds.data_vars


def test_pipeline_get_dataset_none_for_tabular(tabular_pipeline):
    """A non-xarray source has no to_dataset, so get_dataset returns None."""
    assert tabular_pipeline.get_dataset() is None


def test_pipeline_get_dataset_none_with_python_transform(simple_dataset_3d):
    """A Python transform means the pipeline can't defer to the source's
    gridded to_dataset, so get_dataset returns None (callers use the frame)."""
    from lumen.transforms.base import Columns
    source = XArraySQLSource(_dataset=simple_dataset_3d)
    pipeline = Pipeline(
        source=source, table="air", transforms=[Columns(columns=["lat", "lon", "air"])]
    )
    assert pipeline.get_dataset() is None


def test_hvplotview_renders_gridded_from_source_dataset(simple_dataset_3d_pipeline):
    """hvPlotView renders an image from the source's gridded Dataset (paging the
    extra time dim), not by pivoting the long-form frame."""
    from lumen.views.base import hvPlotView
    view = hvPlotView(
        pipeline=simple_dataset_3d_pipeline,
        kind="image", x="lon", y="lat", z="air", groupby="time",
    )
    plot = view.get_plot(view.get_data())
    import holoviews as hv
    assert isinstance(plot, (hv.DynamicMap, hv.HoloMap))


def test_hvplotuiview_uses_grid_explorer_for_gridded(simple_dataset_3d_pipeline):
    """The AI explorer view (hvPlotUIView) renders a gridded 'image' via hvPlot's
    grid explorer over the compact Dataset -- the long-form dataframe explorer
    would raise 'image requires gridded data'."""
    import panel as pn

    from hvplot.ui import hvGridExplorer

    from lumen.views.base import hvPlotUIView
    view = hvPlotUIView(
        pipeline=simple_dataset_3d_pipeline,
        kind="image", x="lon", y="lat", z="air", groupby="time",
    )
    panel = view.get_panel()
    assert isinstance(panel, hvGridExplorer)
    pn.panel(panel).get_root()  # must not raise


def test_hvplotuiview_uses_dataframe_explorer_for_tabular(tabular_pipeline):
    """A non-xarray pipeline still uses the dataframe explorer."""
    from hvplot.ui import hvDataFrameExplorer

    from lumen.views.base import hvPlotUIView
    view = hvPlotUIView(pipeline=tabular_pipeline, kind="scatter", x="x", y="y")
    assert isinstance(view.get_panel(), hvDataFrameExplorer)
