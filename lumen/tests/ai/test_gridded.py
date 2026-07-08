import datetime

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("xarray_sql")

from lumen.ai.config import PROMPTS_DIR
from lumen.ai.utils import gridded_metadata, render_template
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


def test_hvplot_prompt_no_gridded_rules_for_tabular():
    rendered = render_template(
        PROMPTS_DIR / "hvPlotAgent" / "main.jinja2",
        **_base_context(gridded=None),
    )
    assert "For gridded xarray data" not in rendered
