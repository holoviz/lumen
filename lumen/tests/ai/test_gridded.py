import datetime

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("xarray_sql")

from lumen.ai.config import PROMPTS_DIR
from lumen.ai.gridded import detect_gridded
from lumen.ai.utils import render_template
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.sources.xarray_sql import XArraySQLSource
from lumen.util import check_xarray_available

pytestmark = pytest.mark.skipif(
    not check_xarray_available(), reason="xarray not installed"
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


# ---- detect_gridded tests ----

def test_detect_gridded_xarray_source(xarray_pipeline):
    result = detect_gridded(xarray_pipeline)
    assert result is not None
    assert result["source_type"] == "xarray"
    assert "lat" in result["dims"]
    assert "lon" in result["dims"]
    assert "air" in result["data_vars"]


def test_detect_gridded_pandas_source(tabular_pipeline):
    result = detect_gridded(tabular_pipeline)
    assert result is None


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
    assert "kind='quadmesh'" in rendered


def test_baseview_prompt_no_gridded_block_for_tabular():
    rendered = render_template(
        PROMPTS_DIR / "BaseViewAgent" / "main.jinja2",
        **_base_context(gridded=None),
    )
    assert "xarray dataset" not in rendered
    assert "gridded plot kinds" not in rendered


def test_hvplot_prompt_includes_gridded_rules():
    gridded = {
        "source_type": "xarray",
        "dims": ["lat", "lon"],
        "coords": {"lat": [3], "lon": [4]},
        "data_vars": ["air"],
    }
    rendered = render_template(
        PROMPTS_DIR / "hvPlotAgent" / "main.jinja2",
        **_base_context(gridded=gridded),
    )
    assert "quadmesh" in rendered
    assert "z" in rendered


def test_hvplot_prompt_no_gridded_rules_for_tabular():
    rendered = render_template(
        PROMPTS_DIR / "hvPlotAgent" / "main.jinja2",
        **_base_context(gridded=None),
    )
    assert "For gridded xarray data" not in rendered
