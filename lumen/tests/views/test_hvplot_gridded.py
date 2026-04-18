import numpy as np
import pandas as pd
import pytest

from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.views.base import hvPlotView

# ---- Fixtures ----

@pytest.fixture
def gridded_df():
    """Long-form 2D grid: lat x lon with an 'air' value column."""
    lats = np.repeat([0.0, 1.0, 2.0], 4)
    lons = np.tile([10.0, 20.0, 30.0, 40.0], 3)
    air = np.arange(12, dtype=float)
    return pd.DataFrame({"lat": lats, "lon": lons, "air": air})


@pytest.fixture
def gridded_pipeline(gridded_df):
    source = InMemorySource(tables={"grid": gridded_df})
    return Pipeline(source=source, table="grid")


@pytest.fixture
def tabular_df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


@pytest.fixture
def tabular_pipeline(tabular_df):
    source = InMemorySource(tables={"data": tabular_df})
    return Pipeline(source=source, table="data")


# ---- Tests ----

def test_hvplot_C_param_accepted():
    """C param is a valid Selector on hvPlotBaseView — no AttributeError."""
    assert hasattr(hvPlotView.param, "C")


def test_hvplot_quadmesh_in_kind_objects():
    """quadmesh is a recognised kind value."""
    assert "quadmesh" in hvPlotView.param.kind.objects


def test_hvplot_heatmap_from_longform_df(gridded_pipeline):
    """kind=heatmap with x, y, C renders a 2D HeatMap from a long-form grid."""
    import holoviews as hv
    view = hvPlotView(
        pipeline=gridded_pipeline,
        kind="heatmap",
        x="lon",
        y="lat",
        C="air",
    )
    panel = view.get_panel()
    assert panel is not None
    plot = view.get_plot(view.get_data())
    assert isinstance(plot, hv.HeatMap)


def test_hvplot_C_none_does_not_break_scatter(tabular_pipeline):
    """When C is None (default), scatter plots work unchanged."""
    view = hvPlotView(
        pipeline=tabular_pipeline,
        kind="scatter",
        x="x",
        y="y",
    )
    panel = view.get_panel()
    assert panel is not None
