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

def test_hvplot_z_param_accepted():
    """z param is a valid Selector on hvPlotBaseView — no AttributeError."""
    assert "z" in hvPlotView.param


def test_hvplot_quadmesh_in_kind_objects():
    """quadmesh is a recognised kind value."""
    assert "quadmesh" in hvPlotView.param.kind.objects


def test_hvplot_heatmap_from_longform_df(gridded_pipeline):
    """kind=heatmap with x, y, z renders a 2D HeatMap from a long-form grid.

    The public z= param is mapped internally to hvPlot's C= for heatmap.
    """
    import holoviews as hv
    view = hvPlotView(
        pipeline=gridded_pipeline,
        kind="heatmap",
        x="lon",
        y="lat",
        z="air",
    )
    panel = view.get_panel()
    assert panel is not None
    plot = view.get_plot(view.get_data())
    assert isinstance(plot, hv.HeatMap)


def test_hvplot_z_maps_to_C_only_for_heatmap(gridded_pipeline):
    """z= is mapped to C= for heatmap, and passed as z= for other gridded kinds.

    Inspects the processed kwargs dict by stubbing df.hvplot so we can capture
    without hitting hvPlot's kind/data compatibility checks.
    """
    from unittest.mock import MagicMock, patch

    heatmap_view = hvPlotView(
        pipeline=gridded_pipeline, kind="heatmap", x="lon", y="lat", z="air",
    )
    df = heatmap_view.get_data()
    with patch.object(type(df), "hvplot", MagicMock()) as mock:
        heatmap_view.get_plot(df)
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("C") == "air"
        assert "z" not in call_kwargs

    contour_view = hvPlotView(
        pipeline=gridded_pipeline, kind="contourf", x="lon", y="lat", z="air",
    )
    df = contour_view.get_data()
    with patch.object(type(df), "hvplot", MagicMock()) as mock:
        contour_view.get_plot(df)
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("z") == "air"
        assert "C" not in call_kwargs


def test_hvplot_z_none_does_not_break_scatter(tabular_pipeline):
    """When z is None (default), scatter plots work unchanged."""
    view = hvPlotView(
        pipeline=tabular_pipeline,
        kind="scatter",
        x="x",
        y="y",
    )
    panel = view.get_panel()
    assert panel is not None
