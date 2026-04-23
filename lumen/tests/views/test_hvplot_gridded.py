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

    heatmap stays on pandas (no pivot), so we can patch DataFrame.hvplot.
    For contourf/quadmesh/image the view pivots to xarray, so we patch the
    pivoted DataArray's hvplot accessor instead.
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
    gridded = contour_view._to_gridded(df)
    with patch.object(type(gridded), "hvplot", MagicMock()) as mock:
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


def test_hvplot_quadmesh_from_longform_df(gridded_pipeline):
    """kind=quadmesh renders a QuadMesh from a long-form pandas DataFrame.

    Long-form pandas is the only data format XArraySQLSource produces, so the
    view must pivot it to a 2D array before handing to hvPlot, otherwise
    hvPlot raises NotImplementedError for quadmesh on pandas. Addresses
    @ahuang11's review: quadmesh/image are semantically more correct than
    heatmap for continuous-coordinate gridded data.
    """
    import holoviews as hv
    view = hvPlotView(
        pipeline=gridded_pipeline,
        kind="quadmesh",
        x="lon",
        y="lat",
        z="air",
    )
    plot = view.get_plot(view.get_data())
    assert isinstance(plot, hv.QuadMesh)


def test_hvplot_image_from_longform_df(gridded_pipeline):
    """kind=image renders an Image from a long-form pandas DataFrame via pivot."""
    import holoviews as hv
    view = hvPlotView(
        pipeline=gridded_pipeline,
        kind="image",
        x="lon",
        y="lat",
        z="air",
    )
    plot = view.get_plot(view.get_data())
    assert isinstance(plot, hv.Image)


def test_hvplot_gridded_size_guard_raises(gridded_pipeline):
    """Pivoting a grid larger than _GRIDDED_MAX_CELLS raises ValueError."""
    view = hvPlotView(
        pipeline=gridded_pipeline,
        kind="quadmesh",
        x="lon",
        y="lat",
        z="air",
    )
    view._GRIDDED_MAX_CELLS = 4  # the fixture has 3 lats x 4 lons = 12 cells
    with pytest.raises(ValueError, match="exceeding the 4 safety cap"):
        view.get_plot(view.get_data())


def test_hvplot_gridded_skips_pivot_if_xarray_unavailable(gridded_pipeline):
    """If hvplot.xarray failed to import, _to_gridded returns the df unchanged.

    Build a view with a non-gridded kind so construction doesn't trigger the
    quadmesh render path, then flip the flag and call _to_gridded directly.
    """
    import pandas as pd
    view = hvPlotView(
        pipeline=gridded_pipeline,
        kind="scatter",
        x="lon",
        y="lat",
        z="air",
    )
    view._hvplot_xarray_available = False
    result = view._to_gridded(view.get_data())
    assert isinstance(result, pd.DataFrame), "expected df fallback when xarray unavailable"


def test_hvplot_gridded_passes_through_non_dataframe(gridded_pipeline):
    """If the pipeline hands over an already-gridded xarray object, skip pivot."""
    import xarray as xr
    view = hvPlotView(
        pipeline=gridded_pipeline,
        kind="quadmesh",
        x="lon",
        y="lat",
        z="air",
    )
    df = view.get_data()
    already_gridded = df.set_index(["lat", "lon"])["air"].to_xarray()
    assert isinstance(already_gridded, xr.DataArray)
    result = view._to_gridded(already_gridded)
    assert result is already_gridded, "xarray input should pass through unchanged"
