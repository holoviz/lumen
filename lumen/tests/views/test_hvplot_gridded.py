from unittest.mock import MagicMock, patch

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.views import base as views_base
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
def gridded_3d_df():
    """Long-form 3D grid: time x lat x lon (an extra dim to page via groupby)."""
    times = pd.to_datetime(["2013-01-01", "2013-01-02"])
    lats = [0.0, 1.0, 2.0]
    lons = [10.0, 20.0, 30.0, 40.0]
    idx = pd.MultiIndex.from_product([times, lats, lons], names=["time", "lat", "lon"])
    return pd.DataFrame({"air": np.arange(len(idx), dtype=float)}, index=idx).reset_index()


@pytest.fixture
def tabular_df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


@pytest.fixture
def tabular_pipeline(tabular_df):
    source = InMemorySource(tables={"data": tabular_df})
    return Pipeline(source=source, table="data")


# ---- Tests ----

def test_hvplot_z_param_accepted():
    """z param is a valid Selector on hvPlotBaseView (no AttributeError)."""
    assert "z" in hvPlotView.param


def test_hvplot_quadmesh_in_kind_objects():
    """quadmesh is a recognised kind value."""
    assert "quadmesh" in hvPlotView.param.kind.objects


def test_hvplot_heatmap_from_longform_df(gridded_pipeline):
    """kind=heatmap with x, y, z renders a 2D HeatMap from a long-form grid.

    The public z= param is mapped internally to hvPlot's C= for heatmap.
    """
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
    pytest.importorskip("xarray")
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
    hvPlot raises NotImplementedError for quadmesh on pandas. Addresses the
    review on #1823: quadmesh/image are semantically more correct than
    heatmap for continuous-coordinate gridded data.
    """
    pytest.importorskip("xarray")
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
    pytest.importorskip("xarray")
    view = hvPlotView(
        pipeline=gridded_pipeline,
        kind="image",
        x="lon",
        y="lat",
        z="air",
    )
    plot = view.get_plot(view.get_data())
    assert isinstance(plot, hv.Image)


def test_hvplot_image_groupby_pages_extra_dim(gridded_3d_df):
    """A groupby dimension (e.g. time) is folded into the pivot so a 3D grid
    renders with a widget slider instead of being rejected as duplicate rows.

    air_temperature (the #1823 review dataset) is time x lat x lon, so the view
    must page the extra time axis rather than block on repeated (lat, lon) pairs.
    """
    pytest.importorskip("xarray")
    source = InMemorySource(tables={"grid": gridded_3d_df})
    pipeline = Pipeline(source=source, table="grid")
    view = hvPlotView(
        pipeline=pipeline, kind="image", x="lon", y="lat", z="air", groupby="time",
    )
    plot = view.get_plot(view.get_data())
    # paged over time -> a DynamicMap/HoloMap keyed on time, not a bare Image
    assert isinstance(plot, (hv.DynamicMap, hv.HoloMap))
    assert "time" in [d.name for d in plot.kdims]


def test_hvplot_quadmesh_raises_on_duplicate_rows(gridded_df):
    """Duplicate (y, x) rows + a gridded kind must surface a clear ValueError,
    not hvPlot's opaque NotImplementedError on pandas."""
    pytest.importorskip("xarray")
    dup = pd.concat([gridded_df, gridded_df.iloc[[0]]], ignore_index=True)
    source = InMemorySource(tables={"grid": dup})
    pipeline = Pipeline(source=source, table="grid")
    view = hvPlotView(pipeline=pipeline, kind="quadmesh", x="lon", y="lat", z="air")
    with pytest.raises(ValueError, match="duplicate"):
        view.get_plot(view.get_data())


def test_hvplot_quadmesh_raises_when_z_missing(gridded_pipeline):
    """A gridded kind without z= must surface a clear ValueError."""
    pytest.importorskip("xarray")
    view = hvPlotView(pipeline=gridded_pipeline, kind="quadmesh", x="lon", y="lat")
    with pytest.raises(ValueError, match="x, y, and z must all be set"):
        view.get_plot(view.get_data())


def test_hvplot_gridded_size_guard_raises(gridded_pipeline, monkeypatch):
    """Pivoting a grid larger than GRIDDED_MAX_CELLS raises ValueError."""
    pytest.importorskip("xarray")
    view = hvPlotView(
        pipeline=gridded_pipeline,
        kind="quadmesh",
        x="lon",
        y="lat",
        z="air",
    )
    monkeypatch.setattr(views_base, "GRIDDED_MAX_CELLS", 4)  # fixture is 3 lats x 4 lons = 12 cells
    with pytest.raises(ValueError, match="exceeding the 4 safety cap"):
        view.get_plot(view.get_data())


def test_hvplot_gridded_skips_pivot_if_xarray_unavailable(gridded_pipeline, monkeypatch):
    """If xarray is unavailable, _to_gridded returns the df unchanged.

    Build a view with a non-gridded kind so construction doesn't trigger the
    quadmesh render path, then make xarray look unavailable and call
    _to_gridded directly.
    """
    view = hvPlotView(
        pipeline=gridded_pipeline,
        kind="scatter",
        x="lon",
        y="lat",
        z="air",
    )
    monkeypatch.setattr(views_base, "try_import_xarray", lambda: None)
    result = view._to_gridded(view.get_data())
    assert isinstance(result, pd.DataFrame), "expected df fallback when xarray unavailable"


def test_hvplot_gridded_passes_through_non_dataframe(gridded_pipeline):
    """If the pipeline hands over an already-gridded xarray object, skip pivot."""
    xr = pytest.importorskip("xarray")
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


def test_hvplot_gridded_xarray_dataset_renders(gridded_pipeline, gridded_df):
    """An xarray Dataset (not DataArray) handed to get_plot must actually render.

    The pass-through branch in _to_gridded skips pivot for any non-DataFrame;
    we need to confirm that downstream .hvplot(kind='quadmesh', z=...) on a
    Dataset succeeds, not just that the object is returned unchanged. Standard
    Lumen sources only emit DataFrames, so we bypass the pipeline and pass a
    Dataset directly into get_plot.
    """
    xr = pytest.importorskip("xarray")
    ds = (
        gridded_df.set_index(["lat", "lon"])["air"]
        .to_xarray()
        .to_dataset(name="air")
    )
    assert isinstance(ds, xr.Dataset)
    view = hvPlotView(
        pipeline=gridded_pipeline, kind="quadmesh", x="lon", y="lat", z="air",
    )
    plot = view.get_plot(ds)
    assert isinstance(plot, hv.QuadMesh)
