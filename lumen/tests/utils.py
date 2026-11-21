"""Shared test helpers for optional dependencies.

Importing these at module top would break collection where they are not
installed, so guard them once here and let tests import the names and skip via
the ``requires_*`` markers instead of repeating the guard per file.
"""
import pandas as pd
import pytest

try:
    import geopandas as gpd

    from shapely.geometry import Polygon
except ImportError:
    gpd = None
    Polygon = None

# geopandas depends on shapely, so a single geopandas check covers both.
requires_geopandas = pytest.mark.skipif(
    gpd is None, reason="geopandas is not installed"
)

try:
    import datashader
except ImportError:
    datashader = None

# Anything that builds a datashaded hvPlot needs it, including the explorer,
# whose converter asks for it as soon as the plot is constructed.
requires_datashader = pytest.mark.skipif(
    datashader is None, reason="datashader is not installed"
)


def makeMixedDataFrame():
    return pd.DataFrame({
        'A': [0.0, 1.0, 2.0, 3.0, 4.0],
        'B': [0.0, 1.0, 0.0, 1.0, 0.0],
        'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'],
        'D': pd.bdate_range('1/1/2009', periods=5),
    })
