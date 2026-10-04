"""Shared test helpers for optional dependencies.

Importing these at module top would break collection where they are not
installed, so guard them once here and let tests import the names and skip via
the ``requires_*`` markers instead of repeating the guard per file.
"""
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
