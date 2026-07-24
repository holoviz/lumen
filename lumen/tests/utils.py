"""Shared test helpers for optional geospatial dependencies.

Importing geopandas at module top would break collection where it is not
installed, so guard it once here and let tests import ``gpd``/``Polygon`` and
skip via ``requires_geopandas`` instead of repeating the guard per file.
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
