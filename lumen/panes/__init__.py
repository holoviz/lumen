"""
Panel components Lumen renders its views with.

These are written as standalone Panel components rather than Lumen-specific
widgets so they can be contributed upstream; see `mosaic.Mosaic`.
"""
from .mosaic import Mosaic

__all__ = ["Mosaic"]
