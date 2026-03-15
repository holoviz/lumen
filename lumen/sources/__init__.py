from importlib import import_module
from importlib.util import find_spec

from .base import *
from .base import __all__ as _base_all

__all__ = list(_base_all)

if find_spec("xarray") is not None:
    __all__.append("XarraySource")


def __getattr__(name):
    if name == "XarraySource" and find_spec("xarray") is not None:
        return import_module(f"{__name__}.xarray").XarraySource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
