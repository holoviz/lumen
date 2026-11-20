from __future__ import annotations

from typing import TYPE_CHECKING

from .__version import __version__
from .config import config
from .dashboard import Dashboard
from .filters.base import Filter
from .layout import Layout
from .pipeline import Pipeline
from .sources.base import Source
from .state import state
from .transforms.base import Transform
from .views.base import View

if TYPE_CHECKING:
    from .sources.data import DataT

# Declared so type checkers treat these as public now that lumen ships py.typed
__all__ = [
    "Dashboard", "Filter", "Layout", "Pipeline", "Source", "Transform", "View",
    "__version__", "config", "read", "state",
]


def read(
    data: DataT | list[DataT] | dict[str, DataT], *, variables: list[str] | None = None,
    sel: dict | None = None, bbox: tuple[float, float, float, float] | None = None,
) -> Source:
    """
    Load data into a single Lumen Source.

    See :func:`lumen.sources.data.read` for the data it accepts and the
    ``variables``, ``sel`` and ``bbox`` options for remote data.
    """
    # Deferred: reading loads DuckDB, which import lumen avoids
    from .sources.data import read
    return read(data, variables=variables, sel=sel, bbox=bbox)
