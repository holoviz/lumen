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
    from .ai.data import DataT

# Declared so type checkers treat these as public now that lumen ships py.typed
__all__ = [
    "Dashboard", "Filter", "Layout", "Pipeline", "Source", "Transform", "View",
    "__version__", "config", "read", "state",
]


def read(data: DataT | list[DataT] | dict[str, DataT], **options) -> Source:
    """
    Load data into a single Lumen Source.

    See :func:`lumen.ai.data.read` for the data it accepts and the
    ``variables``, ``sel`` and ``bbox`` options for remote data.
    """
    # Deferred: lumen.ai pulls in the Panel UI stack, which import lumen avoids
    from .ai.data import read
    return read(data, **options)
