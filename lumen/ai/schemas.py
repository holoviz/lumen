"""Moved to :mod:`lumen.data.catalog`; this alias keeps existing imports and patches working."""
import sys

from typing import TYPE_CHECKING

from ..data import catalog as _catalog

if TYPE_CHECKING:
    from ..data.catalog import *  # noqa: F403

sys.modules[__name__] = _catalog
