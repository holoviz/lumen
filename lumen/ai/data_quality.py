"""Moved to :mod:`lumen.data.quality`; this alias keeps existing imports and patches working."""
import sys

from typing import TYPE_CHECKING

from ..data import quality as _quality

if TYPE_CHECKING:
    from ..data.quality import *  # noqa: F403

sys.modules[__name__] = _quality
