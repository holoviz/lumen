"""Moved to :mod:`lumen.data.stats`; this alias keeps existing imports and patches working."""
import sys

from typing import TYPE_CHECKING

from ..data import stats as _stats

if TYPE_CHECKING:
    from ..data.stats import *  # noqa: F403

sys.modules[__name__] = _stats
