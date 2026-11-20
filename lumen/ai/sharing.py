"""Moved to :mod:`lumen.data.sharing`; this alias keeps existing imports and patches working."""
import sys

from typing import TYPE_CHECKING

from ..data import sharing as _sharing

if TYPE_CHECKING:
    from ..data.sharing import *  # noqa: F403
    from ..data.sharing import _env_mask_pii, _env_share  # noqa: F401

sys.modules[__name__] = _sharing
