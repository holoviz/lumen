from .__version import __version__  # noqa
from .config import config  # noqa
from .dashboard import Dashboard  # noqa
from .filters.base import Filter  # noqa
from .layout import Layout  # noqa
from .pipeline import Pipeline  # noqa
from .sources.base import Source  # noqa
from .state import state  # noqa
from .transforms.base import Transform  # noqa
from .views.base import View  # noqa


def __getattr__(name):
    # lumen.ai pulls in the Panel UI stack, so only import it on use
    if name == "read":
        from .ai.data import read
        return read
    raise AttributeError(f"module 'lumen' has no attribute {name!r}")
