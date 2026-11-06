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


def read(data):
    """
    Load data into a single Lumen Source.

    See :func:`lumen.sources.data.read` for the data it accepts.
    """
    # Deferred: reading loads DuckDB, which import lumen avoids
    from .sources.data import read
    return read(data)
