import param as _param

from .config import config # noqa
from .dashboard import Dashboard # noqa

__version__ = str(_param.version.Version(
    fpath=__file__, archive_commit="$Format:%h$", reponame="lumen"))
