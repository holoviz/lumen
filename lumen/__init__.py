import param as _param

from .dashboard import Dashboard # noqa


class _config(_param.Parameterized):

    template_vars = _param.Dict(default={})


config = _config()


__version__ = str(_param.version.Version(
    fpath=__file__, archive_commit="$Format:%h$", reponame="lumen"))
