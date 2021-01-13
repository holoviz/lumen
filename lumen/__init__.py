import param as _param

from .dashboard import Dashboard # noqa


class _config(_param.Parameterized):
    """
    Stores shared configuration for the entire Lumen application.
    """

    sources = _param.Dict(default={}, doc="""
      A global dictionary of shared Source objects.""")

    template_vars = _param.Dict(default={}, doc="""
      Template variables which may be referenced in a dashboard yaml
      specification.""")



config = _config()


__version__ = str(_param.version.Version(
    fpath=__file__, archive_commit="$Format:%h$", reponame="lumen"))
