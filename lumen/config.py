import param as param


class _config(param.Parameterized):
    """
    Stores shared configuration for the entire Lumen application.
    """

    filters = param.Dict(default={}, doc="""
      A global dictionary of shared Filter objects.""")

    sources = param.Dict(default={}, doc="""
      A global dictionary of shared Source objects.""")

    template_vars = param.Dict(default={}, doc="""
      Template variables which may be referenced in a dashboard yaml
      specification.""")

    yamls = param.List(default=[], doc="""
      List of yaml files currently being served.""")


config = _config()
