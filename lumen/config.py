import param as param

from .filters import Filter
from .sources import Source


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

    def load_global_sources(self, sources, root, clear_cache=True):
        """
        Loads global sources shared across all targets.
        """
        for name, source_spec in sources.items():
            if not source_spec.get('shared'):
                continue
            source_spec = dict(source_spec)
            filter_specs = source_spec.pop('filters', {})
            source = self.sources.get(name)
            source_type = Source._get_type(source_spec['type'])
            if source is not None:
                # Check if old source is the same as the new source
                params = {
                    k: v for k, v in source.param.get_param_values()
                    if k != 'name'
                }
                new_spec = dict(source_spec, **{
                    k: v for k, v in source_type.param.get_param_values()
                    if k != 'name'
                })
                reinitialize = params == new_spec
            else:
                reinitialize = True
            if reinitialize:
                source_spec['name'] = name
                self.sources[name] = source = Source.from_spec(
                    source_spec, self.sources, root=root
                )
            if source.cache_dir and clear_cache:
                source.clear_cache()
            schema = source.get_schema()
            self.filters[name] = {
                fname: Filter.from_spec(filter_spec, schema)
                for fname, filter_spec in filter_specs.items()
            }



config = _config()
