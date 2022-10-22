from weakref import WeakKeyDictionary

import panel as pn

from .util import extract_refs, is_ref


class _session_state:
    """
    The session_state object holds both global and session specific
    state making it easy to manage access to Sources and Filters
    across components.
    """

    global_sources = {}

    global_filters = {}

    _specs = WeakKeyDictionary() if pn.state.curdoc else {}

    _spec = {}

    _apps = WeakKeyDictionary() if pn.state.curdoc else {}

    _loading = WeakKeyDictionary() if pn.state.curdoc else {}

    _sources = WeakKeyDictionary() if pn.state.curdoc else {}

    _pipelines = WeakKeyDictionary() if pn.state.curdoc else {}

    _filters = WeakKeyDictionary() if pn.state.curdoc else {}

    _variables = WeakKeyDictionary() if pn.state.curdoc else {}

    @property
    def app(self):
        return self._apps.get(pn.state.curdoc)

    @property
    def spec(self):
        if pn.state.curdoc is None:
            return self._spec
        if pn.state.curdoc not in self._specs:
            self.spec = {}
        spec = self._specs.get(pn.state.curdoc, {})
        return spec

    @spec.setter
    def spec(self, spec):
        if pn.state.curdoc is None:
            self._spec = spec
        else:
            self._specs[pn.state.curdoc] = spec

    def to_spec(
        self, auth=None, config=None, defaults=None,
        pipelines={}, sources={}, targets=[], variables=None
    ):
        """
        Exports the full specification of the supplied components including
        the variable definitions.

        Parameters
        ----------
        auth: lumen.dashboard.Auth
        config: lumen.dashboard.Config
        defaults: lumen.dashboard.Defaults
        variables: lumen.variables.Variables
        pipelines: Dict[str, Pipeline]
        sources: Dict[str, Source]
        targets: list[Target]

        Returns
        -------
        Declarative specification of all the supplied components.
        """
        variables = variables or self.variables
        context = {}
        if auth:
            from .dashboard import Auth
            if isinstance(auth, dict):
                context['auth'] = Auth.validate(auth)
            else:
                context['auth'] = auth.to_spec()
        if config:
            from .dashboard import Config
            if isinstance(config, dict):
                context['config'] = Config.validate(config)
            else:
                context['config'] = config.to_spec()
        if defaults:
            from .dashboard import Defaults
            if isinstance(defaults, dict):
                context['defaults'] = Defaults.validate(defaults)
            else:
                context['defaults'] = defaults.to_spec()
        if self.variables._vars:
            context['variables'] = {
                k: v.to_spec() for k, v in self.variables._vars.items()
            }
        if sources:
            context['sources'] = {}
        for k, source in sources.items():
            context['sources'][k] = source.to_spec(context=context)
        if pipelines:
            context['pipelines'] = {}
        for k, pipeline in pipelines.items():
            context['pipelines'][k] = pipeline.to_spec(context=context)
        if targets:
            context['targets'] = [
                target.to_spec(context) for target in targets
            ]
        return context

    @property
    def filters(self):
        from .filters import Filter
        if pn.state.curdoc not in self._filters:
            self._filters[pn.state.curdoc] = {
                source: {
                    name: Filter.from_spec(spec, schema)
                    for name, (spec, schema) in filters.items()
                }
                for source, filters in self.global_filters.items()
            }
        return self._filters[pn.state.curdoc]

    @property
    def sources(self):
        if pn.state.curdoc not in self._sources:
            self._sources[pn.state.curdoc] = dict(self.global_sources)
        return self._sources[pn.state.curdoc]

    @property
    def pipelines(self):
        if pn.state.curdoc not in self._pipelines:
            self._pipelines[pn.state.curdoc] = {}
        return self._pipelines[pn.state.curdoc]

    @property
    def variables(self):
        if pn.state.curdoc not in self._variables:
            from .variables import Variables
            self._variables[pn.state.curdoc] = Variables()
        return self._variables[pn.state.curdoc]

    @property
    def global_refs(self):
        target_refs = []
        source_specs = self.spec.get('sources', {})
        for target in self.spec.get('targets', []):
            if isinstance(target.get('source'), str):
                src_ref = target['source']
                target = dict(target, source=source_specs.get(src_ref))
            target_refs.append(set(extract_refs(target, 'variables')))

        var_refs = extract_refs(self.spec.get('variables'), 'variables')
        if target_refs:
            combined_refs = set.union(*target_refs)
            merged_refs = set.intersection(*target_refs)
        else:
            combined_refs, merged_refs = set(), set()
        return list(merged_refs | (set(var_refs) - combined_refs))

    @property
    def loading_msg(self):
        return self._loading.get(pn.state.curdoc)

    @loading_msg.setter
    def loading_msg(self, loading_msg):
        self._loading[pn.state.curdoc] = loading_msg

    def load_global_sources(self, clear_cache=True):
        """
        Loads global sources shared across all targets.
        """
        from .sources import Source
        for name, source_spec in self.spec.get('sources', {}).items():
            if not source_spec.get('shared'):
                continue
            source_spec = dict(source_spec)
            filter_specs = source_spec.pop('filters', {})
            source = self.global_sources.get(name)
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
            if self.loading_msg is not None:
                self.loading_msg.object = f'Loading {name} source...'
            if reinitialize:
                source_spec['name'] = name
                self.global_sources[name] = source = Source.from_spec(source_spec)
            if source.cache_dir and clear_cache:
                source.clear_cache()
            if not filter_specs:
                continue
            tables = set(filter_spec.get('table') for filter_spec in filter_specs.values())
            if None in tables:
                schemas = source.get_schema()
            else:
                schemas = {table: source.get_schema(table) for table in tables}
            self.global_filters[name] = {
                fname: (filter_spec, schemas)
                for fname, filter_spec in filter_specs.items()
            }

    def load_pipelines(self, **kwargs):
        from .pipeline import Pipeline
        self._pipelines[pn.state.curdoc or None] = pipelines = {}
        for name, pipeline_spec in self.spec.get('pipelines', {}).items():
            pipelines[name] = Pipeline.from_spec(
                dict(pipeline_spec, name=name, **kwargs)
            )
        return pipelines

    def load_source(self, name, source_spec):
        from .filters import Filter
        from .sources import Source
        source_spec = dict(source_spec)
        filter_specs = source_spec.pop('filters', None)
        if self.loading_msg:
            self.loading_msg.object = f'Loading {name} source...'
        if name in self.sources:
            source = self.sources[name]
        else:
            source_spec['name'] = name
            source = Source.from_spec(source_spec)
        self.sources[name] = source
        if not filter_specs:
            return source
        self.filters[name] = filters = dict(self.filters.get(name, {}))
        tables = set(filter_spec.get('table') for filter_spec in filter_specs.values())
        if None in tables:
            schemas = source.get_schema()
        else:
            schemas = {table: source.get_schema(table) for table in tables}
        for fname, filter_spec in filter_specs.items():
            if fname not in filters:
                filters[fname] = Filter.from_spec(filter_spec, schemas)
        return source

    def resolve_views(self):
        from .views import View
        exts = []
        for target in self.spec.get('targets', []):
            views = target.get('views', [])
            if isinstance(views, dict):
                views = list(views.values())
            for view in views:
                view_type = View._get_type(view.get('type', None))
                if view_type and view_type._extension:
                    exts.append(view_type._extension)
        for ext in exts:
            __import__(pn.extension._imports[ext])

    def _resolve_source_ref(self, refs):
        if len(refs) == 3:
            sourceref, table, field = refs
        elif len(refs) == 2:
            sourceref, table = refs
        elif len(refs) == 1:
            (sourceref,) = refs

        from .sources import Source
        source = Source.from_spec(sourceref)
        if len(refs) == 1:
            return source
        if len(refs) == 2:
            return source.get(table)
        table_schema = source.get_schema(table)
        if field not in table_schema:
            raise ValueError(f"Field '{field}' was not found in "
                             f"'{sourceref}' table '{table}'.")
        field_schema = table_schema[field]
        if 'enum' not in field_schema:
            raise ValueError(f"Field '{field}' schema does not "
                             "declare an enum.")
        return field_schema['enum']

    def resolve_reference(self, reference, variables=None):
        if not is_ref(reference):
            raise ValueError('References should be prefixed by $ symbol.')
        from .variables import Variable
        refs = reference[1:].split('.')
        vars = variables or self.variables
        if refs[0] == 'variables':
            value = vars[refs[1]]
            if isinstance(value, Variable):
                value = value.value
            return value
        return self._resolve_source_ref(refs)

    def reset(self):
        self._filters.clear()
        self._sources.clear()
        self._pipelines.clear()
        self._specs.clear()
        self._spec.clear()
        self._variables.clear()



state = _session_state()
