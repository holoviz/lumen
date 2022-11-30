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

    _configs = WeakKeyDictionary()
    _config = None

    _specs = WeakKeyDictionary()
    _spec = {}

    _loadings = WeakKeyDictionary()
    _loading = None

    _sources = WeakKeyDictionary()
    _source = {}

    _pipelines = WeakKeyDictionary()
    _pipeline = {}

    _filters = WeakKeyDictionary()
    _filter = {}

    _variables = WeakKeyDictionary()
    _variable = None

    @property
    def config(self):
        if pn.state.curdoc is None:
            return self._config
        return self._configs.get(pn.state.curdoc, self._config)

    @config.setter
    def config(self, config):
        if pn.state.curdoc is None:
            self._config = config
        else:
            self._configs[pn.state.curdoc] = config

    @property
    def filters(self):
        if pn.state.curdoc is None:
            if self._filter is None:
                self._filter = self._global_filters
            return self._filter
        elif pn.state.curdoc not in self._filters:
            self._filters[pn.state.curdoc] = self._global_filters
        return self._filters[pn.state.curdoc]

    @property
    def loading_msg(self):
        if pn.state.curdoc is None:
            return self._loading
        return self._loadings.get(pn.state.curdoc)

    @loading_msg.setter
    def loading_msg(self, loading_msg):
        if pn.state.curdoc is None:
            self._loading = loading_msg
        else:
            self._loadings[pn.state.curdoc] = loading_msg

    @property
    def pipelines(self):
        if pn.state.curdoc is None:
            return self._pipeline
        elif pn.state.curdoc not in self._pipelines:
            self._pipelines[pn.state.curdoc] = dict(self._pipeline)
        return self._pipelines[pn.state.curdoc]

    @property
    def sources(self):
        if pn.state.curdoc is None:
            return self._source
        elif pn.state.curdoc not in self._sources:
            self._sources[pn.state.curdoc] = dict(
                self.global_sources, **self._source
            )
        return self._sources[pn.state.curdoc]

    @property
    def spec(self):
        if pn.state.curdoc is None:
            return self._spec
        return self._specs.get(pn.state.curdoc, self._spec)

    @spec.setter
    def spec(self, spec):
        if pn.state.curdoc is None:
            self._spec = spec
        else:
            self._specs[pn.state.curdoc] = spec

    @property
    def variables(self):
        from .variables import Variables
        if self._variable is None:
            self._variable = Variables()
        if pn.state.curdoc is None:
            return self._variable
        elif pn.state.curdoc not in self._variables:
            self._variables[pn.state.curdoc] = variables = Variables()
            for var in self._variable._vars.values():
                variables.add_variable(var)
        return self._variables[pn.state.curdoc]

    @property
    def _global_filters(self):
        from .filters import Filter
        return {
            source: {
                name: Filter.from_spec(spec, schema)
                for name, (spec, schema) in filters.items()
            }
            for source, filters in self.global_filters.items()
        }

    def to_spec(
        self, auth=None, config=None, defaults=None,
        pipelines={}, sources={}, layouts=[], variables=None
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
        layouts: list[Layout]

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
        if layouts:
            context['layouts'] = [
                layout.to_spec(context) for layout in layouts
            ]
        return context

    @property
    def global_refs(self):
        layout_refs = []
        source_specs = self.spec.get('sources', {})
        for layout in self.spec.get('layouts', []):
            if isinstance(layout.get('source'), str):
                src_ref = layout['source']
                layout = dict(layout, source=source_specs.get(src_ref))
            layout_refs.append(set(extract_refs(layout, 'variables')))

        var_refs = extract_refs(self.spec.get('variables'), 'variables')
        if layout_refs:
            combined_refs = set.union(*layout_refs)
            merged_refs = set.intersection(*layout_refs)
        else:
            combined_refs, merged_refs = set(), set()
        return list(merged_refs | (set(var_refs) - combined_refs))

    def load_global_sources(self, clear_cache=True):
        """
        Loads global sources shared across all layouts.
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
        pipelines = self.pipelines
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
        for layout in self.spec.get('layouts', []):
            views = layout.get('views', [])
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
