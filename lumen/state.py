from weakref import WeakKeyDictionary

import panel as pn

from .util import is_ref


class _session_state:
    """
    The session_state object holds both global and session specific
    state making it easy to manage access to Sources and Filters
    across components.
    """

    global_sources = {}

    global_filters = {}

    spec = {}

    _apps  = WeakKeyDictionary() if pn.state.curdoc else {}

    _loading = WeakKeyDictionary() if pn.state.curdoc else {}

    _sources = WeakKeyDictionary() if pn.state.curdoc else {}

    _filters = WeakKeyDictionary() if pn.state.curdoc else {}

    _variables = WeakKeyDictionary() if pn.state.curdoc else {}

    @property
    def app(self):
        return self._apps.get(pn.state.curdoc)

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
    def variables(self):
        return self._variables.get(pn.state.curdoc, {})

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
            if filter_specs:
                schema = source.get_schema()
                self.global_filters[name] = {
                    fname: (filter_spec, schema)
                    for fname, filter_spec in filter_specs.items()
                }

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
        if filter_specs:
            schema = source.get_schema()
            for fname, filter_spec in filter_specs.items():
                if fname not in filters:
                    filters[fname] = Filter.from_spec(filter_spec, schema)
        return source

    def resolve_views(self):
        from .views import View
        exts = []
        for target in self.spec.get('targets', []):
            views = target.get('views', [])
            if isinstance(views, dict):
                views = list(views.values())
            for view in views:
                view_type = View._get_type(view.get('type'))
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
        refs = reference[1:].split('.')
        vars = variables or self.variables
        if refs[0] == 'variables':
            return vars[refs[1]]
        return self._resolve_source_ref(refs)


state = _session_state()
