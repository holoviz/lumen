import datetime as dt

from functools import partial
from io import StringIO, BytesIO
from itertools import product

import param
import panel as pn

from .config import _LAYOUTS
from .filters import Filter, FacetFilter, ParamFilter
from .sources import Source
from .state import state
from .views import View


DOWNLOAD_FORMATS = ['csv', 'xlsx', 'json', 'parquet']


class Facet(param.Parameterized):

    by = param.List(default=[], class_=FacetFilter, doc="""
        Fields to facet by.""")

    layout = param.ClassSelector(default=None, class_=(str, dict), doc="""
        How to lay out the facets.""")

    sort = param.ListSelector(default=[], objects=[], doc="""
        List of fields to sort by.""")

    reverse = param.Boolean(default=False, doc="""
        Whether to reverse the sort order.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._sort_widget = pn.widgets.MultiSelect(
            options=self.param.sort.objects,
            sizing_mode='stretch_width',
            size=len(self.sort),
            value=self.sort,
        )
        self._sort_widget.link(self, value='sort')
        self._reverse_widget = pn.widgets.Checkbox(
            value=self.reverse, name='Reverse', margin=(5, 0, 0, 10)
        )
        self._reverse_widget.link(self, value='reverse')

    @param.depends('sort:objects', watch=True)
    def _update_options(self):
        self._sort_widget.options = self.param.sort.objects

    @classmethod
    def from_spec(cls, spec, schema):
        """
        Creates a Facet object from a schema and a set of fields.
        """
        by = []
        for by_spec in spec.pop('by', []):
            if isinstance(by_spec, str):
                f = Filter.from_spec({'type': 'facet', 'field': by_spec}, schema)
            elif isinstance(by_spec, dict):
                f = Filter.from_spec(dict(by_spec, type='facet'), schema)
            elif isinstance(by_spec, Filter):
                f = by_spec
            by.append(f)
        sort = spec.pop('sort', [b.field for b in by])
        sorter = cls(by=by, **spec)
        sorter.param.sort.objects = sort
        return sorter

    def get_sort_key(self, views):
        sort_key = []
        for field in self.sort:
            values = [v.get_value(field) for v in views]
            if values:
                sort_key.append(values[0])
        return tuple(sort_key)

    @property
    def filters(self):
        return product(*[filt.filters for filt in self.by])



class Download(pn.viewable.Viewer):
    """
    The Download object controls options to download the Source tables
    in a variety of formats via the Dashboard UI.
    """

    labels = param.Dict(default={}, doc="""
        A dictionary to override the Button label for each table.""")

    filters = param.List(class_=Filter, doc="""
        A list of filters to be rendered.""")

    format = param.ObjectSelector(default=None, objects=DOWNLOAD_FORMATS, doc="""
        The format to download the data in.""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments passed to the serialization function, e.g. 
        data.to_csv(file_obj, **kwargs).""")

    source = param.ClassSelector(class_=Source, doc="""
        The Source queries the data from some data source.""")

    tables = param.List(default=[], doc="""
        The list of tables to allow downloading.""")

    def __init__(self, **params):
        super().__init__(**params)
        default_table = self.tables[0] if self.tables else None
        button_params = {'filename': f'{default_table}.{self.format}'}
        if default_table in self.labels:
            button_params['label'] = self.labels[default_table]
        self._select_download = pn.widgets.Select(
            name='Select table to download', options=self.tables, value=default_table,
            sizing_mode='stretch_width'
        )
        self._select_download.param.watch(self._update_button, 'value')
        self._download_button = pn.widgets.FileDownload(
            callback=self._table_data, align='end',
            sizing_mode='stretch_width', **button_params
        )
        self._layout = pn.Column(
            pn.pane.Markdown('### Download tables', margin=(0, 5, -10, 5)),
            self._download_button,
            pn.layout.Divider(),
            sizing_mode='stretch_width'
        )
        if len(self.tables) > 1:
            self._layout.insert(1, self._select_download)

    def __panel__(self):
        return self._layout

    def _update_button(self, event):
        params = {'filename': f'{event.new}.{self.format}'}
        if event.new in self.labels:
            params['label'] = self.labels[event.new]
        self._download_button.param.set_param(**params)

    def __bool__(self):
        return self.format is not None

    def _table_data(self):
        if self.format in ('json', 'csv'):
            io = StringIO()
        else:
            io = BytesIO()
        table = self._select_download.value
        query = {
            filt.field: filt.query for filt in self.filters
            if filt.query is not None and
            (filt.table is None or filt.table == table)
        }
        data = self.source.get(table, **query)
        for filt in self.filters:
            if not isinstance(filt, ParamFilter):
                continue
            from holoviews import Dataset
            if filt.value is not None:
                ds = Dataset(data)
                data = ds.select(filt.value).data
        if self.format == 'csv':
            data.to_csv(io, **self.kwargs)
        elif self.format == 'json':
            data.to_json(io, **self.kwargs)
        elif self.format == 'xlsx':
            data.to_excel(io, **self.kwargs)
        elif self.format == 'parquet':
            data.to_parquet(io, **self.kwargs)
        io.seek(0)
        return io

    @classmethod
    def from_spec(cls, spec, source, filters):
        """
        Creates a Download object from a specification.

        Parameters
        ----------
        spec : dict
            Specification declared as a dictionary of parameter values.
        source: Source
            The Source to monitor
        filters: list
            List of Filter objects

        """
        return cls(source=source, filters=filters, **spec)



class Target(param.Parameterized):
    """
    A Target renders the results of a Source query using the defined
    set of filters and views.
    """

    download = param.ClassSelector(class_=Download, default=Download(), doc="""
        The download objects determines whether and how the source tables
        can be downloaded.""")

    facet = param.ClassSelector(class_=Facet, doc="""
        The facet object determines whether and how to facet the cards
        on the target.""")

    filters = param.List(class_=Filter, doc="""
        A list of filters to be rendered.""")

    layout = param.ClassSelector(default='column', class_=(str, list, dict), doc="""
        Defines the layout of the views in the monitor target. Can be
        'column', 'row', 'grid', 'row' or a nested list of indexes
        corresponding to the views, e.g. [[0, 1], [2]] will create
        a Column of one row containing views 0 and 1 and a second Row
        containing view 2.""" )

    reloadable = param.Boolean(default=True, doc="""
        Whether to allow reloading data target's source using a button.""")

    title = param.String(doc="A title for this Target.")

    refresh_rate = param.Integer(default=None, doc="""
        How frequently to refresh the monitor by querying the adaptor.""")

    source = param.ClassSelector(class_=Source, doc="""
        The Source queries the data from some data source.""")

    tsformat = param.String(default="%m/%d/%Y %H:%M:%S")

    views = param.ClassSelector(class_=(list, dict), doc="""
        A list or dictionary of views to be displayed.""")

    def __init__(self, **params):
        if 'facet' not in params:
            params['facet'] = Facet()
        self._application = params.pop('application', None)
        self._cards = []
        self._cache = {}
        self._cb = None
        self._stale = False
        self._updates = {}
        self._view_controls = pn.Column(sizing_mode='stretch_width')
        self.kwargs = {k: v for k, v in params.items() if k not in self.param}
        super().__init__(**{k: v for k, v in params.items() if k in self.param})

        # Set up watchers
        self.facet.param.watch(self._resort, ['sort', 'reverse'])
        for filt in self.filters:
            if isinstance(filt, FacetFilter):
                continue
            filt.param.watch(partial(self._rerender, invalidate_cache=True), 'value')
        self._update_views()

    def _resort(self, *events):
        self._rerender(update_views=False)

    ##################################################################
    # Create UI
    ##################################################################

    def _construct_card(self, title, views):
        kwargs = dict(self.kwargs)
        layout = self.layout
        if isinstance(layout, list):
            item = pn.Column(sizing_mode='stretch_both')
            for row_spec in layout:
                row = pn.Row(sizing_mode='stretch_width')
                for index in row_spec:
                    if isinstance(index, int):
                        view = views[index].panel
                    else:
                        matches = [view for view in views if view.name == index]
                        if matches:
                            view = matches[0].panel
                        else:
                            raise KeyError("Target could not find named "
                                           f"view '{index}'.")
                    row.append(view)
                item.append(row)
        else:
            if isinstance(layout, dict):
                layout_kwargs = dict(layout)
                layout = layout_kwargs.pop('type')
                kwargs.update(layout_kwargs)
            if layout == 'grid' and 'ncols' not in kwargs:
                kwargs['ncols'] = 2
            layout_type = _LAYOUTS[layout]
            item = layout_type(*(view.panel for view in views), **kwargs)
        params = {k: v for k, v in self.kwargs.items() if k in pn.Card.param}
        return pn.Card(item, title=title, name=title, **params)

    def _materialize_views(self, filters):
        views = []
        for view in self.views:
            if isinstance(self.views, dict):
                view_spec = dict(self.views[view], name=view)
            else:
                view_spec = view
            view = View.from_spec(view_spec, self.source, filters)
            views.append(view)
        return views

    def _get_card(self, filters, facet_filters, invalidate_cache=True, update_views=True, events=[]):
        # Get cache key
        if isinstance(self.views, list):
            view_specs = self.views
        else:
            view_specs = list(self.views.values())
        key = (tuple(str(f.value) for f in facet_filters) +
               tuple(id(view) for view in view_specs))

        # Get views
        update_card = False
        if key in self._cache:
            card, views = self._cache[key]
        else:
            view_filters = filters + list(facet_filters)
            card, views = None, self._materialize_views(view_filters)

        # Update views
        if update_views:
            for view in views:
                view_stale = view.update(*events, invalidate_cache=invalidate_cache)
                update_card = update_card or view_stale

        if not any(view for view in views):
            return None, None, views

        sort_key = self.facet.get_sort_key(views)

        if facet_filters:
            title = ' '.join([f'{f.label}: {f.value}' for f in facet_filters])
        else:
            title = self.title

        if card is None:
            card = self._construct_card(title, views)
            self._cache[key] = (card, views)
        else:
            card.title = title
            if update_card:
                self._updates[card] = views
        return sort_key, card, views

    def get_filter_panel(self, skip=None):
        skip = skip or []
        views = []
        source_panel = self.source.panel
        if source_panel:
            source_header = pn.pane.Markdown('### Source', margin=(0, 5, -10, 5))
            views.extend([source_header, source_panel, pn.layout.Divider()])
        if self.download:
            views.append(self.download)
        filters = [filt.panel for filt in self.filters
                   if filt not in skip and filt.panel is not None]
        if filters:
            views.append(pn.pane.Markdown('### Filters', margin=(0, 5, -10, 5)))
            views.extend(filters)
        if self.facet.param.sort.objects:
            views.append(pn.layout.Divider(margin=0, height=5))
            views.extend([
                pn.pane.Markdown('### Sort', margin=(0, 5, -10, 5)),
                self.facet._sort_widget,
                self.facet._reverse_widget
            ])
        if self._view_controls:
            views.append(self._view_controls)
        if self.reloadable:
            if views:
                views.append(pn.layout.Divider(margin=0, height=5))
            self._reload_button = pn.widgets.Button(
                name='\u21bb', width=50, height=40, css_classes=['reload', 'tab'],
                margin=0
            )
            self._reload_button.on_click(self.update)
            self._timestamp = pn.pane.HTML(
                f'Last updated: {dt.datetime.now().strftime(self.tsformat)}',
                align='center', margin=(10, 0), sizing_mode='stretch_width'
            )
            reload_panel = pn.Row(
                self._reload_button, self._timestamp, sizing_mode='stretch_width'
            )
            views.append(reload_panel)
        return pn.Column(*views, name=self.title, sizing_mode='stretch_width')

    ##################################################################
    # Rendering API
    ##################################################################

    def _sync_view(self, view, *events):
        view.param.set_param(**{event.name: event.new for event in events})

    def _update_views(self, invalidate_cache=True, update_views=True, events=[]):
        cards = []
        view_controls = []
        prev_views = None
        for facet_filters in self.facet.filters:
            key, card, views = self._get_card(
                self.filters, facet_filters, invalidate_cache, update_views,
                events=events
            )
            if prev_views:
                for v1, v2 in zip(prev_views, views):
                    v1.param.watch(partial(self._sync_view, v2), v1.controls)
            else:
                for view in views:
                    if not view.controls:
                        continue
                    view_controls.append(view.control_panel)
                    cb = partial(self._rerender, invalidate_cache=False)
                    view.param.watch(cb, view.controls)
            prev_views = views
            if card is None:
                continue
            cards.append((key, card))

        self._view_controls[:] = view_controls

        if self.facet.sort:
            cards = sorted(cards, key=lambda x: x[0])
            if self.facet.reverse:
                cards = cards[::-1]

        cards = [card for _, card in cards]
        if cards != self._cards:
            self._cards[:] = cards
            self._stale = True
        else:
            self._stale = False

    def _rerender(self, *events, invalidate_cache=False, update_views=True):
        self._update_views(invalidate_cache, update_views, events=events)
        rerender = bool(self._updates)
        has_updates = (
            any(view._updates for _, (_, views) in self._cache.items() for view in views)
        )
        if update_views and (has_updates or rerender):
            if self._updates:
                self._application._set_loading(self.title)
                for card, views in self._updates.items():
                    card[0][:] = [view.panel for view in views]
                self._updates = {}
            for _, (_, views) in self._cache.items():
                for view in views:
                    if view._updates:
                        view._panel.param.set_param(**view._updates)
                        view._updates = None
        if self._stale or (update_views and rerender):
            self._application._render()
            self._stale = False

    ##################################################################
    # Public API
    ##################################################################

    @classmethod
    def from_spec(cls, spec, **kwargs):
        """
        Creates a Target object from a specification. If a Target
        specification references an existing Source or Filter by name.

        Parameters
        ----------
        spec : dict
            Specification declared as a dictionary of parameter values.
        kwargs: dict
            Additional kwargs to pass to the Target

        Returns
        -------
        Resolved and instantiated Target object
        """
        # Resolve source
        spec = dict(spec)
        source_spec = spec.pop('source', None)
        source = Source.from_spec(source_spec)
        tables = source.get_tables()

        # Resolve filters
        filter_specs = spec.pop('filters', [])
        if isinstance(source_spec, str):
            source_filters = state.filters.get(source_spec)
        else:
            source_filters = None

        if filter_specs or 'facet' in spec:
            schema = source.get_schema()
            filters = [
                Filter.from_spec(filter_spec, schema, source_filters)
                for filter_spec in filter_specs
            ]

            # Resolve faceting
            facet_spec = spec.pop('facet', {})
            facet_filters = [f for f in filters if isinstance(f, FacetFilter)]
        else:
            # Avoid computing schema unless necessary
            facet_spec, schema = {}, {}
            filters, facet_filters = [], []

        # Resolve download options
        download_spec = spec.pop('download', {})
        if isinstance(download_spec, str):
            download_spec = {'format': download_spec}
        if 'tables' not in download_spec:
            download_spec['tables'] = list(tables)
        spec['download'] = Download.from_spec(download_spec, source, filters) 

        # Backward compatibility
        if facet_spec:
            if facet_filters:
                param.main.warning(
                    "Cannot declare FacetFilters and provide a facet spec. "
                    "Declare the facet.by key of the target instead."
                )
            elif 'sort' in spec:
                param.main.warning(
                    "Cannot declare sort spec and provide a facet spec. "
                    "Declare the sort fields on the facet.sort key of "
                    "the target instead."
                )
        elif 'sort' in spec:
            param.main.param.warning(
                'Specifying a sort key for a target is deprecated. Use '
                'facet key instead and declare facet filters under the '
                '"by" keyword.'
            )
            sort_spec = spec['sort']
            spec['facet'] = dict(
                sort=sort_spec['fields'],
                reverse=sort_spec.get('reverse', False),
                by=facet_filters
            )
        if 'config' in spec:
            param.main.param.warning(
                "Passing config to a target is deprecated use the "
                "facet.layout key on the target instead."
            )
            facet_spec.update(spec.pop('config'))
        if 'facet_layout' in spec:
            param.main.param.warning(
                "Passing facet_layout to a target is deprecated use "
                "the facet.layout key on the target instead."
            )
            facet_spec['layout'] = spec.pop('facet_layout')

        spec['facet'] = Facet.from_spec(facet_spec, schema)
        params = dict(kwargs, **spec)
        return cls(filters=filters, source=source, **params)

    @property
    def panels(self):
        """
        Returns a layout of the rendered View objects on this target.
        """
        if len(self._cards) > 1:
            default = 'flex' if 'flex' in _LAYOUTS else 'grid'
        else:
            default = 'column'
        layout = self.facet.layout or default
        kwargs = dict(name=self.title, sizing_mode='stretch_width')
        if isinstance(layout, dict):
            layout_kwargs = dict(layout)
            layout = layout_kwargs.pop('type')
            kwargs.update(layout_kwargs)
        layout_type = _LAYOUTS[layout]
        if layout == 'grid' and 'ncols' not in kwargs:
            kwargs['ncols'] = 3
        if self._cards:
            content = self._cards
        else:
            content = (
                pn.pane.Alert(
                    '**All views are empty since no data is available. '
                    'Try relaxing the filter constraints.**',
                    sizing_mode='stretch_width'
                ),
            )
        return layout_type(*content, **kwargs)

    @pn.depends('refresh_rate', watch=True)
    def start(self, event=None):
        """
        Starts any periodic callback instantiated on this object.
        """
        refresh_rate = self.refresh_rate if event is None else event.new
        if refresh_rate is None:
            return
        if self._cb:
            self._cb.period = refresh_rate
        else:
            self._cb = pn.state.add_periodic_callback(
                self.update, refresh_rate
            )

    def update(self, *events):
        """
        Updates the views on this target by clearing any caches and
        rerendering the views on this Target.
        """
        self.source.clear_cache()
        self._timestamp.object = f'Last updated: {dt.datetime.now().strftime(self.tsformat)}'
        self._rerender(invalidate_cache=True)
