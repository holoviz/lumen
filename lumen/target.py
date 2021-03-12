import datetime as dt

from functools import partial
from itertools import product

import param
import panel as pn

from .filters import Filter, FacetFilter
from .sources import Source
from .util import _LAYOUTS
from .views import View


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



class Target(param.Parameterized):
    """
    A Target renders the results of a Source query using the defined
    set of filters and views.
    """

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

    title = param.String(doc="A title for this Target.")

    refresh_rate = param.Integer(default=None, doc="""
        How frequently to refresh the monitor by querying the adaptor.""")

    source = param.ClassSelector(class_=Source, doc="""
        The Source queries the data from some data source.""")

    tsformat = param.String(default="%m/%d/%Y %H:%M:%S")

    views = param.ClassSelector(class_=(list, dict), doc="""
        A list or dictionary of views to be displayed.""")

    def __init__(self, **params):
        self._application = params.pop('application', None)
        self._cards = []
        self._cache = {}
        self._cb = None
        self._stale = False
        self._updates = {}
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
            return None, None

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
        return sort_key, card

    def get_filter_panel(self, skip=None):
        skip = skip or []
        views = []
        source_panel = self.source.panel
        if source_panel:
            source_header = pn.pane.Markdown('### Source', margin=(0, 5))
            views.extend([source_header, source_panel, pn.layout.Divider()])
        filters = [filt.panel for filt in self.filters
                   if filt not in skip and filt.panel is not None]
        if filters:
            views.append(pn.pane.Markdown('### Filters', margin=(0, 5)))
            views.extend(filters)
            views.append(pn.layout.Divider())
        if self.facet.param.sort.objects:
            views.extend([
                pn.pane.Markdown('### Sort', margin=(0, 5)),
                self.facet._sort_widget,
                self.facet._reverse_widget
            ])
            views.append(pn.layout.Divider())
        self._reload_button = pn.widgets.Button(
            name='↻', width=50, css_classes=['reload'], margin=0
        )
        self._reload_button.on_click(self.update)
        self._timestamp = pn.pane.HTML(
            f'Last updated: {dt.datetime.now().strftime(self.tsformat)}',
            align='end', margin=10, sizing_mode='stretch_width'
        )
        reload_panel = pn.Row(self._reload_button, self._timestamp, sizing_mode='stretch_width')
        views.append(reload_panel)
        return pn.Column(*views, name=self.title, sizing_mode='stretch_width')

    ##################################################################
    # Rendering API
    ##################################################################

    def _update_views(self, invalidate_cache=True, update_views=True, events=[]):
        cards = []
        for facet_filters in self.facet.filters:
            key, card = self._get_card(
                self.filters, facet_filters, invalidate_cache, update_views,
                events=events
            )
            if card is None:
                continue
            cards.append((key, card))

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
                self._application._loading(self.title)
                for card, views in self._updates.items():
                    card[0][:] = [view.panel for view in views]
                self._updates = {}
            for _, (_, views) in self._cache.items():
                for view in views:
                    if view._updates:
                        view._panel.param.set_param(**view._updates)
                        view._updates = None
        if self._stale or (update_views and rerender):
            self._application._rerender()
            self._stale = False

    ##################################################################
    # Public API
    ##################################################################

    @classmethod
    def from_spec(cls, spec, sources={}, filters={}, root=None, **kwargs):
        """
        Creates a Target object from a specification. If a Target
        specification references an existing Source or Filter by name
        these may be supplied in the sources and filters dictionaries.

        Parameters
        ----------
        spec : dict
            Specification declared as a dictionary of parameter values.
        sources: dict
            Dictionary of Source objects
        filters: dict
            Dictionary of Filter objects
        root: str
            Root directory where dashboard specification was loaded from.
        kwargs: dict
            Additional kwargs to pass to the Target

        Returns
        -------
        Resolved and instantiated Target object
        """
        # Resolve source
        spec = dict(spec)
        source_spec = spec.pop('source', None)
        source = Source.from_spec(source_spec, sources, root=root)
        schema = source.get_schema()

        # Resolve filters
        filter_specs = spec.pop('filters', [])
        if isinstance(source_spec, str):
            source_filters = filters.get(source_spec)
        else:
            source_filters = None
        filters = [
            Filter.from_spec(filter_spec, schema, source_filters)
            for filter_spec in filter_specs
        ]

        # Resolve faceting
        facet_spec = spec.pop('facet', {})
        facet_filters = [f for f in filters if isinstance(f, FacetFilter)]

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
        default = 'grid' if len(self._cards) > 1 else 'column'
        layout = self.facet.layout or default
        kwargs = dict(name=self.title, sizing_mode='stretch_width')
        if isinstance(layout, dict):
            layout_kwargs = dict(layout)
            layout = layout_kwargs.pop('type')
            kwargs.update(layout_kwargs)
        layout_type = _LAYOUTS[layout]
        if layout == 'grid' and 'ncols' not in kwargs:
            kwargs['ncols'] = 3
        return layout_type(*self._cards, **kwargs)

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
