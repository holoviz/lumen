import datetime as dt

from functools import partial
from itertools import product

import param
import panel as pn

from .filters import FacetFilter
from .sources import Source
from .util import LAYOUTS
from .views import View


class Monitor(param.Parameterized):
    """
    A Monitor renders the results of a Source query using the defined
    set of filters and views.
    """

    application = param.Parameter(doc="The overall monitoring application.")

    filters = param.List(doc="A list of filters to be rendered.")

    facet_layout = param.ClassSelector(default=None, class_=(str, dict), doc="""
        If a FacetFilter is specified this declares how to lay out
        the cards.""")

    layout = param.ClassSelector(default='column', class_=(str, list, dict), doc="""
        Defines the layout of the views in the monitor target. Can be
        'column', 'row', 'grid', 'row' or a nested list of indexes
        corresponding to the views, e.g. [[0, 1], [2]] will create
        a Column of one row containing views 0 and 1 and a second Row
        containing view 2.""" )

    sort_fields = param.List(default=[], doc="List of fields to sort by.")

    sort_reverse = param.Boolean(default=False, doc="""
        Whether to reverse the sort order.""")

    schema = param.Dict(doc="Schema for this Monitor target.")

    title = param.String(doc="A title for this monitor.")

    refresh_rate = param.Integer(default=None, doc="""
        How frequently to refresh the monitor by querying the adaptor.""")

    source = param.ClassSelector(class_=Source, doc="""
        The Source queries the data from some data source.""")

    tsformat = param.String(default="%m/%d/%Y %H:%M:%S")

    views = param.List(doc="A list of views to be displayed.")

    def __init__(self, **params):
        self._config = params.pop('config', {})
        if self._config:
            self.param.warning("Passing config to targets is deprecated, "
                               "use facet_layout key to control the layout.")
        sort = params.pop('sort', {})
        params['sort_fields'] = sort_fields = sort.get('fields', [])
        params['sort_reverse'] = sort_reverse = sort.get('reverse', False)
        self.kwargs = {k: v for k, v in params.items() if k not in self.param}
        super().__init__(**{k: v for k, v in params.items() if k in self.param})
        self._cards = []
        self._cache = {}
        self._cb = None
        self._stale = False
        self._updates = {}
        
        # Build UI components
        self._sort_widget = pn.widgets.MultiSelect(
            options=sort_fields,
            sizing_mode='stretch_width',
            size=len(sort_fields),
            value=sort_fields,
        )
        self._sort_widget.param.watch(self._resort, 'value')
        self._reverse_widget = pn.widgets.Checkbox(
            value=sort_reverse, name='Reverse', margin=(5, 0, 0, 10)
        )
        self._reverse_widget.param.watch(self._resort, 'value')
        self._reload_button = pn.widgets.Button(
            name='↻', width=50, css_classes=['reload'], margin=0
        )
        self._reload_button.on_click(self.update)
        self.timestamp = pn.pane.HTML(
            f'Last updated: {dt.datetime.now().strftime(self.tsformat)}',
            align='end', margin=10, sizing_mode='stretch_width'
        )
        for filt in self.filters:
            if isinstance(filt, FacetFilter):
                continue
            filt.param.watch(partial(self._rerender, invalidate_cache=True), 'value')
        self._update_views()

    def _resort(self, *events):
        self.sort_fields = self._sort_widget.value
        self.sort_reverse = self._reverse_widget.value
        self._rerender(update_views=False)

    def _get_sort_key(self, views):
        sort_key = []
        for field in self.sort_fields:
            values = [v.get_value(field) for v in views]
            if values:
                sort_key.append(values[0])
        return tuple(sort_key)

    def _construct_card(self, title, views):
        kwargs = dict(self.kwargs)
        layout = self.layout
        if isinstance(layout, list):
            item = pn.Column(sizing_mode='stretch_both')
            for row_spec in layout:
                row = pn.Row(sizing_mode='stretch_both')
                for index in row_spec:
                    row.append(views[index].panel)
                item.append(row)
        else:
            if isinstance(layout, dict):
                layout_kwargs = dict(layout)
                layout = layout_kwargs.pop('type')
                kwargs.update(layout_kwargs)
            if layout == 'grid' and 'ncols' not in kwargs:
                kwargs['ncols'] = 2
            layout_type = LAYOUTS[layout]
            item = layout_type(*(view.panel for view in views), **kwargs)
        params = {k: v for k, v in self.kwargs.items() if k in pn.Card.param}
        return pn.Card(item, title=title, name=title, **params)

    def _get_card(self, filters, facet_filters, invalidate_cache=True, update_views=True):
        view_filters = filters + list(facet_filters)
        key = (tuple(str(f.value) for f in facet_filters) +
               tuple(id(view) for view in self.views))

        update_card = False
        if key in self._cache:
            card, views = self._cache[key]
            if update_views:
                for view in views:
                    view_stale = view.update(invalidate_cache)
                    update_card = update_card or view_stale
        else:
            card = None
            views = [
                View.from_spec(view_spec, self, self.source, view_filters)
                for view_spec in self.views
            ]
        if not any(view for view in views):
            return None, None

        sort_key = self._get_sort_key(views)

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

    def _update_views(self, invalidate_cache=True, update_views=True):
        filters = [filt for filt in self.filters
                   if not isinstance(filt, FacetFilter)]
        facets = [filt.filters for filt in self.filters
                  if isinstance(filt, FacetFilter)]

        cards = []
        for facet_filters in product(*facets):
            key, card = self._get_card(
                filters, facet_filters, invalidate_cache, update_views
            )
            if card is None:
                continue
            cards.append((key, card))

        if self.sort_fields:
            cards = sorted(cards, key=lambda x: x[0])
            if self.sort_reverse:
                cards = cards[::-1]

        cards = [card for _, card in cards]
        if cards != self._cards:
            self._cards[:] = cards
            self._stale = True

    def _rerender(self, *events, invalidate_cache=False, update_views=True):
        self._update_views(invalidate_cache, update_views)
        has_updates = (
            any(view._updates for _, (_, views) in self._cache.items() for view in views) or
            bool(self._updates)
        )
        if update_views and has_updates:
            self.application._loading(self.title)
            for card, views in self._updates.items():
                card[0][:] = [view.panel for view in views]
            self._updates = {}
            for _, (_, views) in self._cache.items():
                for view in views:
                    if view._updates:
                        view._panel.param.set_param(**view._updates)
                        view._updates = None
        if self._stale or (update_views and has_updates):
            self.application._rerender()
            self._stale = False

    # Public API

    @property
    def filter_panel(self):
        views = []
        source_panel = self.source.panel
        if source_panel:
            source_header = pn.pane.Markdown('### Source', margin=(0, 5))
            views.extend([source_header, source_panel, pn.layout.Divider()])
        filters = [filt.panel for filt in self.filters if filt.panel is not None]
        if filters:
            views.append(pn.pane.Markdown('### Filters', margin=(0, 5)))
            views.extend(filters)
            views.append(pn.layout.Divider())
        if self.sort_fields:
            views.extend([
                pn.pane.Markdown('### Sort', margin=(0, 5)),
                self._sort_widget,
                self._reverse_widget
            ])
            views.append(pn.layout.Divider())
        views.append(pn.Row(self._reload_button, self.timestamp, sizing_mode='stretch_width'))
        return pn.Column(*views, name=self.title, sizing_mode='stretch_width') if views else None

    @property
    def panels(self):
        default = 'grid' if len(self._cards) > 1 else 'column'
        layout = self.facet_layout or self._config.get('layout', default)
        kwargs = dict(name=self.title, sizing_mode='stretch_width')
        if isinstance(layout, dict):
            layout_kwargs = dict(layout)
            layout = layout_kwargs.pop('type')
            kwargs.update(layout_kwargs)
        layout_type = LAYOUTS[layout]
        if layout == 'grid':
            kwargs['ncols'] = self._config.get('ncols', 3)
        return layout_type(*self._cards, **kwargs)

    @pn.depends('refresh_rate', watch=True)
    def start(self, event=None):
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
        self.source.clear_cache()
        self.timestamp.object = f'Last updated: {dt.datetime.now().strftime(self.tsformat)}'
        self._rerender(invalidate_cache=True)
