import datetime as dt

from itertools import product

import param
import panel as pn

from .filters import FacetFilter
from .sources import Source
from .transforms import Transform
from .views import View


class Monitor(param.Parameterized):
    """
    A Monitor renders the results of a Source query using the defined
    set of filters and views.
    """

    application = param.Parameter(doc="The overall monitoring application.")

    current = param.List()

    height = param.Integer(default=400, doc="Height of the monitor cards.")

    filters = param.List(doc="A list of filters to be rendered.")

    sort_fields = param.List(default=[], doc="List of fields to sort by.")

    sort_reverse = param.Boolean(default=False, doc="""
        Whether to reverse the sort order.""")

    schema = param.Dict()

    title = param.String(doc="A title for this monitor.")

    layout = param.ObjectSelector(default='column', objects=['row', 'column', 'grid'])

    sizing_mode = param.ObjectSelector(default='fixed', objects=[
        'stretch_width', 'stretch_both', 'stretch_height', 'fixed'], doc="""
        Sizing mode to apply to the views.""")

    refresh_rate = param.Integer(default=None, doc="""
        How frequently to refresh the monitor by querying the adaptor.""")

    source = param.ClassSelector(class_=Source, doc="""
       The Source queries the data from some data source.""")

    tsformat = param.String(default="%m/%d/%Y %H:%M:%S")

    views = param.List(doc="A list of views to be displayed.")

    width = param.Integer(default=400, doc="Width of the monitor cards.")

    def __init__(self, **params):
        views = params.get('views', [])
        sort = params.pop('sort', {})
        params['sort_fields'] = sort_fields = sort.get('fields', [])
        params['sort_reverse'] = sort_reverse = sort.get('reverse', False)
        self._sort_widget = pn.widgets.MultiSelect(
            options=[m.get('name') for m in views],
            sizing_mode='stretch_width', value=sort_fields,
            size=len(views)
        )
        self._reverse_widget = pn.widgets.Checkbox(
            value=sort_reverse, name='Reverse', margin=(5, 0, 0, 10)
        )
        self._reload_button = pn.widgets.Button(
            name='↻', width=50, css_classes=['reload'], margin=0
        )
        self._reload_button.on_click(self.update)
        super(Monitor, self).__init__(**params)
        self._cards = []
        self._cache = {}
        self._stale = False
        self._cb = None
        self.timestamp = pn.pane.HTML(
            f'Last updated: {dt.datetime.now().strftime(self.tsformat)}',
            align='end', margin=10, sizing_mode='stretch_width'
        )
        self._update_views()
        for filt in self.filters:
            if isinstance(filt, FacetFilter):
                continue
            filt.param.watch(self._rerender, 'value')

    @pn.depends('_sort_widget.value', '_reverse_widget.value', watch=True)
    def _resort(self, *events):
        self.sort_fields = self._sort_widget.value
        self.sort_reverse = self._reverse_widget.value
        self._update_views()
        self.application._rerender()

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

    def _get_transforms(self, transform_specs):
        transforms = []
        for transform in transform_specs:
            transform = dict(transform)
            transform_type = transform.pop('type', None)
            transform = Transform._get_type(transform_type)(**transform)
            transforms.append(transform)
        return transforms

    def _get_view(self, key, view_spec, filters):
        view_type = view_spec.pop('type', None)
        transform_specs = view_spec.pop('transforms', [])
        transforms = self._get_transforms(transform_specs)
        if key not in self._cache:
            view = View._get_type(view_type)(
                source=self.source, filters=filters,
                transforms=transforms, monitor=self, **view_spec
            )
            self._cache[key] = view
        else:
            view = self._cache.get(key)
            view.update(rerender=False)
        return view

    def _get_card(self, title, views):
        if self.layout == 'row':
            item = pn.Row(*(view.panel for view in views))
        elif self.layout == 'grid':
            item = pn.GridBox(*(view.panel for view in views), ncols=2)
        else:
            item = pn.Column(*(view.panel for view in views))
        item.sizing_mode = self.sizing_mode

        return pn.Card(
            item, height=self.height, width=self.width, title=title,
            sizing_mode=self.sizing_mode
        )

    @pn.depends('current')
    def _update_views(self):
        filters = [filt for filt in self.filters
                   if not isinstance(filt, FacetFilter)]
        facets = [filt.filters for filt in self.filters
                  if isinstance(filt, FacetFilter)]
        cards = []
        for facet_filters in product(*facets):
            views = []
            view_filters = filters + list(facet_filters)
            key = tuple(str(f.value) for f in facet_filters)
            for view_spec in self.views:
                view = self._get_view(key+(id(view_spec),), dict(view_spec), view_filters)
                if view:
                    views.append(view)
            if not views:
                continue

            sort_key = []
            for field in self.sort_fields:
                values = [m.get_value() for m in views if m.name == field]
                if values:
                    sort_key.append(values[0])

            if facet_filters:
                title = ' '.join([f'{f.label}: {f.value}' for f in facet_filters])
            else:
                title = self.title

            card = self._get_card(title, views)
            cards.append((tuple(sort_key), card))

        if self.sort_fields:
            cards = sorted(cards, key=lambda x: x[0])
            if self.sort_reverse:
                cards = cards[::-1]

        self._cards[:] = [card for _, card in cards]

    def _rerender(self, *events):
        if not self._stale:
            return
        self._update_views()
        self.application._rerender()

    # Public API
        
    @property
    def filter_panel(self):
        views = []
        filters = [filt.panel for filt in self.filters if filt.panel is not None]
        if filters:
            views.append(pn.pane.Markdown('### Filters', margin=(0, 5)))
            views.extend(filters)
            views.append(pn.layout.Divider())
        views.extend([pn.pane.Markdown('### Sort', margin=(0, 5)), self._sort_widget, self._reverse_widget])
        views.append(pn.layout.Divider())
        views.append(pn.Row(self._reload_button, self.timestamp, sizing_mode='stretch_width'))
        return pn.Card(*views, title=self.title, sizing_mode='stretch_width') if views else None

    @property
    def panels(self):
        return self._cards

    def update(self, *events):
        self.source.clear_cache()
        self.timestamp.object = f'Last updated: {dt.datetime.now().strftime(self.tsformat)}'
        self._update_views()
