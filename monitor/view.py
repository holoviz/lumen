import datetime as dt

from itertools import product

import param
import panel as pn

from .adaptor import QueryAdaptor
from .filter import FacetFilter
from .metric import MetricView


class View(param.Parameterized):
    """
    A View renders the results of a monitoring query using the defined
    set of filters and metrics.
    """

    adaptor = param.ClassSelector(class_=QueryAdaptor, doc="""
       The adaptor queries the data from some data source.""")

    application = param.Parameter(doc="""
       The overall monitoring application.""")

    current = param.List()

    filters = param.List(doc="A list of filters to be rendered.")

    metrics = param.List(doc="A list of metrics to be displayed.")

    sort_fields = param.List(default=[], doc="List of fields to sort by.")

    sort_reverse = param.Boolean(default=False, doc="""
        Whether to reverse the sort order.""")

    schema = param.Dict()

    title = param.String(doc="A title for this monitoring view.")

    layout = param.ObjectSelector(default='column', objects=['row', 'column', 'grid'])

    refresh_rate = param.Integer(default=None, doc="""
        How frequently to refresh the view by querying the adaptor.""")

    tsformat = param.String(default="%m/%d/%Y %H:%M:%S")

    height = param.Integer(default=400, doc="Height of the view cards.")

    width = param.Integer(default=400, doc="Width of the view cards.")

    def __init__(self, **params):
        metrics = params.get('metrics', [])
        sort = params.pop('sort', {})
        params['sort_fields'] = sort_fields = sort.get('fields', [])
        params['sort_reverse'] = sort_reverse = sort.get('reverse', False)
        self._sort_widget = pn.widgets.MultiSelect(
            options=[m.get('name') for m in metrics],
            sizing_mode='stretch_width', value=sort_fields,
            size=len(metrics)
        )
        self._reverse_widget = pn.widgets.Checkbox(
            value=sort_reverse, name='Reverse', margin=(5, 0, 0, 10)
        )
        self._reload_button = pn.widgets.Button(
            name='↻', width=50, css_classes=['reload'], margin=0
        )
        self._reload_button.on_click(self.update)
        super(View, self).__init__(**params)
        self._cards = []
        self._cache = {}
        self._stale = False
        self._cb = None
        self.timestamp = pn.pane.HTML(
            f'Last updated: {dt.datetime.now().strftime(self.tsformat)}',
            align='end', margin=10, sizing_mode='stretch_width'
        )
        self._update_metrics()
        for filt in self.filters:
            if isinstance(filt, FacetFilter):
                continue
            filt.param.watch(self._rerender, 'value')

    @pn.depends('_sort_widget.value', '_reverse_widget.value', watch=True)
    def _resort(self, *events):
        self.sort_fields = self._sort_widget.value
        self.sort_reverse = self._reverse_widget.value
        self._update_metrics()
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

    @pn.depends('current')
    def _update_metrics(self):
        filters = [filt for filt in self.filters
                   if not isinstance(filt, FacetFilter)]
        facets = [filt.filters for filt in self.filters
                  if isinstance(filt, FacetFilter)]
        cards = []
        for facet_filters in product(*facets):
            metrics = []
            metric_filters = filters + list(facet_filters)
            key = tuple(str(f.value) for f in facet_filters)
            for metric in self.metrics:
                metric = dict(metric)
                metric_type = metric.pop('type', None)
                metric_key = key+(metric['name'],)
                if metric_key not in self._cache:
                    metric = MetricView.get(metric_type)(
                        adaptor=self.adaptor, filters=metric_filters, view=self, **metric
                    )
                    self._cache[metric_key] = metric
                else:
                    metric = self._cache.get(metric_key)
                    metric.update(rerender=False)
                if metric:
                    metrics.append(metric)
            if not metrics:
                continue

            sort_key = []
            for field in self.sort_fields:
                values = [m.get_value() for m in metrics if m.name == field]
                if values:
                    sort_key.append(values[0])

            if facet_filters:
                title = ' '.join([f'{f.label}: {f.value}' for f in facet_filters])
            else:
                title = self.title

            if self.layout == 'row':
                item = pn.Row(*(m.panel for m in metrics))
            elif self.layout == 'grid':
                item = pn.GridBox(*(m.panel for m in metrics), ncols=2)
            else:
                item = pn.Column(*(m.panel for m in metrics))

            card = pn.Card(item, height=self.height, width=self.width, title=title)
            cards.append((tuple(sort_key), card))
        if self.sort:
            cards = sorted(cards, key=lambda x: x[0])
            if self.sort_reverse:
                cards = cards[::-1]
        self._cards[:] = [card for _, card in cards]

    def _rerender(self, *events):
        if not self._stale:
            return
        self._update_metrics()
        self.application._rerender()

    def update(self, *events):
        self.adaptor.update()
        self.timestamp.object = f'Last updated: {dt.datetime.now().strftime(self.tsformat)}'
        self._update_metrics()

    @property
    def panels(self):
        return self._cards

    @property
    def filter_panel(self):
        views = [pn.pane.Markdown('### Filters', margin=(0, 5))]
        views.extend([filt.panel for filt in self.filters if filt.panel is not None])
        views.append(pn.layout.Divider())
        views.extend([pn.pane.Markdown('### Sort', margin=(0, 5)), self._sort_widget, self._reverse_widget])
        views.append(pn.layout.Divider())
        views.append(pn.Row(self._reload_button, self.timestamp, sizing_mode='stretch_width'))        
        return pn.Card(*views, title=self.title, sizing_mode='stretch_width') if views else None
