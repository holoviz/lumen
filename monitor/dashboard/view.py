from itertools import product

import param
import panel as pn

from .adaptor import QueryAdaptor
from .filter import FacetFilter
from .metric import MetricView


class View(param.Parameterized):

    adaptor = param.ClassSelector(class_=QueryAdaptor)

    current = param.List()

    filters = param.List()

    metrics = param.List()

    schema = param.Dict()

    title = param.String()

    height = param.Integer(default=400)

    width = param.Integer(default=400)

    def __init__(self, **params):
        super(View, self).__init__(**params)
        self._cards = []
        self._update_metrics()

    @pn.depends('current')
    def _update_metrics(self):
        facets = [filt.filters for filt in self.filters
                  if isinstance(filt, FacetFilter)]
        filters = [filt for filt in self.filters
                   if not isinstance(filt, FacetFilter)]
        cards = []
        for facet_filters in product(*facets):
            metrics = []
            metric_filters = filters + list(facet_filters)
            for metric in self.metrics:
                metric = dict(metric)
                metric_type = metric.pop('type', None)
                metric = MetricView.get(metric_type)(
                    adaptor=self.adaptor, filters=metric_filters, **metric
                )
                metrics.append(metric)
            if facet_filters:
                title = ' '.join([f'{f.label}: {f.value}' for f in facet_filters])
            else:
                title = self.title
            card = pn.Card(*(m.view() for m in metrics), height=self.height,
                           width=self.width, title=title)
            cards.append(card)
        self._cards[:] = cards

    @property
    def panels(self):
        return self._cards
