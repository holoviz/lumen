import param
import panel as pn

from .adaptor import QueryAdaptor
from .metric import Metric


class View(param.Parameterized):

    adaptor = param.ClassSelector(class_=QueryAdaptor)

    title = param.String()

    current = param.List()

    schema = param.Dict()

    metrics = param.List()

    filters = param.List()

    height = param.Integer(default=400)

    width = param.Integer(default=400)

    def __init__(self, **params):
        super(View, self).__init__(**params)
        self._card = pn.Card(height=self.height, width=self.width, title=self.title)
        self._update_metrics()

    @pn.depends('current')
    def _update_metrics(self):
        metrics = []
        for metric in self.metrics:
            metric_type = metric.pop('type', None)
            metric = Metric.get(metric_type)(
                adaptor=self.adaptor, filters=self.filters, **metric
            )
            metrics.append(metric)
        self._metrics = metrics
        self._card[:] = [m.view() for m in metrics]

    @property
    def panel(self):
        return self._card
