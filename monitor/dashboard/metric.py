"""
The MetricView classes render the metrics returned by a QueryAdaptor
as a Panel object. 
"""

import param
import panel as pn

from .adaptor import QueryAdaptor


class MetricView(param.Parameterized):
    """
    A MetricView renders a metric as an object.
    """

    adaptor = param.ClassSelector(class_=QueryAdaptor)

    filters = param.List()

    metric_type = None

    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        self._layout = pn.Row()
        for filt in self.filters:
            filt.param.watch(self._update_view, 'value')
        self._update_view()

    @classmethod
    def get(cls, metric_type):
        """
        Returns the matching 
        """
        for metric in param.concrete_descendents(cls).values():
            if metric.metric_type == metric_type:
                return metric
        return DefaultMetricView

    def get_data(self):
        query = {filt.name: filt.query for filt in self.filters}
        return self.adaptor.get_metric(self.name, **query)

    def get_view(self):
        return pn.panel(self.get_data())

    def _update_view(self, *events):
        self._layout[:] = [self.get_view()]

    def view(self):
        return self._layout


class DefaultMetricView(MetricView):

    metric_type = None
