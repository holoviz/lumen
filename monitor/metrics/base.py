"""
The MetricView classes render the metrics returned by a QueryAdaptor
as a Panel object. 
"""

import param
import panel as pn

from ..adaptors import QueryAdaptor
from ..filters import ConstantFilter


class MetricView(param.Parameterized):
    """
    A MetricView renders a metric as an object.
    """

    adaptor = param.ClassSelector(class_=QueryAdaptor)

    application = param.Parameter()

    filters = param.List()

    view = param.Parameter()

    metric_type = None

    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        self._view = None
        for filt in self.filters:
            filt.param.watch(self._update_view, 'value')
        self._cache = None
        self._update_view(rerender=False)

    @classmethod
    def get(cls, metric_type):
        """
        Returns the matching 
        """
        for metric in param.concrete_descendents(cls).values():
            if metric.metric_type == metric_type:
                return metric
        return DefaultMetricView

    def __bool__(self):
        return len(self._cache) > 0

    def get_data(self):
        if self._cache is not None:
            return self._cache
        query = {filt.name: filt.query for filt in self.filters if filt.query is not None}
        self._cache = self.adaptor.get_metric(self.name, **query)
        return self._cache

    def get_value(self):
        data = self.get_data()
        if not len(data):
            return None
        row = data.iloc[0]
        return row[self.name]

    def get_view(self):
        return pn.panel(self.get_data())

    def _update_view(self, *events, rerender=True):
        self._cache = None
        self._view = self.get_view()
        if rerender and any(isinstance(f, ConstantFilter) for f in self.filters) and not self:
            self.view._stale = True

    def update(self, rerender=True):
        self._update_view(rerender=rerender)

    @property
    def panel(self):
        return self._view


class DefaultMetricView(MetricView):

    metric_type = None


class StringMetricView(MetricView):

    font_size = param.String(default='24pt')

    metric_type = 'string'

    def get_view(self):
        value = self.get_value()
        if value is None:
            return pn.pane.HTML('No info')
        return pn.pane.HTML(f'<p style="font-size: {self.font_size}>{value}</p>')


class IndicatorMetricView(MetricView):

    indicator = param.String()

    indicator_options = param.Dict()

    label = param.String()

    metric_type = 'indicator'

    def __init__(self, **params):
        self_params = {k: v for k, v in params.items() if k in self.param}
        options = {
            k: v for k, v in params.items() if k not in list(self.param)
        }
        options['name'] = self_params.pop('label', self_params.get('name', ''))
        super().__init__(indicator_options=options, **self_params)

    def get_view(self):
        value = self.get_value()
        if value is None:
            return pn.pane.HTML('No info')
        indicators = param.concrete_descendents(pn.widgets.indicators.ValueIndicator)
        indicator_name = self.indicator[0].upper() + self.indicator[1:]
        indicator = indicators.get(indicator_name)
        if indicator is None:
            raise ValueError(f'No such indicator as {self.indicator}, '
                             'ensure the indicator option in the spec '
                             'specifies a Panel ValueIndicator that '
                             'exists and has been imported')
        return indicator(**self.indicator_options, value=self.get_value())
