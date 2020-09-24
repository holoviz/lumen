"""
The View classes render the metrics returned by a QueryAdaptor as a
Panel object.
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

    transforms = param.List()

    monitor = param.Parameter()

    view_type = None

    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        self._panel = None
        for filt in self.filters:
            filt.param.watch(self._update_panel, 'value')
        self._cache = None
        self._update_panel(rerender=False)

    @classmethod
    def get(cls, view_type):
        """
        Returns the matching 
        """
        for view in param.concrete_descendents(cls).values():
            if view.view_type == view_type:
                return view
        return DefaultView

    def __bool__(self):
        return len(self._cache) > 0

    def get_data(self):
        if self._cache is not None:
            return self._cache
        query = {filt.name: filt.query for filt in self.filters if filt.query is not None}
        metric = self.adaptor.get_metric(self.name, **query)
        for transform in self.transforms:
            metric = transform.apply(metric)
        self._cache = metric
        return self._cache

    def get_value(self):
        data = self.get_data()
        if not len(data):
            return None
        row = data.iloc[0]
        return row[self.name]

    def get_panel(self):
        return pn.panel(self.get_data())

    def _update_panel(self, *events, rerender=True):
        self._cache = None
        self._panel = self.get_panel()
        if rerender and any(isinstance(f, ConstantFilter) for f in self.filters) and not self:
            self.view._stale = True

    def update(self, rerender=True):
        self._update_panel(rerender=rerender)

    @property
    def panel(self):
        return self._panel


class DefaultView(MetricView):

    view_type = None


class StringView(MetricView):

    font_size = param.String(default='24pt')

    view_type = 'string'

    def get_panel(self):
        value = self.get_value()
        if value is None:
            return pn.pane.HTML('No info')
        return pn.pane.HTML(f'<p style="font-size: {self.font_size}>{value}</p>')


class IndicatorView(MetricView):

    indicator = param.String()

    indicator_options = param.Dict()

    label = param.String()

    view_type = 'indicator'

    def __init__(self, **params):
        self_params = {k: v for k, v in params.items() if k in self.param}
        options = {
            k: v for k, v in params.items() if k not in list(self.param)
        }
        options['name'] = self_params.pop('label', self_params.get('name', ''))
        super().__init__(indicator_options=options, **self_params)

    def get_panel(self):
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


class hvPlotView(MetricView):

    kind = param.String()

    x = param.String()

    y = param.String()

    kwargs = param.Dict()

    opts = param.Dict()

    view_type = 'hvplot'

    def __init__(self, **params):
        import hvplot.pandas # noqa
        super().__init__(**params)

    def get_panel(self):
        df = self.get_data()
        if df is None or not len(df):
            return pn.pane.HTML('No data')
        return df.hvplot(kind=self.kind, x=self.x, y=self.y,
                         **self.kwargs).opts(**self.opts)
