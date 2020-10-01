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
    A MetricView renders a metric as a Panel Viewable. It provides
    methods which query the QueryAdaptor for the latest data given the
    current `filters`, applying all specified `transforms` and should
    then render this data as a Panel object in the `get_panel` method.
    """

    adaptor = param.ClassSelector(class_=QueryAdaptor, constant=True, doc="""
        The QueryAdaptor to query for metric data.""")

    application = param.Parameter(constant=True, doc="""
        The containing application.""")

    filters = param.List(constant=True, doc="""
        A list of Filter object providing the query parameters for the
        QueryAdaptor.""")

    transforms = param.List(constant=True, doc="""
        A list of transforms to apply to the data returned by the
        QueryAdaptor before visualizing it.""")

    monitor = param.Parameter(constant=True, doc="""
        The Monitor class which owns this view.""")

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
        """
        Queries the QueryAdaptor for the data associated with a
        particular metric applying any filters and transformations
        specified on the MetricView. Unlike `get_value` this should be
        used when multiple return values are expected.

        Returns
        -------
        pandas.DataFrame
            The data associated with a particular metric aftering
            filtering and transformations are applied.
        """
        if self._cache is not None:
            return self._cache
        query = {filt.name: filt.query for filt in self.filters
                 if filt.query is not None}
        metric = self.adaptor.get_metric(self.name, **query)
        for transform in self.transforms:
            metric = transform.apply(metric)
        self._cache = metric
        return self._cache

    def get_value(self):
        """
        Queries the QueryAdaptor for the data associated with a
        particular metric applying any filters and transformations
        specified on the MetricView. Unlike `get_data` this method
        returns a single scalar value associated with the metric and
        should therefore only be used if only a single.

        Returns
        -------
        object
            A single scalar value representing the current value of the
            metric.
        """
        data = self.get_data()
        if not len(data):
            return None
        row = data.iloc[-1]
        return row[self.name]

    def get_panel(self):
        """
        Constructs and returns a Panel object which will represent a
        view of the metric data.

        Returns
        -------
        panel.Viewable
            A Panel Viewable object representing a current
            representation of the metric data.
        """
        return pn.panel(self.get_data())

    def _update_panel(self, *events, rerender=True):
        """
        Updates the cached Panel object and notifies the containing
        Monitor object if it is stale and has to rerender.
        """
        self._cache = None
        self._panel = self.get_panel()
        if (rerender and any(isinstance(f, ConstantFilter) for f in self.filters)
            and not self):
            self.monitor._stale = True

    def update(self, rerender=True):
        """
        Triggers an update in the MetricView.

        Parameters
        ----------
        rerender : bool
            Whether to force a rerender of the containing Monitor.
        """
        self._update_panel(rerender=rerender)

    @property
    def panel(self):
        return self._panel


class DefaultView(MetricView):
    """
    The DefaultView is what is used if no particular view type is
    declared and renders the queried data as a table.
    """

    view_type = None


class StringView(MetricView):
    """
    The StringView renders the latest metric value as a HTML string
    with a specified fontsize.
    """

    font_size = param.String(default='24pt', doc="""
        The font size of the rendered metric string.""")

    view_type = 'string'

    def get_panel(self):
        value = self.get_value()
        if value is None:
            return pn.pane.HTML('No info')
        return pn.pane.HTML(f'<p style="font-size: {self.font_size}>{value}</p>')


class IndicatorView(MetricView):
    """
    The IndicatorView renders the latest metric value as a Panel
    Indicator. An IndicatorView therefore usually represents a single
    scalar value.
    """

    indicator = param.String(doc="The name of the panel Indicator type")

    indicator_options = param.Dict(doc="""
        A dictionary of options to supply to the Indicator.""")

    label = param.String(doc="""
        A custom label to use for the Indicator.""")

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
    """
    The hvPlotView renders the queried metric data as a bokeh plot
    generated with hvPlot. hvPlot allows for a concise declaration
    of a plot via its simple API.
    """

    kind = param.String(doc="The kind of plot, e.g. 'scatter' or 'line'.")

    x = param.String(doc="The column to render on the x-axis.")

    y = param.String(doc="The column to render on the y-axis.")

    kwargs = param.Dict(doc="Dictionary of additional kwargs.x")

    opts = param.Dict(doc="HoloVies option to apply on the plot.")

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
