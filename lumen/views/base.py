"""
The View classes render the data returned by a Source as a Panel
object.
"""

import param
import panel as pn

from ..filters import ConstantFilter
from ..sources import Source


class View(param.Parameterized):
    """
    A View renders the data returned by a Source as a Viewable Panel
    object. The base class provides methods which query the Source for
    the latest data given the current filters and applies all specified
    `transforms`.

    Subclasses should use these methods to query the data and return
    a Viewable Panel object in the `get_panel` method.
    """

    application = param.Parameter(constant=True, doc="""
        The containing application.""")

    filters = param.List(constant=True, doc="""
        A list of Filter object providing the query parameters for the
        Source.""")

    monitor = param.Parameter(constant=True, doc="""
        The Monitor class which owns this view.""")

    source = param.ClassSelector(class_=Source, constant=True, doc="""
        The Source to query for the data.""")

    transforms = param.List(constant=True, doc="""
        A list of transforms to apply to the data returned by the
        Source before visualizing it.""")

    variable = param.String(doc="The variable being visualized.")

    view_type = None

    def __init__(self, **params):
        super().__init__(**params)
        self._panel = None
        for filt in self.filters:
            filt.param.watch(self._update_panel, 'value')
        self._cache = None
        self._update_panel(rerender=False)

    @classmethod
    def _get_type(cls, view_type):
        """
        Returns the matching
        """
        for view in param.concrete_descendents(cls).values():
            if view.view_type == view_type:
                return view
        return View

    def __bool__(self):
        return self._cache is not None and len(self._cache) > 0

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

    def get_data(self):
        """
        Queries the Source for the data associated with a particular
        variable applying any filters and transformations specified on
        the View. Unlike `get_value` this should be used when multiple
        return values are expected.

        Returns
        -------
        DataFrame
            The data associated with a particular variable after
            filtering and transformations are applied.
        """
        if self._cache is not None:
            return self._cache
        query = {filt.name: filt.query for filt in self.filters
                 if filt.query is not None}
        data = self.source.get(self.variable, **query)
        for transform in self.transforms:
            data = transform.apply(data)
        self._cache = data
        return data

    def get_value(self):
        """
        Queries the Source for the data associated with a particular
        variable applying any filters and transformations specified on
        the View. Unlike `get_data` this method returns a single
        scalar value associated with the variable and should therefore
        only be used if only a single.

        Returns
        -------
        object
            A single scalar value representing the current value of
            the queried variable.
        """
        data = self.get_data()
        if not len(data):
            return None
        if len(data) > 1:
            self.param.warning()
        row = data.iloc[-1]
        return row[self.variable]

    def get_panel(self):
        """
        Constructs and returns a Panel object which will represent a
        view of the queried data.

        Returns
        -------
        panel.Viewable
            A Panel Viewable object representing a current
            representation of the data.
        """
        return pn.panel(self.get_data())

    def update(self, rerender=True):
        """
        Triggers an update in the View.

        Parameters
        ----------
        rerender : bool
            Whether to force a rerender of the containing Monitor.
        """
        self._update_panel(rerender=rerender)

    @property
    def panel(self):
        return self._panel


class StringView(View):
    """
    The StringView renders the latest value of the variable as a HTML
    string with a specified fontsize.
    """

    font_size = param.String(default='24pt', doc="""
        The font size of the rendered variable value.""")

    view_type = 'string'

    def get_panel(self):
        value = self.get_value()
        if value is None:
            return pn.pane.HTML('No info')
        return pn.pane.HTML(f'<p style="font-size: {self.font_size}>{value}</p>')


class IndicatorView(View):
    """
    The IndicatorView renders the latest variable value as a Panel
    Indicator.
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


class hvPlotView(View):
    """
    The hvPlotView renders the queried data as a bokeh plot generated
    with hvPlot. hvPlot allows for a concise declaration of a plot via
    its simple API.
    """

    kind = param.String(doc="The kind of plot, e.g. 'scatter' or 'line'.")

    x = param.String(doc="The column to render on the x-axis.")

    y = param.String(doc="The column to render on the y-axis.")

    kwargs = param.Dict(default={}, doc="Dictionary of additional kwargs.")

    opts = param.Dict(default={}, doc="HoloVies option to apply on the plot.")

    view_type = 'hvplot'

    def __init__(self, **params):
        import hvplot.pandas # noqa
        super().__init__(**params)

    def get_panel(self):
        df = self.get_data()
        if df is None or not len(df):
            return pn.pane.HTML('No data')
        plot = df.hvplot(kind=self.kind, x=self.x, y=self.y,
                         **self.kwargs)
        if self.opts:
            plot.opts(**self.opts)
        return plot
