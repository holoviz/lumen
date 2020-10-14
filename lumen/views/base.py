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

    table = param.String(doc="The table being visualized.")

    variable = param.String(doc="The variable being visualized.")

    view_type = None

    def __init__(self, **params):
        self.kwargs = {k: v for k, v in params.items() if k not in self.param}
        super().__init__(**{k: v for k, v in params.items() if k in self.param})
        self._panel = None
        for filt in self.filters:
            filt.param.watch(self.update, 'value')
        self._cache = None
        self.update()

    @classmethod
    def _get_type(cls, view_type):
        """
        Returns the matching View type.
        """
        try:
            __import__(f'lumen.views.{view_type}')
        except Exception:
            pass
        for view in param.concrete_descendents(cls).values():
            if view.view_type == view_type:
                return view
        if view_type is not None:
            self.param.warning(f"View type '{view_type}' could not be found.")
        return View

    def __bool__(self):
        return self._cache is not None and len(self._cache) > 0

    def _update_panel(self, *events):
        """
        Updates the cached Panel object and notifies the containing
        Monitor object if it is stale and has to rerender.
        """
        try:
            if self._panel is not None:
                self.update_panel(self._panel)
                return
        except NotImplementedError:
            pass
        self._panel = self.get_panel()
        self.monitor._stale = True

    def get_data(self):
        """
        Queries the Source for the specified table applying any
        filters and transformations specified on the View. Unlike
        `get_value` this should be used when multiple return values
        are expected.

        Returns
        -------
        DataFrame
            The queried table after filtering and transformations are
            applied.
        """
        if self._cache is not None:
            return self._cache
        query = {filt.name: filt.query for filt in self.filters
                 if filt.query is not None}
        data = self.source.get(self.table, **query)
        for transform in self.transforms:
            data = transform.apply(data)
        if len(data):
            data = self.source._filter_dataframe(data, **query)
        self._cache = data
        return data

    def get_value(self, variable=None):
        """
        Queries the Source for the data associated with a particular
        variable applying any filters and transformations specified on
        the View. Unlike `get_data` this method returns a single
        scalar value associated with the variable and should therefore
        only be used if only a single.

        Parameters
        ----------
        variable: str (optional)
            The variable from the table to return; if None uses
            variable defined on the View.

        Returns
        -------
        object
            A single scalar value representing the current value of
            the queried variable.
        """
        data = self.get_data()
        if not len(data):
            return None
        row = data.iloc[-1]
        return row[self.variable if variable is None else variable]

    def get_panel(self):
        """
        Constructs and returns a Panel object which will represent a
        view of the queried table.

        Returns
        -------
        panel.Viewable
            A Panel Viewable object representing a current
            representation of the queried table.
        """
        return pn.panel(self.get_data())

    def update(self, invalidate_cache=True):
        """
        Triggers an update in the View.

        Parameters
        ----------
        invalidate_cache : bool
            Whether to force a rerender of the containing Monitor.
        """
        if invalidate_cache:
            self._cache = None
        self._update_panel()

    def update_panel(self, panel):
        raise NotImplementedError

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
        return pn.pane.HTML(f'<p style="font-size: {self.font_size}>{value}</p>',
                            **self.kwargs)

    def update_panel(self, panel):
        value = self.get_value()
        if value is None:
            panel.object = 'No info'
        panel.object = f'<p style="font-size: {self.font_size}>{value}</p>'


class IndicatorView(View):
    """
    The IndicatorView renders the latest variable value as a Panel
    Indicator.
    """

    indicator = param.String(doc="The name of the panel Indicator type")

    label = param.String(doc="""
        A custom label to use for the Indicator.""")

    view_type = 'indicator'

    def __init__(self, **params):
        super().__init__(**params)
        name = params.get('label', params.get('variable', ''))
        self.kwargs['name'] = name
        self._panel.name = name

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
        return indicator(**self.kwargs, value=self.get_value())


class hvPlotView(View):
    """
    The hvPlotView renders the queried data as a bokeh plot generated
    with hvPlot. hvPlot allows for a concise declaration of a plot via
    its simple API.
    """

    kind = param.String(doc="The kind of plot, e.g. 'scatter' or 'line'.")

    x = param.String(doc="The column to render on the x-axis.")

    y = param.String(doc="The column to render on the y-axis.")

    opts = param.Dict(default={}, doc="HoloViews option to apply on the plot.")

    view_type = 'hvplot'

    def __init__(self, **params):
        import hvplot.pandas # noqa
        super().__init__(**params)

    def get_plot(self, df):
        plot = df.hvplot(
            kind=self.kind, x=self.x, y=self.y, **self.kwargs
        )
        return plot.opts(**self.opts) if self.opts else plot

    def get_panel(self):
        df = self.get_data()
        if df is None or not len(df):
            return pn.pane.HTML('No data')
        return pn.pane.HoloViews(self.get_plot(df))

    def update_panel(self, panel):
        df = self.get_data()
        if (df is None or not len(df)) and not isinstance(panel, pn.pane.HTML):
            raise NotImplementedError
        panel.object = self.get_plot(df)
