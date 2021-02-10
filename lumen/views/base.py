"""
The View classes render the data returned by a Source as a Panel
object.
"""

from io import StringIO

import numpy as np
import param
import panel as pn

from bokeh.models import NumeralTickFormatter

from ..sources import Source
from ..transforms import Transform


class View(param.Parameterized):
    """
    A View renders the data returned by a Source as a Viewable Panel
    object. The base class provides methods which query the Source for
    the latest data given the current filters and applies all
    specified `transforms`.

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

    field = param.String(doc="The field being visualized.")

    view_type = None

    def __init__(self, **params):
        self.kwargs = {k: v for k, v in params.items() if k not in self.param}
        super().__init__(**{k: v for k, v in params.items() if k in self.param})
        self._panel = None
        for filt in self.filters:
            filt.param.watch(self.update, 'value')
        self._cache = None
        self._updates = None
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
            raise ValueError(f"View type '{view_type}' could not be found.")
        return View

    @classmethod
    def from_spec(cls, spec, monitor, source, filters):
        """
        Resolves a View specification given the schema of the Source
        it will be filtering on.

        Parameters
        ----------
        spec: dict
            Specification declared as a dictionary of parameter values.
        monitor: lumen.monitor.Monitor
            The Monitor object that holds this View.
        source: lumen.sources.Source
            The Source object containing the tables the View renders.
        filters: list(lumen.filters.Filter)
            A list of Filter objects which provide query values for
            the Source.

        Returns
        -------
        The resolved View object.
        """
        spec = dict(spec)
        transform_specs = spec.pop('transforms', [])
        transforms = [Transform.from_spec(tspec) for tspec in transform_specs]
        view_type = View._get_type(spec.pop('type', None))
        return view_type(
            filters=filters, monitor=monitor, source=source,
            transforms=transforms, **spec
        )

    def __bool__(self):
        return self._cache is not None and len(self._cache) > 0

    def _update_panel(self, *events):
        """
        Updates the cached Panel object and notifies the containing
        Monitor object if it is stale and has to rerender.
        """
        if self._panel is not None:
            self._updates = self._get_params()
            if self._updates is not None:
                return False
        self._panel = self.get_panel()
        return True

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
        query = {
            filt.field: filt.query for filt in self.filters
            if filt.query is not None and
            (filt.table is None or filt.table == self.table)
        }
        data = self.source.get(self.table, **query)
        for transform in self.transforms:
            data = transform.apply(data)
        if len(data):
            data = self.source._filter_dataframe(data, **query)
        self._cache = data
        return data

    def get_value(self, field=None):
        """
        Queries the Source for the data associated with a particular
        field applying any filters and transformations specified on
        the View. Unlike `get_data` this method returns a single
        scalar value associated with the field and should therefore
        only be used if only a single.

        Parameters
        ----------
        field: str (optional)
            The field from the table to return; if None uses
            field defined on the View.

        Returns
        -------
        object
            A single scalar value representing the current value of
            the queried field.
        """
        data = self.get_data()
        if not len(data) or field is not None and field not in data.columns:
            return None
        row = data.iloc[-1]
        return row[self.field if field is None else field]

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

        Returns
        -------
        stale : bool
            Whether the panel on the View is stale and needs to be
            rerendered.
        """
        if invalidate_cache:
            self._cache = None
        return self._update_panel()

    def _get_params(self):
        return None

    @property
    def panel(self):
        return self._panel


class StringView(View):
    """
    The StringView renders the latest value of the field as a HTML
    string with a specified fontsize.
    """

    font_size = param.String(default='24pt', doc="""
        The font size of the rendered field value.""")

    view_type = 'string'

    def get_panel(self):
        return pn.pane.HTML(**self._get_params())

    def _get_params(self):
        value = self.get_value()
        params = dict(self.kwargs)
        if value is None:
            params['object'] = 'No info'
        else:
            params['object'] = f'<p style="font-size: {self.font_size}">{value}</p>'
        return params


class IndicatorView(View):
    """
    The IndicatorView renders the latest field value as a Panel
    Indicator.
    """

    indicator = param.String(doc="The name of the panel Indicator type")

    label = param.String(doc="""
        A custom label to use for the Indicator.""")

    view_type = 'indicator'

    def __init__(self, **params):
        super().__init__(**params)
        name = params.get('label', params.get('field', ''))
        self.kwargs['name'] = name
        self._panel.name = name

    def get_panel(self):
        indicators = param.concrete_descendents(pn.widgets.indicators.ValueIndicator)
        indicator_name = self.indicator.title()
        indicator = indicators.get(indicator_name)
        if indicator is None:
            raise ValueError(f'No such indicator as {self.indicator}, '
                             'ensure the indicator option in the spec '
                             'specifies a Panel ValueIndicator that '
                             'exists and has been imported')
        return indicator(**self._get_params())

    def _get_params(self):
        value = self.get_value()
        if (not isinstance(value, (type(None), str)) and np.isnan(value)):
            value = None
        return dict(self.kwargs, value=value)


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

    streaming = param.Boolean(default=False, doc="""
      Whether to stream new data to the plot or rerender the plot.""")

    view_type = 'hvplot'

    def __init__(self, **params):
        import hvplot.pandas # noqa
        self._stream = None
        super().__init__(**params)

    def get_plot(self, df):
        processed = {}
        for k, v in self.kwargs.items():
            if k.endswith('formatter') and isinstance(v, str) and '%' not in v:
                v = NumeralTickFormatter(format=v)
            processed[k] = v
        if self.streaming:
            processed['stream'] = self._stream
        plot = df.hvplot(
            kind=self.kind, x=self.x, y=self.y, **processed
        )
        return plot.opts(**self.opts) if self.opts else plot

    def get_panel(self):
        return pn.pane.HoloViews(**self._get_params())

    def _get_params(self):
        df = self.get_data()
        if self.streaming:
            from holoviews.streams import Pipe
            self._stream = Pipe(data=df)
        return dict(object=self.get_plot(df))

    def update(self, invalidate_cache=True):
        """
        Triggers an update in the View.

        Parameters
        ----------
        invalidate_cache : bool
            Whether to force a rerender of the containing Monitor.

        Returns
        -------
        stale : bool
            Whether the panel on the View is stale and needs to be
            rerendered.
        """
        if invalidate_cache:
            self._cache = None
        if not self.streaming or self._stream is None:
            return self._update_panel()
        self._stream.send(self.get_data())
        return False


class Table(View):
    """
    Renders a Source table using a Panel Table widget.
    """

    view_type = 'table'

    def get_panel(self):
        return pn.widgets.tables.Tabulator(**self._get_params())

    def _get_params(self):
        return dict(value=self.get_data(), disabled=True, **self.kwargs)


class Download(View):
    """
    The Download View allows downloading the current table as a csv or
    xlsx file.
    """

    filename = param.String(default='data', doc="""
      Filename of the downloaded file.""")

    filetype = param.Selector(default='csv', objects=['csv', 'xlsx'], doc="""
      File type of the downloaded file.""")

    save_kwargs = param.Dict(default={}, doc="""
      Options for the to_csv or to_excel methods.""")

    view_type = 'download'

    def __bool__(self):
        return True

    def _get_file_data(self):
        df = self.get_data()
        sio = StringIO()
        if self.filetype == 'csv':
            savefn = df.to_csv
        elif self.filetype == 'xlsx':
            savefn = df.to_excel
        savefn(sio, **self.save_kwargs)
        sio.seek(0)
        return sio

    def get_panel(self):
        return pn.widgets.FileDownload(**self._get_params())

    def _get_params(self):
        filename = f'{self.filename}.{self.filetype}'
        return dict(filename=filename, callback=self._get_file_data, **self.kwargs)
