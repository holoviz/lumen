"""
The View classes render the data returned by a Source as a Panel
object.
"""
import sys

from io import StringIO, BytesIO
from weakref import WeakKeyDictionary

import numpy as np
import param
import panel as pn

from bokeh.models import NumeralTickFormatter
from panel.pane.base import PaneBase
from panel.pane.perspective import (
    THEMES as _PERSPECTIVE_THEMES, Plugin as _PerspectivePlugin
)
from panel.param import Param
from panel.viewable import Viewer

from ..base import Component
from ..config import _INDICATORS
from ..filters import ParamFilter
from ..panel import DownloadButton
from ..sources import Source
from ..state import state
from ..transforms import Transform
from ..util import is_ref

DOWNLOAD_FORMATS = ['csv', 'xlsx', 'json', 'parquet']


class Download(Viewer):

    color = param.Color(default='grey', allow_None=True, doc="""
      The color of the download button.""")

    hide = param.Boolean(default=False, doc="""
      Whether the download button hides when not in focus.""")

    format = param.ObjectSelector(default=None, objects=DOWNLOAD_FORMATS, doc="""
      The format to download the data in.""")

    kwargs = param.Dict(default={}, doc="""
      Keyword arguments passed to the serialization function, e.g.
      data.to_csv(file_obj, **kwargs).""")

    size = param.Integer(default=18, doc="""
      The size of the download button.""")

    view = param.Parameter(doc="Holds the current view.")

    @classmethod
    def from_spec(cls, spec):
        return cls(**spec)

    def __bool__(self):
        return self.format is not None

    def _table_data(self):
        if self.format in ('json', 'csv'):
            io = StringIO()
        else:
            io = BytesIO()
        data = self.view.get_data()
        if self.format == 'csv':
            data.to_csv(io, **self.kwargs)
        elif self.format == 'json':
            data.to_json(io, **self.kwargs)
        elif self.format == 'xlsx':
            data.to_excel(io, **self.kwargs)
        elif self.format == 'parquet':
            data.to_parquet(io, **self.kwargs)
        io.seek(0)
        return io

    def __panel__(self):
        filename = f'{self.view.table}_{self.view.name}_view.{self.format}'
        return DownloadButton(
            callback=self._table_data, filename=filename, color=self.color,
            size=18, hide=self.hide
        )


class View(Component):
    """
    A View renders the data returned by a Source as a Viewable Panel
    object. The base class provides methods which query the Source for
    the latest data given the current filters and applies all
    specified `transforms`.

    Subclasses should use these methods to query the data and return
    a Viewable Panel object in the `get_panel` method.
    """

    controls = param.List(default=[], doc="""
        Parameters that should be exposed as widgets in the UI.""")

    download = param.ClassSelector(class_=Download, default=Download(), doc="""
        The download objects determines whether and how the source tables
        can be downloaded.""")

    filters = param.List(constant=True, doc="""
        A list of Filter object providing the query parameters for the
        Source.""")

    source = param.ClassSelector(class_=Source, constant=True, doc="""
        The Source to query for the data.""")

    selection_group = param.String(default=None, doc="""
        Declares a selection group the plot is part of. This feature
        requires the separate HoloViews library.""")

    transforms = param.List(constant=True, doc="""
        A list of transforms to apply to the data returned by the
        Source before visualizing it.""")

    sql_transforms = param.List(constant=True, doc="""
        A list of sql transforms to apply to the data returned by the
        Source before visualizing it.""")

    table = param.String(doc="The table being visualized.")

    field = param.Selector(doc="The field being visualized.")

    view_type = None

    # Panel extension to load to render this View
    _extension = None

    # Parameters which reference fields in the table
    _field_params = ['field']

    _selections = WeakKeyDictionary()

    _supports_selections = False

    __abstract = True

    def __init__(self, **params):
        self._cache = None
        self._ls = None
        self._panel = None
        self._updates = None
        refs = params.pop('refs', {})
        self.kwargs = {k: v for k, v in params.items() if k not in self.param}

        # Populate field selector parameters
        params = {k: v for k, v in params.items() if k in self.param}
        source, table = params.pop('source', None), params.pop('table', None)
        if source is None:
            raise ValueError("Views must declare a Source.")
        if table is None:
            raise ValueError("Views must reference a table on the declared Source.")
        fields = list(source.get_schema(table))
        for fp in self._field_params:
            if isinstance(self.param[fp], param.Selector):
                self.param[fp].objects = fields
        super().__init__(source=source, table=table, refs=refs, **params)
        self.download.view = self
        for transform in self.transforms:
            for fp in transform._field_params:
                if isinstance(transform.param[fp], param.Selector):
                    transform.param[fp].objects = fields
        if self.selection_group:
            self._init_link_selections()

    def _init_link_selections(self):
        doc = pn.state.curdoc
        if self._ls is not None or doc is None:
            return
        if doc not in View._selections and self.selection_group:
            View._selections[doc] = {}
        self._ls = View._selections.get(doc, {}).get(self.selection_group)
        if self._ls is None:
            from holoviews.selection import link_selections
            self._ls = link_selections.instance()
            if self.selection_group:
                View._selections[doc][self.selection_group] = self._ls
        if 'selection_expr' in self.param:
            self._ls.param.watch(self._update_selection_expr, 'selection_expr')

    def _update_selection_expr(self, event):
        self.selection_expr = event.new

    @classmethod
    def from_spec(cls, spec, source, filters):
        """
        Resolves a View specification given the schema of the Source
        it will be filtering on.

        Parameters
        ----------
        spec: dict
            Specification declared as a dictionary of parameter values.
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
        sql_transform_specs = spec.pop('sql_transforms', [])
        sql_transforms = [Transform.from_spec(tspec) for tspec in sql_transform_specs]
        view_type = View._get_type(spec.pop('type', None))
        resolved_spec, refs = {}, {}
        for p, value in spec.items():
            if p not in view_type.param:
                resolved_spec[p] = value
                continue
            parameter = view_type.param[p]
            if is_ref(value):
                refs[p] = value
                value = state.resolve_reference(value)
            if isinstance(parameter, param.ObjectSelector) and parameter.names:
                try:
                    value = parameter.names.get(value, value)
                except Exception:
                    pass
            resolved_spec[p] = value

        # Resolve download options
        download_spec = spec.pop('download', {})
        if isinstance(download_spec, str):
            download_spec = {'format': download_spec}
        resolved_spec['download'] = Download.from_spec(download_spec)
        view = view_type(
            filters=filters, source=source, transforms=transforms,
            sql_transforms=sql_transforms, refs=refs, **resolved_spec
        )

        # Resolve ParamFilter parameters
        for filt in filters:
            if isinstance(filt, ParamFilter):
                if not isinstance(filt.parameter, str):
                    continue
                name, parameter = filt.parameter.split('.')
                if name == view.name and parameter in view.param:
                    filt.parameter = view.param[parameter]
        return view

    def __bool__(self):
        return self._cache is not None and len(self._cache) > 0

    def _update_panel(self, *events):
        """
        Updates the cached Panel object and returns a boolean value
        indicating whether a rerender is required.
        """
        if self._panel is not None:
            self._cleanup()
            self._updates = self._get_params()
            if self._updates is not None:
                return False
        self._panel = self.get_panel()
        return True

    def _cleanup(self):
        """
        Method that is called on update.
        """

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
        query = {}
        if self.sql_transforms:
            if not self.source._supports_sql:
                raise ValueError(
                    'Can only use sql transforms source that support them. '
                    f'Found source typed {self.source.source_type!r} instead.'
                )
            query['sql_transforms'] = self.sql_transforms

        for filt in self.filters:
            filt_query = filt.query
            if (filt_query is not None and
            not getattr(filt, 'disabled', None) and
            (filt.table is None or filt.table == self.table)):
                query[filt.field] = filt_query
        data = self.source.get(self.table, **query)
        for transform in self.transforms:
            data = transform.apply(data)
        if len(data):
            data = self.source._filter_dataframe(data, **query)
        for filt in self.filters:
            if not isinstance(filt, ParamFilter):
                continue
            from holoviews import Dataset
            if filt.value is not None:
                ds = Dataset(data)
                data = ds.select(filt.value).data
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

    def update(self, *events, invalidate_cache=True):
        """
        Triggers an update in the View.

        Parameters
        ----------
        events: tuple
            param events that may trigger an update.
        invalidate_cache : bool
            Whether to clear the View's cache.

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
    def control_panel(self):
        column = pn.Column(sizing_mode='stretch_width')
        if self.controls:
            column.append(
                Param(
                    self.param, parameters=self.controls, sizing_mode='stretch_width'
                )
            )
        for trnsfm in self.transforms+self.sql_transforms:
            if trnsfm.controls:
                column.append(trnsfm.control_panel)
        index = (1 if self.controls else 0)
        if len(column) > index:
            column.insert(index, '### Transforms')
        return column

    @property
    def panel(self):
        panel = self._panel
        if isinstance(panel, PaneBase):
            if len(panel.layout) == 1 and panel._unpack:
                panel = panel.layout[0]
            else:
                panel = panel._layout
        if self.download:
            return pn.Column(self.download, panel, sizing_mode='stretch_width')
        return panel

    @property
    def refs(self):
        refs = super().refs
        for c in self.controls:
            if c not in refs:
                refs.append(c)
        return refs


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

    indicator = param.Selector(objects=_INDICATORS, doc="""
        The name of the panel Indicator type.""")

    label = param.String(doc="""
        A custom label to use for the Indicator.""")

    view_type = 'indicator'

    def __init__(self, **params):
        if 'indicator' in params and isinstance(params['indicator'], str):
            params['indicator'] = _INDICATORS[params['indicator']]
        super().__init__(**params)
        name = params.get('label', params.get('field', ''))
        self.kwargs['name'] = name

    def get_panel(self):
        return self.indicator(**self._get_params())

    def _get_params(self):
        params = dict(self.kwargs)
        if 'data' in self.indicator.param:
            params['data'] = self.get_data()
        else:
            value = self.get_value()
            if (not isinstance(value, (type(None), str)) and np.isnan(value)):
                value = None
            params['value'] = value
        return params


class hvPlotBaseView(View):

    kind = param.String(default=None, doc="The kind of plot, e.g. 'scatter' or 'line'.")

    x = param.Selector(doc="The column to render on the x-axis.")

    y = param.Selector(doc="The column to render on the y-axis.")

    by = param.ListSelector(doc="The column(s) to facet the plot by.")

    groupby = param.ListSelector(doc="The column(s) to group by.")

    __abstract = True

    def __init__(self, **params):
        import hvplot.pandas # noqa
        if 'dask' in sys.modules:
            try:
                import hvplot.dask # noqa
            except Exception:
                pass
        if 'by' in params and isinstance(params['by'], str):
            params['by'] = [params['by']]
        if 'groupby' in params and isinstance(params['groupby'], str):
            params['groupby'] = [params['groupby']]
        super().__init__(**params)


class hvPlotUIView(hvPlotBaseView):
    """
    The hvPlotUIView displays provides a component for exploring
    datasets using widgets.
    """

    view_type = 'hvplot_ui'

    def _get_args(self):
        from hvplot.ui import hvPlotExplorer
        params = {
            k: v for k, v in self.param.values().items()
            if k in hvPlotExplorer.param and v is not None and k != 'name'
        }
        return (self.get_data(),), dict(params, **self.kwargs)

    def get_panel(self):
        from hvplot.ui import hvPlotExplorer
        args, kwargs = self._get_args()
        return hvPlotExplorer(*args, **kwargs)


class hvPlotView(hvPlotBaseView):
    """
    The hvPlotView renders the queried data as a bokeh plot generated
    with hvPlot. hvPlot allows for a concise declaration of a plot via
    its simple API.
    """

    opts = param.Dict(default={}, doc="HoloViews options to apply on the plot.")

    streaming = param.Boolean(default=False, doc="""
        Whether to stream new data to the plot or rerender the plot.""")

    selection_expr = param.Parameter(doc="""
        A selection expression caputirng the current selection applied
        on the plot.""")

    view_type = 'hvplot'

    _field_params = ['x', 'y', 'by', 'groupby']

    _supports_selections = True

    def __init__(self, **params):
        self._stream = None
        self._linked_objs = []
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
        plot = plot.opts(**self.opts) if self.opts else plot
        if self.selection_group or 'selection_expr' in self._param_watchers:
            plot = self._link_plot(plot)
        return plot

    def _link_plot(self, plot):
        self._init_link_selections()
        if self._ls is None:
            return plot
        linked_objs = list(self._ls._plot_reset_streams)
        plot = self._ls(plot)
        self._linked_objs += [
            o for o in self._ls._plot_reset_streams if o not in linked_objs
        ]
        return plot

    def _cleanup(self):
        if self._ls is None:
            return
        for obj in self._linked_objs:
            reset = self._ls._plot_reset_streams.pop(obj)
            sel_expr = self._ls._selection_expr_streams.pop(obj)
            self._ls._cross_filter_stream.input_streams.remove(sel_expr)
            sel_expr.clear()
            sel_expr.source = None
            reset.clear()
            reset.source = None
        self._linked_objs = []

    def get_panel(self):
        return pn.pane.HoloViews(**self._get_params())

    def _get_params(self):
        df = self.get_data()
        if self.streaming:
            from holoviews.streams import Pipe
            self._stream = Pipe(data=df)
        return dict(object=self.get_plot(df))

    def update(self, *events, invalidate_cache=True):
        """
        Triggers an update in the View.

        Parameters
        ----------
        events: tuple
            param events that may trigger an update.
        invalidate_cache : bool
            Whether to clear the View's cache.

        Returns
        -------
        stale : bool
            Whether the panel on the View is stale and needs to be
            rerendered.
        """
        # Skip events triggered by a parameter change on this View
        own_parameters = [self.param[p] for p in self.param]
        own_events = events and all(
            isinstance(e.obj, ParamFilter) and
            (e.obj.parameter in own_parameters or
            e.new is self._ls.selection_expr)
            for e in events
        )
        if own_events:
            return False
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

    page_size = param.Integer(default=20, bounds=(1, None), doc="""
        Number of rows to render per page, if pagination is enabled.""")

    view_type = 'table'

    _extension = 'tabulator'

    def get_panel(self):
        return pn.widgets.tables.Tabulator(**self._get_params())

    def _get_params(self):
        return dict(value=self.get_data(), disabled=True, page_size=self.page_size,
                    **self.kwargs)


class DownloadView(View):
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



class PerspectiveView(View):

    aggregates = param.Dict(None, allow_None=True, doc="""
        How to aggregate. For example {x: "distinct count"}""")

    columns = param.ListSelector(default=None, allow_None=True, doc="""
        A list of source columns to show as columns. For example ["x", "y"]""")

    computed_columns = param.ListSelector(default=None, allow_None=True, doc="""
        A list of computed columns. For example [""x"+"index""]""")

    column_pivots = param.ListSelector(None, allow_None=True, doc="""
        A list of source columns to pivot by. For example ["x", "y"]""")

    filters = param.List(default=None, allow_None=True, doc="""
        How to filter. For example [["x", "<", 3],["y", "contains", "abc"]]""")

    row_pivots = param.ListSelector(default=None, allow_None=True, doc="""
        A list of source columns to group by. For example ["x", "y"]""")

    selectable = param.Boolean(default=True, allow_None=True, doc="""
        Whether items are selectable.""")

    sort = param.List(default=None, doc="""
        How to sort. For example[["x","desc"]]""")

    plugin = param.ObjectSelector(default=_PerspectivePlugin.GRID.value, objects=_PerspectivePlugin.options(), doc="""
        The name of a plugin to display the data. For example hypergrid or d3_xy_scatter.""")

    theme = param.ObjectSelector(default='material', objects=_PERSPECTIVE_THEMES, doc="""
        The style of the PerspectiveViewer. For example material-dark""")

    view_type = 'perspective'

    _extension = 'perspective'

    _field_params = ['columns', 'computed_columns', 'column_pivots', 'row_pivots']

    def _get_params(self):
        df = self.get_data()
        param_values = dict(self.param.get_param_values())
        params = set(View.param) ^ set(PerspectiveView.param)
        kwargs = dict({p: param_values[p] for p in params}, **self.kwargs)
        return dict(object=df, toggle_config=False, **kwargs)

    def get_panel(self):
        return pn.pane.Perspective(**self._get_params())


class AltairView(View):

    chart = param.Dict(default={}, doc="Keyword argument for Chart.")

    x = param.Selector(doc="The column to render on the x-axis.")

    y = param.Selector(doc="The column to render on the y-axis.")

    marker = param.Selector(default='line', objects=[
        'area', 'bar', 'boxplot', 'circle', 'errorband', 'errorbar',
        'geoshape', 'image', 'line', 'point', 'rect', 'rule', 'square',
        'text', 'tick', 'trail'])

    encode = param.Dict(default={}, doc="Keyword arguments for encode.")

    mark = param.Dict(default={}, doc="Keyword arguments for mark.")

    transform = param.Dict(doc="""
        Keyword arguments for transforms, nested by the type of
        transform, e.g. {'bin': {'as_': 'binned', 'field': 'x'}}.""")

    project = param.Dict(doc="Keyword arguments for project.")

    properties = param.Dict(doc="Keyword arguments for properties.")

    view_type = 'altair'

    _extension = 'vega'

    def _transform_encoding(self, encoding, value):
        import altair as alt
        if isinstance(value, dict):
            value = dict(value)
            for kw, val in value.items():
                if kw == 'scale':
                    if isinstance(val, list):
                        val = alt.Scale(range=val)
                    else:
                        val = alt.Scale(**val)
                if kw == 'tooltip':
                    val = [alt.Tooltip(**v) for v in val]
                value[kw] = val
            value = getattr(alt, encoding.capitalize())(**value)
        return value

    def _get_params(self):
        import altair as alt
        df = self.get_data()
        chart = alt.Chart(df, **self.chart)
        mark = getattr(chart, f'mark_{self.marker}')(**self.mark)
        x = self._transform_encoding('x', self.x)
        y = self._transform_encoding('y', self.y)
        encode = {k: self._transform_encoding(k, v) for k, v in self.encode.items()}
        encoded = mark.encode(x=x, y=y, **encode)
        if self.transform:
            for key, kwargs in self.transform.items():
                encoded = getattr(encoded, f'transform_{key}')(**kwargs)
        if self.project:
            encoded = encoded.project(**self.project)
        if self.properties:
            encoded = encoded.properties(**self.properties)
        return dict(object=encoded, **self.kwargs)

    def get_panel(self):
        return pn.pane.Vega(**self._get_params())
