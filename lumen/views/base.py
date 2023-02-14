"""
The View classes render the data returned by a Pipeline as a Panel
object.
"""
from __future__ import annotations

import sys

from io import BytesIO, StringIO
from typing import (
    IO, TYPE_CHECKING, Any, ClassVar, Dict, List,
)
from weakref import WeakKeyDictionary

import numpy as np
import panel as pn
import param  # type: ignore

from bokeh.models import NumeralTickFormatter  # type: ignore
from panel.pane.base import PaneBase
from panel.pane.perspective import (
    THEMES as _PERSPECTIVE_THEMES, Plugin as _PerspectivePlugin,
)
from panel.param import Param
from panel.viewable import Viewable, Viewer

from ..base import MultiTypeComponent
from ..config import _INDICATORS
from ..downloads import Download
from ..filters.base import Filter, ParamFilter
from ..pipeline import Pipeline
from ..state import state
from ..transforms.base import Transform
from ..transforms.sql import SQLTransform
from ..util import (
    VARIABLE_RE, catch_and_notify, immediate_dispatch, is_ref,
    resolve_module_reference,
)
from ..validation import ValidationError

if TYPE_CHECKING:
    from bokeh.document import Document  # type: ignore
    from holoviews.selection import link_selections  # type: ignore

DOWNLOAD_FORMATS = ['csv', 'xlsx', 'json', 'parquet']


class View(MultiTypeComponent, Viewer):
    """
    `View` components provide a visual representation for the data returned
    by a :class:`lumen.source.base.Source` or :class:`lumen.pipeline.Pipeline`.

    The `View` must return a Panel object or an object that can be
    rendered by Panel. The base class provides methods which query the
    the provided :class:`lumen.pipeline.Pipeline`.

    Subclasses should use these methods to query the data and return
    a Viewable Panel object in the `get_panel` method.
    """

    controls = param.List(default=[], doc="""
        Parameters that should be exposed as widgets in the UI.""")

    download = param.ClassSelector(class_=Download, default=Download(), doc="""
        The download objects determines whether and how the source tables
        can be downloaded.""")

    loading_indicator = param.Boolean(default=True, constant=True, doc="""
        Whether to display a loading indicator on the View when the
        Pipeline is refreshing the data.""")

    limit = param.Integer(default=None, doc="""
        Limits the number of rows that are rendered.""")

    pipeline = param.ClassSelector(class_=Pipeline, doc="""
        The data pipeline that drives the View.""")

    rerender = param.Event(default=False, doc="""
        An event that is triggered whenever the View requests a re-render.""")

    selection_group = param.String(default=None, doc="""
        Declares a selection group the plot is part of. This feature
        requires the separate HoloViews library.""")

    title = param.String(default=None, doc="""
        The title of the view.""")

    field = param.Selector(doc="The field being visualized.")

    view_type: ClassVar[str | None] = None

    # Panel extension to load to render this View
    _extension: ClassVar[str | None] = None

    # Parameters which reference fields in the table
    _field_params: ClassVar[List[str]] = ['field']

    _requires_source: ClassVar[bool] = True

    _selections: ClassVar[WeakKeyDictionary[Document, link_selections]] = WeakKeyDictionary()

    _supports_selections: ClassVar[bool] = False

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
        pipeline = params.pop('pipeline', None)
        if pipeline is None:
            raise ValueError("Views must declare a Pipeline.")
        if isinstance(params.get("download"), str):
            *filenames, ext = params.get("download").split(".")
            filename = ".".join(filenames) or None
            params["download"] = Download(filename=filename, format=ext)
        fields = list(pipeline.schema)
        for fp in self._field_params:
            if isinstance(self.param[fp], param.Selector):
                self.param[fp].objects = fields
        pipeline.param.watch(self.update, 'data')
        if self.loading_indicator:
            pipeline._update_widget.param.watch(self._update_loading, 'loading')
        super().__init__(pipeline=pipeline, refs=refs, **params)
        self.param.watch(self.update, [p for p in self.param if p not in ('rerender', 'selection_expr', 'name')])
        self.download.view = self
        if self.selection_group:
            self._init_link_selections()
        self._initialized = False

    def __panel__(self) -> Viewable:
        if not self._initialized:
            self.update()
        return pn.panel(pn.bind(lambda e: self.panel, self.param.rerender))

    def _update_loading(self, event):
        if self._panel is not None:
            with immediate_dispatch():
                self._panel.loading = event.new

    def _update_ref(self, pname: str, ref: str, *events: param.parameterized.Event) -> None:
        # Note: Do not trigger update in View if Pipeline references
        # the same variable and is not set up to auto-update
        for event in events:
            if self.pipeline.auto_update:
                continue
            refs = []
            current = self.pipeline
            while current is not None:
                for ref in current.refs:
                    for subref in VARIABLE_RE.findall(ref):
                        if subref not in refs:
                            refs.append(subref)
                current = current.pipeline
            if any(ref == event.name for e in events for ref in refs):
                return
        super()._update_ref(pname, ref, *events)

    def _init_link_selections(self):
        doc = pn.state.curdoc or self.pipeline
        if self._ls is not None:
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

    def _update_selection_expr(self, event: param.parameterized.Event):
        self.selection_expr = event.new

    def __bool__(self) -> bool:
        return self._requires_source or self._cache is None or len(self._cache) > 0

    @catch_and_notify
    def _update_panel(self, *events: param.parameterized.Event):
        """
        Updates the cached Panel object and returns a boolean value
        indicating whether a rerender is required.
        """
        self._sync_refs(trigger=False)
        if self._panel is not None:
            self._cleanup()
            try:
                self._stream()
                return False
            except NotImplementedError:
                updates = self._get_params()
                if updates is not None:
                    self._panel.param.set_param(**updates)
                    return False
        self._panel = self.get_panel()
        return True

    @classmethod
    def _validate_filters(cls, *args, **kwargs):
        return cls._validate_list_subtypes('filters', Filter, *args, **kwargs)

    @classmethod
    def _validate_transforms(cls, *args, **kwargs):
        return cls._validate_list_subtypes('transforms', Transform, *args, **kwargs)

    @classmethod
    def _validate_sql_transforms(cls, *args, **kwargs):
        return cls._validate_list_subtypes('sql_transforms', SQLTransform, *args, **kwargs)

    @classmethod
    def _validate_pipeline(cls, *args, **kwargs):
        return cls._validate_str_or_spec('pipeline', Pipeline, *args, **kwargs)

    ##################################################################
    # Public API
    ##################################################################

    @classmethod
    def from_spec(
        cls, spec: Dict[str, Any] | str, source=None, filters=None, pipeline=None
    ) -> 'View':
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
        pipeline: lumen.pipeline.Pipeline
            The Lumen pipeline driving this View. Must not be supplied
            if the spec contains a pipeline definition or reference.

        Returns
        -------
        The resolved View object.
        """
        if isinstance(spec, str):
            raise ValueError(
                "View cannot be materialized by reference. Please pass "
                "full specification for the View."
            )
        spec = spec.copy()
        resolved_spec, refs = {}, {}

        # Resolve pipeline
        if 'pipeline' in spec:
            if pipeline is not None:
                raise ValueError(
                    "Either specify the pipeline as part of the specification "
                    "or pass it in explicitly, not both."
                )
            pipeline = spec['pipeline']
            if isinstance(pipeline, str):
                resolved_spec['pipeline'] = state.pipelines[pipeline]
            else:
                resolved_spec['pipeline'] = Pipeline.from_spec(pipeline)
        else:
            overrides = {
                p: spec.pop(p) for p in Pipeline.param if p != 'name' and p in spec
            }
            for ts in ('transforms', 'sql_transforms'):
                if ts in overrides:
                    overrides[ts] = [Transform.from_spec(t) for t in overrides[ts]]
            if pipeline is None:
                if isinstance(source, str):
                    source = state.sources[source]
                pipeline = Pipeline(source=source, **overrides)
            elif 'table' in overrides and len(overrides) == 1:
                if pipeline.table != overrides['table']:
                    raise ValidationError(
                        'Table declared on view does not match table declared '
                        'on pipeline.', spec, 'table'
                    )
            elif overrides:
                pipeline = pipeline.chain(
                    filters=overrides.get('filters', []),
                    transforms=overrides.get('transforms', []),
                    sql_transforms=overrides.get('sql_transforms', []),
                    _chain_update=True
                )
            resolved_spec['pipeline'] = pipeline

        # Resolve View parameters
        view_type = View._get_type(spec.pop('type', None))
        for p, value in spec.items():
            if p in resolved_spec:
                continue
            elif p not in view_type.param:
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
            *filenames, ext = download_spec.split('.')
            filename = '.'.join(filenames) or None
            download_spec = {'filename': filename, 'format': ext}
        resolved_spec['download'] = Download.from_spec(download_spec)

        view = view_type(refs=refs, **resolved_spec)

        if filters is None:
            filters = view.pipeline.traverse('filters')
            for pipeline in state.pipelines.values():
                for filt in pipeline.traverse('filters'):
                    if filt not in filters:
                        filters.append(filt)

        # Resolve ParamFilter parameters
        for filt in (filters or []):
            if isinstance(filt, ParamFilter):
                if not isinstance(filt.parameter, str):
                    continue
                name, parameter = filt.parameter.split('.')
                if name == view.name and parameter in view.param:
                    filt.parameter = view.param[parameter]
        return view

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
        if self.pipeline.data is None:
            self.pipeline._update_data()
        self._cache = data = self.pipeline.data
        if self.limit is not None:
            data = data.iloc[:self.limit]
        return data.copy()

    def get_value(self, field: str | None = None):
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
        if not len(data) or (field is not None and field not in data.columns):
            return None
        row = data.iloc[-1]
        return row[self.field if field is None else field]

    def to_spec(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Exports the full specification to reconstruct this component.

        Parameters
        ----------
        context: Dict[str, Any]
          Context contains the specification of all previously serialized components,
          e.g. to allow resolving of references.

        Returns
        -------
        Declarative specification of this component.
        """
        spec = super().to_spec(context)
        spec.update(self.kwargs)
        if context is None:
            return spec
        for name, pipeline in context.get('pipelines', {}).items():
            if spec.get('pipeline') == pipeline:
                spec['pipeline'] = name
                return spec
        if 'pipeline' in spec:
            if 'pipelines' not in context:
                context['pipelines'] = {}
            context['pipelines'][self.pipeline.name] = spec['pipeline']
            spec['pipeline'] = self.pipeline.name
        return spec

    def update(self, *events: param.parameterized.Event, invalidate_cache: bool=True):
        """
        Triggers an update in the View.

        Parameters
        ----------
        events: tuple
            param events that may trigger an update.
        invalidate_cache : bool
            Whether to clear the View's cache.
        """
        if invalidate_cache:
            self._cache = None
        stale = self._update_panel()
        self._initialized = True
        if stale:
            self.param.trigger('rerender')

    @property
    def control_panel(self) -> Param:
        return Param(
            self.param, parameters=self.controls, sizing_mode='stretch_width'
        )

    @property
    def panel(self) -> Viewable:
        if not self._initialized:
            self.update()
        panel = self._panel
        if isinstance(panel, PaneBase):
            if len(panel.layout) == 1 and panel._unpack:
                panel = panel.layout[0]
            else:
                panel = panel._layout
        if self.title:
            title_pane = pn.pane.HTML(
                f'<h3 align="center", style="margin-top: 0; margin-bottom: 0;">{self.title}</h3>',
                sizing_mode="stretch_width"
            )
        if self.download and self.title:
            download_pane = pn.panel(self.download)
            download_pane.sizing_mode = "fixed"
            return pn.Column(
                pn.Row(title_pane, download_pane), panel, sizing_mode=panel.sizing_mode
            )
        elif self.download:
            return pn.Column(self.download, panel, sizing_mode="stretch_both")
        elif self.title:
            return pn.Column(title_pane, panel, sizing_mode=panel.sizing_mode)
        return panel

    @property
    def refs(self) -> List[str]:
        refs = super().refs
        for c in self.controls:
            if c not in refs:
                refs.append(c)
        return refs

    ##################################################################
    # Component subclassable API
    ##################################################################

    def get_panel(self) -> Viewable:
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

    def _get_params(self) -> Dict[str, Any] | None:
        """
        Returns a dictionary of parameter values that will update the
        parameters on the existing _panel object inplace.  If not
        implemented each update will re-render the entire object by
        calling get_panel and replacing the existing object.
        """
        return None

    def _stream(self):
        """
        Streams new data to the existing _panel object. Will only
        be called if streaming is enabled.
        """
        raise NotImplementedError('View does not implement streaming.')

    def _cleanup(self):
        """
        Implements any cleanup that has to be performed when the View
        is updated.
        """



class Panel(View):
    """
    `Panel` views provide a way to declaratively wrap a Panel component.

    The `Panel` View is a very general purpose view that allows
    expressing arbitrary Panel objects as a specification. The Panel
    specification may be arbitrarily nested making it possible to
    specify entire layouts.  Additionally the Panel specification also
    supports references, including standard source and variable
    references and a custom `$data` reference that inserts the current
    data of the `View`.

    ```
    type: panel
      spec:
       type: panel.layout.Column
       objects:
         - type: pn.pane.Markdown
           object: '# My custom title'
         - type: pn.pane.DataFrame
           object: $data
    ```
    """

    spec = param.Dict()

    view_type: ClassVar[str] = 'panel'

    _requires_source: ClassVar[bool] = False

    def _resolve_spec(self, spec):
        if not isinstance(spec, dict) or 'type' not in spec:
            return spec
        spec = self.spec.copy()
        ptype = resolve_module_reference(spec.pop('type'), Viewable)
        params = {}
        for p, v in spec.items():
            if isinstance(v, dict) and 'type' in v:
                v = self._resolve_spec(v)
            elif isinstance(v, list):
                v = [self._resolve_spec(sv) for sv in v]
            elif is_ref(v):
                if v == '$data':
                    v = self.get_data()
                else:
                    v = state.resolve_reference(v)
            params[p] = v
        return ptype(**params)

    def get_panel(self):
        return self._resolve_spec(self.spec)


class StringView(View):
    """
    `StringView` renders the latest value of the field as a HTML string.
    """

    font_size = param.String(default='24pt', doc="""
        The font size of the rendered field value.""")

    view_type: ClassVar[str] = 'string'

    def get_panel(self) -> pn.pane.HTML:
        return pn.pane.HTML(**self._get_params())

    def _get_params(self) -> Dict[str, Any]:
        value = self.get_value()
        params = dict(self.kwargs)
        if value is None:
            params['object'] = 'No info'
        else:
            params['object'] = f'<p style="font-size: {self.font_size}">{value}</p>'
        return params


class IndicatorView(View):
    """
    `IndicatorView` renders the latest field value as a Panel `Indicator`.
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

    def _get_params(self) -> Dict[str, Any]:
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
        import hvplot.pandas  # type: ignore # noqa
        if 'dask' in sys.modules:
            try:
                import hvplot.dask  # type: ignore # noqa
            except Exception:
                pass
        if 'by' in params and isinstance(params['by'], str):
            params['by'] = [params['by']]
        if 'groupby' in params and isinstance(params['groupby'], str):
            params['groupby'] = [params['groupby']]
        super().__init__(**params)


class hvPlotUIView(hvPlotBaseView):
    """
    `hvPlotUIView` displays provides a component for exploring datasets interactively.
    """

    view_type = 'hvplot_ui'

    def _get_args(self):
        from hvplot.ui import hvPlotExplorer  # type: ignore
        params = {
            k: v for k, v in self.param.values().items()
            if k in hvPlotExplorer.param and v is not None and k != 'name'
        }
        return (self.get_data(),), dict(params, **self.kwargs)

    def __panel__(self):
        panel = self.get_panel()
        def ui(*events):
            panel._data = self.get_data()
            panel._plot()
            return panel
        return pn.bind(ui, self.param.rerender)

    def get_panel(self):
        from hvplot.ui import hvDataFrameExplorer
        args, kwargs = self._get_args()
        return hvDataFrameExplorer(*args, **kwargs)


class hvPlotView(hvPlotBaseView):
    """
    `hvPlotView` renders the queried data as a bokeh plot generated with hvPlot.

    hvPlot allows for a concise but powerful declaration of a plot via
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

    _ignore_kwargs = ['tables']

    _supports_selections = True

    def __init__(self, **params):
        self._data_stream = None
        self._linked_objs = []
        super().__init__(**params)

    def get_plot(self, df):
        processed = {}
        for k, v in self.kwargs.items():
            if k in self._ignore_kwargs:
                continue
            if k.endswith('formatter') and isinstance(v, str) and '%' not in v:
                v = NumeralTickFormatter(format=v)
            processed[k] = v
        if self.streaming:
            processed['stream'] = self._data_stream
        plot = df.hvplot(
            kind=self.kind, x=self.x, y=self.y, by=self.by, groupby=self.groupby, **processed
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
        params = self._get_params()
        for s in ('width', 'height'):
            if f'frame_{s}' in self.kwargs:
                params[s] = self.kwargs[f'frame_{s}']
        return pn.pane.HoloViews(**params)

    def _get_params(self):
        df = self.get_data()
        if self.streaming:
            from holoviews.streams import Pipe  # type: ignore
            self._data_stream = Pipe(data=df)
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
            return
        if invalidate_cache:
            self._cache = None
        if not self.streaming or self._data_stream is None:
            stale = self._update_panel()
            if stale:
                self.param.trigger('rerender')
        else:
            self._data_stream.send(self.get_data())


class Table(View):
    """
    `Table` renders data using the powerful Panel `Tabulator` component.

    See https://panel.holoviz.org/reference/widgets/Tabulator.html
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
    `DownloadView` renders a button that allows downloading data as CSV, Excel, and parquet files.
    """

    filename = param.String(default='data', doc="""
      Filename of the downloaded file.""")

    format = param.ObjectSelector(default=None, objects=DOWNLOAD_FORMATS, doc="""
      The format to download the data in.""")

    kwargs = param.Dict(default={}, doc="""
      Keyword arguments passed to the serialization function, e.g.
      data.to_csv(file_obj, **kwargs).""")

    view_type = 'download'

    _required_keys = ["format"]

    def __bool__(self) -> bool:
        return True

    @catch_and_notify("Download failed")
    def _table_data(self) -> IO[Any]:
        io: IO[Any]
        if self.format in ('json', 'csv'):
            io = StringIO()
        else:
            io = BytesIO()
        data = self.get_data()
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

    def get_panel(self) -> pn.widgets.FileDownload:
        return pn.widgets.FileDownload(**self._get_params())

    def _get_params(self) -> Dict[str, Any]:
        filename = f'{self.filename}.{self.format}'
        return dict(filename=filename, callback=self._table_data, **self.kwargs)



class PerspectiveView(View):
    """
    `PerspectiveView` renders data into a Perspective widget.

    See https://panel.holoviz.org/reference/panes/Perspective.html for more details.
    """

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

    def _get_params(self) -> Dict[str, Any]:
        df = self.get_data()
        param_values = dict(self.param.get_param_values())
        params = set(View.param) ^ set(PerspectiveView.param)
        kwargs = dict({p: param_values[p] for p in params}, **self.kwargs)
        return dict(object=df, toggle_config=False, **kwargs)

    def get_panel(self) -> pn.pane.Perspective:
        return pn.pane.Perspective(**self._get_params())


class AltairView(View):
    """
    `AltairView` provides a declarative way to render Altair charts.
    """

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

    def _transform_encoding(self, encoding: str, value: Any) -> Any:
        import altair as alt  # type: ignore
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

    def _get_params(self) -> Dict[str, Any]:
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

    def get_panel(self) -> pn.pane.Vega:
        return pn.pane.Vega(**self._get_params())


__all__ = [name for name, obj in locals().items() if isinstance(obj, type) and issubclass(obj, View)] + ["Download"]
