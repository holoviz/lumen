"""
The hvPlot views render a Pipeline's data with hvPlot, either as a plot or as
hvPlot's own explorer UI.

The explorer's controls carry the type, default and bounds of every option they
expose, but roughly half of them document nothing, and a param's ``doc`` is what
the plot agent's model turns into the description the LLM reads. The prose for
those options exists, in ``HoloViewsConverter``'s class docstring, so it is read
from there rather than restated here.
"""
from __future__ import annotations

import copy
import inspect
import logging
import re
import sys

from functools import cache
from typing import ClassVar

import hvplot.pandas  # type: ignore  # noqa: F401
import pandas as pd
import panel as pn
import param  # type: ignore

from bokeh.models import NumeralTickFormatter  # type: ignore
from griffe import Docstring, DocstringSectionKind  # type: ignore
from holoviews import Overlay  # type: ignore
from holoviews.plotting.util import process_cmap  # type: ignore
from holoviews.streams import Pipe  # type: ignore
from hvplot import hvPlotTabular  # type: ignore
from hvplot.converter import HoloViewsConverter  # type: ignore
from hvplot.ui import (  # type: ignore
    Axes, Colormapping, Geographic, Labels, Operations, Style,
    hvDataFrameExplorer, hvGridExplorer, hvPlotExplorer,
)
from panel.util import classproperty

from ..filters.base import ParamFilter
from ..util import is_geodataframe, try_import_xarray, widen_nullable
from .base import View

# Without this, `from .hvplot import *` in the package __init__ would rebind the
# name hvplot to the library imported above, shadowing this module.
__all__ = [
    "AGGREGATORS",
    "GRIDDED_KINDS",
    "GRIDDED_MAX_CELLS",
    "HVPLOT_KINDS",
    "HVPLOT_PASSTHROUGH_PARAMS",
    "HVPLOT_STYLE_PARAMS",
    "MAX_RENDER_ROWS",
    "REDUCING_KINDS",
    "VALUE_AGGREGATORS",
    "hvOverlayView",
    "hvPlotBaseView",
    "hvPlotUIView",
    "hvPlotView",
]

# Safety cap on a pivoted hvPlot grid. 10M cells is ~80MB for float64;
# anything larger is rejected rather than risking OOM.
GRIDDED_MAX_CELLS = 10_000_000

# hvPlot kinds we pivot long-form data into a 2D xarray grid for: the 2D
# scalar-field kinds that map cleanly from x/y/z columns.
GRIDDED_KINDS = ("contour", "contourf", "image", "quadmesh")

# deck.gl serialises every row into the browser as JSON; beyond a few hundred
# hvPlot draws one glyph per row for non-reducing kinds; past this many rows a
# browser tab can exhaust its memory (a gridded xarray source expands to millions
# of long-form rows), so hvPlotView/hvPlotUIView refuse to render such a frame.
MAX_RENDER_ROWS = 250_000

# Kinds whose rendered output size is bounded regardless of row count -- they
# pivot to a grid or aggregate -- so they are exempt from MAX_RENDER_ROWS.
REDUCING_KINDS = GRIDDED_KINDS + (
    "heatmap", "hexbin", "hist", "kde", "box", "violin", "bivariate",
)

# Keep the Selector in sync with tabular kinds that produce plots in hvPlot.
# The explorer is handled by hvPlotUIView, while dataset returns a bare
# hv.Dataset with no plotting class for hvPlotView to render.
HVPLOT_KINDS = [
    kind for kind in hvPlotTabular.__all__ if kind not in {"explorer", "dataset"}
] + list(GRIDDED_KINDS)

# The datashader reductions hvPlot's own explorer offers by name. count_cat is
# left out deliberately: hvPlot builds it from `by`, and naming it here only
# gets as far as a bare string that never becomes a categorical reduction.
AGGREGATORS = [None, "any", "count", "max", "mean", "min", "sum"]

# These reduce a value column rather than counting rows, so hvPlot needs to be
# told which column via `color`; without it datashader cannot pick a dimension.
VALUE_AGGREGATORS = ("max", "mean", "min", "sum")

# hvPlot groups its options under headings of its own ("Data Options",
# "Geographic Options", ...). griffe's numpy parser only treats a section as
# parameters when it is titled "Parameters", and reads everything else as an
# admonition, so the headings are renamed before it sees them.
_SECTION_UNDERLINE = re.compile(r'^-+$')

# Options sharing a description are documented on one line, e.g.
# "logx/logy : bool". griffe keeps the first name and discards the rest.
_GROUPED_OPTION = re.compile(r'^(\w+(?:/\w+)+)\s*:', re.MULTILINE)

# reStructuredText that is meaningful in rendered docs and noise in a prompt.
_ROLE = re.compile(r':\w+:`[^`]*?([^`:/]+)`')
_LITERAL = re.compile(r'``([^`]+)``')
_DIRECTIVE = re.compile(r'\.\.\s+\w+::')

# A sentence break, minus the abbreviations hvPlot's prose actually uses.
_SENTENCE = re.compile(r'(?<!\be\.g)(?<!\bi\.e)(?<!\betc)\.(?:\s|$)')

_MAX_DESCRIPTION = 200


def _as_parameters(doc: str) -> str:
    """Retitle hvPlot's option sections so griffe parses them as parameters."""
    lines = doc.split('\n')
    retitled: list[str] = []
    index = 0
    while index < len(lines) - 1:
        title, underline = lines[index], lines[index + 1]
        if title.strip() and not title.startswith(' ') and _SECTION_UNDERLINE.match(underline.strip()):
            retitled += ['Parameters', '-' * len('Parameters')]
            index += 2
            continue
        retitled.append(lines[index])
        index += 1
    return '\n'.join(retitled + lines[index:])


def _summarize(description: str) -> str:
    """Reduce a docstring entry to one plain sentence."""
    description = _DIRECTIVE.split(description)[0]
    description = _ROLE.sub(r'\1', description)
    description = _LITERAL.sub(r'\1', description)
    description = ' '.join(description.split())
    sentence = _SENTENCE.split(description)[0].strip(' ,;')
    if len(sentence) >= _MAX_DESCRIPTION:
        sentence = sentence[:_MAX_DESCRIPTION - 1].rsplit(' ', 1)[0]
    return f'{sentence}.' if sentence else ''


@cache
def hvplot_param_docs() -> dict[str, str]:
    """Map an hvPlot option name to a one-sentence description of it."""
    doc = inspect.cleandoc(HoloViewsConverter.__doc__ or '')
    # griffe reports every continuation line hvPlot indents by three spaces, and
    # every grouped entry it could not type. None of it is actionable here.
    griffe_log = logging.getLogger('griffe')
    level = griffe_log.level
    griffe_log.setLevel(logging.ERROR)
    try:
        sections = Docstring(_as_parameters(doc), lineno=1, parser='numpy').parse()
    finally:
        griffe_log.setLevel(level)
    docs = {
        parameter.name: summary
        for section in sections if section.kind is DocstringSectionKind.parameters
        for parameter in section.value
        if (summary := _summarize(parameter.description or ''))
    }
    if not docs:
        raise RuntimeError(
            'Could not read any option descriptions from hvPlot. Its converter '
            'docstring is laid out differently than expected.'
        )
    for match in _GROUPED_OPTION.finditer(doc):
        first, *rest = match.group(1).split('/')
        if not (summary := docs.get(first)):
            continue
        for name in rest:
            docs.setdefault(name, summary)
    return docs


# The controls hvPlot's own explorer groups its styling options under. They
# already declare each option's type, default and bounds, so those are copied
# rather than restated; only the description is usually missing, and hvPlot
# documents that on its converter instead.
_STYLE_CONTROLS = (Axes, Labels, Style, Colormapping)

# Back-reference to the explorer, not an option.
_SKIP_INTERNALS = ("name", "explorer")

# responsive, width and height default differently on the explorer than in
# hvPlot itself, so generating them would silently restyle every existing
# dashboard. All three are already set, by the plot agent and by get_panel.
_SKIP_SIZING = ("responsive", "width", "height")

# Options the explorer offers on its advanced panels, for someone already
# looking at a plot. Nothing in a prompt asks for them, so they would only
# spend prompt and give the LLM more to get wrong.
_SKIP_EXPLORER_ONLY = ("shared_axes", "rescale_discrete_levels", "symmetric")

# hvPlot offers 712 colormaps and a Selector becomes an enum in the plot
# agent's schema, which would cost more prompt than the rest of the view put
# together. These are the ones worth naming; hvPlot stays the authority on
# whether a name is real.
_PREFERRED_CMAPS = (
    "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu_r",
    "Blues", "Reds", "Greens", "fire", "kbc_r", "rainbow", "bmy", "gray",
)

# alpha is the one styling option hvPlot documents in neither place.
_MISSING_DOCS = {"alpha": "Opacity of the plotted glyphs, from 0 to 1."}


def _as_llm_param(parameter: param.Parameter, doc: str) -> param.Parameter:
    """Copy one of the explorer's options into a param the plot agent can offer."""
    parameter = copy.deepcopy(parameter)
    parameter.doc = doc
    if isinstance(parameter, param.Selector):
        # The objects are there to steer the LLM, not to narrow what a spec may
        # say: hvPlot reads a dict cmap as a color key and a False legend as no
        # legend, and both predate these params.
        parameter.check_on_set = False
    # Structured output makes the LLM answer with every field, so an option
    # keeping the explorer's default would be restated on every spec and
    # forwarded as if it had been asked for. None is the one value that means
    # untouched, leaving the default to hvPlot.
    parameter.default = None
    parameter.allow_None = True
    return parameter


def _declare_hvplot_style_params(view_type: type[View]) -> tuple[str, ...]:
    """Declare hvPlot's styling options on a view, and name the ones declared.

    An undeclared keyword already reaches hvPlot through kwargs; declaring one
    is what puts it in the schema the plot agent hands the LLM, which is
    otherwise limited to the axes and cannot style a plot at all.
    """
    docs = hvplot_param_docs()
    skip = set(view_type.param).union(
        _SKIP_INTERNALS, _SKIP_SIZING, _SKIP_EXPLORER_ONLY
    )
    declared = []
    for control in _STYLE_CONTROLS:
        for name, parameter in control.param.objects().items():
            if name in skip:
                continue
            doc = parameter.doc or docs.get(name) or _MISSING_DOCS.get(name)
            if not doc:
                continue
            view_type.param.add_parameter(name, _as_llm_param(parameter, doc))
            skip.add(name)
            declared.append(name)
    cmap = view_type.param.cmap
    cmap.objects = [c for c in _PREFERRED_CMAPS if c in cmap.objects]
    return tuple(declared)


class hvPlotBaseView(View):

    kind = param.Selector(
        default=None, doc="The kind of plot, e.g. 'scatter' or 'line'.",
        objects=HVPLOT_KINDS
    )

    x = param.Selector(doc="The column to render on the x-axis.")

    y = param.Selector(doc="The column to render on the y-axis.")

    aggregator = param.Selector(default=None, objects=AGGREGATORS, doc="""
        How datashader reduces the rows landing in one pixel, e.g. 'mean' to
        shade by an average rather than a row count. Only meaningful with
        datashade or rasterize; all but 'count' and 'any' reduce a value
        column, which is named with `color`.""")

    by = param.ListSelector(doc="The column(s) to facet the plot by.")

    color_key = param.Dict(default=None, doc="""
        Mapping of the values in `by` to explicit colors, e.g.
        {'Irish': '#e41a1c', 'Italian': '#377eb8'}. Only meaningful with
        datashade; without it datashader picks a categorical palette.""")

    datashade = param.Boolean(default=False, doc="""
        Aggregate the data server-side with datashader and send an image
        instead of one glyph per row. Combined with `by` this blends the
        categories present in each pixel, rather than overplotting them.""")

    dynspread = param.Boolean(default=False, doc="""
        Grow isolated points so sparse regions stay visible after
        datashading. Has no effect unless datashade is enabled.""")

    groupby = param.ListSelector(doc="The column(s) to group by.")

    z = param.Selector(doc="""
        Column of z-values for gridded plot kinds (image, quadmesh, heatmap, contourf).
        Internally mapped to hvPlot's C= for kind='heatmap' and z= for other kinds.""")

    geo = param.Boolean(
        default=False, doc="Toggle True if the plot is on a geographic map."
    )

    _field_params = ['x', 'y', 'by', 'groupby', 'z']

    __abstract = True

    def __init__(self, **params):
        if 'dask' in sys.modules:
            try:
                # Deferred: registers hvPlot's dask accessor, and dask is optional.
                import hvplot.dask  # type: ignore  # noqa: F401, PLC0415
            except Exception:
                pass
        for key in ('by', 'groupby'):
            if key in params:
                params[key] = self._as_column_list(params[key])
        if params.get("geo") and params.get("kind") in (None, "scatter"):
            params["kind"] = "points"
        super().__init__(**params)

    @staticmethod
    def _as_column_list(value):
        """Accept a bare column name wherever a list of them is expected."""
        return [value] if isinstance(value, str) else value

    @classmethod
    def _validate_by(cls, value, spec, context):
        # Spec validation runs before __init__, so without this a spec saying
        # `by: family` is rejected by the ListSelector before the coercion
        # above ever gets to see it.
        return cls._as_column_list(value)

    @classmethod
    def _validate_groupby(cls, value, spec, context):
        return cls._as_column_list(value)

    @classproperty
    def _valid_keys_(cls):
        return None

    def _complete_color_key(self, df):
        """Fill in a partial ``color_key`` from the categorical palette.

        Not named ``_resolve_color_key``: from_spec treats ``_resolve_<param>``
        as a spec resolver and would call this with the raw spec value.

        Datashader needs a color for every category present, but naming the few
        that matter and leaving the rest is the natural way to ask for one, so
        the remainder are filled in rather than raising.
        """
        if self.color_key is None or not self.by or not isinstance(df, pd.DataFrame):
            return self.color_key
        column = df[self.by[0]]
        categories = list(
            column.cat.categories if isinstance(column.dtype, pd.CategoricalDtype)
            else pd.unique(column)
        )
        missing = [c for c in categories if c not in self.color_key]
        if not missing:
            return self.color_key
        # glasbey_hv carries 256 distinct hues; a Category palette repeats
        # after 10 or 20 and would hand two categories the same color.
        chosen = set(self.color_key.values())
        spare = [c for c in process_cmap('glasbey_hv', categorical=True) if c not in chosen]
        return dict(self.color_key, **dict(zip(missing, spare, strict=False)))

    def _check_aggregator(self, plot_kwargs) -> None:
        """Refuse an aggregator that has nothing to reduce.

        Left to hvPlot this surfaces from inside the datashader operation as
        "Could not determine dimension to apply 'aggregate' operation to",
        which says nothing about the spec that caused it.
        """
        if self.aggregator not in VALUE_AGGREGATORS:
            return
        if not (self.datashade or plot_kwargs.get('rasterize')):
            raise ValueError(
                f"aggregator={self.aggregator!r} only applies when the data is "
                "aggregated server-side; set datashade or rasterize."
            )
        if not (plot_kwargs.get('c') or plot_kwargs.get('color')):
            raise ValueError(
                f"aggregator={self.aggregator!r} reduces a value column, so one "
                "must be named with color; use 'count' or 'any' to reduce rows."
            )

    def get_data(self):
        # Every hvPlot kind can reach datashader, through rasterize/datashade
        # or an operation, and datashader rejects the pandas nullable dtypes
        # the pipeline now preserves. Plots take the numpy widening instead;
        # tables and downloads keep the nullable columns.
        return widen_nullable(super().get_data())

    def _check_render_size(self, df) -> None:
        """Refuse to render more per-row glyphs than a browser tab can hold.

        Kinds that pivot to a grid or aggregate (``REDUCING_KINDS``) bound their
        output regardless of row count, as does server-side ``rasterize``/
        ``datashade``; every other kind draws one glyph per row and can exhaust
        browser memory on a large frame (e.g. a gridded xarray source expanded
        to millions of long-form rows).
        """
        if not isinstance(df, pd.DataFrame):
            return
        n = len(df)
        if n <= MAX_RENDER_ROWS or self.kind in REDUCING_KINDS:
            return
        if self.datashade or self.kwargs.get('rasterize'):
            return
        raise ValueError(
            f"Cannot render {n:,} rows as kind={self.kind!r}: each row becomes a "
            f"glyph in the browser and would exhaust its memory. Reduce the "
            f"pipeline to at most {MAX_RENDER_ROWS:,} rows (e.g. a SQL LIMIT or "
            f"aggregation), use a gridded/aggregating kind, or set rasterize=True."
        )

    def _source_dataset(self):
        """The compact gridded ``xarray.Dataset`` from the pipeline's source
        (xarray-sql ``to_dataset``), or None when the source can't produce one."""
        return self.pipeline.get_dataset() if self.pipeline is not None else None


HVPLOT_STYLE_PARAMS = _declare_hvplot_style_params(hvPlotBaseView)

# Declared params hvPlot takes verbatim, so hvPlotView can forward them without
# knowing what any of them mean.
HVPLOT_PASSTHROUGH_PARAMS = HVPLOT_STYLE_PARAMS + ("datashade", "dynspread")


class hvPlotUIView(hvPlotBaseView):
    """
    `hvPlotUIView` displays provides a component for exploring datasets interactively.
    """

    view_type = 'hvplot_ui'

    def _get_args(self, explorer_cls=None, data=None):
        if explorer_cls is None:
            explorer_cls = hvPlotExplorer
        if data is None:
            data = self.get_data()
        # The explorer keeps styling, colormapping and datashading on nested
        # controls, so a param is only forwarded if one of them claims it;
        # anything else is rejected by hvPlotExplorer.__init__.
        controls = (explorer_cls.param, Geographic.param, Operations.param) + tuple(
            control.param for control in _STYLE_CONTROLS
        )
        # title is left out because every View renders its own above the panel,
        # and the Labels control would draw it a second time inside the plot.
        params = {
            k: v for k, v in self.param.values().items()
            if any(k in control for control in controls)
            and v is not None and k not in ('name', 'title')
        }
        # Only completed once a control has claimed it above: hvPlot gained the
        # color_key control after this was written, and forcing the keyword in
        # regardless makes hvPlotExplorer.__init__ reject it outright on an
        # older hvPlot rather than simply coloring from the default palette.
        if 'color_key' in params:
            params['color_key'] = self._complete_color_key(data)
        self._check_aggregator(self.kwargs)
        return (data,), dict(params, **self.kwargs)

    def __panel__(self):
        panel = self.get_panel()
        def ui(*events):
            gridded = self._source_dataset()
            panel._data = gridded if gridded is not None else self.get_data()
            panel._plot()
            return panel
        return pn.bind(ui, self.param.rerender)

    def get_panel(self):
        # An xarray-backed pipeline explores the compact gridded Dataset (via
        # to_dataset) with hvPlot's grid explorer, so gridded kinds like image
        # and quadmesh work; tabular data uses the dataframe explorer.
        gridded = self._source_dataset()
        if gridded is not None:
            # Deferred: registers hvPlot's xarray accessor, and xarray is optional.
            import hvplot.xarray  # type: ignore  # noqa: F401, PLC0415
            args, kwargs = self._get_args(hvGridExplorer, gridded)
            return hvGridExplorer(*args, **kwargs)
        args, kwargs = self._get_args()
        self._check_render_size(args[0])
        return hvDataFrameExplorer(*args, **kwargs)


class hvPlotView(hvPlotBaseView):
    """
    `hvPlotView` renders the queried data as a bokeh plot generated with hvPlot.

    hvPlot allows for a concise but powerful declaration of a plot via
    its simple API.
    """

    operations = param.List(item_type=param.ParameterizedFunction, doc="""
        Operations to apply to HoloViews plot.""")

    opts = param.Dict(default={}, doc="HoloViews options to apply on the plot.")

    streaming = param.Boolean(default=False, doc="""
        Whether to stream new data to the plot or rerender the plot.""")

    selection_expr = param.Parameter(doc="""
        A selection expression caputirng the current selection applied
        on the plot.""")

    view_type = 'hvplot'

    _ignore_kwargs = ['tables']

    _panel_type = pn.pane.HoloViews

    _supports_selections = True

    def __init__(self, **params):
        self._data_stream = None
        self._linked_objs = []
        super().__init__(**params)

    def _gridded_index(self) -> list[str]:
        """Columns that index the pivoted grid: any groupby axes (e.g. time)
        that hvPlot pages through a widget, then the y/x grid axes."""
        return [*(self.groupby or []), self.y, self.x]

    def _gridded_pivot_blocker(self, df) -> str | None:
        """Return None if df can be pivoted to a grid, else a human-readable reason."""
        if try_import_xarray() is None:
            return "xarray is not installed"
        if not (self.x and self.y and self.z):
            return "x, y, and z must all be set for gridded plot kinds"
        index = self._gridded_index()
        missing = {*index, self.z} - set(df.columns)
        if missing:
            return f"missing required column(s): {sorted(missing)}"
        if df.duplicated(subset=index).any():
            return f"duplicate {tuple(index)} rows prevent pivot to a grid"
        n_cells = 1
        for col in index:
            n_cells *= df[col].nunique()
        if n_cells > GRIDDED_MAX_CELLS:
            return (
                f"pivoted grid would have {n_cells:,} cells, exceeding the "
                f"{GRIDDED_MAX_CELLS:,} safety cap"
            )
        return None

    def _to_gridded(self, df):
        """Pivot a long-form DataFrame to an xarray DataArray for hvPlot's
        quadmesh/image/contourf kinds.

        The y/x columns form the 2D grid; any ``groupby`` columns become extra
        axes hvPlot pages through with a widget. Returns the input unchanged if
        it is already an xarray object or if the pivot is blocked (see
        ``_gridded_pivot_blocker``); callers that require gridded data should
        check the blocker themselves and raise.
        """
        if isinstance(df, pd.DataFrame) and self._gridded_pivot_blocker(df) is not None:
            return df
        # Reached only when xarray is available (an xarray object passed
        # through, or a pivotable DataFrame). Register hvPlot's xarray accessor
        # so .hvplot works on the returned xarray object.
        import hvplot.xarray  # type: ignore  # noqa: F401, PLC0415
        if not isinstance(df, pd.DataFrame):
            return df
        return df.set_index(self._gridded_index())[self.z].to_xarray()

    def _gridded_plot_source(self, df):
        """Object to hand hvPlot for a gridded kind: a compact xarray Dataset
        straight from the source (xarray-sql ``to_dataset``) when available,
        else the long-form frame pivoted to xarray."""
        gridded = self._source_dataset()
        if gridded is not None:
            # Deferred: registers hvPlot's xarray accessor, and xarray is optional.
            import hvplot.xarray  # type: ignore  # noqa: F401, PLC0415
            return gridded
        if isinstance(df, pd.DataFrame):
            blocker = self._gridded_pivot_blocker(df)
            if blocker is not None:
                raise ValueError(
                    f"Cannot render kind={self.kind!r} from this pipeline: "
                    f"{blocker}. Either provide xarray-backed data, switch "
                    f"to a tabular kind (e.g. 'heatmap' for ordinal axes), "
                    f"or fix the spec."
                )
        return self._to_gridded(df)

    def get_plot(self, df):
        self._check_render_size(df)
        processed = {}
        for k, v in self.kwargs.items():
            if k in self._ignore_kwargs:
                continue
            if k.endswith('formatter') and isinstance(v, str) and '%' not in v:
                v = NumeralTickFormatter(format=v)
            processed[k] = v
        if self.streaming:
            processed['stream'] = self._data_stream
        if self.z is not None:
            processed['C' if self.kind == 'heatmap' else 'z'] = self.z
        # Params are stripped out of kwargs by View.__init__, so anything hvPlot
        # needs has to be put back explicitly. These pass straight through; the
        # ones below carry a Lumen meaning hvPlot does not share.
        values = self.param.values()
        processed.update({
            name: values[name] for name in HVPLOT_PASSTHROUGH_PARAMS
            if values[name] != self.param[name].default
        })
        if self.color_key is not None:
            processed['color_key'] = self._complete_color_key(df)
        if self.aggregator is not None:
            self._check_aggregator(processed)
            processed['aggregator'] = self.aggregator

        kind = self.kind
        plot_source = df
        if is_geodataframe(df):
            # hvplot infers the geometry kind (polygons/paths/points); just
            # clear a non-geometry default so it isn't forced to scatter/points
            if kind in (None, 'scatter', 'points'):
                kind = None
            processed['geo'] = self.geo
        elif kind in GRIDDED_KINDS:
            plot_source = self._gridded_plot_source(df)

        plot = plot_source.hvplot(
            kind=kind, x=self.x, y=self.y, by=self.by, groupby=self.groupby, **processed
        )
        if self.operations:
            for operation in self.operations:
                plot = operation(plot)
        plot = plot.opts(**self.opts) if self.opts else plot
        if self.selection_group or 'selection_expr' in self.param.watchers:
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
        return self._panel_type(**params)

    def _get_params(self):
        df = self.get_data()
        if self.streaming:
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


class hvOverlayView(View):
    """
    `hvOverlayView` allows overlaying a list of layers consisting of
    `hvPlotView` components.
    """

    layers = param.List(item_type=hvPlotView)

    view_type = 'hv_overlay'

    _panel_type = pn.pane.HoloViews

    _requires_source: ClassVar[bool] = False

    _supports_selections = True

    def _get_params(self):
        overlay = Overlay([layer.get_plot(layer.get_data()) for layer in self.layers])
        return dict(object=overlay)

    def get_panel(self):
        params = self._get_params()
        return self._panel_type(**params)
