from __future__ import annotations

import datetime as dt
import warnings

from functools import partial
from io import BytesIO, StringIO
from itertools import product
from typing import (
    Any, ClassVar, Dict, List, Tuple,
)

import panel as pn
import param  # type: ignore

from panel.util import PARAM_NAME_PATTERN
from panel.viewable import Layoutable, Viewable, Viewer
from param import edit_constant
from typing_extensions import Literal

from .base import Component
from .config import _LAYOUTS
from .filters.base import FacetFilter, Filter
from .panel import IconButton
from .pipeline import Pipeline
from .sources.base import Source
from .state import state
from .util import catch_and_notify, extract_refs
from .validation import ValidationError, match_suggestion_message
from .views.base import DOWNLOAD_FORMATS, View


def resolve_layout_spec(layout_spec, views, layout_cls=pn.Column, title=None):
    """
    Recursively resolves a layout specification provided as a list-of-lists.

    The function will recurse over the layout specification and
    resolve integer and string indexes into the views of a Layout and
    treat lists as an additonal layer of nesting. The outermost list
    corresponds to a Column layout and each layer of nesting will
    alternate between a Row and a Column.

    As an example the following will resolve to a Column of two Rows,
    where the first row contains one item and the second row contains
    two items:

        [[0], [1, 2]]

    Since the resolution happens recursively the layout may be
    arbitrarily nested, e.g. this example will contain a Column where
    the first item is a Row containing one item and another Column:

        [[0, [1, 2]], [3, 4]]

    This makes it possible to create complex layouts using a simple
    specification.

    Since views can be declared as a list or a dictionary the views
    can be referenced with string or integer indexes.

    Arguments
    ---------
    layout_spec: list[int | str | list]
        The nested layout specification containing references to the views container.
    views: dict[str, View] | list[View]
        The views to be laid out using the layout_spec
    layout_cls: Type[Column] | Type[Row]
        The Panel layout class to nested the views in.
    title: str
       The name of the Layout.

    Returns
    -------
    The layout resolved from the layout specification.
    """
    mode = 'stretch_both' if layout_cls is pn.Column else 'stretch_width'
    layout = layout_cls(sizing_mode=mode)
    view_size = len(views)
    for spec in layout_spec:
        view = None
        if isinstance(spec, int):
            if isinstance(views, dict):
                raise KeyError(
                    f'Layout specification for {title!r} layout used '
                    f'integer index ({spec}) but views were declared as a '
                    'dictionary.'
                )
            elif spec >= view_size:
                raise IndexError(
                    f'Layout specification for {title!r} layout references '
                    f'out-of-bounds index ({spec}) even though the maximum '
                    f'available index is {view_size - 1}.'
                )
            view = views[spec]
        elif isinstance(spec, str):
            if isinstance(views, dict):
                if spec not in views:
                    raise KeyError(
                        f'Layout specification for {title} layout references '
                        'unknown view {spec!r}.'
                    )
                view = views[spec]
            else:
                matches = [view for view in views if view.name == spec]
                if matches:
                    view = matches[0]
                else:
                    raise KeyError(
                        f"Layout could not find named view '{spec}'."
                    )
        if view is not None:
            layout.append(view.panel)
            continue

        if isinstance(spec, list):
            inner_cls = pn.Row if layout_cls is pn.Column else pn.Column
            inner_layout = resolve_layout_spec(
                spec, views, layout_cls=inner_cls, title=title
            )
            layout.append(inner_layout)
        else:
            raise ValidationError(
                'Could not resolve layout specification, found unknown '
                'layout spec', layout_spec, spec
            )
    fixed_width = all(p.sizing_mode in (None, 'fixed') and p.width for p in layout)
    fixed_height = all(p.sizing_mode in (None, 'fixed') and p.height for p in layout)
    if fixed_height and fixed_width:
        layout.sizing_mode = 'fixed'
    elif fixed_width:
        layout.sizing_mode = 'fixed' if layout.sizing_mode == 'stretch_width' else 'stretch_height'
    elif fixed_height:
        layout.sizing_mode = 'stretch_width'
    return layout


class Card(Viewer):
    """
    A Card renders a layout of multiple views given a layout specification.
    """

    layout = param.ClassSelector(default='column', class_=(str, list, dict), doc="""
        Defines the layout of the views in the monitor layout. Can be
        'column', 'row', 'grid', 'row' or a nested list of indexes
        corresponding to the views, e.g. [[0, 1], [2]] will create
        a Column of one row containing views 0 and 1 and a second Row
        containing view 2.""" )

    title = param.String(doc="A title for this Card.")

    views = param.ClassSelector(class_=(list, dict), doc="""
        List or dictionary of View objects.""")

    def __init__(self, **params):
        self.kwargs = params.pop('kwargs', {})
        super().__init__(**params)
        params = {k: v for k, v in self.kwargs.items() if k in pn.Card.param}
        self._card = pn.Card(
            title=self.title, name=self.title, collapsible=False, **params
        )
        self._card[:] = [self._construct_layout()]

    def __panel__(self) -> pn.Card:
        return self._card

    def _construct_layout(self):
        layout = self.layout
        if isinstance(layout, list):
            item = resolve_layout_spec(layout, self.views, title=self.title)
        else:
            kwargs = dict(self.kwargs)
            if isinstance(layout, dict):
                layout_kwargs = dict(layout)
                layout = layout_kwargs.pop('type')
                kwargs.update(layout_kwargs)
            if layout == 'grid' and 'ncols' not in kwargs:
                kwargs['ncols'] = 2
            layout_type = _LAYOUTS[layout]
            layout_params = list(layout_type.param.params())
            kwargs = {k: v for k, v in kwargs.items() if k in layout_params}
            item = layout_type(*(view.panel for view in self.views), **kwargs)
        return item

    @classmethod
    def from_spec(
        cls, spec: Dict[str, Any] | str, filters: List[Filter] | None = None,
        pipelines: Dict[str, Pipeline] = {}
    ) -> 'Card':
        """
        Creates a Card from a specification.

        Parameters
        ----------
        spec : dict
            Specification declared as a dictionary of parameter values
            or a string referencing a source in the sources dictionary.
        filters: Optional[list[Filter]]
            A list of Filter components to add to the pipeline.
        pipelines: dict[str, Pipeline]
            Dictionary of pipelines from which Pipeline references will
            be resolved.

        Returns
        -------
        Resolved and instantiated Card object
        """
        if isinstance(spec, str):
            raise ValueError(
                "Card cannot be materialized by reference. Please pass "
                "full specification for the component."
            )
        spec = dict(spec)
        view_specs = spec.get('views', [])
        spec['views'] = views = []
        for view in view_specs:
            if isinstance(view_specs, dict):
                name = view
                view = view_specs[view]
            else:
                name = None

            if isinstance(view, View):
                if PARAM_NAME_PATTERN.match(view.name) and name:
                    with edit_constant(view):
                        view.name = name
                views.append(view)
                continue

            view_spec = view.copy()
            if 'name' not in view_spec and name:
                view_spec['name'] = name
            if 'pipeline' in view_spec:
                pname = view_spec.pop('pipeline')
                if pname in pipelines:
                    pipeline = pipelines.get(pname)
                else:
                    pipeline = Pipeline.from_spec(pname)
            elif len(pipelines) == 1:
                pipeline = list(pipelines.values())[0]
                if 'pipeline' in view_spec:
                    del view_spec['pipeline']
            if filters:
                pipeline = pipeline.chain(filters=list(filters), _chain_update=True)
            view = View.from_spec(view_spec, pipeline=pipeline)
            views.append(view)
        if filters:
            spec['title'] = ' '.join([f'{f.label}: {f.value}' for f in filters])
        return cls(**spec)

    def rerender(self):
        self._card[:] = [self._construct_layout()]


class Facet(Component):

    by = param.List(default=[], class_=FacetFilter, doc="""
        Fields to facet by.""")

    layout = param.ClassSelector(default=None, class_=(str, dict), doc="""
        How to lay out the facets.""")

    reverse = param.Boolean(default=False, doc="""
        Whether to reverse the sort order.""")

    sort = param.ListSelector(default=[], objects=[], doc="""
        List of fields to sort by.""")

    _valid_keys: ClassVar[Literal['params']] = 'params'
    _required_keys: ClassVar[List[str | Tuple[str, ...]]] = ['by']

    def __init__(self, **params):
        super().__init__(**params)
        self._sort_widget = pn.widgets.MultiSelect(
            options=self.param.sort.objects,
            sizing_mode='stretch_width',
            size=len(self.sort),
            value=self.sort,
        )
        self._sort_widget.link(self, value='sort')
        self._reverse_widget = pn.widgets.Checkbox(
            value=self.reverse, name='Reverse', margin=(5, 0, 0, 10)
        )
        self._reverse_widget.link(self, value='reverse')

    def get_sort_key(self, views) -> Tuple:
        sort_key = []
        for field in self.sort:
            values = [v.get_value(field) for v in views]
            if values:
                sort_key.append(values[0])
        return tuple(sort_key)

    @param.depends('sort:objects', watch=True)
    def _update_options(self):
        self._sort_widget.options = self.param.sort.objects

    @classmethod
    def from_spec(  # type: ignore
        cls, spec: Dict[str, Any] | str, schema: Dict[str, Dict[str, Any]]
    ) -> 'Facet':
        """
        Resolve a Facet object from spec and a schema.
        """
        if isinstance(spec, str):
            raise ValueError(
                "Facet cannot be materialized by reference. Please pass "
                "full specification for the component."
            )
        by = []
        for by_spec in spec.pop('by', []):
            if isinstance(by_spec, str):
                f = Filter.from_spec({'type': 'facet', 'field': by_spec}, schema)
            elif isinstance(by_spec, dict):
                f = Filter.from_spec(dict(by_spec, type='facet'), schema)
            elif isinstance(by_spec, Filter):
                f = by_spec
            by.append(f)
        sort = spec.pop('sort', [b.field for b in by])
        sorter = cls(by=by, **spec)
        sorter.param.sort.objects = sort
        return sorter

    @property
    def filters(self):
        return product(*[filt.filters for filt in self.by])


class Download(Component, Viewer):
    """
    The Download object controls options to download the Source tables
    in a variety of formats via the Dashboard UI.
    """

    labels = param.Dict(default={}, doc="""
        A dictionary to override the Button label for each table.""")

    format = param.ObjectSelector(default=None, objects=DOWNLOAD_FORMATS, doc="""
        The format to download the data in.""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments passed to the serialization function, e.g.
        data.to_csv(file_obj, **kwargs).""")

    pipelines = param.Dict(default={}, doc="""
        Dictionary of pipelines indexed by table name.""")

    tables = param.List(default=[], doc="""
        The list of tables to allow downloading.""")

    _internal_params: ClassVar[List[str]] = ['name', 'pipelines']
    _required_keys: ClassVar[List[str | Tuple[str, ...]]] = []

    def __init__(self, **params):
        super().__init__(**params)
        default_table = self.tables[0] if self.tables else None
        button_params = {'filename': f'{default_table}.{self.format}'}
        if default_table in self.labels:
            button_params['label'] = self.labels[default_table]
        self._select_download = pn.widgets.Select(
            name='Select table to download', options=self.tables, value=default_table,
            sizing_mode='stretch_width'
        )
        self._select_download.param.watch(self._update_button, 'value')
        self._download_button = pn.widgets.FileDownload(
            callback=self._table_data, align='end',
            sizing_mode='stretch_width', **button_params
        )
        self._layout = pn.Column(
            pn.pane.Markdown('### Download tables', margin=(0, 5, -10, 5)),
            self._download_button,
            pn.layout.Divider(),
            sizing_mode='stretch_width'
        )
        if len(self.tables) > 1:
            self._layout.insert(1, self._select_download)

    @classmethod
    def validate(cls, spec, context=None):
        if isinstance(spec, str):
            spec = {'format': spec}
        return super().validate(spec, context)

    def __panel__(self) -> pn.Column:
        return self._layout

    def _update_button(self, event: param.parameterized.Event):
        params = {'filename': f'{event.new}.{self.format}'}
        if event.new in self.labels:
            params['label'] = self.labels[event.new]
        self._download_button.param.set_param(**params)

    def __bool__(self) -> bool:
        return self.format is not None

    @catch_and_notify("Download failed")
    def _table_data(self):
        if self.format in ('json', 'csv'):
            io = StringIO()
        else:
            io = BytesIO()
        table = self._select_download.value
        data = self.pipelines[table].data
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

    @classmethod
    def from_spec(  # type: ignore
        cls, spec: Dict[str, Any] | str, pipelines: Dict[str, Pipeline]
    ) -> 'Download':
        """
        Creates a Download object from a specification.

        Parameters
        ----------
        spec : dict
            Specification declared as a dictionary of parameter values.
        pipelines: Dict[str, Pipeline]
            A dictionary of Pipeline objects indexed by table name.
        """
        if isinstance(spec, str):
            raise ValueError(
                "Download component cannot be materialized by reference. "
                "Please pass full specification for the component."
            )
        return cls(pipelines=pipelines, **spec)


class Layout(Component, Viewer):
    """
    `Layout` renders one or more :class:`lumen.views.base.View` components
    into a layout.

    Additionally it provides functionality for facetting the
    :class:`lumen.views.base.View` objects by grouping the data along
    some dimension.
    """

    auto_update = param.Boolean(default=True, constant=True, doc="""
        Whether changes in filters, transforms and references automatically
        trigger updates in the data or whether an update has to be triggered
        manually using the update event or the update button in the UI."""
    )

    download = param.ClassSelector(class_=Download, constant=True, default=Download(), doc="""
        The download objects determines whether and how the source tables
        can be downloaded.""")

    facet = param.ClassSelector(class_=Facet, constant=True, doc="""
        The facet object determines whether and how to facet the cards
        on the layout.""")

    layout = param.ClassSelector(default='column', class_=(str, list, dict), constant=True, doc="""
        Defines the layout of the views in the monitor layout. Can be
        'column', 'row', 'grid', 'row' or a nested list of indexes
        corresponding to the views, e.g. [[0, 1], [2]] will create
        a Column of one row containing views 0 and 1 and a second Row
        containing view 2.""" )

    reloadable = param.Boolean(default=True, constant=True, doc="""
        Whether to allow reloading data layout's source using a button.""")

    show_title = param.Boolean(default=True, doc="""
        Whether to show the title in Card headers.""")

    title = param.String(doc="A title for this Layout.")

    refresh_rate = param.Integer(default=None, doc="""
        How frequently to refresh the monitor by querying the adaptor.""")

    rerender = param.Event(default=False, doc="""
        An event that is triggered whenever the View requests a re-render.""")

    source = param.ClassSelector(class_=Source, constant=True, doc="""
        The Source queries the data from some data source.""")

    tsformat = param.String(default="%m/%d/%Y %H:%M:%S", constant=True)

    views = param.ClassSelector(class_=(list, dict), constant=True, doc="""
        List or dictionary of View specifications.""")

    _allows_refs: ClassVar[bool] = False

    _header_format: ClassVar[str] = '<div style="font-size: 1.5em; font-weight: bold;">{header}</div>'

    _required_keys: ClassVar[List[str | Tuple[str, ...]]] = ['title', 'views']

    _valid_keys: ClassVar[List[str]] = [
        'config', 'facet_layout', 'sort', # Deprecated
        'layout', 'refresh_rate', 'reloadable', 'show_title', 'title', 'tsformat', # Simple
        'views', 'source', 'filters', 'pipeline', 'facet', 'download' # Objects
    ] + list(Layoutable.param)

    def __init__(self, **params):
        if 'facet' not in params:
            params['facet'] = Facet()
        self._cache = {}
        self._cb = None
        self._scheduled = False
        self._updates = []
        self._update_button = pn.widgets.Button(name='Apply update')
        self._update_button.on_click(self._manual_update)
        self._timestamp = pn.pane.HTML(
            align='center', margin=(10, 0), sizing_mode='stretch_width'
        )
        self._pipelines = params.pop('pipelines', {})
        self.kwargs = {k: v for k, v in params.items() if k not in self.param}
        super().__init__(**{k: v for k, v in params.items() if k in self.param})
        if not self.title:
            self.title = self.name

        # Render content
        if self.views:
            self._construct_cards()
        self._cards = self.get_cards()

        # Set up watchers
        self.facet.param.watch(self._resort, ['sort', 'reverse'])
        for view in list(self._cache.values())[0].views:
            view.param.watch(self._schedule_rerender, 'rerender')

    @param.depends('rerender')
    def __panel__(self) -> Viewable:
        return self.panels

    def _resort(self, *events: param.parameterized.Event):
        self._rerender(update_views=False)

    def _manual_update(self, *events: param.parameterized.Event):
        self._update_button.loading = True
        try:
            for p in self._pipelines.values():
                p.param.trigger('update')
        finally:
            self._update_button.loading = False

    def _sync_component(self, component, *events: param.parameterized.Event):
        component.param.set_param(**{event.name: event.new for event in events})

    def _construct_cards(self):
        controls, all_transforms = [], []
        linked_views = None
        for facet_filters in self.facet.filters:
            key = tuple(str(f.value) for f in facet_filters)
            self._cache[key] = card = Card.from_spec({
                'kwargs': self.kwargs,
                'layout': self.layout,
                'title': self.title,
                'views': self.views,
            }, filters=facet_filters, pipelines=self._pipelines)
            if linked_views is None:
                for view in card.views:
                    if view.controls:
                        controls.append(view.control_panel)
                    transforms = (
                        view.pipeline.traverse('transforms') +
                        view.pipeline.traverse('sql_transforms')
                    )
                    for transform in transforms:
                        if transform not in all_transforms and transform.controls:
                            controls.append(transform.control_panel)
                            all_transforms.append(transform)
                linked_views = card.views
                continue

            # Only the controls for the first facet is shown so link
            # the other facets to the controls of the first
            for v1, v2 in zip(linked_views, card.views):
                v1.param.watch(partial(self._sync_component, v2), v1.refs)
                for t1, t2 in zip(v1.pipeline.transforms, v2.pipeline.transforms):
                    t1.param.watch(partial(self._sync_component, t2), t1.refs)
                for t1, t2 in zip(v1.pipeline.sql_transforms, v2.pipeline.sql_transforms):
                    t1.param.watch(partial(self._sync_component, t2), t1.refs)
        self._view_controls = pn.Column(*controls, sizing_mode='stretch_width')

    ##################################################################
    # Create UI
    ##################################################################

    def get_cards(self) -> List[Card]:
        cards = []
        for card in self._cache.values():
            sort_key = self.facet.get_sort_key(card.views)
            if any(view for view in card.views):
                cards.append((sort_key, card))
        if self.facet.sort:
            cards = sorted(cards, key=lambda x: x[0])
            if self.facet.reverse:
                cards = cards[::-1]
        return [card for _, card in cards]

    def get_filter_panel(self, skip=None, apply_button: bool=True) -> pn.Column:
        skip = list(skip or [])
        views: List[Any] = []

        # Variable controls
        global_refs = state.global_refs
        layout_refs = [ref.split('.')[1] for ref in self.refs if ref not in global_refs]
        var_panel = state.variables.panel(layout_refs)
        if var_panel is not None:
            views.append(var_panel)

        # Source controls
        source_panel = self.source.panel
        if source_panel:
            views.extend([
                self._header_format.format(header='Source'),
                source_panel,
                pn.layout.Divider()
            ])

        # Download controls
        if self.download:
            views.append(self.download)

        # Filter controls
        filters = []
        auto_update = self.auto_update
        for pipeline in self._pipelines.values():
            auto_update &= pipeline.auto_update
            subpipeline = pipeline
            while subpipeline is not None:
                subfilters = []
                for filt in subpipeline.filters:
                    fpanel = filt.panel
                    if filt not in skip and fpanel is not None:
                        subfilters.append(fpanel)
                        skip.append(filt)
                if subfilters:
                    filters.append(subfilters)
                subpipeline = subpipeline.pipeline
        if filters:
            views.append(self._header_format.format(header='Filters'))
            views.extend([filt for sfilters in filters[::-1] for filt in sfilters])

        # Facet controls
        if self.facet.param.sort.objects:
            if views:
                views.append(pn.layout.Divider(margin=0, height=5))
            views.extend([
                self._header_format.format(header='Sort'),
                self.facet._sort_widget,
                self.facet._reverse_widget
            ])

        # View controls
        if self._view_controls:
            views.append(self._view_controls)

        # Add update button
        if not auto_update and apply_button:
            views.append(self._update_button)

        # Reload buttons
        if self.reloadable:
            if views:
                views.append(pn.layout.Divider(margin=0, height=5))
            self._reload_button = IconButton(
                icon='fa-sync', size=18, margin=10
            )
            self._reload_button.on_click(self.update)
            self._timestamp.object = f'Last updated: {dt.datetime.now().strftime(self.tsformat)}'
            reload_panel = pn.Row(
                self._reload_button, self._timestamp, sizing_mode='stretch_width',
                margin=(10, 10, 0, 5)
            )
            views.append(reload_panel)
        return pn.Column(*views, name=self.title, sizing_mode='stretch_width')

    ##################################################################
    # Rendering API
    ##################################################################

    def _schedule_rerender(self, *events: param.parameterized.Event):
        for e in events:
            if isinstance(e.obj, View) and e.obj not in self._updates:
                self._updates.append(e.obj)
        if self._scheduled:
            return
        self._scheduled = True
        if pn.state.curdoc:
            pn.state.curdoc.add_next_tick_callback(self._rerender)
        else:
            self._rerender()

    def _rerender_cards(self, cards: List[Card]):
        for card in cards:
            if any(view in self._updates for view in card.views):
                card.rerender()
        self._updates = []

    def _rerender(self, update_views: bool = True):
        self._scheduled = False
        cards = self.get_cards()
        rerender = False
        if update_views:
            self._rerender_cards(cards)
            rerender = True
        if cards != self._cards:
            self._cards[:] = cards
            rerender = True
        if rerender:
            self.param.trigger('rerender')

    ##################################################################
    # Validation API
    ##################################################################

    @classmethod
    def _validate_config(cls, config: Dict[str, Any], spec: Dict[str, Any], context: Dict[str, Any]) -> None:
        msg = (
            "Passing 'config' to a Layout is deprecated use the 'facet' key "
            "on the layout instead."
        )
        return cls._deprecation(msg, 'facet', spec, config)

    @classmethod
    def _validate_sort(cls, sort: str, spec: Dict[str, Any], context: Dict[str, Any]) -> None:
        msg = (
            "Passing 'sort' to a Layout is deprecated use the 'facet' key "
            "on the layout instead."
        )
        return cls._deprecation(msg, 'facet', spec, {'sort': sort})

    @classmethod
    def _validate_facet_layout(cls, facet_layout: Dict[str, Any], spec: Dict[str, Any], context: Dict[str, Any]):
        msg = (
            "Passing 'facet_layout' to a Layout is deprecated use the 'facet' key "
            "on the layout instead."
        )
        return cls._deprecation(msg, 'facet', spec, {'facet_layout': facet_layout})

    @classmethod
    def _validate_pipeline(cls, *args, **kwargs):
        return cls._validate_str_or_spec('pipeline', Pipeline, *args, **kwargs)

    @classmethod
    def validate(cls, spec: Dict[str, Any] | str, context: Dict[str, Any] | None = None) -> Dict[str, Any] | str:
        if isinstance(spec, str):
            return spec
        if "source" in spec and "pipeline" in spec:
            raise ValueError(f"{cls.name} should either have a source or a pipeline.")
        return super().validate(spec, context)

    @classmethod
    def _validate_filters(cls, filter_specs: List[Dict[str, Any] | str], spec: Dict[str, Any], context: Dict[str, Any]):
        filters = cls._validate_list_subtypes('filters', Filter, filter_specs, spec, context)
        for filter_spec in filter_specs:
            if isinstance(filter_spec, str):
                raise ValidationError(
                    'Layout facet filters may not be passed by reference',
                    spec, 'filters'
                )
            elif filter_spec['type'] == 'facet':
                raise ValidationError(
                    'Layout facetting must be declared via the facet field of the layout specification, '
                    'specifying filters of type \'facet\' is no longer supported.', spec, 'filters'
                )
        if 'pipeline' in spec:
            raise ValidationError(
                'Layout may not declare filters AND a pipeline. Please declare the filters as '
                'part of the pipeline. ', spec, 'filters'
            )
        return filters

    @classmethod
    def _validate_source(cls, source_spec: Dict[str, Any] | str, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        if isinstance(source_spec, str):
            if source_spec not in context['sources']:
                msg = f'Layout specified non-existent source {source_spec!r}.'
                msg = match_suggestion_message(source_spec, list(context['sources']), msg)
                raise ValidationError(msg, spec, source_spec)
            return source_spec

        warnings.warn(
            'Inlining source definitions in a layout is no longer supported. '
            'Please ensure you declare all sources as part of the global \'sources\' '
            'field', DeprecationWarning
        )
        src_spec = Source.validate(source_spec, context)
        src_cls = Source._get_type(src_spec['type'])  # type: ignore
        source_name = f'{src_cls.name}' # NOTE: Create unique name
        if 'sources' not in context:
            context['sources'] = {}
        context['sources'][source_name] = source_spec
        return source_name

    @classmethod
    def _validate_views(
        cls, view_specs: Dict[str, Dict[str, Any] | str] | List[Dict[str, Any] | str],
        spec: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any] | str] | List[Dict[str, Any] | str]:
        view_specs = cls._validate_dict_or_list_subtypes('views', View, view_specs, spec, context)
        if 'source' in spec or 'pipeline' in spec:
            return view_specs
        views = list(view_specs.values()) if isinstance(view_specs, dict) else view_specs
        if not all('pipeline' in spec for spec in views):
            raise ValidationError(
                'Layout (or its views) must declare a source or a pipeline.', spec
            )
        return view_specs

    ##################################################################
    # Public API
    ##################################################################

    @classmethod
    def from_spec(cls, spec: Dict[str, Any] | str, **kwargs) -> 'Layout':
        """
        Creates a Layout object from a specification. If a Layout
        specification references an existing Source or Filter by name.

        Parameters
        ----------
        spec : dict
          Specification declared as a dictionary of parameter values.
        kwargs: dict
          Additional kwargs to pass to the Layout

        Returns
        -------
        Resolved and instantiated Layout object
        """
        if isinstance(spec, str):
            raise ValueError(
                "Layout cannot be materialized by reference. Please pass "
                "full specification for the component."
            )
        # Resolve source
        spec = dict(spec)
        views = spec['views']

        pipelines = {}
        source_filters = None
        filter_specs = spec.pop('filters', [])
        view_specs = views if isinstance(views, list) else list(views.values())
        if 'source' in spec:
            source_spec = spec.pop('source', None)
            if isinstance(source_spec, Source):
                source = source_spec
            else:
                source = Source.from_spec(source_spec)
                if isinstance(source_spec, str):
                    source_filters = state.filters.get(source_spec)
        else:
            if 'pipeline' in spec:
                pspecs = [spec['pipeline']]
            else:
                pspecs = [vspec.get('pipeline') for vspec in view_specs if 'pipeline' in vspec]
            pipeline_spec = pspecs[0]
            if isinstance(pipeline_spec, str):
                if pipeline_spec not in state.pipelines:
                    raise ValueError(f'{pipeline_spec!r} not found in global pipelines.')
                pipeline = state.pipelines[pipeline_spec]
            else:
                pipeline = Pipeline.from_spec(pipeline_spec)
            source = pipeline.source
            pipelines[pipeline.name] = pipeline

        tables = source.get_tables()

        # Resolve facets
        if 'facet' in spec:
            facet_spec = spec.pop('facet', {})
            schema = source.get_schema()
        else:
            facet_spec, schema = {}, {}
        spec['facet'] = Facet.from_spec(facet_spec, schema)

        # Backward compatibility
        processed_views = []
        for view_spec in view_specs:
            view_spec = dict(view_spec)
            processed_views.append(view_spec)
            if 'pipeline' in view_spec:
                pname = view_spec['pipeline']
                pipelines[pname] = Pipeline.from_spec(pname)
                continue
            elif 'pipeline' in spec:
                continue
            elif 'table' in view_spec:
                table = view_spec.pop('table')
            elif len(tables) == 1:
                table = tables[0]
            else:
                raise ValidationError(
                    'View did not declare unambiguous table reference, '
                    'since selected source declares multiple tables. '
                    'Either declare a pipeline or explicitly declare the '
                    'table that the View references.', view_spec
                )
            pspec = {'table': table}
            if filter_specs:
                pspec['filters'] = filter_specs
            view_pipeline = Pipeline.from_spec(pspec, source, source_filters)
            pipelines[view_pipeline.name] = view_pipeline
            view_spec['pipeline'] = view_pipeline.name

        if isinstance(views, dict):
            spec['views'] = dict(zip(views, processed_views))
        else:
            spec['views'] = processed_views

        # Resolve download options
        download_spec = spec.pop('download', {})
        if 'tables' not in download_spec:
            download_spec['tables'] = list(tables)
        spec['download'] = Download.from_spec(download_spec, pipelines)

        params = dict(kwargs, **spec)
        return cls(source=source, pipelines=pipelines, **params)

    @property
    def refs(self) -> List[str]:
        refs = []
        for pipeline in self._pipelines.values():
            for ref in pipeline.refs:
                if ref not in refs:
                    refs.append(ref)
        views = self.views
        for spec in (views if isinstance(views, list) else views.values()):
            for ref in extract_refs(spec, 'variables'):
                if ref not in refs:
                    refs.append(ref)
        return refs

    @property
    def panels(self) -> Viewable:
        """
        Returns a layout of the rendered View objects on this layout.
        """
        if len(self._cards) > 1:
            default = 'flex' if 'flex' in _LAYOUTS else 'grid'
        else:
            default = 'column'
        layout = self.facet.layout or default
        kwargs = dict(name=self.title, margin=0, sizing_mode='stretch_width')
        if isinstance(layout, dict):
            layout_kwargs = dict(layout)
            layout = layout_kwargs.pop('type')
            kwargs.update(layout_kwargs)
        layout_type = _LAYOUTS[layout]
        if layout == 'grid' and 'ncols' not in kwargs:
            kwargs['ncols'] = 3
        if self._cards:
            content = self._cards
            if len(self._cards) == 1 and not self.show_title:
                self._cards[0]._card.param.set_param(
                    hide_header=True, sizing_mode='stretch_width', margin=0
                )
        else:
            content = (
                pn.pane.Alert(
                    f'**All views are empty for {self.title!r} since no data is available. '
                    'Try relaxing the filter constraints.**',
                    sizing_mode='stretch_width',
                    margin=(0, 0, 50, 0),
                ),
            )
        return layout_type(*content, **kwargs)

    def to_spec(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        spec = super().to_spec(context=context)
        if len(self._pipelines) == 1:
            pipeline = list(self._pipelines.values())[0]
            if context:
                if pipeline.name not in context.get('pipelines', {}):
                    context['pipelines'][pipeline.name] = pipeline.to_spec(context=context)
                spec['pipeline'] = pipeline.name
            else:
                spec['pipeline'] = pipeline.to_spec()
        if 'source' in spec and 'pipeline' in spec:
            del spec['source']
        if isinstance(spec['views'], dict):
            spec['views'] = {
                name: view.to_spec(context) if isinstance(view, View) else view
                for name, view in spec['views'].items()
            }
        else:
            spec['views'] = [
                view.to_spec(context) if isinstance(view, View) else view
                for view in spec['views']
            ]
        return spec

    @pn.depends('refresh_rate', watch=True)
    def start(self, event: param.parameterized.Event | None = None):
        """
        Starts any periodic callback instantiated on this object.
        """
        refresh_rate = self.refresh_rate if event is None else event.new
        if refresh_rate is None:
            return
        if self._cb:
            self._cb.period = refresh_rate
        else:
            self._cb = pn.state.add_periodic_callback(
                self.update, refresh_rate
            )

    def update(self, *events: param.parameterized.Event, clear_cache: bool = True):
        """
        Updates the views on this layout by clearing any caches and
        rerendering the views on this Layout.
        """
        if clear_cache:
            self.source.clear_cache()
        self._rerender()
        self._timestamp.object = f'Last updated: {dt.datetime.now().strftime(self.tsformat)}'
