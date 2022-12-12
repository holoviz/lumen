import panel as pn
import param  # type: ignore

from panel.reactive import ReactiveHTML

from lumen.config import _LAYOUTS
from lumen.state import state as lm_state
from lumen.views.base import View

from .base import WizardItem
from .gallery import Gallery, GalleryItem
from .sources import ASSETS_DIR
from .state import state


class LayoutEditor(ReactiveHTML):
    "Select the views on this layout and declare a layout."

    spec = param.Dict(default={})

    title = param.String(default='')

    sizing_mode = param.String(default='stretch_width', readonly=True)

    layout = param.Parameter()

    layout_type = param.Selector(default='flex', objects=list(_LAYOUTS), doc="""
       Select how you want to lay out the selected views.""")

    views = param.List(default=[])

    view_select = param.Parameter()

    _template = """
    <span style="font-size: 1.5em">{{ title }} Layout</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div style="display: flex; width: 100%;">
      <form role="form" style="flex: 25%; max-width: 300px; line-height: 2em;">
        <div id="view-select">${view_select}</div>
        <div style="display: grid; flex: auto;">
          <label for="layout-type-${id}"><b>{{ param.layout_type.label }}</b></label>
          <fast-select id="layout-type" style="max-width: 250px; min-width: 150px; z-index: 100;" value="${layout_type}">
          {% for lt in param.layout_type.objects %}
            <fast-option value="{{ lt }}" {% if lt == layout_type %}selected{% endif %}>
              {{ lt.title() }}
            </fast-option>
          {% endfor %}
          </fast-select>
          <fast-tooltip anchor="layout-type-${id}">{{ param.layout_type.doc }}</fast-tooltip>
        </div>
      </form>
      <div id="layout" style="flex: auto; margin-left: 1em;">${layout}</div>
    </div>
    """

    def __init__(self, **params):
        spec = params.pop('spec', {})
        layout_spec = spec.get('layout', 'flex')
        layout_type, layout = self._construct_layout(layout_spec)
        params['layout_type'] = layout_type
        params['layout'] = layout
        params['view_select'] = vsel = pn.widgets.MultiSelect(
            name='Select views', options=params.get('views', []),
            max_width=250, sizing_mode='stretch_width', margin=0
        )
        params.update(**{
            k: v for k, v in spec.items()
            if k in self.param and k not in params and not self.param[k].readonly
        })
        vsel.link(self, value='views')
        super().__init__(spec=spec, **params)
        if 'views' not in self.spec:
            self.spec['views'] = {}
        self._views = {}
        self._watchers = {}
        state.views.param.watch(self._update_views, 'items')
        self._update_views()
        self.view_select.value = self.view_select.options

    @property
    def thumbnail(self):
        return ASSETS_DIR / 'layout.png'

    def _construct_layout(self, layout_spec):
        layout_kwargs = {'sizing_mode': 'stretch_both'}
        if isinstance(layout_spec, list):
            layout_spec = 'column'
        elif isinstance(layout_spec, dict):
            layout_kwargs = layout_spec
            layout_spec = layout_spec.pop('type')
        return layout_spec, _LAYOUTS.get(layout_spec, pn.FlexBox)(**layout_kwargs)

    def _update_views(self, *events):
        for name, item in state.views.items.items():
            if name not in self._watchers:
                self._watchers[name] = item.param.watch(self._update_views, 'selected')
        views = [name for name, item in state.views.items.items() if item.selected]
        self.view_select.options = views

    def _populate_layout(self, layout):
        views = self.spec['views']
        for i, view_spec in enumerate(views.items() if isinstance(views, dict) else views):
            if isinstance(view_spec, tuple):
                name, view = view_spec
                view['name'] = name
            else:
                view = view_spec
                name = None
            view = dict(view)
            kwargs = {}
            if 'source' not in view and 'pipeline' not in view:
                if 'pipeline' in self.spec:
                    pipeline = self.spec['pipeline']
                    if isinstance(pipeline, str):
                        pipeline = lm_state.pipelines[pipeline]
                    kwargs['pipeline'] = pipeline
                elif 'source' in self.spec:
                    source = self.spec['source']
                    if isinstance(source, str):
                        source = lm_state.sources[source]
                    kwargs['source'] = source
            if name in self._views:
                view = self._views[name]
            else:
                view = View.from_spec(view, **kwargs)
                name = name or view.name
                self._views[name] = view
            if hasattr(layout, 'append'):
                layout.append(view.get_panel())
            else:
                layout[i*3:(i+1)*3, :] = view.get_panel()

    @param.depends('layout_type', 'views', watch=True)
    def _update_layout(self):
        self.spec['layout'] = self.layout_type
        self.spec['views'] = {v: state.views.items[v].spec for v in self.views}
        _, layout = self._construct_layout(self.layout_type)
        self._populate_layout(layout)
        self.layout = layout


class LayoutGalleryItem(GalleryItem):

    editor = param.ClassSelector(class_=LayoutEditor, precedence=-1)

    selected = param.Boolean(default=True)

    views = param.Dict(default={})

    sizing_mode = param.String(default="stretch_both", readonly=True)

    def __init__(self, **params):
        spec = params['spec']
        if 'description' not in params:
            params['description'] = f"Contains {len(spec['views'])} views.."
        super().__init__(**params)
        self.view = pn.pane.PNG(self.thumbnail, height=200, align='center')
        self._modal_content = [self.editor]

    def _open_modal(self, event):
        self.editor._populate_layout(self.editor.layout)
        super()._open_modal(event)


class LayoutGallery(WizardItem, Gallery):
    "Add, select and configure layout groups to add to your dashboard."

    spec = param.ClassSelector(class_=(dict, list), default=[])

    layouts = param.List(precedence=-1)

    _template = """
    <span style="font-size: 1.5em">Layout groups</span>
    <fast-divider></fast-divider>
    <span style="font-size: 1.2em; font-weight: bold;">{{ __doc__ }}</p>
    <div id="items" style="margin: 1em 0; display: flex; flex-wrap: wrap; gap: 1em;">
    {% for item in items.values() %}
      <fast-card id="layout-container" style="width: 350px; height: 400px;">
        ${item}
      </fast-card>
    {% endfor %}
      <fast-card id="view-container-new" style="height: 400px; width: 350px; padding: 1em;">
        <div style="display: grid;">
          <span style="font-size: 1.25em; font-weight: bold;">Create new layout group</span>
          <i id="add-button" onclick="${_open_modal}" class="fa fa-plus" style="font-size: 14em; margin: 0.2em auto;" aria-hidden="true"></i>
        </div>
      </fast-card>
    </div>
    """

    def __init__(self, **params):
        super().__init__(**params)
        self._editor = LayoutsEditor()
        self._save_button = pn.widgets.Button(name='Save layout')
        self._save_button.on_click(self._save_layouts)
        self._modal_content = [self._editor, self._save_button]

    def _save_layouts(self, event):
        for layout in self._editor.layouts:
            item = LayoutGalleryItem(
                name=layout.title, spec=layout.spec, selected=True,
                editor=layout, thumbnail=layout.thumbnail
            )
            self.items[layout.title] = item
            self.layouts.append(layout)
            self.spec.append(layout.spec)
        self.param.trigger('items')
        self.param.trigger('layouts')
        self._editor.layouts = []
        state.template.close_modal()


class LayoutsEditor(WizardItem):
    """
    Add and configure your monitoring layouts.
    """

    title = param.String(default="")

    spec = param.List(precedence=-1)

    sources = param.List(doc="Select a source")

    source = param.String()

    layouts = param.List([], precedence=-1)

    title = param.String(default='')

    _template = """
    <span style="font-size: 2em">Layout editor</span>
    <p>{{ __doc__ }}</p>
    <fast-button id="submit" appearance="accent" style="position: absolute; top: 3em; right: 5px;" onclick="${_add_layout}">
      <b style="font-size: 2em;">+</b>
    </fast-button>
    <fast-divider></fast-divider>
    <div style="display: flex;">
      <div id="layout-list" style="flex: auto; overflow-y: auto; gap: 1em;">
        {% for layout in layouts %}
        <div id="layout-container">${layout}</div>
        <fast-divider></faster-divider>
        {% endfor %}
      </div>
    </div>
    """

    _dom_events = {'layout-title': ['keyup']}

    def __init__(self, **params):
        super().__init__(**params)
        self._source = None
        state.sources.param.watch(self._update_sources, 'sources')

    def _update_sources(self, event):
        self.sources = list(event.new)
        if not self.source and self.sources:
            self.source = self.sources[0]

    def _add_layout(self, event):
        spec = {'title': self.title}
        editor = LayoutEditor(spec=spec, title=self.title)
        self.spec.append(spec)
        self.layouts.append(editor)
        self.param.trigger('layouts')
        self.title = ''
