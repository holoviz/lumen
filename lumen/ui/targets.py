import panel as pn
import param

from panel.reactive import ReactiveHTML

from lumen.config import _LAYOUTS
from lumen.state import state as lumen_state
from lumen.views import View

from .base import WizardItem
from .gallery import Gallery, GalleryItem
from .state import state


class TargetEditor(ReactiveHTML):
    "Select the views on this target and declare a layout."

    spec = param.Dict(default={})

    title = param.String(default='')

    sizing_mode = param.String(default='stretch_width', readonly=True)

    layout = param.Parameter()

    layout_type = param.Selector(default='column', objects=list(_LAYOUTS))

    views = param.List(default=[])

    view_select = param.Parameter()

    _template = """
    <span style="font-size: 1.5em">{{ title }} Target</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div style="display: flex; width: 100%;">
      <form role="form" style="flex: 25%; max-width: 300px; line-height: 2em;">
        <div id="view-select">${view_select}</div>
        <div style="display: grid; flex: auto;">
          <label for="layout_type-${id}"><b>{{ param.layout_type.label }}</b></label>
          <fast-select id="layout_type" style="max-width: 250px; min-width: 150px; z-index: 100;" value="${layout_type}">
          {% for lt in param.layout_type.objects %}
            <fast-option value="{{ lt }}" {% if lt == layout_type %}selected{% endif %}>{{ lt.title() }}</fast-option>
          {% endfor %}
          </fast-select>
          <fast-tooltip anchor="layout_type-${id}">{{ param.layout_type.doc }}</fast-tooltip>
        </div>
      </form>
      <div id="layout" style="flex: auto; margin-left: 1em;">${layout}</div>
    </div>
    """

    _scripts = {'relayout': 'setTimeout(() => view.invalidate_layout(), 100);'}

    def __init__(self, **params):
        spec = params.get('spec')
        layout_spec = spec.get('layout', 'flex')
        layout_type, layout = self._construct_layout(layout_spec)
        params['layout_type'] = layout_type
        params['layout'] = layout
        params['view_select'] = vsel = pn.widgets.MultiSelect(
            name='Select views', options=params.get('views', []),
            max_width=250, sizing_mode='stretch_width', margin=0
        )
        vsel.link(self, value='views')
        super().__init__(**params)
        if 'views' not in self.spec:
            self.spec['views'] = {}
        self._views = {}
        self._populate_layout(self.layout)
        for name, view in state.views.items.items():
            if self.source == view.editor.source:
                self.views.append(name)
                if name not in vsel.options:
                    vsel.options.append(name)

    def _construct_layout(self, layout_spec):
        layout_kwargs = {'sizing_mode': 'stretch_both'}
        if isinstance(layout_spec, list):
            layout_spec = 'column'
        elif isinstance(layout_spec, dict):
            layout_kwargs = layout_spec
            layout_spec = layout_spec.pop('type')
        return layout_spec, _LAYOUTS.get(layout_spec, pn.FlexBox)(**layout_kwargs)

    def _populate_layout(self, layout):
        source = self.spec['source']
        views = self.spec['views']
        view_specs = views.items() if isinstance(views, dict) else views
        for i, view_spec in enumerate(view_specs):
            if isinstance(view_spec, tuple):
                name, view = view_spec
            else:
                view = view_spec
                name = None
            if name in self._views:
                view = self._views[name]
            else:
                source_spec = state.sources.sources[source].spec
                source_obj = lumen_state.load_source(source, source_spec)
                view = View.from_spec(view, source_obj, [])
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


class TargetGalleryItem(GalleryItem):
                 
    editor = param.ClassSelector(class_=TargetEditor, precedence=-1)

    selected = param.Boolean(default=True)

    views = param.Dict(default={})

    sizing_mode = param.String(default="stretch_both", readonly=True)

    _template = """
    <span style="font-size: 1.2em; font-weight: bold;">{{ spec.title }}</p>
    <fast-switch id="selected" checked=${selected} style="float: right"></fast-switch>
    <div id="details" style="margin: 1em 0;">
      ${view}
    </div>
    <p style="height: 4em;">{{ description }}</p>
    <fast-button id="edit-button" style="width: 320px; position: absolute; bottom: 0px;" onclick="${_open_modal}">Edit</fast-button>
    """

    def __init__(self, **params):
        spec = params['spec']
        source_name = spec['source']
        if 'description' not in params:
            params['description'] = f"Contains {len(spec['views'])} views of the {source_name!r} source."
        if 'thumbnail' not in params:
            params['thumbnail'] = state.sources.sources[source_name].thumbnail
        super().__init__(**params)
        self.view = pn.pane.PNG(self.thumbnail, height=200, align='center')
        self._modal_content = [self.editor]


class TargetGallery(WizardItem, Gallery):
    "Add, select and configure layout groups to add to your dashboard."

    spec = param.ClassSelector(class_=(dict, list), default=[])

    targets = param.List(precedence=-1)

    _template = """
    <span style="font-size: 1.5em">Layout groups</span>
    <fast-divider></fast-divider>
    <span style="font-size: 1.2em; font-weight: bold;">{{ __doc__ }}</p>
    <div id="items" style="margin: 1em 0; display: flex; flex-wrap: wrap; gap: 1em;">
    {% for item in items.values() %}
      <fast-card id="target-container" style="width: 350px; height: 400px;">
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
        self._editor = TargetsEditor()
        self._save_button = pn.widgets.Button(name='Save target')
        self._save_button.on_click(self._save_targets)
        self._modal_content = [self._editor, self._save_button]

    def _save_targets(self, event):
        for target in self._editor.targets:
            item = TargetGalleryItem(
                name=target.title, spec=target.spec, selected=True,
                editor=target
            )
            self.items[target.title] = item
            self.targets.append(target)
            self.spec.append(target.spec)
        self.param.trigger('items')
        self.param.trigger('targets')
        self._editor.targets = []
        state.template.close_modal()


class TargetsEditor(WizardItem):
    """
    Add and configure your monitoring targets.
    """

    title = param.String(default="")

    spec = param.List(precedence=-1)

    sources = param.List(doc="Select a source")

    source = param.String()

    targets = param.List([], precedence=-1)
    
    title = param.String(default='')

    _template = """
    <span style="font-size: 2em">Layout editor</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div style="display: flex;">
      <form role="form" style="flex: 20%; max-width: 250px; line-height: 2em;">
        <div style="display: grid;">
          <label for="target-title-${id}"><b>{{ param.title.label }}</b></label>
          <fast-text-field id="target-title" placeholder="{{ param.title.doc }}" value="${title}">
          </fast-text-field>
        </div>
        <div style="display: flex;">
        <div style="display: grid; flex: auto;">
          <label for="sources-${id}"><b>{{ param.sources.label }}</b></label>
          <fast-select id="source" style="max-width: 250px; min-width: 150px;" value="${source}">
          {% for src in sources %}
            <fast-option value="{{ src }}">{{ src.title() }}</fast-option>
          {% endfor %}
          </fast-select>
          <fast-tooltip anchor="sources-${id}">{{ param.sources.doc }}</fast-tooltip>
        </div>
        <fast-button id="submit" appearance="accent" style="margin-top: auto; margin-left: 1em; width: 20px;" onclick="${_add_target}">
            <b style="font-size: 2em;">+</b>
        </fast-button>
        </div>
      </form>
      <div style="flex: auto; overflow-y: auto; gap: 1em;">
        {% for target in targets %}
        <div id="target-container">${target}</div>
        <fast-divider></faster-divider>
        {% endfor %}
      </div>
    </div>
    """

    _dom_events = {'target-title': ['keyup']}

    def __init__(self, **params):
        super().__init__(**params)
        self._source = None
        state.sources.param.watch(self._update_sources, 'sources')

    def _update_sources(self, event):
        self.sources = list(event.new)
        if not self.source and self.sources:
            self.source = self.sources[0]

    def _add_target(self, event):
        spec = {'title': self.title, 'source': self.source}
        editor = TargetEditor(spec=spec, **spec)
        self.spec.append(spec)
        self.targets.append(editor)
        self.param.trigger('targets')
        self.title = ''
