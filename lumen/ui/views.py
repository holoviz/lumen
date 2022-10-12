import panel as pn
import param

from panel.reactive import ReactiveHTML

from lumen.state import state as lm_state
from lumen.util import catch_and_notify
from lumen.views import View

from .base import WizardItem
from .gallery import Gallery, GalleryItem
from .sources import ASSETS_DIR
from .state import state


class ViewsEditor(WizardItem):
    """
    Declare the views for your dashboard.
    """

    spec = param.List(default=[], precedence=-1)

    pipeline  = param.String()

    view_type = param.Selector()

    pipelines = param.List(label='Select a pipeline')

    views = param.List()

    _template = """
    <span style="font-size: 2em">View Editor</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div style="display: flex;">
      <form role="form" style="flex: 30%; max-width: 300px; line-height: 2em;">
        <div style="display: grid; flex: auto;">
          <label for="view-select-${id}"><b>{{ param.view_type.label }}</b></label>
          <fast-select id="view-select" style="min-width: 150px;" value="${view_type}">
          {% for vtype in param.view_type.objects %}
            <fast-option value="{{ vtype }}"  {% if vtype == view_type %}selected{% endif %}>
             {{ vtype.title() }}
            </fast-option>
          {% endfor %}
          </fast-select>
          <fast-tooltip anchor="view-select-${id}">{{ param.view_type.doc }}</fast-tooltip>
        </div>
        <div style="display: flex;">
          <div style="display: grid;">
            <label for="pipelines-${id}"><b>{{ param.pipelines.label }}</b></label>
            <fast-select id="pipelines" style="max-width: 300px; min-width: 240px;" value="${pipeline}">
            {% for ppl in pipelines %}
              <fast-option value="{{ ppl|tojson }}">{{ ppl }}</fast-option>
            {% endfor %}
            </fast-select>
            <fast-tooltip anchor="pipelines-${id}">{{ param.pipelines.doc }}</fast-tooltip>
          </div>
          <fast-button id="submit" appearance="accent" style="margin-top: auto; margin-left: 1em; width: 20px;" onclick="${_add_view}">
            <b style="font-size: 2em;">+</b>
          </fast-button>
        </div>
      </form>
      <div id="view-list" style="flex: auto; margin-left: 2em;">
        {% for view in views %}
        <div id="view-container">${view}</div>
        <fast-divider></faster-divider>
        {% endfor %}
      </div>
    </div>
    """

    def __init__(self, **params):
        super().__init__(**params)
        views = param.concrete_descendents(View)
        self.param.view_type.objects = [
            view.view_type for view in views.values() if view.view_type
        ]
        state.pipelines.param.watch(self._update_pipelines, 'pipelines')

    def _add_view(self, event):
        editor = ViewEditor(
            view_type=self.view_type, pipeline=self.pipeline,
            spec={'pipeline': self.pipeline, 'type': self.view_type}
        )
        editor.param.watch(self._remove_view, 'remove')
        editor.render()
        self.views.append(editor)
        self.param.trigger('views')

    @catch_and_notify
    def _update_pipelines(self, event):
        self.pipelines = list(event.new)
        if not self.pipeline and self.pipelines:
            self.pipeline = self.pipelines[0]

    def _remove_view(self, event):
        self.views.remove(event.obj)
        self.param.trigger('views')


class ViewEditor(ReactiveHTML):

    # Specification parameters

    spec = param.Dict(default={})

    sizing_mode = param.String(default='stretch_both')

    pipeline = param.Parameter(precedence=-1)

    view_type = param.String(default='None')

    # Display parameters

    remove = param.Boolean(default=False)

    view = param.Parameter()

    _template = """
    <span style="font-size: 2em">{{ view_type.title() }} Editor</span>
    <p>Configure the view.</p>
    <div style="width: 300px; display: grid; margin-right: 1em;">
      <label for="view-name"><b>View Name</b></label>
      <fast-text-field id="view-name" placeholder="Enter a name" value="${name}"></fast-text-field>
    </div>
    <fast-divider></fast-divider>
    <div id='view'>${view}</div>
    """

    _dom_events = {'view-name': ['keyup']}

    def __new__(cls, **params):
        if cls is not ViewEditor:
            return super().__new__(cls)
        editors = param.concrete_descendents(cls)
        view_type = params.get('spec', {}).get('type')
        for editor in editors.values():
            if editor.view_type == view_type:
                return super().__new__(editor)
        return super().__new__(cls)

    def __init__(self, **params):
        spec = params.pop('spec', {})
        params.update(**{
            k: v for k, v in spec.items() if k in self.param and k not in params
        })
        super().__init__(spec=spec, **params)

    @property
    def description(self):
        if 'table' in self.spec:
            source = f"{self.spec['table']!r} table"
        elif 'pipeline' in self.spec:
            source = f"{self.spec['pipeline']!r} pipeline"
        else:
            source = 'target source.'
        return f"A {self.view_type} view of the {source}."

    def render(self):
        pass

    @property
    def thumbnail(self):
        return ASSETS_DIR / 'view.png'

    def _remove(self, event):
        self.remove = True

    def _update_spec(self, *events):
        for event in events:
            if not isinstance(event.new, param.Parameterized):
                self.spec[event.name] = event.new


class ViewGalleryItem(GalleryItem):

    editor = param.ClassSelector(class_=ViewEditor, precedence=-1)

    selected = param.Boolean(default=False, doc="""
        Whether the item has been selected.""")

    def __init__(self, **params):
        super().__init__(**params)
        self.thumbnail = self.editor.thumbnail
        self.description = self.editor.description
        self.view = pn.pane.PNG(self.thumbnail, height=200, align='center')
        self._modal_content = [self.editor]

    def _open_modal(self, event):
        self.editor.render()
        super()._open_modal(event)


class ViewGallery(WizardItem, Gallery):
    """
    Select the views to add to your dashboard specification.
    """

    path = param.Foldername()

    spec = param.List()

    views = param.List(default=[], precedence=-1, doc="""
        The list of views added to the dashboard.""")

    _template = """
    <span style="font-size: 1.5em">Views</span>
    <fast-divider></fast-divider>
    <span style="font-size: 1.2em; font-weight: bold;">{{ __doc__ }}</p>
    <div id="items" style="margin: 1em 0; display: flex; flex-wrap: wrap; gap: 1em;">
    {% for item in items %}
      <fast-card id="view-container" class="gallery-item" style="width: 350px; height: 400px;">
        ${item}
      </fast-card>
    {% endfor %}
      <fast-card id="view-container-new" class="gallery-item" style="height: 400px; width: 350px; padding: 1em;">
        <div style="display: grid;">
          <span style="font-size: 1.25em; font-weight: bold;">Add new view</span>
          <i id="add-button" onclick="${_create_new}" class="fa fa-plus" style="font-size: 14em; margin: 0.2em auto;" aria-hidden="true"></i>
        </div>
      </fast-card>
    </div>
    """

    _editor_type = ViewEditor

    _gallery_item = ViewGalleryItem

    def __init__(self, **params):
        super().__init__(**params)
        self._items = self.items
        self._editor = ViewsEditor(spec=self.spec, margin=10)
        self._save_button = pn.widgets.Button(name='Save view')
        self._save_button.on_click(self._save_view)
        self._watchers = {}
        state.pipelines.param.watch(self._update_items, 'items')
        self._update_items()

    def _create_new(self, event):
        if state.modal.objects == [self._editor, self._save_button]:
            state.template.open_modal()
            return
        state.modal.loading = True
        state.template.open_modal()
        state.modal[:] = [self._editor, self._save_button]
        state.modal.loading = False

    def _update_items(self, *events):
        for name, item in state.pipelines.items.items():
            if name not in self._watchers:
                self._watchers[name] = item.param.watch(self._update_items, 'selected')
        for name, item in self.items.items():
            if name not in self._items:
                self._items[name] = item
        self.items = {
            name: item for name, item in self._items.items()
            if item.spec['pipeline'] in lm_state.pipelines
        }
        self.param.trigger('items')

    def _save_view(self, event):
        for view in self._editor.views:
            self.items[view.name] = ViewGalleryItem(
                name=view.name, spec=view.spec, selected=True, editor=view,
                thumbnail=view.thumbnail
            )
            self.views.append(view)
        self.param.trigger('views')
        self.param.trigger('items')
        self._editor.views = []
        state.template.close_modal()

    @param.depends('spec', watch=True)
    def _update_params(self):
        for name, spec in self.spec.items():
            self.views = editor = ViewEditor(name=name, spec=spec)
            self.items[name] = ViewGalleryItem(
                name=name, spec=spec, selected=True, editor=editor,
                thumbnail=editor.thumbnail
            )
        self.param.trigger('views')
        self.param.trigger('items')


class TableViewEditor(ViewEditor):

    page_size = param.Integer(default=15, doc="Declare the page size")

    view_type = param.String(default='table')

    _template = """
    <span style="font-size: 2em">{{ view_type.title() }} Editor</span>
    <p>Configure the view.</p>
    <div style="width: 300px; display: grid; margin-right: 1em;">
      <label for="view-name"><b>View Name</b></label>
      <fast-text-field id="view-name" placeholder="Enter a name" value="${name}"></fast-text-field>
    </div>
    <fast-divider></fast-divider>
    <div id="view">${view}</div>
    """

    _dom_events = {'view-name': ['keyup']}

    def render(self):
        kwargs = dict(self.spec)
        pipeline = lm_state.pipelines[kwargs.pop('pipeline', self.pipeline)]
        if 'sizing_mode' not in kwargs:
            kwargs['sizing_mode'] = 'stretch_width'
        self.tabulator = pn.widgets.Tabulator(
            pipeline.data, pagination='remote', page_size=12, height=400, **kwargs
        )
        controls = ['theme', 'layout', 'page_size', 'pagination']
        control_widgets = self.tabulator.controls(
            controls, margin=(0, 20, 0, 0), jslink=False
        )
        control_widgets.width = 300
        for w in control_widgets:
            w.width = 250
        self.view = pn.Row(
            control_widgets,
            self.tabulator,
            sizing_mode='stretch_width'
        )
        self.tabulator.param.watch(self._update_spec, controls)
        self.tabulator.param.trigger(*controls)

    @property
    def thumbnail(self):
        return ASSETS_DIR / 'tabulator.png'


class PerspectiveViewEditor(ViewEditor):

    view_type = param.String(default='perspective')

    _defaults = dict(sizing_mode='stretch_both', min_height=500, theme='material-dark')

    def __init__(self, **params):
        super().__init__(**{k: v for k, v in params.items() if k in self.param})

    def render(self):
        kwargs = dict(self.spec)
        pipeline = lm_state.pipelines[kwargs.pop('pipeline', self.pipeline)]
        self.view = pn.pane.Perspective(pipeline, **dict(self._defaults, **kwargs))
        self.view.param.watch(self._update_spec, list(self.view.param))

    @property
    def thumbnail(self):
        return ASSETS_DIR / 'perspective.png'


class hvPlotViewEditor(ViewEditor):

    selection_group = param.String(default=None, allow_None=True, precedence=-1)

    view_type = param.String(default='hvplot')

    def __init__(self, **params):
        import hvplot.pandas  # noqa
        super().__init__(**params)

    def render(self):
        from hvplot.ui import hvDataFrameExplorer
        kwargs = dict(self.spec)
        del kwargs['type']
        pipeline = lm_state.pipelines[kwargs.pop('pipeline', self.pipeline)]
        self.view = hvDataFrameExplorer(pipeline.data, **kwargs)
        self.view.param.watch(self._update_spec, list(self.view.param))
        self.view.axes.param.watch(self._update_spec, list(self.view.axes.param))
        self.view.operations.param.watch(self._update_spec, list(self.view.operations.param))
        self.view.style.param.watch(self._update_spec, list(self.view.style.param))
        self.spec['x'] = self.view.x
        self.spec['y'] = self.view.y
        self.spec['kind'] = self.view.kind

    @property
    def thumbnail(self):
        return ASSETS_DIR / 'hvplot.png'

    def _update_spec(self, *events):
        pipeline = self.spec['pipeline']
        self.spec.clear()
        self.spec['type'] = self.view_type
        self.spec['pipeline'] = pipeline
        self.spec.update(self.view.settings())
