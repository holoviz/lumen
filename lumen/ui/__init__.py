import os
import pathlib
import uuid
import tempfile
import yaml

import panel as pn
import param

from panel.layout.base import ListLike
from panel.template.base import BasicTemplate
from panel.template import FastListTemplate

from lumen.config import config
from lumen.sources import Source
from lumen.state import state as lm_state

from .base import Wizard, WizardItem
from .config import ConfigEditor
from .gallery import Gallery, GalleryItem
from .sources import SourceGallery
from .state import state
from .targets import TargetGallery, TargetEditor, TargetGalleryItem
from .views import ViewEditor, ViewGallery, ViewGalleryItem

CSS = [
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
]

CSS_RAW = """
i.fa.fa-plus:hover {
  color: var(--neutral-fill-color) !important;
}

.gallery-item:hover {
  box-shadow: 0 1px 5px;
}

.gallery-item {
  cursor: pointer;
}
"""

pn.extension('ace', 'perspective', 'tabulator', css_files=CSS, raw_css=[CSS_RAW])


class DashboardGalleryItem(GalleryItem):

    _template = """
    <div id="dashboard-item" onclick="${_launch}">
      <span style="font-size: 1.25em; font-weight: bold;">{{ name }}</span>
      <div id="details" style="margin-top: 1em;">
        ${view}
      </div>
      <p style="height: 4em;">${description}</p>
    </div>
    """

    _thumbnail = pathlib.Path(__file__).parent / 'assets' / 'dashboard.png'

    def __init__(self, **params):
        super().__init__(**params)
        metadata = self.spec.get('metadata', {})
        thumbnail = metadata.get('thumbnail', self._thumbnail)
        self.description = metadata.get('description')
        self.view = pn.pane.PNG(str(thumbnail), height=150, margin=0, align='center')

    def _launch(self, event):
        self.selected = True



class DashboardGallery(WizardItem, Gallery):
    "Welcome to the Lumen Dashboard Builder!"

    auto_advance = param.Boolean(default=True)

    path = param.Foldername()

    spec = param.Dict(default={}, precedence=-1)

    _template = """
    <span style="font-size: 2em;">{{ __doc__ }}</span>
    <fast-divider style="margin: 1em 0;"></fast-divider>
    <span style="font-size: 1.5em">Select an existing dashboard, start from scratch or upload a dashboard to edit.</span>
    <fast-button id="upload" onclick="${_open_modal}" appearance="outline" style="float: right;">
        <b style="font-size: 1.5em;">+</b>
    </fast-button>
    <div id="items" style="margin: 1em 0; display: flex; flex-wrap: wrap; gap: 1em;">
    {% for item in items.values() %}
      <fast-card id="dashboard-container" class="gallery-item" style="height: 290px; width: 350px; padding: 1em;">
        ${item}
      </fast-card>
    {% endfor %}
      <fast-card id="dashboard-container-new" class="gallery-item" style="height: 290px; width: 350px; padding: 1em;">
        <div style="display: grid;">
          <span style="font-size: 1.25em; font-weight: bold;">Create new dashboard</span>
          <i id="add-button" onclick="${_create_new}" class="fa fa-plus" style="font-size: 14em; margin: 0.2em auto;" aria-hidden="true"></i>
        </div>
      </fast-card>
    </div>
    """

    _gallery_item = DashboardGalleryItem

    def __init__(self, **params):
        super().__init__(**params)
        self.editor = pn.widgets.Ace(
            language='yml', sizing_mode='stretch_both', theme='dracula',
            margin=(0, 20), height=400
        )
        self.upload = pn.widgets.FileInput(
            sizing_mode='stretch_width', margin=(0, 20), height=50
        )
        self.thumbnail_upload = pn.widgets.FileInput(
            sizing_mode='stretch_width', margin=0, name='Upload a thumbnail'
        )
        self.upload.param.watch(self._yaml_upload, 'value')
        self.namer = pn.widgets.TextInput(
            name='Dashboard name', placeholder='Give the dashboard a name',
            margin=(5, 0)
        )
        self.description = pn.widgets.TextAreaInput(
            name='Provide a description', sizing_mode='stretch_both',
            max_length=100
        )
        self.save = pn.widgets.Button(name='Save', sizing_mode='stretch_width')
        self.save.on_click(self._save)
        self._modal_content = [
            '# Upload dashboard\nUpload a yaml file',
            self.upload,
            pn.Row(
                pn.Column(self.namer, self.thumbnail_upload),
                self.description, sizing_mode='stretch_width',
                margin=(0, 20)
            ),
            self.editor,
            self.save
        ]

    def _save(self, event):
        name = self.namer.value
        spec = yaml.safe_load(self.editor.value)
        spec['metadata'] = {
            'description': self.description.value
        }
        if self.thumbnail_upload.value:
            ext = self.thumbnail_upload.filename.split('.')[-1]
            thumbnail_path = pathlib.Path(self.path) / f'{name}.{ext}'
            with open(thumbnail_path, 'wb') as f:
                f.write(self.thumbnail_upload.value)
            spec['metadata']['thumbnail'] = str(thumbnail_path)
        with open(pathlib.Path(self.path) / f'{name}.yml', 'w', encoding='utf-8') as f:
            f.write(yaml.dump(spec))
        self.items[name] = self._gallery_item(name=name, spec=spec)
        self.param.trigger('items')
        state.template.close_modal()
        self._reset_form()

    def _reset_form(self):
        self.namer.value = ''
        self.editor.value = ''
        self.description.value = ''
        self.upload.filename = ''
        self.upload.value = None
        self.thumbnail_upload.filename = ''
        self.thumbnail_upload.value = None

    def _yaml_upload(self, event):
        if event.new is None:
            return
        name = '.'.join(self.upload.filename.split('.')[:-1])
        self.namer.value = name
        self.editor.value = event.new.decode('utf-8')
        return

    def _create_new(self, event):
        self.spec = {'config': {}, 'sources': {}, 'targets': []}

    def _selected(self, event):
        self.spec = event.obj.spec


class Launcher(WizardItem):
    "WIP: Plugins will allow launching to various cloud providers."

    _template = """
    <span style="font-size: 1.5em">Launcher</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div style="width: 100%">
      <fast-button id="launch-button" style="margin: 2em auto;" onclick="${_launch}">Launch</fast-button>
    </div>
    """

    def _launch(self, event):
        from lumen import Dashboard
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            f.file.write(yaml.dump(state.spec))
        dashboard = Dashboard(f.name)
        dashboard._render_dashboard()
        dashboard.show()


class YAML(WizardItem):
    """
    Download or copy the dashboard yaml specification.
    """

    editor = param.ClassSelector(class_=pn.widgets.Ace)

    _template = """
    <span style="font-size: 1.5em">Launcher</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div id="yaml">${editor}</div>
    """

    def __init__(self, spec, **params):
        yaml_text = yaml.dump(spec)
        params['editor'] = pn.widgets.Ace(value=yaml_text,
            language='yaml', sizing_mode='stretch_both', theme='dracula',
            margin=0
        )
        super().__init__(spec=spec, **params)

    @param.depends('active', watch=True)
    def _update_spec(self):
        self.spec = state.spec
        self.editor.value = yaml.dump(state.spec)


class Builder(param.Parameterized):

    component_dir = param.Path(default=None)

    spec = param.Dict(default={})

    template = param.ClassSelector(class_=BasicTemplate)

    modal = param.ClassSelector(class_=ListLike)

    def __init__(self, **params):
        path = params['component_dir']
        dash_params, source_params, target_params, view_params = {}, {}, {}, {}
        dash_params['path'] = os.path.join(path, 'dashboards')
        source_params['path'] = os.path.join(path, 'sources')
        target_params['path'] = os.path.join(path, 'targets')
        view_params['path'] = os.path.join(path, 'views')
        self.welcome = DashboardGallery(**dash_params)
        super().__init__(**params)

        self.config = ConfigEditor(spec=self.spec['config'])
        state.spec = self.spec
        state.sources = self.sources = SourceGallery(spec=self.spec['sources'], **source_params)
        state.views = self.views = ViewGallery(**view_params)
        state.targets = self.targets = TargetGallery(spec=self.spec['targets'], **target_params)
        self.launcher = YAML(spec=self.spec)
        self.wizard = Wizard(items=[
            self.welcome, self.config, self.sources, self.views, self.targets, self.launcher
        ], sizing_mode='stretch_both')

        preview = pn.widgets.Button(name='Preview', width=100)
        preview.on_click(self._open_dialog)

        state.template.header.append(preview)
        if pn.state.user:
            logout = pn.widgets.Button(
                name='✖', width=40, css_classes=['logout'], margin=0,
                align='center'
            )
            logout.js_on_click(code="""
            window.location.href = '/logout'
            """)
            self.template.header.extend([
                pn.layout.HSpacer(),
                f'<b><font size="4.5em">User: {pn.state.user}</font></b>',
                logout
            ])
        self.template.main.append(self.wizard)
        self.template.modal.append(self.modal)

    def _open_dialog(self, event):
        self.wizard.preview.object = dict(self.spec)
        self.wizard.open_modal()

    @param.depends('welcome.spec', watch=True)
    def _update_spec(self):
        self.wizard.loading = True
        for key, item in self.welcome.spec.items():
            if isinstance(item, list):
                self.spec[key][:] = item
            else:
                if key in self.spec:
                    self.spec[key].clear()
                    self.spec[key].update(item)
                else:
                    self.spec[key] = item
        self.config.param.trigger('spec')
        self.sources.param.trigger('spec')

        for name, source in self.sources.sources.items():
            config.root = str(pathlib.Path(__file__).parent)
            spec = source.spec
            if isinstance(source.spec, dict):
                spec = dict(source.spec)
                spec.pop('filters', None)
            lm_state.sources[name] = Source.from_spec(spec)

        views, editors = [], {}
        targets, target_items = [], {}
        for target in self.spec['targets']:
            source_spec = target['source']
            if isinstance(source_spec, dict):
                source_spec = dict(source_spec, cache_dir=None)
                source_spec.pop('filters', None)
            source = Source.from_spec(source_spec)
            view_specs = target['views']
            if isinstance(view_specs, list):
                view_specs = [(f"{view['type']}: {view['table']}", view) for view in view_specs]
            else:
                view_specs = view_specs.items()
            for name, view in view_specs:
                view = dict(view)
                view_type = view.pop('type')
                editor = ViewEditor(
                    view_type=view_type, name=name, source_obj=source,
                    spec=view
                )
                item = ViewGalleryItem(
                    editor=editor, name=name, selected=True
                )
                while name in editors:
                    editors[name] = item
                views.append(editor)

            editor = TargetEditor(spec=target, views=list(editors))
            item = TargetGalleryItem(spec=target, editor=editor)
            targets.append(editor)
            target_items[f'target{uuid.uuid4().hex}'] = item

        self.targets.param.set_param(targets=targets, items=target_items)
        self.views.param.set_param(views=views, items=editors)
        self.welcome.ready = True
        self.wizard.loading = False
        self.launcher.value = yaml.dump(self.spec)

    def servable(self, title='Lumen Builder'):
        self.template.servable(title=title)


def main():
    path = pathlib.Path('./components').absolute()
    path.mkdir(exist_ok=True)
    params = {'component_dir': str(path)}
    (path / 'dashboards').mkdir(parents=True, exist_ok=True)
    (path / 'sources').mkdir(parents=True, exist_ok=True)
    (path / 'targets').mkdir(parents=True, exist_ok=True)
    (path / 'views').mkdir(parents=True, exist_ok=True)
    state.modal = pn.Column(sizing_mode='stretch_both')
    state.spec = {'config': {}, 'sources': {}, 'targets': []}
    state.template = FastListTemplate(theme='dark', title='Lumen Builder')
    builder = Builder(template=state.template, spec=state.spec, modal=state.modal, **params)
    builder.servable()


if __name__.startswith('bokeh'):
    main()
