import pathlib

import panel as pn
import param  # type: ignore
import yaml

from .base import WizardItem
from .gallery import Gallery, GalleryItem
from .state import state


class DashboardGalleryItem(GalleryItem):

    thumbnail = param.Filename(default=pathlib.Path(__file__).parent / 'assets' / 'dashboard.png', precedence=-1)

    _template = """
    <div id="dashboard-item" onclick="${_launch}">
      <span style="font-size: 1.25em; font-weight: bold;">{{ name }}</span>
      <div id="details" style="margin-top: 1em;">
        ${view}
      </div>
      <p style="height: 4em;">${description}</p>
    </div>
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.view = pn.pane.PNG(
            str(self.thumbnail), height=150, margin=0, align='center'
        )

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
          <i id="add-button" onclick="${_create_new}" class="fa fa-plus" style="font-size: 12em; margin: 0.2em auto;" aria-hidden="true"></i>
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
        self.spec = {'sources': {}, 'layouts': []}

    def _selected(self, event):
        self.spec = event.obj.spec
