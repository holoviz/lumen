import io
import tempfile

import panel as pn
import param  # type: ignore
import yaml

from ..dashboard import Dashboard
from ..state import state as lm_state
from ..util import is_ref, resolve_module_reference
from .base import WizardItem
from .gallery import Gallery, GalleryItem
from .state import state


class LauncherGalleryItem(GalleryItem):
    """
    Gallery Item corresponding to a specific Launcher component.
    """

    icon = param.String(doc="The FontAwesome icon to render.")

    launcher = param.Parameter(precedence=-1, doc="The Launcher instance.")

    selected = param.Boolean(default=False, doc="Whether the item was selected.")

    spec = param.Dict(doc="The specification for the Launcher instance.")

    _template = """
    <div id="launcher-item" onclick="${_select}" style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">
      <div style="font-size: 1.25em; font-weight: bold;">{{ name }}</div>
      <div style="text-align: center; font-size: 8em;">
        <i id="icon" class="fas {{ launcher.icon }}"></i>
      </div>
      <p style="height: 4em;">${description}</p>
    </div>
    """

    def __init__(self, launcher, **params):
        params['name'] = launcher.name.replace('Launcher', '')
        description = ''
        for line in launcher.__doc__.split('\n'):
            if line.strip() and not line.startswith('params('):
                description = line
                break
        super().__init__(launcher=launcher, description=description, **params)

    def _select(self, event):
        self.selected = True


class Launcher(WizardItem):
    """
    A WizardItem that can be used to export or launch the dashboard
    once it has been built.
    """

    __abstract = True


class LauncherGallery(WizardItem, Gallery):
    "Launch or download your dashboard specification."

    auto_advance = param.Boolean(default=True)

    path = param.Foldername()

    spec = param.Dict(default={}, precedence=-1)

    _template = """
    <span style="font-size: 2em;">{{ __doc__ }}</span>
    <fast-divider style="margin: 1em 0;"></fast-divider>
    <div id="items" style="margin: 1em 0; display: flex; flex-wrap: wrap; gap: 1em;">
    {% for item in items.values() %}
      <fast-card id="launcher-container" class="gallery-item" style="height: 290px; width: 350px; padding: 1em;">
        ${item}
      </fast-card>
    {% endfor %}
    </div>
    """

    _gallery_item = LauncherGalleryItem

    def __init__(self, **params):
        self.builder = params.pop('builder')
        super().__init__(**params)
        if not self.items:
            self.items['local'] = LauncherGalleryItem(launcher=LocalLauncher)
            self.items['yaml'] = LauncherGalleryItem(launcher=YAMLLauncher)
        for item in self.items.values():
            item.param.watch(self._advance, 'selected')

    def _preprocess_kwargs(self, kwargs):
        spec = kwargs.get('spec', {})
        launcher_type = spec.pop('type')
        launchers = param.concrete_descendents(Launcher)
        if launcher_type in launchers:
            cls = launchers[launcher_type]
        else:
            cls = resolve_module_reference(launcher_type, Launcher)
        for key, value in list(spec.items()):
            if is_ref(value):
                spec[key] = lm_state.resolve_reference(value)
        kwargs['launcher'] = cls
        return kwargs

    def _advance(self, event):
        launcher = event.obj.launcher(**event.obj.spec)
        items = self.builder.wizard.items
        if isinstance(items[-1], Launcher):
            items = items[:-1]
        items.append(launcher)
        self.builder.wizard.items = items
        self.builder.wizard.next_disable = False
        self.builder.wizard._next()


class LocalLauncher(Launcher):
    """
    Launches the dashboard locally.
    """

    icon = param.String(default='fa-window-maximize')

    _template = """
    <span style="font-size: 1.5em">Launcher</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div style="width: 100%">
      <fast-button id="launch-button" style="margin: 2em auto;" onclick="${_launch}">Launch</fast-button>
    </div>
    """

    def _launch(self, event):
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            f.file.write(yaml.dump(state.spec))
        dashboard = Dashboard(f.name)
        dashboard._render_dashboard()
        dashboard.show()


class YAMLLauncher(Launcher):
    """
    Download or copy the dashboard yaml specification.
    """

    download = param.ClassSelector(class_=pn.widgets.FileDownload)

    editor = param.ClassSelector(class_=pn.pane.JSON)

    icon = param.String(default='fa-file-code')

    _template = """
    <span style="font-size: 1.5em">Launcher</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div id="download">${download}</div>
    <div id="yaml">${editor}</div>
    """

    def __init__(self, **params):
        params['download'] = pn.widgets.FileDownload(
            name='Download Dashboard Yaml',
            callback=self._download_data,
            filename='dashboard.yaml',
            margin=(5, 0)
        )
        params['editor'] = pn.pane.JSON(
            sizing_mode='stretch_both',
            margin=0,
            depth=-1
        )
        super().__init__(**params)

    def _download_data(self):
        sio = io.StringIO()
        text = yaml.dump(state.spec)
        sio.write(text)
        sio.seek(0)
        return sio

    @param.depends('active', watch=True)
    def _update_spec(self):
        self.spec = state.spec
        self.editor.object = state.spec
