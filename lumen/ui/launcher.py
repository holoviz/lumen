import io
import tempfile
import yaml

import panel as pn
import param

from ..dashboard import Dashboard
from .base import WizardItem
from .state import state


class Launcher(WizardItem):
    """
    A WizardItem that can be used to export or launch the dashboard
    once it has been built.
    """


class LocalLauncher(Launcher):
    """
    Launches the dashboard locally.
    """

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

