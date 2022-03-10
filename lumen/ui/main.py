from pathlib import Path

import panel as pn

from panel.template import FastListTemplate

from lumen.ui.builder import Builder
from lumen.ui.launcher import LocalLauncher
from lumen.ui.state import state

def main():
    path = Path(state.components)
    path.mkdir(parents=True, exist_ok=True)
    params = {'component_dir': str(path)}
    (path / 'dashboards').mkdir(parents=True, exist_ok=True)
    (path / 'sources').mkdir(parents=True, exist_ok=True)
    (path / 'targets').mkdir(parents=True, exist_ok=True)
    (path / 'variables').mkdir(parents=True, exist_ok=True)
    (path / 'views').mkdir(parents=True, exist_ok=True)
    state.modal = pn.Column(sizing_mode='stretch_both')
    state.spec = {'config': {}, 'sources': {}, 'targets': [], 'variables': {}}
    state.template = FastListTemplate(theme='dark', title='Lumen Builder')
    if not state.launcher:
        state.launcher = LocalLauncher
    builder = Builder(
        template=state.template, spec=state.spec, modal=state.modal,
        launcher=state.launcher, **params
    )
    builder.servable()

if __name__.startswith('bokeh'):
    main()
