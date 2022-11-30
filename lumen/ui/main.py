from pathlib import Path

import panel as pn

from panel.io.resources import CSS_URLS
from panel.template import FastListTemplate

from lumen.ui.builder import Builder
from lumen.ui.state import state

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

.card-margin {
  height: calc(100% - 40px);
}

.bk-root {
  height: 100% !important;
}
"""

pn.extension(
    'ace', 'perspective', 'tabulator', raw_css=[CSS_RAW],
    css_files=[CSS_URLS['font-awesome']], notifications=True
)

def main():
    path = Path(state.components)
    path.mkdir(parents=True, exist_ok=True)
    params = {'component_dir': str(path)}
    (path / 'dashboards').mkdir(parents=True, exist_ok=True)
    (path / 'launchers').mkdir(parents=True, exist_ok=True)
    (path / 'pipelines').mkdir(parents=True, exist_ok=True)
    (path / 'sources').mkdir(parents=True, exist_ok=True)
    (path / 'layouts').mkdir(parents=True, exist_ok=True)
    (path / 'variables').mkdir(parents=True, exist_ok=True)
    (path / 'views').mkdir(parents=True, exist_ok=True)
    state.modal = pn.Column(sizing_mode='stretch_both')
    state.spec = {'config': {}, 'sources': {}, 'layouts': [], 'variables': {}}
    state.template = FastListTemplate(theme='dark', title='Lumen Builder')
    builder = Builder(
        template=state.template, spec=state.spec, modal=state.modal, **params
    )
    builder.servable()

if __name__.startswith('bokeh'):
    main()
