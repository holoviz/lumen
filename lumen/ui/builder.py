import os
import pathlib
import uuid

import panel as pn
import param

from panel.layout.base import ListLike
from panel.template.base import BasicTemplate

from lumen.config import config
from lumen.sources import Source
from lumen.state import state as lm_state

from .base import Wizard
from .config import ConfigEditor
from .dashboard import DashboardGallery
from .launcher import LauncherGallery
from .sources import SourceGallery
from .state import state
from .targets import TargetGallery, TargetEditor, TargetGalleryItem
from .variables import VariablesEditor
from .views import ViewEditor, ViewGallery, ViewGalleryItem

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

pn.extension('ace', 'perspective', 'tabulator', raw_css=[CSS_RAW])


class Builder(param.Parameterized):

    component_dir = param.Path(default=None)

    spec = param.Dict(default={})

    template = param.ClassSelector(class_=BasicTemplate)

    modal = param.ClassSelector(class_=ListLike)

    def __init__(self, **params):
        path = params['component_dir']
        dash_params, launcher_params, source_params, target_params, var_params, view_params = (
            {}, {}, {}, {}, {}, {}
        )
        dash_params['path'] = os.path.join(path, 'dashboards')
        launcher_params['path'] = os.path.join(path, 'launchers')
        source_params['path'] = os.path.join(path, 'sources')
        target_params['path'] = os.path.join(path, 'targets')
        var_params['path'] = os.path.join(path, 'variables')
        view_params['path'] = os.path.join(path, 'views')

        spec = params.pop('spec', {})
        self.welcome = DashboardGallery(**dash_params)
        if 'launcher' in params:
            params['launcher'] = params['launcher'](spec=spec)
        super().__init__(spec=spec, **params)

        self.config = ConfigEditor(spec=self.spec['config'])
        self.variables = VariablesEditor(spec=self.spec['variables'], **var_params)
        state.spec = self.spec
        state.sources = self.sources = SourceGallery(spec=self.spec['sources'], **source_params)
        state.views = self.views = ViewGallery(**view_params)
        state.targets = self.targets = TargetGallery(spec=self.spec['targets'], **target_params)
        self.launcher = LauncherGallery(builder=self, **launcher_params)
        self.wizard = Wizard(items=[
            self.welcome, self.config, self.variables, self.sources,
            self.views, self.targets, self.launcher
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
        self.wizard.preview.object = dict(state.spec)
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

        for name, source in self.sources.sources.items():
            config.root = str(pathlib.Path(__file__).parent)
            spec = source.spec
            if isinstance(source.spec, dict):
                spec = dict(source.spec)
                spec['name'] = name
                spec.pop('filters', None)
            try:
                lm_state.sources[name] = Source.from_spec(spec)
            except Exception:
                pass

        self.config.param.trigger('spec')
        self.sources.param.trigger('spec')

        views, view_gallery = [], {}
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
            target_views = []
            for name, view in view_specs:
                view = dict(view)
                view_type = view.pop('type')
                view_editor = ViewEditor(
                    view_type=view_type, name=name, source_obj=source,
                    spec=view
                )
                view_gallery[name] = ViewGalleryItem(
                    editor=view_editor, name=name, selected=True
                )
                views.append(view_editor)
                target_views.append(name)

            target_editor = TargetEditor(spec=target, views=target_views)
            item = TargetGalleryItem(spec=target, editor=target_editor)
            targets.append(target_editor)
            target_items[f'target{uuid.uuid4().hex}'] = item

        self.targets.param.set_param(targets=targets, items=target_items)
        self.views.param.set_param(views=views, items=view_gallery)
        self.welcome.ready = True
        self.wizard.loading = False

    def servable(self, title='Lumen Builder'):
        self.template.servable(title=title)
