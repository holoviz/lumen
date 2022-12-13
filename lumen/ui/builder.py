import os
import pathlib
import uuid

import panel as pn
import param  # type: ignore

from panel.layout.base import ListLike
from panel.template.base import BasicTemplate

from lumen.config import config
from lumen.pipeline import Pipeline
from lumen.state import state as lm_state

from .base import Wizard
from .config import ConfigEditor
from .dashboard import DashboardGallery
from .launcher import LauncherGallery
from .layouts import LayoutEditor, LayoutGallery, LayoutGalleryItem
from .pipeline import PipelineGallery
from .sources import SourceGallery
from .state import state
from .variables import VariablesEditor
from .views import ViewEditor, ViewGallery, ViewGalleryItem


class Builder(param.Parameterized):

    component_dir = param.Path(default=None)

    spec = param.Dict(default={})

    template = param.ClassSelector(class_=BasicTemplate)

    modal = param.ClassSelector(class_=ListLike)

    def __init__(self, **params):
        path = params['component_dir']
        dash_params, launcher_params, pipeline_params, source_params, layout_params, var_params, view_params = (
            {}, {}, {}, {}, {}, {}, {}
        )
        dash_params['path'] = os.path.join(path, 'dashboards')
        launcher_params['path'] = os.path.join(path, 'launchers')
        pipeline_params['path'] = os.path.join(path, 'pipelines')
        source_params['path'] = os.path.join(path, 'sources')
        layout_params['path'] = os.path.join(path, 'layouts')
        var_params['path'] = os.path.join(path, 'variables')
        view_params['path'] = os.path.join(path, 'views')

        spec = params.pop('spec', {})
        if 'pipelines' not in spec:
            spec['pipelines'] = {}
        self.welcome = DashboardGallery(**dash_params)
        if 'launcher' in params:
            params['launcher'] = params['launcher'](spec=spec)
        super().__init__(spec=spec, **params)

        self.config = ConfigEditor(spec=self.spec['config'])
        self.variables = VariablesEditor(spec=self.spec['variables'], **var_params)
        lm_state.spec = state.spec = self.spec
        state.sources = self.sources = SourceGallery(spec=self.spec['sources'], **source_params)
        state.pipelines = self.pipelines = PipelineGallery(spec=self.spec['pipelines'], **pipeline_params)
        state.views = self.views = ViewGallery(**view_params)
        state.layouts = self.layouts = LayoutGallery(spec=self.spec['layouts'], **layout_params)
        self.launcher = LauncherGallery(builder=self, **launcher_params)
        self.wizard = Wizard(items=[
            self.welcome, self.config, self.variables, self.sources,
            self.pipelines, self.views, self.layouts, self.launcher
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

        for name, pipeline in self.pipelines.pipelines.items():
            config.root = str(pathlib.Path(__file__).parent)
            spec = pipeline.spec
            if isinstance(pipeline.spec, dict):
                spec = dict(pipeline.spec)
                spec['name'] = name
            try:
                lm_state.pipelines[name] = Pipeline.from_spec(spec)
            except Exception:
                pass

        self.config.param.trigger('spec')
        self.sources.param.trigger('spec')
        self.pipelines.param.trigger('spec')

        views, view_gallery = [], {}
        layouts, layout_items = [], {}
        for layout in self.spec['layouts']:
            view_specs = layout['views']
            if isinstance(view_specs, list):
                specs = {}
                for view in view_specs:
                    name = f"{view['type']}: {view['table']}"
                    while name in specs:
                        if name[-2] == ' ' and name[-1].isdigit():
                            name = f'{name[:-2]} {int(name[-1])+1}'
                        else:
                            name = f'{name} 2'
                    specs[name] = view
                view_specs = specs
            view_specs = view_specs.items()
            layout_views = []
            for name, view in view_specs:
                view = dict(view)
                view_type = view.get('type')
                view_editor = ViewEditor(
                    view_type=view_type, name=name, spec=view, pipeline=layout.get('pipeline')
                )
                view_gallery[name] = ViewGalleryItem(
                    editor=view_editor, name=name, selected=True, spec=view
                )
                views.append(view_editor)
                layout_views.append(name)

            layout_editor = LayoutEditor(spec=layout, views=layout_views)
            item = LayoutGalleryItem(spec=layout, editor=layout_editor)
            layouts.append(layout_editor)
            layout_items[f'layout{uuid.uuid4().hex}'] = item

        self.layouts.param.set_param(layouts=layouts, items=layout_items)
        self.views.param.set_param(views=views, items=view_gallery)
        self.welcome.ready = True
        self.wizard.loading = False

    def servable(self, title='Lumen Builder'):
        self.template.servable(title=title)
