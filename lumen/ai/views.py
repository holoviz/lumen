import asyncio
import traceback

import panel as pn
import param
import yaml

from panel.config import config
from panel.layout import Column, Row, Tabs
from panel.pane import Alert
from panel.param import ParamMethod
from panel.viewable import Viewer
from panel.widgets import (
    Button, ButtonIcon, Checkbox, CodeEditor, LoadingSpinner,
)
from param.parameterized import discard_events

from ..base import Component
from ..dashboard import load_yaml
from ..downloads import Download
from ..pipeline import Pipeline
from ..transforms.sql import SQLLimit
from ..views.base import Table
from .utils import get_data


class LumenOutput(Viewer):

    active = param.Integer(default=1)

    component = param.ClassSelector(class_=Component)

    loading = param.Boolean()

    footer = param.List()

    render_output = param.Boolean(default=True)

    spec = param.String(allow_None=True)

    title = param.String(allow_None=True)

    language = "yaml"

    def __init__(self, **params):
        if 'spec' not in params and 'component' in params and params['component'] is not None:
            component_spec = params['component'].to_spec()
            params['spec'] = yaml.dump(component_spec)
        super().__init__(**params)
        code_editor = CodeEditor(
            value=self.param.spec,
            language=self.language,
            theme="tomorrow_night" if config.theme == "dark" else "tomorrow",
            sizing_mode="stretch_both",
            on_keyup=False
        )
        code_editor.link(self, bidirectional=True, value='spec')
        copy_icon = ButtonIcon(
            icon="copy", active_icon="check", toggle_duration=1000, description="Copy YAML to clipboard"
        )
        copy_icon.js_on_click(
            args={"code_editor": code_editor},
            code="navigator.clipboard.writeText(code_editor.code);",
        )
        download_icon = ButtonIcon(
            icon="download", active_icon="check", toggle_duration=1000, description="Download YAML to file"
        )
        download_icon.js_on_click(
            args={"code_editor": code_editor},
            code="""
            var text = code_editor.code;
            var blob = new Blob([text], {type: 'text/plain'});
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'lumen_spec.yaml';
            document.body.appendChild(a); // we need to append the element to the dom -> otherwise it will not work in firefox
            a.click();
            a.parentNode.removeChild(a);  //afterwards we remove the element again
            """,
        )
        icons = Row(copy_icon, download_icon, *self.footer)
        code_col = Column(code_editor, icons, sizing_mode="stretch_both")
        if self.render_output:
            placeholder = Column(
                ParamMethod(self.render, inplace=True),
                sizing_mode="stretch_width"
            )
            self._main = Tabs(
                ("Code", code_col),
                ("Output", placeholder),
                styles={'min-width': '100%', 'height': 'fit-content', 'min-height': '300px'},
                active=self.active,
                dynamic=True
            )
            self._main.link(self, bidirectional=True, active='active')
        else:
            self._main = code_col
        self._main.loading = self.param.loading
        self._rendered = False
        self._last_output = {}

    async def _render_pipeline(self, pipeline):
        table = Table(
            pipeline=pipeline, pagination='remote',
            min_height=500, sizing_mode="stretch_both", stylesheets=[
                """
                .tabulator-footer {
                display: flex;
                text-align: left;
                padding: 0px;
                }
                """
            ]
        )
        download = Download(
            view=table, hide=False, filename=f'{pipeline.table}',
            format='csv'
        )
        download_pane = download.__panel__()
        download_pane.sizing_mode = 'fixed'
        controls = Row(
            download_pane,
            styles={'position': 'absolute', 'right': '40px', 'top': '-35px'}
        )
        for sql_limit in pipeline.sql_transforms:
            if isinstance(sql_limit, SQLLimit):
                break
        else:
            sql_limit = None
        if sql_limit:
            data = await get_data(pipeline)
            limited = len(data) == sql_limit.limit
            if limited:
                def unlimit(e):
                    sql_limit.limit = None if e.new else 1_000_000
                full_data = Checkbox(
                    name='Full data', width=100, visible=limited
                )
                full_data.param.watch(unlimit, 'value')
                controls.insert(0, full_data)
        return Column(controls, table)

    @param.depends('spec', 'active')
    async def render(self):
        yield LoadingSpinner(
            value=True, name="Rendering component...", height=50, width=50
        )

        if self.render_output and (self.active != (len(self._main)-1)) or self.spec is None:
            return

        if self.spec in self._last_output:
            yield self._last_output[self.spec]
            return
        elif self.component is None:
            yield Alert(
                "No component to render. Please complete the Config tab.",
                alert_type="warning",
            )
            return

        try:
            if self._rendered:
                yaml_spec = load_yaml(self.spec)
                self.component = type(self.component).from_spec(yaml_spec)
            if isinstance(self.component, Pipeline):
                output = await self._render_pipeline(self.component)
            else:
                output = self.component.__panel__()
            self._rendered = True
            self._last_output.clear()
            self._last_output[self.spec] = output
            yield output
        except Exception as e:
            traceback.print_exc()
            yield Alert(
                f"```\n{e}\n```\nPlease press undo, edit the YAML, or continue chatting.",
                alert_type="danger",
            )

    def __panel__(self):
        return self._main

    def __repr__(self):
        return self.spec

    def __str__(self):
        return f"{self.__class__.__name__}:\n```yaml\n{self.spec}\n```"


class AnalysisOutput(LumenOutput):

    analysis = param.Parameter()

    pipeline = param.Parameter()

    def __init__(self, **params):
        if not params['analysis'].autorun:
            params['active'] = 0
        super().__init__(**params)
        controls = self.analysis.controls()
        if controls is not None or not self.analysis.autorun:
            controls = Column() if controls is None else controls
            if not self.render_output:
                self._main = Tabs(
                    ('Specification', self._main),
                    styles={'min-width': '100%', 'height': 'fit-content', 'min-height': '300px'},
                    active=self.active,
                    dynamic=True
                )
            if self.analysis._run_button:
                run_button = self.analysis._run_button
                run_button.param.watch(self._rerun, 'clicks')
                self._main.insert(1, ('Config', controls))
            else:
                self.analysis._run_button = run_button = Button(
                    icon='player-play', name='Run', on_click=self._rerun,
                    button_type='success', margin=(10, 0, 0 , 10)
                )
                self._main.insert(1, ('Config', pn.Column(controls, run_button)))
            with discard_events(self):
                self._main.active = 2 if self.analysis.autorun else 1
        self._rendered = True

    async def _rerun(self, event):
        with self._main.param.update(loading=True):
            if asyncio.iscoroutinefunction(self.analysis.__call__):
                view = await self.analysis(self.pipeline)
            else:
                view = await asyncio.to_thread(self.analysis, self.pipeline)
            self.component = view
            self._rendered = False
            spec = view.to_spec()
            self.param.update(
                spec=yaml.dump(spec),
                active=2
            )


class SQLOutput(LumenOutput):

    language = "sql"

    @param.depends('spec', 'active')
    async def render(self):
        yield LoadingSpinner(
            value=True, name="Executing SQL query...", height=50, width=50
        )
        if self.active != 1:
            return

        pipeline = self.component
        if self.spec in self._last_output:
            yield self._last_output[self.spec]
            return

        try:
            if self._rendered:
                pipeline.source = pipeline.source.create_sql_expr_source(
                    tables={pipeline.table: self.spec}
                )
            output = await self._render_pipeline(pipeline)
            self._rendered = True
            self._last_output.clear()
            self._last_output[self.spec] = output
            yield output
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield Alert(
                f"```\n{e}\n```\nPlease press undo, edit the YAML, or continue chatting.",
                alert_type="danger",
            )

    def __panel__(self):
        return self._main

    def __str__(self):
        return f"{self.__class__.__name__}:\n```sql\n{self.spec}\n```"
