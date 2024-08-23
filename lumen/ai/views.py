import asyncio

import panel as pn
import param
import yaml

from panel.viewable import Viewer
from param.parameterized import discard_events

from ..base import Component
from ..dashboard import load_yaml
from ..downloads import Download
from ..pipeline import Pipeline
from ..transforms.sql import SQLLimit
from ..views.base import Table


class LumenOutput(Viewer):

    active = param.Integer(default=1)

    component = param.ClassSelector(class_=Component)

    spec = param.String(allow_None=True)

    language = "yaml"

    def __init__(self, **params):
        if 'spec' not in params and 'component' in params and params['component'] is not None:
            component_spec = params['component'].to_spec()
            params['spec'] = yaml.safe_dump(component_spec)
        super().__init__(**params)
        code_editor = pn.widgets.CodeEditor(
            value=self.param.spec, language=self.language, sizing_mode="stretch_both",
        )
        code_editor.link(self, bidirectional=True, value='spec')
        copy_icon = pn.widgets.ButtonIcon(
            icon="copy", active_icon="check", toggle_duration=1000
        )
        copy_icon.js_on_click(
            args={"code_editor": code_editor},
            code="navigator.clipboard.writeText(code_editor.code);",
        )
        download_icon = pn.widgets.ButtonIcon(
            icon="download", active_icon="check", toggle_duration=1000
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
        icons = pn.Row(copy_icon, download_icon)
        code_col = pn.Column(code_editor, icons, sizing_mode="stretch_both")
        placeholder = pn.Column(sizing_mode="stretch_width")
        self._tabs = pn.Tabs(
            ("Code", code_col),
            ("Output", placeholder),
            styles={'min-width': '100%', 'height': 'fit-content', 'min-height': '300px'},
            active=self.active,
            dynamic=True
        )
        self._tabs.link(self, bidirectional=True, active='active')
        self._rendered = False
        placeholder.objects = [
            pn.pane.ParamMethod(self._render_component, inplace=True)
        ]
        self._last_output = {}

    def _render_pipeline(self, pipeline):
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
        controls = pn.Row(
            download_pane,
            styles={'position': 'absolute', 'right': '40px', 'top': '-35px'}
        )
        for sql_limit in pipeline.sql_transforms:
            if isinstance(sql_limit, SQLLimit):
                break
        else:
            sql_limit = None
        if sql_limit:
            limited = len(pipeline.data) == sql_limit.limit
            if limited:
                def unlimit(e):
                    sql_limit.limit = None if e.new else 1_000_000
                full_data = pn.widgets.Checkbox(
                    name='Full data', width=100, visible=limited
                )
                full_data.param.watch(unlimit, 'value')
                controls.insert(0, full_data)
        return pn.Column(controls, table)

    @param.depends('spec', 'active')
    async def _render_component(self):
        yield pn.indicators.LoadingSpinner(
            value=True, name="Rendering component...", height=50, width=50
        )

        if (self.active != (len(self._tabs)-1)) or self.spec is None:
            return

        if self.spec in self._last_output:
            yield self._last_output[self.spec]
            return
        elif self.component is None:
            yield pn.pane.Alert(
                "No component to render. Please complete the Config tab.",
                alert_type="warning",
            )
            return

        try:
            if self._rendered:
                yaml_spec = load_yaml(self.spec)
                self.component = type(self.component).from_spec(yaml_spec)
            if isinstance(self.component, Pipeline):
                output = self._render_pipeline(self.component)
            else:
                output = self.component.__panel__()
            self._rendered = True
            self._last_output.clear()
            self._last_output[self.spec] = output
            yield output
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield pn.pane.Alert(
                f"```\n{e}\n```\nPlease press undo, edit the YAML, or continue chatting.",
                alert_type="danger",
            )

    def __panel__(self):
        return self._tabs

    def __repr__(self):
        return self.spec

    def __str__(self):
        return self.spec


class AnalysisOutput(LumenOutput):

    analysis = param.Parameter()

    pipeline = param.Parameter()

    def __init__(self, **params):
        if not params['analysis'].autorun:
            params['active'] = 0
        super().__init__(**params)
        controls = self.analysis.controls()
        if controls is not None:
            if self.analysis._run_button:
                run_button = self.analysis._run_button
                run_button.param.watch(self._rerun, 'clicks')
                self._tabs.insert(1, ('Config', controls))
            else:
                run_button = pn.widgets.Button(
                    icon='player-play', name='Run', on_click=self._rerun,
                    button_type='success', margin=(10, 0, 0 , 10)
                )
                self._tabs.insert(1, ('Config', pn.Column(controls, run_button)))
            with discard_events(self):
                self._tabs.active = 2 if self.analysis.autorun else 1
        self._rendered = True

    async def _rerun(self, event):
        with self._tabs.param.update(loading=True):
            if asyncio.iscoroutinefunction(self.analysis.__call__):
                view = await self.analysis(self.pipeline)
            else:
                view = await asyncio.to_thread(self.analysis, self.pipeline)
            self.component = view
            self._rendered = False
            spec = view.to_spec()
            self.param.update(
                spec=yaml.safe_dump(spec),
                active=2
            )


class SQLOutput(LumenOutput):

    language = "sql"

    @param.depends('spec', 'active')
    async def _render_component(self):
        yield pn.indicators.LoadingSpinner(
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
                pipeline.source = pipeline.source.create_sql_expr_source(tables={pipeline.table: self.spec})
            output = self._render_pipeline(pipeline)
            self._rendered = True
            self._last_output.clear()
            self._last_output[self.spec] = output
            yield output
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield pn.pane.Alert(
                f"```\n{e}\n```\nPlease press undo, edit the YAML, or continue chatting.",
                alert_type="danger",
            )

    def __panel__(self):
        return self._tabs
