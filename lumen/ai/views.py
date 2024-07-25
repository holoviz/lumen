import panel as pn
import param
import yaml

from panel.viewable import Viewer

from ..base import Component
from ..dashboard import load_yaml
from ..downloads import Download
from ..pipeline import Pipeline
from ..views.base import Table
from .memory import memory


class LumenOutput(Viewer):

    active = param.Integer(default=1)

    component = param.ClassSelector(class_=Component)

    spec = param.String()

    language = "yaml"

    def __init__(self, **params):
        if 'spec' not in params and 'component' in params:
            component_spec = params['component'].to_spec()
            params['spec'] = yaml.safe_dump(component_spec)
        super().__init__(**params)
        code_editor = pn.widgets.CodeEditor(
            value=self.spec, language=self.language, sizing_mode="stretch_both",
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
            styles={'min-width': "100%"},
            height=700,
            active=1
        )
        self._tabs.link(self, bidirectional=True, active='active')
        placeholder.objects = [
            pn.pane.ParamMethod(self._render_component, inplace=True)
        ]

    @param.depends('spec', 'active')
    async def _render_component(self):
        yield pn.indicators.LoadingSpinner(
            value=True, name="Rendering component...", height=50, width=50
        )

        if self.active != 1:
            return

        # store the spec in the cache instead of memory to save tokens
        memory["current_spec"] = self.spec
        try:
            self.component = type(self.component).from_spec(load_yaml(self.spec))
            if isinstance(self.component, Pipeline):
                table = Table(
                    pipeline=self.component, pagination='remote', page_size=21,
                    min_height=650, sizing_mode="stretch_both", stylesheets=[
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
                    view=table, hide=False, filename=f'{self.component.table}',
                    format='csv'
                )
                download_pane = download.__panel__()
                download_pane.sizing_mode = 'fixed'
                download_pane.styles = {'position': 'absolute', 'right': '-50px'}
                output = pn.Column(download_pane, table)
            else:
                output = self.component.__panel__()
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
        pipeline.source = pipeline.source.create_sql_expr_source({pipeline.table: self.spec})
        try:
            table = Table(
                pipeline=pipeline, pagination='remote',
                height=458, page_size=12
            )
            download = Download(
                view=table, hide=False, filename=f'{self.component.table}',
                format='csv'
            )
            download_pane = download.__panel__()
            download_pane.sizing_mode = 'fixed'
            download_pane.styles = {'position': 'absolute', 'right': '-50px'}
            output = pn.Column(download_pane, table)
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
