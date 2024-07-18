import panel as pn
import panel.config as _pn_config
import param
import yaml

from panel.io.resources import CSS_URLS
from panel.template.base import BaseTemplate
from panel.viewable import Viewer

from ..base import Component
from ..dashboard import load_yaml
from ..downloads import Download
from ..pipeline import Pipeline
from ..views.base import Table
from .memory import memory

if CSS_URLS["font-awesome"] not in _pn_config.css_files:
    _pn_config.css_files.append(CSS_URLS["font-awesome"])


class LumenOutput(Viewer):

    component = param.ClassSelector(class_=Component)

    template = param.ClassSelector(class_=BaseTemplate)

    spec = param.String()

    language = "yaml"

    def __init__(self, **params):
        if 'spec' not in params and 'component' in params:
            component_spec = params['component'].to_spec()
            params['spec'] = yaml.safe_dump(component_spec)
        super().__init__(**params)
        view_button = pn.widgets.Button(
            name="View Output",
            button_type="primary",
            stylesheets=[
                """
                :host(.solid) .bk-btn.bk-btn-primary {
                    position: relative;
                    background-color: #1a1a1a;
                    font-size: 1.5em;
                    text-align: left;
                    padding-left: 45px;
                    padding-bottom: 10px;
                    font-weight: 350;
                    width: fit-content;
                }
                .bk-btn.bk-btn-primary::before {
                    content: "\\f56e"; /* Unicode for Font Awesome code icon */
                    font-family: "Font Awesome 5 Free"; /* Specify the Font Awesome font */
                    font-weight: 600; /* Ensure the correct font weight for the icon */
                    -webkit-text-stroke: 1px #1a1a1a;
                    position: absolute;
                    top: 50%;
                    left: 8px;
                    transform: translateY(-50%) rotateY(180deg);
                    width: 100%;
                    text-align: right;
                    font-size: 1em;
                    color: white;
                }
                .bk-btn.bk-btn-primary::after {
                    content: "Click to open content in the sidebar";
                    display: block;
                    font-size: 50%;
                    text-align: left;
                    font-weight: 50%;
                }
                """
            ],
            height=75,
            on_click=self._view_content,
        )
        self._view_button = view_button

        placeholder = pn.Column(sizing_mode="stretch_width", min_height=500)
        placeholder.objects = [pn.pane.ParamMethod(self._render_component, inplace=True)]

        divider = pn.layout.Divider(width=10, height=10)

        code_editor = pn.widgets.CodeEditor(
            value=self.spec, language=self.language, min_height=350, sizing_mode="stretch_both",
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

        self._content = pn.Column(placeholder, divider, code_editor, icons, sizing_mode="stretch_both")
        view_button.param.trigger("clicks")

    def _view_content(self, event):
        self.template.sidebar[0].objects = [self._content]
        self.template.sidebar[-1].object = "<script> openNav(); </script>"

    @param.depends('spec')
    async def _render_component(self):
        yield pn.indicators.LoadingSpinner(
            value=True, name="Rendering component...", height=50, width=50
        )

        # store the spec in the cache instead of memory to save tokens
        memory["current_spec"] = self.spec
        try:
            self.component = type(self.component).from_spec(load_yaml(self.spec))
            if isinstance(self.component, Pipeline):
                table = Table(
                    pipeline=self.component, pagination='remote',
                    height=458, page_size=12
                )
                download = Download(
                    view=table, hide=False, filename=f'{self.component.table}',
                    format='csv'
                )
                download_pane = download.__panel__()
                download_pane.sizing_mode = 'fixed'
                download_pane.styles = {'position': 'absolute', 'right': '-50px'}
                output = pn.Column(download_pane, table, sizing_mode='stretch_both')
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
        return self._view_button

    def __repr__(self):
        return self.spec


class SQLOutput(LumenOutput):

    language = "sql"

    @param.depends('spec')
    async def _render_component(self):
        yield pn.indicators.LoadingSpinner(
            value=True, name="Executing SQL query...", height=50, width=50
        )

        pipeline = self.component
        pipeline.source = pipeline.source.create_sql_expr_source({pipeline.table: self.spec})
        try:
            table = Table(
                pipeline=pipeline, pagination='remote',
                height=458, page_size=12, sizing_mode='stretch_both'
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
        return self._view_button
