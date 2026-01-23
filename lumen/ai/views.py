from __future__ import annotations

import asyncio
import json
import re
import textwrap
import traceback

from copy import deepcopy
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Any

import panel as pn
import param
import requests

from jsonschema import Draft7Validator, ValidationError
from panel.config import config
from panel.layout import Row
from panel.pane import panel as as_panel
from panel.param import ParamMethod
from panel.viewable import Viewable, Viewer
from panel.widgets import CodeEditor
from panel_gwalker import GraphicWalker
from panel_material_ui import (
    Alert, Button, Checkbox, CircularProgress, Column, FileDownload,
    MenuButton,
)

from ..base import Component
from ..config import dump_yaml, load_yaml
from ..pipeline import Pipeline
from ..transforms.sql import SQLLimit
from ..views.base import Panel, Table, View
from .analysis import Analysis
from .config import FORMAT_ICONS, FORMAT_LABELS, VEGA_ZOOMABLE_MAP_ITEMS
from .controls import AnnotationControls, CopyControls, RetryControls
from .utils import describe_data, get_data

if TYPE_CHECKING:
    from panel.chat.feed import ChatFeed

    from .report import Task


class LumenOutput(Viewer):

    component = param.ClassSelector(class_=Component)

    loading = param.Boolean()

    footer = param.List(default=[])

    spec = param.String(allow_None=True)

    title = param.String(allow_None=True)

    export_formats = ("yaml",)

    language = "yaml"

    _controls = [CopyControls, RetryControls]
    _label = "Result"

    def __init__(self, **params):
        try:
            spec, spec_dict = self._serialize_component(params["component"])
        except Exception:
            spec_dict = None
            spec = None
        if "spec" not in params:
            params["spec"] = spec
        self._spec_dict = spec_dict
        super().__init__(**params)
        self.editor = self._render_editor()
        self.view = ParamMethod(self.render, inplace=True, sizing_mode='stretch_width')
        self._rendered = False
        self._last_output = {}

    def _render_editor(self):
        self._editor = CodeEditor(
            value=self.param.spec.rx.or_(f'{self.title} output could not be serialized and may therefore not be edited.'),
            language=self.language,
            theme="github_dark" if config.theme == "dark" else "github_light_default",
            sizing_mode="stretch_both",
            soft_tabs=True,
            on_keyup=False,
            indent=2,
            margin=(0, 10),
            disabled=self.param.spec.rx.is_(None),
            styles={"border": "1px solid var(--border-color)"}
        )
        self._editor.link(self, bidirectional=True, value='spec')
        self._icons = Row(
            *self.footer,
            margin=(0, 0, 5, 10)
        )
        return Column(
            self._editor,
            self._icons,
            loading=self.param.loading,
            sizing_mode="stretch_both"
        )

    @classmethod
    def _class_name_to_download_filename(cls, fmt) -> str:
        """Convert class name to snake_case for filenames."""
        name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', cls.__name__)
        return re.sub(r'([a-z\d])([A-Z])', r'\1_\2', name).lower() + f'.{fmt}'

    def render_controls(self, task: Task, interface: ChatFeed):
        export_menu = MenuButton(
            label=f"Export {self._label} as", variant='text', icon="file_download", margin=0,
            items=[
                {
                    "label": FORMAT_LABELS.get(fmt, f"{fmt.upper()}"),
                    "format": fmt,
                    "icon": FORMAT_ICONS.get(fmt, "insert_drive_file")
                } for fmt in self.export_formats
            ],
            on_click=lambda _: file_download._transfer()
        )

        def download_export(item):
            fmt = item["format"]
            file_download.filename = self._class_name_to_download_filename(fmt)
            return self.export(fmt)

        file_download = FileDownload(
            auto=True, callback=param.bind(download_export, export_menu),
            filename=self.title or 'output', visible=False
        )
        export_menu.attached.append(file_download)

        return Row(
            export_menu,
            *(control(interface=interface, task=task, view=self) for control in self._controls),
            sizing_mode="stretch_width",
            margin=(0, 10)
        )

    def export(self, fmt: str) -> StringIO | BytesIO:
        if fmt not in self.export_formats:
            raise ValueError(f"Unknown export format {fmt!r} for {self.__class__.__name__}")
        if fmt == "yaml":
            return StringIO(self.spec)

    @classmethod
    def _serialize_component(cls, component: Component, spec_dict: dict[str, Any] | None = None) -> str:
        component_spec = spec_dict or component.to_spec()
        if isinstance(component, Panel):
            component_spec = component_spec["object"]
        return dump_yaml(component_spec), component_spec

    @classmethod
    def _deserialize_component(
        cls, component: Component, yaml_spec: str, spec_dict: dict[str, Any], pipeline: Pipeline | None = None
    ) -> str:
        spec = load_yaml(yaml_spec)
        cls.validate_spec(spec)
        if isinstance(component, Panel):
            spec = {"type": "panel", "object": spec}
        if pipeline is not None and 'pipeline' in spec:
            del spec['pipeline']
        return type(component).from_spec(spec, pipeline=pipeline)

    @classmethod
    def validate_spec(cls, spec):
        return spec

    @param.depends("footer", watch=True)
    def _update_footer(self):
        self._icons[:] = self.footer

    @param.depends("spec", watch=True)
    def _update_component(self):
        if self.spec in self._last_output and self.component is None:
            return
        pipeline = getattr(self.component, 'pipeline', None)
        self.component = self._deserialize_component(
            self.component, self.spec, self._spec_dict, pipeline=pipeline
        )

    async def _render_pipeline(self, pipeline):
        table = Table(
            pipeline=pipeline, pagination='remote',
            min_height=200, sizing_mode="stretch_both", stylesheets=[
            """
            .tabulator-footer {
            display: flex;
            text-align: left;
            padding: 0px;
            }
            """
            ]
        )
        controls = Row(
            styles={'position': 'absolute', 'right': '40px', 'top': '-35px'}
        )
        for sql_limit in pipeline.sql_transforms:
            if isinstance(sql_limit, SQLLimit):
                break
        else:
            sql_limit = None
        if sql_limit:
            data = pipeline.data
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

    async def render_context(self):
        return {"view": self.component}

    @param.depends('spec')
    async def render(self):
        if self.component is None:
            yield Alert(
                margin=5,
                severity="warning",
                sizing_mode="stretch_width",
                title="Run the analysis."
            )
            return

        yield CircularProgress(value=True, label="Rendering component...")

        if self.spec in self._last_output:
            yield self._last_output[self.spec]
            return

        try:
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
                title=f"**{type(e).__name__}**:{e}\n\nPlease press undo, edit the YAML, or continue chatting.",
                margin=5,
                severity="error",
                sizing_mode="stretch_width"
            )

    def __panel__(self):
        return self.view

    def __repr__(self):
        return self.spec

    def __str__(self):
        return f"{self.__class__.__name__}:\n```yaml\n{self.spec}\n```"


class VegaLiteOutput(LumenOutput):

    export_formats = ("yaml", "png", "jpeg", "pdf", "svg", "html")

    _controls = [CopyControls, RetryControls, AnnotationControls]
    _label = "Plot"

    def export(self, fmt: str) -> StringIO | BytesIO:
        ret = super().export(fmt)
        if ret is not None:
            return ret
        kwargs = {"scale": 2} if fmt in ("png", "jpeg", "pdf") else {}
        # Replace 'container' with actual dimensions for image and HTML exports
        if fmt in ("png", "jpeg", "pdf", "svg", "html"):
            spec = load_yaml(self.spec)
            # Replace 'container' with actual pixel values or use defaults
            if spec.get("width") == "container" or "width" not in spec:
                spec["width"] = 800
            if spec.get("height") == "container" or "height" not in spec:
                spec["height"] = 400
            # Temporarily update spec for export
            with self.param.update(spec=dump_yaml(spec)):
                out = self.component.get_panel().export(fmt, **kwargs)
            return BytesIO(out) if isinstance(out, bytes) else StringIO(out)
        out = self.component.get_panel().export(fmt, **kwargs)
        return BytesIO(out) if isinstance(out, bytes) else StringIO(out)

    @pn.cache
    @staticmethod
    def _load_vega_lite_schema(schema_url: str | None = None):
        return requests.get(schema_url, timeout=0.5).json()

    @staticmethod
    def _format_validation_error(error: ValidationError) -> str:
        """Turn a (possibly nested) JSONSchema ValidationError into readable lines."""

        # 1) Collect leaf errors (ignore parents that just aggregate alternatives)
        GROUP_VALIDATORS = {"anyOf", "oneOf", "allOf", "not"}

        def iter_leaves(err: ValidationError):
            # If this error has nested contexts, descend; otherwise it's a leaf
            if err.context:
                for e in err.context:
                    yield from iter_leaves(e)
            else:
                yield err

        leaves = list(iter_leaves(error))
        if not leaves:
            # Fallback: no leaves; return the top-level message
            return f"{error.json_path}: {error.message}"

        # 2) Remove group-validator parents that slipped through (rare on leaves, but safe)
        leaves = [e for e in leaves if (e.validator or "") not in GROUP_VALIDATORS]

        # 3) If there are any non-root errors, drop root-level `$` to reduce noise
        if any(e.json_path != "$" for e in leaves):
            leaves = [e for e in leaves if e.json_path != "$"]

        # 4) Per-path choose the "best" error by (priority, -depth) then stable order
        # Lower priority number wins; deeper path wins for ties
        PRIORITY = {
            "required": 0,
            "additionalProperties": 1,
            "enum": 2,
            "const": 2,
            "type": 3,
            "format": 4,
            "pattern": 4,
        }

        def path_depth(p: str) -> int:
            # Rough depth metric: count of separators
            return p.count(".") + p.count("[") + p.count("]")

        def priority(err: ValidationError) -> tuple[int, int]:
            return (PRIORITY.get(err.validator or "", 5), -path_depth(err.json_path))

        winners: dict[str, ValidationError] = {}
        for e in leaves:
            p = e.json_path
            if p not in winners or priority(e) < priority(winners[p]):
                winners[p] = e

        # 5) Nicely format each error; include a compact instance snippet when useful
        def fmt_value(val) -> str:
            try:
                s = json.dumps(val, ensure_ascii=False)
            except Exception:
                s = repr(val)
            if len(s) > 80:
                s = s[:77] + "..."
            return s

        lines = []
        # Sort by path, then by priority within the same path for deterministic output
        for p, e in sorted(winners.items(), key=lambda kv: (kv[0], priority(kv[1]))):
            # Some messages are already very clear; optionally append value snippet
            suffix = ""
            # Show the instance if it's short and the message isn't already repeating it
            try:
                inst = e.instance
                if inst is not None and e.validator != "required":
                    snippet = fmt_value(inst)
                    if snippet and snippet not in e.message:
                        suffix = f" (value={snippet})"
            except Exception:
                pass
            lines.append(f"{p}: {e.message}{suffix}")

        return "\n".join(lines)

    @classmethod
    def _serialize_component(cls, component: Component, spec_dict: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        component_spec = spec_dict or component.to_spec()
        vega_spec = component_spec['spec']
        return dump_yaml(vega_spec), component_spec

    @classmethod
    def _deserialize_component(
        cls, component: Component, yaml_spec: str, spec_dict: dict[str, Any], pipeline: Pipeline | None = None
    ) -> str:
        spec = load_yaml(yaml_spec)
        cls.validate_spec(spec)
        spec_dict = dict(spec_dict, spec=spec)
        if pipeline is not None:
            spec_dict.pop('pipeline')
        return type(component).from_spec(spec_dict, pipeline=pipeline)

    @classmethod
    def validate_spec(cls, spec):
        if "spec" in spec:
            spec = spec["spec"]
        try:
            vega_lite_schema = cls._load_vega_lite_schema(
                spec.get("$schema", "https://vega.github.io/schema/vega-lite/v5.json")
            )
        except Exception:
            return super().validate_spec(spec)
        vega_lite_validator = Draft7Validator(vega_lite_schema)
        try:
            # the zoomable params work, but aren't officially valid
            # so we need to remove them for validation
            # https://stackoverflow.com/a/78342773/9324652
            spec_copy = deepcopy(spec)
            for key in VEGA_ZOOMABLE_MAP_ITEMS.get("projection", {}):
                spec_copy.get("projection", {}).pop(key, None)
            spec_copy.pop("params", None)
            vega_lite_validator.validate(spec_copy)
        except ValidationError as e:
            raise RuntimeError(cls._format_validation_error(e)) from e
        return super().validate_spec(spec)

    def __str__(self):
        # Only keep the spec part
        return f"{self.__class__.__name__}:\n```yaml\n{self.spec}\n```"


class DeckGLOutput(LumenOutput):
    """Output class for DeckGL 3D map visualizations.

    Handles serialization/deserialization of DeckGL specs and provides
    appropriate export formats including standalone HTML.
    """

    export_formats = ("yaml", "json", "html")

    _controls = [CopyControls, RetryControls]
    _label = "Map"

    # Required keys for a valid DeckGL spec
    _required_keys = frozenset({"layers", "initialViewState"})

    @classmethod
    def _serialize_component(cls, component: Component, spec_dict: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        component_spec = spec_dict or component.to_spec()
        deckgl_spec = component_spec['spec']
        return dump_yaml(deckgl_spec), component_spec

    @classmethod
    def _deserialize_component(
        cls, component: Component, yaml_spec: str, spec_dict: dict[str, Any], pipeline: Pipeline | None = None
    ) -> Component:
        spec = load_yaml(yaml_spec)
        cls.validate_spec(spec)
        spec_dict = dict(spec_dict, spec=spec)
        if pipeline is not None:
            spec_dict.pop('pipeline', None)
        return type(component).from_spec(spec_dict, pipeline=pipeline)

    @classmethod
    def validate_spec(cls, spec: dict[str, Any]) -> dict[str, Any]:
        """Validate that the DeckGL spec has required structure."""
        if "spec" in spec:
            spec = spec["spec"]

        missing = cls._required_keys - set(spec.keys())
        if missing:
            raise ValueError(f"DeckGL spec missing required keys: {missing}")

        # Validate layers structure
        layers = spec.get("layers", [])
        if not isinstance(layers, list):
            raise ValueError("DeckGL 'layers' must be a list")

        for i, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError(f"Layer {i} must be a dictionary")
            if "@@type" not in layer and "type" not in layer:
                raise ValueError(f"Layer {i} missing '@@type' or 'type' field")

        # Validate initialViewState
        view_state = spec.get("initialViewState", {})
        if not isinstance(view_state, dict):
            raise ValueError("'initialViewState' must be a dictionary")

        return super().validate_spec(spec)

    def export(self, fmt: str) -> StringIO | BytesIO:
        ret = super().export(fmt)
        if ret is not None:
            return ret
        if fmt == "json":
            spec = load_yaml(self.spec)
            return StringIO(json.dumps(spec, indent=2))
        elif fmt == "html":
            spec = load_yaml(self.spec)
            html_content = self._generate_html(spec)
            return StringIO(html_content)
        raise ValueError(f"Unknown export format {fmt!r}")

    def _generate_html(self, spec: dict) -> str:
        """Generate standalone HTML for DeckGL visualization."""
        spec_json = json.dumps(spec, indent=2)
        return textwrap.dedent(f"""\
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>DeckGL Visualization</title>
                <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
                <script src="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.js"></script>
                <link href="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.css" rel="stylesheet" />
                <style>
                    body {{ margin: 0; padding: 0; }}
                    #container {{ width: 100vw; height: 100vh; }}
                </style>
            </head>
            <body>
                <div id="container"></div>
                <script>
                    const spec = {spec_json};
                    new deck.DeckGL({{...spec, container: 'container'}});
                </script>
            </body>
            </html>
        """)

    def __str__(self):
        return f"{self.__class__.__name__}:\n```yaml\n{self.spec}\n```"


class AnalysisOutput(LumenOutput):

    analysis = param.ClassSelector(class_=Analysis)

    context = param.Dict()

    pipeline = param.ClassSelector(class_=Pipeline)

    def __init__(self, **params):
        if 'title' not in params or params['title'] is None:
            params['title'] = type(params['analysis']).__name__
        super().__init__(**params)
        self._rendered = True

    def _render_editor(self):
        controls = self.analysis.controls(self.context)
        if controls is None and self.analysis.autorun:
            return super()._render_editor()

        if self.analysis._run_button:
            run_button = self.analysis._run_button
            run_button.param.watch(self._rerun, 'clicks')
        else:
            self.analysis._run_button = run_button = Button(
                icon='play_circle_outline', label='Run', on_click=self._rerun,
                button_type='success', margin=(10, 0, 0, 10)
            )
        return Column(controls, run_button) if controls else run_button

    async def render_context(self):
        out_context = {"analysis": self.analysis}
        view = self.component
        if isinstance(view, View):
            out_context["view"] = dict(object=self._spec_dict, type=view.view_type)
        elif isinstance(view, Pipeline):
            out_context["pipeline"] = out_context["view"] = pipeline = view
            data = await get_data(pipeline)
            if len(data) > 0:
                out_context["data"] = await describe_data(data)
        return out_context

    async def _rerun(self, event):
        if asyncio.iscoroutinefunction(self.analysis.__call__):
            view = await self.analysis(self.pipeline, self.context)
        else:
            view = await asyncio.to_thread(self.analysis, self.pipeline, self.context)
        view = as_panel(view)
        if isinstance(view, Viewable):
            view = Panel(object=view, pipeline=self.pipeline)
        self.component = view
        self._rendered = False
        self.spec, self._spec_dict = self._serialize_component(view)


class SQLOutput(LumenOutput):

    language = "sql"

    export_formats = ("sql", "csv", "xlsx")
    _label = "Table"

    def export(self, fmt: str) -> str | bytes:
        super().export(fmt)
        if fmt == 'sql':
            return StringIO(self.spec)
        sio = StringIO()
        self.component.data.to_csv(sio)
        sio.seek(0)
        return sio

    def render_explorer(self):
        return GraphicWalker(
            self.component.param.data,
            kernel_computation=True,
            tab='data',
            sizing_mode='stretch_both'
        )

    async def render_context(self):
        return {
            "sql": self.spec,
            "pipeline": self.component,
            "table": self.component.table,
            "source": self.component.source,
            "data": await describe_data(self.component.data)
        }

    @classmethod
    def _serialize_component(cls, component: Component, spec_dict: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        component_spec = spec_dict or component.to_spec()
        sql_spec = component_spec["source"]["tables"][component.table]
        return sql_spec, component_spec

    @classmethod
    def _deserialize_component(
        cls, component: Component, sql_spec: str, spec_dict: dict[str, Any], pipeline: Pipeline | None = None
    ) -> str:
        spec_dict = deepcopy(spec_dict)
        spec_dict["source"]["tables"][component.table] = sql_spec
        return type(component).from_spec(spec_dict)

    def __panel__(self):
        return self.view

    def __str__(self):
        return f"{self.__class__.__name__}:\n```sql\n{self.spec}\n```"
