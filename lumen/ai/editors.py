from __future__ import annotations

import asyncio
import json
import pathlib
import re
import traceback

from copy import deepcopy
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Any

import param

from panel.config import config
from panel.layout import Column, Row
from panel.pane import (
    PDF, DeckGL, Markdown, panel as as_panel,
)
from panel.param import ParamMethod
from panel.viewable import Viewable, Viewer
from panel.widgets import CodeEditor
from panel_gwalker import GraphicWalker
from panel_material_ui import (
    Alert, Button, Checkbox, CircularProgress, FileDownload, FlexBox,
    FloatInput, MenuButton, Tabs,
)
from PIL import Image

from ..base import Component
from ..config import dump_yaml, load_yaml
from ..pipeline import Pipeline
from ..transforms.sql import SQLLimit
from ..views.base import Panel, Table, View
from .analysis import Analysis
from .config import FORMAT_ICONS, FORMAT_LABELS
from .controls import AnnotationControls, CopyControls, RetryControls
from .utils import describe_data, get_data

if TYPE_CHECKING:
    from panel.chat.feed import ChatFeed

    from .report import Task


class LumenEditor(Viewer):

    component = param.ClassSelector(class_=Component)

    loading = param.Boolean()

    footer = param.List(default=[])

    spec = param.String(allow_None=True)

    title = param.String(allow_None=True)

    export_formats = ("yaml",)

    language = "yaml"

    _controls = [RetryControls, CopyControls]
    _label = "Result"

    def __init__(self, **params):
        try:
            spec, spec_dict = self._serialize_component(params["component"])
        except Exception:
            spec_dict = None
            spec = None
        if "spec" not in params:
            params["spec"] = spec
        # The _spec_dict contains the full definition of the Lumen Component including
        # the view_type declaration, while the spec is the user editable specification
        # e.g. for a VegaLiteView the spec_dict is {"spec": vega_spec, "type": "vegalite"}
        # while the spec is just vega_spec
        self._spec_dict = spec_dict
        super().__init__(**params)
        self.editor = self._render_editor()
        self.view = ParamMethod(self.render, inplace=True, sizing_mode='stretch_width')
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
            *(control(interface=interface, task=task, view=self) for control in self._controls),
            export_menu,
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
            object_spec = component_spec["object"]
        else:
            object_spec = component_spec
        return dump_yaml(object_spec), component_spec

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
        view = self.component
        if isinstance(view, View):
            # If output is a view we provide the full View specification
            return {"view": self._spec_dict}
        elif isinstance(view, Pipeline):
            data = await get_data(view)
            return {
                "pipeline": view,
                "table": view.table,
                "source": view.source,
                "data": await describe_data(data)
            }
        else:
            return {}

    @param.depends('component')
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

        # Only use cache for serializable specs; non-serializable (spec=None) always re-render
        if self.spec is not None and self.spec in self._last_output:
            yield self._last_output[self.spec]
            return

        try:
            if isinstance(self.component, Pipeline):
                output = await self._render_pipeline(self.component)
            else:
                output = self.component.__panel__()
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


class VegaLiteEditor(LumenEditor):

    export_formats = ("yaml", "png", "jpeg", "pdf", "svg", "html", "webp", "tiff", "eps")

    _pillow_formats = ("webp", "tiff", "eps")

    _controls = [RetryControls, AnnotationControls, CopyControls]
    _label = "Plot"

    # Default export dimensions. These approximate a typical browser viewport
    # and are used when the spec uses container sizing or the actual rendered
    # dimensions cannot be determined from the live panel.
    _export_width = param.Integer(default=1600, precedence=-1, doc="""
        Minimum width for exported images. Ensures the export is wide
        enough to show full chart content including legends.""")
    _export_height = param.Integer(default=800, precedence=-1, doc="""
        Minimum height for exported images. Ensures geographic maps
        have enough vertical space to show the full projection.""")

    def _get_current_bounds(self) -> dict | None:
        """Extract current zoom/pan bounds from the live browser-rendered panel.

        Returns a dict mapping field names to [min, max] intervals,
        e.g. {'temperature': [10, 30], 'date': ['2024-01-01', '2024-06-01']},
        or None if no zoom state is available.
        """
        panel = getattr(self.component, '_panel', None)
        if panel is None or not hasattr(panel, 'selection') or panel.selection is None:
            return None
        bounds = {}
        for param_name in panel.selection.param:
            if param_name == 'name':
                continue
            value = getattr(panel.selection, param_name)
            if isinstance(value, dict):
                bounds.update(value)
        return bounds if bounds else None

    @staticmethod
    def _set_selection_value(spec: dict, bounds: dict) -> bool:
        """Set the value of an interval selection bound to scales.

        This works for standard x/y charts (scatter, bar, line, etc.)
        where an interval selection with bind="scales" enables zoom/pan.
        Vega-Lite uses the selection value to initialize the scale domains
        at render time, reproducing the user's zoomed view.

        Note: Vega-Lite does not support bind="scales" with geographic
        projections, so geographic maps are handled separately via
        dimension preservation in export().

        Returns True if a matching selection was found and updated.
        """
        if not bounds:
            return False
        params = spec.get('params', [])
        if not isinstance(params, list):
            return False
        for i, p in enumerate(params):
            if not isinstance(p, dict):
                continue
            select = p.get('select', {})
            sel_type = select if isinstance(select, str) else select.get('type', '')
            if sel_type == 'interval' and p.get('bind') == 'scales':
                params = list(params)
                params[i] = dict(p, value=dict(bounds))
                spec['params'] = params
                return True
        return False

    @staticmethod
    def _inject_encoding_domains(spec: dict, bounds: dict) -> None:
        """Fallback: inject scale.domain on matching encoding channels.

        Modifies spec in place. Handles x, y, x2, y2 channels.
        Does not handle longitude/latitude (they use projections, not scales).
        """
        def inject_encoding(enc: dict) -> dict:
            enc = dict(enc)
            for channel in ('x', 'y', 'x2', 'y2'):
                if channel not in enc:
                    continue
                ch_spec = enc[channel]
                field = ch_spec.get('field')
                if field and field in bounds:
                    ch_spec = dict(ch_spec)
                    scale = dict(ch_spec.get('scale', {}))
                    scale['domain'] = list(bounds[field])
                    ch_spec['scale'] = scale
                    enc[channel] = ch_spec
            return enc

        if 'encoding' in spec:
            spec['encoding'] = inject_encoding(spec['encoding'])
        if 'layer' in spec:
            spec['layer'] = [
                dict(layer, encoding=inject_encoding(layer['encoding']))
                if 'encoding' in layer else layer
                for layer in spec['layer']
            ]

    @staticmethod
    def _apply_bounds_to_spec(spec: dict, bounds: dict) -> dict:
        """Inject zoom/pan bounds into a Vega-Lite spec.

        Uses two strategies for standard x/y charts:
        1. Primary: Set the value of an interval selection with bind="scales".
        2. Fallback: Inject scale.domain on encoding channels directly.

        For geographic maps with projections, zoom/pan state is preserved
        by matching the export dimensions to the live panel dimensions
        (handled in export(), not here).

        Parameters
        ----------
        spec : dict
            The Vega-Lite specification.
        bounds : dict
            Field-name-to-[min, max] mapping from the live selection.

        Returns
        -------
        dict
            Modified spec with current view bounds applied.
        """
        spec = dict(spec)

        # Strategy 1: Set selection parameter value (universal approach)
        if VegaLiteEditor._set_selection_value(spec, bounds):
            return spec

        # Also check inside layers for selection params
        if 'layer' in spec:
            for i, layer in enumerate(spec['layer']):
                if not isinstance(layer, dict):
                    continue
                layer = dict(layer)
                if VegaLiteEditor._set_selection_value(layer, bounds):
                    layers = list(spec['layer'])
                    layers[i] = layer
                    spec['layer'] = layers
                    return spec

        # Strategy 2: Fallback to scale.domain injection on encodings
        VegaLiteEditor._inject_encoding_domains(spec, bounds)

        # Recurse into concatenated specs
        for key in ('hconcat', 'vconcat', 'concat'):
            if key in spec:
                spec[key] = [
                    VegaLiteEditor._apply_bounds_to_spec(sub, bounds)
                    for sub in spec[key]
                ]
        return spec

    def export(self, fmt: str) -> StringIO | BytesIO:
        ret = super().export(fmt)
        if ret is not None:
            return ret

        # vl-convert doesn't support webp/tiff/eps; render as png first, then convert with Pillow
        render_fmt = "png" if fmt in self._pillow_formats else fmt
        kwargs = {"scale": 2} if render_fmt in ("png", "jpeg", "pdf") else {}

        # Capture zoom/pan bounds from live panel
        # BEFORE param.update replaces the component
        bounds = self._get_current_bounds()

        spec = load_yaml(self.spec)

        # Resolve dimensions for export rendering.
        # The browser renders charts stretched to fill the container
        # (via sizing_mode='stretch_width'/'stretch_both'), so both width
        # and height in the browser are typically larger than what the spec
        # declares. Ensure export dimensions are at least _export_width and
        # _export_height to approximate the browser viewport.
        spec_width = spec.get("width")
        spec_height = spec.get("height")
        if isinstance(spec_width, (int, float)) and spec_width != "container":
            spec["width"] = max(int(spec_width), self._export_width)
        else:
            spec["width"] = self._export_width
        if isinstance(spec_height, (int, float)) and spec_height != "container":
            spec["height"] = max(int(spec_height), self._export_height)
        else:
            spec["height"] = self._export_height

        # Inject current view bounds so export reflects zoomed view
        # (works for standard x/y charts with interval selections)
        if bounds:
            spec = self._apply_bounds_to_spec(spec, bounds)

        export_width = spec["width"]
        export_height = spec["height"]
        with self.param.update(spec=dump_yaml(spec)):
            panel = self.component.get_panel()
            # Set dimensions directly on the pane so Panel's Vega.export()
            # uses them instead of its own defaults
            if hasattr(panel, 'width'):
                panel.width = export_width
            if hasattr(panel, 'height'):
                panel.height = export_height
            out = panel.export(render_fmt, **kwargs)

        if fmt in self._pillow_formats:
            img = Image.open(BytesIO(out))
            if fmt == "eps":
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format=fmt.upper())
            buf.seek(0)
            return buf

        return BytesIO(out) if isinstance(out, bytes) else StringIO(out)

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
            import vl_convert as vlc
            vlc.vegalite_to_vega(spec)
        except ValueError as e:
            msg = str(e)
            if '\n    at Nc.' in msg:
                msg = msg[:msg.index('\n    at Nc.')]
            raise RuntimeError(msg) from e
        return super().validate_spec(spec)

    def __str__(self):
        # Only keep the spec part
        return f"{self.__class__.__name__}:\n```yaml\n{self.spec}\n```"


class DeckGLEditor(LumenEditor):
    """Output class for DeckGL 3D map visualizations.

    Handles serialization/deserialization of DeckGL specs and provides
    appropriate export formats including standalone HTML.
    """

    export_formats = ("yaml", "json", "html")

    _controls = [RetryControls, CopyControls]
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
            return self._export_html()
        raise ValueError(f"Unknown export format {fmt!r}")

    def _export_html(self) -> StringIO:
        """Export DeckGL visualization as standalone HTML using Panel's save.

        This re-embeds the pipeline data into the spec before saving,
        ensuring the HTML file is self-contained.
        """
        spec = deepcopy(load_yaml(self.spec))

        # Re-embed data from pipeline into layers
        if self.component and self.component.pipeline:
            df = self.component.get_data()
            if df is not None and 'layers' in spec:
                data_records = df.to_dict(orient='records')
                for layer in spec['layers']:
                    if 'data' not in layer:
                        layer['data'] = data_records

        html_buffer = StringIO()
        DeckGL(spec, sizing_mode='stretch_both').save(html_buffer)
        html_buffer.seek(0)
        return html_buffer

    def __str__(self):
        return f"{self.__class__.__name__}:\n```yaml\n{self.spec}\n```"


class AnalysisOutput(LumenEditor):

    analysis = param.ClassSelector(class_=Analysis)

    context = param.Dict()

    pipeline = param.ClassSelector(class_=Pipeline)

    def __init__(self, **params):
        if 'title' not in params or params['title'] is None:
            params['title'] = type(params['analysis']).__name__
        super().__init__(**params)

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
        return FlexBox(*controls, run_button) if controls else run_button

    async def render_context(self):
        out_context = await super().render_context()
        out_context["analysis"] = self.analysis
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
        try:
            self.spec, self._spec_dict = self._serialize_component(view)
        except Exception:
            # Handle non-serializable components (e.g., Matplotlib figures)
            self.spec = None
            self._spec_dict = None


class SQLEditor(LumenEditor):

    language = "sql"

    export_formats = ("sql", "csv", "xlsx", "json", "markdown")
    _label = "Table"

    def export(self, fmt: str) -> StringIO | BytesIO:
        super().export(fmt)
        data = self.component.data
        if fmt == 'sql':
            return StringIO(self.spec)
        sio = StringIO()
        if fmt in ('csv', 'xlsx'):
            data.to_csv(sio)
        elif fmt == 'json':
            data.to_json(sio, orient='records', indent=2)
        elif fmt == 'markdown':
            sio.write(data.to_markdown(index=False))
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


class DocumentEditor(LumenEditor):
    """
    Displays relevant documents in tabs with type-aware rendering.
    """

    documents = param.List(default=[], doc="""
        A list of document dictionaries with at least a `filename` key.
        Optional keys include `content`, `raw_bytes`, and `similarity`.""")

    min_similarity = param.Magnitude(default=0.3, doc="Minimum similarity threshold")

    query = param.String(default="", doc="""
        The user query these documents are relevant to.""")

    title = param.String(default="Relevant documents", doc="""
        The editor title shown above document tabs.""")

    _max_preview_height = None

    def __init__(self, **params):
        if "documents" in params and "component" not in params:
            params["component"] = self._render_component(
                params["documents"],
            )
        super().__init__(**params)

    def _render_editor(self):
        return None

    def render_controls(self, task, interface):
        controls = super().render_controls(task, interface)
        controls.append(FloatInput.from_param(self.param.min_similarity))
        return controls

    def _render_document(self, doc: dict[str, Any]):
        filename = doc.get("filename", "Unknown document")
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        raw_bytes = doc.get("raw_bytes")
        content = doc.get("content", "") or "*No preview available for this document.*"

        # Render PDFs natively when bytes are available for a high-fidelity preview.
        if extension == "pdf":
            if raw_bytes:
                pdf_data = raw_bytes
            elif pathlib.Path(filename).is_file():
                pdf_data = filename
            return PDF(
                pdf_data,
                height=self._max_preview_height,
                sizing_mode="stretch_both",
            )

        return Column(
            Markdown(content, sizing_mode="stretch_width"),
            max_height=self._max_preview_height,
            scroll="y-auto",
            sizing_mode="stretch_both",
        )

    @param.depends("min_similarity", watch=True)
    def _update_documents(self):
        self.component = self._render_component()

    def _render_component(
        self,
        documents: list[dict[str, Any]] | None = None
    ) -> Panel:
        documents = [
            doc for doc in (documents or self.documents)
            if doc.get("similarity", 0) > self.min_similarity
        ]
        if not documents:
            return Panel(
                object=Markdown("*No relevant documents found.*")
            )

        tabs = []
        for doc in documents:
            filename = doc.get("filename", "Unknown document")
            tabs.append((filename, self._render_document(doc)))

        return Panel(object=Tabs(*tabs, dynamic=True, sizing_mode="stretch_both"))

    async def render_context(self):
        return {"view": self._spec_dict}
