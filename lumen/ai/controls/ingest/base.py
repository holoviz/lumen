from __future__ import annotations

import asyncio

import param

from panel.pane.markup import HTML
from panel.viewable import Viewer
from panel_material_ui import Button, Column as MuiColumn, Tabs

from ....sources.duckdb import DuckDBSource
from .progress import Progress
from .result import SourceResult

# ─────────────────────────────────────────────────────────────────────────────
# BASE SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

class BaseSourceControls(Viewer):
    """
    Base class for source controls with lifecycle management.

    Provides progress reporting, message display, load-button wiring,
    and source-output registration. Subclasses implement ``_load()``
    or call ``_run_load()`` manually for custom trigger patterns.
    """

    context = param.Dict(default={})

    disabled = param.Boolean(default=False, doc="Disable controls")

    load_mode = param.Selector(
        default="button",
        objects=["button", "manual"],
        constant=True,
        doc="""
        How data loading is triggered:
        - "button": Show load button, clicking triggers _load()
        - "manual": No button, use _run_load(coro) for custom triggers
        """,
    )

    outputs = param.Dict(default={})

    source_catalog = param.Parameter(default=None, doc="Reference to SourceCatalog instance")

    # Events
    cancel = param.Event(doc="Cancel")

    load = param.Event(doc="Trigger data loading")

    upload_successful = param.Event(doc="Triggered when files are successfully uploaded and processed")

    # Internal state
    _last_table = param.String(default="", doc="Last table added")

    _count = param.Integer(default=0, doc="Count of sources added")

    # UI customization
    label = param.String(default="", constant=True, doc="""
        HTML label shown in the sidebar for this control.""")

    load_button_label = param.String(default="Load Data", doc="""
        Text label for the load button.""")

    load_button_icon = param.String(default="download", doc="""
        Material icon name for the load button.""")

    source_name_prefix = param.String(default="UploadedSource", doc="""
        Prefix for auto-generated source names.""")

    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        self._init_ui_components()
        self._layout = self._render_layout()

    def _init_ui_components(self):
        """Initialize shared UI components. Called before _render_layout()."""
        self._error_placeholder = HTML("", visible=False, margin=(0, 10, 5, 10))
        self._message_placeholder = HTML("", visible=False, margin=(0, 10, 10, 10))

        self._error_text = self._error_placeholder
        self._message_text = self._message_placeholder

        self.progress = self._render_progress()

        self._load_button = Button.from_param(
            self.param.load,
            label=self.param.load_button_label,
            icon=self.param.load_button_icon,
            sizing_mode="stretch_width",
            description="",
            height=42,
        )

        self.tables_tabs = Tabs(sizing_mode="stretch_width")

    async def _load(self) -> SourceResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _load() or set "
            "load_mode='manual' and use _run_load() for custom triggers."
        )

    def _render_controls(self) -> list:
        return []

    def _render_progress(self) -> Progress:
        return Progress()

    def _render_layout(self) -> Viewer:
        controls = self._render_controls()
        components = list(controls)
        if self.load_mode == "button":
            components.append(self._load_button)

        components.extend([
            self._error_placeholder,
            self._message_placeholder,
            self.progress.bar,
            self.progress.description,
        ])

        return MuiColumn(
            *components,
            sizing_mode="stretch_width",
            margin=(10, 15),
        )

    @param.depends("load", watch=True)
    async def _on_load_event(self):
        await self._run_load(self._load())

    async def _run_load(self, coro):
        self._clear_messages()
        self._layout.loading = True
        self._load_button.disabled = True

        try:
            await asyncio.sleep(0.01)
            result = await coro
            self._handle_success(result)
            return result
        except Exception as e:
            self._handle_error(e)
            raise
        finally:
            self._layout.loading = False
            self._load_button.disabled = False
            self.progress.clear()

    def _clear_messages(self):
        self._error_placeholder.visible = False
        self._message_placeholder.visible = False
        self._error_placeholder.object = ""
        self._message_placeholder.object = ""

    def _handle_success(self, result: SourceResult):
        if not result.sources:
            self._show_message(result.message or "No data loaded", error=True)
            return

        for source in result.sources:
            self._register_source_output(source)

        if result.table:
            self.outputs["table"] = result.table
            self._last_table = result.table

        self.param.trigger("outputs")
        self.param.trigger("upload_successful")
        self._show_message(result.message or "✓ Data loaded successfully")

    def _handle_error(self, error: Exception):
        self._show_message(f"⚠️ {error}", error=True)

    def _show_message(self, message: str, error: bool = False):
        target = self._error_placeholder if error else self._message_placeholder
        target.object = message
        target.visible = True

    def _register_source_output(self, source: DuckDBSource):
        """Record the source while keeping ``outputs["sources"]`` deduplicated."""
        self.outputs["source"] = source
        sources = self.outputs.setdefault("sources", [])
        if not any(existing is source for existing in sources):
            sources.append(source)

    def __panel__(self):
        return self._layout
