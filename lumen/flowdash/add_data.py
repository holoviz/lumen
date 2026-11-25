"""
Add data: the dialog that adds a data file people upload, or a source an
admin connects, to a Lumen FlowDash app, and the dashboard bar's button
opening it (see ``lumen.flowdash.uploads`` for what it adds and who may).

The dialog previews what it would add before adding it: the source's
name, its tables and, for the first tables, each column's type and first
values, read off the event loop. Someone the uploads settings let upload
gets the upload alone; an admin can also connect a database, a database
file or a data file by its path, URL or connection string, as
``lumen serve`` takes them. Either may keep the new tables from the
assistant.
"""
from __future__ import annotations

import asyncio
import typing as t

from collections.abc import Callable
from pathlib import Path

import panel as pn
import panel_material_ui as pmui

from panel.viewable import Viewer
from panel_flowdash.admin.kit import esc

from .uploads import (
    Connection, Preview, Upload, UploadError, Uploads, _is_admin, _slug,
    excel_sheets, extension_of,
)

ALL_SHEETS = "All sheets, a table each"

_PREVIEW_CSS = """
.k { font-size: 0.875rem; color: var(--mui-palette-text-primary); }
.k .overline { color: var(--mui-palette-text-secondary); font-size: 0.75rem; font-weight: 500;
  letter-spacing: 1.2px; text-transform: uppercase; margin: 4px 0 8px 0; }
.k .scroll { overflow-x: auto; border: 1px solid var(--mui-palette-divider); border-radius: 4px; }
.k table { border-collapse: collapse; width: 100%; font-size: 0.8125rem; }
.k th { text-align: left; font-weight: 500; color: var(--mui-palette-text-secondary); white-space: nowrap; }
.k th, .k td { padding: 6px 10px; border-bottom: 1px solid var(--mui-palette-divider); vertical-align: top; }
.k tr:last-child td { border-bottom: none; }
.k td.v { max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.k code, .k .col { font-family: "Roboto Mono", ui-monospace, monospace; font-size: 0.8125rem; }
.k .muted { color: var(--mui-palette-text-secondary); }
.k .error-text { color: var(--mui-palette-error-main, #d32f2f); }
"""


def file_size(size: float) -> str:
    """``size`` bytes as people read it, e.g. "14.2 KB"."""
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "bytes" else f"{size:,.1f} {unit}"
        size /= 1024
    return str(size)


def _names(names: t.Sequence[str]) -> str:
    quoted = [f"<code>{esc(n)}</code>" for n in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def preview_html(preview: Preview | None) -> str:
    """What adding would add: the tables and, for the first, each column's type and first values."""
    if preview is None:
        return ""
    noun = "table" if len(preview.tables) == 1 else "tables"
    parts = [f'<div class="muted">Adds the {noun} {_names(preview.tables)}.</div>']
    first = preview.previews[0] if preview.previews else None
    if first is not None:
        rows = "".join(
            f'<tr><td class="col">{esc(name)}</td><td>{esc(kind)}</td><td class="v" title="{esc(values)}">'
            f"{esc(values) or '<span class=muted>-</span>'}</td></tr>"
            for name, kind, values in first.columns
        )
        parts.append(
            f'<div class="overline">Preview of {esc(first.table)}</div>'
            '<div class="scroll"><table><thead><tr><th>Column</th><th>Type</th><th>First rows</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )
    return f'<div class="k">{"".join(parts)}</div>'


class AddDataDialog(Viewer):
    """
    The Add data dialog: a file, the table name it gets and, for Excel,
    its sheet, or for an admin a source to connect; a preview; whether the
    assistant may use it. ``on_added`` is called with the ``Upload`` or
    ``Connection`` added.
    """

    def __init__(
        self, uploads: Uploads, app: t.Any, on_added: Callable[[Upload | Connection], t.Any] | None = None,
        **params,
    ):
        super().__init__(**params)
        self.uploads = uploads
        self._app = app
        self._on_added = on_added
        self._content: bytes | None = None
        self._filename = ""
        self._preview: Preview | None = None
        self._source: t.Any = None
        # Bumped per preview, so the result of one superseded while it was read is dropped.
        self._run = 0
        self._limits = pmui.Typography("", variant="body2", margin=(0, 0, 4, 0), sx={"color": "text.secondary"})
        self._mode = pmui.RadioButtonGroup(
            options={"Upload a file": "upload", "Connect a source": "connect"}, value="upload",
            sizing_mode="stretch_width", margin=0, css_classes=["flowdash-add-data-mode"],
            sx={"& .MuiButton-root, & .MuiToggleButton-root": {"textTransform": "none"}},
        )
        self._mode.param.watch(lambda _event: self._show_mode(), "value")
        # Upload a file
        self._file = pmui.FileInput(
            label="Choose a file", icon="upload_file", variant="outlined", margin=0,
            css_classes=["flowdash-upload-file"],
        )
        self._file.param.watch(self._on_file, "value")
        self._chosen = pmui.Typography("", variant="body2", margin=0, sx={"color": "text.secondary"})
        self._name = pmui.TextInput(
            label="Table name", placeholder="e.g. sales", sizing_mode="stretch_width", margin=0,
            css_classes=["flowdash-upload-name"],
        )
        self._name.param.watch(lambda _event: self._refresh_preview(), "value")
        self._sheet = pmui.Select(
            label="Sheet", options={ALL_SHEETS: ""}, value="", visible=False,
            sizing_mode="stretch_width", margin=0, css_classes=["flowdash-upload-sheet"],
        )
        self._sheet.param.watch(lambda _event: self._refresh_preview(), "value")
        self._upload_pane = pn.Column(
            pn.Row(self._file, self._chosen, margin=0, styles={"align-items": "center", "gap": "12px"}),
            pn.Row(self._name, self._sheet, margin=0, sizing_mode="stretch_width", styles={"gap": "12px"}),
            margin=0, sizing_mode="stretch_width", styles={"gap": "16px"},
        )
        # Connect a source
        self._location = pmui.TextInput(
            label="File, URL or connection string",
            placeholder="e.g. windfarm.duckdb, https://example.com/data.parquet or postgresql://user@host/db",
            sizing_mode="stretch_width", margin=0, css_classes=["flowdash-connect-location"],
        )
        self._ref = pmui.TextInput(
            label="Source name", placeholder="e.g. windfarm", width=220, margin=0,
            css_classes=["flowdash-connect-name"],
        )
        self._read = pmui.Button(
            label="Read", icon="sync", variant="outlined", margin=0, css_classes=["flowdash-connect-read"],
        )
        self._read.on_click(lambda _event: self._refresh_preview())
        self._connect_pane = pn.Column(
            pn.Row(self._location, self._ref, margin=0, sizing_mode="stretch_width", styles={"gap": "12px"}),
            pn.Row(
                pmui.Typography(
                    "Read as lumen serve reads its data. A connection string is kept with its credentials in the "
                    "app's data folder.", variant="caption", margin=0, sx={"color": "text.secondary"},
                    styles={"flex": "1 1 auto"},
                ),
                self._read, margin=0, sizing_mode="stretch_width", styles={"gap": "12px", "align-items": "center"},
            ),
            margin=0, sizing_mode="stretch_width", styles={"gap": "12px"}, visible=False,
        )
        self._previewing = pmui.LinearProgress(variant="indeterminate", visible=False, margin=0, sizing_mode="stretch_width")
        self._preview_pane = pn.pane.HTML("", margin=0, sizing_mode="stretch_width", stylesheets=[_PREVIEW_CSS],
                                          css_classes=["flowdash-add-data-preview"])
        self._alert = pmui.Alert(object="", severity="error", visible=False, sizing_mode="stretch_width", margin=0)
        self._assistant = pmui.Switch(value=True, margin=0, css_classes=["flowdash-add-data-assistant"])
        assistant = pn.Row(
            pn.Column(
                pmui.Typography("The assistant can use it", variant="body1", margin=0),
                pmui.Typography("Off keeps the assistant from listing, describing or querying it.", variant="caption",
                                margin=0, sx={"color": "text.secondary"}),
                margin=0, sizing_mode="stretch_width",
            ),
            self._assistant,
            margin=0, sizing_mode="stretch_width", styles={"align-items": "center", "gap": "12px"},
        )
        self._note = pmui.Typography("", variant="caption", margin=0, sx={"color": "text.secondary"},
                                     styles={"flex": "1 1 auto", "min-width": "0"})
        self._cancel = pmui.Button(label="Cancel", variant="outlined", color="primary", margin=0)
        self._cancel.on_click(lambda _event: self.close())
        self._submit = pmui.Button(
            label="Add", icon="add", variant="contained", margin=0, disabled=True,
            css_classes=["flowdash-upload-submit"],
        )
        self._submit.on_click(self._on_submit)
        self._dialog = pmui.Dialog(
            pn.Column(
                self._limits,
                self._mode,
                self._upload_pane,
                self._connect_pane,
                self._previewing,
                self._preview_pane,
                self._alert,
                assistant,
                pn.Row(self._note, self._cancel, self._submit, margin=0, sizing_mode="stretch_width",
                       styles={"gap": "8px", "align-items": "center"}),
                margin=0, sizing_mode="stretch_width", styles={"gap": "14px", "min-width": "min(560px, 80vw)"},
                css_classes=["flowdash-upload-dialog"],
            ),
            title="Add data", open=False, show_close_button=True, close_on_click=False, width_option="md",
            # Its content stretches, which would otherwise stretch the dashboard bar item holding it.
            sizing_mode="fixed", width=0, height=0, margin=0,
        )

    # ------------------------------------------------------------------
    # State

    @property
    def connecting(self) -> bool:
        return self._mode.value == "connect"

    def _sync_limits(self):
        allowed = self.uploads.allowed()
        self._file.param.update(
            accept=",".join(f".{ext}" for ext in allowed), max_file_size=self.uploads.max_bytes(),
        )
        labels = [f.label for f in allowed.values()]
        types = f"{', '.join(labels[:-1])} or {labels[-1]}" if len(labels) > 1 else (labels[0] if labels else "No file types")
        self._limits.object = (
            "A database, a database file or a data file, as lumen serve takes them." if self.connecting
            else f"{types}, up to {self.uploads.settings().max_mb} MB."
        )

    def _show_mode(self):
        with pn.io.hold():
            self._upload_pane.visible = not self.connecting
            self._connect_pane.visible = self.connecting
            self._preview, self._source = None, None
            self._preview_pane.object = ""
            self._alert.visible = False
            self._sync_limits()
            self._sync_submit()
        if not self.connecting and self._content is not None:
            self._refresh_preview()

    def _sync_submit(self):
        preview = self._preview
        self._submit.disabled = preview is None
        if preview is None:
            self._note.object = "Everyone who can open the app can read what you add in dashboards."
        else:
            self._note.object = (
                f"Added as the source {preview.ref}. Everyone who can open the app can read it in dashboards."
            )

    def open(self, mode: str = "upload"):
        """Open the dialog with nothing chosen, on uploading a file or, for an admin, ``mode="connect"``."""
        admin = _is_admin(self._app)
        with pn.io.hold():
            self._content, self._filename, self._preview, self._source = None, "", None, None
            self._chosen.object = "No file chosen"
            self._name.value = ""
            self._sheet.param.update(options={ALL_SHEETS: ""}, value="", visible=False)
            self._location.value = ""
            self._ref.value = ""
            self._preview_pane.object = ""
            self._alert.visible = False
            self._assistant.value = True
            self._submit.loading = False
            self._mode.visible = admin
            self._mode.value = mode if admin and mode == "connect" else "upload"
            self._show_mode()
            self._dialog.open = True

    def close(self):
        self._dialog.open = False
        self._content = None
        self._source = None

    def _fail(self, message: str):
        self._alert.param.update(object=message, visible=True)

    # ------------------------------------------------------------------
    # Choosing and previewing

    def choose(self, content: bytes, filename: str):
        """Take ``content`` as the file ``filename`` to upload, filling in its table name and, for Excel, its sheets."""
        self._content, self._filename = content, Path(filename).name
        sheets = excel_sheets(content) if extension_of(self._filename) == "xlsx" else []
        with pn.io.hold():
            more = f", {len(sheets)} sheets" if len(sheets) > 1 else ""
            self._chosen.object = f"{self._filename}, {file_size(len(content))}{more}"
            self._file.label = f"Uploaded {self._filename}"
            self._sheet.param.update(
                options={ALL_SHEETS: "", **{s: s for s in sheets}}, value="", visible=len(sheets) > 1,
            )
            name = _slug(Path(self._filename).stem)
            if self._name.value == name:
                self._refresh_preview()
            else:
                self._name.value = name

    def _on_file(self, event):
        if event.new is None:
            return
        self.choose(event.new, self._file.filename or "upload")
        self._file.clear()

    def _refresh_preview(self):
        """Read what adding would add, off the event loop, and show it."""
        from .nodes import in_background

        self._run += 1
        run = self._run
        self._preview, self._source = None, None
        self._alert.visible = False
        self._sync_submit()
        if self.connecting:
            location = (self._location.value_input or self._location.value or "").strip()
            if not location:
                self._preview_pane.object = ""
                return
            name = (self._ref.value_input or self._ref.value or "").strip() or None
            work = lambda: self.uploads.read_location(location, name, app=self._app)
        else:
            if self._content is None:
                return
            refusal = self.uploads.refusal(self._filename, len(self._content), self._app)
            if refusal is not None:
                self._preview_pane.object = ""
                self._fail(refusal)
                return
            content, filename = self._content, self._filename
            name, sheet = self._name.value or None, self._sheet.value or None
            work = lambda: (self.uploads.preview(content, filename, name, app=self._app, sheet=sheet), None)
        result: list[t.Any] = []

        def read():
            try:
                result.append(work())
            except UploadError as exc:
                result.append(exc)

        def done():
            if run != self._run or not result:
                return
            with pn.io.hold():
                self._previewing.visible = False
                if isinstance(result[0], UploadError):
                    self._preview_pane.object = ""
                    self._fail(str(result[0]))
                else:
                    self._preview, self._source = result[0]
                    self._preview_pane.object = preview_html(self._preview)
                self._sync_submit()

        self._previewing.visible = True
        in_background(read, done)

    # ------------------------------------------------------------------
    # Adding

    def _add(self) -> Upload | Connection:
        assistant = bool(self._assistant.value)
        if self.connecting:
            location = (self._location.value_input or self._location.value or "").strip()
            name = (self._ref.value_input or self._ref.value or "").strip() or None
            return self.uploads.connect(location, name, app=self._app, assistant=assistant, source=self._source)
        return self.uploads.add(
            self._content, self._filename, self._name.value or None, app=self._app,
            sheet=self._sheet.value or None, assistant=assistant,
        )

    def submit(self) -> Upload | Connection | None:
        """Add the chosen file or source; show why if it is refused. Reads it, so prefer ``_on_submit`` on a server."""
        if self._preview is None:
            return None
        try:
            added = self._add()
        except UploadError as exc:
            self._fail(str(exc))
            return None
        self._added(added)
        return added

    async def _on_submit(self, _event=None):
        if self._preview is None:
            return
        self._submit.loading = True
        self._alert.visible = False
        try:
            added = await asyncio.to_thread(self._add)
        except UploadError as exc:
            self._fail(str(exc))
            return
        finally:
            self._submit.loading = False
        self._added(added)

    def _added(self, added: Upload | Connection):
        self.close()
        if isinstance(added, Upload):
            tables = ", ".join(added.tables)
            noun = "table" if len(added.tables) == 1 else "tables"
            message = f"Added the {noun} {tables} as source {added.ref}; find it in the component list."
        else:
            message = f"Connected the source {added.ref}; find its tables in the component list."
        if self._app is not None and hasattr(self._app, "_notify"):
            self._app._notify("success", message, duration=6000)
        if self._on_added is not None:
            self._on_added(added)

    def __panel__(self):
        return self._dialog


# The dialog of #2427 that uploads a file, by the name it had.
UploadDialog = AddDataDialog

# While editing, the dashboard bar's breadcrumb needs the room of the button's label below this width.
_EDIT_LABEL_WIDTH = 1600


def _icon_only(editing: bool) -> list[str]:
    """
    The stylesheet that shows only the button's icon, with its name as a
    tooltip: on narrow screens, as the dashboard bar's Share does, and
    below ``_EDIT_LABEL_WIDTH`` while editing.
    """
    try:
        from panel_flowdash.ui.dashboard_bar import (
            ICON_ONLY_WHEN_COMPACT, icon_only_below,
        )
    except ImportError:
        return []
    return [icon_only_below(_EDIT_LABEL_WIDTH) if editing else ICON_ONLY_WHEN_COMPACT]


class UploadButton(Viewer):
    """The dashboard bar's Add data button and its dialog, for someone who may upload."""

    def __init__(self, uploads: Uploads, app: t.Any, **params):
        super().__init__(**params)
        self.uploads = uploads
        self._app = app
        self.dialog = AddDataDialog(uploads, app)
        self.button = pmui.Button(
            label="Add data", icon="upload_file", variant="text", size="small", margin=(5, 2),
            description="Add data: upload a file as a table for everyone", css_classes=["flowdash-add-data"],
        )
        self.button.on_click(lambda _event: self.open())
        self._layout = pn.Row(self.button, self.dialog, margin=0, visible=False, sizing_mode="fixed")
        editor = getattr(app, "_editor", None)
        if editor is not None:
            editor.param.watch(lambda _event: self.sync(), "editable")
        self.sync()

    def sync(self):
        """Show the button if uploads are offered to this session."""
        editor = getattr(self._app, "_editor", None)
        with pn.io.hold():
            self.button.stylesheets = _icon_only(bool(editor is not None and editor.editable))
            self._layout.visible = editor is not None and self.uploads.offered(self._app)

    def open(self):
        if not self.uploads.offered(self._app):
            self.sync()
            return
        self.dialog.open()

    def __panel__(self):
        return self._layout
