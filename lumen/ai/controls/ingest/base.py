from __future__ import annotations

import asyncio
import io
import json
import pathlib
import zipfile

import pandas as pd
import param

from panel.pane.markup import HTML
from panel.viewable import Viewer
from panel_material_ui import Button, Column as MuiColumn, Tabs

from ....sources.duckdb import DuckDBSource
from ....util import detect_file_encoding
from ...utils import log_debug
from .constants import TABLE_EXTENSIONS
from .file_row import UploadedFileRow
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


# ─────────────────────────────────────────────────────────────────────────────
# FILE SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

class FileSourceControls(BaseSourceControls):
    """
    Intermediate base class for controls that process files (upload/download).

    Adds file-card machinery on top of ``BaseSourceControls`` lifecycle
    management. ``UploadSourceControls`` and ``DownloadSourceControls``
    inherit from this class.
    """

    multiple = param.Boolean(default=True, doc="Allow multiple files")

    clear_uploads = param.Boolean(default=True, doc="Clear uploaded file tabs")

    replace_controls = param.Boolean(default=False, doc="Replace controls on add")

    filedropper_kwargs = param.Dict(default={}, doc="""Keyword arguments to pass to FileDropper.
        Common options include 'accepted_filetypes' and 'max_file_size'.
        See https://panel.holoviz.org/reference/widgets/FileDropper.html for all options.""")

    upload_handlers = param.Dict(default={}, doc="Handlers for custom file extensions")

    # Events
    add = param.Event(doc="Use uploaded file(s)")

    # UI customization
    add_button_icon = param.String(default="add", doc="""
        Material icon name for the add/upload confirmation button.""")

    add_button_label = param.String(default="Confirm file(s)", doc="""
        Text label for the add/upload confirmation button.""")

    __abstract = True

    def __init__(self, **params):
        self._markitdown = None
        self._file_cards = []
        super().__init__(**params)

    def _init_ui_components(self):
        """Initialize file-specific UI components on top of base components."""
        super()._init_ui_components()

        self._upload_cards = MuiColumn(
            sizing_mode="stretch_width",
            margin=0,
            styles={"border-top": "1px solid #e0e0e0", "padding-top": "5px"},
        )
        self._upload_cards.visible = False

        files_to_process = self._upload_cards.param["objects"].rx.len() > 0
        self._add_button = Button.from_param(
            self.param.add,
            name=self.param.add_button_label,
            icon=self.param.add_button_icon,
            visible=files_to_process,
            description="",
            align="center",
            sizing_mode="stretch_width",
            height=42,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # File card management
    # ──────────────────────────────────────────────────────────────────────────

    def _create_file_object(self, file_data: bytes | io.BytesIO | io.StringIO, suffix: str):
        if isinstance(file_data, (io.BytesIO, io.StringIO)):
            return file_data
        if suffix == "csv" and isinstance(file_data, bytes):
            encoding = detect_file_encoding(file_data)
            file_data = file_data.decode(encoding).encode("utf-8")
        return io.BytesIO(file_data) if isinstance(file_data, bytes) else io.StringIO(file_data)

    def _generate_file_cards(self, files: dict):
        self._upload_cards.clear()
        self._file_cards.clear()

        if not files:
            self._add_button.visible = False
            self._upload_cards.visible = False
            return

        for filename, file_data in files.items():
            suffix = pathlib.Path(filename).suffix.lstrip(".").lower()
            file_obj = self._create_file_object(file_data, suffix)
            card = UploadedFileRow(file_obj=file_obj, filename=filename)
            card.param.watch(lambda e, c=card: self._remove_card(c), "delete")
            self._upload_cards.append(card)
            self._file_cards.append(card)

        self._upload_cards.visible = bool(self._file_cards)
        self._add_button.visible = bool(self._file_cards)

    def _remove_card(self, card: UploadedFileRow):
        if card in self._file_cards:
            self._file_cards.remove(card)
        if card in self._upload_cards.objects:
            self._upload_cards.remove(card)
        self._add_button.visible = bool(self._file_cards)
        self._upload_cards.visible = bool(self._file_cards)

    def _clear_uploads(self):
        self._upload_cards.clear()
        self._file_cards.clear()
        self._add_button.visible = False
        self._upload_cards.visible = False

    # ──────────────────────────────────────────────────────────────────────────
    # File reading
    # ──────────────────────────────────────────────────────────────────────────

    def _add_table(
        self,
        duckdb_source: DuckDBSource,
        file: io.BytesIO | io.StringIO,
        card: UploadedFileRow,
    ) -> int:
        conn = duckdb_source._connection
        extension = card.extension
        table = card.alias
        sql_expr = f"SELECT * FROM {table}"
        params = {}
        conversion = None
        filename = f"{card.filename}.{extension}"

        try:
            file.seek(0)
            if extension.endswith("csv"):
                df = pd.read_csv(file, parse_dates=True, sep=None, engine="python")
            elif extension.endswith(("parq", "parquet")):
                df = pd.read_parquet(file)
            elif extension.endswith("json"):
                df = self._read_json_file(file, filename)
            elif extension.endswith("xlsx"):
                df = pd.read_excel(file, sheet_name=card.sheet)
            elif extension.endswith(("geojson", "wkt", "zip")):
                df, conversion, params = self._read_geo_file(file, extension, table, conn)
            else:
                self._error_placeholder.object += f"\n⚠️ Could not convert {filename!r}: unsupported format."
                self._error_placeholder.visible = True
                return 0
        except Exception as e:
            self._error_placeholder.object += f"\n⚠️ Error processing {filename!r}: {e}"
            self._error_placeholder.visible = True
            return 0

        if df is None or df.empty:
            self._error_placeholder.object += f"\n⚠️ {filename!r} contains no data."
            self._error_placeholder.visible = True
            return 0

        # Convert pandas StringDtype columns to object for DuckDB compatibility
        for col in df.columns:
            if isinstance(df[col].dtype, pd.StringDtype):
                df[col] = df[col].astype(object)

        duckdb_source.param.update(params)
        df_rel = conn.from_df(df)
        if conversion:
            conn.register(f"{table}_temp", df_rel)
            conn.execute(conversion)
            conn.unregister(f"{table}_temp")
        else:
            df_rel.to_view(table)
        duckdb_source.tables[table] = sql_expr
        self._register_source_output(duckdb_source)
        self.outputs["table"] = table
        self.param.trigger("outputs")
        self._last_table = table
        return 1

    def _read_json_file(self, file: io.BytesIO | io.StringIO, filename: str) -> pd.DataFrame:
        file.seek(0)
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        content = content.strip()
        if not content:
            raise ValueError("JSON file is empty")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON") from e

        if isinstance(data, list):
            if not data:
                raise ValueError("JSON array is empty")
            if all(isinstance(item, dict) for item in data):
                return pd.json_normalize(data)
            return pd.DataFrame({"value": data})

        if isinstance(data, dict):
            for key in ("data", "records", "rows", "items", "results"):
                if key in data and isinstance(data[key], list) and data[key] and all(isinstance(i, dict) for i in data[key]):
                    return pd.json_normalize(data[key])
            if all(isinstance(v, list) for v in data.values()):
                lengths = [len(v) for v in data.values()]
                if len(set(lengths)) == 1:
                    return pd.DataFrame(data)
                raise ValueError(f"JSON object has arrays of different lengths: {lengths}")
            if all(not isinstance(v, (list, dict)) or v is None for v in data.values()):
                return pd.DataFrame([data])
            file.seek(0)
            try:
                return pd.read_json(file)
            except ValueError as e:
                raise ValueError("JSON structure is not tabular.") from e

        raise ValueError(f"JSON root must be an object or array, got {type(data).__name__}")

    def _read_geo_file(self, file: io.BytesIO, extension: str, table: str, conn) -> tuple[pd.DataFrame, str | None, dict]:
        if extension.endswith("zip"):
            zf = zipfile.ZipFile(file)
            if not any(f.filename.endswith("shp") for f in zf.filelist):
                raise ValueError("ZIP file does not contain a shapefile (.shp)")
            file.seek(0)

        import geopandas as gpd
        geo_df = gpd.read_file(file)

        if geo_df.empty:
            raise ValueError("Geospatial file contains no features")

        df = pd.DataFrame(geo_df)
        df["geometry"] = geo_df["geometry"].to_wkb()

        params = {"initializers": ["INSTALL spatial;\nLOAD spatial;"]}
        conn.execute(params["initializers"][0])

        cols = ", ".join(f'"{c}"' for c in df.columns if c != "geometry")
        conversion = f"CREATE TEMP TABLE {table} AS SELECT {cols}, ST_GeomFromWKB(geometry) as geometry FROM {table}_temp"

        return df, conversion, params

    # ──────────────────────────────────────────────────────────────────────────
    # Metadata handling
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_metadata_content(self, file_obj: io.BytesIO, extension: str) -> str:
        file_obj.seek(0)
        if extension in ("md", "txt", "yaml", "yml"):
            content = file_obj.read()
            return content.decode("utf-8") if isinstance(content, bytes) else content
        if extension == "json":
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            try:
                return json.dumps(json.loads(content), indent=2)
            except json.JSONDecodeError:
                return content
        from markitdown import MarkItDown
        if self._markitdown is None:
            self._markitdown = MarkItDown()
        return self._markitdown.convert_stream(file_obj, file_extension=extension).text_content

    def _add_metadata_file(self, card: UploadedFileRow) -> int:
        try:
            content = self._extract_metadata_content(card.file_obj, card.extension)
            base_filename = f"{card.filename}.{card.extension}"
            filename = base_filename
            if self.source_catalog:
                existing = [m["filename"] for m in self.source_catalog._available_metadata]
                if filename in existing:
                    counter = 1
                    while filename in existing:
                        filename = f"{card.filename}_{counter}.{card.extension}"
                        counter += 1
                    self._message_placeholder.param.update(
                        object=f"Renamed duplicate: {base_filename} → {filename}",
                        visible=True,
                    )
                metadata_entry = {
                    "filename": filename,
                    "display_name": filename.rsplit(".", 1)[0],
                    "content": content,
                }
                if card.extension == "pdf":
                    card.file_obj.seek(0)
                    metadata_entry["raw_bytes"] = card.file_obj.read()
                self.source_catalog._available_metadata.append(metadata_entry)
                asyncio.create_task(self._sync_metadata(filename))  # noqa: RUF006

            self.param.trigger("outputs")
            return 1
        except Exception as e:
            self._error_placeholder.object += f"\nCould not process metadata file {card.filename}: {e}"
            self._error_placeholder.visible = True
            return 0

    async def _sync_metadata(self, filename: str):
        if not self.source_catalog:
            return
        await self.source_catalog._sync_metadata_to_vector_store(filename)

    # ──────────────────────────────────────────────────────────────────────────
    # Batch processing
    # ──────────────────────────────────────────────────────────────────────────

    def _process_files(self):
        self._error_placeholder.object = ""
        self._error_placeholder.visible = False

        if not self._file_cards:
            return 0, 0, 0

        source = None
        n_tables = 0
        n_metadata = 0
        table_upload_callbacks = {
            key.lstrip("."): value for key, value in self.upload_handlers.items()
        }
        custom_table_extensions = tuple(table_upload_callbacks)

        data_cards = [c for c in self._file_cards if c.file_type == "data"]
        metadata_cards = [c for c in self._file_cards if c.file_type == "metadata"]

        for card in data_cards:
            log_debug(f"Processing data card: {card.filename}.{card.extension} (alias: {card.alias})")
            if card.extension.endswith(custom_table_extensions):
                source = table_upload_callbacks[card.extension](
                    self.context, card.file_obj, card.alias, card.filename
                )
                if source is not None:
                    n_tables += len(source.get_tables())
                    self._register_source_output(source)
                    self.param.trigger("outputs")
            elif card.extension.endswith(TABLE_EXTENSIONS):
                if source is None:
                    existing_sources = self.outputs.get("sources", [])
                    for existing in existing_sources:
                        if isinstance(existing, DuckDBSource) and existing.ephemeral:
                            source = existing
                            break
                    if source is None:
                        source_id = f"{self.source_name_prefix}{self._count:06d}"
                        source = DuckDBSource(uri=":memory:", ephemeral=True, name=source_id, tables={})
                        self._count += 1
                table_name = card.alias
                filename = f"{card.filename}.{card.extension}"
                source.metadata.setdefault(table_name, {})["filename"] = filename
                n_tables += self._add_table(source, card.file_obj, card)
            else:
                self._error_placeholder.object += f"\n⚠️ Skipped '{card.filename}.{card.extension}': unsupported format."
                self._error_placeholder.visible = True

        for card in metadata_cards:
            n_metadata += self._add_metadata_file(card)

        log_debug(f"Processed files: {n_tables} tables, {n_metadata} metadata files")
        return n_tables, 0, n_metadata

    def _format_bytes(self, bytes_size: int) -> str:
        if bytes_size == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while bytes_size >= 1024 and i < len(size_names) - 1:
            bytes_size /= 1024.0
            i += 1
        return f"{bytes_size:.1f} {size_names[i]}"
