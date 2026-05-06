from __future__ import annotations

import asyncio
import io
import json
import pathlib

import pandas as pd
import param

from panel_material_ui import Button, Column as MuiColumn

from ....sources.duckdb import DuckDBSource
from ....util import detect_file_encoding
from ...utils import log_debug
from .base import BaseSourceControls
from .constants import TABLE_EXTENSIONS
from .file_row import UploadedFileRow
from .utils import FileReadResult, read_file_to_dataframes


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
        alias = card.alias
        filename = f"{card.filename}.{extension}"

        try:
            if extension.endswith("json"):
                # Use the more robust JSON parser on this class
                df = self._read_json_file(file, filename)
                result = FileReadResult(tables={alias: df})
            else:
                result = read_file_to_dataframes(
                    file, extension, alias=alias, sheet=card.sheet,
                )
                if result is None:
                    self._error_placeholder.object += (
                        f"\n⚠️ Could not convert {filename!r}: unsupported format."
                    )
                    self._error_placeholder.visible = True
                    return 0
        except Exception as e:
            self._error_placeholder.object += f"\n⚠️ Error processing {filename!r}: {e}"
            self._error_placeholder.visible = True
            return 0

        # Apply source-level params (e.g. spatial initializers)
        if result.source_params:
            duckdb_source.param.update(result.source_params)
            for init in result.source_params.get("initializers", []):
                conn.execute(init)

        added = 0
        first_table = None
        for tbl_name, df in result.tables.items():
            if df is None or df.empty:
                continue

            # Convert pandas StringDtype columns to object for DuckDB compatibility
            for col in df.columns:
                if isinstance(df[col].dtype, pd.StringDtype):
                    df[col] = df[col].astype(object)

            df_rel = conn.from_df(df)
            if tbl_name in result.conversions:
                conn.register(f"{tbl_name}_temp", df_rel)
                conn.execute(result.conversions[tbl_name])
                conn.unregister(f"{tbl_name}_temp")
            else:
                df_rel.to_view(tbl_name)

            duckdb_source.tables[tbl_name] = f"SELECT * FROM {tbl_name}"
            if first_table is None:
                first_table = tbl_name
            added += 1

        if added > 0:
            self._register_source_output(duckdb_source)
            self.outputs["table"] = first_table
            self.param.trigger("outputs")
            self._last_table = first_table

        if added == 0:
            self._error_placeholder.object += f"\n⚠️ {filename!r} contains no data."
            self._error_placeholder.visible = True

        return added

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
                added = self._add_table(source, card.file_obj, card)
                n_tables += added
            else:
                self._error_placeholder.object += f"\n⚠️ Skipped '{card.filename}.{card.extension}': unsupported format."
                self._error_placeholder.visible = True

        for card in metadata_cards:
            n_metadata += self._add_metadata_file(card)

        log_debug(f"Processed files: {n_tables} tables, {n_metadata} metadata files")
        return n_tables, 0, n_metadata
