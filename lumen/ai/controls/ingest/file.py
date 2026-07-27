from __future__ import annotations

import asyncio
import io
import json
import pathlib

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Callable

    from ....sources.base import Source


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
            label=self.param.add_button_label,
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

    def _read_tables(
        self,
        file: io.BytesIO | io.StringIO,
        card: UploadedFileRow,
    ) -> tuple[FileReadResult | None, str | None]:
        """Read a file into DataFrames. Pure compute: safe to run in a thread.

        Touches no Panel/Bokeh models; errors are returned as a message string
        for the caller to surface on the main thread.

        Returns
        -------
        (result, error) where exactly one is None.
        """
        extension = card.extension
        filename = f"{card.filename}.{extension}"
        try:
            if extension.endswith("json"):
                # Use the more robust JSON parser on this class
                df = self._read_json_file(file, filename)
                return FileReadResult(tables={card.alias: df}), None
            result = read_file_to_dataframes(
                file, extension, alias=card.alias, sheet_name=card.sheet,
            )
            if result is None:
                return None, f"\n⚠️ Could not convert {filename!r}: unsupported format."
        except Exception as e:
            return None, f"\n⚠️ Error processing {filename!r}: {e}"
        return result, None

    def _commit_tables(
        self,
        duckdb_source: DuckDBSource,
        result: FileReadResult,
        card: UploadedFileRow,
    ) -> int:
        """Register read DataFrames with DuckDB and publish outputs.

        Must run on the main thread: mutates Panel models and triggers watchers.
        """
        conn = duckdb_source._connection
        filename = f"{card.filename}.{card.extension}"

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

    def _add_table(
        self,
        duckdb_source: DuckDBSource,
        file: io.BytesIO | io.StringIO,
        card: UploadedFileRow,
    ) -> int:
        """Read and register a file in one blocking call (read + commit)."""
        result, error = self._read_tables(file, card)
        if error is not None:
            self._error_placeholder.object += error
            self._error_placeholder.visible = True
            return 0
        return self._commit_tables(duckdb_source, result, card)

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

    def _ephemeral_source(self, source: DuckDBSource | None) -> DuckDBSource:
        """Reuse the batch's ephemeral in-memory source, or create it.

        Cheap and main-thread only (reads ``self.outputs``, bumps ``_count``).
        """
        if source is not None:
            return source
        for existing in self.outputs.get("sources", []):
            if isinstance(existing, DuckDBSource) and existing.ephemeral:
                return existing
        source_id = f"{self.source_name_prefix}{self._count:06d}"
        self._count += 1
        return DuckDBSource(uri=":memory:", ephemeral=True, name=source_id, tables={})

    def _begin_process_files(
        self,
    ) -> tuple[dict[str, Callable], tuple[str, ...], list[UploadedFileRow], list[UploadedFileRow]]:
        """Reset error state and partition staged cards. Main thread.

        Returns
        -------
        (handlers, handler_extensions, data_cards, metadata_cards)
        """
        self._error_placeholder.object = ""
        self._error_placeholder.visible = False
        handlers = {
            key.lstrip("."): value for key, value in self.upload_handlers.items()
        }
        return (
            handlers,
            tuple(handlers),
            [c for c in self._file_cards if c.file_type == "data"],
            [c for c in self._file_cards if c.file_type == "metadata"],
        )

    def _register_custom_source(self, source: Source | None) -> int:
        """Publish a Source returned by a custom upload handler. Main thread."""
        if source is None:
            return 0
        n_tables = len(source.get_tables())
        self._register_source_output(source)
        self.param.trigger("outputs")
        return n_tables

    def _prepare_table_card(
        self, source: DuckDBSource | None, card: UploadedFileRow
    ) -> DuckDBSource:
        """Ensure an ephemeral source exists and record the card's filename."""
        source = self._ephemeral_source(source)
        source.metadata.setdefault(card.alias, {})["filename"] = (
            f"{card.filename}.{card.extension}"
        )
        return source

    async def _process_files_async(self):
        """Non-blocking :meth:`_process_files`.

        Identical behaviour, except the expensive pure-compute work (custom
        upload handlers such as ``.h5ad``, and file parsing) runs in a worker
        thread so the event loop -- and therefore the UI -- stays responsive.
        All Panel model mutation stays on this (main) thread.

        Cards are processed sequentially rather than concurrently: they share a
        single DuckDB connection, which is not safe for concurrent writes.
        """
        if not self._file_cards:
            return 0, 0, 0

        callbacks, custom_exts, data_cards, metadata_cards = self._begin_process_files()
        source = None
        n_tables = n_metadata = 0
        errors: list[str] = []

        for card in data_cards:
            log_debug(f"Processing data card: {card.filename}.{card.extension} (alias: {card.alias})")
            if card.extension.endswith(custom_exts):
                # Off-thread: handler is pure compute returning a Source.
                source = await asyncio.to_thread(
                    callbacks[card.extension],
                    self.context, card.file_obj, card.alias, card.filename,
                )
                n_tables += self._register_custom_source(source)
            elif card.extension.endswith(TABLE_EXTENSIONS):
                source = self._prepare_table_card(source, card)
                # Off-thread: parse file into DataFrames.
                result, error = await asyncio.to_thread(
                    self._read_tables, card.file_obj, card
                )
                if error is not None:
                    errors.append(error)
                    continue
                n_tables += self._commit_tables(source, result, card)
            else:
                errors.append(
                    f"\n⚠️ Skipped '{card.filename}.{card.extension}': unsupported format."
                )

        for card in metadata_cards:
            n_metadata += self._add_metadata_file(card)

        if errors:
            self._error_placeholder.param.update(
                object=self._error_placeholder.object + "".join(errors), visible=True
            )

        log_debug(f"Processed files: {n_tables} tables, {n_metadata} metadata files")
        return n_tables, 0, n_metadata

    def _process_files(self):
        """Blocking variant of :meth:`_process_files_async`.

        Retained because it is called from synchronous contexts (notably the
        chat-input submit handler in ``lumen.ai.ui``). A coroutine cannot be run
        to completion from inside the already-running event loop, so the two
        entry points share every step via helpers rather than one wrapping the
        other; keep the loop below in sync with the async version.
        """
        if not self._file_cards:
            return 0, 0, 0

        callbacks, custom_exts, data_cards, metadata_cards = self._begin_process_files()
        source = None
        n_tables = n_metadata = 0
        errors: list[str] = []

        for card in data_cards:
            log_debug(f"Processing data card: {card.filename}.{card.extension} (alias: {card.alias})")
            if card.extension.endswith(custom_exts):
                source = callbacks[card.extension](
                    self.context, card.file_obj, card.alias, card.filename
                )
                n_tables += self._register_custom_source(source)
            elif card.extension.endswith(TABLE_EXTENSIONS):
                source = self._prepare_table_card(source, card)
                result, error = self._read_tables(card.file_obj, card)
                if error is not None:
                    errors.append(error)
                    continue
                n_tables += self._commit_tables(source, result, card)
            else:
                errors.append(
                    f"\n⚠️ Skipped '{card.filename}.{card.extension}': unsupported format."
                )

        for card in metadata_cards:
            n_metadata += self._add_metadata_file(card)

        if errors:
            self._error_placeholder.param.update(
                object=self._error_placeholder.object + "".join(errors), visible=True
            )

        log_debug(f"Processed files: {n_tables} tables, {n_metadata} metadata files")
        return n_tables, 0, n_metadata
