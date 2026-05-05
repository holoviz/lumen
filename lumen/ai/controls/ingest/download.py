from __future__ import annotations

import asyncio
import io
import pathlib

from urllib.parse import urlparse

import pandas as pd
import param

from panel_material_ui import (
    Button, Column as MuiColumn, Row, TextAreaInput,
)

from ....sources.duckdb import DuckDBSource
from ....util import normalize_table_name
from .file import FileSourceControls
from .result import SourceResult
from .utils import download_file, read_html_tables, read_json_to_dataframe


class DownloadSourceControls(FileSourceControls):
    """
    Controls for downloading files from URLs and extracting tabular data.

    Supports CSV, JSON, Parquet, Excel, and HTML files. For HTML pages,
    all tables are extracted and loaded as separate database tables.
    """

    download_url = param.String(
        default="",
        label="Download URL(s)",
        doc="Enter one or more URLs, one per line, and press <Shift+Enter> to download",
    )

    input_placeholder = param.String(
        default="Enter URLs, one per line, and press <Shift+Enter> to download"
    )

    _active_download_task = param.ClassSelector(class_=asyncio.Task)

    _cached_tools = param.List(default=None, allow_None=True)

    load_mode = "manual"

    source_name_prefix = "DownloadedSource"

    label = '<span class="material-icons" style="vertical-align: middle;">download</span> Fetch Remote Data'

    _supports_tools = True

    def _render_layout(self):
        """Build download-specific layout with URL input."""
        self._url_input = TextAreaInput.from_param(
            self.param.download_url,
            placeholder=self.param.input_placeholder,
            rows=2,
            margin=10,
            sizing_mode="stretch_width",
            disabled=self.param.disabled,
        )
        self._url_input.param.watch(self._handle_urls, "enter_pressed")

        self._cancel_button = Button.from_param(
            self.param.cancel,
            name="Cancel",
            icon="close",
            on_click=self._handle_cancel,
            visible=self.param._active_download_task.rx.is_not(None),
            height=42,
        )

        self._active_download_task = None

        return MuiColumn(
            self._url_input,
            self._upload_cards,
            Row(self._add_button, self._cancel_button),
            self._error_placeholder,
            self._message_placeholder,
            self.progress.bar,
            self.progress.description,
            sizing_mode="stretch_width",
        )

    def _handle_cancel(self, event):
        """Cancel the active download task."""
        if self._active_download_task and not self._active_download_task.done():
            self._active_download_task.cancel()

    def _on_download_done(self, task):
        """Handle download task completion, cancellation, or failure."""
        if task is not self._active_download_task:
            return

        if task.cancelled():
            self._message_placeholder.param.update(object="Download cancelled.", visible=True)
        elif task.exception() is not None:
            self._message_placeholder.param.update(
                object=f"Download failed: {task.exception()}", visible=True
            )

        self.progress.clear()
        self._active_download_task = None

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Basic URL validation."""
        try:
            result = urlparse(url.strip())
            return bool(result.scheme and result.netloc)
        except Exception:
            return False

    async def _download_file(self, url: str):
        """Download a single URL, delegating to the shared utility."""
        return await download_file(url, progress=self.progress)

    def _handle_urls(self, event):
        """Handle URL input and trigger downloads."""
        url_text = self._url_input.value_input or self._url_input.value
        if not url_text:
            return

        urls = [line.strip() for line in url_text.split("\n") if line.strip()]
        valid_urls = [url for url in urls if self._is_valid_url(url)]
        if not valid_urls:
            return

        self._error_placeholder.visible = False
        self._message_placeholder.param.update(
            object=f"Preparing to download {len(valid_urls)} file(s)...",
            visible=True,
        )

        if self._active_download_task and not self._active_download_task.done():
            self._active_download_task.cancel()

        self._active_download_task = asyncio.create_task(
            self._download_and_process_urls(valid_urls)
        )
        self._active_download_task.add_done_callback(self._on_download_done)

    async def _download_and_process_urls(self, urls: list[str]):
        """Download URLs and generate file cards for processing."""
        downloaded_files = {}
        errors = []

        try:
            for i, url in enumerate(urls, 1):
                if asyncio.current_task().cancelled():
                    break

                self._message_placeholder.object = (
                    f"Processing {i} of {len(urls)}: {url[:50]}{'...' if len(url) > 50 else ''}"
                )
                filename, content, error = await self._download_file(url)

                if error:
                    if "cancelled" in error.lower():
                        break
                    errors.append(f"{url}: {error}")
                else:
                    downloaded_files[filename] = content

        except asyncio.CancelledError:
            self.progress.bar.visible = False
            self.progress.description.visible = False
            raise

        self._generate_file_cards(downloaded_files)

        if errors:
            self._error_placeholder.param.update(
                object=f"Download errors:\n{chr(10).join(errors)}",
                visible=True,
            )

        if downloaded_files:
            self._message_placeholder.object = (
                f"Successfully downloaded {len(downloaded_files)} file(s). "
                "Click 'Confirm file(s)' to process."
            )
        else:
            self._message_placeholder.visible = False

    @param.depends("add", watch=True)
    def _on_add(self):
        """Process downloaded files."""
        if not self._file_cards:
            return

        self._active_download_task = None

        with self._layout.param.update(loading=True):
            n_tables, n_docs, n_metadata = self._process_files()

            total_files = len(self._file_cards)
            if self.clear_uploads:
                self._clear_uploads()
                self._url_input.value = ""

            if (n_tables + n_docs + n_metadata) > 0:
                self._message_placeholder.param.update(
                    object=(
                        f"Successfully processed {total_files} files "
                        f"({n_tables} table(s), {n_metadata} metadata file(s))."
                    ),
                    visible=True,
                )
            self._error_placeholder.object = self._error_placeholder.object.strip()

        self._count += 1

        if (n_tables + n_docs + n_metadata) > 0:
            self.param.trigger("upload_successful")

    # ──────────────────────────────────────────────────────────────────────────
    # Agent integration
    # ──────────────────────────────────────────────────────────────────────────

    def as_tools(
        self, query: str | None = None, top_k: int = 5,
    ) -> list[tuple[str, callable]]:
        """
        Return a tool that allows the agent to download data from URLs.

        The tool accepts a URL string and downloads the file, extracting
        tabular data from CSV, JSON, Parquet, Excel, or HTML files.
        """
        if self._cached_tools is not None:
            return self._cached_tools

        async def fetch_url(url: str) -> SourceResult:
            """
            Download data from a URL and load it into the database.

            Supports CSV, JSON, Parquet, Excel (.xlsx), and HTML files.
            For HTML pages, extracts all tables found on the page.

            Parameters
            ----------
            url : str
                The URL to download data from.

            Returns
            -------
            SourceResult
                Contains the loaded source(s) and table name(s).
            """
            return await self._fetch_url(url)

        self._cached_tools = [("Fetch URL", fetch_url)]
        return self._cached_tools

    async def _fetch_url(self, url: str) -> SourceResult:
        """Download and process a single URL, returning a SourceResult."""
        if not self._is_valid_url(url):
            return SourceResult.empty(f"Invalid URL: {url}")

        self.progress(f"Fetching {url[:80]}{'…' if len(url) > 80 else ''}…")
        filename, content, error = await download_file(url, progress=self.progress)

        if error:
            return SourceResult.empty(f"Download failed: {error}")

        return await self._content_to_result(url, filename, content)

    async def _content_to_result(self, url: str, filename: str, content: bytes) -> SourceResult:
        """Convert downloaded content to a SourceResult with DuckDB tables and/or document."""
        suffix = pathlib.Path(filename).suffix.lstrip(".").lower()

        source_id = f"{self.source_name_prefix}{self._count:06d}"
        source = DuckDBSource(uri=":memory:", ephemeral=True, name=source_id, tables={})
        self._count += 1

        # Derive alias from URL path for better identification
        parsed = urlparse(url)
        url_path = parsed.path.rstrip("/")
        if url_path:
            base_name = url_path.split("/")[-1].rsplit(".", 1)[0] or url_path.split("/")[-2]
        else:
            base_name = parsed.netloc.replace(".", "_")
        alias = normalize_table_name(base_name) or "data"

        file_obj = io.BytesIO(content)
        dfs = {}
        is_html = suffix in ("html", "htm") or suffix not in ("csv", "parq", "parquet", "json", "xlsx")

        # Try to extract tabular data
        try:
            file_obj.seek(0)
            if suffix == "csv":
                dfs = {alias: pd.read_csv(file_obj, parse_dates=True, sep=None, engine="python")}
            elif suffix in ("parq", "parquet"):
                dfs = {alias: pd.read_parquet(file_obj)}
            elif suffix == "json":
                dfs = {alias: read_json_to_dataframe(file_obj.read())}
            elif suffix == "xlsx":
                dfs = {alias: pd.read_excel(file_obj)}
            else:
                # HTML - try to extract tables, but don't fail if none found
                try:
                    dfs = read_html_tables(content, alias)
                except ValueError:
                    dfs = {}  # No tables found, will extract as document
        except Exception as e:
            if not is_html:
                return SourceResult.empty(f"Could not parse {filename!r}: {e}")
            # For HTML, continue to try document extraction

        # Load tables into DuckDB
        total_rows = 0
        tables_loaded = []
        for table_name, df in dfs.items():
            if df.empty:
                continue
            source._connection.from_df(df).to_view(table_name)
            source.tables[table_name] = f"SELECT * FROM {table_name}"
            source.metadata[table_name] = {"url": url, "filename": filename}
            total_rows += len(df)
            tables_loaded.append(f"'{table_name}' ({len(df):,} rows)")

        # For HTML, also extract text content as a document
        doc_added = False
        if is_html and self.source_catalog:
            text_content = self._extract_metadata_content(io.BytesIO(content), ".html")
            if text_content and len(text_content.strip()) > 100:
                doc_added = await self._add_document(url, alias, text_content)

        # Build result message
        messages = []
        if tables_loaded:
            if len(tables_loaded) == 1:
                messages.append(f"Loaded {total_rows:,} rows from {url} into table '{next(iter(dfs.keys()))}'")
            else:
                messages.append(f"Loaded {len(tables_loaded)} tables from {url}: {', '.join(tables_loaded)}")
        if doc_added:
            messages.append(f"Indexed page text as document '{alias}'")

        if not messages:
            return SourceResult.empty(f"No tables or text content found at {url}")

        # If only document was indexed (no tables), return document-only result
        if not tables_loaded and doc_added:
            return SourceResult.from_document(". ".join(messages))

        first_table = next(iter(source.tables), None)
        return SourceResult.from_source(source, table=first_table, message=". ".join(messages))

    async def _add_document(self, url: str, alias: str, text_content: str) -> bool:
        """Add text content as a document to the vector store."""
        if not self.source_catalog:
            return False
        try:
            filename = f"{alias}.txt"
            existing = [m["filename"] for m in self.source_catalog._available_metadata]
            if filename in existing:
                counter = 1
                while filename in existing:
                    filename = f"{alias}_{counter}.txt"
                    counter += 1

            metadata_entry = {
                "filename": filename,
                "display_name": alias,
                "content": text_content,
                "url": url,
            }
            self.source_catalog._available_metadata.append(metadata_entry)
            await self.source_catalog._sync_metadata_to_vector_store(filename)
            self.source_catalog.sync()
            return True
        except Exception:
            return False
