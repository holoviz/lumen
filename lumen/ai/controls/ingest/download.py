from __future__ import annotations

import asyncio
import io
import pathlib
import re

from urllib.parse import urlparse

import param

from panel_material_ui import (
    Button, Column as MuiColumn, Row, TextAreaInput,
)

from ....sources.duckdb import DuckDBSource
from ....util import normalize_table_name
from .constants import TABLE_EXTENSIONS
from .file import FileSourceControls
from .result import SourceResult
from .utils import download_file, read_file_to_dataframes

_KAGGLE_URL_RE = re.compile(
    r'(?:https?://)?(?:www\.)?kaggle\.com/datasets/([^/?#]+/[^/?#]+)',
)


class DownloadSourceControls(FileSourceControls):
    """
    Controls for downloading files from URLs and extracting tabular data.

    Supports CSV, JSON, Parquet, Excel, and HTML files. For HTML pages,
    all tables are extracted and loaded as separate database tables.

    Kaggle dataset URLs (e.g. ``https://www.kaggle.com/datasets/owner/name``)
    are also supported and will download all data files in the dataset into
    a single source with one table per file.
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

    def __init__(self, **params):
        super().__init__(**params)
        try:
            import kagglehub
            self._kagglehub = kagglehub
        except ModuleNotFoundError:
            self._kagglehub = None
        if self._kagglehub is not None:
            self.input_placeholder = (
                "Enter URLs, one per line, and press <Shift+Enter> to download.\n"
                "Kaggle dataset URLs are also supported."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Kaggle helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_kaggle_ref(url: str) -> str | None:
        """Extract ``'owner/dataset'`` from a Kaggle URL, or ``None``."""
        m = _KAGGLE_URL_RE.match(url.strip())
        return m.group(1).rstrip("/") if m else None

    async def _download_kaggle_files(self, ref: str) -> tuple[dict[str, bytes], str | None]:
        """
        Download a Kaggle dataset and read supported files into memory.

        Returns
        -------
        tuple[dict[str, bytes], str | None]
            ``(files_dict, error_message)``
        """
        if self._kagglehub is None:
            return {}, (
                "kagglehub is required to download Kaggle datasets. "
                "Install it with: pip install kagglehub"
            )

        self.progress(f"Downloading Kaggle dataset '{ref}'…")

        try:
            path = await asyncio.get_event_loop().run_in_executor(
                None, self._kagglehub.dataset_download, ref,
            )
        except Exception as e:
            return {}, f"Kaggle download failed for '{ref}': {e}"

        dataset_dir = pathlib.Path(path)
        if not dataset_dir.is_dir():
            return {}, f"Kaggle download path is not a directory: {path}"

        self.progress("Reading downloaded files…")

        files: dict[str, bytes] = {}
        for file_path in sorted(dataset_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lstrip(".").lower() in TABLE_EXTENSIONS:
                files[file_path.name] = file_path.read_bytes()

        if not files:
            return {}, (
                f"No supported data files found in Kaggle dataset '{ref}'. "
                f"Directory contained: {[f.name for f in dataset_dir.rglob('*') if f.is_file()]}"
            )

        return files, None

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

                kaggle_ref = self._parse_kaggle_ref(url)
                if kaggle_ref:
                    files, error = await self._download_kaggle_files(kaggle_ref)
                    if error:
                        errors.append(f"{url}: {error}")
                    else:
                        downloaded_files.update(files)
                else:
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
            Kaggle dataset URLs are also supported (e.g.
            ``https://www.kaggle.com/datasets/owner/dataset``).

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

        kaggle_ref = self._parse_kaggle_ref(url)
        if kaggle_ref:
            return await self._fetch_kaggle(kaggle_ref)

        self.progress(f"Fetching {url[:80]}{'…' if len(url) > 80 else ''}…")
        filename, content, error = await download_file(url, progress=self.progress)

        if error:
            return SourceResult.empty(f"Download failed: {error}")

        return await self._content_to_result(url, filename, content)

    async def _fetch_kaggle(self, ref: str) -> SourceResult:
        """Download a Kaggle dataset and load all files into a single source."""
        files, error = await self._download_kaggle_files(ref)
        if error:
            return SourceResult.empty(error)

        source_id = f"{self.source_name_prefix}{self._count:06d}"
        source = DuckDBSource(uri=":memory:", ephemeral=True, name=source_id, tables={})
        self._count += 1
        conn = source._connection

        total_rows = 0
        tables_loaded = []
        for filename, content in files.items():
            suffix = pathlib.Path(filename).suffix.lstrip(".").lower()
            base_name = pathlib.Path(filename).stem
            alias = normalize_table_name(base_name) or "data"

            file_obj = io.BytesIO(content)
            try:
                result = read_file_to_dataframes(file_obj, suffix, alias=alias)
            except Exception as e:
                tables_loaded.append(f"'{filename}' (error: {e})")
                continue

            if result is None:
                continue

            source.param.update(result.source_params)
            for init in result.source_params.get("initializers", []):
                conn.execute(init)

            for tbl_name, df in result.tables.items():
                if df is None or df.empty:
                    continue
                df_rel = conn.from_df(df)
                if tbl_name in result.conversions:
                    conn.register(f"{tbl_name}_temp", df_rel)
                    conn.execute(result.conversions[tbl_name])
                    conn.unregister(f"{tbl_name}_temp")
                else:
                    df_rel.to_view(tbl_name)
                source.tables[tbl_name] = f"SELECT * FROM {tbl_name}"
                source.metadata[tbl_name] = {"filename": filename, "kaggle_ref": ref}
                total_rows += len(df)
                tables_loaded.append(f"'{tbl_name}' ({len(df):,} rows)")

        if not source.tables:
            return SourceResult.empty(
                f"No tables could be parsed from Kaggle dataset '{ref}'."
            )

        first_table = next(iter(source.tables))
        if len(source.tables) == 1:
            message = f"Loaded {total_rows:,} rows from Kaggle dataset '{ref}' into '{first_table}'"
        else:
            message = (
                f"Loaded {len(source.tables)} tables ({total_rows:,} total rows) "
                f"from Kaggle dataset '{ref}': {', '.join(tables_loaded)}"
            )

        return SourceResult.from_source(source, table=first_table, message=message)

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

        # Try to extract tabular data
        try:
            result = read_file_to_dataframes(file_obj, suffix, alias=alias)
            dfs = result.tables if result is not None else {}
        except Exception as e:
            return SourceResult.empty(f"Could not parse {filename!r}: {e}")

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

        # For HTML pages, also extract text content as a document
        doc_added = False
        if suffix in ("html", "htm") and self.source_catalog:
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

        # All tables are in source.tables; first_table is for default selection
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
