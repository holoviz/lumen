import asyncio
import io
import pathlib
import zipfile

from urllib.parse import urlparse

import aiohttp
import pandas as pd
import param

from panel.chat import ChatAreaInput
from panel.layout import (
    Column, FlexBox, Row, Tabs,
)
from panel.pane.markup import HTML
from panel.viewable import Viewer
from panel.widgets import (
    Button, FileDropper, Select, Tabulator, TextInput, ToggleIcon, Tqdm,
)

from ..sources.duckdb import DuckDBSource
from ..util import detect_file_encoding
from .memory import _Memory, memory

TABLE_EXTENSIONS = ("csv", "parquet", "parq", "json", "xlsx", "geojson", "wkt", "zip")

# Download configuration constants
class DownloadConfig:
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    TIMEOUT_SECONDS = 300     # 5 minutes
    PROGRESS_UPDATE_INTERVAL = 50  # Update every 50 chunks
    DEFAULT_HASH_MODULO = 10000
    UNKNOWN_SIZE_MAX = 1000000000  # 1GB max for unknown file sizes

    # Connection settings
    CONNECTION_LIMIT = 100
    CONNECTION_LIMIT_PER_HOST = 30
    KEEPALIVE_TIMEOUT = 30


class MediaControls(Viewer):

    alias = param.String(default="", doc="What to name the uploaded file when querying it")

    filename = param.String(default="", doc="Filename")

    extension = param.String(default="", doc="File extension")

    _load = param.Event(doc="Load table")

    def __init__(self, file_obj: io.BytesIO, **params):
        filename = params["filename"]
        if "/" in filename:
            filename, extension = filename.rsplit("/")[-1].split(".", maxsplit=1)
        elif not self.filename and filename:
            filename, extension = filename.rsplit(".", maxsplit=1)
        params["filename"] = filename
        params["alias"] = filename.replace("-", "_")
        params["extension"] = extension
        super().__init__(**params)
        self.file_obj = file_obj

    def __panel__(self):
        return self.box


class DocumentControls(MediaControls):

    def __init__(self, file_obj: io.BytesIO, **params):
        super().__init__(file_obj, **params)
        self._name_input = TextInput.from_param(
            self.param.alias, name="Document alias",
        )
        self._metadata_input = TextInput(name="Metadata", description="Comments or notes about this document")
        self.box = FlexBox(
            self._name_input,
            self._metadata_input,
        )


class TableControls(MediaControls):

    sheet = param.Selector(default=None, objects=[], doc="Sheet")

    def __init__(self, file_obj: io.BytesIO, **params):
        super().__init__(file_obj, **params)
        self._name_input = TextInput.from_param(
            self.param.alias, name="Table alias",
        )
        self._sheet_select = Select.from_param(
            self.param.sheet, name="Sheet", visible=False
        )
        self.box = FlexBox(
            self._name_input,
            self._sheet_select,
        )
        self.param.trigger("filename")  # fix name
        self.param.trigger("alias")  # fix alias
        self.param.trigger("_load")  # to prevent blocking

    @param.depends("_load", watch=True)
    async def _select_sheet(self):
        if not self.extension.endswith("xlsx"):
            return
        import openpyxl

        wb = openpyxl.load_workbook(self.file_obj, read_only=True)
        with param.parameterized.batch_call_watchers(self):
            self.param.sheet.objects = wb.sheetnames
            self.sheet = wb.sheetnames[0]
            self._sheet_select.visible = True

    @param.depends("alias", watch=True)
    async def _replace_with_underscore(self):
        self.alias = "".join(
            c if c.isalnum() else "_" for c in self.alias
        ).strip("_").lower()


class SourceControls(Viewer):

    add = param.Event(doc="Use uploaded file(s)")

    cancel = param.Event(doc="Cancel")

    cancellable = param.Boolean(default=True, doc="Show cancel button")

    clear_uploads = param.Boolean(default=True, doc="Clear uploaded file tabs")

    memory = param.ClassSelector(class_=_Memory, default=None, doc="""
        Local memory which will be used to provide the agent context.
        If None the global memory will be used.""")

    multiple = param.Boolean(default=False, doc="Allow multiple files")

    replace_controls = param.Boolean(default=False, doc="Replace controls")

    table_upload_callbacks = param.Dict(default={}, doc="""
        Dictionary mapping from file extensions to callback function,
        e.g. {"hdf5": ...}. The callback function should accept the file bytes and
        table alias.""")

    _last_table = param.String(default="", doc="Last table added")

    _count = param.Integer(default=0, doc="Count of sources added")

    def __init__(self, **params):
        super().__init__(**params)
        self.tables_tabs = Tabs(sizing_mode="stretch_width")
        self._markitdown = None
        self._file_input = FileDropper(
            layout="compact",
            multiple=self.param.multiple,
            margin=(10, 10, 0, 10),
            sizing_mode="stretch_width",
            # accepted_filetypes=[".csv", ".parquet", ".parq", ".json", ".xlsx"],
        )
        self._file_input.param.watch(self._generate_media_controls, "value")
        self._upload_tabs = Tabs(sizing_mode="stretch_width", closable=True)

        # URL input for downloading files
        self._url_input = ChatAreaInput(
            placeholder="Enter URLs, one per line, and press <Enter> to download.",
            rows=4,
            margin=(10, 10, 0, 10),
            sizing_mode="stretch_width",
        )
        self._url_input.param.watch(self._handle_urls, "enter_pressed")

        self._input_tabs = Tabs(
            ("File Input", self._file_input),
            ("Text Input", self._url_input),
            sizing_mode="stretch_both",
            dynamic=True,
        )

        self._add_button = Button.from_param(
            self.param.add,
            name="Use file(s)",
            icon="table-plus",
            visible=self._upload_tabs.param["objects"].rx().rx.len() > 0,
            button_type="success",
        )

        self._cancel_button = Button.from_param(
            self.param.cancel,
            name="Cancel",
            icon="circle-x",
            visible=self.param.cancellable.rx().rx.bool() and self._add_button.param.clicks.rx() == 0,
        )
        self._cancel_button.param.watch(self._handle_cancel, "clicks")

        self._error_placeholder = HTML("", visible=False, margin=(0, 10))
        self._message_placeholder = HTML("", visible=False, margin=(0, 10))

        # Progress bar for downloads
        self._progress_bar = Tqdm(
            visible=False,
            margin=(0, 10, 0, 10),
            sizing_mode="stretch_width"
        )

        self.menu = Column(
            self._input_tabs,
            self._upload_tabs,
            Row(self._add_button, self._cancel_button),
            self.tables_tabs,
            self._error_placeholder,
            self._message_placeholder,
            self._progress_bar,
            sizing_mode="stretch_width",
        )

        self._media_controls = []
        self._downloaded_media_controls = []  # Track downloaded file controls separately
        self._active_download_task = None  # Track active download task for cancellation

    @property
    def _memory(self):
        return memory if self.memory is None else self.memory

    def _handle_cancel(self, event):
        """Handle cancel button click by cancelling active download task"""
        if self._active_download_task and not self._active_download_task.done():
            self._active_download_task.cancel()
            self._progress_bar.visible = False
            self._message_placeholder.param.update(
                object="Download cancelled by user.",
                visible=True
            )

    def _is_valid_url(self, url):
        """Basic URL validation"""
        try:
            result = urlparse(url.strip())
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL with fallback logic"""
        parsed = urlparse(url)
        filename = parsed.path.split('/')[-1] if parsed.path else 'downloaded_file'

        # Remove query parameters from filename if present
        if '?' in filename:
            filename = filename.split('?')[0]

        # If no valid filename or extension, use default
        if not filename or '.' not in filename:
            filename = f"data_{abs(hash(url)) % DownloadConfig.DEFAULT_HASH_MODULO}.json"

        return filename

    def _extract_filename_from_headers(self, response_headers: dict, default_filename: str) -> str:
        """Extract filename from HTTP headers if available"""
        if 'content-disposition' not in response_headers:
            return default_filename

        import re
        cd = response_headers['content-disposition']
        matches = re.findall('filename="?([^"]+)"?', cd)
        if matches:
            suggested_name = matches[0]
            if '.' in suggested_name:  # Only use if it has an extension
                return suggested_name

        return default_filename

    def _setup_progress_bar(self, content_length: str) -> int:
        """Setup progress bar based on content length"""
        if content_length:
            total_size = int(content_length)
            self._progress_bar.max = max(total_size, 1)  # Ensure minimum of 1
            self._progress_bar.visible = True
            self._progress_bar.value = 0
            return total_size
        else:
            # Unknown size, use large number for indeterminate progress
            self._progress_bar.max = DownloadConfig.UNKNOWN_SIZE_MAX
            self._progress_bar.visible = True
            self._progress_bar.value = 0
            return 0

    def _update_download_progress(self, filename: str, downloaded_size: int, total_size: int, chunk_count: int):
        """Update progress bar if it's time for an update"""
        if chunk_count % DownloadConfig.PROGRESS_UPDATE_INTERVAL == 0:
            if total_size > 0:
                # Only set value if downloaded size is reasonable compared to max
                if downloaded_size <= self._progress_bar.max:
                    self._progress_bar.value = downloaded_size
                    progress_desc = f"Downloading {filename}: {self._format_bytes(downloaded_size)}/{self._format_bytes(total_size)}"
                else:
                    # Don't update value if it would exceed max (compressed content case)
                    progress_desc = f"Downloading {filename}: {self._format_bytes(downloaded_size)}"
            else:
                # Unknown size - just show downloaded amount and update progress
                self._progress_bar.value = min(downloaded_size, self._progress_bar.max)
                progress_desc = f"Downloading {filename}: {self._format_bytes(downloaded_size)}"

            self._progress_bar.desc = progress_desc

    def _format_bytes(self, bytes_size):
        """Convert bytes to human readable format"""
        if bytes_size == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while bytes_size >= 1024 and i < len(size_names) - 1:
            bytes_size /= 1024.0
            i += 1
        return f"{bytes_size:.1f} {size_names[i]}"

    def _finalize_download_progress(self, filename: str, downloaded_size: int, total_size: int):
        """Show final progress update when download completes"""
        if total_size > 0:
            # Set final value if it doesn't exceed max
            if downloaded_size <= self._progress_bar.max:
                self._progress_bar.value = downloaded_size
                self._progress_bar.desc = f"Downloaded {filename}: {self._format_bytes(downloaded_size)}/{self._format_bytes(total_size)}"
            else:
                # Size mismatch - show final size without updating value
                self._progress_bar.desc = f"Downloaded {filename}: {self._format_bytes(downloaded_size)}"
        else:
            # Unknown size - set value safely and show final size
            self._progress_bar.value = min(downloaded_size, self._progress_bar.max)
            self._progress_bar.desc = f"Downloaded {filename}: {self._format_bytes(downloaded_size)}"

    def _format_download_error(self, error: Exception) -> str:
        """Format download error messages consistently"""
        if isinstance(error, aiohttp.ClientError):
            return f"Network error: {error!s}"
        elif isinstance(error, asyncio.TimeoutError):
            return f"Download timeout ({DownloadConfig.TIMEOUT_SECONDS}s exceeded)"
        elif isinstance(error, asyncio.CancelledError):
            return "Download cancelled"
        else:
            return f"Download failed: {error!s}"

    def _create_file_object(self, file_data: bytes, suffix: str):
        """Create appropriate file object based on file type"""
        if suffix == "csv":
            encoding = detect_file_encoding(file_obj=file_data)
            return io.BytesIO(file_data.decode(encoding).encode("utf-8")) if isinstance(file_data, bytes) else io.StringIO(file_data)
        else:
            return io.BytesIO(file_data) if isinstance(file_data, bytes) else io.StringIO(file_data)

    def _create_media_controls(self, file_obj: io.BytesIO, filename: str):
        """Factory method to create appropriate media controls"""
        suffix = pathlib.Path(filename).suffix.lstrip(".").lower()
        table_extensions = TABLE_EXTENSIONS + tuple(key.lstrip(".").lower() for key in self.table_upload_callbacks.keys())

        if suffix in table_extensions:
            return TableControls(file_obj, filename=filename)
        else:
            return DocumentControls(file_obj, filename=filename)

    async def _download_file(self, url):
        """Download file from URL with progress bar and return (filename, file_content, error)"""
        try:
            # Extract filename from URL
            filename = self._extract_filename_from_url(url)

            timeout = aiohttp.ClientTimeout(total=DownloadConfig.TIMEOUT_SECONDS)
            connector = aiohttp.TCPConnector(
                limit=DownloadConfig.CONNECTION_LIMIT,
                limit_per_host=DownloadConfig.CONNECTION_LIMIT_PER_HOST,
                keepalive_timeout=DownloadConfig.KEEPALIVE_TIMEOUT,
                enable_cleanup_closed=True
            )

            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(url) as response:
                    response.raise_for_status()

                    # Try to get filename from headers
                    filename = self._extract_filename_from_headers(response.headers, filename)

                    # Setup progress bar
                    content_length = response.headers.get('content-length')
                    total_size = self._setup_progress_bar(content_length)

                    # Download in chunks
                    downloaded_data = io.BytesIO()
                    downloaded_size = 0
                    chunk_count = 0
                    size_mismatch_detected = False

                    async for chunk in response.content.iter_chunked(DownloadConfig.CHUNK_SIZE):
                        # Check for cancellation during download
                        if asyncio.current_task().cancelled():
                            raise asyncio.CancelledError()

                        downloaded_data.write(chunk)
                        downloaded_size += len(chunk)
                        chunk_count += 1

                        # Check if we've exceeded the expected total size (compressed content issue)
                        if not size_mismatch_detected and total_size > 0 and downloaded_size > total_size:
                            # Switch to indeterminate progress - Content-Length was wrong
                            # Use a large max value and don't update progress bar value anymore
                            self._progress_bar.max = max(downloaded_size * 2, DownloadConfig.UNKNOWN_SIZE_MAX)
                            total_size = 0  # Treat as unknown size from now on
                            size_mismatch_detected = True

                        # Update progress periodically
                        self._update_download_progress(filename, downloaded_size, total_size, chunk_count)
                        await asyncio.sleep(0)  # Allow UI updates

                    # Final progress update
                    self._finalize_download_progress(filename, downloaded_size, total_size)

                    # Get the downloaded data
                    downloaded_data.seek(0)
                    content = downloaded_data.read()

                    self._progress_bar.visible = False
                    return filename, content, None

        except (aiohttp.ClientError, asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
            self._progress_bar.visible = False
            return None, None, self._format_download_error(e)

    def _handle_urls(self, event):
        """Handle URL input and trigger downloads"""
        # Get the current value from the input widget
        url_text = self._url_input.value
        if not url_text:
            return

        urls = [line.strip() for line in url_text.split('\n') if line.strip()]
        valid_urls = [url for url in urls if self._is_valid_url(url)]

        if not valid_urls:
            return

        # Clear previous errors and show loading message
        self._error_placeholder.visible = False
        self._message_placeholder.param.update(
            object=f"Preparing to download {len(valid_urls)} file(s)...",
            visible=True
        )

        # Cancel any existing download task
        if self._active_download_task and not self._active_download_task.done():
            self._active_download_task.cancel()

        self._active_download_task = asyncio.create_task(self._download_and_process_urls(valid_urls))
        self._active_download_task.add_done_callback(
            lambda t: self._message_placeholder.param.update(
                object="Download complete." if not t.cancelled() else "Download cancelled.",
                visible=True
            ) if not t.cancelled() or t.exception() is None else None
        )

    async def _download_and_process_urls(self, urls):
        """Download URLs and add them to file input for processing"""
        downloaded_files = {}
        errors = []

        try:
            for i, url in enumerate(urls, 1):
                # Check if task was cancelled
                if asyncio.current_task().cancelled():
                    break

                self._message_placeholder.object = f"Processing {i} of {len(urls)}: {url[:50]}{'...' if len(url) > 50 else ''}"
                filename, content, error = await self._download_file(url)

                if error:
                    if "cancelled" in error.lower():
                        # Stop processing if download was cancelled
                        break
                    errors.append(f"{url}: {error}")
                else:
                    downloaded_files[filename] = content

        except asyncio.CancelledError:
            # Clean up and re-raise
            self._progress_bar.visible = False
            raise

        # Add downloaded files as individual tabs to the main input tabs
        if downloaded_files:
            self._add_downloaded_files_as_tabs(downloaded_files)

        # Update status messages
        if errors:
            error_msg = "\n".join(errors)
            self._error_placeholder.param.update(
                object=f"Download errors:\n{error_msg}",
                visible=True
            )

        success_count = len(downloaded_files)
        if success_count > 0:
            self._message_placeholder.object = f"Successfully downloaded {success_count} file(s)"
        else:
            self._message_placeholder.visible = False

    def _add_downloaded_files_as_tabs(self, downloaded_files):
        """Add downloaded files as individual tabs in the main input tabs"""
        for filename, file_data in downloaded_files.items():
            suffix = pathlib.Path(filename).suffix.lstrip(".").lower()

            # Create file object using helper
            file_obj = self._create_file_object(file_data, suffix)

            # Create media controls using helper
            media_controls = self._create_media_controls(file_obj, filename)

            # Add as a new tab
            display_name = pathlib.Path(filename).stem
            self._upload_tabs.append((display_name, media_controls))
            self._downloaded_media_controls.append(media_controls)

        # Show add button since we have files to process
        self._add_button.visible = True

    def _generate_media_controls(self, event):
        """Generate media controls for uploaded files"""
        self._upload_tabs.clear()
        self._media_controls.clear()

        if not self._file_input.value:
            self._add_button.visible = len(self._downloaded_media_controls) > 0
            return

        for filename, file_data in self._file_input.value.items():
            suffix = pathlib.Path(filename).suffix.lstrip(".").lower()

            # Create file object and media controls using helpers
            file_obj = self._create_file_object(file_data, suffix)
            media_controls = self._create_media_controls(file_obj, filename)

            self._upload_tabs.append((filename, media_controls))
            self._media_controls.append(media_controls)

        # Show add button if we have files to process
        self._add_button.visible = len(self._upload_tabs) > 0 or len(self._downloaded_media_controls) > 0

    def _add_table(
        self,
        duckdb_source: DuckDBSource,
        file: io.BytesIO | io.StringIO,
        table_controls: TableControls,
    ) -> int:
        conn = duckdb_source._connection
        extension = table_controls.extension
        table = table_controls.alias
        sql_expr = f"SELECT * FROM {table}"
        params = {}
        conversion = None
        if extension.endswith("csv"):
            df = pd.read_csv(file, parse_dates=True)
        elif extension.endswith(("parq", "parquet")):
            df = pd.read_parquet(file)
        elif extension.endswith("json"):
            df = pd.read_json(file)
        elif extension.endswith("xlsx"):
            sheet = table_controls.sheet
            df = pd.read_excel(file, sheet_name=sheet)
        elif extension.endswith(('geojson', 'wkt', 'zip')):
            if extension.endswith('zip'):
                zf = zipfile.ZipFile(file)
                if not any(f.filename.endswith('shp') for f in zf.filelist):
                    raise ValueError("Could not interpret zip file contents")
                file.seek(0)
            import geopandas as gpd
            geo_df = gpd.read_file(file)
            df = pd.DataFrame(geo_df)
            df['geometry'] = geo_df['geometry'].to_wkb()
            params['initializers'] = init = ["""
            INSTALL spatial;
            LOAD spatial;
            """]
            conn.execute(init[0])
            cols = ', '.join(f'"{c}"' for c in df.columns if c != 'geometry')
            conversion = f'CREATE TEMP TABLE {table} AS SELECT {cols}, ST_GeomFromWKB(geometry) as geometry FROM {table}_temp'
        else:
            self._error_placeholder.object += f"\nCould not convert {table_controls.filename}.{extension}."
            self._error_placeholder.visible = True
            return 0

        duckdb_source.param.update(params)
        df_rel = conn.from_df(df)
        if conversion:
            conn.register(f'{table}_temp', df_rel)
            conn.execute(conversion)
            conn.unregister(f'{table}_temp')
        else:
            df_rel.to_view(table)
        duckdb_source.tables[table] = sql_expr
        self._memory["source"] = duckdb_source
        self._memory["table"] = table
        if "sources" in self._memory:
            self._memory["sources"].append(duckdb_source)
            self._memory.trigger("sources")
        else:
            self._memory["sources"] = [duckdb_source]
        self._last_table = table
        return 1

    def _add_document(
        self,
        file: io.BytesIO,
        document_controls: DocumentControls
    ) -> int:
        if self._markitdown is None:
            from markitdown import MarkItDown, UnsupportedFormatException
            self._markitdown = MarkItDown()
            self._unsupported_exception = UnsupportedFormatException
        extension = document_controls.extension
        try:
            text = self._markitdown.convert_stream(
                file, file_extension=extension
            ).text_content
        except self._unsupported_exception:
            self._error_placeholder.object += f"\nCould not convert {document_controls.filename}.{extension}."
            self._error_placeholder.visible = True
            return 0

        metadata = {
            "filename": f"{document_controls.filename}.{document_controls.extension}",
            "comments": document_controls._metadata_input.value,
        }
        document = {"text": text, "metadata": metadata}
        if "document_sources" in self._memory:
            for i, source in enumerate(self._memory["document_sources"]):
                if source.get("metadata", {})["filename"] == metadata["filename"]:
                    self._memory["document_sources"][i] = document
                    break
            else:
                self._memory["document_sources"].append(document)
        else:
            self._memory["document_sources"] = [document]
        return 1

    @param.depends("add", watch=True)
    def add_medias(self):
        # Combine both uploaded files and downloaded files for processing
        all_media_controls = self._media_controls + self._downloaded_media_controls

        # Only proceed if we have files to process
        if len(all_media_controls) == 0:
            return

        with self.menu.param.update(loading=True):

            source = None
            n_tables = 0
            n_docs = 0
            table_upload_callbacks = {
                key.lstrip("."): value
                for key, value in self.table_upload_callbacks.items()
            }
            custom_table_extensions = tuple(table_upload_callbacks)

            for media_controls in all_media_controls:
                if media_controls.extension.endswith(custom_table_extensions):
                    n_tables += int(table_upload_callbacks[media_controls.extension](
                        media_controls.file_obj, media_controls.alias
                    ))
                elif media_controls.extension.endswith(TABLE_EXTENSIONS):
                    if source is None:
                        source_id = f"UploadedSource{self._count:06d}"
                        source = DuckDBSource(uri=":memory:", ephemeral=True, name=source_id, tables={})
                    n_tables += self._add_table(source, media_controls.file_obj, media_controls)
                else:
                    n_docs += self._add_document(media_controls.file_obj, media_controls)

            if self.replace_controls:
                src = self._memory.get("source")
                if src:
                    self.tables_tabs[:] = [
                        (t, Tabulator(src.get(t), sizing_mode="stretch_both", pagination="remote"))
                        for t in src.get_tables()
                    ]
                self.menu[0].visible = False
                self._add_button.visible = False
                self._cancel_button.visible = False
                if n_tables == 0:
                    self.tables_tabs.visible = False
                    self.menu.height = 70

            if self.clear_uploads:
                # Clear uploaded files from view
                self._upload_tabs.clear()
                self._media_controls.clear()
                # Also clear downloaded file tabs (anything beyond the first 2 tabs)
                while len(self._input_tabs) > 2:
                    self._input_tabs.pop()
                self._downloaded_media_controls.clear()
                self._add_button.visible = False
                self._file_input.value = {}
                self._url_input.value = ""

            if n_docs > 0:
                # Rather than triggering document sources on every upload, trigger it once
                self._memory.trigger("document_sources")

            # Clear uploaded files and URLs from memory
            if (n_tables + n_docs) > 0:
                total_files = len(self._upload_tabs) + len(self._downloaded_media_controls)
                self._message_placeholder.param.update(
                    object=f"Successfully processed {total_files} files ({n_tables} table(s), {n_docs} document(s)).",
                    visible=True,
                )
            self._error_placeholder.object = self._error_placeholder.object.strip()

        self._count += 1

    def __panel__(self):
        return self.menu


class RetryControls(Viewer):

    active = param.Boolean(False, doc="Click to retry")

    reason = param.String(doc="Reason for retry")

    def __init__(self, **params):
        super().__init__(**params)
        icon = ToggleIcon.from_param(
            self.param.active,
            name=" ",
            description="Prompt LLM to retry",
            icon="repeat-once",
            active_icon="x",
            margin=5,
        )
        self._text_input = TextInput(
            placeholder="Enter feedback and press the <Enter> to retry.",
            visible=icon.param.value,
            max_length=200,
            margin=(5, 0),
        )
        row = Row(icon, self._text_input)
        self._row = row

        self._text_input.param.watch(self._enter_reason, "enter_pressed")

    def _enter_reason(self, _):
        self.param.update(
            reason=self._text_input.value,
            active=False,
        )

    def __panel__(self):
        return self._row
