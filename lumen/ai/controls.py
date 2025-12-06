from __future__ import annotations

import asyncio
import io
import pathlib
import re
import traceback
import zipfile

from typing import TYPE_CHECKING
from urllib.parse import urlparse

import aiohttp
import pandas as pd
import param

from panel.io import state
from panel.layout import Column, HSpacer, Row
from panel.pane.markup import HTML, Markdown
from panel.viewable import Viewer
from panel.widgets import FileDropper, Tabulator, Tqdm
from panel_material_ui import (
    Button, Card, ChatAreaInput, CheckBoxGroup, Column as MuiColumn,
    IconButton, Popup, Select, Switch, Tabs, TextInput, ToggleIcon, Typography,
)

from ..config import load_yaml
from ..pipeline import Pipeline
from ..sources.duckdb import DuckDBSource
from ..util import detect_file_encoding
from .config import SOURCE_TABLE_SEPARATOR
from .context import TContext
from .utils import generate_diff

if TYPE_CHECKING:
    from .views import SQLOutput

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
        self.box = Row(
            self._name_input,
            self._metadata_input,
        )


class TableControls(MediaControls):

    sheet = param.Selector(default=None, objects=[], doc="Sheet")

    def __init__(self, file_obj: io.BytesIO, **params):
        super().__init__(file_obj, **params)
        self._name_input = TextInput.from_param(
            self.param.alias, name="Table alias", margin=(15, 10, 0, 10)
        )
        self._sheet_select = Select.from_param(
            self.param.sheet, name="Sheet", visible=False
        )
        self.box = Row(
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

    active = param.Integer(default=0, doc="Active source index")

    cancellable = param.Boolean(default=False, doc="Show cancel button")

    clear_uploads = param.Boolean(default=True, doc="Clear uploaded file tabs")

    context = param.Dict(default={})

    disabled = param.Boolean(default=False, doc="Disable controls")

    downloaded_files = param.Dict(default={}, doc="Downloaded files to add as tabs")

    download_url = param.String(default="", doc="URL input for downloading files")

    input_placeholder = param.String(default="Enter URLs, one per line, and press <Enter> to download")

    add = param.Event(doc="Use uploaded file(s)")

    cancel = param.Event(doc="Cancel")

    multiple = param.Boolean(default=True, doc="Allow multiple files")

    show_input = param.Boolean(default=True, doc="Whether to show the input controls")

    replace_controls = param.Boolean(default=False, doc="Replace controls on add")

    outputs = param.Dict(default={})

    table_upload_callbacks = {}

    _active_download_task = param.ClassSelector(class_=asyncio.Task)

    _last_table = param.String(default="", doc="Last table added")

    _count = param.Integer(default=0, doc="Count of sources added")

    def __init__(self, **params):
        super().__init__(**params)
        self.tables_tabs = Tabs(sizing_mode="stretch_width")
        self._markitdown = None
        self._file_input = FileDropper(
            layout="compact",
            multiple=self.param.multiple,
            margin=0,
            sizing_mode="stretch_width",
            disabled=self.param.disabled,
            # accepted_filetypes=[".csv", ".parquet", ".parq", ".json", ".xlsx"],
        )
        self._file_input.param.watch(self._generate_media_controls, "value")
        self._upload_tabs = Tabs(sizing_mode="stretch_width", closable=True)
        files_to_process = self._upload_tabs.param["objects"].rx.len() > 0
        self._upload_tabs.visible = files_to_process

        # URL input for downloading files
        self._url_input = ChatAreaInput.from_param(
            self.param.download_url,
            placeholder=self.param.input_placeholder,
            rows=4,
            margin=(10, 10, 0, 10),
            sizing_mode="stretch_width",
            disabled=self.param.disabled,
            enable_upload=False
        )
        self._url_input.param.watch(self._handle_urls, "enter_pressed")

        self._input_tabs = Tabs(
            ("Upload Files", self._file_input),
            ("Download from URL", self._url_input),
            sizing_mode="stretch_width",
            dynamic=True,
            active=self.param.active
        )

        self._add_button = Button.from_param(
            self.param.add,
            name="Use file(s)",
            icon="table-plus",
            visible=files_to_process,
            button_type="success"
        )

        self._cancel_button = Button.from_param(
            self.param.cancel,
            name="Cancel",
            icon="circle-x",
            on_click=self._handle_cancel,
            visible=self.param._active_download_task.rx.is_not(None)
        )

        self._error_placeholder = HTML("", visible=False, margin=(0, 10, 5, 10))
        self._message_placeholder = HTML("", visible=False, margin=(0, 10, 10, 10))

        # Progress bar for downloads
        self._progress_bar = Tqdm(
            visible=False,
            margin=(0, 10, 0, 10),
            sizing_mode="stretch_width"
        )

        self.menu = MuiColumn(
            *((self._input_tabs,) if self.show_input else ()),
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
        self._add_downloaded_files_as_tabs()

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
            return all([result.scheme, result.netloc])
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

    def _create_file_object(self, file_data: bytes | io.BytesIO | io.StringIO, suffix: str):
        """Create appropriate file object based on file type"""
        if isinstance(file_data, (io.BytesIO, io.StringIO)):
            return file_data

        if suffix == "csv" and isinstance(file_data, bytes):
            encoding = detect_file_encoding(file_data)
            file_data = file_data.decode(encoding).encode("utf-8")
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

        except (TimeoutError, aiohttp.ClientError, asyncio.CancelledError, Exception) as e:
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
        self.downloaded_files = downloaded_files

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

    @param.depends("downloaded_files", watch=True)
    def _add_downloaded_files_as_tabs(self):
        """Add downloaded files as individual tabs in the main input tabs"""
        for filename, file_data in self.downloaded_files.items():
            suffix = pathlib.Path(filename).suffix.lstrip(".").lower()

            # Create file object using helper
            file_obj = self._create_file_object(file_data, suffix)

            # Create media controls using helper
            media_controls = self._create_media_controls(file_obj, filename)

            # Add as a new tab
            display_name = pathlib.Path(filename).stem
            self._upload_tabs.append((display_name, media_controls))
            self._downloaded_media_controls.append(media_controls)

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
            df = pd.read_csv(file, parse_dates=True, sep=None, engine='python')
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
        self.outputs["source"] = duckdb_source
        if "sources" not in self.outputs:
            self.outputs["sources"] = [duckdb_source]
        else:
            self.outputs["sources"].append(duckdb_source)
        self.outputs["table"] = table
        self.param.trigger('outputs')
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
        if "document_sources" in self.outputs:
            for i, source in enumerate(self.outputs["document_sources"]):
                if source.get("metadata", {})["filename"] == metadata["filename"]:
                    self.outputs["document_sources"][i] = document
                    break
            else:
                self.outputs["document_sources"].append(document)
        else:
            self.outputs["document_sources"] = [document]
        self.param.trigger('outputs')
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
                    source = table_upload_callbacks[media_controls.extension](
                        self.context, media_controls.file_obj, media_controls.alias, media_controls.filename
                    )
                    if source is not None:
                        n_tables += len(source.get_tables())
                        self.outputs["source"] = source
                        if "sources" not in self.outputs:
                            self.outputs["sources"] = [source]
                        else:
                            self.outputs["sources"].append(source)
                        self.param.trigger("outputs")
                elif media_controls.extension.endswith(TABLE_EXTENSIONS):
                    if source is None:
                        source_id = f"UploadedSource{self._count:06d}"
                        source = DuckDBSource(uri=":memory:", ephemeral=True, name=source_id, tables={})
                    table_name = media_controls.alias
                    filename = f"{media_controls.filename}.{media_controls.extension}"
                    if table_name not in source.metadata:
                        source.metadata[table_name] = {}
                    source.metadata[table_name]["filename"] = filename
                    n_tables += self._add_table(source, media_controls.file_obj, media_controls)
                else:
                    n_docs += self._add_document(media_controls.file_obj, media_controls)

            if self.replace_controls:
                src = self.outputs.get("source")
                if src:
                    self.tables_tabs[:] = [
                        (t, Tabulator(src.get(t), sizing_mode="stretch_width", pagination="remote"))
                        for t in src.get_tables()
                    ]
                self.menu[0].visible = False
                self._add_button.visible = False
                self._cancel_button.visible = False
                if n_tables == 0:
                    self.tables_tabs.visible = False
                    self.menu.height = 70

            total_files = len(self._upload_tabs) + len(self._downloaded_media_controls)
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

            # Clear uploaded files and URLs from memory
            if (n_tables + n_docs) > 0:
                self._message_placeholder.param.update(
                    object=f"Successfully processed {total_files} files ({n_tables} table(s), {n_docs} document(s)).",
                    visible=True,
                )
            self._error_placeholder.object = self._error_placeholder.object.strip()

        self._count += 1

    def __panel__(self):
        return self.menu


class RevisionControls(Viewer):

    active = param.Boolean(False, doc="Click to revise")

    instruction = param.String(doc="Instruction to LLM to revise output")

    interface = param.Parameter()

    layout_kwargs = param.Dict(default={})

    task = param.Parameter()

    view = param.Parameter(doc="The View to revise")

    toggle_kwargs = {}

    input_kwargs = {}

    def __init__(self, **params):
        super().__init__(**params)
        input_kwargs = dict(
            max_length=200,
            margin=10,
            size="small"
        )
        input_kwargs.update(self.input_kwargs)
        self._text_input = TextInput(**input_kwargs)
        popup = Popup(
            self._text_input,
            open=self.param.active,
            anchor_origin={"horizontal": "right", "vertical": "center"},
            styles={"z-index": '1300'}
        )
        icon = ToggleIcon.from_param(
            self.param.active,
            active_icon="cancel",
            attached=[popup],
            icon_size="1.1em",
            label="",
            margin=(5, 0),
            size="small",
            **self.toggle_kwargs
        )
        popup.param.watch(self._close, "open")
        self._text_input.param.watch(self._enter_reason, "enter_pressed")
        self._row = Row(icon, **self.layout_kwargs)

    @param.depends("active", watch=True)
    def _open(self):
        if self.active:
            state.execute(self._text_input.focus, schedule=True)

    def _close(self, event):
        self.active = event.new

    def _enter_reason(self, _):
        instruction = self._text_input.value_input
        self._text_input.value = ""
        self.param.update(active=False, instruction=instruction)

    def __panel__(self):
        return self._row


class RetryControls(RevisionControls):

    instruction = param.String(doc="Reason for retry")

    input_kwargs = {"placeholder": "Enter feedback and press the <Enter> to retry."}

    toggle_kwargs = {"icon": "auto_awesome", "description": "Prompt LLM to retry"}

    @param.depends("instruction", watch=True)
    async def _revise(self):
        if not self.instruction:
            return
        try:
            await self.task.revise(
                self.instruction, self.task.out_context, self.view, {'interface': self.interface}
            )
        except Exception as e:
            traceback.print_exc()
            self.interface.stream(f"**Error during revision:**\n```\n{e}\n```", user="Assistant")



class AnnotationControls(RevisionControls):
    """Controls for adding annotations to visualizations."""

    input_kwargs = {
        "placeholder": "Describe what to annotate (e.g., 'highlight peak values', 'mark outliers')..."
    }

    toggle_kwargs = {
        "description": "Add annotations to highlight key insights",
        "icon": "chat-bubble",
    }

    @param.depends("instruction", watch=True)
    async def _annotate(self):
        if not self.instruction:
            return
        try:
            old_spec = self.view.spec
            with self.view.editor.param.update(loading=True):
                new_spec = await self.task.actor.annotate(
                    self.instruction, list(self.task.history), self.task.out_context, {"spec": load_yaml(self.view.spec)}
                )
                self.view.spec = new_spec

            # Generate and stream diff to show changes made
            diff = generate_diff(old_spec, new_spec, filename="vega-lite")
            if diff:
                self.interface.stream(f"**Annotations applied:**\n```diff\n{diff}```", user="Assistant")
        except Exception as e:
            traceback.print_exc()
            self.view.spec = old_spec
            self.interface.stream(f"**Error during annotation:**\n```\n{e}\n```", user="Assistant")


class TableSourceCard(Viewer):
    """
    A component that displays a single data source as a card with table selection controls.

    The card includes:
    - A header with the source name and a checkbox to toggle all tables
    - A delete button (if multiple sources exist)
    - Individual checkboxes for each table in the source
    - Metadata display showing source information like filenames and other key-value pairs
    """

    all_selected = param.Boolean(default=True, doc="""
        Whether all tables should be selected by default.""")

    collapsed = param.Boolean(default=False, doc="""
        Whether the card should start collapsed.""")

    context = param.Dict()

    deletable = param.Boolean(default=True, doc="""
        Whether to show the delete button.""")

    delete = param.Event(doc="""Action to delete this source from memory.""")

    selected = param.List(default=None, doc="""
        List of currently selected table names.""")

    source = param.Parameter(doc="""
        The data source to display in this card.""")

    def __init__(self, **params):
        super().__init__(**params)
        self.all_tables = self.source.get_tables()

        # Determine which tables are currently visible
        if self.selected is None:
            visible_tables = []
            for table in self.all_tables:
                visible_tables.append(table)
            self.selected = visible_tables

        # Create widgets once in init
        self.source_toggle = Switch.from_param(
            self.param.all_selected,
            name=f"{self.source.name}",
            margin=(5, -5, 0, 3),
            sizing_mode='fixed',
        )

        self.delete_button = IconButton.from_param(
            self.param.delete,
            icon='delete',
            icon_size='1em',
            color="danger",
            margin=(5, 0, 0, 0),
            sizing_mode='fixed',
            width=40,
            height=40,
            visible=self.param.deletable
        )

        # Create table checkboxes with metadata
        self.table_controls = self._create_table_controls()

        # Create source-level metadata display (if any non-table metadata exists)
        self.metadata_display = self._create_source_metadata_display()

    def _create_table_controls(self):
        """Create table checkboxes with per-table metadata displayed next to each checkbox."""
        table_controls = []

        for table in self.all_tables:
            # Create checkbox for this table
            checkbox = CheckBoxGroup(
                value=[table] if table in self.selected else [],
                options=[table],
                sizing_mode='stretch_width',
                margin=(2, 10),
                name="",
            )

            # Get metadata for this table
            table_metadata = self.source.metadata.get(table, {}) if self.source.metadata else {}
            metadata_parts = []

            for key, value in table_metadata.items():
                if isinstance(value, list):
                    value_str = ', '.join(str(v) for v in value)
                else:
                    value_str = str(value)
                metadata_parts.append(f"{key}: {value_str}")

            if metadata_parts:
                metadata_text = '; '.join(metadata_parts)
                metadata_display = Typography(
                    metadata_text,
                    variant="caption",
                    color="text.secondary",
                    margin=(-10, 10, 0, 42),
                    sizing_mode='stretch_width',
                    styles={"min-height": "unset"} # Override ChatMessage CSS
                )
                table_controls.extend([checkbox, metadata_display])
            else:
                table_controls.append(checkbox)

            # Watch checkbox changes
            checkbox.param.watch(self._on_table_checkbox_change, 'value')

        return Column(*table_controls, margin=0, sizing_mode='stretch_width')

    def _on_table_checkbox_change(self, event):
        """Handle individual table checkbox changes."""
        # Collect all selected tables from all checkboxes
        selected_tables = []
        for obj in self.table_controls.objects:
            if obj.value:
                selected_tables.extend(obj.value)

        # Update selected parameter
        self.selected = selected_tables

    def _create_source_metadata_display(self):
        """Create a metadata display widget for source-level metadata (non-table metadata)."""
        metadata_parts = []

        if self.source.metadata:
            # Only show metadata that's not table-specific
            for key, value in self.source.metadata.items():
                if key not in self.all_tables:  # Skip table-specific metadata
                    if isinstance(value, list):
                        value_str = ', '.join(str(v) for v in value)
                    else:
                        value_str = str(value)
                    metadata_parts.append(f"{key}: {value_str}")

        if metadata_parts:
            metadata_text = '; '.join(metadata_parts)
            return Typography(
                metadata_text,
                variant="caption",
                color="text.secondary",
                margin=(0, 10, 5, 10),
                sizing_mode='stretch_width',
            )
        else:
            return Typography(
                "",
                margin=0,
                sizing_mode='stretch_width',
                visible=False
            )

    @param.depends('all_selected', watch=True)
    def _on_source_toggle(self):
        """Handle source checkbox toggle (all tables on/off)."""
        if not self.all_selected and len(self.selected) == len(self.all_tables):
            # Important to check to see if all tables are selected for intuitive behavior
            self.selected = []
        elif self.all_selected:
            self.selected = self.all_tables

    @param.depends('selected', watch=True)
    def _update_visible_slugs(self):
        """Update visible_slugs in memory based on selected tables."""
        self.all_selected = len(self.selected) == len(self.all_tables)
        for table in self.all_tables:
            table_slug = f"{self.source.name}{SOURCE_TABLE_SEPARATOR}{table}"
            if table in self.selected:
                self.context['visible_slugs'].add(table_slug)
            else:
                self.context['visible_slugs'].discard(table_slug)

    @param.depends('delete', watch=True)
    def _delete_source(self):
        """Handle source deletion via param.Action."""
        if self.source in self.context.get("sources", []):
            # Remove all tables from this source from visible_slugs
            for table in self.all_tables:
                table_slug = f"{self.source.name}{SOURCE_TABLE_SEPARATOR}{table}"
                self.context['visible_slugs'].discard(table_slug)

            self.context["sources"] = [
                source for source in self.context.get("sources", [])
                if source is not self.source
            ]

    def __panel__(self):
        card_header = Row(
            self.source_toggle,
            HSpacer(),
            self.delete_button,
            sizing_mode='stretch_width',
            align='start',
            height=35,
            margin=0
        )

        # Create the card content with metadata display
        card_content = Column(
            self.metadata_display,
            self.table_controls,
            margin=0,
            sizing_mode='stretch_width'
        )

        # Create the card
        return Card(
            card_content,
            header=card_header,
            collapsible=True,
            collapsed=self.param.collapsed,
            sizing_mode='stretch_width',
            margin=0,
            name="TableSourceCard"
        )


class SourceCatalog(Viewer):
    """
    A component that displays all data sources with table selection controls.

    This component shows each source as a collapsible card with:
    - A header checkbox to toggle all tables in the source
    - Individual checkboxes for each table
    - A delete button to remove the source (if multiple sources exist)

    Tables can be selectively shown/hidden using the checkboxes, which updates
    the 'visible_slugs' set in memory.
    """

    context = param.Dict(default={})

    sources = param.List(default=[], doc="""
        List of data sources to display in the catalog.""")

    def __init__(self, /, context: TContext | None = None, **params):
        self._title = Markdown(margin=(0, 10))
        self._cards_column = Column()
        self._layout = Column(
            self._title,
            self._cards_column,
            sizing_mode='stretch_width'
        )
        if context is None:
            raise ValueError("SourceCatalog must be given a context dictionary.")
        if "source" in context and "sources" not in context:
            context["sources"] = [context["source"]]
        if "visible_slugs" not in context:
            context["visible_slugs"] = set()
        super().__init__(context=context, **params)

    @param.depends("sources", watch=True, on_init=True)
    async def sync(self, context: TContext | None = None):
        """
        Trigger the catalog with new sources.

        Args:
            sources: Optional list of sources. If None, uses sources from memory.
        """
        context = context or self.context
        sources = self.sources or context.get('sources', [])

        # Create a lookup of existing cards by source
        existing_cards = {
            card.source: card for card in self._cards_column.objects
            if isinstance(card, TableSourceCard) and card.source in sources
        }

        # Build the new cards list
        source_cards = []
        multiple_sources = len(sources) > 1
        for source in sources:
            if source in existing_cards:
                # Reuse existing card and update its deletable property
                card = existing_cards[source]
                card.deletable = multiple_sources
                source_cards.append(card)
            else:
                # Create new card for new source
                source_card = TableSourceCard(
                    context=context,
                    source=source,
                    deletable=multiple_sources,
                    collapsed=multiple_sources,
                )
                source_cards.append(source_card)

        self._cards_column.objects = source_cards

        if len(sources) == 0:
            self._title.object = "No sources available. Add a source to get started."
        else:
            self._title.object = "Select the table and document sources you want visible to the LLM."

    def __panel__(self):
        """
        Create the source catalog UI.

        Returns a Column containing all source cards or a message if no sources exist.
        """
        return self._layout


class TableExplorer(Viewer):
    """
    TableExplorer provides a high-level entrypoint to explore tables in a split UI.
    It allows users to load tables, explore them using Graphic Walker, and then
    interrogate the data via a chat interface.
    """

    add_exploration = param.Event(label='Explore table(s)')

    table_slug = param.Selector(label="Select table(s) to preview")

    context = param.Dict(default={})

    def __init__(self, **params):
        self._initialized = False
        super().__init__(**params)
        self._table_select = Select.from_param(
            self.param.table_slug, sizing_mode='stretch_width',
            max_height=200, margin=0
        )
        self._explore_button = Button.from_param(
            self.param.add_exploration,
            icon='add_chart', color='primary', icon_size="2em",
            disabled=self._table_select.param.value.rx().rx.not_(),
            margin=(0, 0, 0, 10), width=200, align='end'
        )
        self._input_row = Row(self._table_select, self._explore_button, margin=(0, 10, 0, 10))
        self.source_map = {}
        self._layout = self._input_row

    @param.depends("context", watch=True)
    async def sync(self, context: TContext | None = None):
        init = not self._initialized
        self._initialized = True
        context = context or self.context
        if "sources" in context:
            sources = context["sources"]
        elif "source" in context:
            sources = [context["source"]]
        else:
            return
        selected = [self.table_slug] if self.table_slug else []
        deduplicate = len(sources) > 1
        new = {}

        # Build the source map for UI display
        for source in sources:
            tables = source.get_tables()
            for table in tables:
                if deduplicate:
                    table = f'{source.name}{SOURCE_TABLE_SEPARATOR}{table}'

                if (table.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)[-1] not in self.source_map and
                    not init and state.loaded):
                    selected.append(table)
                new[table] = source

        self.source_map.clear()
        self.source_map.update(new)
        selected = selected[-1] if len(selected) == 1 else None
        self._table_select.param.update(options=list(self.source_map), value=selected)
        self._input_row.visible = bool(self.source_map)
        self._initialized = True

    def create_sql_output(self) -> SQLOutput | None:
        if not self.table_slug:
            return
        from .views import SQLOutput

        source = self.source_map[self.table_slug]
        if SOURCE_TABLE_SEPARATOR in self.table_slug:
            _, table = self.table_slug.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)
        else:
            table = self.table_slug
        new_table = f"select_{table}"
        sql_expr = f"SELECT * FROM \"{table}\""
        new_source = source.create_sql_expr_source({new_table: sql_expr})
        pipeline = Pipeline(source=new_source, table=new_table)
        return SQLOutput(spec=sql_expr, component=pipeline)

    def _explore_table_if_single(self, event):
        """
        If only one table is uploaded, help the user load it
        without requiring them to click twice. This step
        only triggers when the Upload in the Overview tab is used,
        i.e. does not trigger with uploads through the SourceAgent
        """
        if len(self._table_select.options) == 1:
            self._explore_button.param.trigger("value")

    def __panel__(self):
        return self._layout
