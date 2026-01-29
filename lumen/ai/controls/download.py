from __future__ import annotations

import asyncio
import io
import re

from urllib.parse import parse_qs, urlparse

import aiohttp
import param

from panel_material_ui import (
    Button, Column as MuiColumn, Row, TextAreaInput,
)

from .base import (
    METADATA_EXTENSIONS, TABLE_EXTENSIONS, BaseSourceControls, DownloadConfig,
)


class DownloadControls(BaseSourceControls):
    """
    Controls for downloading files from URLs.
    """

    download_url = param.String(default="", doc="Enter one or more URLs, one per line, and press <Shift+Enter> to download", label="Download URL(s)")

    input_placeholder = param.String(default="Enter URLs, one per line, and press <Shift+Enter> to download")

    _active_download_task = param.ClassSelector(class_=asyncio.Task)

    load_mode = "manual"  # URL entry triggers, not a button

    label = '<span class="material-icons" style="vertical-align: middle;">download</span> Fetch Remote Data'

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
        """Handle cancel button click by cancelling active download task"""
        if self._active_download_task and not self._active_download_task.done():
            self._active_download_task.cancel()
            self.progress.clear()
            self._message_placeholder.param.update(
                object="Download cancelled by user.",
                visible=True
            )
        # Clear the active task so Cancel button hides
        self._active_download_task = None

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

        if '?' in filename:
            filename = filename.split('?')[0]

        # Check if the extracted extension is a valid table/metadata extension
        current_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        valid_extensions = TABLE_EXTENSIONS + METADATA_EXTENSIONS

        # If extension is not valid, check for format= query parameter
        if current_ext not in valid_extensions:
            query_params = parse_qs(parsed.query)
            format_param = query_params.get('format', [None])[0]
            if format_param and format_param.lower() in valid_extensions:
                # Replace or add the correct extension
                base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
                filename = f"{base_name}.{format_param.lower()}"

        if not filename or '.' not in filename:
            filename = f"data_{abs(hash(url)) % DownloadConfig.DEFAULT_HASH_MODULO}.json"

        return filename

    def _extract_filename_from_headers(self, response_headers: dict, default_filename: str) -> str:
        """Extract filename from HTTP headers if available"""
        # First check Content-Disposition header
        if 'content-disposition' in response_headers:
            cd = response_headers['content-disposition']
            matches = re.findall('filename="?([^"]+)"?', cd)
            if matches:
                suggested_name = matches[0]
                if '.' in suggested_name:
                    return suggested_name

        # Check if current filename has invalid extension, try to fix from Content-Type
        current_ext = default_filename.rsplit('.', 1)[-1].lower() if '.' in default_filename else ''
        valid_extensions = TABLE_EXTENSIONS + METADATA_EXTENSIONS

        if current_ext not in valid_extensions and 'content-type' in response_headers:
            content_type = response_headers['content-type'].lower().split(';')[0].strip()
            content_type_map = {
                'text/csv': 'csv',
                'application/csv': 'csv',
                'application/json': 'json',
                'application/geo+json': 'geojson',
                'application/vnd.geo+json': 'geojson',
                'application/parquet': 'parquet',
                'application/vnd.apache.parquet': 'parquet',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'application/vnd.ms-excel': 'xlsx',
                'text/plain': 'csv',  # Often CSVs are served as text/plain
            }
            if content_type in content_type_map:
                ext = content_type_map[content_type]
                base_name = default_filename.rsplit('.', 1)[0] if '.' in default_filename else default_filename
                return f"{base_name}.{ext}"

        return default_filename

    def _setup_progress_bar(self, content_length: str) -> int:
        """Setup progress bar based on content length"""
        self.progress.description.visible = True
        if content_length:
            total_size = int(content_length)
            self.progress.bar.param.update(
                visible=True,
                value=0,
                variant="determinate"
            )
            return total_size
        else:
            self.progress.bar.variant = "indeterminate"
            self.progress.bar.visible = True
            return 0

    def _update_download_progress(self, filename: str, downloaded_size: int, total_size: int, chunk_count: int):
        """Update progress bar if it's time for an update"""
        if chunk_count % DownloadConfig.PROGRESS_UPDATE_INTERVAL == 0:
            if total_size > 0:
                progress = min((downloaded_size / total_size) * 100, 100)
                if progress < 100:
                    self.progress.bar.value = progress
                    progress_desc = f"Downloading {filename}: {self._format_bytes(downloaded_size)}/{self._format_bytes(total_size)}"
                else:
                    progress_desc = f"Downloading {filename}: {self._format_bytes(downloaded_size)}"
            else:
                progress_desc = f"Downloading {filename}: {self._format_bytes(downloaded_size)}"

            self.progress.description.object = progress_desc

    def _finalize_download_progress(self, filename: str, downloaded_size: int, total_size: int):
        """Show final progress update when download completes"""
        if total_size > 0:
            progress = min((downloaded_size / total_size) * 100, 100)
            self.progress.bar.value = progress
            self.progress.description.object = f"Downloaded {filename}: {self._format_bytes(downloaded_size)}/{self._format_bytes(total_size)}"
        else:
            self.progress.bar.value = 100
            self.progress.description.object = f"Downloaded {filename}: {self._format_bytes(downloaded_size)}"

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

    async def _download_file(self, url):
        """Download file from URL with progress bar and return (filename, file_content, error)"""
        try:
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

                    filename = self._extract_filename_from_headers(response.headers, filename)
                    content_length = response.headers.get('content-length')
                    total_size = self._setup_progress_bar(content_length)

                    downloaded_data = io.BytesIO()
                    downloaded_size = 0
                    chunk_count = 0
                    size_mismatch_detected = False

                    async for chunk in response.content.iter_chunked(DownloadConfig.CHUNK_SIZE):
                        if asyncio.current_task().cancelled():
                            raise asyncio.CancelledError()

                        downloaded_data.write(chunk)
                        downloaded_size += len(chunk)
                        chunk_count += 1

                        if not size_mismatch_detected and total_size > 0 and downloaded_size > total_size:
                            # Content-Length was inaccurate, switch to indeterminate mode
                            total_size = 0
                            size_mismatch_detected = True

                        self._update_download_progress(filename, downloaded_size, total_size, chunk_count)
                        await asyncio.sleep(0)

                    self._finalize_download_progress(filename, downloaded_size, total_size)

                    downloaded_data.seek(0)
                    content = downloaded_data.read()

                    self.progress.bar.visible = False
                    self.progress.description.visible = False
                    return filename, content, None

        except (TimeoutError, aiohttp.ClientError, asyncio.CancelledError, Exception) as e:
            self.progress.bar.visible = False
            self.progress.description.visible = False
            return None, None, self._format_download_error(e)

    def _handle_urls(self, event):
        """Handle URL input and trigger downloads"""
        url_text = self._url_input.value
        if not url_text:
            return

        urls = [line.strip() for line in url_text.split('\\n') if line.strip()]
        valid_urls = [url for url in urls if self._is_valid_url(url)]

        if not valid_urls:
            return

        self._error_placeholder.visible = False
        self._message_placeholder.param.update(
            object=f"Preparing to download {len(valid_urls)} file(s)...",
            visible=True
        )

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
        """Download URLs and generate file cards for processing"""
        downloaded_files = {}
        errors = []

        try:
            for i, url in enumerate(urls, 1):
                if asyncio.current_task().cancelled():
                    break

                self._message_placeholder.object = f"Processing {i} of {len(urls)}: {url[:50]}{'...' if len(url) > 50 else ''}"
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

        # Generate file cards from downloaded files
        self._generate_file_cards(downloaded_files)

        if errors:
            error_msg = "\\n".join(errors)
            self._error_placeholder.param.update(
                object=f"Download errors:\\n{error_msg}",
                visible=True
            )

        success_count = len(downloaded_files)
        if success_count > 0:
            self._message_placeholder.object = f"Successfully downloaded {success_count} file(s). Click 'Confirm file(s)' to process."
        else:
            self._message_placeholder.visible = False

        # Clear the active task so Cancel button hides
        self._active_download_task = None

    @param.depends("add", watch=True)
    def _on_add(self):
        """Process downloaded files."""
        if len(self._file_cards) == 0:
            return

        # Clear the active task so Cancel button hides
        self._active_download_task = None

        with self._layout.param.update(loading=True):
            n_tables, n_docs, n_metadata = self._process_files()

            total_files = len(self._file_cards)
            if self.clear_uploads:
                self._clear_uploads()
                self._url_input.value = ""

            if (n_tables + n_docs + n_metadata) > 0:
                self._message_placeholder.param.update(
                    object=f"Successfully processed {total_files} files ({n_tables} table(s), {n_metadata} metadata file(s)).",
                    visible=True,
                )
            self._error_placeholder.object = self._error_placeholder.object.strip()

        self._count += 1

        if (n_tables + n_docs + n_metadata) > 0:
            self.param.trigger('upload_successful')
