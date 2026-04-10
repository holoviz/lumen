from __future__ import annotations

import asyncio

import param

from panel_material_ui import (
    Button, Column as MuiColumn, Row, TextAreaInput,
)

from .base import FileSourceControls
from .utils import download_file


class DownloadSourceControls(FileSourceControls):
    """
    Controls for downloading files from URLs.
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

    load_mode = "manual"

    source_name_prefix = "DownloadedSource"

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

    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation."""
        from urllib.parse import urlparse
        try:
            result = urlparse(url.strip())
            return all([result.scheme, result.netloc])
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

        urls = [line.strip() for line in url_text.split('\n') if line.strip()]
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
            self.param.trigger('upload_successful')
