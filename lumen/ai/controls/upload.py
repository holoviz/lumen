from __future__ import annotations

import param

from panel.widgets import FileDropper
from panel_material_ui import Column as MuiColumn

from .base import BaseSourceControls


class UploadControls(BaseSourceControls):
    """
    Controls for uploading files from the local filesystem.
    """

    load_mode = "manual"  # File selection triggers, not a button

    label = '<span class="material-icons" style="vertical-align: middle;">upload</span> Upload Data'

    def _render_layout(self):
        """Build upload-specific layout with FileDropper."""
        # Build FileDropper kwargs with defaults
        file_dropper_params = dict(
            layout="compact",
            multiple=self.param.multiple,
            margin=1,
            sizing_mode="stretch_width",
            disabled=self.param.disabled,
            stylesheets=[".bk-input.filepond--root { box-shadow: unset; cursor: grab; } .bk-input.filepond--root:not([disabled]):hover { box-shadow: unset; }"],
            visible=self._upload_cards.param.visible.rx.not_()
        )
        # Override with user-provided kwargs
        file_dropper_params.update(self.filedropper_kwargs)

        self._file_input = FileDropper(**file_dropper_params)
        self._file_input.param.watch(self._on_file_upload, "value")

        return MuiColumn(
            self._file_input,
            self._upload_cards,
            self._add_button,
            self._error_placeholder,
            self._message_placeholder,
            self.progress.bar,
            self.progress.description,
            sizing_mode="stretch_width",
        )

    def _on_file_upload(self, event):
        """Handle file upload from FileDropper."""
        self._generate_file_cards(self._file_input.value or {})

    @param.depends("add", watch=True)
    def _on_add(self):
        """Process uploaded files."""
        if len(self._file_cards) == 0:
            return

        with self._layout.param.update(loading=True):
            n_tables, n_docs, n_metadata = self._process_files()

            total_files = len(self._file_cards)
            n_successful = n_tables + n_docs + n_metadata

            # Always clear on completion to reset UI
            if self.clear_uploads:
                self._clear_uploads()
            self._file_input.value = {}

            if n_successful > 0:
                self._message_placeholder.param.update(
                    object=f"Successfully processed {total_files} files ({n_tables} table(s), {n_metadata} metadata file(s)).",
                    visible=True,
                )

            self._error_placeholder.object = self._error_placeholder.object.strip()

        self._count += 1

        if (n_tables + n_docs + n_metadata) > 0:
            self.param.trigger('upload_successful')
