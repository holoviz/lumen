from __future__ import annotations

import param

from panel.viewable import Viewer
from panel_material_ui import IconButton, Row


class CopyControls(Viewer):
    """Controls for copying spec to clipboard."""

    interface = param.Parameter()

    layout_kwargs = param.Dict(default={})

    task = param.Parameter()

    view = param.Parameter(doc="The View to copy")

    def __init__(self, **params):
        super().__init__(**params)
        copy_icon = IconButton(
            icon="content_copy",
            active_icon="check",
            margin=(5, 0),
            toggle_duration=1000,
            description="Copy YAML to clipboard",
            size="small",
            color="primary",
            icon_size="0.9em",
        )
        copy_icon.js_on_click(
            args={"code_editor": self.view.editor},
            code="navigator.clipboard.writeText(code_editor.code);",
        )
        self._row = Row(copy_icon, **self.layout_kwargs)

    def __panel__(self):
        return self._row
