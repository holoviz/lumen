from __future__ import annotations

import warnings

from pathlib import Path
from typing import Any

import param

from panel.custom import Child, JSComponent


class Details(JSComponent):
    """
    A component wrapper for the HTML details element with proper content rendering.

    This component provides an expandable details/summary element that can be
    used to hide/show content. It supports any Panel component as content,
    including Markdown panes for rendering markdown content.
    """

    title = param.String(default="Details", doc="""
        The text to display in the summary element (the clickable header).""")

    object = Child(doc="""
        The content to display within the details element. Can be any Panel component,
        including a Markdown pane for rendering markdown content.""")

    collapsed = param.Boolean(default=True, doc="""
        Whether the details element is collapsed by default.""")

    _esm = Path(__file__).parent / "models" / "details.js"

    def __init__(self, object: Any, **params):
        warnings.warn(
            "lumen.ai.Details is deprecated and will be removed in a future release. "
            "Please use panel_material_ui.Details instead.", DeprecationWarning, stacklevel=2
        )
        super().__init__(object=object, **params)

    @param.depends("collapsed", watch=True)
    def _send_collapsed_update(self):
        """Send message to JS when collapsed state changes in Python"""
        self._send_msg({"type": "update_collapsed", "collapsed": self.collapsed})

    def _handle_msg(self, msg):
        """Handle messages from JS"""
        if 'collapsed' in msg:
            collapsed = msg['collapsed']
            with param.discard_events(self):
                self.collapsed = collapsed
