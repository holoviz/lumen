"""Delivery: turn a Lumen view into a PNG preview and a self-contained HTML file.

Both reuse existing machinery and need no server or browser:
- PNG via ``VegaLiteEditor.export("png")`` (vl-convert, already a Lumen dependency).
- HTML via Panel's ``.save(embed=True)`` (all JS/CSS inlined).
"""

from __future__ import annotations

import io
import os
from typing import Optional

import panel as pn

_OUT_DIR = os.environ.get(
    "LUMEN_MCP_OUT", os.path.join(os.path.expanduser("~"), ".lumen-mcp", "out")
)
_INITIALIZED = False


def _ensure_panel() -> None:
    global _INITIALIZED
    if not _INITIALIZED:
        pn.extension("vega", "tabulator")
        os.makedirs(_OUT_DIR, exist_ok=True)
        _INITIALIZED = True


def save_png(view, chart_id: str, scale: int = 2) -> Optional[str]:
    """Export a PNG for a VegaLiteView via vl-convert. Returns the path, or None if unavailable.

    Static export needs concrete sizes, so ``"container"`` width/height are replaced with defaults.
    """
    _ensure_panel()
    pane = view.get_panel()
    spec = dict(pane.object)
    if spec.get("width") in (None, "container"):
        spec["width"] = 800
    if spec.get("height") in (None, "container"):
        spec["height"] = 400
    pane.object = spec
    try:
        out = pane.export("png", scale=scale)
    except Exception:
        return None
    data = out.getvalue() if isinstance(out, io.BytesIO) else out
    path = os.path.join(_OUT_DIR, f"{chart_id}.png")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def save_html(view, chart_id: str, title: Optional[str] = None) -> tuple[str, str]:
    """Render a self-contained (offline) HTML page for a view; save it and return (path, html)."""
    _ensure_panel()
    buf = io.StringIO()
    pn.Column(view.get_panel()).save(buf, embed=True, title=title or chart_id)
    html = buf.getvalue()
    path = os.path.join(_OUT_DIR, f"{chart_id}.html")
    with open(path, "w") as fh:
        fh.write(html)
    return path, html
