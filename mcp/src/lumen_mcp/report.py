"""Report tool: assemble charts and markdown into a downloadable HTML and/or notebook.

Reuses existing machinery with no Lumen change: Panel ``.save(embed=True)`` for offline HTML and
``lumen.ai.export`` (``format_output`` / ``make_preamble`` / ``write_notebook``) for a reproducible
notebook (each chart serializes to ``lm.View.from_spec(...)``).
"""

from __future__ import annotations

import io
import os
from typing import Optional

import panel as pn
from lumen.ai.export import format_output, make_md_cell, make_preamble, write_notebook

from . import rendering
from .session import session


def _slug(title: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-") or "report"


def build_report(
    items: list,
    title: str = "Report",
    formats: Optional[list] = None,
) -> dict:
    """Build a downloadable report from ordered ``items`` and save it to disk.

    Each item is either ``{"markdown": "..."}`` or ``{"chart": "<chart_id>"}``. ``formats`` defaults
    to ``["html", "ipynb"]``. Returns the saved file paths.
    """
    formats = formats or ["html", "ipynb"]
    rendering._ensure_panel()

    panes: list = []
    cells: list = []
    extensions: list = []
    for item in items:
        if "markdown" in item:
            panes.append(pn.pane.Markdown(item["markdown"]))
            cells.append(make_md_cell(item["markdown"]))
        elif "chart" in item:
            chart_id = item["chart"]
            if chart_id not in session.charts:
                raise KeyError(f"Unknown chart_id {chart_id!r}. See list_charts.")
            view = session.charts[chart_id]["view"]
            panes.append(view.get_panel())
            cell, ext = format_output(view)
            if cell is not None:
                cells.append(cell)
                if ext and ext not in extensions:
                    extensions.append(ext)
        else:
            raise ValueError(f"Report item must have a 'markdown' or 'chart' key: {item!r}")

    slug = _slug(title)
    out: dict = {"title": title, "items": len(items)}

    if "html" in formats:
        buf = io.StringIO()
        pn.Column(*panes).save(buf, embed=True, title=title)
        path = os.path.join(rendering._OUT_DIR, f"{slug}.html")
        with open(path, "w") as fh:
            fh.write(buf.getvalue())
        out["html_path"] = path

    if "ipynb" in formats:
        notebook = write_notebook(make_preamble("", extensions=extensions or None, title=title) + cells)
        path = os.path.join(rendering._OUT_DIR, f"{slug}.ipynb")
        with open(path, "w") as fh:
            fh.write(notebook)
        out["ipynb_path"] = path

    return out
