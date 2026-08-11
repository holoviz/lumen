"""Visualization tools: render, refine, and fetch host-authored Vega-Lite charts.

Flow (all reuse, no LLM): normalize the spec (adds ``$schema`` etc.) -> bind the table via a Lumen
``Pipeline`` -> ``VegaLiteView`` -> PNG + self-contained HTML. Each chart's interactive HTML is kept
in the registry so it can be served as a ``ui://lumen/chart/{id}`` app resource.
"""

from __future__ import annotations

from typing import Optional

from lumen.pipeline import Pipeline
from lumen.views import VegaLiteView

from . import rendering
from ._shims import normalize_spec
from .session import session


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge ``patch`` into ``base``; patch wins on scalars and list replacement."""
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _result(chart_id: str, entry: dict) -> dict:
    return {
        "chart_id": chart_id,
        "table": entry["table"],
        "spec": entry["spec"],
        "png_path": entry["png_path"],
        "html_path": entry["html_path"],
        "ui_uri": f"ui://lumen/chart/{chart_id}",
    }


def render_vegalite(spec: dict, table: str, chart_id: Optional[str] = None) -> dict:
    """Render a Vega-Lite ``spec`` bound to workspace ``table``.

    The spec should reference field names but omit ``data`` (the server injects the table's data).
    Returns the normalized spec, a ``chart_id``, a ``ui_uri``, and paths to a PNG preview and a
    self-contained HTML file.
    """
    source = session.require_source()
    rendering._ensure_panel()

    normalized = normalize_spec(spec)["spec"]
    view = VegaLiteView(pipeline=Pipeline(source=source, table=table), spec=normalized)
    cid = chart_id or session.next_chart_id()

    png_path = rendering.save_png(view, cid)
    html_path, html = rendering.save_html(view, cid)
    session.charts[cid] = {
        "view": view,
        "spec": normalized,
        "table": table,
        "png_path": png_path,
        "html_path": html_path,
        "html": html,
    }
    return _result(cid, session.charts[cid])


def refine_chart(chart_id: str, spec_patch: dict) -> dict:
    """Deep-merge ``spec_patch`` into an existing chart's spec and re-render under the same id.

    Use this to iterate (add a color encoding, a title, a sort, a tooltip) without restating the
    whole spec.
    """
    if chart_id not in session.charts:
        raise KeyError(f"Unknown chart_id {chart_id!r}. Render a chart first, or see list_charts.")
    entry = session.charts[chart_id]
    return render_vegalite(_deep_merge(entry["spec"], spec_patch), entry["table"], chart_id=chart_id)


def get_chart(chart_id: str) -> dict:
    """Fetch an existing chart (spec, saved paths, and ui_uri) by id."""
    if chart_id not in session.charts:
        raise KeyError(f"Unknown chart_id {chart_id!r}. See list_charts.")
    return _result(chart_id, session.charts[chart_id])


def list_charts() -> dict:
    """List rendered charts (id and the table each is bound to)."""
    return {
        "charts": [
            {"chart_id": cid, "table": entry["table"]} for cid, entry in session.charts.items()
        ]
    }
