"""FastMCP server exposing the keyless lumen-mcp tools.

Run with ``lumen-mcp`` (stdio) after ``pip install -e .``, or register with a client, e.g.
``claude mcp add lumen-mcp -- lumen-mcp``.

Chart tools return the rendered PNG as an inline ``Image`` (so it displays in the chat and the model
can see it) alongside structured data (chart_id, normalized spec, saved paths). Hosts that do not
render images inline still get the paths.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent

from . import agentic, live, report, session_io, sources, viz
from .session import session

mcp = FastMCP(
    "lumen-mcp",
    instructions=(
        "Drive Lumen from any MCP client (keyless mode). You, the host LLM, write the SQL and the "
        "Vega-Lite spec; this server runs them through Lumen: a DuckDB workspace, spec "
        "normalization, rendering, and report export. Workflow: connect_source -> describe_table "
        "-> run_sql (each result becomes a named table) -> render_vegalite(spec, table) -> "
        "refine_chart -> build_report. Bind charts to the table names returned by run_sql, and "
        "omit 'data' from the spec (the server injects the table's data)."
    ),
)


def _safe(fn):
    """Return tool errors to the model instead of crashing the server."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    return wrapper


def _tool_result(structured: dict, text: str, image_paths=()) -> ToolResult:
    """Build a ToolResult with inline PNG previews (for paths that exist) plus a text summary."""
    content = [Image(path=path, format="png").to_image_content()
               for path in image_paths if path and os.path.exists(path)]
    content.append(TextContent(type="text", text=text))
    return ToolResult(content=content, structured_content=structured)


def _chart_result(result: dict) -> ToolResult:
    """Wrap a render result as an inline PNG preview plus structured data."""
    text = (f"Rendered chart '{result['chart_id']}' on table '{result['table']}'. "
            f"Interactive HTML saved to {result.get('html_path')}.")
    return _tool_result(result, text, [result.get("png_path")])


def render_vegalite(spec: dict, table: str, chart_id: Optional[str] = None) -> ToolResult:
    """Render a Vega-Lite spec bound to a workspace table.

    Omit 'data' from the spec; the server injects the table's data. Returns an inline PNG preview
    plus the chart_id, normalized spec, and saved PNG/HTML paths.
    """
    return _chart_result(viz.render_vegalite(spec, table, chart_id))


def refine_chart(chart_id: str, spec_patch: dict) -> ToolResult:
    """Deep-merge a spec patch into an existing chart and re-render under the same id.

    Returns an inline PNG preview plus the updated structured result.
    """
    return _chart_result(viz.refine_chart(chart_id, spec_patch))


def get_chart(chart_id: str) -> ToolResult:
    """Fetch an existing chart by id: an inline PNG preview plus spec, saved paths, and ui_uri."""
    return _chart_result(viz.get_chart(chart_id))


def view(target: str) -> ToolResult:
    """Show a chart or a saved file inline. ``target`` is a chart_id, or a path to a .png/.html file.

    A chart_id or an image file is returned as an inline image; an HTML file is returned as a path to
    open in a browser (rasterizing HTML to an inline image would need a headless browser).
    """
    if target in session.charts:
        return _chart_result(viz.get_chart(target))
    if os.path.isfile(target):
        ext = os.path.splitext(target)[1].lower()
        if ext == ".png":
            return _tool_result({"path": target, "kind": "image"}, target, [target])
        note = (f"HTML at {target} - open it in a browser for the interactive view."
                if ext in (".html", ".htm") else f"File: {target}")
        return _tool_result({"path": target, "kind": ext.lstrip(".") or "file"}, note)
    raise ValueError(f"Unknown chart id or file path: {target!r}")


def build_report(items: list, title: str = "Report", formats: Optional[list] = None) -> ToolResult:
    """Assemble charts + markdown into a downloadable HTML and .ipynb.

    Each item is ``{"markdown": "..."}`` or ``{"chart": "<chart_id>"}``. Returns inline previews of the
    report's charts plus the saved file paths.
    """
    result = report.build_report(items, title=title, formats=formats)
    pngs = [session.charts.get(item["chart"], {}).get("png_path")
            for item in items[:8] if isinstance(item, dict) and item.get("chart")]
    saved = ", ".join(f"{key.replace('_path', '')}: {value}"
                      for key, value in result.items() if key.endswith("_path"))
    return _tool_result(result, f"Report '{result.get('title')}' saved. {saved}", pngs)


def launch_dashboard() -> ToolResult:
    """Start a live interactive Lumen dashboard server (charts + sortable tables).

    Returns inline PNG previews of the charts plus a localhost URL. Open the URL in a browser for the
    full interactive experience (filter/sort tables, pan/zoom charts); the previews are static.
    """
    result = live.launch_dashboard()
    pngs = [session.charts.get(cid, {}).get("png_path") for cid in result.get("charts", [])[:4]]
    status = "ready" if result.get("ready") else "starting"
    text = (f"Live dashboard {status} at {result['url']} - open it in a browser to interact "
            f"(filter/sort tables, pan/zoom charts). Tables: {result.get('tables')}. "
            f"Chart previews are shown inline above.")
    return _tool_result(result, text, pngs)


def lumen_ask(prompt: str) -> ToolResult:
    """Answer a natural-language request by running Lumen's OWN agents (keyed mode).

    Lumen writes and runs the SQL and builds the chart itself over the current workspace. Returns the
    chart inline plus the generated SQL and a summary. Requires a connected source and a configured
    LLM key.
    """
    result = agentic.lumen_ask(prompt)
    lines = []
    if result.get("sql"):
        lines.append(f"SQL:\n{result['sql']}")
    if result.get("summary"):
        lines.append(result["summary"])
    if result.get("chart_id"):
        lines.append(f"Chart: {result['chart_id']} (table {result.get('table')}).")
    return _tool_result(result, "\n\n".join(lines) or "Done.", [result.get("png_path")])


def set_llm_key(api_key: str, provider: str = "openai", model: Optional[str] = None) -> dict:
    """Enable keyed mode at runtime by providing an LLM key, so lumen_ask can run Lumen's own agents.

    Security: prefer starting the server with the key in the environment (OPENAI_API_KEY or
    ANTHROPIC_API_KEY). Passing it through this tool sends it through the conversation, so rotate the
    key afterward. The in-chat setup pane (ui://lumen/setup) submits it without routing through the
    model on Apps-capable hosts.
    """
    resolved = agentic.set_key(api_key, provider=provider, model=model)
    return {"keyed": True, "provider": resolved, "model": os.environ.get("LUMEN_MCP_LLM_MODEL")}


# The plain-dict tools carry their own docstrings + type hints for the schema; the chart tools above
# add an inline image.
_TOOLS = [
    sources.connect_source,
    sources.list_tables,
    sources.describe_table,
    sources.run_sql,
    render_vegalite,
    refine_chart,
    get_chart,
    view,
    viz.list_charts,
    build_report,
    session_io.save_session,
    session_io.load_session,
    launch_dashboard,
    live.stop_dashboard,
]

for _fn in _TOOLS:
    mcp.tool()(_safe(_fn))


# Keyed mode: configure from the environment at startup, and always expose lumen_ask + set_llm_key so
# keyed mode can also be enabled at runtime. lumen_ask returns a helpful error until a key is set.
agentic.configure_llm()
for _keyed_fn in (lumen_ask, set_llm_key):
    mcp.tool()(_safe(_keyed_fn))


@mcp.resource(
    "ui://lumen/chart/{chart_id}",
    app=True,
    mime_type="text/html",
    name="Lumen chart viewer",
    description="Interactive, self-contained Vega chart for a rendered chart id.",
)
def chart_app(chart_id: str) -> str:
    entry = session.charts.get(chart_id)
    if entry is None:
        return f"<p>Unknown chart id: {chart_id}</p>"
    return entry["html"]


_APPS_DIR = os.path.join(os.path.dirname(__file__), "apps")


@mcp.resource(
    "ui://lumen/setup",
    app=True,
    mime_type="text/html",
    name="Enable keyed mode",
    description="Enter an LLM key to enable Lumen's own agents (keyed mode). Apps-capable hosts only.",
)
def setup_app() -> str:
    with open(os.path.join(_APPS_DIR, "setup.html")) as fh:
        return fh.read()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
