"""Live MCP round-trip through FastMCP's in-memory client.

Exercises the real protocol path (list_tools + call_tool) rather than the tool functions directly.
Requires ``fastmcp``. Run with:
    python tests/test_roundtrip.py
or under pytest.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import make_sample_db  # noqa: E402
from fastmcp import Client  # noqa: E402

from lumen_mcp.server import mcp  # noqa: E402

_EXPECTED_TOOLS = {
    "connect_source", "list_tables", "describe_table", "run_sql",
    "render_vegalite", "refine_chart", "get_chart", "view", "list_charts", "build_report",
    "save_session", "load_session",
}


def _data(result):
    """Extract the structured return value from a FastMCP CallToolResult."""
    if getattr(result, "data", None) is not None:
        return result.data
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def _run() -> None:
    db = os.path.join(tempfile.mkdtemp(), "sample.db")
    make_sample_db.main(db)

    async with Client(mcp) as client:
        tools = {tool.name for tool in await client.list_tools()}
        assert _EXPECTED_TOOLS.issubset(tools), tools
        print("tools:", sorted(tools))

        connected = _data(await client.call_tool("connect_source", {"uri": db}))
        assert "sales" in connected["tables"], connected
        print("connect_source:", connected)

        result = _data(await client.call_tool(
            "run_sql",
            {"sql": "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC",
             "name": "by_region"},
        ))
        assert result["row_count"] == 4, result
        print("run_sql:", {key: result[key] for key in ("table", "row_count", "columns")})

        spec = {
            "mark": "bar",
            "encoding": {
                "x": {"field": "region", "type": "nominal", "sort": "-y"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }
        raw = await client.call_tool("render_vegalite", {"spec": spec, "table": "by_region"})
        chart = _data(raw)
        images = [block for block in raw.content if getattr(block, "type", None) == "image"]
        assert images, "render_vegalite did not return an inline image block"
        assert chart["html_path"] and os.path.exists(chart["html_path"]), chart
        assert chart["ui_uri"].startswith("ui://lumen/chart/"), chart
        print("render_vegalite:", {key: chart[key] for key in ("chart_id", "png_path", "html_path")},
              "| inline image:", bool(images))

        got = _data(await client.call_tool("get_chart", {"chart_id": chart["chart_id"]}))
        assert got["chart_id"] == chart["chart_id"] and got["ui_uri"] == chart["ui_uri"], got
        templates = [str(t) for t in await client.list_resource_templates()]
        assert any("ui://lumen/chart" in t for t in templates), templates
        print("get_chart:", got["ui_uri"], "| ui:// resource template registered:", True)

        rep_raw = await client.call_tool(
            "build_report",
            {"items": [{"markdown": "# Sales"}, {"chart": chart["chart_id"]}], "title": "Sales"},
        )
        rep = _data(rep_raw)
        rep_images = [block for block in rep_raw.content if getattr(block, "type", None) == "image"]
        assert os.path.exists(rep["html_path"]) and os.path.exists(rep["ipynb_path"]), rep
        assert rep_images, "build_report returned no inline preview"
        print("build_report:", {key: rep.get(key) for key in ("html_path", "ipynb_path")},
              "| inline previews:", len(rep_images))

        view_raw = await client.call_tool("view", {"target": chart["chart_id"]})
        view_images = [block for block in view_raw.content if getattr(block, "type", None) == "image"]
        assert view_images, "view(chart_id) returned no inline image"
        print("view:", "chart_id -> inline image OK")

        save_path = os.path.join(tempfile.mkdtemp(), "session")
        saved = _data(await client.call_tool("save_session", {"path": save_path}))
        assert "by_region" in saved["tables"], saved
        loaded = _data(await client.call_tool("load_session", {"path": saved["saved"]}))
        assert "by_region" in loaded["tables"], loaded
        assert chart["chart_id"] in loaded["charts"], loaded
        print("save/load session:", {"tables": loaded["tables"], "charts": loaded["charts"]})

    print("ROUND-TRIP PASS")


def test_roundtrip() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    test_roundtrip()
