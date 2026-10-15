"""Keyed mode: lumen_ask runs Lumen's own agents (SQLAgent + VegaLiteAgent) headless.

Skipped unless an LLM key is configured (OPENAI_API_KEY or ANTHROPIC_API_KEY). Makes real LLM calls,
so it is slow. Run with:
    OPENAI_API_KEY=... python tests/test_keyed.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import make_sample_db  # noqa: E402

from lumen_mcp import agentic, sources  # noqa: E402


def test_keyed() -> None:
    provider = agentic.configure_llm()
    if not provider:
        print("SKIP: no LLM key configured (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        return

    db = os.path.join(tempfile.mkdtemp(), "sample.db")
    make_sample_db.main(db)
    sources.connect_source(db)

    from lumen_mcp import server  # imports after key is set -> lumen_ask registered

    result = server.lumen_ask("Show total sales by region as a bar chart")
    data = result.structured_content
    assert data.get("sql"), data
    images = [block for block in result.content if getattr(block, "type", None) == "image"]

    print("provider:", provider)
    print("lumen_ask sql:", " ".join(data["sql"].split())[:140])
    print("table:", data.get("table"), "| chart_id:", data.get("chart_id"),
          "| png:", bool(data.get("png_path")), "| inline image:", bool(images))
    assert data.get("chart_id"), "lumen_ask did not produce a chart"
    print("KEYED PASS")


if __name__ == "__main__":
    test_keyed()
