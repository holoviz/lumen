"""Live dashboard server: build a session, launch it, confirm the URL serves, then stop.

Spawns a real ``panel serve`` subprocess, so it is slower than the other tests. Run with:
    python tests/test_dashboard.py
or under pytest.
"""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import make_sample_db  # noqa: E402

from lumen_mcp import live, server, sources, viz  # noqa: E402


def test_dashboard() -> None:
    db = os.path.join(tempfile.mkdtemp(), "sample.db")
    make_sample_db.main(db)
    sources.connect_source(db)
    sources.run_sql(
        "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC",
        name="by_region",
    )
    viz.render_vegalite(
        {"mark": "bar", "encoding": {"x": {"field": "region", "type": "nominal"},
                                     "y": {"field": "total", "type": "quantitative"}}},
        "by_region",
    )

    try:
        tool_result = server.launch_dashboard()
        data = tool_result.structured_content
        assert data["ready"], data
        assert "by_region" in data["tables"], data
        images = [b for b in tool_result.content if getattr(b, "type", None) == "image"]
        assert images, "launch_dashboard returned no inline chart preview"
        with urllib.request.urlopen(data["url"], timeout=5) as response:
            assert response.status == 200
        print("launch_dashboard ->", {k: data[k] for k in ("url", "ready", "charts", "tables")},
              "| inline previews:", len(images))
    finally:
        stopped = live.stop_dashboard()
        print("stop_dashboard   ->", stopped)
    print("DASHBOARD PASS")


if __name__ == "__main__":
    test_dashboard()
