"""End-to-end keyless slice: connect -> describe -> run_sql -> render.

Exercises the tool *logic* directly (no MCP client, no LLM, no key). Run with:
    PYTHONPATH=src python tests/test_slice.py
or under pytest.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import make_sample_db  # noqa: E402
from lumen_mcp import report, sources, viz  # noqa: E402


def test_keyless_slice() -> None:
    db = os.path.join(tempfile.mkdtemp(), "sample.db")
    make_sample_db.main(db)

    connected = sources.connect_source(db)
    assert "sales" in connected["tables"], connected

    described = sources.describe_table("sales")
    assert "region" in described["columns"], described

    result = sources.run_sql(
        "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC",
        name="by_region",
    )
    assert result["table"] == "by_region", result
    assert result["row_count"] == 4, result

    spec = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "region", "type": "nominal", "sort": "-y"},
            "y": {"field": "total", "type": "quantitative"},
        },
    }
    chart = viz.render_vegalite(spec, "by_region")
    assert "$schema" in chart["spec"], "spec was not normalized"
    assert chart["html_path"] and os.path.exists(chart["html_path"]), chart

    refined = viz.refine_chart(
        chart["chart_id"], {"encoding": {"color": {"field": "region", "type": "nominal"}}}
    )
    assert refined["chart_id"] == chart["chart_id"], refined
    assert "color" in refined["spec"]["encoding"], refined["spec"]["encoding"]

    charts = viz.list_charts()
    assert any(c["chart_id"] == chart["chart_id"] for c in charts["charts"]), charts

    rep = report.build_report(
        [{"markdown": "# Sales report"}, {"chart": chart["chart_id"]}], title="Sales report"
    )
    assert os.path.exists(rep["html_path"]), rep
    assert os.path.exists(rep["ipynb_path"]), rep

    print("connect_source ->", connected)
    print("describe_table  ->", described["columns"], described["dtypes"])
    print("run_sql         ->", {k: result[k] for k in ("table", "row_count", "columns")})
    print("render_vegalite -> chart_id=%s png=%s html=%s" % (
        chart["chart_id"], chart["png_path"], chart["html_path"]))
    print("refine_chart    -> color encoding merged:", "color" in refined["spec"]["encoding"])
    print("build_report    ->", {k: rep.get(k) for k in ("html_path", "ipynb_path")})
    print("PASS")


if __name__ == "__main__":
    test_keyless_slice()
