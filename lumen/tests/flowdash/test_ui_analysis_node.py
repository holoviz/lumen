"""
Dragging a host analysis in from the component palette places a Lumen
analysis node whose tile shows what the analysis returns once a pipeline
feeds it. Runs where pytest-playwright is installed.
"""
import re

import panel as pn
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("instructor")
pytest.importorskip("pytest_playwright")

from panel.tests.util import serve_component, wait_until
from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps
from playwright.sync_api import expect

from lumen.ai.analysis import Analysis
from lumen.flowdash import (
    ANALYSES_KEY, ANALYSIS_ID, COMPONENTS, SOURCES_KEY, TABLE_ID,
)
from lumen.flowdash.analysis import analysis_presets
from lumen.flowdash.choices import palette_presets
from lumen.sources.duckdb import DuckDBSource

pytestmark = pytest.mark.skipif(
    not palette_presets(analysis_presets), reason="panel-flowdash without palette_presets"
)


class TotalCard(Analysis):
    """Show the total amount."""

    columns = ["amount"]

    def __call__(self, pipeline, context):
        return pn.pane.Markdown(f"Total amount: {pipeline.data['amount'].sum()}")


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


def test_an_analysis_dragged_from_the_palette_shows_its_view(page):
    source = DuckDBSource(uri=":memory:", tables={
        "sales": "SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, amount)",
    })
    editor = FlowDash(
        components=COMPONENTS, notifications=False,
        context={SOURCES_KEY: {"shop": source}, ANALYSES_KEY: [TotalCard]},
    )
    ops = DashboardOps(editor)
    ops.add_node(TABLE_ID, {"ref": "shop", "table": "sales"}, name="sales")
    page.set_viewport_size({"width": 1400, "height": 900})
    serve_component(page, editor)

    primary = page.locator(".MuiListItemText-primary", has_text=re.compile(r"^TotalCard \(analysis\)$"))
    item = page.locator(".MuiListItemButton-root", has=primary)
    expect(item).to_be_visible(timeout=20000)
    item.drag_to(page.locator(".react-flow__pane"), target_position={"x": 400, "y": 300})
    wait_until(lambda: any(i["component_id"] == ANALYSIS_ID for i in editor._tile_items), page, timeout=10000)

    (placed,) = [i for i in editor._tile_items if i["component_id"] == ANALYSIS_ID]
    assert placed["config"]["analysis"] == "TotalCard"
    name = next(n["name"] for n in ops.nodes() if n["component"] == ANALYSIS_ID)
    ops.connect("sales.pipeline", f"{name}.pipeline")
    node = editor._component_instances[placed["instance_id"]]
    wait_until(lambda: node._object is not None and node._error is None, page, timeout=10000)
    expect(page.locator(".react-flow__node", has_text="view: Panel Markdown")).to_be_visible(timeout=10000)

    page.get_by_role("button", name="Layout").click()
    expect(page.get_by_text("Total amount: 30")).to_be_visible(timeout=10000)
