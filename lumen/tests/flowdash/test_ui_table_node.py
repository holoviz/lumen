"""
Dragging a table of the app's data in from the component palette places
one working Lumen Table node. Runs where pytest-playwright is installed.
"""
import re

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")

from panel.tests.util import serve_component, wait_until
from panel_flowdash import FlowDash
from playwright.sync_api import expect

from lumen.flowdash import COMPONENTS, SOURCES_KEY, TABLE_ID
from lumen.flowdash.choices import palette_presets, table_presets
from lumen.sources.duckdb import DuckDBSource

pytestmark = pytest.mark.skipif(
    not palette_presets(table_presets), reason="panel-flowdash without palette_presets"
)


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


def test_a_table_dragged_from_the_palette_reads_its_source(page):
    source = DuckDBSource(uri=":memory:", tables={
        "sales": "SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, amount)",
    })
    editor = FlowDash(components=COMPONENTS, notifications=False, context={SOURCES_KEY: {"shop": source}})
    page.set_viewport_size({"width": 1400, "height": 900})
    serve_component(page, editor)

    primary = page.locator(".MuiListItemText-primary", has_text=re.compile(r"^sales \(shop\)$"))
    item = page.locator(".MuiListItemButton-root", has=primary)
    expect(item).to_be_visible(timeout=20000)
    item.drag_to(page.locator(".react-flow__pane"), target_position={"x": 120, "y": 120})
    wait_until(lambda: len(editor._tile_items) == 1, page, timeout=10000)

    (placed,) = editor._tile_items
    assert placed["component_id"] == TABLE_ID
    assert placed["config"] == {"ref": "shop", "table": "sales"}
    node = editor._component_instances[placed["instance_id"]]
    wait_until(lambda: node._object is not None and node._error is None, page, timeout=10000)
    assert node._object.source is source
    expect(page.locator(".react-flow__node")).to_have_count(1)
