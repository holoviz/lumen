"""
Brushing a Lumen chart in the browser filters the rest of a FlowDash
dashboard in the source's query. Runs where pytest-playwright is installed.
"""
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")
pytest.importorskip("panel_flowdash.selection")

from panel.tests.util import serve_component, wait_until
from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps

from lumen.flowdash import (
    COMPONENTS, DATA_ID, FILTER_ID, SOURCE_ID, TABLE_ID, VIEW_ID, bind_source,
)
from lumen.sources.duckdb import DuckDBSource

N = 2000


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


ROWS = pd.DataFrame({
    "category": np.random.default_rng(0).choice(list("ABCDEF"), N),
    "price": np.round(np.random.default_rng(1).gamma(2.0, 20.0, N), 2),
    "quantity": np.random.default_rng(2).integers(1, 4, N),
})

BAR = {
    "width": 300, "height": 200,
    "params": [{"name": "brush", "select": {"type": "interval", "encodings": ["x"]}}],
    "mark": "bar",
    "encoding": {
        "x": {"field": "category", "type": "nominal", "sort": None},
        "y": {"aggregate": "sum", "field": "price", "type": "quantitative"},
        "opacity": {"condition": {"param": "brush", "value": 1}, "value": 0.3},
    },
}

SCATTER = {
    "width": 300, "height": 200,
    "params": [{"name": "brush", "select": {"type": "interval", "encodings": ["x", "y"]}}],
    "mark": "point",
    "encoding": {
        "x": {"field": "price", "type": "quantitative"},
        "y": {"field": "quantity", "type": "quantitative"},
    },
}


def data_queries(executed):
    return [q for q in executed if "__count" not in q and not (q.endswith(" LIMIT 1") and "subquery" not in q)]


def open_dashboard(page, spec):
    source = DuckDBSource.from_df({"sales": ROWS})
    editor = FlowDash(components=COMPONENTS, editable=False, notifications=False)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node(VIEW_ID, {"spec": spec}, name="chart")
    ops.connect("sales.pipeline", "chart.pipeline")
    ops.add_node(FILTER_ID, {}, name="brushed")
    ops.connect("sales.pipeline", "brushed.pipeline")
    ops.connect("chart.selection", "brushed.selection")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("brushed.filtered", "rows.pipeline")
    ops.add_node("Widgets/KPI", {"label": "Orders"}, name="orders")
    ops.connect("rows.data", "orders.value", [{"op": "column", "column": "price"}, {"op": "aggregate", "how": "count"}])
    page.set_viewport_size({"width": 1400, "height": 1000})
    serve_component(page, editor)
    canvas = page.locator(".vega-embed canvas, .vega-embed svg").first
    canvas.wait_for(state="visible", timeout=20000)
    return ops, stable_box(page, canvas)


def stable_box(page, locator, timeout=10):
    """The bounding box of ``locator`` once tiles stop moving it."""
    boxes = [locator.bounding_box()]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        page.wait_for_timeout(250)
        boxes.append(locator.bounding_box())
        if len(boxes) >= 3 and boxes[-1] == boxes[-2] == boxes[-3]:
            break
    return boxes[-1]


def node(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def drag(page, box, x0, x1, y0=0.5, y1=0.5, steps=30, pause=0):
    # The 300x200 plot sits at the right of the canvas, below a few pixels of padding.
    left, top = box["x"] + box["width"] - 300 - 5, box["y"] + 5
    page.mouse.move(left + 300 * x0, top + 200 * y0)
    page.mouse.down()
    for i in range(1, steps + 1):
        page.mouse.move(left + 300 * (x0 + (x1 - x0) * i / steps), top + 200 * (y0 + (y1 - y0) * i / steps))
        if pause:
            page.wait_for_timeout(pause)
    page.mouse.up()
    return left, top


def test_brushing_a_chart_filters_the_dashboard_in_the_source_query(page, executed):
    ops, box = open_dashboard(page, BAR)
    kpi, chart, rows = node(ops, "orders"), node(ops, "chart"), node(ops, "rows")
    assert kpi.value == N
    executed.clear()

    left, top = drag(page, box, 0.02, 0.45)
    wait_until(lambda: kpi.value is not None and kpi.value < N, page, timeout=10000)
    page.wait_for_timeout(500)

    (predicate,) = chart.selection.predicates
    selected = ROWS[ROWS.category.isin(predicate.values)]
    assert 0 < len(selected) < N
    assert kpi.value == len(selected)
    assert len(rows._object) == len(selected)
    page.get_by_text(f"{len(selected):,}").first.wait_for(timeout=8000)
    queries = data_queries(executed)
    assert len(queries) == 1, queries
    assert 'WHERE "category" IN (' in queries[0]
    assert len(chart._object.get_data()) == N

    executed.clear()
    page.mouse.click(left + 300 * 0.95, top + 100)
    wait_until(lambda: kpi.value == N, page, timeout=10000)
    assert len(rows._object) == N
    assert data_queries(executed) == []


def test_a_fast_drag_runs_one_query(page, executed):
    ops, box = open_dashboard(page, SCATTER)
    kpi = node(ops, "orders")
    executed.clear()

    drag(page, box, 0.05, 0.5, y0=0.05, y1=0.9, steps=30)
    wait_until(lambda: kpi.value is not None and kpi.value < N, page, timeout=10000)
    page.wait_for_timeout(800)

    queries = data_queries(executed)
    assert len(queries) == 1, queries
    assert '"price" BETWEEN' in queries[0] and '"quantity" BETWEEN' in queries[0]
