import asyncio
import datetime as dt
import time

import pandas as pd
import pytest

pytest.importorskip("duckdb")
config_choices = pytest.importorskip("panel_flowdash.config_choices")

import panel as pn
import panel_material_ui as pmui

from panel_flowdash import FlowDash, columns_of
from panel_flowdash.agent import DashboardOps
from panel_flowdash.ui.status_bar import select_on_canvas

from lumen.flowdash import (
    COMPONENTS, FILTER_ID, PIPELINE_TYPE, SOURCE_ID, SOURCES_KEY, SQL_ID,
    TABLE_ID, choices as lumen_choices,
)
from lumen.flowdash.choices import pipeline_column_types
from lumen.flowdash.nodes import table_pipeline
from lumen.sources.base import InMemorySource
from lumen.sources.duckdb import DuckDBSource

from .test_host import settle

TABLES = {
    "sales": (
        "SELECT * FROM (VALUES ('A', 10, 1.5, TRUE, TIMESTAMP '2020-01-01', DATE '2020-01-02')) "
        "AS t(category, amount, price, paid, sold_at, sold_on)"
    ),
    "regions": "SELECT * FROM (VALUES ('A', 'North')) AS t(category, region)",
}


@pytest.fixture
def source():
    return DuckDBSource(uri=":memory:", tables=TABLES)


def dashboard(source, *sources):
    refs = {"data": source, **{f"data_{i + 2}": s for i, s in enumerate(sources)}}
    editor = FlowDash(components=COMPONENTS, context={SOURCES_KEY: refs}, notifications=False)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {}, name="table")
    ops.connect("data.source", "table.source")
    return ops


def test_pipeline_columns_have_types_without_fetching_rows(source):
    pipeline = table_pipeline(source, "sales")
    assert pipeline_column_types(pipeline) == {
        "category": "string", "amount": "integer", "price": "number", "paid": "boolean",
        "sold_at": "datetime", "sold_on": "datetime",
    }
    assert columns_of(pipeline, PIPELINE_TYPE)[0] == ("category", "string")


async def test_source_node_offers_the_host_refs(source):
    ops = dashboard(source, DuckDBSource(uri=":memory:", tables=TABLES))
    choices = await config_choices.node_choices(ops.editor, ops._instance("data"))
    assert choices["ref"].values() == ["data", "data_2"]


async def test_table_node_offers_the_tables_of_its_source(source):
    ops = dashboard(source)
    choices = await config_choices.node_choices(ops.editor, ops._instance("table"))
    assert choices["table"].values() == ["regions", "sales"]


async def test_table_node_offers_nothing_without_a_source(source):
    ops = dashboard(source)
    ops.disconnect("data.source", "table.source")
    assert await config_choices.node_choices(ops.editor, ops._instance("table")) == {}


async def test_filter_node_offers_the_columns_of_its_pipeline(source):
    ops = dashboard(source)
    ops.set_config("table", {"table": "sales"})
    ops.add_node(FILTER_ID, {}, name="filter")
    ops.connect("table.pipeline", "filter.pipeline")
    choices = await config_choices.node_choices(ops.editor, ops._instance("filter"))
    assert choices["field"].labelled() == {
        "category (string)": "category", "amount (integer)": "amount", "price (number)": "price",
        "paid (boolean)": "paid", "sold_at (datetime)": "sold_at", "sold_on (datetime)": "sold_on",
    }


async def test_inspector_shows_the_tables_as_a_dropdown(source):
    ops = dashboard(source)
    editor = ops.editor
    node = ops._instance("table")
    select_on_canvas(editor._flow, node_id=node)
    await settle()
    form = next(iter(pn.Column(*editor._inspector._body).select(pn.Param)))
    table = form._widgets["table"]
    assert isinstance(table, pmui.Select)
    assert list(table.options.values()) == ["", "regions", "sales"]
    table.value = "sales"
    await settle()
    assert editor._component_instances[node].table == "sales"


ORDERS = (
    "SELECT * FROM (VALUES ('B', 1, TRUE, DATE '2020-01-01'), ('A', 2, FALSE, DATE '2020-01-02'), "
    "('B', 3, TRUE, DATE '2020-01-03'), (NULL, 2, NULL, NULL)) AS t(category, amount, paid, sold_on)"
)


class SlowSource(DuckDBSource):
    """A DuckDB source whose queries take half a second, blocking the thread they run on."""

    async def execute_with_timeout(self, sql_query, timeout, fetch=False):
        time.sleep(0.5)
        return await super().execute_with_timeout(sql_query, timeout, fetch)


def filter_dashboard(source, limit=None):
    ops = dashboard(source)
    ops.set_config("table", {"table": "orders", "limit": limit} if limit else {"table": "orders"})
    ops.add_node(FILTER_ID, {"field": "category"}, name="filter")
    ops.connect("table.pipeline", "filter.pipeline")
    return ops


@pytest.fixture
def orders():
    return DuckDBSource(uri=":memory:", tables={"orders": ORDERS})


async def value_choices(ops, name="filter", **kwargs):
    choices = await config_choices.node_choices(ops.editor, ops._instance(name), **kwargs)
    return choices["value"].values() if "value" in choices else None


async def test_filter_node_offers_the_distinct_values_of_its_field(orders):
    ops = filter_dashboard(orders, limit=1)
    assert await value_choices(ops) == ["A", "B"]
    ops.set_config("filter", {"field": "amount"})
    assert await value_choices(ops) == [1, 2, 3]
    ops.set_config("filter", {"field": "paid"})
    assert await value_choices(ops) == [False, True]


async def test_filter_value_choices_follow_upstream_filters(orders):
    ops = filter_dashboard(orders)
    ops.add_node(FILTER_ID, {"field": "amount"}, name="amount")
    ops.set_inputs("amount", {"value": [1, 3]})
    ops.disconnect("table.pipeline", "filter.pipeline")
    ops.connect("table.pipeline", "amount.pipeline")
    ops.connect("amount.filtered", "filter.pipeline")
    assert await value_choices(ops) == ["B"]


async def test_filter_offers_no_values_for_dates_or_too_many(orders, monkeypatch):
    ops = filter_dashboard(orders)
    ops.set_config("filter", {"field": "sold_on"})
    assert await value_choices(ops) is None
    monkeypatch.setattr(lumen_choices, "DISTINCT_LIMIT", 2)
    ops.set_config("filter", {"field": "amount"})
    assert await value_choices(ops) is None
    ops.set_config("filter", {"field": "category"})
    assert await value_choices(ops) == ["A", "B"]


async def test_filter_value_choices_are_read_once_per_pipeline(orders, monkeypatch):
    from lumen.flowdash import nodes

    ops = filter_dashboard(orders)
    calls = []
    execute = nodes.execute
    monkeypatch.setattr(nodes, "execute", lambda *args, **kw: calls.append(args[1]) or execute(*args, **kw))
    assert await value_choices(ops) == ["A", "B"]
    assert await value_choices(ops) == ["A", "B"]
    assert len(calls) == 1 and "DISTINCT" in calls[0]


async def test_filter_value_choices_from_rows_of_a_source_without_sql():
    frame = pd.DataFrame({"category": ["B", "A", None, "B"], "amount": [1, 2, 3, 4]})
    ops = filter_dashboard(InMemorySource(tables={"orders": frame}))
    assert await value_choices(ops) == ["A", "B"]


async def test_filter_value_choices_of_a_sql_result(orders):
    ops = dashboard(orders)
    ops.add_node(SQL_ID, {"sql": "SELECT category FROM orders WHERE amount > 1"}, name="query")
    ops.connect("data.source", "query.source")
    ops.add_node(FILTER_ID, {"field": "category"}, name="filter")
    ops.connect("query.pipeline", "filter.pipeline")
    assert await value_choices(ops) == ["A", "B"]


async def test_filter_value_choices_keep_the_event_loop_free():
    ops = filter_dashboard(SlowSource(uri=":memory:", tables={"orders": ORDERS}))
    ticks = 0

    async def tick():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    ticker = asyncio.ensure_future(tick())
    try:
        start = time.monotonic()
        assert await value_choices(ops) == ["A", "B"]
        elapsed = time.monotonic() - start
    finally:
        ticker.cancel()
    assert elapsed >= 0.5
    assert ticks >= 20


async def test_filter_value_choices_finish_after_the_inspector_stops_waiting():
    ops = filter_dashboard(SlowSource(uri=":memory:", tables={"orders": ORDERS}))
    assert await value_choices(ops, timeout=0.05) is None
    await settle(0.7)
    start = time.monotonic()
    assert await value_choices(ops, timeout=0.05) == ["A", "B"]
    assert time.monotonic() - start < 0.05


async def test_inspector_offers_the_filter_values_as_a_dropdown(orders):
    ops = filter_dashboard(orders)
    editor = ops.editor
    node = ops._instance("filter")
    select_on_canvas(editor._flow, node_id=node)
    await settle()
    selects = [
        w for w in pn.Column(*editor._inspector._body).select(pmui.Select)
        if list(w.options.values())[-2:] == ["A", "B"]
    ]
    assert selects


async def test_filter_value_choices_sort_text_naturally():
    frame = pd.DataFrame({"category": ["region_10", "Region_2", "region_1"], "amount": [1, 2, 3]})
    ops = filter_dashboard(InMemorySource(tables={"orders": frame}))
    assert await value_choices(ops) == ["region_1", "Region_2", "region_10"]


async def test_filter_offers_no_values_for_an_object_column_of_dates():
    frame = pd.DataFrame({"category": [dt.date(2020, 1, 2), dt.date(2020, 1, 1)], "amount": [1, 2]}, dtype=object)
    ops = filter_dashboard(InMemorySource(tables={"orders": frame}))
    assert await value_choices(ops) is None
