import asyncio
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
    COMPONENTS, FILTER_ID, PIPELINE_TYPE, SOURCE_ALIASES_KEY, SOURCE_ID,
    SOURCES_KEY, SQL_ID, TABLE_ID, choices as lumen_choices,
)
from lumen.flowdash.choices import (
    palette_presets, pipeline_column_types, source_presets, table_presets,
)
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


async def test_source_node_offers_the_alias_it_names_a_source_by(source):
    context = {SOURCES_KEY: {"sales": source}, SOURCE_ALIASES_KEY: {"data": "sales"}}
    ops = DashboardOps(FlowDash(components=COMPONENTS, context=context, notifications=False))
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="old")
    ops.add_node(SOURCE_ID, {"ref": "sales"}, name="new")
    old = await config_choices.node_choices(ops.editor, ops._instance("old"))
    new = await config_choices.node_choices(ops.editor, ops._instance("new"))
    assert old["ref"].values() == ["sales", "data"]
    assert new["ref"].values() == ["sales"]


async def test_table_node_offers_the_tables_of_its_source(source):
    ops = dashboard(source)
    choices = await config_choices.node_choices(ops.editor, ops._instance("table"))
    assert choices["table"].values() == ["regions", "sales"]


async def test_table_node_offers_only_the_refs_without_a_source(source):
    ops = dashboard(source)
    ops.disconnect("data.source", "table.source")
    choices = await config_choices.node_choices(ops.editor, ops._instance("table"))
    assert list(choices) == ["ref"]
    assert choices["ref"].values() == ["data"]


async def test_table_node_offers_the_tables_of_the_source_its_ref_names(source):
    other = DuckDBSource(uri=":memory:", tables={"plants": "SELECT 1 AS x"})
    editor = FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"shop": source, "energy": other}},
                      notifications=False)
    ops = DashboardOps(editor)
    ops.add_node(TABLE_ID, {"ref": "energy"}, name="table")
    choices = await config_choices.node_choices(editor, ops._instance("table"))
    assert choices["ref"].values() == ["energy", "shop"]
    assert choices["table"].values() == ["plants"]


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


needs_presets = pytest.mark.skipif(
    not palette_presets(source_presets), reason="panel-flowdash without palette_presets"
)


def test_presets_list_each_source_and_table(source):
    other = DuckDBSource(uri=":memory:", tables={"plants": "SELECT 1 AS x"})
    context = {SOURCES_KEY: {"shop": source, "energy": other}}
    assert [p["title"] for p in source_presets(context)] == ["shop (source)", "energy (source)"]
    assert source_presets(context)[0]["config"] == {"ref": "shop"}
    assert source_presets(context)[0]["tooltip"] == "Source shop: regions, sales"
    tables = table_presets(context)
    assert [p["title"] for p in tables] == ["regions (shop)", "sales (shop)", "plants (energy)"]
    assert tables[1]["config"] == {"ref": "shop", "table": "sales"}
    assert tables[1]["tooltip"] == "Source shop, table sales"
    assert source_presets({}) == table_presets({}) == []


@needs_presets
def test_a_table_preset_places_one_working_node(source):
    editor = FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"shop": source}}, notifications=False)
    items = {item["label"]: item for group in editor._palette.items for item in group["items"]}
    assert items["sales (shop)"]["tooltip"] == "Source shop, table sales"

    editor._on_palette_clicked(items["sales (shop)"])
    ops = DashboardOps(editor)
    (table_node,) = (node["name"] for node in ops.nodes())
    assert ops.node_config(table_node) == {"ref": "shop", "table": "sales"}
    pipeline = ops.node_output(f"{table_node}.pipeline")
    assert pipeline.source is source and pipeline.table == "sales"


@needs_presets
def test_a_table_preset_wired_to_a_source_node_reads_it(source):
    editor = FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"shop": source}}, notifications=False)
    items = {item["label"]: item for group in editor._palette.items for item in group["items"]}

    editor._on_palette_clicked(items["shop (source)"])
    editor._on_palette_clicked(items["sales (shop)"])
    ops = DashboardOps(editor)
    source_node, table_node = (node["name"] for node in ops.nodes())
    assert ops.node_config(source_node) == {"ref": "shop"}
    ops.connect(f"{source_node}.source", f"{table_node}.source")
    assert ops.node_output(f"{source_node}.source") is source
    assert ops.node_output(f"{table_node}.pipeline").table == "sales"


def test_presets_read_each_sources_tables_once_until_it_reloads(source, monkeypatch):
    calls = []
    get_tables = DuckDBSource.get_tables
    monkeypatch.setattr(DuckDBSource, "get_tables", lambda self: calls.append(1) or get_tables(self))
    context = {SOURCES_KEY: {"shop": source}}
    for _ in range(3):
        table_presets(context)
        source_presets(context)
    assert len(calls) == 1
    source.clear_cache()
    table_presets(context)
    assert len(calls) == 2
    source.tables = {**source.tables, "extra": "SELECT 1 AS x"}
    assert "extra (shop)" in [p["title"] for p in table_presets(context)]


def test_presets_name_tables_after_the_served_source(tmp_path, monkeypatch):
    from lumen.flowdash.app import resolve_sources, source_refs

    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("turbine,mw\na,1\n")
    (tmp_path / "plants.csv").write_text("plant,mw\nx,1\n")
    context = {SOURCES_KEY: source_refs(resolve_sources(["turbines.csv", "plants.csv"]))}
    assert [p["title"] for p in source_presets(context)] == ["local (source)"]
    assert source_presets(context)[0]["tooltip"] == "Source local: plants, turbines"
    assert [p["title"] for p in table_presets(context)] == ["plants (local)", "turbines (local)"]


def test_table_presets_name_the_file_a_numbered_table_reads(tmp_path, monkeypatch):
    from lumen.flowdash.app import resolve_sources, source_refs

    monkeypatch.chdir(tmp_path)
    for folder in ("x", "y"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "plants.csv").write_text("plant\nq\n")
    context = {SOURCES_KEY: source_refs(resolve_sources(["y/plants.csv", "x/plants.csv"]))}
    tooltips = {p["title"]: p["tooltip"] for p in table_presets(context)}
    assert tooltips == {
        "plants (local)": "Source local, table plants, file y/plants.csv",
        "plants_2 (local)": "Source local, table plants_2, file x/plants.csv",
    }


async def test_table_node_shows_its_ref_as_ignored_while_a_source_is_connected(source):
    ops = dashboard(source)
    choices = await config_choices.node_choices(ops.editor, ops._instance("table"))
    assert list(choices) == ["table"]
    ops.editor._apply_config_patch(ops._instance("table"), {"ref": "data"})
    choices = await config_choices.node_choices(ops.editor, ops._instance("table"))
    assert choices["ref"].labelled() == {"data (ignored while a source is connected)": "data"}
    assert choices["table"].values() == ["regions", "sales"]


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
