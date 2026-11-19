import pytest

pytest.importorskip("duckdb")
config_choices = pytest.importorskip("panel_flowdash.config_choices")

import panel as pn
import panel_material_ui as pmui

from panel_flowdash import FlowDash, columns_of
from panel_flowdash.agent import DashboardOps
from panel_flowdash.ui.status_bar import select_on_canvas

from lumen.flowdash import (
    COMPONENTS, FILTER_ID, PIPELINE_TYPE, SOURCE_ID, SOURCES_KEY, TABLE_ID,
)
from lumen.flowdash.choices import pipeline_column_types
from lumen.flowdash.nodes import table_pipeline
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
