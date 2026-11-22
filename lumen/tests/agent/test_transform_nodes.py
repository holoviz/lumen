from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession
from lumen.agent.authoring import add_chart
from lumen.agent.data import list_tables
from lumen.flowdash import (
    AGGREGATE_ID, DERIVE_ID, SORT_ID, TABLE_ID,
)
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "orders": "SELECT * FROM (VALUES (101, 'east', 30), (102, 'east', 20), (103, 'west', 40)) AS t(order_id, region, amount)",
}
BAR = {"mark": "bar", "encoding": {"x": {"field": "region", "type": "nominal"}, "y": {"field": "amount", "type": "quantitative"}}}


@pytest.fixture
def session():
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(lambda m, i: ModelResponse([TextPart("")])))
    ops = session.ops
    ops.add_node(TABLE_ID, {"table": "orders"}, name="orders")
    ops.connect("data.source", "orders.source")
    ops.add_node(DERIVE_ID, {"column": "net", "expression": '"amount" * 0.8'}, name="with_net")
    ops.connect("orders.pipeline", "with_net.pipeline")
    ops.add_node(AGGREGATE_ID, {"by": ["region"], "columns": ["amount", "net"], "method": "sum"}, name="by_region")
    ops.connect("with_net.result", "by_region.pipeline")
    ops.add_node(SORT_ID, {"by": ["amount"], "descending": True, "limit": 1}, name="top_region")
    ops.connect("by_region.result", "top_region.pipeline")
    yield session
    session.close()


@pytest.fixture
def ctx(session):
    return SimpleNamespace(deps=session.deps)


async def test_list_tables_shows_each_transform_step(ctx):
    text = await list_tables(ctx)
    assert "- top_region (sorted by 'amount' descending, first 1 of by_region)" in text
    assert "- by_region (sum of 'amount', 'net' by 'region' of with_net)" in text
    assert "- with_net (column 'net' = \"amount\" * 0.8 of orders)" in text


def test_add_chart_charts_a_transform_node(ctx, session):
    assert add_chart(ctx, "by_region", "region_chart", BAR) == "Placed chart 'region_chart' (bar) of 'by_region'."
    assert "from: by_region.result\n    to: region_chart.pipeline" in session.graph()
