from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

from pydantic_ai import ModelRetry
from pydantic_ai.messages import (
    ModelResponse, TextPart, ToolCallPart, ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession, _graph
from lumen.agent.assistant import INSTRUCTIONS
from lumen.agent.authoring import add_sql
from lumen.agent.data import list_tables, resolve_input, run_sql
from lumen.flowdash import bind_source, set_row_ceiling
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "customers": "SELECT * FROM (VALUES (1, 'Ada', 'east'), (2, 'Bo', 'west'), (3, 'Cy', 'east')) AS t(customer_id, customer, region)",
    "orders": "SELECT * FROM (VALUES (101, 1, 30, 'paid'), (102, 1, 20, 'paid'), (103, 2, 40, 'paid'), (104, 3, 5, 'cancelled')) AS t(order_id, customer_id, amount, status)",
}
PAID = (
    "SELECT c.customer, c.region, o.amount FROM orders o JOIN customers c "
    "ON o.customer_id = c.customer_id WHERE o.status = 'paid'"
)
BY_REGION = "SELECT region, SUM(amount) AS total FROM paid_by_region GROUP BY region"


def script(*responses):
    """A model answering each request with the next response, recording the requests it got."""
    requests = []

    def respond(messages, info):
        requests.append(messages[-1])
        return ModelResponse(parts=list(responses[len(requests) - 1]))

    return FunctionModel(respond, model_name="scripted"), requests


@pytest.fixture
def session():
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(lambda m, i: ModelResponse([TextPart("")])))
    yield session
    session.close()


@pytest.fixture
def ctx(session):
    return SimpleNamespace(deps=session.deps)


def _follower(session):
    session.add_sql("paid_by_region", "SELECT c.region, SUM(o.amount) AS total FROM orders o JOIN customers c "
                    "ON o.customer_id = c.customer_id WHERE o.status = 'paid' GROUP BY c.region")
    session.add_sql("top_region", "SELECT * FROM paid_by_region", source="paid_by_region")


async def test_add_sql_on_an_earlier_sql_node_reads_its_result(ctx, session):
    session.add_sql("paid", PAID)

    text = await add_sql(ctx, "paid", "by_customer", "SELECT customer, SUM(amount) AS total FROM paid GROUP BY customer ORDER BY customer")

    assert text.startswith("Placed SQL node 'by_customer' on the result of 'paid'.")
    assert "from: paid.result\n    to: by_customer.source" in session.graph()
    assert _graph.output(session.ops, "by_customer", "data").values.tolist() == [["Ada", 50], ["Bo", 40]]


async def test_add_sql_says_when_the_upstream_tile_is_capped(ctx, session):
    set_row_ceiling(session.ops.editor, 2)
    session.add_sql("paid", PAID)

    text = await add_sql(ctx, "paid", "n", "SELECT COUNT(*) AS n FROM paid")

    assert "It reads all 3 rows of 'paid'; its tile shows the first 2." in text
    assert _graph.output(session.ops, "n", "data").values.tolist() == [[3]]


async def test_add_sql_says_nothing_of_an_upstream_with_exactly_its_limit(ctx, session):
    set_row_ceiling(session.ops.editor, 3)
    session.add_sql("paid", PAID)

    text = await add_sql(ctx, "paid", "n", "SELECT COUNT(*) AS n FROM paid")

    assert "its tile shows" not in text


async def test_add_sql_refuses_a_node_that_is_neither_a_source_nor_a_query(ctx, session):
    session.add_sql("paid", PAID)
    session.ops.add_node("Lumen/view", {"spec": {"mark": "bar"}}, name="chart")

    with pytest.raises(ModelRetry, match="'chart' is a Lumen/view node; pass a source or an earlier SQL node"):
        await add_sql(ctx, "chart", "other", "SELECT 1 FROM orders")
    with pytest.raises(ModelRetry, match="not a table of this source"):
        await add_sql(ctx, "data", "other", "SELECT * FROM paid")


async def test_a_name_the_app_provides_a_source_under_names_the_source(ctx, session):
    bind_source(session.editor, session.source, "energy")
    session.add_sql("energy", PAID)

    with pytest.raises(ModelRetry, match="not a table of this source"):
        await add_sql(ctx, "energy", "n", "SELECT COUNT(*) AS n FROM energy")


def test_resolve_input_falls_back_to_the_only_source_only_for_unknown_names(ctx, session):
    session.add_sql("paid", PAID)

    assert resolve_input(ctx.deps, "orders")[0] == "data"
    name, src = resolve_input(ctx.deps, "paid")
    assert name == "paid" and "paid" in src.get_tables()


async def test_run_sql_queries_an_earlier_result(ctx, session):
    session.add_sql("paid", PAID)

    text = await run_sql(ctx, "SELECT COUNT(*) AS n FROM paid", "paid")

    assert "3" in text and "error" not in text.lower()


async def test_list_tables_lists_earlier_results_with_typed_columns(ctx, session):
    set_row_ceiling(session.ops.editor, 2)
    session.add_sql("paid", PAID)
    session.add_sql("regions", "SELECT DISTINCT region FROM customers WHERE region = 'west'")

    text = await list_tables(ctx)

    results = text.split("Earlier results you can query by table name (pass the node as add_sql's source):\n")[1]
    assert results.splitlines()[:2] == [
        "- regions: table 'regions' (1 row); columns: region string",
        "- paid: table 'paid' (2 rows shown, the tile's limit; a query reads every row); columns: customer string, region string, amount integer",
    ]


def test_instructions_edit_answers_in_place_and_build_new_questions_on_results():
    assert "When the user narrows or changes an answer already on the dashboard" in INSTRUCTIONS
    assert "answer's nodes in place; do not add a node for it" in INSTRUCTIONS
    assert "add_sql(source=<its SQL node>)" in INSTRUCTIONS


async def test_set_config_tests_a_followers_sql_against_its_upstreams_result():
    top = "SELECT region FROM paid_by_region ORDER BY total DESC LIMIT 1"
    model, requests = script(
        [ToolCallPart("set_config", {"name": "top_region", "config": {"sql": top}}, tool_call_id="s1")],
        [TextPart("Done.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    try:
        _follower(session)
        await session.ask("Only the top region")
        frame = _graph.output(session.ops, "top_region", "data")
    finally:
        session.close()

    (part,) = requests[1].parts
    assert isinstance(part, ToolReturnPart) and "Result: 1 rows" in part.model_response_str()
    assert frame.values.tolist() == [["east"]]


async def test_set_config_keeps_the_table_name_of_a_node_with_followers():
    model, requests = script(
        [ToolCallPart("set_config", {"name": "paid_by_region", "config": {"table": "totals"}}, tool_call_id="s1")],
        [TextPart("Done.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    try:
        _follower(session)
        await session.ask("Rename the table")
        config = _graph.config_of(session.ops, "paid_by_region")
    finally:
        session.close()

    (part,) = requests[1].parts
    assert part.outcome == "retried" and "'top_region' query 'paid_by_region'" in str(part.content)
    assert config["table"] == "paid_by_region"


async def test_set_config_names_a_follower_its_change_breaks():
    model, requests = script(
        [ToolCallPart("set_config", {"name": "paid_by_region", "config": {"sql": "SELECT region FROM customers"}}, tool_call_id="s1")],
        [TextPart("Done.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    try:
        session.add_sql("paid_by_region", "SELECT region, 1 AS total FROM customers")
        session.add_sql("ranked", "SELECT region FROM paid_by_region ORDER BY total DESC", source="paid_by_region")
        await session.ask("Just the regions")
    finally:
        session.close()

    (part,) = requests[1].parts
    content = part.model_response_str()
    assert isinstance(part, ToolReturnPart) and "Follower 'ranked' now fails:" in content and "total" in content


JOIN_SQL = (
    "SELECT r.region, c.customers, r.total FROM paid_by_region AS r JOIN customers_per_region AS c "
    "ON r.region = c.region ORDER BY r.region"
)


def _two_results(session):
    session.add_sql("paid", PAID)
    session.add_sql("paid_by_region", "SELECT region, SUM(amount) AS total FROM paid GROUP BY region", source="paid")
    session.add_sql("customers_per_region", "SELECT region, COUNT(*) AS customers FROM customers GROUP BY region")


async def test_add_sql_joins_earlier_results_given_as_a_list(ctx, session):
    _two_results(session)

    text = await add_sql(ctx, ["paid_by_region", "customers_per_region"], "joined", JOIN_SQL)

    assert text.startswith("Placed SQL node 'joined' on the results of 'paid_by_region' and 'customers_per_region'.")
    graph = session.graph()
    assert "from: paid_by_region.result\n    to: joined.source" in graph
    assert "from: customers_per_region.result\n    to: joined.source" in graph
    assert _graph.output(session.ops, "joined", "data").values.tolist() == [["east", 2, 50], ["west", 1, 40]]


async def test_add_sql_joins_only_earlier_sql_nodes(ctx, session):
    _two_results(session)

    with pytest.raises(ModelRetry, match="To join, pass earlier SQL nodes; 'data' is not one"):
        await add_sql(ctx, ["paid_by_region", "data"], "joined", JOIN_SQL)
    assert "joined" not in _graph.nodes(session.ops)


async def test_set_config_on_a_join_tests_its_sql_against_both_results():
    model, requests = script(
        [ToolCallPart("set_config", {"name": "joined", "config": {"sql": JOIN_SQL.replace("ORDER BY r.region", "WHERE r.region = 'west'")}}, tool_call_id="s1")],
        [TextPart("Done.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    try:
        _two_results(session)
        session.add_sql("joined", JOIN_SQL, source=["paid_by_region", "customers_per_region"])
        await session.ask("Only the west region")
        frame = _graph.output(session.ops, "joined", "data")
    finally:
        session.close()

    (part,) = requests[1].parts
    assert isinstance(part, ToolReturnPart) and "Result: 1 rows" in part.model_response_str()
    assert frame.values.tolist() == [["west", 1, 40]]


async def test_add_sql_takes_a_list_of_nodes_written_as_a_json_string(ctx, session):
    _two_results(session)

    await add_sql(ctx, '["paid_by_region", "customers_per_region"]', "joined", JOIN_SQL)

    assert _graph.output(session.ops, "joined", "data").values.tolist() == [["east", 2, 50], ["west", 1, 40]]


@pytest.mark.parametrize("source", [[], "[]"])
async def test_add_sql_refuses_an_empty_list(ctx, session, source):
    with pytest.raises(ModelRetry, match="Pass a source, an earlier SQL node, or a list"):
        await add_sql(ctx, source, "joined", JOIN_SQL)


async def test_add_sql_joins_a_node_named_twice_once(ctx, session):
    _two_results(session)

    await add_sql(ctx, ["paid_by_region", "paid_by_region"], "again", "SELECT * FROM paid_by_region")

    assert session.graph().count("to: again.source") == 1
