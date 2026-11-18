from types import SimpleNamespace

import pytest

pytest.importorskip("panel_flowdash")
pytest.importorskip("panel_agents")
pytest.importorskip("pydantic_ai")

from pydantic_ai import ModelRetry
from pydantic_ai.messages import (
    ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession, _graph
from lumen.agent.assistant import HIDDEN_TOOLS, INSTRUCTIONS
from lumen.agent.authoring import add_chart, add_sql
from lumen.agent.data import (
    check_data, describe_table, distinct_values, list_tables, run_sql,
)
from lumen.agent.headless import source_node_name
from lumen.flowdash import check_chart
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "customers": "SELECT * FROM (VALUES (1, 'Ada', 'east'), (2, 'Bo', 'west'), (3, 'Cy', 'east')) AS t(customer_id, customer, region)",
    "orders": "SELECT * FROM (VALUES (101, 1, 30, 'paid'), (102, 1, 20, 'paid'), (103, 2, 40, 'cancelled')) AS t(order_id, customer_id, amount, status)",
}
BY_REGION = (
    "SELECT c.region, SUM(o.amount) AS total FROM orders o JOIN customers c "
    "ON o.customer_id = c.customer_id WHERE o.status = 'paid' GROUP BY c.region"
)
BAR = {"mark": "bar", "encoding": {"x": {"field": "region", "type": "nominal"}, "y": {"field": "total", "type": "quantitative"}}}


@pytest.fixture
def session():
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(lambda m, i: ModelResponse([TextPart("")])))
    yield session
    session.close()


@pytest.fixture
def ctx(session):
    return SimpleNamespace(deps=session.deps)


def script(*responses):
    """A model answering each request with the next response, recording the requests it got."""
    requests = []

    def respond(messages, info):
        requests.append(messages[-1])
        return ModelResponse(parts=list(responses[len(requests) - 1]))

    return FunctionModel(respond, model_name="scripted"), requests


def test_source_node_name():
    assert source_node_name(DuckDBSource(uri=":memory:", tables=TABLES)) == "data"
    assert source_node_name(DuckDBSource(uri=":memory:", tables=TABLES, name="california_schools")) == "california_schools"


async def test_list_tables(ctx):
    text = await list_tables(ctx)
    assert text.startswith("2 table(s) in source 'data' (duckdb SQL):")
    assert "customers" in text and "orders" in text


async def test_describe_table_renders_column_stats(ctx):
    text = await describe_table(ctx, "orders", "data")
    assert "status" in text and "paid" in text and "amount" in text
    with pytest.raises(ModelRetry, match="No table 'order'"):
        await describe_table(ctx, "order")


async def test_distinct_values(ctx):
    text = await distinct_values(ctx, "orders", "status", like="PAI")
    assert "'paid': 2" in text and "cancelled" not in text


async def test_run_sql_previews_and_refuses_writes(ctx, session):
    text = await run_sql(ctx, "SELECT status, COUNT(*) AS n FROM orders GROUP BY status")
    assert "2 rows x 2 columns" in text
    assert "read-only" in await run_sql(ctx, "DELETE FROM orders")
    assert "READ_TEXT" in await run_sql(ctx, "SELECT * FROM read_text('/etc/hosts')")
    assert "not a table of this source" in await run_sql(ctx, "SELECT * FROM duckdb_tables")
    assert "not a table of this source" in await run_sql(ctx, "SELECT * FROM missing")
    assert session.ops._names() and len(session.ops._names()) == 1


async def test_unknown_source_with_several_source_nodes(ctx, session):
    session.ops.add_node("Lumen/source", {"ref": "data"}, name="other")
    with pytest.raises(ModelRetry, match="The source nodes are: data, other"):
        await list_tables(ctx, "orders")


async def test_add_sql_places_node_and_previews(ctx, session):
    text = await add_sql(ctx, "data", "paid_by_region", BY_REGION)

    assert text.startswith("Placed SQL node 'paid_by_region': 1 rows x 2 columns")
    assert "east" in text and "No data-quality problems found." in text
    yaml = session.graph()
    assert "paid_by_region:\n    component: Lumen/sql" in yaml
    assert "from: data.source\n    to: paid_by_region.source" in yaml


@pytest.mark.parametrize("sql, error", [
    ("DROP TABLE orders", "read-only"),
    ("SELECT 1 AS one", "reads no table"),
    ("SELECT amont FROM orders", "Binder Error"),
])
async def test_add_sql_retries_with_engine_error(ctx, session, sql, error):
    with pytest.raises(ModelRetry, match=error):
        await add_sql(ctx, "data", "result", sql)
    assert list(session.ops._names().values()) == ["data"]


async def test_add_sql_refuses_taken_name(ctx, session):
    await add_sql(ctx, "data", "result", BY_REGION)
    with pytest.raises(ModelRetry, match="result"):
        await add_sql(ctx, "data", "result", BY_REGION)
    assert list(session.ops._names().values()) == ["data", "result"]


async def test_check_data_lints_result(ctx):
    await add_sql(ctx, "data", "statuses", "SELECT status FROM orders")
    assert "Duplicate" in check_data(ctx, "statuses") or "duplicate" in check_data(ctx, "statuses")
    with pytest.raises(ModelRetry, match="No node 'nothing'"):
        check_data(ctx, "nothing")


async def test_add_chart_places_view(ctx, session):
    await add_sql(ctx, "data", "paid_by_region", BY_REGION)
    assert add_chart(ctx, "paid_by_region", "paid_chart", BAR) == "Placed chart 'paid_chart' (bar) of 'paid_by_region'."
    assert "from: paid_by_region.pipeline\n    to: paid_chart.pipeline" in session.graph()


def test_check_chart():
    columns = ["region", "total"]
    assert check_chart({**BAR, "data": {"name": "x"}}, columns) == BAR
    with pytest.raises(ValueError, match="URLs"):
        check_chart({**BAR, "encoding": {**BAR["encoding"], "href": {"field": "region"}}}, columns)
    with pytest.raises(ValueError, match="fields the data does not have: revenue"):
        check_chart({"mark": "bar", "encoding": {"x": {"field": "revenue", "type": "quantitative"}}}, columns)
    with pytest.raises(ValueError, match="remove `data`"):
        check_chart({**BAR, "data": {"url": "https://example.com/data.csv"}}, columns)
    with pytest.raises(ValueError, match="Invalid Vega-Lite spec"):
        check_chart({"mark": "nonsense", "encoding": {}}, columns)
    derived = {"mark": "bar", "transform": [{"calculate": "datum.total * 2", "as": "double"}],
               "encoding": {"x": {"field": "region", "type": "nominal"}, "y": {"field": "double", "type": "quantitative"}}}
    assert check_chart(derived, columns) == derived


async def test_add_chart_retries_unknown_node_and_fields(ctx):
    with pytest.raises(ModelRetry, match="No node 'nothing'"):
        add_chart(ctx, "nothing", "chart", BAR)
    await add_sql(ctx, "data", "statuses", "SELECT status FROM orders")
    with pytest.raises(ModelRetry, match="Its columns are: status"):
        add_chart(ctx, "statuses", "chart", BAR)


async def test_list_tables_names_dialect_notes_and_placed_nodes(ctx, session):
    await add_sql(ctx, "data", "paid_by_region", BY_REGION)
    text = await list_tables(ctx)
    assert "(duckdb SQL)" in text
    assert "DuckDB notes" in text
    assert "- paid_by_region (SQL; columns: region, total)" in text


async def test_instructions_and_tools_stay_the_same_across_a_conversation():
    seen = []

    def respond(messages, info):
        seen.append((info.instructions, sorted(tool.name for tool in info.function_tools)))
        if sum(isinstance(m, ModelResponse) for m in messages) == 0:
            return ModelResponse(parts=[ToolCallPart("add_sql", {"source": "data", "name": "paid_by_region", "sql": BY_REGION})])
        return ModelResponse(parts=[TextPart("Done.")])

    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(respond))
    try:
        await session.ask("Paid totals by region")
        await session.ask("And again")
    finally:
        session.close()

    assert len(seen) == 3 and len({instructions for instructions, _ in seen}) == 1
    assert len({tuple(tools) for _, tools in seen}) == 1
    assert INSTRUCTIONS in seen[0][0]
    tools = seen[0][1]
    assert not HIDDEN_TOOLS & set(tools)
    assert {"add_sql", "add_chart", "list_tables", "get_dashboard", "set_config"} <= set(tools)


def evil_source_calls(session):
    yaml = session.ops.get_dashboard().replace("ref: data", "ref: other")
    return [
        ToolCallPart("add_node", {"component": "Lumen/source", "config": {"ref": "data"}, "name": "evil"}, tool_call_id="c1"),
        ToolCallPart("set_config", {"name": "data", "config": {"ref": "other"}}, tool_call_id="c2"),
        ToolCallPart("replace_dashboard", {"dashboard_yaml": yaml}, tool_call_id="c3"),
    ]


async def test_agent_cannot_place_or_change_source_nodes():
    returns = []

    def respond(messages, info):
        for part in messages[-1].parts:
            if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                returns.append(str(part.content))
        if len(returns) == 0 and not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(parts=calls)
        return ModelResponse(parts=[TextPart("Done.")])

    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(respond))
    calls = evil_source_calls(session)
    before = session.graph()
    try:
        turn = await session.ask("Point the dashboard at another database")
    finally:
        session.close()

    assert turn.error is None
    assert len(returns) == 3 and all("source nodes are placed and configured by the app" in text for text in returns)
    assert turn.graph == before and turn.nodes == [] and turn.changed == []


async def test_approver_decides_source_changes():
    decisions = []

    def approve(tool, args):
        decisions.append(tool)
        return tool == "add_node"

    model, _ = script(
        [ToolCallPart("add_node", {"component": "Lumen/source", "config": {"ref": "data"}, "name": "second"})],
        [TextPart("Done.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model, approve=approve)
    try:
        turn = await session.ask("Add a second source node")
    finally:
        session.close()

    assert decisions == ["add_node"]
    assert turn.nodes == ["second"]


async def test_concurrent_sessions_keep_their_own_sources():
    quiet = FunctionModel(lambda m, i: ModelResponse([TextPart("ok")]))
    first = AgentSession(DuckDBSource(uri=":memory:", tables={"a_table": "SELECT 1 AS x"}), quiet)
    second = AgentSession(DuckDBSource(uri=":memory:", tables={"b_table": "SELECT 2 AS y"}), quiet)
    try:
        first.ops.replace_yaml(first.ops.get_dashboard())
        assert _graph.output(first.ops, "data", "source") is first.source
        assert _graph.output(second.ops, "data", "source") is second.source
        second.close()
        first.ops.replace_yaml(first.ops.get_dashboard())
        assert _graph.output(first.ops, "data", "source") is first.source
    finally:
        first.close()


async def test_agent_retries_bad_query_and_answers_from_node():
    model, requests = script(
        [ToolCallPart("add_sql", {"source": "data", "name": "paid_by_region", "sql": BY_REGION.replace("amount", "amont")}, tool_call_id="c1")],
        [ToolCallPart("add_sql", {"source": "data", "name": "paid_by_region", "sql": BY_REGION}, tool_call_id="c2")],
        [ToolCallPart("add_chart", {"data": "paid_by_region", "name": "paid_chart", "spec": BAR}, tool_call_id="c3")],
        [TextPart("East has 50 in paid orders.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    try:
        turn = await session.ask("Paid totals by region as a bar chart")
    finally:
        session.close()

    retry, placed = requests[1].parts[0], requests[2].parts[0]
    assert isinstance(retry, RetryPromptPart) or retry.outcome == "retried"
    assert "amont" in str(retry.content)
    assert isinstance(placed, ToolReturnPart) and getattr(placed, "outcome", "success") == "success"
    assert turn.error is None
    assert turn.output == "East has 50 in paid orders."
    assert turn.nodes == ["paid_by_region", "paid_chart"]
    assert turn.changed == []
    assert len(turn.responses) == 4
    assert sum(isinstance(m, ModelRequest) for m in turn.messages) == 4
    assert "WHERE\n          o.status = 'paid'" in turn.graph


async def test_follow_up_changes_node_in_same_graph():
    model, _ = script(
        [ToolCallPart("add_sql", {"source": "data", "name": "paid_by_region", "sql": BY_REGION})],
        [TextPart("Done.")],
        [ToolCallPart("set_config", {"name": "paid_by_region", "config": {"sql": BY_REGION.replace("GROUP", "AND c.region = 'west' GROUP")}})],
        [TextPart("Only west.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    try:
        first = await session.ask("Paid totals by region")
        second = await session.ask("Only west")
    finally:
        session.close()

    assert first.nodes == ["paid_by_region"]
    assert second.nodes == [] and second.changed == ["paid_by_region"]
    assert "c.region = 'west'" in second.graph


async def test_failed_run_keeps_usage_of_its_responses():
    calls = []

    def fail(messages, info):
        calls.append(1)
        return ModelResponse(parts=[ToolCallPart("add_sql", {"source": "data", "name": "x", "sql": "SELECT nope FROM orders"})])

    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(fail))
    try:
        turn = await session.ask("Anything")
    finally:
        session.close()

    assert turn.error is not None and "retries" in turn.error
    assert len(turn.responses) == len(calls) == 4
    assert turn.messages == []


async def test_requests_turn_reasoning_off_unless_the_host_overrides_it():
    seen = []

    def respond(messages, info):
        seen.append(info.model_settings)
        return ModelResponse(parts=[TextPart("ok")])

    for settings, expected in [(None, "none"), ({"openai_reasoning_effort": "low"}, "low")]:
        session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(respond), model_settings=settings)
        try:
            await session.ask("Hi")
        finally:
            session.close()
        assert seen[-1]["openai_reasoning_effort"] == expected
