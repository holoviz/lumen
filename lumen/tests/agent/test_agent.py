import json

from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

from pydantic_ai import ModelRetry
from pydantic_ai.messages import (
    ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession, _graph
from lumen.agent.assistant import (
    INSTRUCTIONS, MODEL_SETTINGS, OTHER_MODEL_SETTINGS, create_agent,
    model_settings_for,
)
from lumen.agent.authoring import add_chart, add_sql
from lumen.agent.data import (
    check_data, data_toolset, describe_table, distinct_values, list_tables,
    run_sql,
)
from lumen.agent.headless import source_node_name
from lumen.flowdash import FILTER_ID, TABLE_ID, check_chart
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
    assert "status VARCHAR {paid:2, cancelled:1}" in text
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
    assert len(session.ops.nodes()) == 1


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
    assert [node["name"] for node in session.ops.nodes()] == ["data"]


async def test_add_sql_refuses_taken_name(ctx, session):
    await add_sql(ctx, "data", "result", BY_REGION)
    with pytest.raises(ModelRetry, match="result"):
        await add_sql(ctx, "data", "result", BY_REGION)
    assert [node["name"] for node in session.ops.nodes()] == ["data", "result"]


async def test_check_data_lints_result(ctx):
    assert "check_data" in data_toolset(check=True).tools and "check_data" not in data_toolset().tools
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
    assert "BY c.region" in text or "GROUP BY c.region" in text
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
    assert INSTRUCTIONS.strip() in seen[0][0]
    tools = seen[0][1]
    assert tools == ["add_chart", "add_sql", "describe_table", "distinct_values", "list_tables", "remove_node", "run_sql", "set_config"]


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

    model = FunctionModel(respond)
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model, agent=create_agent(model, dashboard_tools=None))
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
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model, approve=approve,
                           agent=create_agent(model, dashboard_tools=None))
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


class OpenAIFunctionModel(FunctionModel):
    @property
    def system(self) -> str:
        return "openai"


def test_openai_models_keep_openai_settings_and_others_get_neutral_ones():
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    openai = OpenAIProvider(api_key="x")
    assert model_settings_for(OpenAIResponsesModel("gpt-6-luna", provider=openai)) == MODEL_SETTINGS
    assert model_settings_for(OpenAIChatModel("gpt-6-luna", provider=openai)) == MODEL_SETTINGS
    azure = OpenAIChatModel("gpt-6-luna", provider=AzureProvider(
        azure_endpoint="https://x.openai.azure.com", api_version="2024-12-01-preview", api_key="x"))
    assert azure.system == "azure" and model_settings_for(azure) == MODEL_SETTINGS
    haiku = OpenRouterModel("anthropic/claude-haiku-5.5", provider=OpenRouterProvider(api_key="x"))
    assert model_settings_for(haiku) == OTHER_MODEL_SETTINGS == {"thinking": False, "max_tokens": 1000, "cache": True}


async def test_requests_to_an_anthropic_model_on_openrouter_turn_reasoning_off_and_cache_the_prompt():
    import httpx

    from openai import AsyncOpenAI
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "provider": "Anthropic",
            "model": "anthropic/claude-haiku-5.5", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        })

    client = AsyncOpenAI(api_key="x", base_url="https://openrouter.test/api/v1",
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    model = OpenRouterModel("anthropic/claude-haiku-5.5", provider=OpenRouterProvider(openai_client=client))
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    try:
        await session.ask("Hi")
    finally:
        session.close()

    body = bodies[0]
    assert body["reasoning"] == {"effort": "none"} and "reasoning_effort" not in body
    assert body["max_tokens"] == 1000
    breakpoints = {"type": "ephemeral", "ttl": "5m"}
    assert body["messages"][0]["content"][-1]["cache_control"] == breakpoints
    assert body["messages"][-1]["content"][-1]["cache_control"] == breakpoints
    assert body["tools"][-1]["cache_control"] == breakpoints


async def test_a_model_picked_for_a_run_gets_its_own_settings_under_the_hosts():
    seen = []

    def respond(messages, info):
        seen.append(info.model_settings)
        return ModelResponse(parts=[TextPart("ok")])

    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(respond))
    try:
        await session.agent.run("Hi", deps=session.deps)
        await session.agent.run("Hi", deps=session.deps, model=OpenAIFunctionModel(respond), model_settings={"temperature": 0})
    finally:
        session.close()

    assert seen[0] == {"max_tokens": 1000}
    assert seen[1] == {**MODEL_SETTINGS, "temperature": 0}


async def test_host_model_settings_replace_the_defaults():
    seen = []

    def respond(messages, info):
        seen.append(info.model_settings)
        return ModelResponse(parts=[TextPart("ok")])

    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), OpenAIFunctionModel(respond),
                           model_settings={"openai_reasoning_effort": "low"})
    try:
        await session.ask("Hi")
    finally:
        session.close()
    assert seen == [{"openai_reasoning_effort": "low"}]


async def test_data_agent_leaves_out_flowdash_instructions_and_catalogue():
    from panel_flowdash.agent.assistant import (
        INSTRUCTIONS as FLOWDASH_INSTRUCTIONS,
    )

    seen = []

    def respond(messages, info):
        seen.append(info.instructions)
        return ModelResponse(parts=[TextPart("ok")])

    model = FunctionModel(respond)
    for agent in (None, create_agent(model, dashboard_tools=None)):
        session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model, agent=agent)
        try:
            await session.ask("Hi")
        finally:
            session.close()
    data, dashboard = seen
    assert data.startswith(INSTRUCTIONS.strip()) and FLOWDASH_INSTRUCTIONS not in data and "Components you may place" not in data
    assert FLOWDASH_INSTRUCTIONS in dashboard and "Components you may place" in dashboard


async def test_lookups_are_stubbed_when_the_next_turn_starts():
    from lumen.agent.history import SUPERSEDED

    model, requests = script(
        [ToolCallPart("list_tables", {}, tool_call_id="l1")],
        [ToolCallPart("add_sql", {"source": "data", "name": "paid_by_region", "sql": BY_REGION}, tool_call_id="a1")],
        [TextPart("Done.")],
        [TextPart("Still done.")],
    )
    seen = []
    inner = model.function

    def respond(messages, info):
        seen.append([part.content for m in messages if isinstance(m, ModelRequest) for part in m.parts
                     if isinstance(part, ToolReturnPart) and part.tool_name == "list_tables"])
        return inner(messages, info)

    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(respond))
    try:
        turn = await session.ask("Paid totals by region")
        await session.ask("And now?")
    finally:
        session.close()

    # Within the run the history only grows, so its prefix stays cacheable.
    assert seen[1][0].startswith("2 table(s)") and seen[2][0].startswith("2 table(s)")
    assert seen[3] == [SUPERSEDED]
    assert turn.error is None and turn.nodes == ["paid_by_region"]


async def test_model_trace_records_the_stubbed_history(tmp_path):
    from panel_agents.adapters.model_trace import load_messages
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    model, _ = script(
        [ToolCallPart("list_tables", {}, tool_call_id="l1")],
        [ToolCallPart("add_sql", {"source": "data", "name": "paid_by_region", "sql": BY_REGION}, tool_call_id="a1")],
        [TextPart("Done.")],
        [TextPart("Still done.")],
    )
    trace = tmp_path / "trace.jsonl"
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model, trace=trace)
    try:
        await session.ask("Paid totals by region")
        await session.ask("And now?")
        held = ModelMessagesTypeAdapter.dump_python(session.runner.messages, mode="json")
    finally:
        session.close()
    recorded = ModelMessagesTypeAdapter.dump_python(load_messages(trace), mode="json")
    strip = lambda ms: [{k: v for k, v in m.items() if k not in ("metadata", "timestamp")} for m in ms]  # noqa: E731
    assert strip(recorded) == strip(held)
    assert any("rewind" in line for line in trace.read_text().splitlines())


async def test_instructions_and_tools_are_the_same_for_every_source():
    import json

    seen = []

    def respond(messages, info):
        seen.append((info.instructions, json.dumps([(t.name, t.description, t.parameters_json_schema) for t in info.function_tools])))
        return ModelResponse(parts=[TextPart("ok")])

    for tables in (TABLES, {"other": "SELECT 1 AS x"}):
        session = AgentSession(DuckDBSource(uri=":memory:", tables=tables, name="warehouse"), FunctionModel(respond))
        try:
            await session.ask("Hi")
        finally:
            session.close()
    assert seen[0] == seen[1]


async def test_dashboard_tool_presets():
    seen = []

    def respond(messages, info):
        seen.append(sorted(tool.name for tool in info.function_tools))
        return ModelResponse(parts=[TextPart("ok")])

    model = FunctionModel(respond)
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model, agent=create_agent(model, dashboard_tools="layout"))
    try:
        await session.ask("Hi")
    finally:
        session.close()
    assert {"connect", "disconnect", "place_tile", "set_config", "remove_node"} <= set(seen[0])
    assert "replace_dashboard" not in seen[0]


async def test_list_tables_summarizes_placed_charts(ctx, session):
    await add_sql(ctx, "data", "paid_by_region", BY_REGION)
    add_chart(ctx, "paid_by_region", "paid_chart", BAR)
    text = await list_tables(ctx)
    assert "- paid_chart (chart of paid_by_region; bar; x=region, y=total)" in text


async def test_list_tables_shortens_long_node_sql(ctx, monkeypatch):
    import lumen.agent.data as data

    monkeypatch.setattr(data, "NODE_SQL_CHARS", 20)
    await add_sql(ctx, "data", "paid_by_region", BY_REGION)
    text = await list_tables(ctx)
    line = next(line for line in text.splitlines() if line.startswith("- paid_by_region"))
    shown = " ".join(_graph.config_of(ctx.deps.ops, "paid_by_region")["sql"].split())
    assert line.endswith(f"{shown[:20]} [... {len(shown) - 20} more characters]")


async def test_node_listing_is_bounded(ctx, session):
    from lumen.agent.data import NODE_SQL_CHARS, NODES_SHOWN

    long_sql = "SELECT status, " + ", ".join(f"amount AS a{i}" for i in range(80)) + " FROM orders"
    for i in range(NODES_SHOWN + 3):
        session.add_sql(f"node_{i:02d}", long_sql if i == NODES_SHOWN + 2 else "SELECT status FROM orders")
    text = await list_tables(ctx)
    listed = [line for line in text.split("\n\n")[0].splitlines()[1:]]
    assert listed[0].startswith(f"- node_{NODES_SHOWN + 2:02d} (SQL;") and listed[0].endswith("more characters]")
    assert len(listed[0]) < NODE_SQL_CHARS + 200
    assert listed[-1] == "- and 3 more; list_tables(node=...) shows one in full."
    detail = await list_tables(ctx, node=f"node_{NODES_SHOWN + 2:02d}")
    assert detail.endswith("FROM orders")
    with pytest.raises(ModelRetry, match="No node 'missing'"):
        await list_tables(ctx, node="missing")


async def test_answer_cut_off_at_the_output_limit_is_retried():
    from lumen.agent.assistant import CUT_OFF

    requests = []

    def respond(messages, info):
        requests.append(messages[-1])
        if len(requests) == 1:
            return ModelResponse(parts=[TextPart("The total is   ")], finish_reason="length")
        return ModelResponse(parts=[TextPart("The total is 35.")])

    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(respond))
    try:
        result = await create_agent(FunctionModel(respond)).run("Total?", deps=session.deps)
    finally:
        session.close()
    assert result.output == "The total is 35."
    assert CUT_OFF in str(requests[1].parts[0].content)


async def test_set_config_checks_and_runs_sql_and_chart_specs():
    model, requests = script(
        [ToolCallPart("add_sql", {"source": "data", "name": "paid_by_region", "sql": BY_REGION}, tool_call_id="a1")],
        [ToolCallPart("add_chart", {"data": "paid_by_region", "name": "paid_chart", "spec": BAR}, tool_call_id="a2")],
        [ToolCallPart("set_config", {"name": "paid_by_region", "config": {"sql": "SELECT region, SUM(missing) FROM customers GROUP BY region"}}, tool_call_id="s1")],
        [ToolCallPart("set_config", {"name": "paid_by_region", "config": {"sql": "SELECT * FROM read_text('/etc/hosts')"}}, tool_call_id="s2")],
        [ToolCallPart("set_config", {"name": "paid_chart", "config": {"spec": {**BAR, "mark": {"type": "bar", "href": "https://example.com"}}}}, tool_call_id="s3")],
        [ToolCallPart("set_config", {"name": "paid_by_region", "config": {"sql": "SELECT region, COUNT(*) AS total FROM customers GROUP BY region"}}, tool_call_id="s4")],
        [TextPart("Done.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    try:
        turn = await session.ask("Count customers per region instead")
        config = _graph.config_of(session.ops, "paid_by_region")
    finally:
        session.close()

    def returned(index):
        (part,) = requests[index].parts
        return part

    retried = [str(returned(index).content) for index in (3, 4, 5)]
    assert "missing" in retried[0] and "READ_TEXT" in retried[1] and "URL" in retried[2]
    result = returned(6)
    assert isinstance(result, ToolReturnPart) and "Result: 2 rows" in result.model_response_str()
    assert all(not isinstance(returned(index), ToolReturnPart) or returned(index).outcome == "retried" for index in (3, 4, 5))
    assert "COUNT(*)" in config["sql"] and turn.error is None


def _filtered_customers(session):
    ops = session.ops
    ops.add_node(TABLE_ID, {"table": "customers"}, name="customers")
    ops.connect("data.source", "customers.source")
    ops.add_node(FILTER_ID, {"field": "region"}, name="by_region")
    ops.connect("customers.pipeline", "by_region.pipeline")


REGION_COUNT = {"mark": "bar", "encoding": {"x": {"field": "region", "type": "nominal"}, "y": {"aggregate": "count", "type": "quantitative"}}}


async def test_add_chart_charts_a_filter_node_and_checks_its_columns(ctx, session):
    _filtered_customers(session)

    assert add_chart(ctx, "by_region", "regions", REGION_COUNT) == "Placed chart 'regions' (bar) of 'by_region'."
    assert "from: by_region.filtered\n    to: regions.pipeline" in session.graph()
    with pytest.raises(ModelRetry, match="revenue"):
        add_chart(ctx, "by_region", "other", {"mark": "bar", "encoding": {"x": {"field": "revenue", "type": "nominal"}}})
    text = await list_tables(ctx)
    assert "- by_region (filter on 'region' of customers)" in text
    assert "- customers (table 'customers' of data)" in text


async def test_set_config_checks_a_chart_fed_by_a_filter_node():
    model, requests = script(
        [ToolCallPart("set_config", {"name": "regions", "config": {"spec": {"mark": "bar", "encoding": {"x": {"field": "revenue", "type": "nominal"}}}}}, tool_call_id="s1")],
        [TextPart("Done.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    try:
        _filtered_customers(session)
        add_chart(SimpleNamespace(deps=session.deps), "by_region", "regions", REGION_COUNT)
        await session.ask("Chart revenue")
        spec = _graph.config_of(session.ops, "regions")["spec"]
    finally:
        session.close()

    (part,) = requests[1].parts
    assert "revenue" in str(part.content) and part.outcome == "retried"
    assert spec == REGION_COUNT
