from types import SimpleNamespace

import panel as pn
import pytest

pytest.importorskip("duckdb")

from panel_flowdash import register
from pydantic_ai.messages import (
    ModelResponse, TextPart, ToolCallPart, ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from lumen.agent import (
    AgentSession, _graph, create_agent, create_explore_agent, explore,
    headless,
)
from lumen.agent.assistant import (
    EXPLORE_FOLLOW_UPS, EXPLORE_INSTRUCTIONS, FOLLOW_UPS, INSTRUCTIONS,
)
from lumen.agent.explore import OwnNodesOnly
from lumen.flowdash.app import build_routes
from lumen.sources.duckdb import DuckDBSource


def tools(agent):
    return {tool for toolset in agent.toolsets for tool in getattr(toolset, "tools", {})}


def test_the_explore_agent_has_lumens_tools_but_cannot_remove_nodes():
    agent = create_explore_agent("test")

    assert {"list_tables", "add_sql", "add_chart", "run_sql", "set_config"} <= tools(agent)
    assert "remove_node" not in tools(agent)
    assert agent.name == "Explore"


def test_explore_builds_follow_ups_on_earlier_answers_and_the_drawer_edits_them():
    assert FOLLOW_UPS in INSTRUCTIONS and EXPLORE_FOLLOW_UPS not in INSTRUCTIONS
    assert EXPLORE_FOLLOW_UPS in EXPLORE_INSTRUCTIONS and FOLLOW_UPS not in EXPLORE_INSTRUCTIONS
    assert "but never\n  change them" in EXPLORE_INSTRUCTIONS and "remove_node" not in EXPLORE_INSTRUCTIONS
    assert "use set_config only to fix a node you placed for the current question" in EXPLORE_INSTRUCTIONS.replace("\n  ", " ")
    assert "edit that answer's node with set_config" in INSTRUCTIONS
    assert "remove_node" in tools(create_agent("test"))


@pytest.mark.parametrize("models", [["test"], ["test=Test", "openai:gpt-6-luna=Luna"]])
def test_lumen_serve_gives_explore_lumens_explore_agent(tmp_path, models):
    csv = tmp_path / "sales.csv"
    csv.write_text("category,amount\nA,10\nB,20\n")
    app = build_routes([str(csv)], db_path=tmp_path / "dash.db", agent_model=models)["/"]()

    assert app.explore_agent.name == "Explore"
    assert "add_sql" in tools(app.explore_agent)
    assert app._agent_session.explore_runner is not None
    assert "add_sql" in tools(app.agent) and app.agent is not app.explore_agent


def explore_session(*responses, before=None):
    """
    An AgentSession on Lumen's Explore agent with a model answering each
    request with the next response; ``before(n)``, if given, runs before
    the n-th request is answered, e.g. to change the dashboard mid-run.
    """
    requests = []

    def respond(messages, info):
        requests.append(messages[-1])
        if before is not None:
            before(len(requests))
        return ModelResponse(parts=list(responses[len(requests) - 1]))

    model = FunctionModel(respond, model_name="scripted")
    tables = {"sales": "SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, amount)"}
    return AgentSession(DuckDBSource(uri=":memory:", tables=tables), model, agent=create_explore_agent(model)), requests


async def test_explore_cannot_change_a_dashboard_node_or_an_earlier_answer():
    session, requests = explore_session(
        [ToolCallPart("set_config", {"name": "totals", "config": {"sql": "SELECT 1 AS x FROM sales"}}, tool_call_id="s1")],
        [TextPart("Refused.")],
        [ToolCallPart("add_sql", {"source": "data", "name": "mine", "sql": "SELECT category FROM sales"}, tool_call_id="a1")],
        [ToolCallPart("set_config", {"name": "mine", "config": {"sql": "SELECT amount FROM sales"}}, tool_call_id="s2")],
        [TextPart("Fixed my own node.")],
        [ToolCallPart("set_config", {"name": "mine", "config": {"sql": "SELECT 2 AS y FROM sales"}}, tool_call_id="s3")],
        [TextPart("Refused.")],
    )
    try:
        session.add_sql("totals", "SELECT category, SUM(amount) AS total FROM sales GROUP BY category")
        await session.ask("Change the totals")
        await session.ask("Show the categories")
        await session.ask("Change that earlier answer")
        configs = {name: _graph.config_of(session.ops, name)["sql"] for name in ("totals", "mine")}
    finally:
        session.close()

    refused = [part for index in (1, 6) for part in requests[index].parts]
    assert all(part.outcome == "retried" and "stay as they are in Explore" in str(part.content) for part in refused)
    assert isinstance(requests[4].parts[0], ToolReturnPart) and requests[4].parts[0].outcome != "retried"
    assert configs["totals"].startswith("SELECT category, SUM(amount)")
    assert "amount" in configs["mine"] and "2 AS y" not in configs["mine"]


def test_checked_config_previews_the_copy_explore_makes():
    from panel_flowdash import agent as flowdash_agent

    if not hasattr(flowdash_agent, "ExplorationOps"):
        pytest.skip("panel-flowdash without ExplorationOps (fd #238): Explore edits nothing earlier to copy")
    pytest.xfail(
        "With ExplorationOps an edit to an earlier answer makes a copy, and CheckedConfig previews the "
        "frozen original by its name; make it read the copy, then turn this into a real test."
    )


async def test_explore_refuses_a_frozen_node_before_testing_its_sql():
    session, requests = explore_session(
        [ToolCallPart("set_config", {"name": "totals", "config": {"sql": "SELECT nope FROM sales"}}, tool_call_id="s1")],
        [TextPart("Refused.")],
    )
    try:
        session.add_sql("totals", "SELECT category FROM sales")
        await session.ask("Change the totals")
    finally:
        session.close()

    (part,) = requests[1].parts
    assert "stay as they are in Explore" in str(part.content) and "nope" not in str(part.content)


async def test_explore_cannot_change_a_node_someone_else_adds_mid_run():
    sessions = []

    def add_late(n):
        if n == 2:
            sessions[0].add_sql("late", "SELECT amount FROM sales")

    session, requests = explore_session(
        [ToolCallPart("list_tables", {}, tool_call_id="l1")],
        [ToolCallPart("set_config", {"name": "late", "config": {"sql": "SELECT 99 AS z FROM sales"}}, tool_call_id="s1")],
        [TextPart("Refused.")],
        before=add_late,
    )
    sessions.append(session)
    try:
        session.add_sql("totals", "SELECT category FROM sales")
        await session.ask("Change the late node")
        sql = _graph.config_of(session.ops, "late")["sql"]
    finally:
        session.close()

    assert "stay as they are in Explore" in str(requests[2].parts[0].content)
    assert "99" not in sql


async def test_explore_cannot_change_a_node_that_replaced_its_own_mid_run():
    sessions = []

    def replace_mine(n):
        if n == 3:
            sessions[0].ops.remove_node("mine")
            sessions[0].add_sql("mine", "SELECT amount FROM sales")

    session, requests = explore_session(
        [ToolCallPart("add_sql", {"source": "data", "name": "mine", "sql": "SELECT category FROM sales"}, tool_call_id="a1")],
        [ToolCallPart("list_tables", {}, tool_call_id="l1")],
        [ToolCallPart("set_config", {"name": "mine", "config": {"sql": "SELECT 99 AS z FROM sales"}}, tool_call_id="s1")],
        [TextPart("Refused.")],
        before=replace_mine,
    )
    sessions.append(session)
    try:
        await session.ask("Show the categories, then change them")
        sql = _graph.config_of(session.ops, "mine")["sql"]
    finally:
        session.close()

    assert "stay as they are in Explore" in str(requests[3].parts[0].content)
    assert "amount" in sql and "99" not in sql


@register(component=True, config_schema={"type": "object", "properties": {"text": {"type": "string", "default": ""}}})
def note(config=None, **kwargs):
    return pn.pane.Markdown("note")


async def test_explore_cannot_change_a_function_component_it_did_not_place(monkeypatch):
    monkeypatch.setattr(headless, "COMPONENTS", {**headless.COMPONENTS, "Test/note": note})
    session, requests = explore_session(
        [ToolCallPart("set_config", {"name": "note", "config": {"text": "changed"}}, tool_call_id="s1")],
        [TextPart("Refused.")],
    )
    try:
        session.ops.add_node("Test/note", {"text": "kept"}, name="note")
        assert session.ops._instance("note") not in session.editor._component_instances
        await session.ask("Change the note")
        config = session.ops.node_config("note")
    finally:
        session.close()

    assert "stay as they are in Explore" in str(requests[1].parts[0].content)
    assert config == {"text": "kept"}


class _Context:
    def __init__(self, ops):
        self.deps = SimpleNamespace(ops=ops)
        self.run_id = "run"


async def _step_aside(ops_class, monkeypatch):
    monkeypatch.setattr(explore, "ExplorationOps", ops_class)
    called = []

    async def handler(args):
        called.append(args)
        return "changed"

    result = await OwnNodesOnly().wrap_tool_execute(
        _Context(ops_class.__new__(ops_class)), call=None, tool_def=SimpleNamespace(name="set_config"),
        args={"name": "frozen"}, handler=handler,
    )
    return result, called


async def test_the_guard_steps_aside_on_exploration_ops(monkeypatch):
    class ExplorationOps:
        pass

    assert await _step_aside(ExplorationOps, monkeypatch) == ("changed", [{"name": "frozen"}])


async def test_the_guard_steps_aside_on_panel_flowdashs_exploration_ops(monkeypatch):
    from panel_flowdash import agent as flowdash_agent

    if not hasattr(flowdash_agent, "ExplorationOps"):
        pytest.skip("panel-flowdash without ExplorationOps (fd #238)")
    assert await _step_aside(flowdash_agent.ExplorationOps, monkeypatch) == ("changed", [{"name": "frozen"}])


async def test_explore_fixes_a_table_it_placed_but_cannot_rewrite_an_earlier_node_with_add_metric():
    session, requests = explore_session(
        [ToolCallPart("add_sql", {"source": "data", "name": "mine", "sql": "SELECT category FROM sales"}, tool_call_id="a1")],
        [ToolCallPart("add_table", {"data": "mine", "name": "rows"}, tool_call_id="a2")],
        [ToolCallPart("set_config", {"name": "rows", "config": {"spec": {"page_size": 5}}}, tool_call_id="s1")],
        [TextPart("Placed and fixed my table.")],
        [ToolCallPart("add_metric", {"metric": "capacity", "name": "totals"}, tool_call_id="m1")],
        [TextPart("Refused.")],
    )
    try:
        session.add_sql("totals", "SELECT category, SUM(amount) AS total FROM sales GROUP BY category")
        await session.ask("Show the categories as a table of five rows")
        await session.ask("Turn the totals into a metric")
        spec = _graph.config_of(session.ops, "rows")["spec"]
    finally:
        session.close()

    fixed = requests[3].parts[0]
    assert isinstance(fixed, ToolReturnPart) and fixed.outcome != "retried"
    assert spec == {"page_size": 5}
    (refused,) = requests[5].parts
    assert refused.outcome == "retried" and "stay as they are in Explore" in str(refused.content)
