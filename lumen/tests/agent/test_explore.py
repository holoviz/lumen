from types import SimpleNamespace

import panel as pn
import pytest

pytest.importorskip("duckdb")

from panel_flowdash import register
from pydantic_ai.messages import (
    ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart,
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
from lumen.agent.headless import streamable
from lumen.flowdash import SOURCE_ID, SQL_ID, bound_sources
from lumen.flowdash.app import build_routes
from lumen.sources.duckdb import DuckDBSource

TOTALS = 'SELECT category, SUM(amount) AS total FROM "{table}" GROUP BY category'


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


def _exploration_ops():
    from panel_flowdash import agent as flowdash_agent

    if not hasattr(flowdash_agent, "ExplorationOps"):
        pytest.skip("panel-flowdash without ExplorationOps")
    return flowdash_agent.ExplorationOps


def scripted(*steps, before=None):
    """
    A streaming model making one step per request: a tool call as
    ``(tool, args)``, or a final text; it records the requests it got.
    ``before(n)``, if given, runs before the n-th request is answered.
    """
    requests = []

    def respond(messages, info):
        requests.append(messages[-1])
        if before is not None:
            before(len(requests))
        step = steps[len(requests) - 1]
        if isinstance(step, str):
            return ModelResponse(parts=[TextPart(step)])
        tool, args = step
        return ModelResponse(parts=[ToolCallPart(tool, args, tool_call_id=f"call-{len(requests)}")])

    return streamable(FunctionModel(respond, model_name="scripted")), requests


def results(requests):
    """What each tool call returned, or the retry it asked for, by tool call id."""
    return {
        part.tool_call_id: str(part.content) for request in requests for part in request.parts
        if isinstance(part, (ToolReturnPart, RetryPromptPart))
    }


def exploring(tmp_path, **routes):
    """
    ``lumen serve --flowdash`` on a CSV of sales with Lumen's Explore agent
    on a fresh dashboard, ``routes`` passed to ``build_routes``.
    ``script(*steps)`` scripts the agent's next run, which ``ask`` runs, as
    FlowDash's Explore view does.
    """
    _exploration_ops()
    csv = tmp_path / "sales.csv"
    csv.write_text("category,amount\nA,10\nB,20\nB,5\n")
    app = build_routes([str(csv)], db_path=tmp_path / "dash.db", agent_model=["test"], **routes)["/"]()
    app._editor.new_dashboard("Sales")
    session = app._agent_session
    source = next(iter(bound_sources(session.ops.editor).values()))

    class Explore:
        table = source.get_tables()[0]
        totals = TOTALS.format(table=table)
        ops = session.ops
        explorations = session.explorations
        runner = session.explore_runner
        requests: list = []

        def script(self, *steps, before=None):
            model, self.requests = scripted(*steps, before=before)
            self.runner.agent = create_explore_agent(model)
            self.runner.messages = []

        async def ask(self, question):
            await self.runner.run(question)
            return self.runner.run_id

        def close(self):
            session.explorations.dispose()

    return Explore()


@pytest.fixture
def explore_app(tmp_path):
    explore = exploring(tmp_path)
    yield explore
    explore.close()


def _source(explore_app):
    return next(iter(bound_sources(explore_app.ops.editor)))


async def test_explore_runs_lumens_tools_as_its_own_on_the_exploration(explore_app):
    ref = _source(explore_app)
    explore_app.ops.add_node(SOURCE_ID, {"ref": ref}, name="data")
    explore_app.script(("add_sql", {"source": "data", "name": "totals", "sql": explore_app.totals}), "Done.")
    run = await explore_app.ask("Total by category")

    exploration = explore_app.explorations.exploration()
    assert results(explore_app.requests)["call-1"].startswith("Placed SQL node 'totals'")
    assert exploration.leaves(run) == ["totals"]
    assert [node["name"] for node in explore_app.ops.nodes()] == ["data"]


async def test_explore_previews_the_copy_it_makes_of_an_earlier_answer(explore_app):
    explore_app.ops.add_node(SOURCE_ID, {"ref": _source(explore_app)}, name="data")
    explore_app.script(("add_sql", {"source": "data", "name": "totals", "sql": explore_app.totals}), "Done.")
    first = await explore_app.ask("Total by category")
    explore_app.script(
        ("set_config", {"name": "totals", "config": {"sql": f'SELECT category, COUNT(*) AS n FROM "{explore_app.table}" GROUP BY category'}}),
        "Done.",
    )
    await explore_app.ask("Count them instead")

    result = results(explore_app.requests)["call-1"]
    ex = explore_app.explorations
    assert "'totals' as 'totals_2'" in result and "Use 'totals_2' from now on." in result
    assert "n" in result.split("Result")[1] and "total" not in result.split("Result")[1].split("Data checks")[0]
    assert "COUNT(*)" in ex.node_config("totals_2")["sql"]
    assert "SUM(amount)" in ex.node_config("totals")["sql"] and ex.exploration().leaves(first) == ["totals"]


async def test_explore_reports_a_refused_change_of_a_dashboard_node_without_a_preview(explore_app):
    explore_app.ops.add_node(SOURCE_ID, {"ref": _source(explore_app)}, name="data")
    explore_app.ops.add_node(SQL_ID, {"sql": explore_app.totals, "table": "totals"}, name="totals")
    explore_app.ops.connect("data.source", "totals.source")
    explore_app.script(("set_config", {"name": "totals", "config": {"sql": f'SELECT 1 AS x FROM "{explore_app.table}"'}}), "Done.")
    await explore_app.ask("Change the totals")

    result = results(explore_app.requests)["call-1"]
    assert result.startswith("Refused:") and "Result" not in result
    assert explore_app.ops.node_config("totals")["sql"] == explore_app.totals


async def test_explore_turns_a_refusal_of_lumens_tools_into_a_retry(explore_app):
    explore_app.ops.add_node(SOURCE_ID, {"ref": _source(explore_app)}, name="data")
    explore_app.ops.editor.read_only = True
    explore_app.script(("add_sql", {"source": "data", "name": "totals", "sql": explore_app.totals}), "I can't.")
    await explore_app.ask("Total by category")

    (refused,) = explore_app.requests[1].parts
    assert refused.outcome == "retried" and str(refused.content).startswith("Refused: ")
    assert explore_app.runner.error is None


async def test_explore_stops_lumens_tools_once_the_user_opens_another_dashboard(explore_app):
    explore_app.ops.add_node(SOURCE_ID, {"ref": _source(explore_app)}, name="data")
    def switch(n):
        if n == 2:
            explore_app.ops.editor.new_dashboard("Other")

    explore_app.script(
        ("list_tables", {}), ("add_sql", {"source": "data", "name": "totals", "sql": explore_app.totals}), "Stopped.",
        before=switch,
    )
    await explore_app.ask("Total by category")

    assert results(explore_app.requests)["call-2"].startswith("Refused: the user switched to another dashboard")
