"""The FlowDash agent as an eval subject, recorded sessions as cases, and cassette playback."""

import json

from pathlib import Path

import pytest

pytest.importorskip("panel_flowdash")
pytest.importorskip("panel_agents")
pytest.importorskip("pydantic_ai")

from panel_agents.adapters.cassette import Cassette
from panel_agents.adapters.model_trace import load_messages
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_evals import Case, Dataset

from lumen.agent import AgentSession
from lumen.agent.evals import FlowDashSubject, agent_fingerprint
from lumen.agent.sessions import (
    dump_session, load_session, save_session, session_case,
)
from lumen.ai.evals.harness import (
    CheckResult, Expected, Inputs, Output, artifacts_path, evaluate,
    resolve_subject,
)

from .cases import DATASET, source

SESSIONS = Path(__file__).parent / "sessions"
SESSION_NAMES = sorted(path.stem for path in SESSIONS.glob("*.yaml"))


def scripted(*responses):
    def respond(messages, info):
        count = sum(isinstance(message, ModelResponse) for message in messages)
        return ModelResponse(parts=list(responses[count % len(responses)]))

    return FunctionModel(respond, model_name="gpt-6-luna")


def by_name(*names):
    return [case for case in DATASET.cases if case.name in names]


def test_flowdash_subject_is_registered_and_offline_with_a_model():
    subject = resolve_subject("flowdash", model=scripted([TextPart("x")]))
    assert isinstance(subject, FlowDashSubject)
    assert subject.supports_architecture is False
    assert subject.requires_api_key is False
    assert resolve_subject("flowdash").requires_api_key is True


def test_fingerprint_hashes_agent_surface_without_a_run():
    subject = FlowDashSubject()
    assert subject.fingerprint() == agent_fingerprint()
    assert len(subject.fingerprint()) == 16


async def test_subject_scored_end_to_end_on_outcomes(tmp_path):
    model = scripted(
        [ToolCallPart("list_tables", {})],
        [ToolCallPart("add_sql", {"source": "data", "name": "amount_by_category",
                                  "sql": "SELECT category, SUM(amount) AS total FROM sales GROUP BY category"})],
        [ToolCallPart("add_chart", {"data": "amount_by_category", "name": "amount_chart", "spec": {
            "mark": "bar", "encoding": {"x": {"field": "category", "type": "nominal"},
                                        "y": {"field": "total", "type": "quantitative"}}}})],
        [TextPart("A totals 15 and B 20.")],
    )
    subject = FlowDashSubject(model=model)
    dataset = Dataset[Inputs, Output, Expected](
        name="flowdash", cases=by_name("sales_by_category", "sales_chart", "agent_table_list", "validation_incomplete"),
        evaluators=[CheckResult()],
    )
    output = tmp_path / "flowdash.json"
    report = await evaluate(subject.build_model("gpt-6-luna"), dataset, source, output, subject=subject, scoring="all")
    cases = {case.name: case for case in report.cases}

    assert not report.failures
    assert {name: r.value for name, r in cases["sales_by_category"].assertions.items()} == {
        "all_turns_completed": True, "status": True, "output": True, "rows": True}
    assert {name: r.value for name, r in cases["sales_chart"].assertions.items()} == {
        "all_turns_completed": True, "status": True, "output": True, "chart_mark": True}
    assert cases["agent_table_list"].assertions["listing"].value is True
    assert not cases["validation_incomplete"].assertions

    saved = json.loads(output.read_text())
    assert saved["run"]["subject"] == "flowdash"
    assert saved["run"]["scoring"] == "outcome"
    assert saved["run"]["subject_fingerprint"] == agent_fingerprint()
    assert saved["run"]["settings"]["subject_settings"] == {"model_settings": {"openai_reasoning_effort": "none", "max_tokens": 1000}}
    turn = next(case for case in saved["cases"] if case["name"] == "sales_chart")["turns"][0]
    assert turn["chart_marks"] == ["bar"]
    assert turn["chart_encodings"] == [{"x": {"field": "category", "type": "nominal"},
                                        "y": {"field": "total", "type": "quantitative"}}]
    assert sorted(turn["rows"]) == [["A", 15.0], ["B", 20.0]]
    assert [call["name"] for call in turn["tool_calls"]] == ["list_tables", "add_sql", "add_chart"]
    assert turn["usage"]["input_tokens"] > 0 and turn["usage"]["cost_usd"] > 0
    sidecar = [json.loads(line) for line in artifacts_path(output).read_text().splitlines()]
    artifacts = next(line for line in sidecar if line["name"] == "sales_chart")["artifacts"][0]
    assert "amount_chart:\n    component: Lumen/view" in artifacts["graph"]
    assert artifacts["messages"][0]["parts"][0]["content"] == "Plot the total amount by category as a bar chart."


async def test_seeded_query_is_placed_before_the_prompt():
    seen = {}

    def respond(messages, info):
        count = sum(isinstance(m, ModelResponse) for m in messages)
        if count == 0:
            return ModelResponse(parts=[ToolCallPart("list_tables", {})])
        if count == 1:
            seen["tables"] = messages[-1].parts[0].content
            return ModelResponse(parts=[ToolCallPart("add_chart", {"data": "result", "name": "chart", "spec": {
                "mark": "bar", "encoding": {"x": {"field": "category", "type": "nominal"},
                                            "y": {"field": "total", "type": "quantitative"}}}})])
        return ModelResponse(parts=[TextPart("Charted.")])

    subject = FlowDashSubject(model=FunctionModel(respond))
    case = by_name("agent_vega_bar")[0]
    output = await subject.run(case.inputs, subject.build_model("x"), source(case.inputs))

    assert "- result (SQL; columns: category, total)" in seen["tables"]
    turn = output.turns[0]
    assert turn.status == "success"
    assert turn.chart_marks == ["bar"]
    assert turn.sql.startswith("SELECT category, SUM(amount)")
    assert sorted(turn.rows) == [["A", 15.0], ["B", 20.0]]


async def test_run_error_is_reported_with_usage():
    subject = FlowDashSubject(model=scripted([ToolCallPart("add_sql", {"source": "data", "name": "x", "sql": "SELECT nope FROM sales"})]))
    case = by_name("sales_total")[0]
    output = await subject.run(case.inputs, subject.build_model("x"), source(case.inputs))

    turn = output.turns[0]
    assert turn.status == "error"
    assert "retries" in turn.task_errors[0]
    assert turn.usage.input_tokens > 0


def test_session_round_trip(tmp_path):
    for name in SESSION_NAMES:
        session = load_session(SESSIONS / f"{name}.yaml")
        assert dump_session(session) == (SESSIONS / f"{name}.yaml").read_text()


def test_session_becomes_case_with_rerun_expectations():
    case = session_case(SESSIONS / "paid_by_region.yaml", source)

    assert case.name == "session_paid_by_region"
    assert case.inputs.fixture == "commerce"
    assert case.inputs.prompts == ["What is the total paid order amount per region?", "Only keep the west region."]
    expected = case.metadata
    assert [sorted(rows) for rows in expected.turn_rows] == [[["east", 65.0], ["west", 40.0]], [["west", 40.0]]]
    assert expected.rows == [["west", 40.0]]
    assert expected.columns is None
    assert expected.turn_tables == ["orders", "orders"]
    assert session_case(SESSIONS / "paid_by_region.yaml", source, columns=True).metadata.columns[0] == "region"

    top = session_case(SESSIONS / "top_customer.yaml", source).metadata
    assert top.rows == [["Ada", 2]]

    chart = session_case(SESSIONS / "sales_chart.yaml", source).metadata
    assert chart.chart_mark == "bar"
    assert [sorted(rows) for rows in chart.turn_rows] == [[["A", 15.0], ["B", 20.0]]] * 2


def recorded_fingerprints(name):
    return {
        (message.metadata or {}).get("panel_agents", {}).get("fingerprint")
        for message in load_messages(SESSIONS / f"{name}.trace.jsonl") if isinstance(message, ModelResponse)
    }


STALE = (
    "The recorded sessions were made with another agent surface (fingerprints {recorded}, now {current}): "
    "the agent's instructions or tool schemas changed, possibly through an upstream panel-flowdash docstring. "
    "Record them again with `python -m tests.evals.record_sessions` and review the answers."
)


def stale_sessions():
    current = agent_fingerprint()
    recorded = set().union(*(recorded_fingerprints(name) for name in SESSION_NAMES))
    return None if recorded == {current} else STALE.format(recorded=sorted(map(str, recorded)), current=current)


def test_recorded_sessions_match_agent_surface():
    assert stale_sessions() is None, stale_sessions()


@pytest.fixture
def playback():
    if (reason := stale_sessions()) is not None:
        pytest.skip("Stale cassette, see test_recorded_sessions_match_agent_surface")
    return reason


def fixture(inputs):
    data = source(inputs)
    # Tool results show rows in the engine's order, which only one thread keeps stable.
    data._connection.execute("SET threads TO 1")
    return data


@pytest.mark.parametrize("name", SESSION_NAMES)
async def test_cassette_playback_rebuilds_recorded_graph(playback, name):
    session = load_session(SESSIONS / f"{name}.yaml")
    cassette = Cassette.from_trace(SESSIONS / f"{name}.trace.jsonl", strict=True)
    run = AgentSession(fixture(Inputs(session.questions, fixture=session.fixture)), cassette)
    try:
        turns = [await run.ask(question) for question in session.questions]
    finally:
        run.close()

    assert [turn.error for turn in turns] == [None] * len(turns)
    assert cassette.remaining == 0
    assert [turn.graph for turn in turns] == [turn.dashboard for turn in session.turns]
    assert [[*turn.nodes, *turn.changed] for turn in turns] == [turn.nodes for turn in session.turns]


@pytest.mark.parametrize("name", SESSION_NAMES)
async def test_session_scored_by_harness_on_cassette(playback, name):
    subject = FlowDashSubject(cassette=SESSIONS / f"{name}.trace.jsonl")
    dataset = Dataset[Inputs, Output, Expected](
        name="sessions", cases=[session_case(SESSIONS / f"{name}.yaml", fixture)], evaluators=[CheckResult()])
    report = await evaluate(subject.build_model("cassette"), dataset, fixture, subject=subject)

    assert not report.failures
    assert all(result.value for result in report.cases[0].assertions.values())


async def test_rerun_during_live_session_keeps_its_source():
    live = AgentSession(source(Inputs(["x"], fixture="sales")), scripted([TextPart("ok")]))
    try:
        session_case(SESSIONS / "paid_by_region.yaml", source)
        live.ops.replace_yaml(live.ops.get_dashboard())
        assert live.ops.node_output("data", "source") is live.source
    finally:
        live.close()


def test_recorded_sessions_hold_no_credentials_or_local_paths():
    for path in SESSIONS.iterdir():
        text = path.read_text()
        for needle in ("/Users/", "/home/", "/private/", "sk-", "api_key", "Bearer "):
            assert needle not in text, f"{needle!r} in {path.name}"
        assert load_messages(path) if path.suffix == ".jsonl" else load_session(path)


async def test_save_session_writes_loadable_file(tmp_path):
    model = scripted(
        [ToolCallPart("add_sql", {"source": "data", "name": "total", "sql": "SELECT SUM(amount) AS total FROM sales"})],
        [TextPart("35")],
    )
    inputs = Inputs(["Total?"])
    run = AgentSession(source(inputs), model)
    try:
        turns = [await run.ask("Total?")]
    finally:
        run.close()
    save_session(tmp_path / "total.yaml", turns, "sales")
    case = session_case(tmp_path / "total.yaml", source)
    assert case.metadata.rows == [[35.0]]
    assert isinstance(case, Case)


async def test_timed_out_case_keeps_its_usage(tmp_path):
    import asyncio as _asyncio

    async def respond(messages, info):
        if sum(isinstance(m, ModelResponse) for m in messages) == 0:
            return ModelResponse(parts=[ToolCallPart("list_tables", {})])
        await _asyncio.sleep(30)
        return ModelResponse(parts=[TextPart("late")])

    subject = FlowDashSubject(model=FunctionModel(respond, model_name="gpt-6-luna"))
    dataset = Dataset[Inputs, Output, Expected](name="timeout", cases=by_name("sales_total"), evaluators=[CheckResult()])
    output = tmp_path / "timeout.json"
    await evaluate(subject.build_model("x"), dataset, source, output, subject=subject, prompt_timeout=2)
    (case,) = json.loads(output.read_text())["cases"]
    assert case["turns"][0]["status"] == "timeout"
    assert case["usage"]["input_tokens"] > 0
    assert subject._session is None


async def test_partial_usage_belongs_to_the_current_case():
    subject = FlowDashSubject(model=scripted([TextPart("ok")]))
    case = by_name("sales_total")[0]
    await subject.run(case.inputs, subject.build_model("x"), source(case.inputs))
    assert subject.partial_usage().input_tokens > 0
    assert subject._session is None

    class Boom(Exception):
        pass

    def fail(*args, **kwargs):
        raise Boom()

    from lumen.agent import evals as agent_evals

    original = agent_evals.AgentSession
    agent_evals.AgentSession = fail
    try:
        with pytest.raises(Boom):
            await subject.run(case.inputs, subject.build_model("x"), source(case.inputs))
    finally:
        agent_evals.AgentSession = original
    assert subject.partial_usage() is None
