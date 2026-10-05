"""Focused tests for optional decision-model routing in coordinators."""

import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lumen.ai.agents import SQLAgent
from lumen.ai.coordinator import Coordinator, Planner
from lumen.ai.decisions import (
    Choice, DecisionResult, Jev, Noul,
)
from lumen.ai.models import FollowUpClassification, ThinkingYesNo
from lumen.ai.tool_trace import DecisionCall
from lumen.ai.ui import UI

MESSAGES = [{"role": "user", "content": "Show the latest sales"}]


@pytest.fixture
def decision_model(monkeypatch):
    model = Jev(api_key="private-provider-key")
    invoke = AsyncMock()
    monkeypatch.setattr(Jev, "invoke", invoke)
    return model, invoke


def answer(site, payload):
    return DecisionResult(model="fake", answers={site: payload}, usage={})


async def test_no_model_uses_existing_llm_path(llm, monkeypatch):
    """An unconfigured planner still uses the original follow-up and clarification prompts."""
    planner = Planner(llm=llm, agents=[], planner_tools=[])
    prompt = AsyncMock(side_effect=[
        FollowUpClassification(follow_up_type="direct", chain_of_thought="Cached data"),
        ThinkingYesNo(yes=True, chain_of_thought="Ambiguous"),
    ])
    monkeypatch.setattr(planner, "_invoke_prompt", prompt)

    assert await planner._check_follow_up_question(MESSAGES, {"data": "cached"}) == "direct"
    assert await planner._check_clarification_needed(MESSAGES, {}) is True
    assert [call.args[0] for call in prompt.await_args_list] == ["follow_up", "clarification_check"]


@pytest.mark.parametrize(("confidence", "expected"), [
    (0.90, "direct"), (0.89, "fallback"),
])
async def test_choice_confidence_boundary(llm, decision_model, confidence, expected):
    """Accept a valid choice at the default threshold, otherwise use the LLM."""
    model, invoke = decision_model
    invoke.return_value = answer("planner.follow_up", {
        "type": "choice", "choice": "direct", "confidence": confidence,
        "probabilities": {"direct": 0.91, "derived": 0.04, "new": 0.05},
    })
    planner = Planner(llm=llm, agents=[], planner_tools=[], decision_model=model)
    fallback = AsyncMock(return_value="fallback")

    result = await planner._decide_or_fallback(
        "planner.follow_up", {"user_request": "Show sales"},
        Choice(instructions="Classify", criteria={"direct": "cached", "derived": "filter", "new": "fetch"}), fallback,
    )

    assert result == expected
    assert fallback.await_count == (expected == "fallback")
    assert invoke.await_args.kwargs == {}


@pytest.mark.parametrize(("noul", "expected"), [
    (0.99, True), (0.01, False), (0.95, True), (0.05, False), (0.949, "fallback"), (0.051, "fallback"),
])
async def test_noul_certainty_is_symmetric(llm, decision_model, noul, expected):
    """Strong yes and no responses are equally actionable; uncertain ones fall back."""
    model, invoke = decision_model
    invoke.return_value = answer("planner.clarification", {"type": "noul", "noul": noul})
    planner = Planner(llm=llm, agents=[], planner_tools=[], decision_model=model)
    fallback = AsyncMock(return_value="fallback")

    result = await planner._decide_or_fallback("planner.clarification", {}, Noul(instructions="Clarify?"), fallback)

    assert result == expected
    assert fallback.await_count == (expected == "fallback")


@pytest.mark.parametrize("payload", [
    {"type": "choice", "choice": "unknown", "confidence": 1, "probabilities": {"direct": 0, "new": 1}},
    {"type": "choice", "choice": "direct", "confidence": 1, "probabilities": {"direct": 1}},
    {"type": "choice", "choice": "direct", "confidence": 1, "probabilities": {"direct": 2, "new": -1}},
    {"type": "noul", "noul": 1},
])
async def test_invalid_choice_falls_back(llm, decision_model, payload):
    """Reject unknown options, incomplete or invalid distributions, and wrong answer types."""
    model, invoke = decision_model
    invoke.return_value = answer("route", payload)
    coordinator = Coordinator(llm=llm, agents=[], decision_model=model)
    fallback = AsyncMock(return_value="LLM route")

    assert await coordinator._decide_or_fallback(
        "route", {}, Choice(instructions="Route", criteria={"direct": "cached", "new": "fetch"}), fallback,
    ) == "LLM route"
    fallback.assert_awaited_once()


async def test_site_threshold_override(llm, decision_model):
    """Per-site thresholds override the default without affecting other sites."""
    model, invoke = decision_model
    invoke.return_value = answer("route", {"type": "noul", "noul": 0.8})
    coordinator = Coordinator(llm=llm, agents=[], decision_model=model, decision_thresholds={"route": 0.6})
    fallback = AsyncMock(return_value="fallback")

    assert await coordinator._decide_or_fallback("route", {}, Noul(instructions="Relevant?"), fallback) is True
    fallback.assert_not_awaited()


async def test_provider_failure_falls_back_but_cancellation_propagates(llm, decision_model):
    """Provider failures use the LLM; task cancellation must never trigger a new request."""
    model, invoke = decision_model
    coordinator = Coordinator(llm=llm, agents=[], decision_model=model)
    fallback = AsyncMock(return_value=False)
    invoke.side_effect = RuntimeError("provider unavailable")

    assert await coordinator._decide_or_fallback("route", {}, Noul(instructions="Relevant?"), fallback) is False
    fallback.assert_awaited_once()

    fallback.reset_mock()
    invoke.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await coordinator._decide_or_fallback("route", {}, Noul(instructions="Relevant?"), fallback)
    fallback.assert_not_awaited()


async def test_stalled_provider_times_out_and_falls_back(llm, decision_model):
    """A provider that hangs costs the decision timeout, not the provider's request timeout."""
    model, invoke = decision_model

    async def stall(*args, **kwargs):
        await asyncio.sleep(60)

    invoke.side_effect = stall
    coordinator = Coordinator(llm=llm, agents=[], decision_model=model, decision_timeout=0.05)
    fallback = AsyncMock(return_value=False)

    with llm.trace() as events:
        assert await asyncio.wait_for(
            coordinator._decide_or_fallback("route", {}, Noul(instructions="Relevant?"), fallback), timeout=5
        ) is False
    fallback.assert_awaited_once()
    assert [(e.site, e.route) for e in events if isinstance(e, DecisionCall)] == [("route", "error")]


def test_coordinator_shares_decision_timeout_with_agents(llm, decision_model):
    model, _ = decision_model
    coordinator = Coordinator(llm=llm, agents=[SQLAgent], decision_model=model, decision_timeout=2)
    assert coordinator.agents[0].decision_timeout == 2


@pytest.mark.parametrize(("route", "pipeline_kept"), [("new", False), ("derived", True)])
async def test_follow_up_routes_and_clears_pipeline(llm, decision_model, monkeypatch, route, pipeline_kept):
    """A new data request invalidates the pipeline; derived follow-ups preserve it."""
    model, invoke = decision_model
    invoke.return_value = answer("planner.follow_up", {
        "type": "choice", "choice": route, "confidence": 0.98,
        "probabilities": {"direct": 0.01, "derived": 0.01 if route == "new" else 0.98, "new": 0.98 if route == "new" else 0.01},
    })
    planner = Planner(llm=llm, agents=[], planner_tools=[], decision_model=model)
    prompt = AsyncMock()
    monkeypatch.setattr(planner, "_invoke_prompt", prompt)
    pipeline = object()
    source = SimpleNamespace(metadata={
        "weekly": {"source_action": "fetch_weather"},
        "sales_by_region": {"derived_from": ["orders"], "created_order": 1},
    })
    context = {
        "data": "cached sales", "pipeline": pipeline, "sql": "SELECT sales FROM orders",
        "source": source, "api_key": "private-context-key",
    }

    assert await planner._check_follow_up_question(MESSAGES, context) == route
    assert ("pipeline" in context) is pipeline_kept
    if pipeline_kept:
        assert context["pipeline"] is pipeline
    state, questions = invoke.await_args.args
    assert set(questions) == {"planner.follow_up"}
    assert set(questions["planner.follow_up"].criteria) == {"direct", "derived", "new"}
    assert state["user_request"] == MESSAGES[0]["content"]
    assert state["external_api_tables"] == ["weekly"]
    assert state["derived_tables"] == ["sales_by_region"]
    assert "private-context-key" not in str(state)
    assert "private-provider-key" not in str(state)
    prompt.assert_not_awaited()


async def test_follow_up_without_data_skips_both_models(llm, decision_model, monkeypatch):
    """There is no follow-up to classify when no data is cached."""
    model, invoke = decision_model
    planner = Planner(llm=llm, agents=[], planner_tools=[], decision_model=model)
    prompt = AsyncMock()
    monkeypatch.setattr(planner, "_invoke_prompt", prompt)

    assert await planner._check_follow_up_question(MESSAGES, {}) == "new"
    invoke.assert_not_awaited()
    prompt.assert_not_awaited()


@pytest.mark.parametrize(("noul", "expected"), [(0.98, True), (0.02, False), (0.5, True)])
async def test_clarification_routes_or_falls_back(llm, decision_model, monkeypatch, noul, expected):
    """Clarification uses a confident decision and consults the LLM if uncertain."""
    model, invoke = decision_model
    invoke.return_value = answer("planner.clarification", {"type": "noul", "noul": noul})
    planner = Planner(llm=llm, agents=[], planner_tools=[], decision_model=model)
    prompt = AsyncMock(return_value=ThinkingYesNo(yes=True, chain_of_thought="Unclear"))
    monkeypatch.setattr(planner, "_invoke_prompt", prompt)

    assert await planner._check_clarification_needed(MESSAGES, {"password": "private-context-key"}) is expected
    state, questions = invoke.await_args.args
    assert state == {"user_request": "Show the latest sales"}
    assert isinstance(questions["planner.clarification"], Noul)
    assert prompt.await_count == (noul == 0.5)


@pytest.mark.parametrize(("follow_up_type", "decided"), [("direct", False), ("derived", False), ("new", True), (None, True)])
async def test_clarification_of_follow_ups_uses_the_llm(llm, decision_model, monkeypatch, follow_up_type, decided):
    """Follow-ups like "now by month" depend on earlier turns, which only the LLM sees."""
    model, invoke = decision_model
    invoke.return_value = answer("planner.clarification", {"type": "noul", "noul": 0.02})
    planner = Planner(llm=llm, agents=[], planner_tools=[], decision_model=model)
    prompt = AsyncMock(return_value=ThinkingYesNo(yes=False, chain_of_thought="Clear in context"))
    monkeypatch.setattr(planner, "_invoke_prompt", prompt)

    assert await planner._check_clarification_needed(MESSAGES, {}, follow_up_type) is False
    assert invoke.await_count == decided
    assert prompt.await_count == (not decided)


async def test_pre_plan_passes_follow_up_type_to_clarification(llm, monkeypatch):
    planner = Planner(llm=llm, agents=[], planner_tools=[])
    monkeypatch.setattr(planner, "_check_follow_up_question", AsyncMock(return_value="derived"))
    clarification = AsyncMock(return_value=False)
    monkeypatch.setattr(planner, "_check_clarification_needed", clarification)
    monkeypatch.setattr(planner, "_execute_planner_tools", AsyncMock())

    await planner._pre_plan(MESSAGES, {}, {}, {})

    assert clarification.await_args.args[2] == "derived"


@pytest.mark.parametrize(("noul", "expected"), [(0.97, True), (0.03, False), (0.5, True)])
async def test_tool_relevance_routes_or_falls_back(llm, decision_model, monkeypatch, noul, expected):
    """Tool relevance includes task and output, but not unrelated credentials."""
    model, invoke = decision_model
    invoke.return_value = answer("coordinator.tool_relevance", {"type": "noul", "noul": noul})
    coordinator = Coordinator(llm=llm, agents=[], decision_model=model)
    prompt = AsyncMock(return_value=ThinkingYesNo(yes=True, chain_of_thought="Useful"))
    monkeypatch.setattr(coordinator, "_invoke_prompt", prompt)
    tool = SimpleNamespace(name="Lookup", purpose="Find sales")
    actor = SimpleNamespace(name="Analyst", purpose="Analyze sales")

    assert await coordinator._check_tool_relevance(
        tool, "Sales table found", actor, "Analyze the latest sales", MESSAGES, {"password": "private-context-key"},
    ) is expected
    state, questions = invoke.await_args.args
    assert state["task"] == "Analyze the latest sales"
    assert state["tool_output"] == "Sales table found"
    assert state["user_request"] == "Show the latest sales"
    assert isinstance(questions["coordinator.tool_relevance"], Noul)
    assert "private-context-key" not in str(state)
    assert prompt.await_count == (noul == 0.5)


def test_ui_passes_decision_configuration_to_coordinator(llm, decision_model, monkeypatch):
    """UI configuration reaches the planner without embedding credentials in thresholds."""
    model, _ = decision_model
    monkeypatch.setattr(UI, "_configure_session", lambda self: None)
    monkeypatch.setattr(UI, "_render_page", lambda self: None)
    ui = UI(llm=llm, default_agents=[], agents=[], tools=[], source_controls=[],
            decision_model=model, decision_thresholds={"planner.follow_up": 0.75})

    assert ui._coordinator.decision_model is model
    assert ui._coordinator.decision_thresholds == {"planner.follow_up": 0.75}


@pytest.mark.parametrize(("noul", "route"), [(0.99, "accepted"), (0.6, "fallback")])
async def test_decisions_are_recorded_on_the_llm_trace(llm, decision_model, noul, route):
    """Evals see which route answered each decision alongside the LLM calls."""
    model, invoke = decision_model
    invoke.return_value = DecisionResult(model="fake", answers={"route": {"type": "noul", "noul": noul}}, usage={"cost": 0.00001})
    coordinator = Coordinator(llm=llm, agents=[], decision_model=model)
    fallback = AsyncMock(return_value=False)

    with llm.trace() as events:
        await coordinator._decide_or_fallback("route", {}, Noul(instructions="Relevant?"), fallback)

    [call] = [event for event in events if isinstance(event, DecisionCall)]
    assert (call.site, call.route, call.value) == ("route", route, True)
    assert call.certainty == pytest.approx(2 * abs(noul - 0.5))
    assert call.usage == {"cost": 0.00001}
    assert call.fallback_value == (False if route == "fallback" else None)


async def test_clarification_state_includes_available_tables(llm, decision_model, monkeypatch):
    """The table list lets the decision model see when the data settles an ambiguity."""
    model, invoke = decision_model
    invoke.return_value = answer("planner.clarification", {"type": "noul", "noul": 0.02})
    planner = Planner(llm=llm, agents=[], planner_tools=[], decision_model=model)
    monkeypatch.setattr(planner, "_invoke_prompt", AsyncMock())
    metaset = SimpleNamespace(table_context=lambda include_metadata: "sales: category, amount")

    assert await planner._check_clarification_needed(MESSAGES, {"metaset": metaset}) is False
    state, questions = invoke.await_args.args
    assert state == {"user_request": "Show the latest sales", "available_tables": "sales: category, amount"}
    assert set(questions["planner.clarification"].criteria) == {"true", "false"}
