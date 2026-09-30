"""Decision-model sites inside agents: projection pruning, cleanup gating and validation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from lumen.ai.agents import SQLAgent, ValidationAgent
from lumen.ai.agents.sql import drop_projections
from lumen.ai.coordinator import Coordinator
from lumen.ai.decisions import DecisionResult, Jev
from lumen.ai.tool_trace import DecisionCall
from lumen.sources.duckdb import DuckDBSource

MESSAGES = [{"role": "user", "content": "Which bond type is the most common?"}]


@pytest.fixture
def jev(monkeypatch):
    model = Jev(api_key="test-key")
    invoke = AsyncMock()
    monkeypatch.setattr(Jev, "invoke", invoke)
    return model, invoke


def nouls(site, values):
    return DecisionResult(model="fake", answers={
        f"{site}:{key}": {"type": "noul", "noul": p} for key, p in values.items()
    }, usage={})


class Step:
    def __init__(self):
        self.text = ""

    def stream(self, text):
        self.text += text


@pytest.mark.parametrize(("sql", "drop", "expected"), [
    ('SELECT "t", COUNT(*) AS "n" FROM b GROUP BY "t" ORDER BY "n" DESC LIMIT 1', {1},
     'SELECT "t" FROM b GROUP BY "t" ORDER BY COUNT(*) DESC LIMIT 1'),
    ("WITH x AS (SELECT 1 AS a, 2 AS b) SELECT a, b AS bb FROM x ORDER BY bb", {1},
     "WITH x AS (SELECT 1 AS a, 2 AS b) SELECT a FROM x ORDER BY b"),
    ("SELECT DISTINCT a, b FROM t", {1}, None),
    ("SELECT a, b FROM t ORDER BY 2", {1}, None),
    ("SELECT * FROM t", {0}, None),
    ("SELECT a FROM t UNION SELECT b FROM u", {0}, None),
    ("SELECT a, b FROM t", {0, 1}, None),
])
def test_drop_projections(sql, drop, expected):
    assert drop_projections(sql, "duckdb", drop) == expected


def test_drop_projections_requires_matching_column_count():
    assert drop_projections("SELECT a, b FROM t", "duckdb", {1}, n_columns=3) is None


async def test_projection_drops_confident_no_and_keeps_the_rest(llm, jev):
    model, invoke = jev
    invoke.return_value = nouls("sql.projection", {"0": 0.95, "1": 0.05})
    source = DuckDBSource(tables={"bond": "SELECT * FROM (VALUES ('-'), ('-'), ('=')) AS t(bond_type)"})
    agent = SQLAgent(llm=llm, decision_model=model)
    sql = 'SELECT "bond_type", COUNT(*) AS "count" FROM bond GROUP BY "bond_type" ORDER BY "count" DESC LIMIT 1'
    preview = source.execute(sql)
    step = Step()

    pruned, frame = await agent._prune_projection(MESSAGES, sql, preview, source, step)

    assert "count" not in pruned.split("FROM")[0].lower()
    assert list(pd.DataFrame(frame).columns) == ["bond_type"]
    assert "`count`" in step.text
    state, questions = invoke.await_args.args
    assert [c["name"] for c in state["result_columns"]] == ["bond_type", "count"]
    assert set(questions) == {"sql.projection:0", "sql.projection:1"}


@pytest.mark.parametrize("noul", [0.3, 0.9])
async def test_projection_keeps_query_unless_confident_no(llm, jev, noul):
    model, invoke = jev
    invoke.return_value = nouls("sql.projection", {"0": 0.99, "1": noul})
    source = DuckDBSource(tables={"t": "SELECT * FROM (VALUES (1, 2)) AS t(a, b)"})
    agent = SQLAgent(llm=llm, decision_model=model)
    preview = source.execute("SELECT a, b FROM t")

    pruned, _ = await agent._prune_projection(MESSAGES, "SELECT a, b FROM t", preview, source, Step())

    assert pruned == "SELECT a, b FROM t"


async def test_projection_is_skipped_without_a_model(llm):
    source = DuckDBSource(tables={"t": "SELECT * FROM (VALUES (1, 2)) AS t(a, b)"})
    preview = source.execute("SELECT a, b FROM t")
    assert await SQLAgent(llm=llm)._prune_projection(MESSAGES, "SELECT a, b FROM t", preview, source, Step()) == (
        "SELECT a, b FROM t", preview
    )


async def test_cleanup_gate_drops_only_confidently_irrelevant_findings(llm, jev):
    model, invoke = jev
    invoke.return_value = nouls("sql.cleanup_gate", {"0": 0.02, "1": 0.6})
    agent = SQLAgent(llm=llm, decision_model=model)
    step = Step()

    kept = await agent._gate_cleanup(MESSAGES, "SELECT 1", ["padded names", "placeholder -9999"], step)

    assert kept == ["placeholder -9999"]
    assert "Skipped 1" in step.text


async def test_cleanup_gate_keeps_findings_when_the_provider_fails(llm, jev):
    model, invoke = jev
    invoke.side_effect = RuntimeError("down")
    agent = SQLAgent(llm=llm, decision_model=model)
    assert await agent._gate_cleanup(MESSAGES, "SELECT 1", ["padded names"], Step()) == ["padded names"]


@pytest.mark.parametrize(("noul", "skipped"), [(0.99, True), (0.6, False), (0.01, False)])
async def test_validation_skips_llm_only_on_confident_complete(llm, jev, monkeypatch, noul, skipped):
    model, invoke = jev
    invoke.return_value = DecisionResult(model="fake", answers={"validation.complete": {"type": "noul", "noul": noul}}, usage={})
    agent = ValidationAgent(llm=llm, decision_model=model)
    prompt = AsyncMock(return_value=SimpleNamespace(correct=True, chain_of_thought="ok", missing_elements=[], suggestions=[]))
    monkeypatch.setattr(agent, "_invoke_prompt", prompt)

    [result], _ = await agent.respond(MESSAGES, {"sql": "SELECT 1", "data": "1 row"})

    assert result.correct
    assert prompt.await_count == (not skipped)
    state, _ = invoke.await_args.args
    assert state == {"user_request": MESSAGES[0]["content"], "sql": "SELECT 1", "data": "1 row"}


def test_coordinator_shares_its_decision_model_with_agents(llm, jev):
    model, _ = jev
    coordinator = Coordinator(llm=llm, agents=[SQLAgent], decision_model=model, decision_thresholds={"sql.projection": 0.6})
    [agent] = coordinator.agents
    assert agent.decision_model is model
    assert agent.decision_thresholds == {"sql.projection": 0.6}


async def test_batched_decisions_are_traced_per_question(llm, jev):
    model, invoke = jev
    invoke.return_value = nouls("sql.cleanup_gate", {"0": 0.01, "1": 0.99})
    agent = SQLAgent(llm=llm, decision_model=model)

    with llm.trace() as events:
        await agent._gate_cleanup(MESSAGES, "SELECT 1", ["a", "b"], Step())

    calls = [e for e in events if isinstance(e, DecisionCall)]
    assert [(c.site, c.route, c.value) for c in calls] == [
        ("sql.cleanup_gate:0", "accepted", False), ("sql.cleanup_gate:1", "accepted", True),
    ]


@pytest.mark.parametrize(("noul", "plausible"), [(0.99, True), (0.7, False), (0.01, False)])
async def test_empty_result_plausible_only_on_confident_yes(llm, jev, noul, plausible):
    model, invoke = jev
    invoke.return_value = DecisionResult(model="fake", answers={"sql.empty_result": {"type": "noul", "noul": noul}}, usage={})
    agent = SQLAgent(llm=llm, decision_model=model)
    assert await agent._empty_result_plausible("Paid orders above 1000?", "SELECT * FROM orders WHERE amount > 1000") is plausible


async def test_empty_result_is_never_plausible_without_a_model(llm):
    assert await SQLAgent(llm=llm)._empty_result_plausible("Paid orders above 1000?", "SELECT 1") is False


async def test_validation_state_includes_the_chart_spec(llm, jev, monkeypatch):
    model, invoke = jev
    invoke.return_value = DecisionResult(model="fake", answers={"validation.complete": {"type": "noul", "noul": 0.99}}, usage={})
    agent = ValidationAgent(llm=llm, decision_model=model)
    monkeypatch.setattr(agent, "_invoke_prompt", AsyncMock())

    await agent.respond(MESSAGES, {"view": {"mark": "bar", "encoding": {"x": {"field": "region"}}}})

    state, _ = invoke.await_args.args
    assert '"mark": "bar"' in state["view"]
