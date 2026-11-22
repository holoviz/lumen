"""The assistant scenario suite: fixtures, checks, the runner and replay of the recorded baseline, offline."""

import json

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import pytest

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from lumen.agent.headless import AgentSession
from lumen.agent.scenarios import (
    Scenario, Turn, agent_fingerprint, cassette_path, compress, diff,
    load_cassette, recorded_fingerprints, run_scenario, scenario_agent, seed,
    summarize, table, to_json, turn_state,
)
from lumen.agent.scenarios.__main__ import main
from lumen.agent.scenarios.catalog import (
    BY_NAME, CALIFORNIA, CAPACITY_BY_YEAR, CAPACITY_MW, PROJECTS, SCENARIOS,
    TURBINES, WIND, WIND_KPIS_LAST,
)
from lumen.agent.scenarios.checks import (
    before_views, chart_mark, charts_titled, edited_in_place, filter_on,
    holds_value, kpi_values, kpis, matches_series, max_requests,
    no_hidden_leftovers, same_row, shows_series, shows_value, width,
)
from lumen.agent.scenarios.fixtures import (
    fixture_source, penguins, quakes, states, turbines,
)

BASELINE_DIR = Path(__file__).parent / "scenarios"
BASELINE = BASELINE_DIR / "baseline.json"

GOOD_SQL = "SELECT p_year, SUM(t_cap) / 1000.0 AS capacity_mw FROM turbines GROUP BY p_year ORDER BY p_year"
BAD_SQL = "SELECT p_year, SUM(p_cap) AS capacity_mw FROM turbines GROUP BY p_year ORDER BY p_year"
LINE = {"title": "Capacity by year", "mark": "line", "encoding": {
    "x": {"field": "p_year", "type": "ordinal"}, "y": {"field": "capacity_mw", "type": "quantitative"}}}


def scripted(*turns):
    """A model that makes each list of tool calls in turn, one request each, then answers ``done``."""
    calls = [call for turn in turns for call in [*turn, None]]

    def respond(messages, info):
        count = sum(isinstance(message, ModelResponse) for message in messages)
        call = calls[count] if count < len(calls) else None
        if call is None:
            return ModelResponse(parts=[TextPart("done")])
        name, args = call
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f"call-{count}")])

    return FunctionModel(respond, model_name="gpt-6-luna")


@asynccontextmanager
async def state_after(fixture, model, prompt="x", seed_yaml=None):
    """The state of one turn of ``model``, checked while its session is still open."""
    session = AgentSession(fixture_source(fixture), model, agent=scenario_agent(model))
    try:
        if seed_yaml:
            seed(session, seed_yaml)
        before = session.graph()
        result = await session.ask(prompt)
        yield turn_state(session, result, before)
    finally:
        session.close()


def test_fixtures_are_deterministic_and_keep_their_traps():
    assert turbines().equals(turbines.__wrapped__())
    assert quakes().equals(quakes.__wrapped__())
    frame = turbines()
    # p_cap repeats on every turbine row of its project, so summing it overcounts.
    assert frame.groupby("p_name")["p_cap"].nunique().max() == 1
    assert frame["p_cap"].sum() > 5 * frame["t_cap"].sum() / 1000
    assert frame["t_state"].nunique() > 20 and set(frame["t_state"]) <= set(states()["state"])
    places = quakes()["place"]
    assert places.str.endswith(", CA").sum() > 5 * places.str.contains("California").sum() > 0
    assert str(quakes()["time"].max().date()) == "2025-12-31"
    assert len(penguins()) == 120 and penguins()["body_mass_g"].isna().sum() == 2


def test_scenarios_have_unique_names_themes_and_check_ids():
    assert len(SCENARIOS) == len(BY_NAME) >= 25
    assert {s.theme for s in SCENARIOS} == {
        "kpis", "filters", "titles", "layout", "edits", "numbers", "charts", "claims", "runaway"}
    for scenario in SCENARIOS:
        for turn in scenario.turns:
            ids = [check.id for check in turn.checks]
            assert len(ids) == len(set(ids)), scenario.name
            assert all(check.workstream in {f"W{i}" for i in range(1, 9)} for check in turn.checks)
        for key, workstream in scenario.xfail.items():
            index, check = key.split("/")
            assert check in [c.id for c in scenario.turns[int(index) - 1].checks], (scenario.name, key)
            assert workstream.startswith("W")


@pytest.mark.parametrize("name", sorted(s.name for s in SCENARIOS if s.seed))
async def test_seed_dashboards_load(name):
    async with state_after(BY_NAME[name].fixture, scripted(), seed_yaml=BY_NAME[name].seed) as state:
        assert state.error is None
        shown = [n for n in state.nodes if state.visible(n)]
        assert shown and all(state.data(n) is not None for n in shown)


def test_number_matching_accepts_units_and_partial_rows():
    expected = pd.DataFrame({"year": [2000, 2001, 2002], "mw": [1.5, 2.0, 3.25]})
    assert matches_series(pd.DataFrame({"Year": [2001, 2000], "GW": [0.002, 0.0015]}), expected)
    assert matches_series(pd.DataFrame({"Year": ["2000", "2001"], "MW": [1.5, 2.0]}), expected)
    assert not matches_series(pd.DataFrame({"Year": [2000, 2001], "MW": [1.5, 24.0]}), expected)
    assert not matches_series(pd.DataFrame({"State": ["TX"], "MW": [1.5]}), expected)
    assert holds_value(pd.DataFrame({"n": [242]}), 242, scales=(1.0,))
    assert not holds_value(pd.DataFrame({"n": [21]}), 242, scales=(1.0,))
    assert holds_value(pd.DataFrame({"share": [27.4]}), 0.274, scales=(1.0, 100.0))


@pytest.mark.parametrize(("sql", "expected"), [(GOOD_SQL, True), (BAD_SQL, False)], ids=["t_cap", "p_cap"])
async def test_capacity_must_not_sum_the_repeated_project_column(sql, expected):
    async with state_after("turbines", scripted([
        ("add_sql", {"source": "data", "name": "capacity_by_year", "sql": sql}),
        ("add_chart", {"data": "capacity_by_year", "name": "capacity_chart", "spec": LINE}),
    ])) as state:
        assert state.error is None
        assert shows_series(CAPACITY_BY_YEAR).run(state)[0] is expected
        assert charts_titled().run(state) == (True, "")
        assert no_hidden_leftovers().run(state) == (True, "")
        assert state.requests == 3 and state.refusals == 0


async def test_answer_left_in_a_hidden_node_is_a_leftover_but_holds_the_value():
    async with state_after("quakes", scripted([
        ("add_sql", {"source": "data", "name": "ca", "sql": "SELECT COUNT(*) AS n FROM quakes WHERE place LIKE '%, CA'"}),
    ])) as state:
        passed, detail = no_hidden_leftovers().run(state)
        assert not passed and "ca" in detail
        assert not shows_value(CALIFORNIA, scales=(1.0,)).run(state)[0]
        assert shows_value("SELECT COUNT(*) FROM quakes WHERE place LIKE '%, CA'", scales=(1.0,)).run(state)[0]


async def test_kpi_checks_on_seeded_kpis():
    async with state_after("turbines", scripted(), seed_yaml=WIND_KPIS_LAST) as state:
        assert kpis(3, top=False).run(state) == (True, "")
        assert kpi_values(TURBINES, PROJECTS, CAPACITY_MW).run(state) == (True, "")
        assert not kpis(3).run(state)[0]
        assert not before_views("kpi").run(state)[0]


async def test_kpis_moved_to_the_top_row():
    async with state_after("turbines", scripted([
        *[("place_tile", {"name": name, "index": i, "width": 33.33}) for i, name in
          enumerate(["kpi_turbines", "kpi_projects", "kpi_capacity"])],
    ]), seed_yaml=WIND_KPIS_LAST) as moved:
        assert moved.grid[:3] == ["kpi_turbines", "kpi_projects", "kpi_capacity"]
        assert kpis(3).run(moved) == (True, "")
        assert before_views("kpi").run(moved) == (True, "")
        assert same_row("kpi_turbines", "kpi_projects", "kpi_capacity").run(moved) == (True, "")
        assert width("kpi_turbines", 33.33).run(moved) == (True, "")


async def test_text_mark_charts_are_not_kpis():
    async with state_after("turbines", scripted([
        ("add_sql", {"source": "data", "name": "totals", "sql": "SELECT COUNT(*) AS turbines FROM turbines"}),
        ("add_chart", {"data": "totals", "name": "kpi_card", "spec": {
            "mark": {"type": "text"}, "encoding": {"text": {"field": "turbines", "type": "quantitative"}}}}),
    ])) as state:
        passed, detail = kpis(1, top=False).run(state)
        assert not passed and "kpi_card" in detail


LINE_BY_STATE = {"mark": {"type": "line"}, "encoding": {
    "x": {"field": "t_state", "type": "nominal"}, "y": {"field": "capacity_mw", "type": "quantitative"}}}


async def test_edit_in_place():
    async with state_after("turbines", scripted([
        ("set_config", {"name": "state_capacity_chart", "config": {"spec": LINE_BY_STATE}}),
    ]), seed_yaml=WIND) as edited:
        assert edited_in_place("state_capacity_chart").run(edited) == (True, "")
        assert chart_mark("line", "state_capacity_chart").run(edited) == (True, "")


async def test_remove_and_recreate_is_not_an_edit_in_place():
    async with state_after("turbines", scripted([
        ("remove_node", {"name": "state_capacity_chart"}),
        ("add_chart", {"data": "state_capacity", "name": "state_capacity_line", "spec": LINE_BY_STATE}),
    ]), seed_yaml=WIND) as recreated:
        passed, detail = edited_in_place("state_capacity_chart").run(recreated)
        assert not passed and "state_capacity_line" in detail
        assert recreated.removed == ["state_capacity_chart"]


async def test_filter_check_reads_the_global_filter_bar():
    async with state_after("turbines", scripted(), seed_yaml=WIND) as state:
        assert not filter_on("t_state").run(state)[0]
        state.graph["filters"] = [{"column": "t_state", "widget": "select"}]
        assert filter_on("t_state").run(state) == (True, "")


async def test_refusals_and_requests_are_counted():
    async with state_after("turbines", scripted([
        ("add_chart", {"data": "nope", "name": "x", "spec": {"mark": "bar"}}),
        ("place_tile", {"name": "state_capacity_chart", "area": "top"}),
    ]), seed_yaml=WIND) as state:
        assert state.requests == 3
        assert state.refusals == 2
        assert max_requests(2).run(state)[0] is False


async def test_run_scenario_scores_turns_and_reports(tmp_path):
    scenario = Scenario("demo", "numbers", "turbines", playtest="test", turns=[
        Turn("Chart capacity by year.", [shows_series(CAPACITY_BY_YEAR), charts_titled()]),
        Turn("Now the wrong way.", [shows_series(CAPACITY_BY_YEAR, where="turn"), max_requests(1)]),
    ], xfail={"2/numbers": "W4"})
    model = scripted(
        [("add_sql", {"source": "data", "name": "capacity_by_year", "sql": GOOD_SQL}),
         ("add_chart", {"data": "capacity_by_year", "name": "capacity_chart", "spec": LINE})],
        [("add_sql", {"source": "data", "name": "capacity_wrong", "sql": BAD_SQL})],
    )
    result = await run_scenario(scenario, model)
    assert result.error is None
    assert [t.verdict for t in result.turns] == ["did", "not"]
    assert result.turns[1].checks[0].xfail == "W4"
    summary = summarize([result])
    assert summary["total"] == {**summary["total"], "turns": 2, "did": 1, "partly": 0, "not": 1}
    assert summary["themes"]["numbers"]["requests_per_turn"] == 2.5
    text = table(summary)
    assert "demo" in text and "total" in text.splitlines()[-1]
    report = to_json([result], mode="test")
    json.dumps(report)
    other = json.loads(json.dumps(report))
    other["results"][0]["turns"][1]["verdict"] = "did"
    other["results"][0]["turns"][1]["checks"][1]["passed"] = True
    assert diff(report, other) == ["demo t2: not -> did (requests pass)"]


async def test_recorded_cassette_replays_the_same_verdicts(tmp_path):
    scenario = Scenario("demo", "edits", "turbines", seed=WIND, turns=[
        Turn("Make it a line chart.", [chart_mark("line", "state_capacity_chart"), edited_in_place("state_capacity_chart")]),
    ])
    model = scripted([("set_config", {"name": "state_capacity_chart", "config": {"spec": {
        "mark": "line", "encoding": {"x": {"field": "t_state", "type": "nominal"},
                                     "y": {"field": "capacity_mw", "type": "quantitative"}}}}})])
    trace = tmp_path / "demo.trace.jsonl"
    recorded = await run_scenario(scenario, model, trace=trace)
    compress(trace, cassette_path(tmp_path, "demo"))
    assert recorded_fingerprints(cassette_path(tmp_path, "demo")) == {agent_fingerprint()}
    cassette = load_cassette(cassette_path(tmp_path, "demo"))
    replayed = await run_scenario(scenario, cassette)
    assert cassette.remaining == 0
    assert [t.verdict for t in replayed.turns] == [t.verdict for t in recorded.turns] == ["did"]
    assert replayed.turns[0].requests == recorded.turns[0].requests == 2


def test_cli_replays_cassettes_and_writes_json(tmp_path, capsys, monkeypatch):
    import asyncio

    scenario = BY_NAME["line_chart"]
    model = scripted([("set_config", {"name": "turbines_per_year_chart", "config": {"spec": {
        "title": "Turbines per year", "mark": "line", "encoding": {
            "x": {"field": "p_year", "type": "ordinal"}, "y": {"field": "turbines", "type": "quantitative"}}}}})])
    trace = tmp_path / "line_chart.trace.jsonl"
    asyncio.run(run_scenario(scenario, model, trace=trace))
    compress(trace, cassette_path(tmp_path, "line_chart"))
    out = tmp_path / "run.json"
    main(["--cassettes", str(tmp_path), "--scenario", "line_chart", "--themes", "--output", str(out)])
    printed = capsys.readouterr().out
    assert "line_chart" in printed and "edits" in printed
    report = json.loads(out.read_text())
    assert report["mode"] == "replay"
    assert report["summary"]["total"]["did"] == 1
    main(["--cassettes", str(tmp_path), "--scenario", "line_chart", "--diff", str(out)])
    assert "No verdicts changed." in capsys.readouterr().out


# -- the recorded baseline -------------------------------------------------------------------------

RECORDED = sorted(path.name.split(".")[0] for path in BASELINE_DIR.glob("*.trace.jsonl.gz"))


def stale(name: str) -> bool:
    return recorded_fingerprints(cassette_path(BASELINE_DIR, name)) != {agent_fingerprint()}


_REPLAYS: dict[str, object] = {}


async def replay(name: str):
    if name not in _REPLAYS:
        _REPLAYS[name] = await run_scenario(BY_NAME[name], load_cassette(cassette_path(BASELINE_DIR, name)))
    return _REPLAYS[name]


def test_every_scenario_has_a_recorded_baseline():
    assert RECORDED == sorted(BY_NAME)
    report = json.loads(BASELINE.read_text())
    assert report["model"] == "openai/gpt-6-luna"
    assert sorted(r["name"] for r in report["results"]) == sorted(BY_NAME)


CHECKS = [
    (scenario.name, index, check.id)
    for scenario in SCENARIOS if scenario.name in RECORDED
    for index, turn in enumerate(scenario.turns, 1) for check in turn.checks
]


@pytest.mark.parametrize(("name", "turn", "check"), CHECKS, ids=[f"{n}-t{t}-{c}" for n, t, c in CHECKS])
async def test_recorded_baseline(request, name, turn, check):
    """
    Each check of the recorded baseline, replayed: it passes, or it is
    expected to fail until its workstream lands, and then reports XPASS.
    """
    if stale(name):
        pytest.skip("Stale cassette: the agent's instructions or tools changed; record the scenarios again.")
    result = await replay(name)
    assert result.error is None, result.error
    outcome = next(c for c in result.turns[turn - 1].checks if c.id == check)
    if workstream := BY_NAME[name].xfail.get(f"{turn}/{check}"):
        request.applymarker(pytest.mark.xfail(reason=f"fixed by {workstream}", strict=False))
    assert outcome.passed, outcome.detail


def test_recorded_scenarios_hold_no_credentials_or_local_paths():
    import gzip

    for path in BASELINE_DIR.iterdir():
        text = gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()
        assert "sk-or-" not in text and "/Users/" not in text and "/home/" not in text, path.name
