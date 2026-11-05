import asyncio
import base64
import json
import logging
import subprocess
import sys
import threading
import time
import warnings

from typing import Any, get_args
from unittest.mock import AsyncMock, patch

import pandas as pd
import panel as pn
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.config import panel_extension
from panel.tests.util import async_wait_until
from panel_material_ui import Card

import lumen.ai.answer as answer_module

from lumen.ai.actor import NullStep
from lumen.ai.agents import ChatAgent, SQLAgent, VegaLiteAgent
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.answer import (
    Answer, _default_agents, _default_llm, _env_llm, _planner, _planners,
    _resolve, _run, _StatusLine, aask, ask, chat, clear_cache, configure,
)
from lumen.ai.config import QueryError, RetriesExceededError
from lumen.ai.coordinator import Plan
from lumen.ai.coordinator.planner import make_plan_model
from lumen.ai.data import read
from lumen.ai.editors import VegaLiteEditor
from lumen.ai.models import ReplaceLine, RetrySpec
from lumen.ai.report import ActorTask
from lumen.ai.usage import Usage, record_usage
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.filters.base import ConstantFilter
from lumen.pipeline import Pipeline
from lumen.tests.ai.conftest import MockLLM
from lumen.tests.utils import Polygon, gpd
from lumen.util import log as lumen_log
from lumen.views.base import VegaLiteView, hvPlotView

QUESTION = [{"role": "user", "content": "Total value?"}]


def _plan(*steps, unanswerable=None):
    PlanModel = make_plan_model(["ChatAgent", "SQLAgent"], [])
    (StepModel,) = get_args(PlanModel.__annotations__["steps"])
    return PlanModel(
        chain_of_thought="Query then explain",
        title="Total value",
        steps=[StepModel(actor=actor, instruction=title, title=title) for actor, title in steps],
        unanswerable=unanswerable,
    )


@pytest.fixture
def frame_source():
    return read({"tiny": pd.DataFrame({"category": ["A", "B", "A"], "value": [10.5, 20.0, 7.5]})})


async def test_aask_answers_with_sql_and_text(llm, frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm.set_responses([
        _plan(("SQLAgent", "Sum value"), ("ChatAgent", "Explain")),
        SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="tiny_total", tables=["tiny"]),
        "The total value is 38.",
    ])

    answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])

    assert isinstance(answer, Answer)
    assert answer.status == "success"
    assert answer.value == "The total value is 38."
    assert "SUM(value)" in answer.sql
    assert answer.pipeline.data.iloc[0, 0] == 38.0
    assert answer.view is None and answer.spec is None


@pytest.mark.parametrize("notifications", [False, True])
def test_a_chart_that_fails_to_draw_shows_its_error_in_the_card(chart_answer, notifications):
    with (
        patch.object(VegaLiteView, "get_panel", side_effect=ValueError("bad encoding")),
        pn.config.set(notifications=notifications),
    ):
        chart = chart_answer.__panel__()[1]
    assert chart.title == "The chart could not be drawn" and "bad encoding" in chart.object


@pytest.mark.parametrize("question, reason", [
    ("What is the elevation of each category?", "No table has an elevation column."),
    ("What was the value in 2030?", "The table has no dates, so no year can be picked out."),
    ("Who bought each item?", "No table records buyers."),
])
async def test_an_impossible_question_is_unanswerable_without_running(llm, frame_source, question, reason):
    llm.set_responses([_plan(unanswerable=reason)])

    answer = await aask(frame_source, question, llm=llm, agents=[ChatAgent(), SQLAgent()])

    assert answer.status == "unanswerable" and answer.error == reason
    assert llm._index == 1  # only the plan was asked for; no agent ran
    alert = answer.__panel__()[0]
    assert alert.severity == "info" and reason in alert.object
    assert reason in answer._static_bundle(None)["text/html"]


async def test_a_reason_with_steps_still_runs_the_steps(llm, frame_source):
    llm.set_responses([_plan(("ChatAgent", "Explain"), unanswerable="Unsure"), "Ten rows."])
    answer = await aask(frame_source, "How many rows?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert answer.status == "success" and answer.value == "Ten rows."


def test_a_plan_without_the_reason_parses_as_before():
    PlanModel = make_plan_model(["ChatAgent"], [])
    assert PlanModel(chain_of_thought="c", title="t", steps=[]).unanswerable is None


async def test_retrying_an_unanswerable_answer_tells_the_model_why(llm, frame_source):
    llm.set_responses([_plan(unanswerable="No table has an elevation column.")])
    answer = await aask(frame_source, "Elevation?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    with patch("lumen.ai.answer._aask_messages", new_callable=AsyncMock) as ask:
        await answer.aretry(hint="Use the value column")
    messages = ask.call_args.args[2]
    assert "No table has an elevation column." in messages[1]["content"]


async def test_a_plan_that_neither_queries_nor_answers_is_an_error(llm, frame_source):
    llm.set_responses([_plan(("ChatAgent", "Explain")), ""])

    answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])

    assert answer.status == "error"
    assert "without querying the data" in answer.error
    assert answer.sql is None and answer.view is None


def _costing(llm, response, tokens, cost):
    """A response that reports usage, as a provider does when it answers."""
    def respond():
        record_usage(llm, llm.usage, Usage("mock", tokens, tokens // 10, cost_usd=cost))
        return response
    return respond


async def test_answer_usage_sums_the_calls_it_made(llm, frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm.set_responses([
        _costing(llm, _plan(("SQLAgent", "Sum value"), ("ChatAgent", "Explain")), 1000, 0.001),
        _costing(llm, SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="tiny_total", tables=["tiny"]), 2000, 0.002),
        _costing(llm, "The total value is 38.", 500, 0.0005),
    ])
    record_usage(llm, llm.usage, Usage("mock", 9999, 9999))  # an earlier question, not this one

    answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])

    assert len(answer.usage.records) == 3
    assert (answer.usage.input_tokens, answer.usage.output_tokens) == (3500, 350)
    assert answer.usage.cost_usd == pytest.approx(0.0035)
    assert answer._usage_note == "3,500 in / 350 out tokens · $0.0035"
    assert answer.__panel__()[-1].object == answer._usage_note
    assert "3,500 in / 350 out tokens" in answer._static_bundle(None)["text/html"]


async def test_answer_usage_counts_an_agent_with_its_own_llm(llm, frame_source):
    chat_llm = MockLLM()
    chat_llm.set_responses([_costing(chat_llm, "Ten rows.", 400, 0.004)])
    llm.set_responses([_costing(llm, _plan(("ChatAgent", "Explain")), 100, 0.001)])

    answer = await aask(frame_source, "How many rows?", llm=llm, agents=[ChatAgent(llm=chat_llm), SQLAgent()])

    assert answer.usage.input_tokens == 500 and answer.usage.cost_usd == pytest.approx(0.005)


async def test_answer_usage_leaves_out_the_cost_when_a_call_is_unpriced(llm, frame_source):
    llm.set_responses([
        _costing(llm, _plan(("ChatAgent", "Explain")), 100, 0.001),
        _costing(llm, "Ten rows.", 50, None),
    ])
    answer = await aask(frame_source, "How many rows?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert answer.usage.cost_usd is None
    assert answer._usage_note == "150 in / 15 out tokens"


def test_answer_without_counted_calls_has_no_usage_line(chart_answer):
    assert chart_answer._usage_note == ""
    assert all("tokens" not in str(getattr(part, "object", "")) for part in chart_answer.__panel__())
    assert chart_answer.with_spec(chart_answer._chart_spec()[1]).usage.records == []


async def test_aask_returns_an_error_answer_when_the_plan_fails(llm, frame_source):
    def fail():
        raise RuntimeError("LLM unavailable")

    llm.set_responses([
        _plan(("ChatAgent", "Explain")),
        fail, fail, fail, fail,
    ])
    answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])

    assert answer.status == "error"
    assert answer.value == "Lumen could not answer 'Total value?': 'Explain' failed."
    assert "LLM unavailable" in answer.error
    assert repr(answer).startswith("Answer(status='error'")


def _failing_sql(fail_times=1):
    """Patch SQLAgent.respond to fail like a query that never ran, then work."""
    original = SQLAgent.respond
    calls = []

    async def respond(self, *args, **kwargs):
        calls.append(1)
        if len(calls) <= fail_times:
            raise QueryError("BinderException: column nope not found", "SELECT nope FROM tiny")
        return await original(self, *args, **kwargs)

    return patch.object(SQLAgent, "respond", new=respond)


async def test_failed_sql_is_kept_on_the_error_answer(frame_source):
    llm = FollowUpLLM([])
    with _failing_sql(fail_times=9):
        answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])

    assert answer.status == "error"
    assert answer.sql == "SELECT nope FROM tiny"
    assert "column nope not found" in answer.error
    assert answer.pipeline is None and answer.view is None


async def test_error_answer_prints_no_traceback(frame_source, capsys):
    with _failing_sql(fail_times=9):
        answer = await aask(frame_source, "Total value?", llm=FollowUpLLM([]), agents=[ChatAgent(), SQLAgent()])
    assert answer.status == "error"
    assert "Traceback" not in capsys.readouterr().err


async def test_error_card_shows_the_message_and_open_sql(frame_source):
    llm = FollowUpLLM([])
    with _failing_sql(fail_times=9):
        answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])

    alert, sql, steps = answer.__panel__()
    assert alert.severity == "error" and "column nope" in alert.object
    assert steps.title == "Steps" and steps.collapsed
    assert not sql.collapsed and "SELECT nope FROM tiny" in sql[0].object
    html = answer._static_bundle(None)["text/html"]
    assert "column nope not found" in html
    assert "<details open><summary>Failed SQL</summary>" in html


async def test_a_later_step_failing_keeps_the_working_query(frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm = FollowUpLLM([SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="total", tables=["tiny"])])

    async def respond(self, *args, **kwargs):
        raise RetriesExceededError("Maximum number of retries exceeded.") from ValueError("chart spec is invalid")

    with patch.object(ChatAgent, "respond", new=respond):
        answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])

    assert answer.status == "error"
    assert answer.error == "ValueError: chart spec is invalid"
    assert "SUM(value)" in answer.sql
    assert answer.pipeline.data.iloc[0, 0] == 38.0
    _alert, sql, _steps = answer.__panel__()
    assert sql.title == "SQL" and sql.collapsed
    assert "<details><summary>SQL</summary>" in answer._static_bundle(None)["text/html"]
    with pytest.raises(ValueError, match=r"answer\.retry"):
        await answer.arefine("Use a line")


async def test_retry_sends_the_hint_after_the_failure(frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm = FollowUpLLM([SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="total", tables=["tiny"])])
    with _failing_sql(fail_times=1):
        failed = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])
        fixed = await failed.aretry(hint="The column is called value")

    assert failed.status == "error" and fixed.status == "success"
    assert fixed.question == "Total value?"
    assert fixed.pipeline.data.iloc[0, 0] == 38.0
    history, latest = llm.plan_calls[-1]
    assert "SELECT nope FROM tiny" in history["content"]
    assert latest == {"role": "user", "content": "The column is called value"}


async def test_chat_does_not_build_on_an_error_answer(frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm = FollowUpLLM([SQLModel(query="SELECT COUNT(*) AS n FROM tiny", table_slug="n", tables=["tiny"])])
    conversation = frame_source.chat(llm=llm, agents=[ChatAgent(), SQLAgent()])
    with _failing_sql(fail_times=1):
        failed = await conversation.aask("Broken?")
        second = await conversation.aask("How many rows?")

    assert conversation.answers == [failed, second]
    assert failed.status == "error" and second.status == "success"
    assert llm.plan_calls[-1] == [{"role": "user", "content": "How many rows?"}]


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    """No test sees another's configure() call, environment or cached LLM."""
    monkeypatch.setattr(answer_module, "_settings", {"agents": None, "llm": None, "cache": False})
    monkeypatch.delenv("LUMEN_AI_PROVIDER", raising=False)
    monkeypatch.delenv("LUMEN_AI_MODEL", raising=False)
    _env_llm.cache_clear()
    clear_cache()
    max_rows = SQLAgent.max_rows
    yield
    _env_llm.cache_clear()
    # Plain assignment: monkeypatch would restore the Parameter object itself
    SQLAgent.max_rows = max_rows


async def test_aask_without_a_provider_explains_the_options(frame_source):
    with patch("lumen.ai.answer.get_available_llm", return_value=None):
        with pytest.raises(RuntimeError, match=r"No LLM provider found\. Pass llm=lumen\.ai\.llm\.Kilo"):
            await aask(frame_source, "Total value?")


def test_default_llm_is_shared_between_questions(llm):
    with patch("lumen.ai.answer.get_available_llm", return_value=type(llm)):
        assert _default_llm() is _default_llm()


async def test_configure_sets_the_llm_and_a_question_can_override_it(frame_source):
    configured, passed = MockLLM(), MockLLM()
    assert configure(llm=configured)["llm"] is configured
    _chat_only(configured)
    assert (await aask(frame_source, "How many rows?"))._llm is configured
    _chat_only(passed)
    assert (await aask(frame_source, "How many rows?", llm=passed))._llm is passed


async def test_configure_sets_the_agents(llm, frame_source):
    agents = [ChatAgent()]
    configure(agents=agents)
    _chat_only(llm)
    answer = await aask(frame_source, "How many rows?", llm=llm)
    assert answer._agents is None
    assert _planners.get(frame_source) is None  # explicit agents get their own Planner
    assert configure() == {
        "agents": agents, "cache": False, "llm": None, "mask_pii": False, "max_rows": 1_000_000, "share": "sample",
    }


async def _ask_every_row(source):
    SQLModel = make_sql_model([(source.name, "rows")])
    llm = FollowUpLLM([SQLModel(query="SELECT * FROM rows", table_slug="all_rows", tables=["rows"])])
    return await aask(source, "Show every row", llm=llm, agents=[ChatAgent(), SQLAgent()])


@pytest.fixture
def ten_rows():
    return read({"rows": pd.DataFrame({"a": range(10)})})


async def test_max_rows_caps_the_answer_data_and_says_so(ten_rows):
    configure(max_rows=4)
    answer = await _ask_every_row(ten_rows)
    assert len(answer.pipeline.data) == 4
    assert answer.truncated_at == 4
    assert "stops at the first 4 rows" in answer._static_bundle(None)["text/html"]
    assert any("stops at the first 4 rows" in getattr(part, "object", "") for part in answer.__panel__())


async def test_no_note_below_the_cap(ten_rows):
    answer = await _ask_every_row(ten_rows)
    assert len(answer.pipeline.data) == 10
    assert answer.truncated_at is None and "stops at" not in answer._text


async def test_max_rows_none_returns_every_row(ten_rows):
    configure(max_rows=4)
    configure(max_rows=None)
    answer = await _ask_every_row(ten_rows)
    assert len(answer.pipeline.data) == 10


def test_environment_picks_the_provider_and_model(monkeypatch):
    monkeypatch.setenv("LUMEN_AI_PROVIDER", "kilo")
    monkeypatch.setenv("LUMEN_AI_MODEL", "kilo-auto/free")
    llm = _default_llm()
    assert type(llm).__name__ == "Kilo"
    assert llm.model_kwargs["default"]["model"] == "kilo-auto/free"
    assert _default_llm() is llm


def test_configure_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("LUMEN_AI_PROVIDER", "kilo")
    configured = MockLLM()
    configure(llm=configured)
    assert _default_llm() is configured


def test_unknown_provider_lists_the_valid_ones(monkeypatch):
    monkeypatch.setenv("LUMEN_AI_PROVIDER", "nope")
    with pytest.raises(ValueError, match="Could not find LLM Provider 'nope', valid providers include"):
        _default_llm()


async def test_aask_reuses_one_planner_per_source(llm, frame_source):
    _chat_only(llm)
    await aask(frame_source, "How many rows?", llm=llm)
    planner = _planners[frame_source]
    _chat_only(llm)
    second = await aask(frame_source, "How many rows?", llm=llm)
    assert _planners[frame_source] is planner
    assert second.value == "Three rows."
    assert "metaset" not in planner.context


def test_planner_is_rebuilt_for_another_llm_or_explicit_agents(llm, frame_source):
    first = _planner(frame_source, llm, None)
    assert _planner(frame_source, llm, None) is first
    assert _planner(frame_source, type(llm)(), None) is not first
    assert _planner(frame_source, llm, [ChatAgent()]) is not _planner(frame_source, llm, [ChatAgent()])


def test_default_agents_plain_table(frame_source):
    agents = _default_agents(frame_source)
    assert [type(a).__name__ for a in agents] == ["ChatAgent", "SQLAgent", "VegaLiteAgent"]


def test_default_agents_use_hvplot_for_gridded_data():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_sql")
    ds = xr.Dataset({"temp": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    assert [type(a).__name__ for a in _default_agents(read(ds))] == ["ChatAgent", "SQLAgent", "hvPlotAgent"]


def test_default_agents_use_hvplot_for_geometry():
    if gpd is None:
        pytest.skip("geopandas is not installed")
    gdf = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1)])])
    assert [type(a).__name__ for a in _default_agents(read({"shapes": gdf}))] == ["ChatAgent", "SQLAgent", "hvPlotAgent"]


def test_default_agents_use_hvplot_for_lon_lat_columns():
    source = read({"quakes": pd.DataFrame({"Lat": [1.0], "lon": [2.0], "mag": [4.5]})})
    assert [type(a).__name__ for a in _default_agents(source)] == ["ChatAgent", "SQLAgent", "hvPlotAgent"]


def test_default_agents_use_hvplot_when_a_table_has_geometry(frame_source):
    with patch("lumen.ai.answer.geometry_columns", return_value=["geometry"]):
        assert [type(a).__name__ for a in _default_agents(frame_source)] == ["ChatAgent", "SQLAgent", "hvPlotAgent"]


def test_resolve_plain_pipeline_limits_to_its_table(frame_source):
    pipeline = Pipeline(source=frame_source, table="tiny")
    source, visible = _resolve(pipeline)
    assert source is frame_source
    assert visible == {f"{frame_source.name}{SOURCE_TABLE_SEPARATOR}tiny"}


def test_resolve_filtered_pipeline_queries_only_its_rows(frame_source):
    pipeline = Pipeline(source=frame_source, table="tiny", filters=[ConstantFilter(field="category", value="A")])
    source, visible = _resolve(pipeline)
    assert visible is None
    assert source.get_tables() == ["tiny"]
    assert sorted(source.get("tiny")["value"]) == [7.5, 10.5]


def _chat_only(llm):
    llm.set_responses([
        _plan(("ChatAgent", "Explain")),
        "Three rows.",
    ])


def test_ask_without_a_running_loop(llm, frame_source):
    _chat_only(llm)
    answer = ask(frame_source, "How many rows?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert answer.value == "Three rows."


def test_ask_errors_do_not_chain_the_loop_check(frame_source):
    with patch("lumen.ai.answer.aask", side_effect=ValueError("boom")):
        with pytest.raises(ValueError) as info:
            ask(frame_source, "q")
    assert info.value.__context__ is None


async def test_aask_quiets_instructor_retry_log(llm, frame_source, monkeypatch):
    logger = logging.getLogger("instructor.v2.retry")
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    # ExplorerUI tests in the same process leave Lumen's logger at DEBUG
    monkeypatch.setattr(lumen_log, "level", logging.WARNING)
    _chat_only(llm)
    await aask(frame_source, "How many rows?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert logger.level == logging.CRITICAL


async def test_ask_inside_a_running_loop(llm, frame_source):
    _chat_only(llm)
    answer = ask(frame_source, "How many rows?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert answer.value == "Three rows."


async def test_source_and_pipeline_methods_delegate(frame_source):
    pipeline = Pipeline(source=frame_source, table="tiny")
    with patch("lumen.ai.answer.ask") as sync_ask, patch("lumen.ai.answer.aask", new_callable=AsyncMock) as async_ask:
        frame_source.ask("q", llm="x", returns=int)
        pipeline.ask("q")
        await frame_source.aask("q")
        await pipeline.aask("q", llm="x")
    assert [c.args[0] for c in sync_ask.call_args_list] == [frame_source, pipeline]
    assert sync_ask.call_args_list[0].kwargs == {"llm": "x", "agents": None, "confirm": False, "returns": int}
    assert [c.args[0] for c in async_ask.call_args_list] == [frame_source, pipeline]


def test_core_imports_do_not_load_lumen_ai():
    code = "import sys, lumen, lumen.sources, lumen.pipeline; assert 'lumen.ai' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.fixture
def chart_answer(frame_source):
    view = VegaLiteView(
        pipeline=Pipeline(source=frame_source, table="tiny"),
        spec={"mark": "bar", "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        }},
    )
    return Answer(
        question="Value by category?", value="**B** is largest.\nA is second.",
        sql="SELECT * FROM tiny", spec=view.spec, view=view, status="success",
    )


@pytest.fixture
def refinable_answer(chart_answer, llm):
    """The chart answer as a run leaves it: with its editor and the Vega-Lite task."""
    agent = VegaLiteAgent(n_doc_pages=0)
    return Answer(
        editor=VegaLiteEditor(component=chart_answer.view), llm=llm,
        plan=Plan(ActorTask(agent, title="Chart", instruction="Chart"), title="Chart"),
        **{k: getattr(chart_answer, k) for k in ("question", "value", "sql", "spec", "view", "status", "source")},
    )


def _line_edit(llm, line_no, line):
    llm.set_responses([RetrySpec(chain_of_thought="Change the mark", edits=[ReplaceLine(line_no=line_no, line=line)])])


async def test_refine_changes_the_chart_with_one_llm_call(refinable_answer, llm):
    _line_edit(llm, 8, "mark: line")
    with patch.object(SQLAgent, "respond", side_effect=AssertionError("refine must not query")):
        refined = await refinable_answer.arefine("Use a line")

    assert llm._index == 1
    assert refined.spec["mark"] == "line" and refined.view.spec["mark"] == "line"
    assert refinable_answer.spec["mark"] == "bar" and refinable_answer.view.spec["mark"] == "bar"
    assert refined.sql == refinable_answer.sql and refined.question == refinable_answer.question


async def test_refine_can_be_refined_again(refinable_answer, llm):
    _line_edit(llm, 8, "mark: line")
    once = await refinable_answer.arefine("Use a line")
    _line_edit(llm, 8, "mark: point")
    twice = await once.arefine("Use points")
    assert (once.spec["mark"], twice.spec["mark"]) == ("line", "point")


def test_refine_from_sync_code(refinable_answer, llm):
    _line_edit(llm, 8, "mark: area")
    assert refinable_answer.refine("Use an area").spec["mark"] == "area"


@pytest.mark.parametrize("missing", ["spec", "editor"])
async def test_refine_needs_a_vega_lite_chart(refinable_answer, missing):
    if missing == "spec":
        refinable_answer.spec = None
    else:
        refinable_answer._editor = None
    with pytest.raises(ValueError, match="refine changes a Vega-Lite chart"):
        await refinable_answer.arefine("Use a line")


def test_answer_code_rebuilds_the_chart_without_an_llm(chart_answer):
    namespace = {}
    exec(chart_answer.code, namespace)
    view = namespace["view"]
    assert view.spec["mark"] == "bar"
    pd.testing.assert_frame_equal(view.pipeline.data, chart_answer.view.pipeline.data)


async def test_answer_code_rebuilds_the_queried_data(llm, frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm.set_responses([
        _plan(("SQLAgent", "Sum value"), ("ChatAgent", "Explain")),
        SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="tiny_total", tables=["tiny"]),
        "38.", "38.",
    ])
    answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    namespace = {}
    exec(answer.code, namespace)
    assert namespace["pipeline"].data.iloc[0, 0] == 38.0


async def test_answer_export_keeps_large_data_out_of_the_script(llm, tmp_path):
    big = read({"big": pd.DataFrame({"category": ["A", "B"] * 1000, "value": range(2000)})})
    SQLModel = make_sql_model([(big.name, "big")])
    llm.set_responses([
        _plan(("SQLAgent", "Sum value"), ("ChatAgent", "Explain")),
        SQLModel(query="SELECT category, SUM(value) AS total FROM big GROUP BY category", table_slug="totals", tables=["big"]),
        "Done.", "Done.",
    ])
    answer = await aask(big, "Total per category?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    script = answer.export(tmp_path / "out")

    assert script.stat().st_size < 5_000
    assert sorted(p.name for p in (tmp_path / "out" / "data").iterdir()) == ["big.parquet", "totals.parquet"]
    code = (
        "import runpy, sys; g = runpy.run_path(sys.argv[1]); "
        "print(sorted(map(tuple, g['pipeline'].data.values.tolist())))"
    )
    run = subprocess.run([sys.executable, "-c", code, str(script)], cwd=tmp_path, capture_output=True, text=True, check=True)
    expected = sorted(map(tuple, answer.pipeline.data.values.tolist()))
    assert run.stdout.strip() == str(expected)


def test_answer_export_rebuilds_a_chart(chart_answer, tmp_path):
    script = chart_answer.export(tmp_path)
    code = "import runpy, sys; g = runpy.run_path(sys.argv[1]); print(g['view'].spec['mark'], len(g['view'].pipeline.data))"
    run = subprocess.run([sys.executable, "-c", code, str(script)], cwd=tmp_path.parent, capture_output=True, text=True, check=True)
    assert run.stdout.strip() == "bar 3"


def test_answer_export_needs_data():
    with pytest.raises(ValueError, match="no chart or data"):
        Answer(question="q", value="Hi", status="success").export("unused")


def test_answer_code_is_empty_without_data():
    assert Answer(question="q", value="Hi", status="success").code == ""


def test_answer_repr_is_one_line(chart_answer):
    assert repr(chart_answer) == "Answer(status='success', value='**B** is largest.')"


def test_answer_static_bundle_outside_a_kernel(chart_answer):
    bundle = chart_answer._repr_mimebundle_()
    assert bundle["text/markdown"].startswith("**B** is largest.\nA is second.\n\n```sql\nSELECT * FROM tiny\n```\n\n```json\n")
    spec = bundle["application/vnd.vegalite.v5+json"]
    assert spec["data"]["values"][0] == {"category": "A", "value": 10.5}
    assert base64.b64decode(bundle["image/png"]).startswith(b"\x89PNG")
    html = bundle["text/html"]
    assert "<strong>B</strong> is largest." in html
    assert f'src="data:image/png;base64,{bundle["image/png"]}" width="600"' in html
    assert "<details><summary>SQL</summary><pre><code>SELECT * FROM tiny</code></pre></details>" in html
    assert "<details><summary>Spec</summary><pre><code>{" in html


def test_answer_static_spec_sends_missing_values_as_null():
    source = read({"fuels": pd.DataFrame({"year": [1900, 1950], "coal": [1.0, 2.0], "gas": [float("nan"), 3.0]})})
    view = VegaLiteView(pipeline=Pipeline(source=source, table="fuels"), spec={"mark": "area", "encoding": {
        "x": {"field": "year", "type": "quantitative"}, "y": {"field": "gas", "type": "quantitative"},
    }})
    answer = Answer(question="?", spec=view.spec, view=view, status="success")
    spec = answer._chart_bundle()["application/vnd.vegalite.v5+json"]
    # Strict JSON, which a kernel message and the Vega stack both need
    json.dumps(spec, allow_nan=False)
    assert [row["gas"] for row in spec["data"]["values"]] == [None, 3.0]


def test_answer_static_spec_keeps_data_given_by_url(frame_source):
    url = {"url": "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json"}
    view = VegaLiteView(
        pipeline=Pipeline(source=frame_source, table="tiny"),
        spec={"data": url, "mark": "point", "encoding": {"x": {"field": "Horsepower", "type": "quantitative"}}},
    )
    answer = Answer(question="q", value="v", spec=view.spec, view=view, status="success")
    assert answer._vega_lite_spec()["data"] == url


def test_answer_chart_label_reads_a_title_object(chart_answer):
    chart_answer.view.spec = dict(chart_answer.view.spec, title={"text": "Value by category"})
    chart_answer.spec = chart_answer.view.spec
    assert chart_answer._chart_bundle()["text/plain"] == "<Vega-Lite chart: Value by category>"


def test_answer_static_bundle_without_a_chart():
    answer = Answer(question="q", value="Three rows.", status="success")
    bundle = answer._repr_mimebundle_()
    assert set(bundle) == {"text/plain", "text/markdown", "text/html"}
    assert bundle["text/html"] == "<p>Three rows.</p>\n"


def test_answer_in_a_kernel_loads_the_extension_once(chart_answer):
    def load(*args):
        panel_extension._loaded_extensions.extend(args)

    with (
        patch("lumen.ai.answer._in_kernel", return_value=True),
        patch.object(panel_extension, "_loaded", True),
        patch.object(panel_extension, "_loaded_extensions", []),
        patch("lumen.ai.answer.pn.extension", side_effect=load) as extension,
        patch.object(type(chart_answer.__panel__()), "_repr_mimebundle_", return_value=({"text/html": "card"}, {})),
    ):
        assert chart_answer._repr_mimebundle_() == ({"text/html": "card"}, {})
        chart_answer._repr_mimebundle_()
    extension.assert_called_once_with("vega", "codeeditor")


def test_answer_falls_back_to_separate_outputs_when_panel_cannot_render(chart_answer):
    pytest.importorskip("IPython")
    with (
        patch("lumen.ai.answer._in_kernel", return_value=True),
        patch.object(panel_extension, "_loaded_extensions", ["vega", "codeeditor"]),
        patch.object(type(chart_answer.__panel__()), "_repr_mimebundle_", side_effect=ImportError("jupyter_bokeh")),
        patch("IPython.display.display") as display,
    ):
        last = chart_answer._repr_mimebundle_()
    (text,), (chart,) = (c.args for c in display.call_args_list)
    assert text["text/markdown"].startswith("**B** is largest.")
    # Its own output, so front-ends draw it as interactive Vega-Lite
    assert chart["application/vnd.vegalite.v5+json"]["mark"] == "bar"
    assert "image/png" in chart
    assert "SELECT * FROM tiny" in last["text/html"]


def test_answer_card_has_answer_chart_and_collapsed_sql(chart_answer):
    card = chart_answer.__panel__()
    assert card is chart_answer.__panel__()
    text, _chart, ask_box, sql, spec = card
    assert text.object.startswith("**B** is largest.")
    assert not ask_box.visible
    assert sql.collapsed and "SELECT * FROM tiny" in sql[0].object
    assert spec.title == "Spec" and spec.collapsed
    code = spec[0].object
    assert code.startswith("```json\n")
    assert json.loads(code.removeprefix("```json\n").removesuffix("\n```"))["mark"] == "bar"


def test_with_spec_redraws_a_vega_lite_chart(chart_answer):
    edited = json.loads(chart_answer._chart_spec()[1])
    edited["mark"] = "line"
    redrawn = chart_answer.with_spec(json.dumps(edited))
    assert redrawn.spec["mark"] == "line" and redrawn.view is not chart_answer.view
    assert chart_answer.spec["mark"] == "bar"
    assert redrawn.sql == chart_answer.sql and redrawn.value == chart_answer.value


def test_with_spec_redraws_an_hvplot_chart(frame_source):
    pipeline = Pipeline(source=frame_source, table="tiny")
    view = hvPlotView(pipeline=pipeline, kind="bar", x="category", y="value")
    answer = Answer(question="?", status="success", pipeline=pipeline, view=view)
    redrawn = answer.with_spec(answer._chart_spec()[1].replace("kind: bar", "kind: scatter"))
    assert redrawn.view.kind == "scatter" and redrawn.view.pipeline is pipeline


def test_with_spec_rejects_what_is_not_a_spec(chart_answer):
    with pytest.raises(ValueError, match="mapping"):
        chart_answer.with_spec("- just\n- a list")
    with pytest.raises(ValueError, match="no chart"):
        Answer(question="?", status="success").with_spec({})


def test_edit_then_redraw_puts_the_edited_chart_in_the_card(chart_answer):
    card = chart_answer.__panel__()
    code, edit, editor, error, actions = card[-1]
    redraw, cancel = actions
    assert code.visible and edit.visible and not editor.visible and not actions.visible

    edit.clicks += 1
    assert editor.visible and actions.visible and not code.visible
    old_chart = card[1]
    editor.value = editor.value.replace('"bar"', '"point"')
    redraw.clicks += 1
    assert chart_answer.spec["mark"] == "point" and card[1] is not old_chart
    # Back to the code block, which now shows the redrawn spec
    assert code.visible and not editor.visible and '"point"' in code.object

    edit.clicks += 1
    editor.value = "{not: [valid"
    redraw.clicks += 1
    assert error.visible and editor.visible and chart_answer.spec["mark"] == "point"
    cancel.clicks += 1
    assert code.visible and not editor.visible and not error.visible


def test_answer_spec_leaves_out_the_data(chart_answer):
    chart_answer.spec = dict(chart_answer.spec, data={"values": [{"category": "A"}] * 1000})
    language, text = chart_answer._chart_spec()
    assert language == "json" and "values" not in text


def test_answer_spec_of_an_hvplot_chart_is_its_yaml(frame_source):
    pipeline = Pipeline(source=frame_source, table="tiny")
    view = hvPlotView(pipeline=pipeline, kind="bar", x="category", y="value")
    language, text = Answer(question="?", status="success", pipeline=pipeline, view=view)._chart_spec()
    assert language == "yaml" and "kind: bar" in text and "pipeline" not in text


def test_answer_chart_of_an_hvplot_answer_is_its_holoviews_object(frame_source):
    import holoviews as hv
    pipeline = Pipeline(source=frame_source, table="tiny")
    view = hvPlotView(pipeline=pipeline, kind="bar", x="category", y="value")
    chart = Answer(question="?", status="success", pipeline=pipeline, view=view).chart
    assert isinstance(chart, hv.Bars)
    assert list(chart.data["value"]) == [10.5, 20.0, 7.5]
    assert isinstance(chart + chart, hv.Layout)


@pytest.mark.parametrize("key, kind", [
    (None, "Chart"), ("layer", "LayerChart"), ("facet", "FacetChart"), ("hconcat", "HConcatChart"),
])
def test_answer_chart_of_a_vega_lite_answer_is_an_altair_chart(chart_answer, monkeypatch, key, kind):
    built = []

    class TopLevel:
        @classmethod
        def from_dict(cls, spec):
            built.append(spec)
            return cls()

    altair = type(sys)("altair")
    for name in ("Chart", "LayerChart", "FacetChart", "RepeatChart", "HConcatChart", "VConcatChart", "ConcatChart"):
        setattr(altair, name, type(name, (TopLevel,), {}))
    # Panel's Vega pane checks for altair charts through altair.api
    altair.api = type(sys)("altair.api")
    altair.api.TopLevelMixin = TopLevel
    monkeypatch.setitem(sys.modules, "altair", altair)
    layer = {"mark": "bar", "encoding": chart_answer.spec["encoding"]}
    spec = {None: chart_answer.spec, "layer": {"layer": [layer]}, "facet": {"facet": {"field": "category"}, "spec": layer},
            "hconcat": {"hconcat": [layer, layer]}}[key]
    chart_answer.view.spec = chart_answer.spec = spec
    assert type(chart_answer.chart).__name__ == kind
    assert built == [chart_answer.to_vega()]


def test_the_static_card_of_a_concatenated_chart_has_its_image(chart_answer):
    layer = {"mark": "bar", "encoding": chart_answer.spec["encoding"]}
    chart_answer.view.spec = chart_answer.spec = {"hconcat": [layer, layer]}
    html_card = chart_answer._static_bundle(chart_answer._chart_bundle())["text/html"]
    assert '<img src="data:image/png;base64,' in html_card and "width=" not in html_card


def test_answer_to_vega_sizes_only_the_views_that_take_a_size(chart_answer):
    assert (chart_answer.to_vega()["width"], chart_answer.to_vega()["height"]) == (600, 400)
    layer = {"mark": "bar", "encoding": chart_answer.spec["encoding"]}
    chart_answer.view.spec = chart_answer.spec = {"hconcat": [layer, layer]}
    assert "width" not in chart_answer.to_vega() and "height" not in chart_answer.to_vega()


def test_answer_chart_without_altair_says_how_to_get_it(chart_answer, monkeypatch):
    monkeypatch.setitem(sys.modules, "altair", None)
    with pytest.raises(ImportError, match="pip install altair"):
        _ = chart_answer.chart


def test_answer_to_vega_inlines_the_data(chart_answer):
    spec = chart_answer.to_vega()
    assert spec["mark"] == "bar"
    assert [row["value"] for row in spec["data"]["values"]] == [10.5, 20.0, 7.5]
    assert chart_answer.spec.get("data") is None


def test_answer_without_a_chart_has_no_chart_or_vega(frame_source):
    answer = Answer(question="?", value="Ten rows.", sql="SELECT 1", status="success")
    assert answer.chart is None
    with pytest.raises(ValueError, match="no Vega-Lite chart"):
        answer.to_vega()


def test_pn_panel_of_an_answer_is_its_card(chart_answer):
    assert pn.panel(chart_answer) is chart_answer.__panel__()


def test_answer_without_a_chart_has_no_spec(frame_source):
    answer = Answer(question="?", value="Ten rows.", sql="SELECT 1", status="success")
    assert answer._chart_spec() is None
    assert [part.title for part in answer.__panel__() if isinstance(part, Card)] == ["SQL"]
    assert "Spec" not in answer._static_bundle(None)["text/html"]


def test_answer_card_skips_empty_text(chart_answer):
    chart_answer.value = ""
    assert len(chart_answer.__panel__()) == 4


def test_importing_answer_does_not_load_the_panel_extension():
    code = (
        "import lumen.ai.answer; from panel.config import panel_extension; "
        "assert not panel_extension._loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


async def test_chat_follow_up_sees_history_and_last_result(frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm = FollowUpLLM([
        SQLModel(query="SELECT category, SUM(value) AS total FROM tiny GROUP BY category", table_slug="totals", tables=["tiny"]),
        SQLModel(query="SELECT * FROM tiny WHERE category = 'A'", table_slug="only_a", tables=["tiny"]),
    ])
    conversation = frame_source.chat(llm=llm, agents=[ChatAgent(), SQLAgent()])
    first = await conversation.aask("Total value per category?")
    second = await conversation.aask("Only A")

    assert conversation.answers == [first, second]
    history, latest = llm.plan_calls[-1]
    assert "User: Total value per category?" in history["content"]
    assert latest == {"role": "user", "content": "Only A"}
    # The classifier only runs when the last answer's data is in context
    assert llm.follow_up_checks == 1
    assert "SUM(value)" in first.sql and "category = 'A'" in second.sql
    assert repr(conversation) == "Chat(answers=2)"


def test_chat_ask_works_from_sync_code(frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm = FollowUpLLM([SQLModel(query="SELECT COUNT(*) AS n FROM tiny", table_slug="n", tables=["tiny"])])
    answer = chat(frame_source, llm=llm, agents=[ChatAgent(), SQLAgent()]).ask("How many rows?")
    assert answer.pipeline.data.iloc[0, 0] == 3


async def test_answer_explore_opens_the_ui_on_the_answer(llm, frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm.set_responses([
        _plan(("SQLAgent", "Sum value"), ("ChatAgent", "Explain")),
        SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="tiny_total", tables=["tiny"]),
        "38.", "38.",
    ])
    answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert answer.source is frame_source

    ui = answer.explore()
    await async_wait_until(lambda: len(ui._explorations.items) == 2)

    assert ui.llm is llm
    # The UI also adds the answer's derived table, as it does after a chat answer
    assert ui.context["sources"][0] is frame_source
    exploration = ui._explorations.items[1]["view"]
    assert exploration.context["pipeline"] is answer.pipeline
    assert ui.interface.objects[0].object == "Total value?"


class FollowUpLLM(MockLLM):
    """Replies by response model, and records what each Plan call was sent."""

    def __init__(self, sql_models, **params):
        super().__init__(**params)
        self.sql_models = list(sql_models)
        self.plan_calls = []
        self.follow_up_checks = 0

    async def invoke(self, messages, system="", response_model=None, **kwargs):
        name = getattr(response_model, "__name__", "")
        if "Plan" in name:
            self.plan_calls.append(messages)
            return _plan(("SQLAgent", "Query"), ("ChatAgent", "Explain"))
        if name == "FollowUpClassification":
            self.follow_up_checks += 1
            return response_model(chain_of_thought="Builds on the last result", follow_up_type="derived")
        if "SQLQuery" in name:
            return self.sql_models.pop(0)
        if name:
            return await super().invoke(messages, system=system, response_model=response_model, **kwargs)
        return "Done."


class FakeShell:
    def __init__(self):
        self.next_inputs = []

    def set_next_input(self, text, replace=False):
        self.next_inputs.append((text, replace))


def test_to_cell_puts_the_code_in_the_next_cell(chart_answer):
    shell = FakeShell()
    with patch("lumen.ai.answer._in_kernel", return_value=True), patch("IPython.get_ipython", return_value=shell):
        chart_answer.to_cell()
    assert shell.next_inputs == [(chart_answer.code, False)]


def test_to_cell_needs_a_kernel(chart_answer):
    with pytest.raises(RuntimeError, match=r"answer\.code holds the code"):
        chart_answer.to_cell()


def test_to_cell_needs_code():
    with pytest.raises(ValueError, match="no chart or data"):
        Answer(question="?", status="success").to_cell()


class FakeHandle:
    def __init__(self):
        self.updates = []

    def update(self, obj, raw=False):
        self.updates.append(obj["text/plain"])


async def test_run_shows_the_current_step_in_place():
    handle = FakeHandle()

    async def question():
        NullStep().add("Generating SQL...")
        await asyncio.sleep(0.6)
        return "done"

    with patch("lumen.ai.answer._in_kernel", return_value=True), \
            patch("IPython.display.display", return_value=handle) as display:
        assert _run(question()) == "done"

    display.assert_called_once()
    assert any(text.startswith("Generating SQL (") for text in handle.updates)
    assert handle.updates[-1] == ""


async def test_run_cancels_the_question_on_interrupt():
    ended = []

    async def question():
        try:
            await asyncio.sleep(30)
        finally:
            ended.append(asyncio.current_task().cancelling())

    threads = threading.active_count()
    start = time.monotonic()
    with patch.object(_StatusLine, "show", side_effect=KeyboardInterrupt), pytest.raises(KeyboardInterrupt):
        _run(question())

    assert time.monotonic() - start < 5
    assert ended == [1]
    assert threading.active_count() == threads


async def test_run_names_the_first_step_until_one_reports():
    handle = FakeHandle()

    async def revise():
        await asyncio.sleep(0.4)

    with patch("lumen.ai.answer._in_kernel", return_value=True), \
            patch("IPython.display.display", return_value=handle) as display:
        _run(revise(), "Revising the chart")

    assert display.call_args.args[0] == {"text/plain": "Revising the chart"}
    assert any(text.startswith("Revising the chart (") for text in handle.updates)


def _geo_answer(crs="EPSG:4326"):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point
    source = read({"places": gpd.GeoDataFrame({"n": ["a", "b"]}, geometry=[Point(13.4, 52.5), Point(2.3, 48.9)], crs=crs)})
    return Answer(question="?", status="success", pipeline=Pipeline(source=source, table="places"))


def test_answer_gdf_keeps_the_crs():
    gdf = _geo_answer().gdf
    assert gdf.crs.to_epsg() == 4326 and len(gdf) == 2


def test_answer_gdf_is_none_without_geometry(frame_source):
    assert Answer(question="?", status="success", pipeline=Pipeline(source=frame_source, table="tiny")).gdf is None


def test_answer_gdf_warns_once_when_lon_lat_has_no_crs():
    answer = _geo_answer(crs=None)
    with pytest.warns(UserWarning, match="looks like longitude and latitude"):
        assert answer.gdf.crs is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert answer.gdf.crs is None


def _grid_answer(sql):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_sql")
    import numpy as np
    grid = xr.Dataset(
        {"t": (("time", "lat", "lon"), np.arange(24.0).reshape(2, 3, 4))},
        coords={"time": [0, 1], "lat": [10.0, 20.0, 30.0], "lon": [0.0, 1.0, 2.0, 3.0]},
    )
    source = read({"t": grid})
    derived = source.create_sql_expr_source({"result": sql})
    return grid, Answer(question="?", status="success", pipeline=Pipeline(source=derived, table="result"))


def test_answer_ds_is_the_gridded_result():
    grid, answer = _grid_answer("SELECT lat, lon, AVG(t) AS t FROM t GROUP BY lat, lon")
    ds = answer.ds
    assert dict(ds.sizes) == {"lat": 3, "lon": 4}
    assert (ds.t.transpose("lat", "lon").sortby(["lat", "lon"]).values == grid.t.mean("time").values).all()


def test_answer_ds_is_none_for_tables(frame_source):
    assert Answer(question="?", status="success", pipeline=Pipeline(source=frame_source, table="tiny")).ds is None


@pytest.mark.parametrize("suffix", [".parquet", ".geojson", ".gpkg"])
def test_to_file_writes_geometry_with_its_crs(tmp_path, suffix):
    gpd = pytest.importorskip("geopandas")
    path = _geo_answer().to_file(tmp_path / f"places{suffix}")
    back = gpd.read_parquet(path) if suffix == ".parquet" else gpd.read_file(path)
    assert back.crs.to_epsg() == 4326 and list(back["n"]) == ["a", "b"]


def test_to_file_writes_a_table(tmp_path, frame_source):
    answer = Answer(question="?", status="success", pipeline=Pipeline(source=frame_source, table="tiny"))
    assert len(pd.read_csv(answer.to_file(tmp_path / "t.csv"))) == 3
    assert len(pd.read_parquet(answer.to_file(tmp_path / "t.parquet"))) == 3
    with pytest.raises(ValueError, match="hold geometry"):
        answer.to_file(tmp_path / "t.geojson")
    with pytest.raises(ValueError, match="hold grids"):
        answer.to_file(tmp_path / "t.zarr")
    with pytest.raises(ValueError, match=r"Cannot write '\.xlsx'"):
        answer.to_file(tmp_path / "t.xlsx")


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_to_file_writes_a_grid(tmp_path, suffix):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr" if suffix == ".zarr" else "netCDF4")
    _, answer = _grid_answer("SELECT lat, lon, AVG(t) AS t FROM t GROUP BY lat, lon")
    path = answer.to_file(tmp_path / f"g{suffix}")
    back = xr.open_zarr(path, consolidated=False) if suffix == ".zarr" else xr.open_dataset(path)
    assert dict(back.sizes) == {"lat": 3, "lon": 4}


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3857"])
def test_a_map_answer_gets_a_static_geoshape(crs):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point
    points = gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=[Point(13.4, 52.5), Point(2.3, 48.9)], crs="EPSG:4326").to_crs(crs)
    answer = Answer(question="Where?", status="success", pipeline=Pipeline(source=read({"pts": points}), table="pts"))
    bundle = answer._chart_bundle()
    spec = bundle["application/vnd.vegalite.v5+json"]
    assert spec["mark"]["type"] == "geoshape" and spec["encoding"]["color"]["field"] == "properties.v"
    assert spec["encoding"]["color"]["title"] == "v"
    lon, lat = spec["data"]["values"][0]["geometry"]["coordinates"]
    # Back in longitude and latitude, whatever CRS the answer came in
    assert abs(lon - 13.4) < 1e-6 and abs(lat - 52.5) < 1e-6
    assert base64.b64decode(bundle["image/png"])[:4] == b"\x89PNG"


def test_map_uses_lonboard_when_installed():
    pytest.importorskip("lonboard")
    assert type(_geo_answer().map()).__name__ == "Map"


def test_map_falls_back_to_hvplot_on_a_basemap(monkeypatch):
    pytest.importorskip("geoviews")
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "lonboard" else real(name, *a))
    plot = _geo_answer().map()
    assert any("Tiles" in type(el).__name__ or "WMTS" in type(el).__name__ for el in plot)


def test_map_needs_geometry(frame_source):
    with pytest.raises(ValueError, match="no geometry to map"):
        Answer(question="?", status="success", pipeline=Pipeline(source=frame_source, table="tiny")).map()


def test_answer_ds_has_sorted_coordinates():
    _, answer = _grid_answer("SELECT lat, lon, AVG(t) AS t FROM t GROUP BY lat, lon ORDER BY lat DESC, lon DESC")
    ds = answer.ds
    assert list(ds.lat.values) == [10.0, 20.0, 30.0] and list(ds.lon.values) == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize("responsive", [True, False])
def test_card_gives_a_responsive_plot_a_height(frame_source, responsive):
    pipeline = Pipeline(source=frame_source, table="tiny")
    view = hvPlotView(pipeline=pipeline, kind="bar", x="category", y="value", responsive=responsive)
    card = Answer(question="?", status="success", pipeline=pipeline, view=view).__panel__()
    assert (card[0].min_height == 400) is responsive


def _answer_over(data, sql):
    source = read(data)
    pipeline = Pipeline(source=source.create_sql_expr_source({"result": sql}), table="result")
    return Answer(question="?", status="success", source=source, sql=sql, pipeline=pipeline)


def test_answer_data_is_pandas_for_pandas_input(frame_source):
    answer = _answer_over({"t": pd.DataFrame({"a": [1, 2]})}, "SELECT a FROM t")
    assert isinstance(answer.data, pd.DataFrame) and answer.data.a.tolist() == [1, 2]


def test_answer_data_is_polars_for_polars_input():
    pl = pytest.importorskip("polars")
    answer = _answer_over(pl.DataFrame({"a": [1, 2]}), "SELECT a * 2 AS b FROM data")
    assert isinstance(answer.data, pl.DataFrame) and answer.data["b"].to_list() == [2, 4]


def test_answer_data_is_pyarrow_for_pyarrow_input():
    pa = pytest.importorskip("pyarrow")
    answer = _answer_over({"t": pa.table({"a": [1, 2]})}, "SELECT a FROM t")
    assert isinstance(answer.data, pa.Table) and answer.data.column_names == ["a"]


def test_answer_data_is_a_geodataframe_for_geopandas_input():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point
    places = gpd.GeoDataFrame({"n": ["a"]}, geometry=[Point(13.4, 52.5)], crs="EPSG:4326")
    answer = _answer_over({"places": places}, "SELECT n, geometry FROM places")
    assert isinstance(answer.data, gpd.GeoDataFrame) and answer.data.crs.to_epsg() == 4326


def test_answer_data_is_a_dataset_for_xarray_input():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_sql")
    import numpy as np
    grid = xr.Dataset({"t": (("lat", "lon"), np.arange(6.0).reshape(2, 3))}, coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]})
    answer = _answer_over({"t": grid}, "SELECT lat, lon, t FROM t")
    assert isinstance(answer.data, xr.Dataset) and dict(answer.data.sizes) == {"lat": 2, "lon": 3}


def test_answer_data_follows_the_tables_the_query_read():
    pl = pytest.importorskip("polars")
    data = {"fast": pl.DataFrame({"a": [1]}), "plain": pd.DataFrame({"b": [2]})}
    assert isinstance(_answer_over(data, "SELECT a FROM fast").data, pl.DataFrame)
    # Tables of two types read together come back as pandas
    mixed = _answer_over(data, "SELECT a, b FROM fast CROSS JOIN plain")
    assert type(mixed.data) is pd.DataFrame


def test_answer_data_imports_no_library_the_input_did_not_use():
    code = (
        "import sys, pandas as pd, lumen; from lumen.ai.answer import Answer; from lumen.pipeline import Pipeline\n"
        "src = lumen.read({'t': pd.DataFrame({'a': [1]})})\n"
        "p = Pipeline(source=src.create_sql_expr_source({'r': 'SELECT a FROM t'}), table='r')\n"
        "Answer(question='?', status='success', source=src, sql='SELECT a FROM t', pipeline=p).data\n"
        "assert not {'polars', 'geopandas', 'xarray'} & set(sys.modules), {'polars', 'geopandas', 'xarray'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


class _RecordingLLM(MockLLM):
    """A MockLLM that keeps the system prompt of every call."""

    async def invoke(self, messages, system="", response_model=None, **kwargs):
        self.systems = [*getattr(self, "systems", []), system]
        return await super().invoke(messages, system=system, response_model=response_model, **kwargs)


async def test_column_descriptions_in_source_metadata_reach_the_sql_prompt():
    source = read({"tiny": pd.DataFrame({"value": [1.0]})})
    source.metadata = {"tiny": {"columns": {"value": "Euro amount of each sale"}}}
    llm = _RecordingLLM()
    SQLModel = make_sql_model([(source.name, "tiny")])
    llm.set_responses([
        _plan(("SQLAgent", "Sum")), SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="t", tables=["tiny"]),
        "The total is 1.",
    ])
    await aask(source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert any("Euro amount of each sale" in system for system in llm.systems)


async def test_described_text_reaches_the_sql_prompt_even_after_a_first_question(frame_source):
    llm = _RecordingLLM()
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    sql_step = lambda: SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="t", tables=["tiny"])
    llm.set_responses([
        _plan(("SQLAgent", "Sum")), sql_step(), "The total is 38.",
        _plan(("SQLAgent", "Sum")), sql_step(), "The total is 38.",
    ])
    agents = [ChatAgent(), SQLAgent()]

    await aask(frame_source, "Total value?", llm=llm, agents=agents)
    assert not any("Euro amount" in system for system in llm.systems)
    frame_source.describe("tiny", "Sales by category", columns={"value": "Euro amount of each sale"})
    await aask(frame_source, "Total value?", llm=llm, agents=agents)

    later = llm.systems[3:]
    assert any("Euro amount of each sale" in system for system in later)
    assert any("Sales by category" in system for system in later)


async def test_confirm_returns_the_plan_without_running_it(llm, frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm.set_responses([
        _costing(llm, _plan(("SQLAgent", "Sum value"), ("ChatAgent", "Explain")), 100, 0.001),
        _costing(llm, SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="t", tables=["tiny"]), 200, 0.002),
        _costing(llm, "The total value is 38.", 50, 0.0005),
    ])

    answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()], confirm=True)

    assert answer.status == "planned" and llm._index == 1  # planned, nothing ran
    assert "1. **SQLAgent**: Sum value" in answer._plan_steps and "2. **ChatAgent**: Explain" in answer._plan_steps
    card = answer.__panel__()
    assert card[0].title == "Planned: Total value" and card[1].label == "Run"
    assert "Run it with answer.run()" in answer._static_bundle(None)["text/markdown"]

    assert await answer.arun() is answer
    assert answer.status == "success" and answer.value == "The total value is 38."
    assert "SUM(value)" in answer.sql and llm._index == 3  # planning was not paid twice
    assert answer.usage.input_tokens == 350  # planning and running together
    assert answer.__panel__() is card and card[0].object.startswith("The total value is 38.")
    with pytest.raises(ValueError, match="only once"):
        await answer.arun()


async def test_an_unanswerable_question_is_not_held_for_confirmation(llm, frame_source):
    llm.set_responses([_plan(unanswerable="No table has an elevation column.")])
    answer = await aask(frame_source, "Elevation?", llm=llm, agents=[ChatAgent(), SQLAgent()], confirm=True)
    assert answer.status == "unanswerable"


def test_only_a_planned_answer_runs(chart_answer):
    with pytest.raises(ValueError, match="confirm=True"):
        chart_answer.run()


async def test_a_planned_chat_question_joins_the_chat_when_run(llm, frame_source):
    llm.set_responses([_plan(("ChatAgent", "Explain")), "Ten rows."])
    conversation = chat(frame_source, llm=llm, agents=[ChatAgent(), SQLAgent()])

    planned = await conversation.aask("How many rows?", confirm=True)
    assert conversation.answers == []
    await planned.arun()
    assert conversation.answers == [planned] and conversation._messages[-1]["content"] == "Ten rows."


def _cached_ask(llm, source, question="How many rows?"):
    _chat_only(llm)
    return aask(source, question, llm=llm, agents=[ChatAgent(), SQLAgent()])


async def test_a_cached_question_makes_no_llm_calls(llm, frame_source):
    configure(cache=True)
    first = await _cached_ask(llm, frame_source)
    calls = llm._index
    again = await aask(frame_source, "  how many  ROWS? ", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert llm._index == calls and again is not first
    assert again.cached and again.value == first.value == "Three rows."
    assert again._usage_note == "From the answer cache, no LLM calls"
    assert again.__panel__() is not first.__panel__()


async def test_the_cache_is_off_unless_configured(llm, frame_source):
    await _cached_ask(llm, frame_source)
    second = await _cached_ask(llm, frame_source)
    assert not second.cached and llm._index == 2


@pytest.mark.parametrize("change", ["describe", "model", "prompts", "source"])
async def test_a_change_that_could_alter_the_answer_misses_the_cache(llm, frame_source, change):
    configure(cache=True)
    await _cached_ask(llm, frame_source)
    source = frame_source
    if change == "describe":
        frame_source.describe("tiny", "Sales by category")
    elif change == "model":
        llm.model_kwargs = {"default": {"model": "another"}}
    elif change == "source":
        source = read({"tiny": pd.DataFrame({"category": ["A", "B", "A"], "value": [10.5, 20.0, 7.5]})})
    with patch("lumen.ai.answer._prompt_version", return_value="edited" if change == "prompts" else answer_module._prompt_version()):
        again = await _cached_ask(llm, source)
    assert not again.cached


async def test_failed_answers_are_not_cached(llm, frame_source):
    configure(cache=True)
    llm.set_responses([_plan(("ChatAgent", "Explain")), ""])
    await aask(frame_source, "Total?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    again = await _cached_ask(llm, frame_source, "Total?")
    assert not again.cached


async def test_the_cache_keeps_only_the_latest_answers(llm, frame_source, monkeypatch):
    monkeypatch.setattr(answer_module, "_ANSWER_CACHE_SIZE", 2)
    configure(cache=True)
    for question in ("One?", "Two?", "Three?"):
        await _cached_ask(llm, frame_source, question)
    assert len(answer_module._answer_cache) == 2
    assert not (await _cached_ask(llm, frame_source, "One?")).cached
    clear_cache()
    assert not answer_module._answer_cache


async def test_editing_a_frame_after_reading_it_cannot_make_a_cached_answer_stale(llm):
    """lumen.read copies frames into DuckDB, so a later edit changes no answer."""
    frame = pd.DataFrame({"category": ["A", "B"], "value": [1.0, 2.0]})
    source = read({"tiny": frame})
    frame.loc[0, "value"] = 99.0
    assert source.execute("SELECT SUM(value) AS total FROM tiny").iloc[0, 0] == 3.0


async def test_the_cache_bounds_the_rows_it_holds(llm, frame_source, monkeypatch):
    monkeypatch.setattr(answer_module, "_ANSWER_CACHE_ROWS", 4)
    configure(cache=True)
    for question in ("One?", "Two?", "Three?"):
        answer = await _cached_ask(llm, frame_source, question)
        answer.pipeline = Pipeline(source=frame_source, table="tiny")  # three rows each
        answer_module._answer_cache[next(reversed(answer_module._answer_cache))] = answer
    assert len(answer_module._answer_cache) <= 2


class _TypingLLM(MockLLM):
    """Answers the plan and SQL from its responses, and the typed result with result."""

    result = None
    typed_prompt = None

    async def invoke(self, messages, system="", response_model=None, **kwargs):
        if response_model is not None and response_model.__name__ == "Result":
            self.typed_prompt = messages[-1]["content"]
            return response_model(result=self.result)
        return await super().invoke(messages, system, response_model, **kwargs)


def _sql_answer(llm, source, query):
    SQLModel = make_sql_model([(source.name, "tiny")])
    llm.set_responses([
        _plan(("SQLAgent", "Query"), ("ChatAgent", "Explain")),
        SQLModel(query=query, table_slug="tiny_result", tables=["tiny"]),
        "The answer is in the data.",
    ])


@pytest.mark.parametrize("query, returns, expected", [
    ("SELECT SUM(value) AS total FROM tiny", int, 38),
    ("SELECT SUM(value) AS total FROM tiny", float, 38.0),
    ("SELECT DISTINCT category FROM tiny ORDER BY category", list[str], ["A", "B"]),
    ("SELECT category, SUM(value) AS total FROM tiny GROUP BY category ORDER BY category",
     dict[str, float] | list[dict[str, Any]], [{"category": "A", "total": 18.0}, {"category": "B", "total": 20.0}]),
])
async def test_returns_takes_a_result_already_in_the_shape_without_an_llm_call(frame_source, query, returns, expected):
    llm = _TypingLLM()
    _sql_answer(llm, frame_source, query)
    answer = await aask(frame_source, "?", llm=llm, agents=[ChatAgent(), SQLAgent()], returns=returns)
    assert answer.status == "success" and answer.result == expected
    assert type(answer.result) is type(expected)
    assert llm._index == 3 and llm.typed_prompt is None


async def test_returns_a_model_from_a_matching_row(frame_source):
    from pydantic import BaseModel

    class Total(BaseModel):
        category: str
        total: float

    llm = _TypingLLM()
    _sql_answer(llm, frame_source, "SELECT 'B' AS category, MAX(value) AS total FROM tiny")
    answer = await aask(frame_source, "?", llm=llm, agents=[ChatAgent(), SQLAgent()], returns=Total)
    assert answer.result == Total(category="B", total=20.0)


async def test_returns_reads_the_type_from_the_answer_when_the_data_has_another_shape(frame_source):
    llm = _TypingLLM()
    llm.result = "B"
    _sql_answer(llm, frame_source, "SELECT category, SUM(value) AS total FROM tiny GROUP BY category")
    answer = await aask(frame_source, "Which category is largest?", llm=llm, agents=[ChatAgent(), SQLAgent()], returns=str)
    assert answer.status == "success" and answer.result == "B"
    assert "Which category is largest?" in llm.typed_prompt and "category,total" in llm.typed_prompt


async def test_returns_never_takes_one_cell_of_a_larger_result(frame_source):
    """Two rows are not one int; the typed call decides, not the first cell."""
    llm = _TypingLLM()
    llm.result = 2
    _sql_answer(llm, frame_source, "SELECT COUNT(*) AS n FROM tiny GROUP BY category")
    answer = await aask(frame_source, "How many categories?", llm=llm, agents=[ChatAgent(), SQLAgent()], returns=int)
    assert answer.result == 2 and llm.typed_prompt is not None


async def test_returns_keeps_data_out_of_the_typed_prompt_unless_samples_are_shared(frame_source):
    llm = _TypingLLM()
    llm.result = "B"
    _sql_answer(llm, frame_source, "SELECT category, SUM(value) AS total FROM tiny GROUP BY category")
    configure(share="schema")
    try:
        await aask(frame_source, "?", llm=llm, agents=[ChatAgent(), SQLAgent()], returns=str)
    finally:
        configure(share="sample")
    assert "category,total" not in llm.typed_prompt


async def test_an_answer_that_is_not_the_type_becomes_an_error(frame_source):
    class Refusing(_TypingLLM):
        async def invoke(self, messages, system="", response_model=None, **kwargs):
            if response_model is not None and response_model.__name__ == "Result":
                raise ValueError("no number in the answer")
            return await super().invoke(messages, system, response_model, **kwargs)

    llm = Refusing()
    _sql_answer(llm, frame_source, "SELECT 'many' AS category")
    answer = await aask(frame_source, "?", llm=llm, agents=[ChatAgent(), SQLAgent()], returns=int)
    assert answer.status == "error" and "could not be read as int" in answer.error
    assert answer.result is None and answer.sql


async def test_returns_is_part_of_the_cache_key(frame_source):
    configure(cache=True)
    try:
        llm = _TypingLLM()
        _sql_answer(llm, frame_source, "SELECT SUM(value) AS total FROM tiny")
        await aask(frame_source, "Total?", llm=llm, agents=[ChatAgent(), SQLAgent()], returns=int)
        _sql_answer(llm, frame_source, "SELECT SUM(value) AS total FROM tiny")
        other = await aask(frame_source, "Total?", llm=llm, agents=[ChatAgent(), SQLAgent()], returns=float)
        again = await aask(frame_source, "Total?", llm=llm, agents=[ChatAgent(), SQLAgent()], returns=int)
    finally:
        configure(cache=False)
        clear_cache()
    assert not other.cached and other.result == 38.0
    assert again.cached and again.result == 38 and again._returns is int


async def test_a_planned_answer_is_typed_when_run(frame_source):
    llm = _TypingLLM()
    _sql_answer(llm, frame_source, "SELECT SUM(value) AS total FROM tiny")
    answer = await aask(frame_source, "?", llm=llm, agents=[ChatAgent(), SQLAgent()], confirm=True, returns=int)
    assert answer.status == "planned" and answer.result is None
    await answer.arun()
    assert answer.result == 38


async def test_retry_asks_for_the_same_type(frame_source):
    llm = _TypingLLM()
    _sql_answer(llm, frame_source, "SELECT SUM(value) AS total FROM tiny")
    answer = await aask(frame_source, "?", llm=llm, agents=[ChatAgent(), SQLAgent()], returns=int)
    _sql_answer(llm, frame_source, "SELECT SUM(value) AS total FROM tiny")
    assert (await answer.aretry()).result == 38


async def test_chat_ask_takes_returns(frame_source):
    llm = _TypingLLM()
    _sql_answer(llm, frame_source, "SELECT SUM(value) AS total FROM tiny")
    answer = await chat(frame_source, llm=llm, agents=[ChatAgent(), SQLAgent()]).aask("?", returns=int)
    assert answer.result == 38


@pytest.fixture
def penguins_answer():
    frame = pd.DataFrame({
        "year": [2007, 2008, 2008, 2009], "island": ["A", "B", "A", "B"], "body mass": [1.0, 2.0, 3.0, 4.0],
    })
    sql = (
        "SELECT COUNT(*) AS n FROM penguins WHERE year = 2008 AND island IN ('A', 'B') "
        'AND "body mass" BETWEEN 0 AND 10'
    )
    return Answer(question="How many in 2008?", status="success", source=read({"penguins": frame}), sql=sql)


def test_compile_reruns_the_query_with_new_values_and_no_llm(penguins_answer):
    count = penguins_answer.compile(["year", "island", "body mass"])
    assert count().iloc[0, 0] == 2
    assert count(year=2009).iloc[0, 0] == 1
    assert count(island=["A"]).iloc[0, 0] == 1
    assert count(**{"body mass": (2.5, 10)}).iloc[0, 0] == 1


def test_compile_defaults_are_the_answers_values(penguins_answer):
    import inspect
    count = penguins_answer.compile(["year", "island"])
    params = inspect.signature(count).parameters
    assert {name: p.default for name, p in params.items()} == {"year": 2008, "island": ["A", "B"]}
    assert "$p0" in count.sql and "2008" not in count.sql
    assert count.__doc__.startswith("How many in 2008?")


def test_compile_binds_values_instead_of_writing_them_into_the_sql(penguins_answer):
    count = penguins_answer.compile(["island"])
    assert count(island=["x'; DROP TABLE penguins; --"]).iloc[0, 0] == 0
    assert penguins_answer.source.execute("SELECT COUNT(*) FROM penguins").iloc[0, 0] == 4


def test_compile_names_the_columns_it_can_take(penguins_answer):
    with pytest.raises(ValueError, match=r"no column 'sex'.*island, year"):
        penguins_answer.compile(["sex"])
    with pytest.raises(TypeError, match="Unknown arguments"):
        penguins_answer.compile(["year"])(island=["A"])


def test_compile_refuses_a_column_compared_with_several_values():
    source = read({"t": pd.DataFrame({"year": [2007, 2008]})})
    answer = Answer(question="?", status="success", source=source, sql="SELECT * FROM t WHERE year >= 2007 AND year < 2009")
    with pytest.raises(ValueError, match="different values"):
        answer.compile(["year"])


def test_compile_returns_the_input_type():
    pl = pytest.importorskip("polars")
    source = read({"t": pl.DataFrame({"year": [2007, 2008]})})
    answer = Answer(question="?", status="success", source=source, sql="SELECT * FROM t WHERE year = 2007")
    assert isinstance(answer.compile(["year"])(year=2008), pl.DataFrame)


def test_compile_needs_a_query_that_worked(frame_source):
    with pytest.raises(ValueError, match="no query that worked"):
        Answer(question="?", status="success").compile()
    with pytest.raises(ValueError, match="no query that worked"):
        Answer(question="?", status="error", sql="SELECT 1", sql_failed=True, source=frame_source).compile()


async def test_answer_trace_has_each_step_as_it_ran(llm, frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm.set_responses([
        _plan(("SQLAgent", "Sum value"), ("ChatAgent", "Explain")),
        _costing(llm, SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="tiny_total", tables=["tiny"]), 10, None),
        _costing(llm, "The total value is 38.", 10, None),
    ])
    answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])

    sql_step, chat_step = answer.trace
    assert (sql_step.actor, chat_step.actor) == ("SQLAgent", "ChatAgent")
    assert sql_step.status == chat_step.status == "success"
    assert sql_step.seconds > 0 and chat_step.seconds > 0
    assert sql_step.llm_calls == chat_step.llm_calls == 1
    assert "SUM(value)" in sql_step.sql and chat_step.sql is None
    assert "sql" in sql_step.provides and "source" in sql_step.requires and chat_step.provides == ("chat",)
    assert answer.why == "Query then explain"


def test_plan_graph_follows_what_steps_need():
    plan = Plan(
        ActorTask(SQLAgent(), title="Query", instruction="Query"),
        ActorTask(VegaLiteAgent(n_doc_pages=0), title="Chart", instruction="Chart"),
        ActorTask(ChatAgent(), title="Explain", instruction="Explain"),
        title="Chart it",
    )
    assert plan.graph() == [("Query", "Chart", "data"), ("Query", "Chart", "pipeline"), ("Query", "Chart", "table")]
    mermaid = plan.to_mermaid()
    assert mermaid.startswith("graph TD") and 's0["Query"]' in mermaid and "s0 -->|pipeline| s1" in mermaid
    assert 's2["Explain"]' in mermaid


async def test_answer_steps_card_lists_the_trace(llm, frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm.set_responses([
        _plan(("SQLAgent", "Sum value"), ("ChatAgent", "Explain")),
        SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="tiny_total", tables=["tiny"]),
        "The total value is 38.",
    ])
    answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    steps = next(part for part in answer.__panel__() if isinstance(part, Card) and part.title == "Steps")
    assert steps.collapsed
    text = steps[0].object
    assert text.startswith("*Query then explain*") and "1. **SQLAgent**: Sum value · " in text
    assert "<summary>Steps</summary>" in answer._static_bundle(None)["text/html"]


def test_a_plan_that_did_not_run_has_no_timings(chart_answer):
    agent = VegaLiteAgent(n_doc_pages=0)
    answer = Answer(question="?", status="planned", plan=Plan(ActorTask(agent, title="Chart", instruction="Chart"), title="Chart"))
    (step,) = answer.trace
    assert step.seconds is None and step.llm_calls is None
    assert answer._steps_text.endswith("Chart · not run")


def test_an_answer_without_a_plan_has_no_trace():
    answer = Answer(question="?", status="success")
    assert answer.trace == [] and answer.why == "" and answer._steps_text == ""


def _total_responses(frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    return [
        _plan(("SQLAgent", "Sum value"), ("ChatAgent", "Explain")),
        SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="tiny_total", tables=["tiny"]),
        "The total value is 38.",
    ]


async def test_a_question_typed_into_the_chat_widget_joins_the_chat(llm, frame_source):
    llm.set_responses(_total_responses(frame_source))
    conversation = chat(frame_source, llm=llm, agents=[ChatAgent(), SQLAgent()])
    widget = conversation.__panel__()
    assert widget is conversation.__panel__()

    widget.send("Total value?")
    await async_wait_until(lambda: len(conversation.answers) == 1, timeout=10000)
    (answer,) = conversation.answers
    await async_wait_until(lambda: widget.objects[-1].object is answer.__panel__(), timeout=10000)

    assert answer.value == "The total value is 38."
    assert [message.object for message in widget.objects][0] == "Total value?"
    assert conversation._messages[-1] == {"role": "assistant", "content": "The total value is 38."}


def test_the_chat_widget_starts_with_the_answers_so_far(chart_answer, frame_source):
    conversation = chat(frame_source)
    conversation.answers = [chart_answer]
    widget = conversation.__panel__()
    question, card = widget.objects
    assert question.object == "Value by category?" and card.object is chart_answer.__panel__()


async def test_a_failed_call_in_the_chat_widget_shows_an_error(frame_source):
    class Failing(MockLLM):
        async def invoke(self, *args, **kwargs):
            raise RuntimeError("rate limited")

    conversation = chat(frame_source, llm=Failing(), agents=[ChatAgent(), SQLAgent()])
    reply = await conversation._reply("Total value?", "User", conversation.__panel__())
    assert reply.severity == "error" and "rate limited" in reply.object


def test_a_chat_outside_a_kernel_is_its_answers_as_html(chart_answer, frame_source):
    conversation = chat(frame_source)
    conversation.answers = [chart_answer]
    bundle = conversation._repr_mimebundle_()
    assert bundle["text/plain"] == "Chat(answers=1)"
    assert "<h3>Value by category?</h3>" in bundle["text/html"] and "<img" in bundle["text/html"]


@pytest.fixture
def two_answer_chat(refinable_answer, frame_source):
    table = Answer(
        question="Total value?", value="The total is 38.", status="success", source=frame_source,
        sql="SELECT SUM(value) AS total FROM tiny",
        pipeline=Pipeline(source=frame_source, table="tiny"),
    )
    failed = Answer(question="Wingspan?", value="Lumen could not answer 'Wingspan?'.", error="No such column", status="error")
    conversation = chat(frame_source)
    conversation.answers = [refinable_answer, table, failed]
    return conversation


def test_chat_to_html_is_one_page_of_every_answer(two_answer_chat, tmp_path):
    page = two_answer_chat.to_html(tmp_path / "chat.html")
    assert page.startswith("<!doctype html>") and (tmp_path / "chat.html").read_text(encoding="utf-8") == page
    for heading in ("Value by category?", "Total value?", "Wingspan?"):
        assert f"<h3>{heading}</h3>" in page
    assert "SUM(value)" in page and "No such column" in page


def test_chat_to_notebook_runs_without_an_llm(two_answer_chat, tmp_path):
    import nbformat
    text = two_answer_chat.to_notebook(tmp_path / "chat.ipynb")
    notebook = nbformat.reads(text, as_version=4)
    nbformat.validate(notebook)
    markdown = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
    assert markdown[0].startswith("## Value by category?") and "No such column" in markdown[2]
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    # Bare expressions such as `view` are fine in a script; their objects are what count
    script = code + "\nassert isinstance(view, lm.views.View) and len(pipeline.data) == 3\n"
    (tmp_path / "chat.py").write_text(script)
    subprocess.run([sys.executable, str(tmp_path / "chat.py")], check=True, cwd=tmp_path, timeout=300)


def test_chat_to_report_has_each_answer_and_its_chart(two_answer_chat, refinable_answer):
    report = two_answer_chat.to_report("Penguins")
    assert report.title == "Penguins"
    titles = [view.object.splitlines()[0] for view in report.views if isinstance(view, pn.pane.Markdown)]
    assert titles == ["## Value by category?", "## Total value?", "## Wingspan?"]
    assert refinable_answer._editor in report.views
    assert "## Total value?" in report.to_notebook()


def test_a_vega_lite_chart_reports_clicks_on_its_dimension_fields():
    from lumen.ai.selection import CLICK, with_click as _with_click
    spec = {"mark": "bar", "encoding": {
        "x": {"field": "category", "type": "nominal"},
        "y": {"field": "value", "aggregate": "mean", "type": "quantitative"},
        "color": {"field": "island", "type": "nominal"},
    }}
    assert _with_click(spec)["params"] == [{"name": CLICK, "select": {"type": "point", "fields": ["category", "island"]}}]
    assert _with_click(spec)["encoding"]["opacity"] == {"condition": {"param": CLICK, "value": 1}, "value": 0.35}
    faded = {**spec, "encoding": {**spec["encoding"], "opacity": {"value": 0.5}}}
    assert _with_click(faded)["encoding"]["opacity"] == {"value": 0.5}
    layered = _with_click({"encoding": spec["encoding"], "layer": [{"mark": "bar"}, {"mark": "text"}]})
    assert layered["layer"][0]["params"][0]["select"]["fields"] == ["category", "island"]
    assert "params" not in layered["layer"][1]
    facet = {"facet": {"field": "island"}, "spec": spec}
    assert _with_click(facet) is facet
    assert _with_click(_with_click(spec)) == _with_click(spec)


def test_a_click_on_a_vega_lite_chart_selects_its_rows_and_shows_the_ask_box(chart_answer):
    from lumen.ai.selection import CLICK
    chart_answer.pipeline = chart_answer.view.pipeline
    card = chart_answer.__panel__()
    box = card[2]
    pane = chart_answer.view._panel
    assert not box.visible and CLICK in pane._selections

    pane.selection.param.update(**{CLICK: [{"category": "A"}]})
    assert list(chart_answer.selection["value"]) == [10.5, 7.5]
    assert box.visible and box[0].object == "**Selected:** 2 rows"

    pane.selection.param.update(**{CLICK: [{"category": "B"}]})
    assert box[0].object == "**Selected:** category: B, value: 20"
    pane.selection.param.update(**{CLICK: []})
    assert chart_answer.selection is None and not box.visible


def test_a_tap_on_an_hvplot_chart_selects_the_tapped_rows(frame_source):
    pipeline = Pipeline(source=frame_source, table="tiny")
    view = hvPlotView(pipeline=pipeline, kind="scatter", x="category", y="value")
    answer = Answer(question="?", status="success", pipeline=pipeline, view=view)
    answer.__panel__()
    stream = answer._streams[-1]
    stream.event(index=[1, 2])
    assert list(answer.selection["value"]) == [20.0, 7.5]
    stream.event(index=[])
    assert answer.selection is None


def test_a_tap_on_a_grid_selects_the_nearest_cell():
    _grid, answer = _grid_answer("SELECT lat, lon, AVG(t) AS t FROM t GROUP BY lat, lon")
    answer.view = hvPlotView(pipeline=answer.pipeline, kind="quadmesh", x="lon", y="lat", z="t")
    answer.__panel__()
    stream = answer._streams[-1]
    stream.event(x=2.2, y=19.0)
    (row,) = answer.selection.to_dict("records")
    assert (row["lon"], row["lat"]) == (2.0, 20.0)


async def test_ask_about_sends_the_rows_and_the_answer_before(llm, frame_source):
    llm.set_responses([_plan(("ChatAgent", "Explain")), "That is category A, at 10.5."])
    original = Answer(
        question="Values by category?", value="A and B.", sql="SELECT * FROM tiny", status="success",
        source=frame_source, pipeline=Pipeline(source=frame_source, table="tiny"), llm=llm,
        agents=[ChatAgent(), SQLAgent()],
    )
    sent = []
    invoke = llm.invoke

    async def recording(messages, *args, **kwargs):
        sent.append(messages)
        return await invoke(messages, *args, **kwargs)

    llm.invoke = recording
    answer = await original.aask_about("What is this?", rows=[0])
    assert answer.value == "That is category A, at 10.5." and answer.sql is None
    assert answer.question == "What is this?"
    history, latest = sent[0]
    # The planner folds the earlier turns into one history message
    assert "Values by category?" in history["content"] and "SELECT * FROM tiny" in history["content"]
    assert latest["content"].startswith("What is this?\n\nThis is about 1 row")
    assert "category,value\nA,10.5" in latest["content"]


async def test_ask_about_in_a_chat_joins_the_chat(frame_source):
    SQLModel = make_sql_model([(frame_source.name, "tiny")])
    llm = FollowUpLLM([
        SQLModel(query="SELECT category, SUM(value) AS total FROM tiny GROUP BY category", table_slug="totals", tables=["tiny"]),
        SQLModel(query="SELECT * FROM tiny WHERE category = 'A'", table_slug="only_a", tables=["tiny"]),
    ])
    conversation = chat(frame_source, llm=llm, agents=[ChatAgent(), SQLAgent()])
    first = await conversation.aask("Total value per category?")
    follow_up = await first.aask_about("Show its rows", rows={"category": "A", "total": 18.0})
    assert conversation.answers == [first, follow_up] and follow_up.question == "Show its rows"
    history, latest = llm.plan_calls[-1]
    assert "User: Total value per category?" in history["content"]
    assert latest["content"].startswith("Show its rows\n\nThis is about 1 row") and "A,18.0" in latest["content"]
    assert "category = 'A'" in follow_up.sql


def test_ask_about_needs_rows_and_shared_values(chart_answer):
    with pytest.raises(ValueError, match="Nothing is selected"):
        chart_answer.ask_about("What is this?")
    configure(share="schema")
    try:
        with pytest.raises(ValueError, match="share='schema' does not allow"):
            chart_answer.ask_about("What is this?", rows={"category": "A"})
    finally:
        configure(share="sample")


def test_ask_about_withholds_personal_columns():
    from lumen.ai.selection import about as _about
    configure(mask_pii=True)
    try:
        text = _about("Who?", pd.DataFrame({"email": ["a@b.com"], "value": [1]}))
    finally:
        configure(mask_pii=False)
    assert "a@b.com" not in text and "value\n1" in text


def test_the_static_card_has_no_ask_box(chart_answer):
    html_card = chart_answer._static_bundle(chart_answer._chart_bundle())["text/html"]
    assert "Ask about this" not in html_card


async def test_the_ask_box_asks_about_the_selection(chart_answer, monkeypatch):
    chart_answer.pipeline = chart_answer.view.pipeline
    asked = []

    async def aask_about(question, rows=None):
        asked.append((question, list(chart_answer.selection["category"])))
        return Answer(question=question, value="Category B.", status="success")

    monkeypatch.setattr(chart_answer, "aask_about", aask_about)
    box = chart_answer.__panel__()[2]
    chart_answer.selection = chart_answer.pipeline.data.iloc[[1]]
    _selected, (question, ask), answers = box
    question.value_input = "What is this?"
    ask.clicks += 1
    await async_wait_until(lambda: len(answers) == 1)
    assert asked == [("What is this?", ["B"])] and answers[0][0].object == "Category B."


def test_a_pick_on_a_lonboard_map_selects_the_row():
    pytest.importorskip("lonboard")
    answer = _geo_answer()
    m = answer.map()
    m.layers[0].selected_index = 1
    assert len(answer.selection) == 1 and answer.selection.index[0] == answer.gdf.index[1]


def test_the_click_selection_stays_out_of_the_spec_and_survives_a_redraw(chart_answer):
    from lumen.ai.selection import CLICK
    chart_answer.pipeline = chart_answer.view.pipeline
    chart_answer.__panel__()
    pane = chart_answer.view._panel
    assert "params" not in chart_answer.view.spec and "params" not in chart_answer._chart_spec()[1]
    # A redraw, e.g. on a filter, draws the spec without the selection again
    pane.object = {key: value for key, value in pane.object.items() if key != "params"}
    assert CLICK in pane._selections
    pane.selection.param.update(**{CLICK: [{"category": "B"}]})
    assert list(chart_answer.selection["category"]) == ["B"]


@pytest.fixture
def many_points():
    import numpy as np
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.random(6_000), "y": rng.random(6_000), "kind": rng.choice(["a", "b"], 6_000)})
    return Pipeline(source=read({"points": frame}), table="points")


def _point_spec(mark="point", color=True, **x):
    encoding = {"x": {"field": "x", "type": "quantitative", **x}, "y": {"field": "y", "type": "quantitative"}}
    if color:
        encoding["color"] = {"field": "kind", "type": "nominal"}
    return {"mark": mark, "encoding": encoding}


@pytest.mark.parametrize("mark, color, kind", [("point", True, "scatter"), ({"type": "circle"}, False, "scatter"), ("line", True, "line")])
def test_a_large_scatter_or_line_chart_is_drawn_as_a_density_image(many_points, mark, color, kind):
    pytest.importorskip("datashader")
    from lumen.ai.answer import _density_view
    view = _density_view(VegaLiteView(pipeline=many_points, spec=_point_spec(mark, color)), many_points)
    assert isinstance(view, hvPlotView) and view.kind == kind and (view.x, view.y) == ("x", "y")
    if color:
        assert view.datashade and view.by == ["kind"]
    else:
        assert view.kwargs["rasterize"] and not view.datashade
    import holoviews as hv
    assert isinstance(view.get_plot(view.get_data()), hv.DynamicMap)


@pytest.mark.parametrize("spec", [
    {"mark": "bar", "encoding": {"x": {"field": "kind"}, "y": {"field": "y", "aggregate": "mean"}}},
    _point_spec(aggregate="mean"),
    {"layer": [_point_spec()]},
])
def test_other_large_charts_stay_vega_lite(many_points, spec):
    from lumen.ai.answer import _density_view
    view = VegaLiteView(pipeline=many_points, spec=spec)
    assert _density_view(view, many_points) is view


def test_a_small_scatter_chart_stays_vega_lite(frame_source):
    from lumen.ai.answer import _density_view
    pipeline = Pipeline(source=frame_source, table="tiny")
    view = VegaLiteView(pipeline=pipeline, spec={"mark": "point", "encoding": {
        "x": {"field": "category", "type": "nominal"}, "y": {"field": "value", "type": "quantitative"}}})
    assert _density_view(view, pipeline) is view


def test_without_datashader_a_large_chart_stays_and_says_how_to_draw_it(many_points, monkeypatch):
    from lumen.ai.answer import _density_view
    real = answer_module.importlib.util.find_spec
    monkeypatch.setattr(answer_module.importlib.util, "find_spec", lambda name: None if name == "datashader" else real(name))
    view = VegaLiteView(pipeline=many_points, spec=_point_spec())
    assert _density_view(view, many_points) is view
    answer = Answer(question="?", status="success", pipeline=many_points, view=view, spec=view.spec)
    assert "draws all 6,000 points" in answer._cap_note and "Install datashader" in answer._cap_note


def test_an_answer_with_a_large_scatter_chart_gets_the_density_chart_and_a_note(many_points):
    pytest.importorskip("datashader")
    from lumen.ai.answer import _build_answer
    view = VegaLiteView(pipeline=many_points, spec=_point_spec())
    plan = Plan(title="Points")
    answer = _build_answer(
        "Where are the points?", plan, [VegaLiteEditor(component=view)], {"pipeline": many_points, "sql": "SELECT * FROM points"},
        many_points.source, MockLLM(), None, None,
    )
    assert isinstance(answer.view, hvPlotView) and answer.spec is None and answer._editor is None
    assert answer._cap_note == "The chart draws 6,000 points as a density image, blending the categories where they overlap."
    assert answer._cap_note in answer._static_bundle(None)["text/html"]


def test_a_tap_on_a_density_image_selects_the_nearest_point(many_points):
    pytest.importorskip("datashader")
    from lumen.ai.answer import _density_view
    view = _density_view(VegaLiteView(pipeline=many_points, spec=_point_spec(color=False)), many_points)
    answer = Answer(question="?", status="success", pipeline=many_points, view=view)
    answer.__panel__()
    stream = answer._streams[-1]
    data = many_points.data
    stream.event(x=data["x"].iloc[7], y=data["y"].iloc[7])
    assert answer.selection.index.tolist() == [7]


def test_an_edit_to_a_large_scatter_chart_is_drawn_as_a_density_image(many_points):
    pytest.importorskip("datashader")
    bars = VegaLiteView(pipeline=many_points, spec={"mark": "bar", "encoding": {
        "x": {"field": "kind"}, "y": {"field": "y", "aggregate": "mean"}}})
    answer = Answer(question="?", status="success", pipeline=many_points, view=bars, spec=bars.spec)
    redrawn = answer.with_spec(_point_spec())
    assert isinstance(redrawn.view, hvPlotView) and redrawn.view.datashade and redrawn.spec is None


def test_a_click_on_a_point_matches_its_numbers_as_numbers():
    from lumen.ai.selection import _matching
    df = pd.DataFrame({"bill": [58.0, 46.5], "species": ["Chinstrap", "Gentoo"], "year": [2007, 2008]})
    # Vega-Lite sends JSON numbers: a whole float arrives as 58, not 58.0
    assert _matching(df, {"bill": 58, "species": "Chinstrap"}).index.tolist() == [0]
    assert _matching(df, {"year": 2008}).index.tolist() == [1]
    dates = pd.DataFrame({"time": pd.to_datetime(["2024-06-01", "2024-07-01"])})
    assert _matching(dates, {"time": 1717200000000}).index.tolist() == [0]



@pytest.mark.parametrize("kind, callback", [("scatter", "Selection1DCallback"), ("density", "TapCallback")])
def test_the_drawn_hvplot_chart_reports_clicks(frame_source, many_points, kind, callback):
    """The view draws its plot again on render; the stream has to follow the plot that is shown."""
    if kind == "density":
        pytest.importorskip("datashader")
        from lumen.ai.answer import _density_view
        pipeline = many_points
        view = _density_view(VegaLiteView(pipeline=pipeline, spec=_point_spec()), pipeline)
    else:
        pipeline = Pipeline(source=frame_source, table="tiny")
        view = hvPlotView(pipeline=pipeline, kind="scatter", x="category", y="value")
    answer = Answer(question="?", status="success", pipeline=pipeline, view=view)
    answer.__panel__().get_root()
    plots = [plot for plot, _ in view._panel._plots.values()]
    assert any(callback in [type(c).__name__ for c in plot.callbacks] for plot in plots)
