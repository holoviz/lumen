import asyncio
import base64
import json
import logging
import subprocess
import sys
import threading
import time

from typing import get_args
from unittest.mock import AsyncMock, MagicMock, patch

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
from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.answer import (
    _LOOP_LOCK, Answer, _after_fork, _busy, _default_agents, _default_llm,
    _drop_exceptions, _env_llm, _loop, _planner, _planners, _records, _resolve,
    _row_cap_hit, _run, _StatusLine, aask, ask, chat, configure,
)
from lumen.ai.config import QueryError, RetriesExceededError
from lumen.ai.coordinator import Plan
from lumen.ai.coordinator.planner import make_plan_model
from lumen.ai.editors import VegaLiteEditor
from lumen.ai.models import ReplaceLine, RetrySpec
from lumen.ai.report import ActorTask, TaskGroup
from lumen.ai.sharing import sharing
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.filters.base import ConstantFilter
from lumen.pipeline import Pipeline
from lumen.sources.data import read
from lumen.tests.ai.conftest import MockLLM
from lumen.tests.utils import Polygon, gpd
from lumen.transforms.sql import SQLLimit
from lumen.util import log as lumen_log
from lumen.views.base import VegaLiteView, hvPlotView

QUESTION = [{"role": "user", "content": "Total value?"}]


def _plan(*steps):
    PlanModel = make_plan_model(["ChatAgent", "SQLAgent"], [])
    (StepModel,) = get_args(PlanModel.__annotations__["steps"])
    return PlanModel(
        chain_of_thought="Query then explain",
        title="Total value",
        steps=[StepModel(actor=actor, instruction=title, title=title) for actor, title in steps],
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


async def test_a_plan_that_neither_queries_nor_answers_is_an_error(llm, frame_source):
    llm.set_responses([_plan(("ChatAgent", "Explain")), ""])

    answer = await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])

    assert answer.status == "error"
    assert "without querying the data" in answer.error
    assert answer.sql is None and answer.view is None


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

    alert, sql = answer.__panel__()
    assert alert.severity == "error" and "column nope" in alert.object
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
    _alert, sql = answer.__panel__()
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
    monkeypatch.setattr(answer_module, "_settings", {"agents": None, "llm": None})
    monkeypatch.delenv("LUMEN_AI_PROVIDER", raising=False)
    monkeypatch.delenv("LUMEN_AI_MODEL", raising=False)
    _env_llm.cache_clear()
    max_rows = SQLAgent.max_rows
    yield
    _env_llm.cache_clear()
    # Plain assignment: monkeypatch would restore the Parameter object itself
    SQLAgent.max_rows = max_rows
    sharing.param.update(share="sample", mask_pii=False)


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
    assert configure() == {"agents": agents, "llm": None, "mask_pii": False, "max_rows": 1_000_000, "share": "sample"}


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


async def test_a_result_of_exactly_max_rows_is_not_reported_as_cut(ten_rows):
    configure(max_rows=10)
    answer = await _ask_every_row(ten_rows)
    assert len(answer.pipeline.data) == 10 and answer.truncated_at is None


def test_a_failing_cap_check_does_not_break_the_answer(ten_rows):
    pipeline = Pipeline(source=ten_rows, table="rows", sql_transforms=[SQLLimit(limit=10)])
    assert len(pipeline.data) == 10
    with patch.object(type(ten_rows), "get", side_effect=RuntimeError("warehouse down")):
        assert _row_cap_hit(pipeline) is None


def test_chart_spec_leaves_out_inline_data_at_any_depth():
    from lumen.ai.answer import _without_values

    spec = {"data": {"values": [1]}, "layer": [{"data": {"values": [2], "name": "x"}, "mark": "bar"}], "datasets": {"d": [3]}}
    assert _without_values(spec) == {"data": {}, "layer": [{"data": {"name": "x"}, "mark": "bar"}]}


async def test_chat_forgets_context_built_under_another_sharing_level(llm, frame_source):
    conversation = chat(frame_source, llm=llm, agents=[ChatAgent(), SQLAgent()])
    _chat_only(llm)
    await conversation.aask("How many rows?")
    assert conversation._context is not None
    configure(share="schema")
    _chat_only(llm)
    await conversation.aask("And again?")
    assert [m["role"] for m in conversation._messages] == ["user", "user", "assistant"]


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


async def test_planner_is_rebuilt_for_another_llm_or_explicit_agents(llm, frame_source):
    first = await _planner(frame_source, llm, None)
    assert await _planner(frame_source, llm, None) is first
    assert await _planner(frame_source, type(llm)(), None) is not first
    assert await _planner(frame_source, llm, [ChatAgent()]) is not await _planner(frame_source, llm, [ChatAgent()])


async def test_a_busy_planner_is_not_shared(llm, frame_source):
    first = await _planner(frame_source, llm, None)
    _busy.add(first)
    try:
        assert await _planner(frame_source, llm, None) is not first
    finally:
        _busy.discard(first)
    assert await _planner(frame_source, llm, None) is first


async def test_default_agents_plain_table(frame_source):
    agents = await _default_agents(frame_source)
    assert [type(a).__name__ for a in agents] == ["ChatAgent", "SQLAgent", "VegaLiteAgent"]


async def test_default_agents_add_hvplot_for_gridded_data():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_sql")
    ds = xr.Dataset({"temp": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    assert any(isinstance(a, hvPlotAgent) for a in await _default_agents(read(ds)))


async def test_default_agents_add_hvplot_for_geometry():
    if gpd is None:
        pytest.skip("geopandas is not installed")
    gdf = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1)])])
    assert any(isinstance(a, hvPlotAgent) for a in await _default_agents(read({"shapes": gdf})))


async def test_default_agents_add_hvplot_when_a_table_has_geometry(frame_source):
    with patch("lumen.ai.answer.geometry_columns", return_value=["geometry"]):
        assert any(isinstance(a, hvPlotAgent) for a in await _default_agents(frame_source))


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
    assert _resolve(pipeline)[0] is source


def test_filter_change_drops_the_cached_planner(frame_source):
    pipeline = Pipeline(source=frame_source, table="tiny", filters=[ConstantFilter(field="category", value="A")])
    source, _ = _resolve(pipeline)
    _planners[source] = MagicMock()
    pipeline.filters[0].value = "B"
    assert source not in _planners
    assert _resolve(pipeline)[0] is source


def test_table_change_rebuilds_the_filtered_source():
    source = read({
        "tiny": pd.DataFrame({"category": ["A", "B"], "value": [1, 2]}),
        "other": pd.DataFrame({"category": ["A", "A"], "value": [3, 4]}),
    })
    pipeline = Pipeline(source=source, table="tiny", filters=[ConstantFilter(field="category", value="A")])
    first, _ = _resolve(pipeline)
    pipeline.table = "other"
    second, _ = _resolve(pipeline)
    assert second is not first
    assert second.get_tables() == ["other"]
    assert sorted(second.get("other")["value"]) == [3, 4]


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
    with patch("lumen.ai.answer._aask_messages", side_effect=ValueError("boom")):
        with pytest.raises(ValueError) as info:
            ask(frame_source, "q")
    assert info.value.__context__ is None


async def test_aask_quiets_instructor_retry_log_only_while_asking(llm, frame_source, monkeypatch, caplog):
    # ExplorerUI tests in the same process leave Lumen's logger at DEBUG
    monkeypatch.setattr(lumen_log, "level", logging.WARNING)
    logger = logging.getLogger("instructor.v2.retry")

    def reply():
        logger.error("retrying")
        return "Three rows."

    llm.set_responses([_plan(("ChatAgent", "Explain")), reply])
    with caplog.at_level(logging.ERROR, logger="instructor.v2.retry"):
        await aask(frame_source, "How many rows?", llm=llm, agents=[ChatAgent(), SQLAgent()])
        logger.error("outside")
    assert caplog.messages == ["outside"]


def test_sync_ask_runs_every_question_on_one_loop(llm, frame_source):
    loops = []

    def reply():
        loops.append(asyncio.get_running_loop())
        return "Three rows."

    for _ in range(2):
        llm.set_responses([_plan(("ChatAgent", "Explain")), reply])
        ask(frame_source, "How many rows?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert loops[0] is loops[1]


async def test_aask_runs_on_the_loop_sync_ask_uses(llm, frame_source):
    loops = []

    def reply():
        loops.append(asyncio.get_running_loop())
        return "Three rows."

    llm.set_responses([_plan(("ChatAgent", "Explain")), reply])
    await aask(frame_source, "How many rows?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert loops == [_loop()]


def test_ask_from_lumens_loop_raises_instead_of_deadlocking(frame_source):
    async def nested():
        ask(frame_source, "q")

    with pytest.raises(RuntimeError, match="await aask"):
        asyncio.run_coroutine_threadsafe(nested(), _loop()).result(timeout=5)


def test_fork_resets_the_loop_and_caches(frame_source):
    loop = _loop()
    _planners[frame_source] = MagicMock()
    _LOOP_LOCK.acquire()  # as the before-fork hook does
    _after_fork()
    assert frame_source not in _planners
    assert _loop() is not loop
    loop.call_soon_threadsafe(loop.stop)


def test_interrupting_sync_ask_cancels_the_question(frame_source):
    future = MagicMock()
    future.result.side_effect = KeyboardInterrupt
    with patch("lumen.ai.answer._aask_messages", new=MagicMock()), \
            patch("lumen.ai.answer.asyncio.run_coroutine_threadsafe", return_value=future):
        with pytest.raises(KeyboardInterrupt):
            ask(frame_source, "q")
    future.cancel.assert_called_once()


def test_answer_html_escapes_html_in_the_answer():
    html = Answer(value="Hi <script>alert(1)</script>")._static_bundle(None)["text/html"]
    assert "<script>" not in html


async def test_ask_inside_a_running_loop(llm, frame_source):
    _chat_only(llm)
    answer = ask(frame_source, "How many rows?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert answer.value == "Three rows."


async def test_source_and_pipeline_methods_delegate(frame_source):
    pipeline = Pipeline(source=frame_source, table="tiny")
    with patch("lumen.ai.answer.ask") as sync_ask, patch("lumen.ai.answer.aask", new_callable=AsyncMock) as async_ask:
        frame_source.ask("q", llm="x")
        pipeline.ask("q")
        await frame_source.aask("q")
        await pipeline.aask("q", llm="x")
    assert [c.args[0] for c in sync_ask.call_args_list] == [frame_source, pipeline]
    assert sync_ask.call_args_list[0].kwargs == {"llm": "x", "agents": None}
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


async def test_refined_answer_explores_with_its_new_chart(refinable_answer, llm):
    _line_edit(llm, 8, "mark: line")
    refined = await refinable_answer.arefine("Use a line")
    assert refined._views == [refined._editor]
    assert refinable_answer._editor not in refined._views


async def test_retrying_a_chat_answer_keeps_the_conversation(llm, frame_source):
    conversation = chat(frame_source, llm=llm, agents=[ChatAgent(), SQLAgent()])
    _chat_only(llm)
    await conversation.aask("How many rows?")
    _chat_only(llm)
    second = await conversation.aask("Only A")

    _chat_only(llm)
    with patch("lumen.ai.answer._respond", new=AsyncMock(wraps=answer_module._respond)) as respond:
        retried = await second.aretry()

    messages = respond.call_args.args[2]
    assert [m["content"] for m in messages if m["role"] == "user"] == ["How many rows?", "Only A"]
    assert conversation.answers[-1] is retried and len(conversation.answers) == 3


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
    text, _chart, sql, spec = card
    assert text.object.startswith("**B** is largest.")
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


def test_answer_without_a_chart_has_no_spec(frame_source):
    answer = Answer(question="?", value="Ten rows.", sql="SELECT 1", status="success")
    assert answer._chart_spec() is None
    assert [part.title for part in answer.__panel__() if isinstance(part, Card)] == ["SQL"]
    assert "Spec" not in answer._static_bundle(None)["text/html"]


def test_answer_card_skips_empty_text(chart_answer):
    chart_answer.value = ""
    assert len(chart_answer.__panel__()) == 3


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


async def test_answer_explore_keeps_a_pipeline_answer_to_its_table(llm):
    source = read({"tiny": pd.DataFrame({"value": [1.0]}), "other": pd.DataFrame({"x": [1]})})
    llm.set_responses([_plan(("ChatAgent", "Explain")), "One row.", "One row."])
    answer = await aask(Pipeline(source=source, table="tiny"), "How many rows?", llm=llm, agents=[ChatAgent(), SQLAgent()])

    ui = answer.explore()
    await async_wait_until(lambda: len(ui._explorations.items) == 2)
    assert ui.context["visible_slugs"] == {f"{source.name}{SOURCE_TABLE_SEPARATOR}tiny"}


def test_answer_code_explains_joined_grids_cannot_be_rebuilt():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_sql")
    grid = xr.Dataset({"temp": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    source = read({"climate": grid, "tiny": pd.DataFrame({"x": [0]})})
    answer = Answer(source=source, pipeline=Pipeline(source=source, table="climate"))
    with pytest.raises(ValueError, match="joins xarray data"):
        answer.code


async def test_answer_code_explains_joined_grids_after_a_chat_follow_up():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_sql")
    grid = xr.Dataset({"temp": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    source = read({"climate": grid, "tiny": pd.DataFrame({"x": [0]})})
    SQLModel = make_sql_model([(source.name, "climate")])
    llm = FollowUpLLM([
        SQLModel(query="SELECT x, temp FROM climate", table_slug="temps", tables=["climate"]),
        SQLModel(query="SELECT * FROM climate WHERE temp > 1", table_slug="warm", tables=["climate"]),
    ])
    conversation = source.chat(llm=llm, agents=[ChatAgent(), SQLAgent()])
    await conversation.aask("Temperatures?")
    follow_up = await conversation.aask("Only the warm ones")
    assert follow_up.source is not source
    with pytest.raises(ValueError, match="joins xarray data"):
        follow_up.code


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
        with NullStep().add("Generating SQL..."):
            pass
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

    _loop()  # started once, on the first sync ask
    threads = threading.active_count()
    start = time.monotonic()
    with patch.object(_StatusLine, "show", side_effect=KeyboardInterrupt), pytest.raises(KeyboardInterrupt):
        _run(question())

    # The interrupt returns at once, and the task is cancelled on the loop
    assert time.monotonic() - start < 5
    await async_wait_until(lambda: ended == [1])
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


def test_records_turn_every_missing_value_into_null():
    frame = pd.DataFrame({
        "f32": pd.Series([1.0, float("nan")], dtype="float32"),
        "ints": pd.Series([1, None], dtype="Int64"),
    })
    assert _records(frame) == [{"f32": 1.0, "ints": 1}, {"f32": None, "ints": None}]


def test_failed_answers_drop_exceptions_in_nested_task_groups():
    inner = ActorTask(ChatAgent(), title="Explain")
    inner.out_context = {"__exception__": ValueError("boom"), "sql": "SELECT 1"}
    _drop_exceptions(TaskGroup(TaskGroup(inner, title="Inner"), title="Outer"))
    assert inner.out_context == {"sql": "SELECT 1"}
