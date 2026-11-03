import asyncio
import base64
import logging
import subprocess
import sys
import threading
import time

from typing import get_args
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.config import panel_extension
from panel.tests.util import async_wait_until

from lumen.ai.actor import NullStep
from lumen.ai.agents import ChatAgent, SQLAgent, VegaLiteAgent
from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.answer import (
    Answer, _default_agents, _default_llm, _planner, _planners, _resolve, _run,
    _StatusLine, aask, ask, chat,
)
from lumen.ai.config import QueryError
from lumen.ai.coordinator import Plan
from lumen.ai.coordinator.planner import make_plan_model
from lumen.ai.data import read
from lumen.ai.editors import VegaLiteEditor
from lumen.ai.models import ReplaceLine, RetrySpec
from lumen.ai.report import ActorTask
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.filters.base import ConstantFilter
from lumen.pipeline import Pipeline
from lumen.tests.ai.conftest import MockLLM
from lumen.tests.utils import Polygon, gpd
from lumen.util import log as lumen_log
from lumen.views.base import VegaLiteView

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


async def test_aask_without_a_provider_explains_the_options(frame_source):
    _default_llm.cache_clear()
    with patch("lumen.ai.answer.get_available_llm", return_value=None):
        with pytest.raises(RuntimeError, match=r"No LLM provider found\. Pass llm=lumen\.ai\.llm\.Kilo"):
            await aask(frame_source, "Total value?")


def test_default_llm_is_shared_between_questions(llm):
    _default_llm.cache_clear()
    with patch("lumen.ai.answer.get_available_llm", return_value=type(llm)):
        assert _default_llm() is _default_llm()
    _default_llm.cache_clear()


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


def test_default_agents_add_hvplot_for_gridded_data():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_sql")
    ds = xr.Dataset({"temp": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    assert any(isinstance(a, hvPlotAgent) for a in _default_agents(read(ds)))


def test_default_agents_add_hvplot_for_geometry():
    if gpd is None:
        pytest.skip("geopandas is not installed")
    gdf = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1)])])
    assert any(isinstance(a, hvPlotAgent) for a in _default_agents(read({"shapes": gdf})))


def test_default_agents_add_hvplot_when_a_table_has_geometry(frame_source):
    with patch("lumen.ai.answer.geometry_columns", return_value=["geometry"]):
        assert any(isinstance(a, hvPlotAgent) for a in _default_agents(frame_source))


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
        frame_source.ask("q", llm="x")
        pipeline.ask("q")
        await frame_source.aask("q")
        await pipeline.aask("q", llm="x")
    assert [c.args[0] for c in sync_ask.call_args_list] == [frame_source, pipeline]
    assert sync_ask.call_args_list[0].kwargs == {"llm": "x"}
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
    assert bundle["text/markdown"] == "**B** is largest.\nA is second.\n\n```sql\nSELECT * FROM tiny\n```"
    spec = bundle["application/vnd.vegalite.v5+json"]
    assert spec["data"]["values"][0] == {"category": "A", "value": 10.5}
    assert base64.b64decode(bundle["image/png"]).startswith(b"\x89PNG")
    html = bundle["text/html"]
    assert "<strong>B</strong> is largest." in html
    assert f'src="data:image/png;base64,{bundle["image/png"]}" width="600"' in html
    assert "<details><summary>SQL</summary><pre><code>SELECT * FROM tiny</code></pre></details>" in html


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
        panel_extension._loaded_extensions.append("vega")

    with (
        patch("lumen.ai.answer._in_kernel", return_value=True),
        patch.object(panel_extension, "_loaded", True),
        patch.object(panel_extension, "_loaded_extensions", []),
        patch("lumen.ai.answer.pn.extension", side_effect=load) as extension,
        patch.object(type(chart_answer.__panel__()), "_repr_mimebundle_", return_value=({"text/html": "card"}, {})),
    ):
        assert chart_answer._repr_mimebundle_() == ({"text/html": "card"}, {})
        chart_answer._repr_mimebundle_()
    extension.assert_called_once_with("vega")


def test_answer_falls_back_to_separate_outputs_when_panel_cannot_render(chart_answer):
    pytest.importorskip("IPython")
    with (
        patch("lumen.ai.answer._in_kernel", return_value=True),
        patch.object(panel_extension, "_loaded_extensions", ["vega"]),
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
    text, _chart, sql = card
    assert text.object.startswith("**B** is largest.")
    assert sql.collapsed and "SELECT * FROM tiny" in sql[0].object


def test_answer_card_skips_empty_text(chart_answer):
    chart_answer.value = ""
    assert len(chart_answer.__panel__()) == 2


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
    with pytest.raises(RuntimeError, match="answer.code holds the code"):
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
