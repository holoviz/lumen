import base64
import subprocess
import sys

from typing import get_args
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.config import panel_extension

from lumen.ai.agents import ChatAgent, SQLAgent
from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.answer import (
    Answer, _default_agents, _default_llm, _resolve, aask, ask,
)
from lumen.ai.coordinator.planner import make_plan_model
from lumen.ai.data import read
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.filters.base import ConstantFilter
from lumen.pipeline import Pipeline
from lumen.tests.utils import Polygon, gpd
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


async def test_aask_raises_when_the_plan_fails(llm, frame_source):
    def fail():
        raise RuntimeError("LLM unavailable")

    llm.set_responses([
        _plan(("ChatAgent", "Explain")),
        fail, fail, fail, fail,
    ])
    with pytest.raises(RuntimeError, match="could not answer 'Total value\\?'"):
        await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])


async def test_aask_without_a_provider_explains_the_options(frame_source):
    _default_llm.cache_clear()
    with patch("lumen.ai.answer.get_available_llm", return_value=None):
        with pytest.raises(RuntimeError, match="No LLM provider found"):
            await aask(frame_source, "Total value?")


def test_default_llm_is_shared_between_questions(llm):
    _default_llm.cache_clear()
    with patch("lumen.ai.answer.get_available_llm", return_value=type(llm)):
        assert _default_llm() is _default_llm()
    _default_llm.cache_clear()


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


def test_answer_repr_is_one_line(chart_answer):
    assert repr(chart_answer) == "Answer(status='success', value='**B** is largest.')"


def test_answer_static_bundle_outside_a_kernel(chart_answer):
    bundle = chart_answer._repr_mimebundle_()
    assert bundle["text/markdown"] == "**B** is largest.\nA is second.\n\n```sql\nSELECT * FROM tiny\n```"
    spec = bundle["application/vnd.vegalite.v5+json"]
    assert spec["data"]["values"][0] == {"category": "A", "value": 10.5}
    assert base64.b64decode(bundle["image/png"]).startswith(b"\x89PNG")


def test_answer_static_spec_keeps_data_given_by_url(frame_source):
    url = {"url": "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json"}
    view = VegaLiteView(
        pipeline=Pipeline(source=frame_source, table="tiny"),
        spec={"data": url, "mark": "point", "encoding": {"x": {"field": "Horsepower", "type": "quantitative"}}},
    )
    answer = Answer(question="q", value="v", spec=view.spec, view=view, status="success")
    assert answer._vega_lite_spec()["data"] == url


def test_answer_static_bundle_without_a_chart():
    answer = Answer(question="q", value="Three rows.", status="success")
    assert set(answer._repr_mimebundle_()) == {"text/plain", "text/markdown"}


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


def test_answer_falls_back_when_panel_cannot_render(chart_answer):
    with (
        patch("lumen.ai.answer._in_kernel", return_value=True),
        patch.object(panel_extension, "_loaded_extensions", ["vega"]),
        patch.object(type(chart_answer.__panel__()), "_repr_mimebundle_", side_effect=ImportError("jupyter_bokeh")),
    ):
        assert "application/vnd.vegalite.v5+json" in chart_answer._repr_mimebundle_()


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
