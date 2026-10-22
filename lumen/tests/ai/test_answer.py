from typing import get_args
from unittest.mock import patch

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.agents import ChatAgent, SQLAgent
from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.answer import (
    Answer, _default_agents, _resolve, aask,
)
from lumen.ai.coordinator.planner import make_plan_model
from lumen.ai.data import read
from lumen.ai.models import ThinkingYesNo
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.filters.base import ConstantFilter
from lumen.pipeline import Pipeline
from lumen.tests.utils import Polygon, gpd

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
        ThinkingYesNo(chain_of_thought="Clear question", yes=False),
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
        ThinkingYesNo(chain_of_thought="Clear question", yes=False),
        _plan(("ChatAgent", "Explain")),
        fail, fail, fail, fail,
    ])
    with pytest.raises(RuntimeError, match="could not answer 'Total value\\?'"):
        await aask(frame_source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])


async def test_aask_without_a_provider_explains_the_options(frame_source):
    with patch("lumen.ai.answer.get_available_llm", return_value=None):
        with pytest.raises(RuntimeError, match="No LLM provider found"):
            await aask(frame_source, "Total value?")


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
