from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import param
import pytest

from pydantic import BaseModel

from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.config import PROMPTS_DIR
from lumen.ai.translate import param_to_pydantic
from lumen.views.base import hvPlotBaseView


async def extract(spec, n_rows):
    """Run _extract_spec against a frame of the given size."""
    data = pd.DataFrame({"x": range(n_rows), "y": range(n_rows), "c": ["a"] * n_rows})
    with patch("lumen.ai.agents.hvplot.get_data", return_value=data):
        return await hvPlotAgent()._extract_spec(
            {"pipeline": SimpleNamespace(table="t")}, dict(spec)
        )


# ---- The LLM has to be able to express the spec ----

def test_datashade_params_reach_the_pydantic_schema():
    """The agent derives the LLM's schema from the view's params, so a param
    that param_to_pydantic cannot map is a field the model can never fill."""
    class Probe(param.Parameterized):
        by = hvPlotBaseView.param.by
        color_key = hvPlotBaseView.param.color_key
        datashade = hvPlotBaseView.param.datashade
        dynspread = hvPlotBaseView.param.dynspread

    models = param_to_pydantic(Probe, base_model=BaseModel, process_subclasses=False)
    properties = models["Probe"].model_json_schema()["properties"]

    assert {"datashade", "dynspread", "color_key"} <= set(properties)


# ---- The large-frame default has to preserve the category ----

async def test_large_categorical_frame_uses_datashade():
    """rasterize reduces each pixel to one number, which discards the category."""
    spec = await extract({"kind": "points", "x": "x", "y": "y", "by": ["c"]}, 20_001)

    assert spec["datashade"] is True
    assert "rasterize" not in spec
    assert "cnorm" not in spec


async def test_large_frame_without_by_still_rasterizes():
    spec = await extract({"kind": "points", "x": "x", "y": "y"}, 20_001)

    assert spec["rasterize"] is True
    assert spec["cnorm"] == "log"
    assert "datashade" not in spec


async def test_small_categorical_frame_is_left_alone():
    """Under the threshold every point is drawn, so neither operation applies."""
    spec = await extract({"kind": "points", "x": "x", "y": "y", "by": ["c"]}, 10)

    assert "datashade" not in spec
    assert "rasterize" not in spec


@pytest.mark.parametrize("kind", ["bar", "heatmap", "hist"])
async def test_non_point_kinds_are_untouched(kind):
    spec = await extract({"kind": kind, "x": "x", "y": "y", "by": ["c"]}, 20_001)

    assert "datashade" not in spec
    assert "rasterize" not in spec


# ---- The prompt has to mention it, or the model never tries ----

def test_prompt_documents_categorical_datashading():
    prompt = (PROMPTS_DIR / "hvPlotAgent" / "main.jinja2").read_text()

    assert "datashade" in prompt
    assert "color_key" in prompt


# ---- The schema the LLM is actually handed ----

def test_get_model_builds_the_view_schema():
    """_get_model walked the whole component tree until it hit a param type it
    could not map, so no hvPlot view schema could be built at all."""
    schema = {
        "lon": {"type": "number"},
        "lat": {"type": "number"},
        "family": {"type": "string", "enum": ["Irish", "Italian"]},
    }

    model = hvPlotAgent()._get_model("main", schema)
    properties = model.model_json_schema()["properties"]

    assert {"kind", "x", "y", "by", "datashade", "dynspread", "color_key"} <= set(properties)


def test_get_model_omits_excluded_names():
    """The generated models inherit, so anything excluded has to stay out of the
    parent models too."""
    model = hvPlotAgent()._get_model("main", {"lon": {"type": "number"}})
    properties = model.model_json_schema()["properties"]

    excluded = {"pipeline", "source", "transforms", "download", "controls",
                "field", "selection_group"}

    assert not (excluded & set(properties))
