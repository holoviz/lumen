"""The hvPlot agent could not produce a renderable spec at all.

Each test here stands for one step of that failure: the schema offered the
model a value its own view rejects, the recovery path raised instead of
recovering, and the one field the agent adds to the schema reached the view.
"""
import pandas as pd
import param
import pytest

from pydantic import BaseModel

from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.translate import param_to_pydantic
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.views.base import hvPlotUIView


@pytest.fixture
def pipeline():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    return Pipeline(source=InMemorySource(tables={"points": df}), table="points")


# ---- A Selector holding None offered the model the string "None" ----

class _Aggregating(param.Parameterized):

    aggregator = param.Selector(default=None, objects=[None, "count", "mean"], doc="""
        How to reduce the rows landing in one pixel.""")


def test_a_none_option_is_not_offered_as_a_string():
    """param names the None option 'None' for display, and that name was being
    handed to the model as though it were a value."""
    model = param_to_pydantic(_Aggregating, base_model=BaseModel)[_Aggregating.__name__]

    enum = model.model_json_schema()["properties"]["aggregator"]["enum"]

    assert "None" not in enum
    assert set(enum) >= {"count", "mean"}


def test_an_optional_selector_is_not_required():
    """Requiring it leaves the model no way to say 'no aggregation', which is
    what its default means."""
    model = param_to_pydantic(_Aggregating, base_model=BaseModel)[_Aggregating.__name__]

    schema = model.model_json_schema()

    assert "aggregator" not in schema.get("required", [])
    assert schema["properties"]["aggregator"]["default"] is None


def test_the_view_accepts_every_aggregator_the_schema_offers(pipeline):
    """The schema and the view have to agree, or a valid answer fails to render."""
    model = hvPlotAgent()._get_model("main", {"x": {}, "y": {}})
    enum = model.model_json_schema()["properties"]["aggregator"]["enum"]

    for aggregator in enum:
        hvPlotUIView(pipeline=pipeline, kind="scatter", x="x", y="y", aggregator=aggregator)


# ---- The agent's own extra field reached the view ----

async def test_chain_of_thought_does_not_reach_the_view(pipeline):
    """It is added to the schema to make the model reason, not to be plotted;
    hvPlotExplorer rejects any keyword none of its controls claims."""
    spec = {"kind": "scatter", "x": "x", "y": "y", "chain_of_thought": "because"}

    extracted = await hvPlotAgent()._extract_spec({"pipeline": pipeline}, spec)

    assert "chain_of_thought" not in extracted
    hvPlotUIView(pipeline=pipeline, **extracted).get_panel()


# ---- The recovery path raised instead of recovering ----

def test_a_prompt_other_than_main_still_resolves_its_model():
    """_get_model is called for every prompt, so overriding it with a required
    extra argument broke revise, which is what runs when a spec fails."""
    assert hvPlotAgent()._get_model("revise_output") is not None


class _Colormapped(param.Parameterized):

    cmap = param.Selector(default="viridis", objects=["viridis", "fire"], check_on_set=False, doc="""
        The colormap to use.""")


def test_a_value_outside_the_choices_does_not_break_the_model():
    """A Selector that accepts more than it lists (hvPlot reads a dict cmap as a
    color key) puts that value in range, and it is not a Literal member."""
    instance = _Colormapped(cmap={"Irish": "#e41a1c"})

    model = param_to_pydantic(type(instance), base_model=BaseModel)[type(instance).__name__]

    assert set(model.model_json_schema()["properties"]["cmap"]["enum"]) == {"viridis", "fire"}
