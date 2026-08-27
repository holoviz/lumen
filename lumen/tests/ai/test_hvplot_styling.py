"""What the generated styling options cost the plot agent's prompt.

The view-level half of this lives in lumen/tests/views/test_hvplot_styling.py,
which must not import the AI stack.
"""
import json

from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.config import PROMPTS_DIR
from lumen.views.base import HVPLOT_STYLE_PARAMS


def test_style_params_reach_the_agent_schema():
    schema = {"lon": {"type": "number"}, "lat": {"type": "number"}}

    properties = hvPlotAgent()._get_model("main", schema).model_json_schema()["properties"]

    assert set(HVPLOT_STYLE_PARAMS) <= set(properties)
    for name in HVPLOT_STYLE_PARAMS:
        assert properties[name].get("description"), name


def test_the_schema_stays_affordable():
    """The whole point of an allowlist is that it does not grow by accident.

    Raise this number deliberately, having looked at what was added.
    """
    schema = {"lon": {"type": "number"}, "lat": {"type": "number"},
              "family": {"type": "string"}, "pop": {"type": "number"}}

    rendered = json.dumps(hvPlotAgent()._get_model("main", schema).model_json_schema())

    assert len(rendered) <= 7000


def test_prompt_asks_for_styling_only_when_requested():
    """The options are worth having only if the model leaves them alone by
    default, so the prompt has to say so."""
    prompt = (PROMPTS_DIR / "hvPlotAgent" / "main.jinja2").read_text()

    assert "only when the request asks" in prompt
