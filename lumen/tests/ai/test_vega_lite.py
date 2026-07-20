import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip(
        "lumen.ai could not be imported, skipping tests.",
        allow_module_level=True,
    )

import vl_convert

from lumen.ai.agents.vega_lite import VegaLiteAgent, category_palette
from lumen.ai.editors import VegaLiteEditor
from lumen.ai.utils import normalize_vegalite_spec
from lumen.config import dump_yaml


def test_normalize_vegalite_spec_adds_schema_and_container_sizing():
    """A bare spec gains $schema and container sizing and stays valid."""
    raw = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"},
        },
    }

    result = normalize_vegalite_spec(raw)

    spec = result["spec"]
    assert spec["$schema"] == "https://vega.github.io/schema/vega-lite/v5.json"
    assert spec["width"] == "container"
    assert spec["height"] == "container"
    assert result["sizing_mode"] == "stretch_both"
    assert result["min_height"] == 200
    # Normalized spec must pass the editor's own validation.
    VegaLiteEditor.validate_spec(spec)


def test_normalize_vegalite_spec_strips_sizing_for_compound_charts():
    """Compound charts drop top-level width/height (sub-charts size themselves)."""
    raw = {
        "hconcat": [
            {
                "data": {"values": [{"A": 1, "B": 2}]},
                "mark": "bar",
                "encoding": {
                    "x": {"field": "A", "type": "quantitative"},
                    "y": {"field": "B", "type": "quantitative"},
                },
            }
        ],
        "width": 400,
        "height": 300,
    }

    spec = normalize_vegalite_spec(raw)["spec"]

    assert "width" not in spec
    assert "height" not in spec
    assert spec["$schema"] == "https://vega.github.io/schema/vega-lite/v5.json"


def test_normalize_vegalite_spec_adds_geographic_interactivity():
    """Specs with lat/long encodings gain a projection, zoom params and map layer."""
    raw = {
        "mark": "circle",
        "encoding": {
            "latitude": {"field": "lat", "type": "quantitative"},
            "longitude": {"field": "lon", "type": "quantitative"},
        },
    }

    spec = normalize_vegalite_spec(raw)["spec"]

    assert spec["projection"]["type"] == "mercator"
    assert "scale" in {p.get("name") for p in spec["params"]}
    assert "layer" in spec


CATEGORICAL_SPEC = {
    "mark": "bar",
    "encoding": {
        "x": {"field": "c", "type": "nominal"},
        "y": {"field": "v", "type": "quantitative"},
        "color": {"field": "c", "type": "nominal"},
    },
}


async def test_palette_supplied_as_a_default_the_model_can_override(llm):
    """Charts share a palette so a report does not shift colors between views,
    but a chart that chose its own colors keeps them."""
    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    spec = (await agent._extract_spec({}, {"yaml_spec": dump_yaml(dict(CATEGORICAL_SPEC))}))["spec"]
    assert spec["config"]["range"]["category"] == category_palette()

    chosen = dict(CATEGORICAL_SPEC, config={"range": {"category": ["#111111"]}, "view": {"stroke": None}})
    spec = (await agent._extract_spec({}, {"yaml_spec": dump_yaml(chosen)}))["spec"]
    assert spec["config"]["range"]["category"] == ["#111111"]
    assert spec["config"]["view"] == {"stroke": None}


async def test_palette_only_added_where_it_can_be_used(llm):
    """A palette is only meaningful for categorical color, so charts that do not
    use it are left alone rather than carrying twenty unused colors."""
    agent = VegaLiteAgent(llm=llm, code_execution="disabled")
    axes = {"x": {"field": "c", "type": "nominal"}, "y": {"field": "v", "type": "quantitative"}}

    for encoding in (axes, dict(axes, color={"field": "v", "type": "quantitative"})):
        chart = {"mark": "bar", "encoding": encoding}
        spec = (await agent._extract_spec({}, {"yaml_spec": dump_yaml(chart)}))["spec"]
        assert "range" not in spec.get("config", {})

    layered = {"layer": [{"mark": "bar", "encoding": CATEGORICAL_SPEC["encoding"]}]}
    spec = (await agent._extract_spec({}, {"yaml_spec": dump_yaml(layered)}))["spec"]
    assert spec["config"]["range"]["category"] == category_palette()


async def test_palette_not_reapplied_once_the_chart_exists(llm):
    """Annotating or polishing a chart must not restore a palette the user
    removed, so the defaults are only supplied on the way in."""
    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    spec = (await agent._extract_spec(
        {}, {"yaml_spec": dump_yaml(dict(CATEGORICAL_SPEC))}, apply_defaults=False
    ))["spec"]

    assert "range" not in spec.get("config", {})


def test_palette_reaches_the_compiled_vega_scale():
    """The palette is only useful if Vega resolves it, which it does through the
    named 'category' range rather than by inlining the colors in the scale."""
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": [{"c": "a", "v": 1}, {"c": "b", "v": 2}]},
        "mark": "bar",
        "encoding": {
            "x": {"field": "c", "type": "nominal"},
            "y": {"field": "v", "type": "quantitative"},
            "color": {"field": "c", "type": "nominal"},
        },
        "config": {"range": {"category": category_palette()}},
    }

    compiled = vl_convert.vegalite_to_vega(spec)

    assert compiled["config"]["range"]["category"] == category_palette()
