import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip(
        "lumen.ai could not be imported, skipping tests.",
        allow_module_level=True,
    )

from lumen.ai.editors import VegaLiteEditor
from lumen.ai.utils import normalize_vegalite_spec


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
