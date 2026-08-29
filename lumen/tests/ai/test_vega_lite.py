import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip(
        "lumen.ai could not be imported, skipping tests.",
        allow_module_level=True,
    )

import vl_convert

from lumen.ai.agents.vega_lite import VegaLiteAgent
from lumen.ai.config import PROMPTS_DIR
from lumen.ai.editors import VegaLiteEditor
from lumen.ai.utils import category_palette, normalize_vegalite_spec
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


class TestGeoshapeGeometryValidation:
    """A geoshape over a table with no geometry compiles cleanly and draws nothing,
    so validation has to reject it or the only symptom is an empty canvas."""

    BOUNDARIES = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json"

    def test_rejects_geoshape_bound_to_the_table(self):
        spec = {
            "data": {"name": "avg_pm25_by_state"},
            "projection": {"type": "albersUsa"},
            "layer": [{
                "mark": {"type": "geoshape", "stroke": "white"},
                "encoding": {"color": {"field": "avg_pm25", "type": "quantitative"}},
            }],
            "transform": [{
                "lookup": "properties.name",
                "from": {
                    "data": {"url": self.BOUNDARIES, "format": {"type": "topojson", "feature": "states"}},
                    "key": "properties.name",
                    "fields": ["state"],
                },
            }],
        }
        with pytest.raises(RuntimeError, match="carries no geometry"):
            VegaLiteEditor.validate_spec(spec)

    def test_accepts_boundaries_as_primary_data(self):
        spec = {
            "data": {"url": self.BOUNDARIES, "format": {"type": "topojson", "feature": "states"}},
            "transform": [{
                "lookup": "properties.name",
                "from": {"data": {"name": "t"}, "key": "state", "fields": ["avg_pm25"]},
            }],
            "mark": "geoshape",
            "projection": {"type": "albersUsa"},
            "encoding": {"color": {"field": "avg_pm25", "type": "quantitative"}},
        }
        VegaLiteEditor.validate_spec(spec)

    def test_accepts_omitted_data_for_own_geometry(self):
        """The table's own geometry is injected at render time, so the spec has no data."""
        spec = {
            "mark": "geoshape",
            "projection": {"type": "naturalEarth1"},
            "encoding": {"color": {"field": "properties.avg_pm25", "type": "quantitative"}},
        }
        VegaLiteEditor.validate_spec(spec)

    def test_accepts_the_two_layer_base_plus_join_pattern(self):
        """The base layer keeps regions the table does not cover on the map."""
        boundaries = {"url": self.BOUNDARIES, "format": {"type": "topojson", "feature": "states"}}
        spec = {
            "projection": {"type": "albersUsa"},
            "layer": [
                {"data": boundaries, "mark": {"type": "geoshape", "fill": "#eeeeee"}},
                {
                    "data": boundaries,
                    "transform": [{
                        "lookup": "properties.name",
                        "from": {"data": {"name": "t"}, "key": "state", "fields": ["avg_pm25"]},
                    }],
                    "mark": "geoshape",
                    "encoding": {"color": {"field": "avg_pm25", "type": "quantitative"}},
                },
            ],
        }
        VegaLiteEditor.validate_spec(spec)

    def test_rejects_a_layer_bound_to_the_table(self):
        spec = {
            "projection": {"type": "albersUsa"},
            "layer": [{
                "data": {"name": "t"},
                "mark": "geoshape",
                "encoding": {"color": {"field": "avg_pm25", "type": "quantitative"}},
            }],
        }
        with pytest.raises(RuntimeError, match="carries no geometry"):
            VegaLiteEditor.validate_spec(spec)

    def test_rejects_a_layer_inheriting_table_bound_data(self):
        """A layer with no data of its own uses its parent's, so the check follows it."""
        spec = {
            "data": {"name": "t"},
            "layer": [{
                "mark": "geoshape",
                "encoding": {"color": {"field": "avg_pm25", "type": "quantitative"}},
            }],
        }
        with pytest.raises(RuntimeError, match="carries no geometry"):
            VegaLiteEditor.validate_spec(spec)

    def test_leaves_non_geographic_charts_alone(self):
        spec = {
            "data": {"name": "t"},
            "mark": "bar",
            "encoding": {
                "x": {"field": "state", "type": "nominal"},
                "y": {"field": "avg_pm25", "type": "quantitative"},
            },
        }
        VegaLiteEditor.validate_spec(spec)


class TestMultiSeriesLineSplit:
    """A line over several series needs a split channel or every series lands on
    one polyline, which zigzags back across a perfectly correct time axis.

    Vega-Lite splits the path by wrapping the mark in a group whose
    ``from.facet.groupby`` names the split field, so the compiled output says
    plainly whether the chart draws one line or several.
    """

    ROWS = [
        {"time": time, "band": band, "temp": temp}
        for time in ("2020-01-01", "2020-04-01", "2020-07-01")
        for band, temp in (("tropics", 28), ("midlat", 12), ("arctic", -15))
    ]

    @classmethod
    def _line_spec(cls, **encoding):
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": cls.ROWS},
            "mark": "line",
            "encoding": {
                "x": {"field": "time", "type": "temporal"},
                "y": {"field": "temp", "type": "quantitative"},
                **encoding,
            },
        }

    @classmethod
    def _split_field(cls, **encoding):
        """The field the compiled chart splits its path on, None if it draws one."""
        mark = vl_convert.vegalite_to_vega(cls._line_spec(**encoding))["marks"][0]
        return ((mark.get("from") or {}).get("facet") or {}).get("groupby")

    def test_a_line_with_no_split_channel_draws_a_single_path(self):
        assert self._split_field() is None

    def test_color_and_detail_each_split_the_path(self):
        band = {"field": "band", "type": "nominal"}
        assert self._split_field(color=band) == ["band"]
        assert self._split_field(detail=band) == ["band"]

    def test_shape_and_a_constant_color_leave_the_path_joined(self):
        """Neither reaches the path, so a chart that names a series through them
        still draws one polyline. A check that trusted the presence of a channel
        rather than the compiled output would call both of these split.
        """
        assert self._split_field(shape={"field": "band", "type": "nominal"}) is None
        assert self._split_field(color={"value": "#d62728"}) is None

    def test_sorting_the_x_encoding_leaves_the_path_untouched(self):
        """A path mark already sorts by x, so an explicit sort on a continuous
        axis cannot be what joins or separates the series.
        """
        plain = vl_convert.vegalite_to_vega(self._line_spec())
        sorted_x = vl_convert.vegalite_to_vega(
            self._line_spec(x={"field": "time", "type": "temporal", "sort": "ascending"})
        )

        assert [mark.get("sort") for mark in plain["marks"]] == [{"field": "x"}]
        assert sorted_x == plain

    def test_the_altair_template_states_the_split_rule_and_shows_it(self):
        """main.jinja2 carries this rule; main_altair.jinja2 is a separate
        template that inherits none of it, so it needs its own copy and its own
        worked example.
        """
        # ponytail: reads the template as text, which holds while the rule and
        # the example both sit outside every conditional block.
        altair = (PROMPTS_DIR / "VegaLiteAgent" / "main_altair.jinja2").read_text()
        time_series = altair.split("Time series:")[1].split("```")[1]

        assert "split on it with `color`" in altair
        assert "mark_line" in time_series and "color=" in time_series
