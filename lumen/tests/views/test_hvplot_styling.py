import pandas as pd
import pytest

from hvplot.ui import hvDataFrameExplorer

from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.views.hvplot import (
    HVPLOT_STYLE_PARAMS, hvPlotBaseView, hvPlotUIView, hvPlotView,
)

from .test_hvplot_datashade import record_hvplot_call

# Every option hvPlot >=0.11.3 documents well enough to declare, against a value
# the explorer accepts for it. Doubles as the expected set: anything missing
# means hvPlot moved its docs and the views quietly lost an option.
STYLE_VALUES = {
    "alpha": 0.5, "clabel": "c", "clim": (0, 1), "cmap": "viridis",
    "cnorm": "log", "color": "red", "colorbar": True, "fontscale": 1.2,
    "legend": "top_left", "logx": True, "logy": True, "rot": 45,
    "xlabel": "x", "xlim": (0, 1), "ylabel": "y", "ylim": (0, 1),
}


@pytest.fixture
def df():
    return pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})


@pytest.fixture
def pipeline(df):
    return Pipeline(source=InMemorySource(tables={"points": df}), table="points")


# ---- The params exist and carry usable metadata ----

def test_style_params_are_declared():
    """Nothing raises on an hvPlot that documents fewer options, so the set is
    asserted rather than left to shrink quietly."""
    assert set(HVPLOT_STYLE_PARAMS) == set(STYLE_VALUES)
    for name in HVPLOT_STYLE_PARAMS:
        assert name in hvPlotBaseView.param, name


def test_every_style_param_documents_itself():
    """A param's doc is what the agent's model turns into the LLM description."""
    for name in HVPLOT_STYLE_PARAMS:
        assert hvPlotBaseView.param[name].doc, name


def test_style_params_do_not_shadow_the_lumen_params():
    """The hand-written params carry Lumen semantics the generated ones must not
    overwrite -- x/y/by/groupby/z bind to the table schema, kind and aggregator
    to Lumen's own object lists."""
    for name in ("kind", "x", "y", "by", "groupby", "z", "geo", "datashade",
                 "dynspread", "aggregator", "color_key", "title"):
        assert name not in HVPLOT_STYLE_PARAMS, name


def test_style_params_start_unset():
    """Structured output answers with every field, so an option holding the
    explorer's default would be restated on every spec and forwarded as though
    the request had asked for it."""
    for name in HVPLOT_STYLE_PARAMS:
        assert hvPlotBaseView.param[name].default is None, name


def test_style_params_are_accepted_by_the_explorer(df):
    """A generated spec renders as hvplot_ui, so a param the explorer rejects
    would raise rather than style anything."""
    hvDataFrameExplorer(df, x="x", y="y", kind="line", **STYLE_VALUES)


# ---- Trap A: hvPlotUIView drops params no control claims ----

def test_ui_view_forwards_style_params(pipeline, df):
    view = hvPlotUIView(
        pipeline=pipeline, kind="line", x="x", y="y",
        logy=True, xlabel="Across", ylabel="Up", cmap="viridis",
    )

    _, kwargs = view._get_args(hvDataFrameExplorer, df)

    assert kwargs["logy"] is True
    assert kwargs["xlabel"] == "Across"
    assert kwargs["ylabel"] == "Up"
    assert kwargs["cmap"] == "viridis"


# ---- Trap B: hvPlotView strips params out of kwargs ----

def test_hvplot_view_forwards_style_params(pipeline, df):
    view = hvPlotView(
        pipeline=pipeline, kind="line", x="x", y="y",
        logy=True, xlabel="Across", cmap="viridis", rot=45,
    )

    recorded = record_hvplot_call(view, df)

    assert recorded["logy"] is True
    assert recorded["xlabel"] == "Across"
    assert recorded["cmap"] == "viridis"
    assert recorded["rot"] == 45


def test_hvplot_view_omits_unset_style_params(pipeline, df):
    """An unstyled plot must not start carrying every styling keyword."""
    view = hvPlotView(pipeline=pipeline, kind="line", x="x", y="y")

    recorded = record_hvplot_call(view, df)

    assert not set(recorded) & set(HVPLOT_STYLE_PARAMS)


def test_style_params_round_trip_through_the_spec(pipeline):
    view = hvPlotView(pipeline=pipeline, kind="line", x="x", y="y", logy=True, rot=45)

    spec = view.to_spec()

    assert spec["logy"] is True
    assert spec["rot"] == 45


# ---- The colormap choices ----

def test_the_colormap_enum_stays_short():
    """hvPlot offers 712 colormaps and every one would be spent as prompt."""
    assert len(hvPlotBaseView.param.cmap.objects) <= 20


def test_a_dict_colormap_is_still_accepted(pipeline, df):
    """Specs predating color_key passed the categorical mapping as cmap, and
    hvPlot still reads a dict cmap as one. The enum steers the LLM; it must not
    narrow what a spec may say."""
    color_key = {"Irish": "#e41a1c", "Italian": "#377eb8"}
    view = hvPlotView(pipeline=pipeline, kind="line", x="x", y="y", cmap=color_key)

    assert record_hvplot_call(view, df)["cmap"] == color_key


def test_the_view_title_is_not_drawn_twice(pipeline, df):
    """Every View renders its own title above the panel, and the explorer's
    Labels control would draw the same string inside the plot."""
    view = hvPlotUIView(pipeline=pipeline, kind="line", x="x", y="y", title="Cities")

    _, kwargs = view._get_args(hvDataFrameExplorer, df)

    assert "title" not in kwargs
