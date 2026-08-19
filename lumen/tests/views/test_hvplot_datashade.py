import pandas as pd
import pytest

from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.views import base as views_base
from lumen.views.base import hvPlotBaseView, hvPlotUIView, hvPlotView

# ---- Fixtures ----

@pytest.fixture
def categorical_df():
    """Points carrying a categorical column, the shape datashader blends by."""
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "ancestry": ["Irish", "Italian", "German", "Irish"],
        }
    )


@pytest.fixture
def categorical_pipeline(categorical_df):
    source = InMemorySource(tables={"points": categorical_df})
    return Pipeline(source=source, table="points")


COLOR_KEY = {"Irish": "#e41a1c", "Italian": "#377eb8", "German": "#4daf4a"}


class RecordingFrame(pd.DataFrame):
    """DataFrame that records the kwargs its .hvplot call receives."""

    _metadata = ["recorded"]

    @property
    def _constructor(self):
        return RecordingFrame

    def hvplot(self, **kwargs):
        self.recorded.update(kwargs)
        return object()


def record_hvplot_call(view, df):
    """Return the kwargs hvPlot would be called with for this view."""
    frame = RecordingFrame(df)
    frame.recorded = {}
    view.get_plot(frame)
    return frame.recorded


# ---- Params exist ----

def test_datashade_params_declared():
    """The AI agent's schema is derived from these params, so they must exist."""
    for name in ("datashade", "dynspread", "color_key"):
        assert name in hvPlotBaseView.param


def test_datashade_params_are_not_kwargs(categorical_pipeline):
    """Promoting a name to a param removes it from kwargs, which is why the
    views have to forward it explicitly."""
    view = hvPlotView(
        pipeline=categorical_pipeline,
        kind="points",
        x="x",
        y="y",
        by=["ancestry"],
        datashade=True,
        color_key=COLOR_KEY,
    )

    assert view.kwargs == {}
    assert view.datashade is True
    assert view.color_key == COLOR_KEY


# ---- hvPlotView forwards to hvPlot ----

def test_hvplot_view_forwards_datashade(categorical_pipeline, categorical_df):
    view = hvPlotView(
        pipeline=categorical_pipeline,
        kind="points",
        x="x",
        y="y",
        by=["ancestry"],
        datashade=True,
        dynspread=True,
        color_key=COLOR_KEY,
    )

    recorded = record_hvplot_call(view, categorical_df)

    assert recorded["datashade"] is True
    assert recorded["dynspread"] is True
    assert recorded["color_key"] == COLOR_KEY
    assert recorded["by"] == ["ancestry"]


def test_hvplot_view_omits_unset_datashade(categorical_pipeline, categorical_df):
    """An ordinary plot must not start carrying datashader keywords."""
    view = hvPlotView(pipeline=categorical_pipeline, kind="scatter", x="x", y="y")

    recorded = record_hvplot_call(view, categorical_df)

    assert "datashade" not in recorded
    assert "dynspread" not in recorded
    assert "color_key" not in recorded


def test_hvplot_view_keeps_dict_cmap_kwarg(categorical_pipeline, categorical_df):
    """Specs predating color_key passed the mapping as cmap; hvPlot still maps a
    dict cmap onto color_key, so those must keep working."""
    view = hvPlotView(
        pipeline=categorical_pipeline,
        kind="points",
        x="x",
        y="y",
        by=["ancestry"],
        datashade=True,
        cmap=COLOR_KEY,
    )

    recorded = record_hvplot_call(view, categorical_df)

    assert recorded["cmap"] == COLOR_KEY
    assert recorded["datashade"] is True


# ---- hvPlotUIView forwards to the explorer ----

def test_hvplot_ui_view_forwards_datashade(categorical_pipeline):
    """The explorer keeps these on nested controls rather than on
    hvPlotExplorer itself, so _get_args has to look there."""
    view = hvPlotUIView(
        pipeline=categorical_pipeline,
        kind="points",
        x="x",
        y="y",
        by=["ancestry"],
        datashade=True,
        dynspread=True,
        color_key=COLOR_KEY,
    )

    _args, kwargs = view._get_args()

    assert kwargs["datashade"] is True
    assert kwargs["dynspread"] is True
    assert kwargs["color_key"] == COLOR_KEY


def test_hvplot_ui_view_builds_explorer(categorical_pipeline):
    """hvPlotExplorer.__init__ raises on any keyword no control claims, so
    constructing it proves the forwarded names are routable."""
    view = hvPlotUIView(
        pipeline=categorical_pipeline,
        kind="points",
        x="x",
        y="y",
        by=["ancestry"],
        datashade=True,
        color_key=COLOR_KEY,
    )

    explorer = view.get_panel()

    assert explorer.operations.datashade is True
    assert explorer.colormapping.color_key == COLOR_KEY


# ---- Render-size cap ----

def test_render_cap_exempts_datashade_param(categorical_pipeline, monkeypatch):
    """The cap reads the param now; reading only kwargs would reject every
    datashaded plot, which is exactly the large-frame case datashade is for."""
    monkeypatch.setattr(views_base, "MAX_RENDER_ROWS", 2)
    view = hvPlotView(
        pipeline=categorical_pipeline, kind="points", x="x", y="y", datashade=True
    )
    over_cap = pd.DataFrame({"x": range(10), "y": range(10)})

    view._check_render_size(over_cap)  # datashade -> exempt -> must not raise


def test_render_cap_still_exempts_rasterize_kwarg(categorical_pipeline, monkeypatch):
    """rasterize stays a plain kwarg and must keep its exemption."""
    monkeypatch.setattr(views_base, "MAX_RENDER_ROWS", 2)
    view = hvPlotView(
        pipeline=categorical_pipeline, kind="points", x="x", y="y", rasterize=True
    )
    over_cap = pd.DataFrame({"x": range(10), "y": range(10)})

    view._check_render_size(over_cap)


def test_render_cap_still_raises_without_aggregation(categorical_pipeline, monkeypatch):
    monkeypatch.setattr(views_base, "MAX_RENDER_ROWS", 2)
    view = hvPlotView(pipeline=categorical_pipeline, kind="points", x="x", y="y")
    over_cap = pd.DataFrame({"x": range(10), "y": range(10)})

    with pytest.raises(ValueError, match="10 rows"):
        view._check_render_size(over_cap)


# ---- A single column name where a list is expected ----

@pytest.mark.parametrize("key", ["by", "groupby"])
def test_spec_accepts_a_bare_column_name(categorical_pipeline, key):
    """__init__ coerces a string, but validate() runs on the raw spec first, so
    without a matching hook a spec saying `by: family` was rejected before the
    coercion written for it ever ran."""
    spec = {"type": "hvplot", "kind": "points", "x": "x", "y": "y", key: "ancestry"}

    validated = hvPlotView.validate(dict(spec))

    assert validated[key] == ["ancestry"]


@pytest.mark.parametrize("key", ["by", "groupby"])
def test_spec_leaves_a_list_alone(categorical_pipeline, key):
    spec = {"type": "hvplot", "kind": "points", "x": "x", "y": "y", key: ["ancestry"]}

    assert hvPlotView.validate(dict(spec))[key] == ["ancestry"]


@pytest.mark.parametrize("key", ["by", "groupby"])
def test_constructor_still_coerces_a_bare_column_name(categorical_pipeline, key):
    view = hvPlotView(
        pipeline=categorical_pipeline, kind="points", x="x", y="y", **{key: "ancestry"}
    )

    assert getattr(view, key) == ["ancestry"]


# ---- A colour key naming only some categories ----

def test_partial_color_key_is_completed(categorical_pipeline, categorical_df):
    """Datashader needs a color per category, but naming the few that matter is
    the natural way to ask, so the rest are filled rather than raising."""
    view = hvPlotView(
        pipeline=categorical_pipeline,
        kind="points",
        x="x",
        y="y",
        by=["ancestry"],
        datashade=True,
        color_key={"Irish": "#e41a1c"},
    )

    resolved = view._complete_color_key(categorical_df)

    assert set(resolved) == {"Irish", "Italian", "German"}
    assert resolved["Irish"] == "#e41a1c"
    assert len(set(resolved.values())) == 3


def test_completed_key_never_reuses_a_chosen_color(categorical_pipeline, categorical_df):
    view = hvPlotView(
        pipeline=categorical_pipeline,
        kind="points",
        x="x",
        y="y",
        by=["ancestry"],
        datashade=True,
        color_key={"Irish": "#1f77b4"},
    )

    resolved = view._complete_color_key(categorical_df)

    assert list(resolved.values()).count("#1f77b4") == 1


def test_complete_key_is_left_alone(categorical_pipeline, categorical_df):
    view = hvPlotView(
        pipeline=categorical_pipeline,
        kind="points",
        x="x",
        y="y",
        by=["ancestry"],
        datashade=True,
        color_key=COLOR_KEY,
    )

    assert view._complete_color_key(categorical_df) == COLOR_KEY


def test_completion_covers_a_categorical_dtype(categorical_pipeline, categorical_df):
    """count_cat keys off the dtype's categories, which can list values that
    never appear in the frame."""
    framed = categorical_df.copy()
    framed["ancestry"] = framed["ancestry"].astype("category")
    view = hvPlotView(
        pipeline=categorical_pipeline,
        kind="points",
        x="x",
        y="y",
        by=["ancestry"],
        datashade=True,
        color_key={"Irish": "#e41a1c"},
    )

    resolved = view._complete_color_key(framed)

    assert set(resolved) == set(framed["ancestry"].cat.categories)
