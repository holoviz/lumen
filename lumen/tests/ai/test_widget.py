import importlib.util
import sys

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

pytest.importorskip("anywidget")

from lumen.ai.answer import Answer, _StaticHTML
from lumen.ai.selection import CLICK
from lumen.ai.widget import TABLE_ROWS, AnswerWidget
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource
from lumen.views import VegaLiteView, hvPlotView

SPEC = {"mark": "bar", "encoding": {"x": {"field": "region", "type": "nominal"}, "y": {"field": "amount", "type": "quantitative"}}}


@pytest.fixture
def pipeline():
    sales = pd.DataFrame({"region": ["N", "S", "E"], "amount": [10.0, 20.0, 5.0]})
    return Pipeline(source=DuckDBSource.from_df({"sales": sales}), table="sales")


def _answer(pipeline, view=None, **kwargs):
    return Answer(question="Sales?", value="South sold **most**.", status="success", pipeline=pipeline, view=view, **kwargs)


def test_a_vega_lite_answer_is_a_clickable_chart_whose_click_selects_rows(pipeline):
    answer = _answer(pipeline, VegaLiteView(pipeline=pipeline, spec=SPEC), spec=SPEC)
    widget = answer.widget()
    assert isinstance(widget, AnswerWidget) and widget.question == "Sales?"
    assert widget.clickable and [p["name"] for p in widget.vega["params"]] == [CLICK]
    assert widget.vega["data"]["values"][1] == {"region": "S", "amount": 20.0}
    assert "<strong>most</strong>" in widget.card and widget.bokeh is None
    # As vega-embed reports a click on the S bar, and then on nothing
    widget.clicked = {"vlPoint": {"or": [{"_vgsid_": 2}]}, "region": ["S"]}
    assert answer.selection.to_dict("records") == [{"region": "S", "amount": 20.0}]
    widget.clicked = {}
    assert answer.selection is None


def test_an_hvplot_answer_is_standalone_bokeh(pipeline):
    widget = _answer(pipeline, hvPlotView(pipeline=pipeline, kind="bar", x="region", y="amount")).widget()
    assert widget.vega is None and {"doc", "root_id"} <= set(widget.bokeh)
    assert not widget.clickable


def test_an_answer_without_a_chart_shows_its_first_rows(pipeline):
    widget = _answer(pipeline).widget()
    assert "<table" in widget.card and ">S<" in widget.card and widget.vega is None and widget.bokeh is None
    many = Pipeline(source=DuckDBSource.from_df({"t": pd.DataFrame({"i": range(50)})}), table="t")
    assert f"The first {TABLE_ROWS} of 50 rows" in _answer(many).widget().card


def test_marimo_shows_the_widget_or_the_static_card_without_anywidget(pipeline, monkeypatch):
    answer = _answer(pipeline, VegaLiteView(pipeline=pipeline, spec=SPEC), spec=SPEC)
    assert isinstance(answer._display_(), AnswerWidget)
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None if name == "anywidget" else find_spec(name, *args))
    static = answer._display_()
    assert isinstance(static, _StaticHTML) and "<strong>most</strong>" in static._repr_html_()


def test_the_widget_says_to_install_anywidget(pipeline, monkeypatch):
    monkeypatch.setitem(sys.modules, "lumen.ai.widget", None)
    with pytest.raises(ImportError, match="pip install anywidget"):
        _answer(pipeline).widget()
