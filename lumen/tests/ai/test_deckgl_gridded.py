import datetime

import pandas as pd
import pytest

from lumen.ai.config import PROMPTS_DIR
from lumen.ai.utils import render_template
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.views.base import DECKGL_MAX_ROWS, DeckGLView


def _base_context(gridded=None):
    class _FakeSource:
        dialect = None

    class _FakePipeline:
        table = "air"
        source = _FakeSource()

    class _FakeMemory(dict):
        pass

    memory = _FakeMemory({
        "pipeline": _FakePipeline(),
        "data": "lat,lon,air\n0,10,1.0",
        "table": "air",
    })
    return dict(
        memory=memory,
        doc="DeckGLView description",
        gridded=gridded,
        current_datetime=datetime.datetime(2026, 1, 1),
        actor_name="DeckGLAgent",
    )


_GRIDDED = {
    "source_type": "xarray",
    "dims": ["lat", "lon"],
    "coords": {"lat": [3], "lon": [4]},
    "data_vars": ["air"],
    "regular": True,
}


def test_deckgl_prompt_includes_gridded_block():
    rendered = render_template(
        PROMPTS_DIR / "DeckGLAgent" / "main.jinja2",
        **_base_context(gridded=_GRIDDED),
    )
    assert "ColumnLayer" in rendered
    assert "air" in rendered
    assert "lat" in rendered
    assert "lon" in rendered


def test_deckgl_prompt_no_gridded_block_for_tabular():
    rendered = render_template(
        PROMPTS_DIR / "DeckGLAgent" / "main.jinja2",
        **_base_context(gridded=None),
    )
    assert "Gridded xarray data" not in rendered


def test_deckgl_pydeck_prompt_includes_gridded_block():
    rendered = render_template(
        PROMPTS_DIR / "DeckGLAgent" / "main_pydeck.jinja2",
        **_base_context(gridded=_GRIDDED),
    )
    assert "ColumnLayer" in rendered
    assert "air" in rendered


def _deckgl_view(df):
    source = InMemorySource(tables={"t": df})
    pipeline = Pipeline(source=source, table="t")
    return DeckGLView(
        pipeline=pipeline,
        spec={
            "initialViewState": {"latitude": 0, "longitude": 0, "zoom": 2},
            "layers": [{"@@type": "ScatterplotLayer", "getPosition": "@@=[lon, lat]"}],
            "views": [{"@@type": "MapView", "controller": True}],
        },
    )


def test_deckglview_row_limit_raises():
    n = DECKGL_MAX_ROWS + 1
    df = pd.DataFrame({"lon": range(n), "lat": range(n), "value": range(n)})
    view = _deckgl_view(df)
    with pytest.raises(ValueError, match="250,000"):
        view._get_params()


def test_deckglview_row_limit_passes_under_threshold():
    df = pd.DataFrame({"lon": [0.0, 1.0], "lat": [0.0, 1.0], "value": [1.0, 2.0]})
    view = _deckgl_view(df)
    params = view._get_params()
    assert "object" in params
    assert params["object"]["layers"][0]["data"] == [
        {"lon": 0.0, "lat": 0.0, "value": 1.0},
        {"lon": 1.0, "lat": 1.0, "value": 2.0},
    ]
