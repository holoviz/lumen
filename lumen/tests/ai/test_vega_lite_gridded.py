import datetime

from lumen.ai.config import PROMPTS_DIR
from lumen.ai.utils import render_template


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
        doc="VegaLiteView description",
        gridded=gridded,
        current_datetime=datetime.datetime(2026, 1, 1),
        actor_name="VegaLiteAgent",
    )


def test_vegalite_prompt_includes_gridded_block():
    gridded = {
        "source_type": "xarray",
        "dims": ["lat", "lon"],
        "coords": {"lat": [3], "lon": [4]},
        "data_vars": ["air"],
    }
    rendered = render_template(
        PROMPTS_DIR / "VegaLiteAgent" / "main.jinja2",
        **_base_context(gridded=gridded),
    )
    assert "mark: rect" in rendered
    assert "viridis" in rendered
    assert "lat" in rendered
    assert "air" in rendered


def test_vegalite_prompt_no_gridded_block_for_tabular():
    rendered = render_template(
        PROMPTS_DIR / "VegaLiteAgent" / "main.jinja2",
        **_base_context(gridded=None),
    )
    assert "Gridded xarray data" not in rendered
    assert "Gridded heatmap" not in rendered
