"""Tests for lumen.ai.editors module."""
import json

from io import BytesIO

import pandas as pd
import param
import pytest

from PIL import Image

import lumen.ai.editors as editors_module

from lumen.ai.editors import LumenEditor, SQLEditor, VegaLiteEditor
from lumen.base import Component
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource


class MockComponent(Component):
    """Minimal component for testing editor exports."""

    data = param.Parameter()

    table = param.String(default="test")


class MockVegaComponent(Component):
    _mock_panel = param.Parameter()

    def get_panel(self):
        return self._mock_panel


def _make_png_bytes(mode="RGBA"):
    img = Image.new(mode, (10, 10), color=(255, 0, 0, 128) if mode == "RGBA" else (255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_MINIMAL_VEGALITE_SPEC = """\
$schema: https://vega.github.io/schema/vega-lite/v5.json
mark: bar
encoding:
  x:
    field: A
    type: quantitative
  y:
    field: B
    type: quantitative
data:
  values:
    - {A: 1, B: 2}
"""


@pytest.fixture
def mock_panel():
    class FakePanel:
        def __init__(self):
            self.calls = []

        def export(self, fmt, **kwargs):
            self.calls.append((fmt, kwargs))
            return _make_png_bytes()

    return FakePanel()


@pytest.fixture
def vegalite_editor(monkeypatch, mock_panel):
    monkeypatch.setattr(editors_module, 'ParamMethod', lambda *args, **kwargs: None)
    monkeypatch.setattr(VegaLiteEditor, '_update_component', lambda self, *a, **kw: None)
    component = MockVegaComponent(_mock_panel=mock_panel)
    return VegaLiteEditor(component=component, spec=_MINIMAL_VEGALITE_SPEC)


@pytest.fixture
def sql_editor(monkeypatch):
    """Return a SQLEditor with a minimal valid component and spec."""
    df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 25]})
    component = MockComponent(data=df)
    monkeypatch.setattr(editors_module, 'ParamMethod', lambda *args, **kwargs: None)
    return SQLEditor(component=component, spec="SELECT * FROM test")


def test_sqleditor_export_sql(sql_editor):
    result = sql_editor.export('sql')
    assert 'SELECT' in result.getvalue()


def test_sqleditor_export_csv(sql_editor):
    result = sql_editor.export('csv')
    content = result.getvalue()
    assert 'name' in content
    assert 'Alice' in content


def test_sqleditor_export_json(sql_editor):
    result = sql_editor.export('json')
    data = json.loads(result.getvalue())
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]['name'] == 'Alice'


def test_sqleditor_export_markdown(sql_editor):
    result = sql_editor.export('markdown')
    content = result.getvalue()
    assert '|' in content  # markdown tables use pipes
    assert 'name' in content
    assert 'Alice' in content



def test_vegalite_export_formats_includes_pillow_formats():
    assert {"webp", "tiff", "eps"} <= set(VegaLiteEditor.export_formats)


@pytest.mark.parametrize("fmt,validate", [
    ("webp", lambda r: Image.open(r).format == "WEBP"),
    ("tiff", lambda r: Image.open(r).format == "TIFF"),
    ("eps",  lambda r: r.read().startswith(b"%!PS")),
])
def test_vegalite_pillow_export(vegalite_editor, mock_panel, fmt, validate):
    result = vegalite_editor.export(fmt)
    assert isinstance(result, BytesIO)
    assert validate(result)
    assert mock_panel.calls[-1] == ("png", {"scale": 2})


@pytest.fixture
def sql_pipeline_editor(monkeypatch):
    """Return a SQLEditor wrapping a real Pipeline over an in-memory DuckDB source."""
    source = DuckDBSource(tables={
        'tiny': """
            SELECT * FROM (
              VALUES (1,'A',10.5),
                     (2,'B',20.0),
                     (3,'A', 7.5)
            ) AS t(id, category, value)
        """
    })
    pipeline = Pipeline(source=source, table='tiny')
    monkeypatch.setattr(editors_module, 'ParamMethod', lambda *args, **kwargs: None)
    return SQLEditor(component=pipeline, spec="SELECT * FROM tiny")


def test_filter_menu_items_skip_len_and_pick_icons(sql_pipeline_editor):
    items = sql_pipeline_editor._filter_items()
    by_label = {it["label"]: it["icon"] for it in items}
    assert "__len__" not in by_label
    assert by_label["id"] == "numbers"            # integer
    assert by_label["category"] == "format_list_bulleted"  # enum string
    assert by_label["value"] == "calculate"       # number


def test_add_filter_creates_labeled_widget(sql_pipeline_editor):
    editor = sql_pipeline_editor
    editor._add_filter({"label": "value", "toggled": True})
    filt = editor._filters["value"]
    # Label falls back to the field name when no explicit label is given.
    assert filt.widget.label == "value"
    assert filt in editor.component.filters
    assert filt.widget in list(editor._filter_area)


def test_filter_subsets_and_restores_data(sql_pipeline_editor):
    editor = sql_pipeline_editor
    pipeline = editor.component
    assert len(pipeline.data) == 3

    editor._add_filter({"label": "value", "toggled": True})
    editor._filters["value"].widget.value = (15.0, 25.0)
    assert list(pipeline.data["value"]) == [20.0]

    editor._add_filter({"label": "value", "toggled": False})
    assert all(f.field != "value" for f in pipeline.filters)
    assert editor._filters["value"].widget not in list(editor._filter_area)
    assert len(pipeline.data) == 3


def test_render_controls_inserts_add_filter_menu(sql_pipeline_editor):
    controls = sql_pipeline_editor.render_controls(task=None, interface=None)
    labels = [getattr(c, "label", None) for c in controls]
    assert "Add Filter" in labels


@pytest.fixture
def xarray_pipeline_editor(monkeypatch):
    """SQLEditor over an xarray source so coordinate dimensions are exercised."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_sql")
    import numpy as np

    from lumen.sources.xarray_sql import XArraySQLSource

    times = pd.date_range("2020-01-01", periods=6, freq="D")
    ds = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], np.random.default_rng(0).random((6, 3, 2)))},
        coords={"time": times,
                "lat": ("lat", np.array([10.0, 20.0, 30.0])),
                "lon": ("lon", np.array([100.0, 110.0]))},
    )
    pipeline = Pipeline(source=XArraySQLSource(_dataset=ds), table="temperature")
    monkeypatch.setattr(editors_module, 'ParamMethod', lambda *args, **kwargs: None)
    return SQLEditor(component=pipeline, spec="SELECT * FROM temperature")


def test_coordinate_dimensions_listed_first(xarray_pipeline_editor):
    items = xarray_pipeline_editor._filter_items()
    labels = [it["label"] for it in items]
    # The data variable comes after all coordinate dimensions.
    assert set(labels[:3]) == {"time", "lat", "lon"}
    assert labels[-1] == "temperature"
    icons = {it["label"]: it["icon"] for it in items}
    assert icons["time"] == "schedule"        # datetime axis
    assert icons["lat"] == "straighten"       # numeric axis
    assert icons["temperature"] == "calculate"  # data variable


def test_datetime_axis_widget_supports_label(xarray_pipeline_editor):
    # Regression: the datetime range widget must accept a label (PMUI, not
    # plain panel) or constructing the time filter raises TypeError.
    xarray_pipeline_editor._add_filter({"label": "time", "toggled": True})
    widget = xarray_pipeline_editor._filters["time"].widget
    assert "label" in widget.param
    assert widget.label == "time"


def test_coordinate_range_filter_subsets_like_sel(xarray_pipeline_editor):
    editor = xarray_pipeline_editor
    pipeline = editor.component
    editor._add_filter({"label": "lat", "toggled": True})
    editor._filters["lat"].widget.value = (20.0, 30.0)
    ds = pipeline.source.dataset
    expected = ds.sel(lat=slice(20.0, 30.0)).to_dataframe().reset_index().shape[0]
    assert len(pipeline.data) == expected
    assert set(pipeline.data["lat"].unique()) == {20.0, 30.0}


def test_class_name_to_filename():
    """Test CamelCase to snake_case conversion for base class."""
    assert LumenEditor._class_name_to_download_filename("yaml") == "lumen_editor.yaml"
    assert LumenEditor._class_name_to_download_filename("json") == "lumen_editor.json"


def test_class_name_with_acronyms():
    """Test handling of acronyms in class names."""
    class SQLEditor(LumenEditor):
        pass

    assert SQLEditor._class_name_to_download_filename("yaml") == "sql_editor.yaml"


def test_class_name_with_numbers():
    """Test handling of numbers in class names."""
    class Editor2D(LumenEditor):
        pass

    assert Editor2D._class_name_to_download_filename("png") == "editor2_d.png"
