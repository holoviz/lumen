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
