"""Tests for lumen.ai.editors module."""

import param

import lumen.ai.editors as editors_module

from lumen.ai.editors import LumenEditor
from lumen.base import Component


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


def test_sql_editor_explorer_defaults(monkeypatch):
    """SQLEditor should default Graphic Walker to the visualization tab."""
    calls = {}

    def fake_graphic_walker(data, **kwargs):
        calls["data"] = data
        calls["kwargs"] = kwargs
        return "walker"

    class DummyComponent(Component):
        data = param.Parameter(default="table_data")

        @classmethod
        def from_spec(cls, spec):
            return cls()

        def to_spec(self, context=None):
            return {"type": "dummy"}

        def __panel__(self):
            return "dummy-panel"

    editor = editors_module.SQLEditor(component=DummyComponent())
    monkeypatch.setattr(editors_module, "GraphicWalker", fake_graphic_walker)

    result = editor.render_explorer()

    assert result == "walker"
    assert calls["data"] is editor.component.param.data
    assert calls["kwargs"]["tab"] == "vis"
    assert calls["kwargs"]["hide_profiling"] is True
    assert calls["kwargs"]["config"]["i18nResources"]["en-US"]["App"]["segments"] == {
        "data": "Data View",
        "vis": "Visualization",
    }
