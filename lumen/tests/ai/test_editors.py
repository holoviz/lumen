"""Tests for lumen.ai.editors module."""
import json

import pandas as pd
import param
import pytest

import lumen.ai.editors as editors_module

from lumen.ai.editors import LumenEditor, SQLEditor
from lumen.base import Component


class MockComponent(Component):
    """Minimal component for testing editor exports."""

    data = param.Parameter()

    table = param.String(default="test")


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
