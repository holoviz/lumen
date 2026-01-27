"""Tests for lumen.ai.editors module."""
from lumen.ai.editors import LumenEditor


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
