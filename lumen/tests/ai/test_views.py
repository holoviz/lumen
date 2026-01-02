"""Tests for lumen.ai.views module."""
from lumen.ai.views import LumenOutput


def test_class_name_to_filename():
    """Test CamelCase to snake_case conversion for base class."""
    assert LumenOutput._class_name_to_download_filename("yaml") == "lumen_output.yaml"
    assert LumenOutput._class_name_to_download_filename("json") == "lumen_output.json"


def test_class_name_with_acronyms():
    """Test handling of acronyms in class names."""
    class SQLOutput(LumenOutput):
        pass
    
    assert SQLOutput._class_name_to_download_filename("yaml") == "sql_output.yaml"


def test_class_name_with_numbers():
    """Test handling of numbers in class names."""
    class Output2D(LumenOutput):
        pass
    
    assert Output2D._class_name_to_download_filename("png") == "output2_d.png"