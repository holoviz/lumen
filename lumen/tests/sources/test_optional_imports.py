"""Tests for optional dependency import guards in source modules.

Verifies that source classes:
1. Can be imported without their optional dependency installed
2. Raise ImportError with a helpful message on instantiation
3. The error message includes the correct pip install command
"""
from __future__ import annotations

import importlib

from pathlib import Path

import pytest

# Each entry: (module_path, class_name, guard_package, pip_extra)
OPTIONAL_SOURCES = [
    ("lumen.sources.sqlalchemy", "SQLAlchemySource", "sqlalchemy", "sql"),
    ("lumen.sources.bigquery", "BigQuerySource", "google.auth", "bigquery"),
    ("lumen.sources.snowflake", "SnowflakeSource", "snowflake", "snowflake"),
    ("lumen.sources.ae5", "AE5Source", "ae5_tools", "ae5"),
]


@pytest.mark.parametrize(
    "module_path,class_name,guard_package,pip_extra",
    OPTIONAL_SOURCES,
    ids=["sqlalchemy", "bigquery", "snowflake", "ae5"],
)
def test_import_succeeds_without_optional_dep(module_path, class_name, guard_package, pip_extra):
    """Importing the module should not crash even if the optional dep is missing."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, class_name)


@pytest.mark.parametrize(
    "module_path,class_name,guard_package,pip_extra",
    OPTIONAL_SOURCES,
    ids=["sqlalchemy", "bigquery", "snowflake", "ae5"],
)
def test_error_message_contains_pip_install(module_path, class_name, guard_package, pip_extra):
    """Error messages should include the correct pip install command."""
    mod = importlib.import_module(module_path)
    source_text = Path(mod.__file__).read_text()
    expected = f"pip install lumen[{pip_extra}]"
    assert expected in source_text, (
        f"{module_path} should contain '{expected}' in its error message"
    )


@pytest.mark.parametrize(
    "module_path,class_name,guard_package,pip_extra",
    OPTIONAL_SOURCES,
    ids=["sqlalchemy", "bigquery", "snowflake", "ae5"],
)
def test_import_guard_uses_try_except(module_path, class_name, guard_package, pip_extra):
    """Each module should have a try/except guard around the optional import."""
    mod = importlib.import_module(module_path)
    source_text = Path(mod.__file__).read_text()
    assert "except ImportError:" in source_text, (
        f"{module_path} should have 'except ImportError:' guard"
    )
