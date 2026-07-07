"""Tests for optional dependency import guards in source modules.

Verifies that source classes raise ImportError at module level with a
helpful message when their optional dependency is not installed.
"""
from __future__ import annotations

import builtins
import importlib
import sys

from pathlib import Path
from unittest.mock import patch

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
def test_error_message_contains_pip_install(module_path, class_name, guard_package, pip_extra):
    """Error messages should include the correct pip install command."""
    module_file = module_path.replace(".", "/") + ".py"
    source_text = Path(module_file).read_text()
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
    module_file = module_path.replace(".", "/") + ".py"
    source_text = Path(module_file).read_text()
    assert "except ImportError" in source_text, (
        f"{module_path} should have 'except ImportError' guard"
    )
    assert "raise ImportError(" in source_text, (
        f"{module_path} should re-raise ImportError with a helpful message"
    )


@pytest.mark.parametrize(
    "module_path,class_name,guard_package,pip_extra",
    OPTIONAL_SOURCES,
    ids=["sqlalchemy", "bigquery", "snowflake", "ae5"],
)
def test_import_raises_when_dependency_missing(module_path, class_name, guard_package, pip_extra):
    """Importing the module should raise ImportError with a helpful message
    when the optional dependency is not installed."""
    real_import = builtins.__import__
    blocked_root = guard_package.split(".")[0]

    def mock_import(name, *args, **kwargs):
        if name == blocked_root or name.startswith(blocked_root + "."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    # Remove the module and all submodules from sys.modules so it gets re-imported
    to_remove = [key for key in sys.modules if key == module_path or key.startswith(module_path + ".")]
    # Also remove the guard package itself so the mock takes effect
    to_remove += [key for key in sys.modules if key == blocked_root or key.startswith(blocked_root + ".")]
    saved = {key: sys.modules.pop(key) for key in to_remove}

    try:
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match=f"pip install lumen\\[{pip_extra}\\]"):
                importlib.import_module(module_path)
    finally:
        # Restore original modules so other tests are not affected
        sys.modules.update(saved)


def test_try_import_xarray_returns_module_when_installed():
    """try_import_xarray returns the xarray module when xarray and xarray-sql are installed."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_sql")
    from lumen.util import try_import_xarray
    assert try_import_xarray() is xr


def test_try_import_xarray_none_when_missing():
    """try_import_xarray returns None (not raises) when xarray-sql is absent."""
    from lumen.util import try_import_xarray

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "xarray_sql" or name.startswith("xarray_sql."):
            raise ImportError("No module named 'xarray_sql'")
        return real_import(name, *args, **kwargs)

    saved = {
        key: sys.modules.pop(key)
        for key in [k for k in sys.modules if k == "xarray_sql" or k.startswith("xarray_sql.")]
    }
    try:
        with patch("builtins.__import__", side_effect=mock_import):
            assert try_import_xarray() is None
    finally:
        sys.modules.update(saved)
