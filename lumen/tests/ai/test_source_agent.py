# Postponed annotations make every annotation in this module a string,
# which is what user modules with this import hand to SourceAgent.
from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest

try:
    from lumen.ai.agents.source import SourceAgent, _sanitize_tool_callable
    from lumen.ai.controls import CodeSourceControls
    from lumen.ai.translate import function_to_model
    from lumen.util import normalize_table_name
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)


def load_data(kind: Literal["a", "b"] = "a") -> pd.DataFrame:
    """Load some data."""
    return pd.DataFrame({"kind": [kind]})


def load_secret_data(kind: Literal["a", "b"] = "a", token: str = "") -> pd.DataFrame:
    """Load some data with a token."""
    return pd.DataFrame({"kind": [kind]})


def test_build_tools_keeps_action_with_postponed_annotations():
    controls = CodeSourceControls(functions=load_data)

    tools = SourceAgent._build_tools({"source_controls": [controls]}, result_store=[])

    assert [tool.name for tool in tools] == ["load_data"]


def test_sanitize_tool_callable_resolves_postponed_annotations():
    result_store = []
    wrapped = _sanitize_tool_callable(load_data, "Load Data", [], result_store)

    schema = function_to_model(wrapped).model_json_schema()
    assert schema["properties"]["kind"]["enum"] == ["a", "b"]

    wrapped(kind="b")
    assert result_store[0]["kwargs"] == {"kind": "b"}


def test_sanitize_tool_callable_resolves_postponed_annotations_with_skip_params():
    wrapped = _sanitize_tool_callable(load_secret_data, "Load Secret Data", ["token"])

    schema = function_to_model(wrapped).model_json_schema()
    assert list(schema["properties"]) == ["kind"]
    assert schema["properties"]["kind"]["enum"] == ["a", "b"]


# ---- Derived table names must be valid SQL identifiers ----

def test_derived_table_name_prefixes_digit_leading_numeric_argument():
    """A numeric argument value (min_size=20) must not open the name; DuckDB
    rejects identifiers like ``20_load_cells``. The name is prefixed rather
    than reordered, so existing valid names keep their shape."""
    name = SourceAgent._derive_table_name(
        tool_results=[{"action": "Load Cells", "kwargs": {"min_size": 20}}]
    )

    assert name == "table_20_load_cells"
    assert not name[0].isdigit()


def test_derived_table_name_preserves_existing_valid_names():
    name = SourceAgent._derive_table_name(
        tool_results=[
            {"action": "List Aggs", "kwargs": {"ticker": "AAPL", "timespan": "day"}}
        ]
    )

    assert name == "day_aapl_list_aggs"


def test_derived_table_name_digit_leading_action_gets_a_prefix():
    """An action named like a number ("3D Plot") still yields a valid
    identifier."""
    name = SourceAgent._derive_table_name(
        tool_results=[{"action": "3D Plot", "kwargs": {}}]
    )

    assert name == "table_3d_plot"
    assert not name[0].isdigit()


def test_derived_table_name_falls_back_for_symbol_only_action():
    """``"***"`` normalizes to an empty string, which cannot name a table."""
    name = SourceAgent._derive_table_name(
        tool_results=[{"action": "***", "kwargs": {}}]
    )

    assert name == "source_data"
    assert not name[0].isdigit()


def test_derived_table_name_from_single_source_action_is_sql_safe():
    name = SourceAgent._derive_table_name(source_actions={"2024_report": object()})

    assert name == "table_2024_report"
    assert not name[0].isdigit()


def test_derived_table_name_from_symbol_only_source_action_falls_back():
    name = SourceAgent._derive_table_name(source_actions={"***": object()})

    assert name == "source_data"


def test_derived_table_name_still_deduplicates_against_existing_tables():
    existing = {"table_20_load_cells"}

    name = SourceAgent._derive_table_name(
        tool_results=[{"action": "Load Cells", "kwargs": {"min_size": 20}}],
        existing_tables=existing,
    )

    assert name == "table_20_load_cells_2"


def test_normalize_table_name_leaves_valid_names_unchanged():
    assert normalize_table_name("my_data_file") == "my_data_file"
    assert normalize_table_name("Load Cells") == "load_cells"


def test_source_result_normalizes_payload_table_before_duckdb():
    """``SourceResult.table`` is the ``payload.table`` the agent registers; it
    reaches DuckDB unquoted, so raw upload/action names must be normalized
    there rather than only in the derived-name path."""
    pytest.importorskip("duckdb")
    from lumen.ai.controls.ingest.result import SourceResult

    result = SourceResult.from_dataframe(pd.DataFrame({"value": [1, 2]}), "2024 report")

    assert result.table == "table_2024_report"
    assert len(result.sources[0].get(result.table)) == 2

    symbol_only = SourceResult.from_dataframe(pd.DataFrame({"value": [1]}), "***")

    assert symbol_only.table == "data"
    assert len(symbol_only.sources[0].get(symbol_only.table)) == 1
