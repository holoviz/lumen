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


def test_derive_table_name_numeric_value_leads_with_action_name():
    # Issue #2134: a numeric kwarg value sorted first, so the derived name
    # started with a digit ("20_load_cells") and DuckDB rejected it.
    name = SourceAgent._derive_table_name(tool_results=[{"action": "Load Cells", "kwargs": {"min_size": 20}}])
    assert name == "load_cells_20"
    assert not name[0].isdigit()


def test_derive_table_name_numeric_value_loads_from_duckdb():
    # The issue's end-to-end reproducer: the derived name must be usable
    # as an unquoted table name in DuckDB.
    from lumen.sources.duckdb import DuckDBSource

    name = SourceAgent._derive_table_name(tool_results=[{"action": "Load Cells", "kwargs": {"min_size": 20}}])
    source = DuckDBSource.from_df(tables={name: pd.DataFrame({"area": [1.0]})})
    source.tables[name] = f"SELECT * FROM {name}"
    assert source.get(name).shape == (1, 1)


def test_derive_table_name_keeps_short_values_first_otherwise():
    # The shortest-first ordering is unchanged when no value is numeric.
    name = SourceAgent._derive_table_name(
        tool_results=[
            {
                "action": "List Aggs",
                "kwargs": {"ticker": "AAPL", "timespan": "day", "from_": "2024-01-01"},
            }
        ]
    )
    assert name == "day_aapl_2024_01_01_list_aggs"


def test_derive_table_name_digit_leading_source_action_gets_prefix():
    name = SourceAgent._derive_table_name(source_actions={"2024 report": None})
    assert name == "table_2024_report"
    assert not name[0].isdigit()
