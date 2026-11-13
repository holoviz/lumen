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


# ---- Derived table names must be valid SQL identifiers ----

def test_derived_table_name_does_not_start_with_a_digit():
    """A numeric argument value (min_size=20) must not open the name; DuckDB
    rejects identifiers like ``20_load_cells``."""
    name = SourceAgent._derive_table_name(
        tool_results=[{"action": "Load Cells", "kwargs": {"min_size": 20}}]
    )

    assert name == "load_cells_20"
    assert not name[0].isdigit()


def test_derived_table_name_leads_with_the_action_name():
    name = SourceAgent._derive_table_name(
        tool_results=[
            {"action": "List Aggs", "kwargs": {"ticker": "AAPL", "timespan": "day"}}
        ]
    )

    assert name == "list_aggs_day_aapl"


def test_derived_table_name_digit_leading_action_gets_a_prefix():
    """An action named like a number ("3D Plot") still yields a valid
    identifier."""
    name = SourceAgent._derive_table_name(
        tool_results=[{"action": "3D Plot", "kwargs": {}}]
    )

    assert name == "table_3d_plot"
    assert not name[0].isdigit()


def test_derived_table_name_still_deduplicates_against_existing_tables():
    existing = {"load_cells_20"}

    name = SourceAgent._derive_table_name(
        tool_results=[{"action": "Load Cells", "kwargs": {"min_size": 20}}],
        existing_tables=existing,
    )

    assert name == "load_cells_20_2"
