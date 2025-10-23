from datetime import datetime
from unittest.mock import MagicMock, patch

import jinja2
import numpy as np
import pandas as pd
import pytest
import yaml

from panel.chat import ChatStep

try:
    from lumen.ai.config import PROMPTS_DIR
    from lumen.ai.utils import (
        UNRECOVERABLE_ERRORS, clean_sql, describe_data, get_schema,
        merge_dicts, parse_huggingface_url, render_template, report_error,
        retry_llm_output,
    )
except ImportError:
    pytest.skip("Skipping tests that require lumen.ai", allow_module_level=True)


def test_render_template_with_valid_template():
    result = render_template(
        PROMPTS_DIR / "_Testing" / "topic.jinja2",
        {"tools": ""},
        current_datetime=datetime(2024, 1, 1, 12, 0),
        memory={}
    ).strip()
    assert "What is the topic of the data?" in result


def test_render_template_with_override():
    result = render_template(
        PROMPTS_DIR / "_Testing" / "topic.jinja2",
        {"context": "Its Lumen", "tools": ""},
        current_datetime=datetime(2024, 1, 1, 12, 0),
        memory={}
    ).strip()
    assert "What is the topic of the data?" in result
    assert "Its Lumen" in result


def test_render_template_with_missing_variable():
    with pytest.raises(jinja2.exceptions.UndefinedError):
        render_template(PROMPTS_DIR / "Actor" / "main.jinja2")


class TestRetryLLMOutput:

    @patch("time.sleep", return_value=None)
    def test_success(self, mock_sleep):
        @retry_llm_output(retries=2)
        def mock_func(errors=None):
            return "Success"

        result = mock_func()
        assert result == "Success"
        assert mock_sleep.call_count == 0

    @patch("time.sleep", return_value=None)
    def test_failure(self, mock_sleep):
        @retry_llm_output(retries=2)
        def mock_func(errors=None):
            if errors is not None:
                assert errors == ["Failed"]
            raise Exception("Failed")

        with pytest.raises(Exception):
            mock_func()
        assert mock_sleep.call_count == 1

    @patch("time.sleep", return_value=None)
    def test_failure_unrecoverable(self, mock_sleep):
        @retry_llm_output(retries=2)
        def mock_func(errors=None):
            if errors is not None:
                assert errors == ["Failed"]
            raise unrecoverable_error("Failed")

        unrecoverable_error = UNRECOVERABLE_ERRORS[0]
        with pytest.raises(unrecoverable_error, match="Failed"):
            mock_func(errors=["Failed"])
        assert mock_sleep.call_count == 0

    @patch("asyncio.sleep", return_value=None)
    async def test_async_success(self, mock_sleep):
        @retry_llm_output(retries=2)
        async def mock_func(errors=None):
            return "Success"

        result = await mock_func()
        assert result == "Success"
        assert mock_sleep.call_count == 0

    @patch("asyncio.sleep", return_value=None)
    async def test_async_failure(self, mock_sleep):
        @retry_llm_output(retries=2)
        async def mock_func(errors=None):
            if errors is not None:
                assert errors == ["Failed"]
            raise Exception("Failed")

        with pytest.raises(Exception):
            await mock_func()
        assert mock_sleep.call_count == 1

    @patch("asyncio.sleep", return_value=None)
    async def test_async_failure_unrecoverable(self, mock_sleep):
        @retry_llm_output(retries=2)
        async def mock_func(errors=None):
            if errors is not None:
                assert errors == ["Failed"]
            raise unrecoverable_error("Failed")

        unrecoverable_error = UNRECOVERABLE_ERRORS[0]
        with pytest.raises(unrecoverable_error, match="Failed"):
            await mock_func(errors=["Failed"])
        assert mock_sleep.call_count == 0


class TestGetSchema:

    async def test_get_schema_from_source(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {"field1": {"type": "integer"}}
        schema = await get_schema(mock_source)
        assert "field1" in schema
        assert schema["field1"]["type"] == "int"

    async def test_min_max(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {
            "field1": {
                "type": "integer",
                "inclusiveMinimum": 0,
                "inclusiveMaximum": 100,
            }
        }
        schema = await get_schema(mock_source, include_min_max=True)
        assert "min" in schema["field1"]
        assert "max" in schema["field1"]
        assert schema["field1"]["min"] == 0
        assert schema["field1"]["max"] == 100

    async def test_no_min_max(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {
            "field1": {
                "type": "integer",
                "inclusiveMinimum": 0,
                "inclusiveMaximum": 100,
            }
        }
        schema = await get_schema(mock_source, include_min_max=False)
        assert "min" not in schema["field1"]
        assert "max" not in schema["field1"]
        assert "inclusiveMinimum" not in schema["field1"]
        assert "inclusiveMaximum" not in schema["field1"]

    async def test_enum(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {
            "field1": {"type": "string", "enum": ["value1", "value2"]}
        }
        schema = await get_schema(mock_source, include_enum=True)
        assert "enum" in schema["field1"]
        assert schema["field1"]["enum"] == ["value1", "value2"]

    async def test_no_enum(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {
            "field1": {"type": "string", "enum": ["value1", "value2"]}
        }
        schema = await get_schema(mock_source, include_enum=False)
        assert "enum" not in schema["field1"]

    async def test_enum_limit(self):
        mock_source = MagicMock()
        # Return more than 10 enum values to trigger truncation
        mock_source.get_schema.return_value = {
            "field1": {"type": "string", "enum": [f"value{i}" for i in range(15)]}
        }
        schema = await get_schema(mock_source, include_enum=True, limit=2)
        assert "enum" in schema["field1"]
        assert "..." in schema["field1"]["enum"]

    async def test_count(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {
            "field1": {"type": "integer"},
            "__len__": 1000,
        }
        schema = await get_schema(mock_source, include_count=True)
        assert schema["__len__"] == 1000

    async def test_no_count(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {
            "field1": {"type": "integer"},
            "__len__": 1000,
        }
        schema = await get_schema(mock_source, include_count=False)
        assert "__len__" not in schema

    async def test_table(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {"field1": {"type": "integer"}}
        schema = await get_schema(mock_source, table="test_table")
        # Check that get_schema was called with shuffle parameter
        mock_source.get_schema.assert_called_once()
        call_args = mock_source.get_schema.call_args
        assert call_args[0][0] == "test_table"
        assert "shuffle" in call_args[1]
        assert "field1" in schema

    async def test_custom_limit(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {"field1": {"type": "integer"}}
        schema = await get_schema(mock_source, table="test_table", limit=50)
        # Check that get_schema was called with shuffle parameter
        mock_source.get_schema.assert_called_once()
        call_args = mock_source.get_schema.call_args
        assert call_args[0][0] == "test_table"
        assert call_args[1]["limit"] == 50
        assert "field1" in schema


class TestDescribeData:

    async def test_describe_numeric_data(self):
        df = pd.DataFrame({"col1": np.arange(0, 100000), "col2": np.arange(0, 100000)})
        result_str = await describe_data(df)
        result = yaml.safe_load(result_str)
        assert "col1" in result["stats"]
        assert "col2" in result["stats"]
        # nulls key is only present if nulls > 0
        assert result["stats"]["col1"].get("nulls", 0) == 0

    async def test_describe_with_nulls(self):
        df = pd.DataFrame({"col1": np.arange(0, 100000), "col2": np.arange(0, 100000)})
        df.loc[:5000, "col1"] = np.nan
        df.loc[:5000, "col2"] = np.nan
        result_str = await describe_data(df)
        result = yaml.safe_load(result_str)
        # With nulls present, the key should exist
        assert "nulls" in result["stats"]["col1"]
        assert result["stats"]["col1"]["nulls"] > 0
        assert result["stats"]["col2"]["nulls"] > 0

    async def test_describe_string_data(self):
        df = pd.DataFrame(
            {
                "col1": ["apple", "banana", "cherry", "date", "elderberry"] * 2000,
                "col2": ["a", "b", "c", "d", "e"] * 2000,
            }
        )
        result_str = await describe_data(df)
        result = yaml.safe_load(result_str)
        assert result["stats"]["col1"]["nunique"] == 5
        # max_length is added for string columns
        if "max_length" in result["stats"]["col2"]:
            assert result["stats"]["col2"]["max_length"] == 1

    async def test_describe_datetime_data(self):
        df = pd.DataFrame(
            {
                "col1": pd.date_range("2018-08-18", periods=10000),
                "col2": pd.date_range("2018-08-18", periods=10000),
            }
        )
        result_str = await describe_data(df)
        result = yaml.safe_load(result_str)
        assert "col1" in result["stats"]
        assert "col2" in result["stats"]

    async def test_describe_large_data(self):
        df = pd.DataFrame({"col1": range(6000), "col2": range(6000, 12000)})
        result_str = await describe_data(df)
        result = yaml.safe_load(result_str)
        # Check the summary exists and has is_sampled key
        assert "summary" in result
        assert result["summary"]["is_sampled"] is True

    async def test_describe_small_data(self):
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result_str = await describe_data(df)
        # For small dataframes, it returns yaml of the dataframe itself
        result = yaml.safe_load(result_str)
        assert isinstance(result, (list, dict))


def test_clean_sql_removes_backticks():
    sql_expr = "```sql SELECT * FROM `table`; ```"
    cleaned_sql = clean_sql(sql_expr)
    assert cleaned_sql == 'SELECT * FROM "table"'


def test_clean_sql_strips_whitespace_and_semicolons():
    sql_expr = "SELECT * FROM table;    "
    cleaned_sql = clean_sql(sql_expr)
    assert cleaned_sql == "SELECT * FROM table"


def test_report_error():
    step = ChatStep()
    report_error(Exception("Test error"), step)
    assert step.failed_title == "Test error"
    assert step.status == "failed"
    # The Details component is appended, check that something was added
    assert len(step.objects) > 0


class TestParseHuggingFaceUrl:

    def test_no_query_params(self):
        repo, file, model_kwargs = parse_huggingface_url("https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF/blob/main/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf")
        assert repo == "unsloth/Mistral-Small-24B-Instruct-2501-GGUF"
        assert file == "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
        assert model_kwargs == {}

    def test_query_params(self):
        repo, file, model_kwargs = parse_huggingface_url("https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF/blob/main/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf?chat_format=mistral-instruct&n_ctx=1028")
        assert repo == "unsloth/Mistral-Small-24B-Instruct-2501-GGUF"
        assert file == "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
        assert model_kwargs == {"chat_format": "mistral-instruct", "n_ctx": 1028}

    def test_bad_error(self):
        with pytest.raises(ValueError):
            parse_huggingface_url("https://huggingface.co/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf?chat_format=mistral-instruct&n_ctx=1028")


class TestMergeDicts:
    """Test cases for merge_dicts utility function based on Vega-Lite use cases."""

    def test_update_mode_simple_addition(self):
        """Test case 1: Add title (simple top-level addition)"""
        base = {
            "data": {"url": "data/seattle-weather.csv"},
            "mark": "bar",
            "encoding": {
                "x": {"bin": True, "field": "precipitation"},
                "y": {"aggregate": "count"}
            }
        }
        update = {"title": "Precipitation Distribution"}
        
        result = merge_dicts(base, update, mode="update")
        
        assert result["title"] == "Precipitation Distribution"
        assert result["mark"] == "bar"
        assert result["data"] == {"url": "data/seattle-weather.csv"}

    def test_update_mode_deep_nested(self):
        """Test case 3: Change xlabel (deep nested update)"""
        base = {
            "data": {"url": "data/seattle-weather.csv"},
            "mark": "bar",
            "encoding": {
                "x": {"bin": True, "field": "precipitation"},
                "y": {"aggregate": "count"}
            }
        }
        update = {
            "encoding": {
                "x": {
                    "title": "Precipitation Amount (mm)"
                }
            }
        }
        
        result = merge_dicts(base, update, mode="update")
        
        # Should preserve existing x fields and add title
        assert result["encoding"]["x"]["bin"] is True
        assert result["encoding"]["x"]["field"] == "precipitation"
        assert result["encoding"]["x"]["title"] == "Precipitation Amount (mm)"
        assert result["encoding"]["y"] == {"aggregate": "count"}

    def test_replace_mode_complete_replacement(self):
        """Test case 4: Change to timeseries (complete replacement)"""
        base = {
            "data": {"url": "data/seattle-weather.csv"},
            "mark": "bar",
            "encoding": {
                "x": {"bin": True, "field": "precipitation"},
                "y": {"aggregate": "count"}
            }
        }
        update = {
            "mark": "line",
            "encoding": {
                "x": {
                    "field": "date",
                    "type": "temporal",
                    "title": "Date"
                },
                "y": {
                    "field": "precipitation",
                    "type": "quantitative",
                    "aggregate": "mean",
                    "title": "Average Precipitation"
                }
            }
        }
        
        result = merge_dicts(base, update, mode="replace")
        
        # Should completely replace mark and encoding
        assert result["mark"] == "line"
        assert result["encoding"]["x"]["field"] == "date"
        assert result["encoding"]["x"]["type"] == "temporal"
        assert "bin" not in result["encoding"]["x"]

    def test_append_layers_mode(self):
        """Test case 2: Add red reference line (append layers)"""
        base = {
            "data": {"url": "data/seattle-weather.csv"},
            "layer": [
                {
                    "mark": "bar",
                    "encoding": {
                        "x": {"bin": True, "field": "precipitation"},
                        "y": {"aggregate": "count"}
                    }
                }
            ]
        }
        update = {
            "layer": [
                {
                    "mark": {"type": "rule", "color": "red", "strokeWidth": 2},
                    "encoding": {
                        "y": {"datum": 1000}
                    }
                }
            ]
        }
        
        result = merge_dicts(base, update, mode="append_layers")
        
        # Should have 2 layers now
        assert len(result["layer"]) == 2
        assert result["layer"][0]["mark"] == "bar"
        assert result["layer"][1]["mark"]["type"] == "rule"
        assert result["layer"][1]["mark"]["color"] == "red"

    def test_update_mode_hconcat_merge(self):
        """Test hconcat element-by-element merge"""
        base = {
            "hconcat": [
                {"mark": "bar", "encoding": {"x": {"field": "a"}}},
                {"mark": "line", "encoding": {"x": {"field": "b"}}}
            ]
        }
        update = {
            "hconcat": [
                {"encoding": {"y": {"field": "count"}}},
                {"mark": "area"}
            ]
        }
        
        result = merge_dicts(base, update, mode="update")
        
        # First item should merge
        assert result["hconcat"][0]["mark"] == "bar"
        assert result["hconcat"][0]["encoding"]["x"] == {"field": "a"}
        assert result["hconcat"][0]["encoding"]["y"] == {"field": "count"}
        
        # Second item should update mark
        assert result["hconcat"][1]["mark"] == "area"
        assert result["hconcat"][1]["encoding"]["x"] == {"field": "b"}

    def test_update_mode_vconcat_merge(self):
        """Test vconcat element-by-element merge"""
        base = {
            "vconcat": [
                {"mark": "bar"},
                {"mark": "line"}
            ]
        }
        update = {
            "vconcat": [
                {"title": "First Plot"}
            ]
        }
        
        result = merge_dicts(base, update, mode="update")
        
        # First item should get title
        assert result["vconcat"][0]["mark"] == "bar"
        assert result["vconcat"][0]["title"] == "First Plot"
        
        # Second item unchanged
        assert result["vconcat"][1]["mark"] == "line"

    def test_empty_update(self):
        """Test that empty update returns base unchanged"""
        base = {"mark": "bar", "data": {"url": "test.csv"}}
        
        result = merge_dicts(base, {}, mode="update")
        
        assert result == base

    def test_update_mode_scalar_replacement(self):
        """Test that scalar values are replaced in update mode"""
        base = {"width": 400, "height": 300}
        update = {"width": 600}
        
        result = merge_dicts(base, update, mode="update")
        
        assert result["width"] == 600
        assert result["height"] == 300

    def test_append_layers_removes_conflicting_props(self):
        """Test that append_layers mode removes top-level mark and encoding"""
        base = {
            "mark": "bar",
            "encoding": {"x": {"field": "a"}},
            "layer": [{"mark": "point"}]
        }
        update = {
            "layer": [{"mark": "rule"}]
        }
        
        result = merge_dicts(base, update, mode="append_layers")
        
        # Top-level mark and encoding should be removed
        assert "mark" not in result
        assert "encoding" not in result
        assert len(result["layer"]) == 2

    def test_append_layers_with_list_input(self):
        """Test appending layers when update is already a list (auto-wrapping case)"""
        base = {
            "layer": [{"mark": "bar"}]
        }
        # Simulate what happens when LLM returns a list directly
        update_list = [{"mark": {"type": "rule", "color": "red"}}]
        
        # Code should wrap it like this before calling merge_dicts
        update = {"layer": update_list}
        
        result = merge_dicts(base, update, mode="append_layers")
        
        assert len(result["layer"]) == 2
        assert result["layer"][0]["mark"] == "bar"
        assert result["layer"][1]["mark"]["color"] == "red"

    def test_update_mode_layered_to_layered(self):
        """Test updating a layered spec with a layered update (full hierarchy)"""
        base = {
            "layer": [
                {
                    "mark": "bar",
                    "encoding": {
                        "x": {"aggregate": "sum", "field": "frequency"},
                        "y": {"bin": True, "field": "input_tokens"}
                    }
                }
            ]
        }
        # LLM returns full hierarchy with only the changed property
        update = {
            "layer": [
                {
                    "encoding": {
                        "x": {"aggregate": "count"}  # Changed from sum to count
                    }
                }
            ]
        }
        
        result = merge_dicts(base, update, mode="update")
        
        # Should preserve unchanged properties and update the changed one
        assert result["layer"][0]["mark"] == "bar"
        assert result["layer"][0]["encoding"]["x"]["aggregate"] == "count"
        assert result["layer"][0]["encoding"]["x"]["field"] == "frequency"
        assert result["layer"][0]["encoding"]["y"]["bin"] is True

    def test_append_layers_requires_base_layer(self):
        """Test that append_layers expects base to already have layers (per design)"""
        # Per our design, LLM should ALWAYS create layered specs from the start
        # So base should always have layer array
        base = {
            "layer": [{"mark": "bar", "encoding": {"x": {"field": "a"}}}]
        }
        update = {
            "layer": [{"mark": "rule", "encoding": {"y": {"datum": 100}}}]
        }
        
        result = merge_dicts(base, update, mode="append_layers")
        
        assert len(result["layer"]) == 2
        # Conflicting top-level props removed
        assert "mark" not in result
        assert "encoding" not in result
