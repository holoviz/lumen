from pathlib import Path
from unittest.mock import MagicMock, patch

import jinja2
import numpy as np
import pandas as pd
import pytest

from panel.chat import ChatStep

try:
    from lumen.ai.utils import (
        UNRECOVERABLE_ERRORS, clean_sql, describe_data, format_schema,
        get_schema, render_template, report_error, retry_llm_output,
    )
except ImportError:
    pytest.skip("Skipping tests that require lumen.ai", allow_module_level=True)


def test_render_template_with_valid_template():
    template_content = "Hello {{ name }}!"
    with patch.object(Path, "read_text", return_value=template_content):
        result = render_template("test_template.txt", name="World")
        assert result == "Hello World!"


def test_render_template_missing_key():
    template_content = "Hello {{ name }}!"
    with patch.object(Path, "read_text", return_value=template_content):
        with pytest.raises(jinja2.exceptions.UndefinedError):
            render_template("test_template.txt")


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

        with pytest.raises(Exception, match="Failed"):
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

        with pytest.raises(Exception, match="Failed"):
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


def test_format_schema_with_enum():
    schema = {
        "field1": {"type": "string", "enum": ["a", "b", "c", "d", "e", "f"]},
        "field2": {"type": "integer"},
    }
    expected = {
        "field1": {"type": "str", "enum": ["a", "b", "c", "d", "e", "..."]},
        "field2": {"type": "int"},
    }
    assert format_schema(schema) == expected


def test_format_schema_no_enum():
    schema = {
        "field1": {"type": "boolean"},
        "field2": {"type": "integer"},
    }
    expected = {
        "field1": {"type": "bool"},
        "field2": {"type": "int"},
    }
    assert format_schema(schema) == expected


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
        mock_source.get_schema.return_value = {
            "field1": {"type": "string", "enum": ["value1", "value2", "value3"]}
        }
        schema = await get_schema(mock_source, include_enum=True, limit=2)
        assert "enum" in schema["field1"]
        assert "..." in schema["field1"]["enum"]

    async def test_count(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {
            "field1": {"type": "integer"},
            "count": 1000,
        }
        schema = await get_schema(mock_source, include_count=True)
        assert "count" in schema["field1"]
        assert schema["field1"]["count"] == 1000

    async def test_no_count(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {
            "field1": {"type": "integer"},
            "count": 1000,
        }
        schema = await get_schema(mock_source, include_count=False)
        assert "count" not in schema

    async def test_table(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {"field1": {"type": "integer"}}
        schema = await get_schema(mock_source, table="test_table")
        mock_source.get_schema.assert_called_with("test_table", limit=100)
        assert "field1" in schema

    async def test_custom_limit(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {"field1": {"type": "integer"}}
        schema = await get_schema(mock_source, table="test_table", limit=50)
        mock_source.get_schema.assert_called_with("test_table", limit=50)
        assert "field1" in schema

class TestDescribeData:

    async def test_describe_numeric_data(self):
        df = pd.DataFrame({
            "col1": np.arange(0, 100000),
            "col2": np.arange(0, 100000)
        })
        result = await describe_data(df)
        assert "col1" in result["stats"]
        assert "col2" in result["stats"]
        assert result["stats"]["col1"]["nulls"] == '0'
        assert result["stats"]["col2"]["nulls"] == '0'

    async def test_describe_with_nulls(self):
        df = pd.DataFrame({
            "col1": np.arange(0, 100000),
            "col2": np.arange(0, 100000)
        })
        df.loc[:5000, "col1"] = np.nan
        df.loc[:5000, "col2"] = np.nan
        result = await describe_data(df)
        assert result["stats"]["col1"]["nulls"] != '0'
        assert result["stats"]["col2"]["nulls"] != '0'

    async def test_describe_string_data(self):
        df = pd.DataFrame({
            "col1": ["apple", "banana", "cherry", "date", "elderberry"] * 2000,
            "col2": ["a", "b", "c", "d", "e"] * 2000
        })
        result = await describe_data(df)
        assert result["stats"]["col1"]["nunique"] == 5
        assert result["stats"]["col2"]["lengths"]["max"] == 1
        assert result["stats"]["col1"]["lengths"]["max"] == 10

    async def test_describe_datetime_data(self):
        df = pd.DataFrame({
            "col1": pd.date_range("2018-08-18", periods=10000),
            "col2": pd.date_range("2018-08-18", periods=10000),
        })
        result = await describe_data(df)
        assert "col1" in result["stats"]
        assert "col2" in result["stats"]

    async def test_describe_large_data(self):
        df = pd.DataFrame({
            "col1": range(6000),
            "col2": range(6000, 12000)
        })
        result = await describe_data(df)
        assert result["summary"]["is_summarized"] is True
        assert len(df.sample(5000)) == 5000  # Should summarize to 5000 rows

    async def test_describe_small_data(self):
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": [3, 4]
        })
        result = await describe_data(df)
        assert result.equals(df)


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
    assert step.objects[0].object == "\n```python\nTest error\n```"
