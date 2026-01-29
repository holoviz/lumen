import datetime as dt

from unittest.mock import MagicMock, patch

import jinja2
import numpy as np
import pandas as pd
import pytest
import yaml

from panel.chat import ChatStep
from pydantic import ValidationError

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.config import PROMPTS_DIR
from lumen.ai.models import DeleteLine, InsertLine, ReplaceLine
from lumen.ai.utils import (
    UNRECOVERABLE_ERRORS, apply_changes, clean_sql, describe_data, get_schema,
    parse_huggingface_url, render_template, report_error, retry_llm_output,
)


def test_render_template_with_valid_template():
    now = dt.datetime.now()
    expected = (
        "Do not excessively reason in responses; there are chain_of_thought fields for that, but those should also be concise (1-2 sentences).\n"
        f"The current date time is {now.strftime('%b %d, %Y %I:%M %p')}\n"
        "What is the topic of the data?"
    )
    assert (
        render_template(PROMPTS_DIR / "_Testing" / "topic.jinja2", {"tools": ""}, current_datetime=now).strip()
        == expected
    )


def test_render_template_with_override():
    now = dt.datetime.now()
    expected = (
        "Do not excessively reason in responses; there are chain_of_thought fields for that, but those should also be concise (1-2 sentences).\n"
        f"The current date time is {now.strftime('%b %d, %Y %I:%M %p')}\n"
        "What is the topic of the data?\n"
        "Its Lumen"
    )
    assert (
        render_template(PROMPTS_DIR / "_Testing" / "topic.jinja2", {"context": "Its Lumen", "tools": ""}, current_datetime=now).strip()
        == expected
    )


def test_render_template_with_missing_variable():
    with pytest.raises(jinja2.exceptions.UndefinedError):
        render_template(PROMPTS_DIR / "SQLAgent" / "main.jinja2")


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

        with pytest.raises(Exception, match="Maximum number of retries exceeded."):
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

        with pytest.raises(Exception, match="Maximum number of retries exceeded."):
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
        mock_source.get_schema.return_value = {
            "field1": {"type": "string", "enum": ["value1", "value2", "value3"]}
        }
        schema = await get_schema(mock_source, include_enum=True, limit=2)
        assert "enum" in schema["field1"]
        assert schema["field1"]["enum"] == ['value1', 'value2', 'value3']

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
        mock_source.get_schema.assert_called_with("test_table", shuffle=True, limit=100)
        assert "field1" in schema

    async def test_custom_limit(self):
        mock_source = MagicMock()
        mock_source.get_schema.return_value = {"field1": {"type": "integer"}}
        schema = await get_schema(mock_source, table="test_table", limit=50)
        mock_source.get_schema.assert_called_with("test_table", shuffle=True, limit=50)
        assert "field1" in schema


class TestDescribeData:

    async def test_describe_numeric_data(self):
        df = pd.DataFrame({"col1": np.arange(0, 100000), "col2": np.arange(0, 100000)})
        result = yaml.load(await describe_data(df), yaml.SafeLoader)
        assert "col1" in result["stats"]
        assert "col2" in result["stats"]
        assert result["stats"]["col1"]["count"] == 5000
        assert result["stats"]["col2"]["count"] == 5000

    async def test_describe_with_nulls(self):
        df = pd.DataFrame({"col1": np.arange(0, 100000), "col2": np.arange(0, 100000)})
        df.loc[:5000, "col1"] = np.nan
        df.loc[:5000, "col2"] = np.nan
        result = yaml.load(await describe_data(df), yaml.SafeLoader)
        assert result["stats"]["col1"]["nulls"] != "0"
        assert result["stats"]["col2"]["nulls"] != "0"

    async def test_describe_string_data(self):
        df = pd.DataFrame(
            {
                "col1": ["apple", "banana", "cherry", "date", "elderberry"] * 2000,
                "col2": ["a", "b", "c", "d", "e"] * 2000,
            }
        )
        result = yaml.load(await describe_data(df), yaml.SafeLoader)
        assert result["stats"]["col1"]["nunique"] == 5
        assert result["stats"]["col1"]["max_length"] == 10
        assert result["stats"]["col2"]["max_length"] == 1

    async def test_describe_datetime_data(self):
        df = pd.DataFrame(
            {
                "col1": pd.date_range("2018-08-18", periods=10000),
                "col2": pd.date_range("2018-08-18", periods=10000),
            }
        )
        result = yaml.load(await describe_data(df), yaml.SafeLoader)
        assert "col1" in result["stats"]
        assert "col2" in result["stats"]

    async def test_describe_large_data(self):
        df = pd.DataFrame({"col1": range(6000), "col2": range(6000, 12000)})
        result = yaml.load(await describe_data(df), yaml.SafeLoader)
        assert "summary" in result

    async def test_describe_small_data(self):
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result = await describe_data(df)
        assert "|   col1 |   col2 |" in result


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
    assert step.objects[1].objects[0].object == "```python\nTest error\n```"


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


def test_no_edits_returns_original():
    lines = ["a", "b", "c"]
    out = apply_changes(lines, [])
    assert out == "a\nb\nc"


def test_single_replace_middle_line():
    lines = ["a", "b", "c"]
    edits = [ReplaceLine(line_no=2, line="B")]  # replace "b"
    out = apply_changes(lines, edits)
    assert out == "a\nB\nc"


def test_single_delete_first_line():
    lines = ["a", "b", "c"]
    edits = [DeleteLine(line_no=1)]  # delete "a"
    out = apply_changes(lines, edits)
    assert out == "b\nc"


def test_single_delete_last_line():
    lines = ["a", "b", "c"]
    edits = [DeleteLine(line_no=3)]  # delete "c"
    out = apply_changes(lines, edits)
    assert out == "a\nb"


def test_single_insert_before_middle_line():
    lines = ["a", "b", "c"]
    edits = [InsertLine(line_no=2, line="X")]  # insert BEFORE 2 ("b")
    out = apply_changes(lines, edits)
    assert out == "a\nX\nb\nc"


def test_append_insert_at_len_plus_one():
    lines = ["a", "b", "c"]
    edits = [InsertLine(line_no=len(lines) + 1, line="X")]  # append
    out = apply_changes(lines, edits)
    assert out == "a\nb\nc\nX"


def test_multiple_inserts_same_index_preserve_order():
    lines = ["a", "b", "c"]
    edits = [
        InsertLine(line_no=2, line="X1"),
        InsertLine(line_no=2, line="X2"),
    ]
    # Both inserted BEFORE line 2 ("b") in the given order
    out = apply_changes(lines, edits)
    assert out == "a\nX1\nX2\nb\nc"


def test_replace_and_delete_indices_based_on_original_descending():
    # Original: 1:A, 2:B, 3:C, 4:D
    lines = ["A", "B", "C", "D"]
    edits = [
        DeleteLine(line_no=2),                 # remove B
        ReplaceLine(line_no=3, line="X"),      # replace C
    ]
    # After replace/delete (applied using original positions):
    # 1:A, 2:X, 4:D
    out = apply_changes(lines, edits)
    assert out == "A\nX\nD"


def test_mix_insert_replace_delete():
    # Start: 1:A, 2:B, 3:C
    lines = ["A", "B", "C"]
    edits = [
        InsertLine(line_no=1, line="X"),   # before A
        ReplaceLine(line_no=2, line="b2"), # replace B -> b2
        DeleteLine(line_no=3),             # delete C
    ]

    # Steps:
    # replace/delete first:
    #   Replace line 2 → A, b2, C
    #   Delete line 3 → A, b2
    #
    # Insert before line 1 → X, A, b2
    out = apply_changes(lines, edits)
    assert out == "X\nA\nb2"


def test_replace_line_no_out_of_range_raises():
    lines = ["a", "b"]
    # replace line_no must be 1..len
    with pytest.raises(IndexError):
        apply_changes(lines, [ReplaceLine(line_no=3, line="X")])
    with pytest.raises(ValidationError):
        apply_changes(lines, [ReplaceLine(line_no=0, line="X")])


def test_delete_line_no_out_of_range_raises():
    lines = ["a", "b"]
    with pytest.raises(IndexError):
        apply_changes(lines, [DeleteLine(line_no=3)])
    with pytest.raises(ValidationError):
        apply_changes(lines, [DeleteLine(line_no=0)])
