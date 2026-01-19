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
from lumen.ai.models import SearchReplace, SearchReplaceSpec
from lumen.ai.utils import (
    UNRECOVERABLE_ERRORS, apply_search_replace, clean_sql, describe_data,
    get_schema, parse_huggingface_url, render_template, report_error,
    retry_llm_output,
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
    assert step.objects[1].object.object == "```python\nTest error\n```"


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


# ============================================================================
# SearchReplaceSpec Tests - apply_search_replace
# ============================================================================

class TestApplySearchReplace:
    """Test the apply_search_replace function for search/replace based editing."""

    def test_no_edits_returns_original(self):
        """Empty edits list should return original text unchanged."""
        text = "line one\nline two\nline three"
        result = apply_search_replace(text, [])
        assert result == text

    def test_single_simple_replace(self):
        """Basic single search/replace operation."""
        text = "Hello World"
        edits = [SearchReplace(old_str="World", new_str="Universe")]
        result = apply_search_replace(text, edits)
        assert result == "Hello Universe"

    def test_replace_entire_line(self):
        """Replace an entire line."""
        text = "line one\nline two\nline three"
        edits = [SearchReplace(old_str="line two", new_str="LINE TWO MODIFIED")]
        result = apply_search_replace(text, edits)
        assert result == "line one\nLINE TWO MODIFIED\nline three"

    def test_delete_text_empty_new_str(self):
        """Using empty new_str should delete the matched text."""
        text = "Hello World"
        edits = [SearchReplace(old_str=" World", new_str="")]
        result = apply_search_replace(text, edits)
        assert result == "Hello"

    def test_multiple_edits_applied_sequentially(self):
        """Multiple edits should be applied in order."""
        text = "apple banana cherry"
        edits = [
            SearchReplace(old_str="apple", new_str="APPLE"),
            SearchReplace(old_str="cherry", new_str="CHERRY"),
        ]
        result = apply_search_replace(text, edits)
        assert result == "APPLE banana CHERRY"

    def test_multiline_old_str(self):
        """old_str can span multiple lines."""
        text = "line one\nline two\nline three"
        edits = [SearchReplace(
            old_str="line one\nline two",
            new_str="COMBINED LINES"
        )]
        result = apply_search_replace(text, edits)
        assert result == "COMBINED LINES\nline three"

    def test_multiline_new_str(self):
        """new_str can expand to multiple lines."""
        text = "line one\nline two\nline three"
        edits = [SearchReplace(
            old_str="line two",
            new_str="line two-a\nline two-b\nline two-c"
        )]
        result = apply_search_replace(text, edits)
        assert result == "line one\nline two-a\nline two-b\nline two-c\nline three"

    def test_whitespace_preserved_in_match(self):
        """Whitespace is matched exactly when present."""
        text = "  indented line"
        edits = [SearchReplace(old_str="  indented line", new_str="  modified line")]
        result = apply_search_replace(text, edits)
        assert result == "  modified line"

    # =========================================================================
    # Edge Case: Duplicate text (THE CRITICAL CASE)
    # =========================================================================

    def test_duplicate_text_workaround_with_context(self):
        """Workaround: include surrounding context to make old_str unique."""
        text = "item: foo\nother: bar\nitem: foo\nlast: baz"
        # "item: foo" appears twice, but with different surrounding context
        # Include the line before to disambiguate
        edits = [SearchReplace(
            old_str="other: bar\nitem: foo",
            new_str="other: bar\nitem: MODIFIED_FOO"
        )]
        result = apply_search_replace(text, edits)
        assert "item: MODIFIED_FOO" in result
        # First "item: foo" should be unchanged
        assert result.startswith("item: foo")

    def test_duplicate_lines_exact_same_content(self):
        """When lines are exactly the same, must use context to disambiguate."""
        text = "header\n- item\n- item\n- item\nfooter"
        # All three "- item" lines are identical
        # Must include context to target specific one
        edits = [SearchReplace(
            old_str="header\n- item",
            new_str="header\n- FIRST_ITEM"
        )]
        result = apply_search_replace(text, edits)
        lines = result.split("\n")
        assert lines[1] == "- FIRST_ITEM"
        assert lines[2] == "- item"  # Second unchanged
        assert lines[3] == "- item"  # Third unchanged

    def test_duplicate_at_different_indent_levels(self):
        """Same content at different indentation levels are different strings."""
        text = "field: value\n  field: value\n    field: value"
        # Each "field: value" has different indentation, so they're unique
        edits = [SearchReplace(old_str="  field: value", new_str="  field: MODIFIED")]
        result = apply_search_replace(text, edits)
        assert result == "field: value\n  field: MODIFIED\n    field: value"

    # =========================================================================
    # YAML-specific edge cases
    # =========================================================================

    def test_yaml_indentation_preserved(self):
        """YAML indentation should be preserved exactly."""
        text = """encoding:
  x:
    field: old_value
    type: quantitative"""
        edits = [SearchReplace(
            old_str="    field: old_value",
            new_str="    field: new_value"
        )]
        result = apply_search_replace(text, edits)
        assert "    field: new_value" in result
        assert "    type: quantitative" in result

    def test_yaml_add_new_property(self):
        """Add a new property by including anchor context."""
        text = """title: Chart
encoding:
  x:
    field: value"""
        # Insert new line after title by replacing title line with title + new line
        edits = [SearchReplace(
            old_str="title: Chart\nencoding:",
            new_str="title: Chart\nsubtitle: Added\nencoding:"
        )]
        result = apply_search_replace(text, edits)
        assert "subtitle: Added" in result

    def test_yaml_delete_property(self):
        """Delete a property by replacing with empty context."""
        text = """title: Chart
subtitle: To Remove
encoding:"""
        edits = [SearchReplace(
            old_str="\nsubtitle: To Remove",
            new_str=""
        )]
        result = apply_search_replace(text, edits)
        assert result == "title: Chart\nencoding:"

    def test_yaml_nested_dict_replace(self):
        """Replace a nested dictionary value."""
        text = """encoding:
  x:
    axis:
      title: Old Title
      labelAngle: 45"""
        edits = [SearchReplace(
            old_str="      title: Old Title",
            new_str="      title: New Title"
        )]
        result = apply_search_replace(text, edits)
        assert "      title: New Title" in result

    def test_yaml_list_item_modification(self):
        """Modify an item in a YAML list."""
        text = """transform:
- filter: datum.x > 0
- calculate: datum.x * 2
- filter: datum.y > 0"""
        edits = [SearchReplace(
            old_str="- calculate: datum.x * 2",
            new_str="- calculate: datum.x * 10"
        )]
        result = apply_search_replace(text, edits)
        assert "- calculate: datum.x * 10" in result

    def test_yaml_special_characters(self):
        """Handle YAML special characters like colons and quotes."""
        text = """mark:
  color: '#1f77b4'
  tooltip: 'Value: {datum.value}'"""
        edits = [SearchReplace(
            old_str="  color: '#1f77b4'",
            new_str="  color: '#ff0000'"
        )]
        result = apply_search_replace(text, edits)
        assert "  color: '#ff0000'" in result

    def test_yaml_boolean_values(self):
        """Handle YAML boolean values."""
        text = """axis:
  grid: true
  labels: false"""
        edits = [SearchReplace(old_str="  grid: true", new_str="  grid: false")]
        result = apply_search_replace(text, edits)
        assert "  grid: false" in result

    def test_yaml_null_values(self):
        """Handle YAML null values."""
        text = """axis:
  title: Original
  grid: true"""
        edits = [SearchReplace(old_str="  title: Original", new_str="  title: null")]
        result = apply_search_replace(text, edits)
        assert "  title: null" in result

    # =========================================================================
    # Vega-Lite specific scenarios
    # =========================================================================

    def test_vegalite_add_layer(self):
        """Add a new layer to a Vega-Lite spec."""
        text = """layer:
- mark: bar
  encoding:
    x: {field: category}"""
        edits = [SearchReplace(
            old_str="    x: {field: category}",
            new_str="""    x: {field: category}
- mark: rule
  encoding:
    y: {datum: 50}"""
        )]
        result = apply_search_replace(text, edits)
        assert "- mark: rule" in result
        assert "    y: {datum: 50}" in result

    def test_vegalite_modify_encoding(self):
        """Modify an encoding in a Vega-Lite spec."""
        text = """encoding:
  x:
    field: category
    type: nominal
  y:
    field: value
    type: quantitative"""
        edits = [SearchReplace(
            old_str="    field: category\n    type: nominal",
            new_str="    field: category\n    type: nominal\n    axis:\n      labelAngle: -45"
        )]
        result = apply_search_replace(text, edits)
        assert "axis:" in result
        assert "labelAngle: -45" in result

    def test_vegalite_full_spec_modification(self):
        """Test a realistic full Vega-Lite spec modification."""
        text = """$schema: https://vega.github.io/schema/vega-lite/v5.json
data:
  name: source_data
mark: bar
encoding:
  x:
    field: category
    type: nominal
  y:
    field: value
    type: quantitative
title: Simple Chart"""
        edits = [
            SearchReplace(
                old_str="title: Simple Chart",
                new_str="title:\n  text: Improved Chart\n  subtitle: With better formatting"
            ),
            SearchReplace(
                old_str="mark: bar",
                new_str="mark:\n  type: bar\n  color: steelblue"
            ),
        ]
        result = apply_search_replace(text, edits)
        assert "  text: Improved Chart" in result
        assert "  subtitle: With better formatting" in result
        assert "  type: bar" in result
        assert "  color: steelblue" in result

    # =========================================================================
    # Interaction with SearchReplaceSpec model validation
    # =========================================================================

    def test_searchreplacespec_duplicate_old_str_validation(self):
        """SearchReplaceSpec validates that old_str values are unique within edits."""
        with pytest.raises(ValidationError, match="Duplicate old_str"):
            SearchReplaceSpec(
                chain_of_thought="Test",
                edits=[
                    SearchReplace(old_str="same", new_str="A"),
                    SearchReplace(old_str="same", new_str="B"),
                ]
            )

    def test_searchreplacespec_valid_model(self):
        """SearchReplaceSpec accepts valid input."""
        spec = SearchReplaceSpec(
            chain_of_thought="Making changes",
            edits=[
                SearchReplace(old_str="old1", new_str="new1"),
                SearchReplace(old_str="old2", new_str="new2"),
            ]
        )
        assert len(spec.edits) == 2

    # =========================================================================
    # Chaining edits - second edit operates on result of first
    # =========================================================================

    def test_chained_edits_operate_on_intermediate_result(self):
        """Each edit operates on the result of previous edits."""
        text = "A B C"
        edits = [
            SearchReplace(old_str="A", new_str="X"),
            SearchReplace(old_str="X B", new_str="Y"),  # This works because A->X already happened
        ]
        result = apply_search_replace(text, edits)
        assert result == "Y C"

    def test_chained_edits_can_reference_new_content(self):
        """Later edits can reference content added by earlier edits."""
        text = "start end"
        edits = [
            SearchReplace(old_str="start", new_str="start MIDDLE"),
            SearchReplace(old_str="MIDDLE end", new_str="MIDDLE and end"),
        ]
        result = apply_search_replace(text, edits)
        assert result == "start MIDDLE and end"

    # =========================================================================
    # Edge cases with newlines and empty strings
    # =========================================================================

    def test_replace_with_empty_string_deletes(self):
        """Replacing with empty string effectively deletes."""
        text = "keep this\ndelete this\nkeep this too"
        edits = [SearchReplace(old_str="\ndelete this", new_str="")]
        result = apply_search_replace(text, edits)
        assert result == "keep this\nkeep this too"

    def test_empty_text_input(self):
        """Empty input text with no edits returns empty."""
        result = apply_search_replace("", [])
        assert result == ""

    def test_newline_only_text(self):
        """Handle text that's just newlines."""
        text = "\n\n\n"
        edits = [SearchReplace(old_str="\n\n", new_str="\n")]
        result = apply_search_replace(text, edits)
        assert result == "\n\n"

    def test_trailing_newline_preserved(self):
        """Trailing newlines should be preserved."""
        text = "line one\nline two\n"
        edits = [SearchReplace(old_str="line one", new_str="LINE ONE")]
        result = apply_search_replace(text, edits)
        assert result == "LINE ONE\nline two\n"

    # =========================================================================
    # Unicode and special characters
    # =========================================================================

    def test_unicode_characters(self):
        """Handle unicode characters in search/replace."""
        text = "emoji: 🎉 and text"
        edits = [SearchReplace(old_str="🎉", new_str="🚀")]
        result = apply_search_replace(text, edits)
        assert result == "emoji: 🚀 and text"

    def test_regex_special_chars_are_literal(self):
        """Regex special characters should be treated literally."""
        text = "value: a.*b+c?d"
        # These are regex special chars but should match literally
        edits = [SearchReplace(old_str="a.*b+c?d", new_str="replaced")]
        result = apply_search_replace(text, edits)
        assert result == "value: replaced"

    # =========================================================================
    # Insert-like operations using search/replace
    # =========================================================================

    def test_insert_using_anchor_context(self):
        """Insert new content by using anchor context."""
        text = "line1\nline2\nline3"
        # To insert between line1 and line2, replace line1 with line1 + new content
        edits = [SearchReplace(
            old_str="line1\nline2",
            new_str="line1\nNEW_LINE\nline2"
        )]
        result = apply_search_replace(text, edits)
        assert result == "line1\nNEW_LINE\nline2\nline3"

    def test_append_to_end(self):
        """Append content to end of document."""
        text = "line1\nline2"
        edits = [SearchReplace(
            old_str="line2",
            new_str="line2\nline3"
        )]
        result = apply_search_replace(text, edits)
        assert result == "line1\nline2\nline3"

    def test_prepend_to_beginning(self):
        """Prepend content to beginning of document."""
        text = "line1\nline2"
        edits = [SearchReplace(
            old_str="line1",
            new_str="line0\nline1"
        )]
        result = apply_search_replace(text, edits)
        assert result == "line0\nline1\nline2"
