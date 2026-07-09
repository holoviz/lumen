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
    IMAGE_MIME_TYPES, UNRECOVERABLE_ERRORS, apply_changes, clean_sql,
    content_to_text, describe_data, find_slug_by_table_name,
    format_msg_content, fuse_messages, get_schema, mutate_user_message,
    parse_huggingface_url, render_template, report_error, retry_llm_output,
    serialize_image_content, set_content_text, slug_to_table_name,
)
from lumen.config import SOURCE_TABLE_SEPARATOR as SEP


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
        assert "col1: 1" in result
        assert "col2: 3" in result


def test_clean_sql_removes_backticks():
    sql_expr = "```sql SELECT * FROM `table`; ```"
    cleaned_sql = clean_sql(sql_expr)
    assert cleaned_sql == 'SELECT * FROM "table"'


def test_clean_sql_strips_whitespace_and_semicolons():
    sql_expr = "SELECT * FROM table;    "
    cleaned_sql = clean_sql(sql_expr)
    assert cleaned_sql == "SELECT * FROM table"


def test_clean_sql_prettify_with_legacy_any_dialect():
    """BaseSQLSource.dialect defaults to 'any', a legacy value that sqlglot
    no longer recognises (28.x raises 'Unknown dialect any'). clean_sql
    must normalise it to the dialect-agnostic mode so calling code on a
    fresh BaseSQLSource (e.g. STACSource) does not break the SQLAgent
    chat path."""
    cleaned = clean_sql("SELECT 1", dialect="any", prettify=True)
    assert "SELECT" in cleaned and "1" in cleaned


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


class TestSlugToTableName:

    def test_source_qualified(self):
        assert slug_to_table_name(f"DuckDB001{SEP}my_table") == "my_table"

    def test_bare_name_unchanged(self):
        assert slug_to_table_name("my_table") == "my_table"

    def test_multiple_separators_splits_on_first(self):
        slug = f"src{SEP}schema{SEP}table"
        assert slug_to_table_name(slug) == f"schema{SEP}table"


class TestFindSlugByTableName:

    def test_finds_in_dict(self):
        candidates = {
            f"SrcA{SEP}t1": "a",
            f"SrcA{SEP}t2": "b",
        }
        assert find_slug_by_table_name("t2", candidates) == f"SrcA{SEP}t2"

    def test_finds_in_list(self):
        candidates = [f"SrcA{SEP}t1", f"SrcB{SEP}t2"]
        assert find_slug_by_table_name("t2", candidates) == f"SrcB{SEP}t2"

    def test_returns_none_when_absent(self):
        assert find_slug_by_table_name("missing", {f"S{SEP}x": 1}) is None

    def test_returns_first_match(self):
        candidates = {
            f"SrcOld{SEP}tbl": 1,
            f"SrcNew{SEP}tbl": 2,
        }
        result = find_slug_by_table_name("tbl", candidates)
        assert result in (f"SrcOld{SEP}tbl", f"SrcNew{SEP}tbl")


# -------------------------------------------------------------------
# content_to_text
# -------------------------------------------------------------------

class TestContentToText:

    @pytest.mark.parametrize("content, expected", [
        ("hello", "hello"),
        ("", ""),
        (["hello"], "hello"),
        (["hello", "world"], "hello\nworld"),
        ([], ""),
        (42, ""),
    ])
    def test_plain_and_list(self, content, expected):
        assert content_to_text(content) == expected

    def test_list_with_non_text_items(self):
        sentinel = object()
        result = content_to_text(["prompt", sentinel, "follow-up"])
        assert result == "prompt\nfollow-up"

    def test_list_with_only_non_text_items(self):
        assert content_to_text([object(), object()]) == ""

# -------------------------------------------------------------------
# set_content_text
# -------------------------------------------------------------------

class TestSetContentText:

    def test_plain_string(self):
        assert set_content_text("new", "old") == "new"

    def test_list_all_strings(self):
        # Only strings in list → collapse to plain string
        assert set_content_text("new", ["a", "b"]) == "new"

    def test_list_with_non_text_preserves_objects(self):
        img = object()
        result = set_content_text("updated", ["old text", img])
        assert result == ["updated", img]

    def test_list_with_multiple_non_text(self):
        img1, img2 = object(), object()
        result = set_content_text("txt", ["old", img1, "more", img2])
        assert result == ["txt", img1, img2]

    def test_empty_list(self):
        assert set_content_text("new", []) == "new"

    def test_roundtrip_with_content_to_text(self):
        img = object()
        original = ["hello world", img]
        text = content_to_text(original)
        restored = set_content_text(text + " extra", original)
        assert restored == ["hello world extra", img]
        assert content_to_text(restored) == "hello world extra"


# -------------------------------------------------------------------
# format_msg_content
# -------------------------------------------------------------------

class TestFormatMsgContent:

    def test_string_passthrough(self):
        assert format_msg_content("hello") == "hello"

    def test_list_of_strings(self):
        result = format_msg_content(["a", "b"])
        assert result == "a + b"

    def test_list_with_non_string_items(self):
        class FakeImage:
            pass
        img = FakeImage()
        result = format_msg_content(["prompt", img])
        assert "prompt" in result
        assert f"<FakeImage[{id(img)}]>" in result

    def test_non_string_non_list(self):
        obj = object()
        result = format_msg_content(obj)
        assert result == f"<object[{id(obj)}]>"

    def test_empty_list(self):
        assert format_msg_content([]) == ""


# -------------------------------------------------------------------
# serialize_image_content
# -------------------------------------------------------------------

class TestMakeImageContent:

    # Minimal valid magic bytes for each format
    _MAGIC_BYTES = {
        "photo.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,
        "photo.jpg": b"\xff\xd8\xff\xe0" + b"\x00" * 8,
        "photo.jpeg": b"\xff\xd8\xff\xe0" + b"\x00" * 8,
        "photo.gif": b"GIF89a" + b"\x00" * 8,
        "photo.webp": b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8,
    }

    @pytest.mark.parametrize("filename", [
        "photo.png", "photo.jpg", "photo.jpeg", "photo.gif", "photo.webp",
    ])
    def test_recognised_extensions(self, filename):
        data = self._MAGIC_BYTES[filename]
        result = serialize_image_content(filename, data)
        assert result is not None

    @pytest.mark.parametrize("filename", [
        "data.csv", "report.pdf", "archive.zip", "notes.txt",
    ])
    def test_non_image_returns_none(self, filename):
        assert serialize_image_content(filename, b"data") is None

    def test_explicit_mime_type_overrides_extension(self):
        # Use valid PNG magic bytes so instructor doesn't reject
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        result = serialize_image_content("file.bin", data, mime_type="image/png")
        assert result is not None

    def test_non_image_mime_type_returns_none(self):
        assert serialize_image_content("photo.png", b"data", mime_type="text/plain") is None


# -------------------------------------------------------------------
# IMAGE_MIME_TYPES
# -------------------------------------------------------------------

class TestImageMimeTypes:

    @pytest.mark.parametrize("ext, expected_prefix", [
        (".png", "image/"),
        (".jpg", "image/"),
        (".jpeg", "image/"),
        (".gif", "image/"),
        (".webp", "image/"),
        (".svg", "image/"),
        (".bmp", "image/"),
    ])
    def test_all_entries_are_image_types(self, ext, expected_prefix):
        assert IMAGE_MIME_TYPES[ext].startswith(expected_prefix)

    def test_non_image_extension_absent(self):
        assert ".csv" not in IMAGE_MIME_TYPES
        assert ".txt" not in IMAGE_MIME_TYPES


# -------------------------------------------------------------------
# fuse_messages — multimodal content
# -------------------------------------------------------------------

class TestFuseMessagesMultimodal:

    def test_plain_string_messages(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        result = fuse_messages(msgs, max_user_messages=2)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "first" in result[0]["content"]
        assert result[1] == msgs[-1]

    def test_multimodal_user_content_in_history(self):
        img = object()
        msgs = [
            {"role": "user", "content": ["describe this image", img]},
            {"role": "assistant", "content": "It shows a chart"},
            {"role": "user", "content": "thanks"},
        ]
        result = fuse_messages(msgs, max_user_messages=2)
        # History should contain the text portion of the multimodal message
        assert "describe this image" in result[0]["content"]
        # The last user message is kept as-is
        assert result[1]["content"] == "thanks"

    def test_multimodal_last_user_message_preserved(self):
        img = object()
        msgs = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": ["what is this?", img]},
        ]
        result = fuse_messages(msgs, max_user_messages=2)
        # Last user message should be preserved with the image object intact
        assert result[1]["content"] == ["what is this?", img]

    def test_empty_multimodal_content_skipped_in_history(self):
        """Multimodal list with only non-text items yields empty text -> skipped."""
        img = object()
        msgs = [
            {"role": "user", "content": [img]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "final"},
        ]
        result = fuse_messages(msgs, max_user_messages=2)
        assert result[0]["role"] == "system"
        # The image-only user message text is empty and skipped,
        # but the assistant reply is still in history
        assert "Assistant: ok" in result[0]["content"]

    def test_multiple_user_messages_all_multimodal(self):
        img1, img2, img3 = object(), object(), object()
        msgs = [
            {"role": "user", "content": ["first image", img1]},
            {"role": "assistant", "content": "got it"},
            {"role": "user", "content": ["second image", img2]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": ["third image", img3]},
        ]
        result = fuse_messages(msgs, max_user_messages=2)
        # Last user msg preserved as multimodal list
        assert result[-1]["content"] == ["third image", img3]
        # History includes second image text
        assert "second image" in result[0]["content"]

    def test_trailing_assistant_message_kept_in_history(self):
        """An assistant response generated after the last user turn (e.g. an
        answer awaiting validation) must remain in the history."""
        msgs = [
            {"role": "user", "content": "tell me about this dataset"},
            {"role": "assistant", "content": "here is a summary"},
            {"role": "user", "content": "what could be interesting to analyze?"},
            {"role": "assistant", "content": "you could cluster the cells"},
        ]
        result = fuse_messages(msgs, max_user_messages=2)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        # The freshly generated assistant answer is present in the history
        assert "you could cluster the cells" in result[0]["content"]
        # The last user message is still surfaced as the live query
        assert result[1] == msgs[2]


# -------------------------------------------------------------------
# mutate_user_message — multimodal content
# -------------------------------------------------------------------

class TestMutateUserMessageMultimodal:

    def test_suffix_plain_string(self):
        msgs = [{"role": "user", "content": "hello"}]
        mutate_user_message("extra", msgs, suffix=True, inplace=True)
        assert msgs[0]["content"] == "'hello' extra"

    def test_suffix_multimodal_preserves_images(self):
        img = object()
        msgs = [{"role": "user", "content": ["hello", img]}]
        mutate_user_message("extra", msgs, suffix=True, inplace=True)
        content = msgs[0]["content"]
        # Should be a list with updated text + original image
        assert isinstance(content, list)
        assert img in content
        assert any("hello" in item and "extra" in item for item in content if isinstance(item, str))

    def test_prefix_multimodal(self):
        img = object()
        msgs = [{"role": "user", "content": ["query", img]}]
        mutate_user_message("Context: ", msgs, suffix=False, inplace=True)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert img in content
        text = content_to_text(content)
        assert text.startswith("Context: ")

    def test_not_inplace_multimodal(self):
        img = object()
        original = [{"role": "user", "content": ["query", img]}]
        result = mutate_user_message("extra", original, suffix=True, inplace=False)
        # Original should be unchanged
        assert original[0]["content"] == ["query", img]
        # Result should have the mutation
        assert img in result[0]["content"]
