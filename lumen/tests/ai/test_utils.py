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
    FALLBACK_CHARS_PER_TOKEN, IMAGE_MIME_TYPES, LINT_SAMPLE_ROWS,
    UNRECOVERABLE_ERRORS, apply_changes, clean_sql, collapse_indexed_columns,
    content_to_text, count_tokens, describe_data, find_slug_by_table_name,
    format_msg_content, fuse_messages, get_schema, lint_data,
    mutate_user_message, parse_huggingface_url, render_template, report_error,
    retry_llm_output, serialize_image_content, set_content_text,
    slug_to_table_name, truncate_to_tokens,
)
from lumen.config import SOURCE_TABLE_SEPARATOR as SEP


def test_collapse_indexed_columns_collapses_large_series():
    """A dense numbered series (e.g. an embedding matrix) collapses to one entry."""
    names = ["obs_id"] + [f"X_pca_{i}" for i in range(100)]
    assert collapse_indexed_columns(names) == ["obs_id", "X_pca_0..X_pca_99 (100 cols)"]


def test_collapse_indexed_columns_leaves_unique_names():
    """Genuinely distinct columns are never collapsed."""
    names = ["gender", "age", "smoking_status", "tissue_site"]
    assert collapse_indexed_columns(names) == names


def test_collapse_indexed_columns_short_series_untouched():
    """Runs below the threshold stay expanded (e.g. tSNE/UMAP 2-D embeddings)."""
    names = ["X_tsne_0", "X_tsne_1", "obs_id"]
    assert collapse_indexed_columns(names) == names


def test_collapse_indexed_columns_preserves_position_and_interleaving():
    """Each series collapses at its first occurrence, even when interleaved."""
    names = []
    for i in range(10):
        names.extend([f"a_{i}", f"b_{i}"])
    result = collapse_indexed_columns(names)
    assert result == ["a_0..a_9 (10 cols)", "b_0..b_9 (10 cols)"]


def test_collapse_indexed_columns_near_complete_gap_named():
    """A near-complete run collapses and names its missing index."""
    names = [f"X_pca_{i}" for i in range(100) if i != 50]
    assert collapse_indexed_columns(names) == ["X_pca_0..X_pca_99 (99 cols, missing 50)"]


def test_collapse_indexed_columns_multiple_gaps_listed():
    """Several holes (within budget) are all named, in order."""
    names = [f"x_{i}" for i in range(100) if i not in (50, 73)]
    assert collapse_indexed_columns(names) == ["x_0..x_99 (98 cols, missing 50, 73)"]


def test_collapse_indexed_columns_too_many_gaps_expanded():
    """More holes than max_gaps: left expanded, since the gaps likely matter."""
    drop = {1, 3, 5, 7, 9, 11}  # 6 gaps > default max_gaps of 5
    names = [f"x_{i}" for i in range(20) if i not in drop]
    assert collapse_indexed_columns(names) == names


def test_collapse_indexed_columns_step_series_expanded():
    """A step-2 series is too gappy to be a run and stays expanded."""
    names = [f"x_{2 * i}" for i in range(10)]  # span 19, 9 holes
    assert collapse_indexed_columns(names) == names


def test_collapse_indexed_columns_nonzero_start():
    """Contiguous runs that don't start at zero collapse honestly."""
    names = [f"p_{i}" for i in range(5, 21)]  # p_5..p_20, 16 members
    assert collapse_indexed_columns(names) == ["p_5..p_20 (16 cols)"]


def test_collapse_indexed_columns_non_numeric_suffix_ignored():
    """Names whose suffix isn't a bare integer don't match the series pattern."""
    names = ["total_counts", "total_counts_mt", "n_genes_by_counts"]
    assert collapse_indexed_columns(names) == names


def test_collapse_indexed_columns_no_separator():
    """The classic PCA convention (no separator, e.g. PC1..PC50) collapses too."""
    names = [f"PC{i}" for i in range(1, 51)]
    assert collapse_indexed_columns(names) == ["PC1..PC50 (50 cols)"]


def test_collapse_indexed_columns_alternate_separators():
    """Hyphen- and dot-separated series are recognised, not just underscores."""
    hyphen = [f"dim-{i}" for i in range(8)]
    dot = [f"emb.{i}" for i in range(8)]
    assert collapse_indexed_columns(hyphen) == ["dim-0..dim-7 (8 cols)"]
    assert collapse_indexed_columns(dot) == ["emb.0..emb.7 (8 cols)"]


def test_render_template_with_valid_template():
    now = dt.datetime.now()
    expected = (
        "Do not excessively reason in responses; chain_of_thought fields for that, but should also be concise (1-2 sentences).\n"
        f"Current date time {now.strftime('%b %d, %Y %I:%M %p')}\n"
        "What topic of data?"
    )
    assert (
        render_template(PROMPTS_DIR / "_Testing" / "topic.jinja2", {"tools": ""}, current_datetime=now).strip()
        == expected
    )


def test_render_template_with_override():
    now = dt.datetime.now()
    expected = (
        "Do not excessively reason in responses; chain_of_thought fields for that, but should also be concise (1-2 sentences).\n"
        f"Current date time {now.strftime('%b %d, %Y %I:%M %p')}\n"
        "What topic of data?\n"
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

    async def test_describe_selects_relevant_columns(self):
        # Wide frame: many numerics, one low-cardinality categorical placed
        # last, and one near-unique id column. Relevance selection should
        # keep the categorical (despite its position) and drop the id.
        n = 200
        data = {f"num_{i}": np.arange(n) + i for i in range(15)}
        data["uid"] = [f"id_{i}" for i in range(n)]
        data["category"] = (["a", "b", "c"] * n)[:n]
        df = pd.DataFrame(data)
        result = yaml.load(await describe_data(df), yaml.SafeLoader)
        assert result["summary"]["sampled_cols"] is True
        assert result["summary"]["columns_shown"] == 16
        # Low-cardinality categorical kept even though it is the last column.
        assert "category" in result["stats"]
        # Near-unique id column dropped in favour of more informative columns.
        assert "uid" not in result["stats"]

    async def test_describe_priority_columns_forced_in(self):
        n = 200
        data = {f"num_{i}": np.arange(n) + i for i in range(15)}
        data["uid"] = [f"id_{i}" for i in range(n)]
        df = pd.DataFrame(data)
        result = yaml.load(
            await describe_data(df, priority_columns=["uid"]), yaml.SafeLoader
        )
        # Explicitly requested column is included despite low relevance.
        assert "uid" in result["stats"]

    async def test_describe_narrow_data_not_sampled(self):
        df = pd.DataFrame({f"col_{i}": np.arange(0, 200) for i in range(5)})
        result = yaml.load(await describe_data(df), yaml.SafeLoader)
        assert result["summary"]["sampled_cols"] is False
        assert "columns_shown" not in result["summary"]

    async def test_describe_categorical_dtype_emits_enum(self):
        # Sources that return low-cardinality columns as
        # pandas `category` must still surface their enum values.
        n = 300
        df = pd.DataFrame({
            "smoking": pd.Categorical((["Former", "Current", "Never"] * n)[:n]),
            "flag": pd.array(([True, False] * n)[:n], dtype="boolean"),
            "value": np.random.rand(n),
        })
        result = yaml.load(await describe_data(df), yaml.SafeLoader)
        assert set(result["stats"]["smoking"]["enum"]) <= {
            "Former", "Current", "Never", "...",
        }
        assert result["stats"]["smoking"]["nunique"] == 3
        assert "enum" in result["stats"]["flag"]


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


# -------------------------------------------------------------------
# count_tokens / truncate_to_tokens
# -------------------------------------------------------------------

class TestCountTokens:

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_counts_tokens(self):
        # Exact counts are tokenizer-specific; assert the shape of the answer
        # rather than a magic number, so a vocab change doesn't break the suite.
        assert count_tokens("hello world") >= 2
        assert count_tokens("hello world " * 100) > count_tokens("hello world")

    def test_falls_back_when_encoder_unavailable(self):
        """A missing tokenizer must degrade to a char estimate, not raise."""
        text = "a" * 300
        with patch("lumen.ai.utils._get_token_encoder", return_value=None):
            assert count_tokens(text) == pytest.approx(300 / FALLBACK_CHARS_PER_TOKEN, abs=1)

    def test_encoder_is_cached(self):
        from lumen.ai.utils import _get_token_encoder
        assert _get_token_encoder() is _get_token_encoder()

    def test_encoder_failure_is_soft(self):
        """A tokenizer that fails to load yields None rather than propagating."""
        from lumen.ai import utils
        with patch.dict(utils._TOKEN_ENCODER_CACHE, clear=True):
            with patch.dict("sys.modules", {"tiktoken": None}):
                # `import tiktoken` raises ImportError when the module is None.
                assert utils._get_token_encoder() is None
                assert utils.count_tokens("some text") > 0


class TestTruncateToTokens:

    def test_under_budget_is_unchanged(self):
        text = "line one\nline two"
        assert truncate_to_tokens(text, 1000) == text

    def test_over_budget_respects_cap(self):
        text = "\n".join(f"row {i} has a value of {i * 3.14159}" for i in range(300))
        result = truncate_to_tokens(text, 100)
        assert count_tokens(result) <= 100
        assert len(result) < len(text)

    def test_reports_what_was_dropped(self):
        text = "\n".join(f"row {i}" for i in range(500))
        result = truncate_to_tokens(text, 60)
        assert "truncated, showing" in result
        assert "tokens)" in result

    def test_cuts_on_a_line_boundary(self):
        """The last retained line must be whole, so YAML/tables stay parseable."""
        lines = [f"key_{i}: value_{i}" for i in range(300)]
        result = truncate_to_tokens("\n".join(lines), 100)
        body = result.split("\n")[:-1]  # drop the appended note
        assert body
        assert all(line in lines for line in body)

    def test_single_oversized_line(self):
        """One line longer than the whole budget still respects the cap."""
        result = truncate_to_tokens("x" * 5000, 50)
        assert count_tokens(result) <= 50

    def test_custom_marker(self):
        text = "\n".join(f"row {i}" for i in range(500))
        assert "elided, showing" in truncate_to_tokens(text, 60, marker="elided")


class TestLintData:
    """Deterministic data-quality findings used to drive SQLAgent's cleaning pass."""

    def test_clean_frame_reports_nothing(self):
        df = pd.DataFrame({
            "id": range(20),
            "category": ["a", "b"] * 10,
            "value": [float(i) for i in range(20)],
        })
        assert lint_data(df) == []

    def test_empty_frame_reports_nothing(self):
        assert lint_data(pd.DataFrame({"a": [], "b": []})) == []

    def test_nulls_reported_with_percentage(self):
        df = pd.DataFrame({"value": [1.0] * 90 + [None] * 10, "other": range(100)})
        findings = lint_data(df)
        assert any('Missing values' in f and '"value" (10.0% null)' in f for f in findings)

    def test_nulls_below_threshold_ignored(self):
        """One missing value in a thousand is not worth an extra LLM round trip."""
        df = pd.DataFrame({"value": [1.0] * 999 + [None]})
        assert not any("Missing values" in f for f in lint_data(df))

    def test_duplicate_rows_reported(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
        findings = lint_data(df)
        assert any("1 of 4 rows are exact duplicates" in f for f in findings)

    def test_sentinel_numbers_reported(self):
        df = pd.DataFrame({"temp": [12.5, 13.0, -9999.0, 14.0, -9999.0, 15.0]})
        findings = lint_data(df)
        assert any("Placeholder numbers" in f and '"temp" (2 rows)' in f for f in findings)

    def test_untrimmed_and_empty_text_reported(self):
        df = pd.DataFrame({"name": ["alice", " bob ", "", "dave"]})
        findings = lint_data(df)
        assert any("Untrimmed or empty text" in f and "1 padded, 1 empty" in f for f in findings)

    def test_numbers_stored_as_text_reported(self):
        df = pd.DataFrame({"amount": ["1.5", "2.5", "3.5", "4.5"]})
        findings = lint_data(df)
        assert any("Numbers stored as text" in f and '"amount"' in f for f in findings)

    def test_genuine_text_not_flagged_as_numeric(self):
        df = pd.DataFrame({"name": ["alice", "bob", "carol", "dave"]})
        assert not any("Numbers stored as text" in f for f in lint_data(df))

    def test_constant_column_reported(self):
        df = pd.DataFrame({"region": ["north"] * 10, "value": range(10)})
        findings = lint_data(df)
        assert any("Constant column" in f and '"region"' in f for f in findings)

    def test_single_row_frame_reports_no_constant_columns(self):
        """Every column of a one-row result is trivially constant."""
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        assert not any("Constant column" in f for f in lint_data(df))

    def test_iqr_outliers_reported(self):
        df = pd.DataFrame({"revenue": [float(i) for i in range(50)] + [100000.0]})
        findings = lint_data(df)
        assert any("IQR outliers" in f and '"revenue" (1 outside' in f for f in findings)

    def test_degenerate_iqr_column_reports_no_outliers(self):
        """A flag-shaped column has a zero IQR, where Tukey fences flag every
        minority value; skipping it is deliberate, not an oversight."""
        df = pd.DataFrame({"is_active": [0.0] * 50 + [1.0]})
        assert not any("IQR outliers" in f for f in lint_data(df))

    def test_uniform_numeric_column_has_no_outliers(self):
        df = pd.DataFrame({"value": [float(i) for i in range(100)]})
        assert not any("IQR outliers" in f for f in lint_data(df))

    def test_ordinary_random_data_reports_nothing(self):
        """The property the cleaning pass depends on: clean data costs no LLM call.
        At the usual 1.5 IQR fence a gaussian column always reports outliers."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "measure": rng.normal(size=5000),
            "count": rng.integers(0, 100, size=5000),
            "label": rng.choice(["alpha", "beta", "gamma"], size=5000),
        })
        assert lint_data(df) == []

    def test_large_frame_notes_that_counts_are_sampled(self):
        df = pd.DataFrame({
            "value": [1.0] * (LINT_SAMPLE_ROWS * 2),
            "flag": [None] * (LINT_SAMPLE_ROWS * 2),
        })
        findings = lint_data(df)
        assert findings
        assert findings[-1].startswith(f"Counts above come from a random {LINT_SAMPLE_ROWS}-row sample")

    def test_unhashable_values_do_not_raise(self):
        """A geometry or list column must not break a query that already succeeded."""
        df = pd.DataFrame({"geom": [[1, 2], [3, 4], [1, 2]], "value": [1.0, 2.0, 3.0]})
        assert isinstance(lint_data(df), list)

    def test_findings_cap_the_columns_they_name(self):
        df = pd.DataFrame({f"c{i}": ["  padded  "] * 5 for i in range(20)})
        finding = next(f for f in lint_data(df) if "Untrimmed" in f)
        assert "and 12 more" in finding
