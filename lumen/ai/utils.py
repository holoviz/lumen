from __future__ import annotations

import asyncio
import base64
import difflib
import functools
import hashlib
import html
import inspect
import json
import math
import re
import textwrap
import time
import traceback

from collections.abc import Callable, Generator, Iterator
from functools import reduce, wraps
from operator import getitem
from pathlib import Path
from shutil import get_terminal_size
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs

import numpy as np
import pandas as pd
import param
import sqlglot
import yaml

from instructor import Image
from jinja2 import (
    ChoiceLoader, DictLoader, Environment, FileSystemLoader, StrictUndefined,
    nodes,
)
from jinja2.visitor import NodeVisitor
from jsonschema import ValidationError
from markupsafe import escape
from panel_material_ui import Details

from ..config import dump_yaml
from ..filters.base import ConstantFilter
from ..pipeline import Pipeline
from ..sources.base import Source
from ..sources.xarray_sql import XArraySQLSource
from ..transforms import SQLRemoveSourceSeparator
from ..util import log, try_import_xarray
from .config import (
    PROMPTS_DIR, SOURCE_TABLE_SEPARATOR, UNRECOVERABLE_ERRORS, VEGA_MAP_LAYER,
    VEGA_ZOOMABLE_MAP_ITEMS, MissingContextError, RetriesExceededError,
)

if TYPE_CHECKING:
    from panel.chat.step import ChatStep

    from .editors import VegaLiteEditor
    from .models import (
        DeleteLine, InsertLine, LineEdit, ReplaceLine,
    )


IMAGE_MIME_TYPES = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.svg': 'image/svg+xml',
    '.bmp': 'image/bmp',
}

# Column-selection tuning for describe_data_sync.
DEFAULT_MAX_SUMMARY_COLS = 16
# Columns with at most this many distinct values are treated as
# human-meaningful "enum"/categorical columns and prioritised.
LOW_CARDINALITY_MAX = 10


def deterministic_hash(text: str) -> int:
    """Stable hash using MD5, consistent across Python sessions."""
    return int.from_bytes(
        hashlib.md5(text.encode("utf-8")).digest()[:4], byteorder="big"
    )


def format_float(num):
    """
    Process a float value, returning numeric types instead of strings.
    For very large/small numbers, returns a float that will display in scientific notation.
    """
    if not isinstance(num, (int, float)):
        return num
    if pd.isna(num) or math.isinf(num):
        return num

    # For integers, return as int
    if num == int(num):
        # But if it's a very large integer, keep as float for potential sci notation
        if abs(num) >= 1e5:
            return f"{num:.1e}"  # Format as scientific notation
        return int(num)

    # For normal range numbers, round but keep as float
    elif 0.01 <= abs(num) < 1e6:
        return round(num, 1)

    # For very small or very large numbers, return as-is
    # These will naturally display in scientific notation when converted to string
    else:
        return f"{num:.1e}"  # Format as scientific notation


def fuse_messages(messages: list[dict], max_user_messages: int = 2) -> list[dict]:
    """
    Fuses the chat history into a single system message, followed by the last user message.
    This function is reusable across different components that need to combine
    multiple messages into a simplified format for LLM processing.

    Parameters
    ----------
    messages : list[dict]
        List of message dictionaries with 'role' and 'content' keys
    max_user_messages : int
        Maximum number of user messages to include in the history

    Returns
    -------
    list[dict]
        Processed messages with the chat history as a system message
    """
    user_indices = [i for i, msg in enumerate(messages) if msg['role'] == 'user']
    if user_indices:
        first_user_index = user_indices[0] if len(user_indices) <= max_user_messages else user_indices[-max_user_messages]
        last_user_index = user_indices[-1]
        last_user_message = messages[last_user_index]
    else:
        first_user_index = 0
        last_user_index = -1
    history_messages = messages[first_user_index:last_user_index]
    if not history_messages:
        return [last_user_message] if user_indices else []
    formatted_history = "\n\n".join(
        f"{msg['role'].capitalize()}: {content_to_text(msg['content'])}"
        for msg in history_messages
        if content_to_text(msg['content']).strip()
    )
    system_prompt = {
        "role": "system",
        "content": f"<Chat History>\n{formatted_history}\n<\\Chat History>"
    }
    return [system_prompt] if last_user_index == -1 else [system_prompt, last_user_message]


def get_template_loader(
    template_path: Path | str, relative_to: Path = PROMPTS_DIR
) -> tuple[FileSystemLoader, str]:
    if isinstance(template_path, str):
        template_path = Path(template_path)

    template_name = None

    # Create a list of search paths, starting with the default
    search_paths = [relative_to]
    # If template_path is absolute, add its parent directory to search paths
    if template_path.is_absolute():
        # Add the parent directory of the template to the search paths
        search_paths.append(template_path.parent)
        template_name = template_path.name
    else:
        try:
            # Try to make it relative to the default template directory
            template_name = template_path.relative_to(relative_to).as_posix()
        except ValueError:
            # If that fails, treat it as a plain file path
            # Add parent dir to search paths and use filename as template_name
            if template_path.exists():
                search_paths.append(template_path.parent)
                template_name = template_path.name
            else:
                # Final fallback - assume it's a relative path but not to PROMPTS_DIR
                template_name = str(template_path)
    return FileSystemLoader(search_paths), template_name


def json_to_yaml(data):
    """
    Convert JSON string or Python dict/list to YAML format.

    Parameters
    ----------
    data : str | dict | list
        JSON string or Python data structure to convert to YAML

    Returns
    -------
    str
        YAML formatted string, or original data if conversion fails
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return data


def render_template(template_path: Path | str, overrides: dict | None = None, relative_to: Path = PROMPTS_DIR, **context):
    fs_loader, template_name = get_template_loader(template_path, relative_to)
    if overrides:
        # Dynamically create block definitions based on dictionary keys with proper escaping
        block_definitions = "\n".join(
            f"{{% block {escape(key)} %}}{escape(value)}{{% endblock %}}"
            for key, value in overrides.items()
        )
        # Create the dynamic template content by extending the base template and adding blocks
        dynamic_template_content = dedent(
            f"""
            {{% extends "{template_name}" %}}
            {block_definitions}
            """
        )
        dynamic_loader = DictLoader({"dynamic_template": dynamic_template_content})
        choice_loader = ChoiceLoader([dynamic_loader, fs_loader])
        env = Environment(loader=choice_loader, undefined=StrictUndefined)
        template_name = "dynamic_template"
    else:
        env = Environment(loader=fs_loader, undefined=StrictUndefined)

    env.globals["dedent"] = lambda text: textwrap.dedent(text).strip()
    env.filters["json_to_yaml"] = json_to_yaml
    template = env.get_template(template_name)
    return template.render(**context)


class BlockNameCollector(NodeVisitor):
    def __init__(self):
        self.blocks = []

    def visit_Block(self, node: nodes.Block):
        self.blocks.append(node.name)
        # keep traversing in case of nested blocks
        self.generic_visit(node)


_BLOCK_RE = re.compile(
    r"{%-?\s*block\s+(?P<name>\w+)\s*-?%}(?P<body>.*?){%-?\s*endblock(?:\s+(?P=name))?\s*-?%}",
    re.DOTALL,
)

def get_block_names(template_path: Path | str, relative_to: Path = PROMPTS_DIR):
    env = Environment()
    parsed = env.parse(Path(template_path).read_text())
    collector = BlockNameCollector()
    collector.visit(parsed)
    return list(collector.blocks)


def extract_block_source(template_path: Path | str, block_name: str, relative_to: Path = PROMPTS_DIR):
    src = Path(template_path).read_text(encoding='utf-8')
    for m in _BLOCK_RE.finditer(src):
        if m.group("name") == block_name:
            return m.group("body").strip()
    raise KeyError(f"Block '{block_name}' not found in {template_path!r}.")


def warn_on_unused_variables(string, kwargs, prompt_label):
    used_keys = set()

    for key in kwargs:
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, string):
            used_keys.add(key)

    unused_keys = set(kwargs.keys()) - used_keys
    if unused_keys:
        # TODO: reword this concisely... what do you call those variables for formatting?
        log.warning(
            f"The prompt template, {prompt_label}, is missing keys, "
            f"which could mean the LLM is lacking the context provided "
            f"from these variables: {unused_keys}. If this is unintended, "
            f"please create a template that contains those keys."
        )


def get_root_exception(e: Exception, depth: int = 5, exceptions: tuple[Exception] | None = None) -> Exception | None:
    """
    Recursively get the root cause of an exception up to a specified depth.

    Parameters
    ----------
    e : Exception
        The exception to analyze
    depth : int
        Maximum recursion depth to avoid infinite loops
    exceptions : list[Exception]
        List of exception types to consider as root causes. If None, all exceptions are considered.

    Returns
    -------
    Exception | None
        The root cause exception if found, otherwise None
    """
    for _ in range(depth):
        if exceptions and isinstance(e, tuple(exceptions)):
            return e
        if e.__cause__ is not None:
            if isinstance(e.__cause__, ValidationError):
                return e
            e = e.__cause__
    return e


def retry_llm_output(retries=3, sleep=1):
    """
    Retry a function that returns a response from the LLM API.
    If the function raises an exception, retry it up to `retries` times.
    If `sleep` is provided, wait that many seconds between retries.
    If an error occurs, pass the error message into kwargs for the function
    to manually handle by including it.
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                errors = []
                for i in range(retries):
                    if i > 0:
                        log_debug(f"Retrying LLM function {func.__name__} ({i} / {retries})...")
                    if errors:
                        kwargs["errors"] = errors
                    try:
                        output = await func(*args, **kwargs)
                        if not output:
                            raise Exception("No valid output from LLM.")
                        return output
                    except Exception as e:
                        e = get_root_exception(e, exceptions=UNRECOVERABLE_ERRORS)
                        if isinstance(e, UNRECOVERABLE_ERRORS):
                            if not isinstance(e, MissingContextError):
                                log_debug(f"LLM encountered unrecoverable error: {e}")
                            raise e
                        if i == retries - 1:
                            raise RetriesExceededError("Maximum number of retries exceeded.") from e
                        error_str = str(e)
                        if error_str not in errors:
                            errors.append(error_str)
                        else:
                            errors.append(
                                f"{error_str}\n"
                                "(Same error repeated — try a fundamentally different approach)."
                            )
                        traceback.print_exc()
                        if sleep:
                            await asyncio.sleep(sleep)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                errors = []
                for i in range(retries):
                    if i > 0:
                        log_debug(f"Retrying LLM function {func.__name__} ({i} / {retries})...")
                    if errors:
                        kwargs["errors"] = errors
                    try:
                        output = func(*args, **kwargs)
                        if not output:
                            raise Exception("No valid output from LLM.")
                        return output
                    except Exception as e:
                        e = get_root_exception(e, exceptions=UNRECOVERABLE_ERRORS)
                        if isinstance(e, UNRECOVERABLE_ERRORS):
                            if not isinstance(e, MissingContextError):
                                log_debug(f"LLM encountered unrecoverable error: {e}")
                            raise e
                        if i == retries - 1:
                            raise RetriesExceededError("Maximum number of retries exceeded.") from e
                        error_str = str(e)
                        if error_str not in errors:
                            errors.append(error_str)
                        else:
                            errors.append(
                                f"{error_str}\n"
                                "(Same error repeated — try a fundamentally different approach)."
                            )
                        traceback.print_exc()
                        if sleep:
                            time.sleep(sleep)

            return sync_wrapper

    return decorator


def process_enums(spec, num_cols, limit=None, include_enum=True, reduce_enums=True, include_empty_fields=False):
    """
    Process enum values in a schema field specification.

    Parameters
    ----------
    spec : dict
        The specification dictionary for this field
    num_cols : int
        The total number of columns in the schema (used for enum reduction calculation)
    limit : int, optional
        The limit used in schema generation
    include_enum : bool, default True
        Whether to include enum values
    reduce_enums : bool, default True
        Whether to reduce the number of enum values for readability
    include_empty_fields : bool, default False
        Whether to include fields that are empty

    Returns
    -------
    tuple[dict, bool]
        The updated spec dictionary and a boolean indicating if this field is empty
    """
    is_empty = False

    if "enum" not in spec:
        return spec, is_empty

    # Scale max_enums based on average enum length (inverse relationship)
    # Short enums (categories) = show more, long enums (text) = show fewer
    if reduce_enums:
        enum_values = spec["enum"]
        # Calculate average length, excluding None values
        non_none_values = [v for v in enum_values if v is not None]
        if non_none_values:
            avg_length = sum(len(str(v)) for v in non_none_values) / len(non_none_values)
        else:
            avg_length = 0

        # Exponential decay: short enums get more slots, long enums get fewer
        # Formula: base_max = max(2, 10 * e^(-0.025 * avg_length))
        # 20: Maximum base enums (for very short values)
        # 0.025: Decay rate (higher = faster drop-off)
        # max(2, ...): Minimum floor (always show at least 2)
        base_max_enums = max(2, int(10 * math.exp(-0.04 * avg_length)))

        # Still consider number of columns but with less weight
        max_enums = max(2, int(base_max_enums * math.exp(-0.05 * max(0, num_cols - 10))))
    else:
        max_enums = 2
    truncate_limit = min(limit or 2, max_enums)

    if not include_enum or len(spec["enum"]) == 0:
        spec.pop("enum")
    elif len(spec["enum"]) > max_enums:
        # if there are only 10 enums, fine; let's keep them all
        # else, that assumes this column is like glasbey 100+ categories vs category10
        # so truncate to 5 and add "..."
        spec["enum"] = spec["enum"][:truncate_limit] + ["..."]
    elif limit and len(spec["enum"]) == 1 and spec["enum"][0] is None:
        spec["enum"] = [f"(unknown; truncated to {limit} rows)"]
    elif not include_empty_fields and len(spec["enum"]) == 1 and (
        spec["enum"][0] is None or (isinstance(spec["enum"][0], str) and spec["enum"][0].strip() == "")
    ):
        is_empty = True

    # truncate each enum to 100 characters if they exist
    if "enum" in spec:
        spec["enum"] = [
            enum if enum is None or not isinstance(enum, str) or len(enum) < 100 else f"{enum[:100]} ..."
            for enum in spec["enum"]
        ]

    return spec, is_empty


async def get_schema(
    source: Source | Pipeline,
    table: str | None = None,
    include_type: bool = True,
    include_min_max: bool = True,
    include_enum: bool = True,
    include_count: bool = False,
    include_empty_fields: bool = False,
    include_nans: bool = False,
    reduce_enums: bool = True,
    shuffle: bool = True,
    **get_kwargs
):
    if isinstance(source, Pipeline):
        schema = await asyncio.to_thread(source.get_schema)
    else:
        if "limit" not in get_kwargs:
            get_kwargs["limit"] = 100
        schema = await asyncio.to_thread(source.get_schema, table, shuffle=shuffle, **get_kwargs)
    schema = dict(schema)

    # first pop regardless to prevent
    # argument of type 'numpy.int64' is not iterable
    count = schema.pop("__len__", None)

    if not include_type:
        for spec in schema.values():
            if "type" in spec:
                spec.pop("type")

    empty_fields = []
    if include_min_max:
        for field, spec in schema.items():
            min_val = 0
            max_val = 0
            if "inclusiveMinimum" in spec:
                min_val = spec.pop("inclusiveMinimum")
                if isinstance(min_val, (int, float)):
                    min_val = format_float(min_val)
                if include_nans or not pd.isna(min_val):
                    spec["min"] = min_val
            if "inclusiveMaximum" in spec:
                max_val = spec.pop("inclusiveMaximum")
                if isinstance(max_val, (int, float)):
                    max_val = format_float(max_val)
                if include_nans or not pd.isna(max_val):
                    spec["max"] = max_val
            if not include_empty_fields and pd.isna(min_val) and pd.isna(max_val):
                empty_fields.append(field)
                continue
    else:
        for field, spec in schema.items():
            min_val = 0
            max_val = 0
            if "inclusiveMinimum" in spec:
                min_val = spec.pop("inclusiveMinimum")
            if "inclusiveMaximum" in spec:
                max_val = spec.pop("inclusiveMaximum")
            if "min" in spec:
                min_val = spec.pop("min")
            if "max" in spec:
                max_val = spec.pop("max")
            if not include_empty_fields and pd.isna(min_val) and pd.isna(max_val):
                empty_fields.append(field)
                continue

    num_cols = len(schema)
    for field, spec in schema.items():
        if "type" in spec:
            if not include_empty_fields and spec["type"] is None:
                spec.pop("type")
                continue
            if spec["type"] == "string":
                spec["type"] = "str"
            elif spec["type"] == "integer":
                spec["type"] = "int"
            elif spec["type"] == "number":
                spec["type"] = "num"
            elif spec["type"] == "boolean":
                spec["type"] = "bool"

            # If it's obvious that this is a numeric range, remove type
            # Yes `{'min': 2017, 'max': 2020}`  # implicitly infer
            # Not `{'type': 'num', 'min': '2.0e+06', 'max': '6.2e+06'}`  # explicitly show type
            if isinstance(spec.get("min"), (int, float)) and isinstance(spec.get("max"), (int, float)):
                spec.pop("type")

        # Process enums using the extracted function
        if "enum" in spec:
            spec, is_empty = process_enums(
                spec, num_cols,
                limit=get_kwargs.get("limit"),
                include_enum=include_enum,
                reduce_enums=reduce_enums,
                include_empty_fields=include_empty_fields
            )
            if is_empty:
                empty_fields.append(field)
                continue

    # Format any other numeric values in the schema
    for spec in schema.values():
        for key, value in spec.items():
            if isinstance(value, (int, float)) and key not in ['type']:
                spec[key] = format_float(value)

    # remove any fields that are empty
    for field in empty_fields:
        schema[field] = "<null>"

    if count is not None and include_count:
        schema["__len__"] = count
    return schema


async def get_pipeline(**kwargs):
    """
    A wrapper be able to use asyncio.to_thread and not
    block the main thread when calling Pipeline
    """
    def get_pipeline_sync():
        return Pipeline(**kwargs)
    return await asyncio.to_thread(get_pipeline_sync)


async def get_data(pipeline):
    """
    A wrapper be able to use asyncio.to_thread and not
    block the main thread when calling pipeline.data
    """
    def get_data_sync():
        return pipeline.data
    return await asyncio.to_thread(get_data_sync)


def _score_column_relevance(series: pd.Series, n_rows: int) -> float:
    """
    Cheap, dataset-agnostic relevance score used to pick which columns
    to include in a summary. Higher is more informative.

    The ordering favours low-cardinality categoricals (the columns a
    model most often filters/groups on) and numerics with actual spread,
    while penalising constant columns and near-unique id/free-text
    columns. It deliberately relies only on dtype and cardinality so it
    generalises across datasets rather than matching specific names.

    Parameters
    ----------
    series : pd.Series
        The (already row-sampled) column to score.
    n_rows : int
        Number of rows in the sample, used for the cardinality ratio.

    Returns
    -------
    float
        Relevance score; larger values are kept first.
    """
    try:
        nunique = int(series.nunique(dropna=True))
    except TypeError:
        # Unhashable values (e.g. lists/dicts) — treat as uninformative.
        return 0.5

    if nunique <= 1:
        # Constant (or all-null) column carries no signal.
        return 0.0

    ratio = nunique / n_rows if n_rows else 0.0

    # Low-cardinality categorical/enum (incl. bool and small int codes):
    # cheap to summarise and usually the most query-relevant. nunique is
    # already capped, so no additional ratio gate is needed here — on small
    # frames a low-card column can have a high ratio (e.g. 8/12) and should
    # still be surfaced.
    if nunique <= LOW_CARDINALITY_MAX:
        return 3.0

    # Continuous numerics: informative ranges/distributions.
    if pd.api.types.is_numeric_dtype(series):
        return 2.0

    # Near-unique non-numeric column: likely an id or free text.
    if ratio > 0.9:
        return 0.5

    # Moderate-cardinality text.
    return 1.5


def _select_relevant_columns(
    df: pd.DataFrame,
    max_cols: int,
    priority_columns: list[str] | None = None,
) -> tuple[list[str], bool]:
    """
    Choose up to *max_cols* columns to summarise, ordered by relevance.

    Any caller-supplied *priority_columns* (e.g. columns referenced in
    the query or in active exploration filters) are always kept and
    placed first; the remaining budget is filled by intrinsic relevance
    score. Columns are returned relevance-first so that if the rendered
    summary is later truncated to a character/token budget, the most
    useful columns survive.

    Parameters
    ----------
    df : pd.DataFrame
        The (already row-sampled) frame to select columns from.
    max_cols : int
        Soft cap on the number of columns to keep. Priority columns are
        always kept even if they exceed this.
    priority_columns : list[str] | None
        Columns to force-include, in order of importance.

    Returns
    -------
    tuple[list[str], bool]
        The selected column names and whether any columns were dropped.
    """
    columns = list(df.columns)
    if len(columns) <= max_cols and not priority_columns:
        return columns, False

    priority = [c for c in (priority_columns or []) if c in df.columns]
    priority_set = set(priority)

    n_rows = len(df)
    scored = [
        (_score_column_relevance(df[col], n_rows), idx, col)
        for idx, col in enumerate(columns)
        if col not in priority_set
    ]
    # Highest score first; ties fall back to original column order.
    scored.sort(key=lambda t: (-t[0], t[1]))

    remaining = max(0, max_cols - len(priority))
    selected = priority + [col for _, _, col in scored[:remaining]]
    return selected, len(selected) < len(columns)


def describe_data_sync(
    df: pd.DataFrame,
    enum_limit: int = 3,
    reduce_enums: bool = True,
    row_limit: int | None = None,
    max_cols: int = DEFAULT_MAX_SUMMARY_COLS,
    priority_columns: list[str] | None = None,
) -> str:
    """
    Synchronous version of describe_data that generates a YAML summary of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to describe
    enum_limit : int
        Maximum number of enum values to show per column
    reduce_enums : bool
        Whether to reduce enum values for readability
    row_limit : int | None
        The row cap applied to the result (e.g. by SQLLimit). When the
        returned frame reaches this many rows the result was truncated, so
        the summary flags it instead of presenting the capped count as the
        table's true size.
    max_cols : int
        Soft cap on how many columns to include. When the frame is wider,
        columns are selected by relevance (see _select_relevant_columns)
        rather than by position, so informative fields aren't dropped just
        for appearing late in the table.
    priority_columns : list[str] | None
        Columns to always include (e.g. columns referenced in the query
        or active filters), placed first.

    Returns
    -------
    str
        YAML-formatted summary of the DataFrame
    """
    size = df.size
    shape = df.shape
    shape_header = {"data_shape": [int(shape[0]), int(shape[1])], "is_sampled": False}
    if shape[0] == 1 or size < 10 or (shape[1] > 8 and size < 100):
        records = df.to_dict(orient='records')
        result = {**shape_header, "records": records}
        return yaml.dump(result, default_flow_style=False, allow_unicode=True, sort_keys=False)
    if size < 100:
        header = yaml.dump(shape_header, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return header + df.to_markdown(index=False)

    is_sampled = False
    if shape[0] > 5000:
        is_sampled = True
        df = df.sample(5000)

    df = df.sort_index()

    # Select the most informative columns before the (more expensive)
    # per-column stats below, so wide tables don't crowd out or truncate
    # the columns a model actually needs.
    selected_columns, sampled_columns = _select_relevant_columns(
        df, max_cols=max_cols, priority_columns=priority_columns
    )
    if sampled_columns:
        df = df[selected_columns]

    for col in df.columns:
        if isinstance(df[col].iloc[0], pd.Timestamp):
            df[col] = pd.to_datetime(df[col])

    describe_df = df.describe(percentiles=[])
    columns_to_drop = ["min", "max", "count", "top", "freq"] # present if any numeric or object
    columns_to_drop = [col for col in columns_to_drop if col in describe_df.columns]
    df_describe_dict = describe_df.drop(columns=columns_to_drop).to_dict()

    for col in df.select_dtypes(include=["object", "string", "category", "boolean", "bool"]).columns:
        if col not in df_describe_dict:
            df_describe_dict[col] = {}
        try:
            # Get unique values for enum processing
            unique_values = df[col].dropna().unique().tolist()
            if unique_values:
                temp_spec = {"enum": unique_values}
                updated_spec, _ = process_enums(
                    temp_spec,
                    num_cols=len(df.columns),
                    limit=enum_limit,
                    include_enum=True,
                    reduce_enums=reduce_enums,
                )
                if "enum" in updated_spec:
                    df_describe_dict[col]["enum"] = updated_spec["enum"]

            df_describe_dict[col]["nunique"] = df[col].nunique()
        except Exception:
            df_describe_dict[col]["nunique"] = 'unknown'
        try:
            max_length = df[col].str.len().max()
            if pd.notna(max_length):
                df_describe_dict[col]["max_length"] = int(max_length)
        except AttributeError:
            pass

    # Low-cardinality integer columns (e.g. status codes, ratings)
    for col in df.select_dtypes(include=["int64"]).columns:
        nunique = df[col].nunique()
        if nunique <= 10:
            if col not in df_describe_dict:
                df_describe_dict[col] = {}
            unique_values = sorted(df[col].dropna().unique().tolist())
            temp_spec = {"enum": unique_values}
            updated_spec, _ = process_enums(
                temp_spec,
                num_cols=len(df.columns),
                limit=enum_limit,
                include_enum=True,
                reduce_enums=reduce_enums,
            )
            if "enum" in updated_spec:
                df_describe_dict[col]["enum"] = updated_spec["enum"]
            df_describe_dict[col]["nunique"] = int(nunique)

    for col in df.columns:
        if col not in df_describe_dict:
            df_describe_dict[col] = {}
        nulls = int(df[col].isnull().sum())
        if nulls > 0:
            df_describe_dict[col]["nulls"] = nulls

    # select datetime64 columns
    for col in df.select_dtypes(include=["datetime64"]).columns:
        for key in df_describe_dict[col]:
            df_describe_dict[col][key] = str(df_describe_dict[col][key])
        df[col] = df[col].astype(str)  # shorten output

    # select all numeric columns and round
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        for key in df_describe_dict[col]:
            values = df_describe_dict[col][key]
            if isinstance(values, list):
                values = [format_float(v) for v in values]
            else:
                values = format_float(values)
            df_describe_dict[col][key] = values

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].apply(format_float)

    # Emit stats in the (relevance-first) column order so that if the
    # rendered summary is later truncated to a character/token budget by
    # a caller, the most informative columns are the ones that survive.
    ordered_describe = {c: df_describe_dict[c] for c in df.columns if c in df_describe_dict}
    for key, value in df_describe_dict.items():
        if key not in ordered_describe:
            ordered_describe[key] = value
    df_describe_dict = ordered_describe

    n_rows, n_cols = shape

    # Add head and tail samples (2 rows each); deduplicate if identical
    head_sample = df.head(2).to_dict('records')
    tail_sample = df.tail(2).to_dict('records')

    summary = {
        "n_cells": size,
        "data_shape": [n_rows, n_cols],
        "sampled_cols": sampled_columns,
        "is_sampled": is_sampled,
    }
    notes = []
    if row_limit is not None and n_rows >= row_limit:
        summary["rows_capped_at_limit"] = row_limit
        notes.append(
            f"Result capped at the {row_limit}-row display limit; "
            "the full table may contain more rows."
        )
    if sampled_columns:
        summary["columns_shown"] = len(selected_columns)
        notes.append(
            f"Showing {len(selected_columns)} of {n_cols} columns, chosen by "
            "relevance (query/filter columns and low-cardinality fields first).")
    if notes:
        summary["note"] = " ".join(notes)
    result = {
        "summary": summary,
        "stats": df_describe_dict,
    }
    if head_sample == tail_sample:
        result["sample"] = head_sample[0] if head_sample else {}
    else:
        result["head"] = head_sample[0] if head_sample else {}
        result["tail"] = tail_sample[0] if tail_sample else {}

    return yaml.dump(result, default_flow_style=False, allow_unicode=True, sort_keys=False)


async def describe_data(
    df: pd.DataFrame,
    enum_limit: int = 3,
    reduce_enums: bool = True,
    row_limit: int | None = None,
    max_cols: int = DEFAULT_MAX_SUMMARY_COLS,
    priority_columns: list[str] | None = None,
) -> str:
    """
    Async wrapper for describe_data_sync that generates a YAML summary of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to describe
    enum_limit : int
        Maximum number of enum values to show per column
    reduce_enums : bool
        Whether to reduce enum values for readability
    row_limit : int | None
        The row cap applied to the result; used to flag truncation.
    max_cols : int
        Soft cap on how many columns to include in the summary. When the
        frame is wider, columns are chosen by relevance rather than
        position.
    priority_columns : list[str] | None
        Columns to always include (e.g. columns referenced in the query
        or active filters), placed first.

    Returns
    -------
    str
        YAML-formatted summary of the DataFrame
    """
    return await asyncio.to_thread(
        describe_data_sync, df, enum_limit, reduce_enums, row_limit,
        max_cols, priority_columns,
    )



def clean_sql(sql_expr: str, dialect: str | None = None, prettify: bool = False) -> str:
    """
    Cleans up a SQL expression generated by an LLM by removing
    backticks, fencing and extraneous space and semi-colons.
    """
    sql_expr = SQLRemoveSourceSeparator.apply_to(sql_expr)
    cleaned_sql = sql_expr.replace("```sql", "").replace("```", "")
    if dialect != 'bigquery':
        cleaned_sql = cleaned_sql.replace('`', '"')
    clean_sql = cleaned_sql.strip().rstrip(";")
    if prettify:
        # 'any' is a legacy BaseSQLSource dialect value that sqlglot no
        # longer accepts (28.x raises 'Unknown dialect any'); normalise it
        # to sqlglot's dialect-agnostic mode so SQLAgent's chat path keeps
        # working for sources that have not declared a concrete dialect.
        sql_dialect = None if dialect == "any" else dialect
        clean_sql = sqlglot.transpile(
            sql_expr,
            read=sql_dialect,
            pretty=True
        )[0]
    return clean_sql


def report_error(exc: Exception, step: ChatStep, language: str = "python", context: str = "", status: str = "failed"):
    error_msg = str(exc)
    stream_details(f'\n\n```{language}\n{error_msg}\n```\n', step, title="Error")
    if context:
        step.stream(context)
    if len(error_msg) > 50:
        error_msg = error_msg[:50] + "..."

    if status == "failed":
        step.failed_title = error_msg
        step.status = "failed"


def stream_details(content: Any, step: Any, title: str = "Expand for details", auto: bool = True, **stream_kwargs) -> str:
    """
    Process content to place code blocks inside collapsible details elements

    Parameters
    ----------
    content : str
        The content to format
    step : Any
        The chat step to stream the formatted content to, or NullStep if interface is None
    title : str
        The title of the details component
    auto : bool
        Whether to automatically determine what to put in details or all.
        If False, all of content will be placed in details components.

    Returns
    -------
    str
        The formatted content with code blocks in details components
    """
    if not hasattr(step, 'stream') or not hasattr(step, 'append'):
        # If the step is a NullStep without append method, just log the content
        log_debug(content)
        return content

    pattern = r'```([\w-]*)\n(.*?)```'
    last_end = 0
    if not auto:
        details = Details(content, title=title, collapsed=True, margin=(0, 20, 5, 20), sizing_mode="stretch_width")
        step.append(details, **stream_kwargs)
        return content

    for match in re.finditer(pattern, content, re.DOTALL):
        if match.start() > last_end:
            step.stream(content[last_end:match.start()], **stream_kwargs)

        language = match.group(1)
        code = match.group(2)

        details = Details(
            f"```{language}\n{code}```",
            title=title,
            collapsed=True,
            margin=(0, 20, 5, 20),
            sizing_mode="stretch_width",
        )
        step.append(details)
        last_end = match.end()

    if last_end < len(content):
        step.stream(content[last_end:], **stream_kwargs)
    return content


def format_msg_content(content: Any) -> str:
    """
    Format a message content field for logging.

    Handles plain strings, multimodal lists (text + images), and
    arbitrary objects.  Non-string items are represented as compact
    ``Type[id]`` placeholders so that base64 blobs never flood logs.

    Parameters
    ----------
    content : Any
        The message content — a string, a list of mixed items, or any object.

    Returns
    -------
    str
        A human-readable, log-safe representation of the content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(f"<{type(item).__name__}[{id(item)}]>")
        return " + ".join(parts)
    return f"<{type(content).__name__}[{id(content)}]>"


def log_debug(msg: Any, offset: int = 24, prefix: str = "", suffix: str = "", show_sep: Literal["above", "below"] | None = None, show_length: bool = False):
    """
    Log a debug message with a separator line above and below.
    """
    terminal_cols, _ = get_terminal_size()
    terminal_cols -= offset  # Account for the timestamp and log level
    delimiter = "_" * terminal_cols
    if show_sep == "above":
        log.debug(f"\033[90m{delimiter}\033[0m")
    if prefix:
        log.debug(prefix)

    if not isinstance(msg, (str, list)):
        log_debug(format_msg_content(msg))
    elif isinstance(msg, list):
        for m in msg:
            log.debug(m if isinstance(m, str) else format_msg_content(m))
    else:
        log.debug(msg)
    if show_length:
        log.debug(f"Characters: \033[94m{len(msg)}\033[0m")
    if suffix:
        log.debug(suffix)
    if show_sep == "below":
        log.debug(f"\033[90m{delimiter}\033[0m")


def mutate_user_message(content: str, messages: list[dict[str, str]], suffix: bool = True, wrap: bool | str = False, inplace: bool = True) -> list[dict[str, str]]:
    """
    Helper to mutate the last user message in a list of messages. Suffixes the content by default, else prefixes.
    """
    mutated = False
    new = []
    for message in messages[::-1]:
        if message["role"] == "user" and not mutated:
            user_message = content_to_text(message["content"])
            if isinstance(wrap, bool):
                user_message = f"{user_message!r}"
            elif isinstance(wrap, str):
                user_message = f"{wrap}{user_message}{wrap}"

            if suffix:
                user_message += " " + content
            else:
                user_message = content + user_message

            final_content = set_content_text(user_message, message["content"])

            mutated = True
            if inplace:
                message["content"] = final_content
                break
            else:
                message = dict(message, content=final_content)
        new.append(message)
    return messages if inplace else new[::-1]


def format_exception(exc: Exception, limit: int = 0) -> str:
    """
    Format and return a string representation of an exception.

    Parameters
    ----------
    exc : Exception
        The exception to be formatted.
    limit : int
        The maximum number of layers of traceback to include in the output.
        Default is 0.

    Returns
    -------
    str
        A formatted string describing the exception and its traceback.
    """
    e_msg = str(exc).replace('\033[1m', '<b>').replace('\033[0m', '</b>')
    tb = html.escape('\n'.join(traceback.format_exception(exc, limit=limit))).replace('\033[1m', '<b>').replace('\033[0m', '</b>')
    return f'<b>{type(exc).__name__}</b>: {e_msg}\n<pre style="overflow-y: auto">{tb}</pre>'


def cast_value(value):
    if len(value) == 1:
        value = value[0]
        if ',' in value and value.replace(',', '').isdigit():
            return [cast_value(v) for v in value.split(",")]
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return [cast_value(v) for v in value]


def parse_huggingface_url(url: str) -> tuple[str, str, dict]:
    """Parse a HuggingFace URL to extract repo, model file, and model_kwargs.

    Args:
        url: HuggingFace URL in format https://huggingface.co/org/repo/blob/main/file.gguf?param1=value1&param2=value2

    Returns:
        tuple: (repo, model_file, model_kwargs)

    Raises:
        ValueError: If URL format is invalid
    """
    parts = url.split('/')
    hf_index = parts.index('huggingface.co')
    if len(parts) < hf_index + 4:  # Need at least org/repo after huggingface.co
        raise ValueError
    repo = f"{parts[hf_index+1]}/{parts[hf_index+2]}"
    model_file = parts[-1]  # Last component is the filename

    # Parse query parameters
    model_kwargs = {}
    if "?" in model_file:
        model_file, query_params = model_file.split('?')
        for key, value in parse_qs(query_params).items():
            model_kwargs[key] = cast_value(value)
    return repo, model_file, model_kwargs


def load_json(json_spec: str) -> dict:
    """
    Load a JSON string, handling unicode escape sequences.
    """
    return json.loads(json_spec.encode().decode('unicode_escape'))


def slug_to_table_name(slug: str) -> str:
    """Extract just the table name from a source-qualified slug."""
    return slug.split(SOURCE_TABLE_SEPARATOR, 1)[-1]


def find_slug_by_table_name(table_name: str, candidates: dict | list) -> str | None:
    """Find a slug in *candidates* whose table-name suffix matches *table_name*."""
    keys = candidates if isinstance(candidates, (list, set, frozenset)) else candidates.keys()
    return next(
        (k for k in keys if slug_to_table_name(k) == table_name),
        None,
    )


def parse_table_slug(table: str, sources: list[Source], normalize: bool = True) -> tuple[Source | None, str]:
    if SOURCE_TABLE_SEPARATOR in table:
        a_source_name, a_table = table.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)
        a_source_obj = next((s for s in sources if s.name == a_source_name), None)
    else:
        a_source_obj = sources[0] if sources else None
        a_table = table
    if normalize:
        a_table = a_source_obj.normalize_table(a_table)
    return a_source_obj, a_table


def truncate_string(s, max_length=30, ellipsis="..."):
    if len(s) <= max_length:
        return s
    part_length = (max_length - len(ellipsis)) // 2
    return f"{s[:part_length]}{ellipsis}{s[-part_length:]}"


def truncate_iterable(iterable, max_length=150) -> tuple[list, list, bool]:
    iterable_list = list(iterable)
    if len(iterable_list) > max_length:
        half = max_length // 2
        first_half_items = iterable_list[:half]
        second_half_items = iterable_list[-half:]

        cols_to_show = first_half_items + second_half_items
        show_ellipsis = True
    else:
        cols_to_show = iterable_list
        show_ellipsis = False
    return cols_to_show, show_ellipsis


async def with_timeout(coro, timeout_seconds=10, default_value=None, error_message=None):
    """
    Executes a coroutine with a timeout.

    Parameters
    ----------
    coro : coroutine
        The coroutine to execute with a timeout
    timeout_seconds : float, default=10
        Maximum number of seconds to wait before timing out
    default_value : Any, default=None
        Value to return if timeout occurs
    error_message : str, default=None
        Custom error message to log on timeout

    Returns
    -------
    Any
        Result of the coroutine if it completes in time, otherwise default_value
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except TimeoutError:
        if error_message:
            log_debug(error_message)
        return default_value

def generate_diff(old_text: str, new_text: str, filename: str = "spec") -> str:
    """
    Generate a unified diff between old and new text.

    Parameters
    ----------
    old_text : str
        The original text
    new_text : str
        The modified text
    filename : str
        The filename to use in the diff header

    Returns
    -------
    str
        A unified diff string showing the changes
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    # Ensure last lines have newlines for proper diff formatting
    if old_lines and not old_lines[-1].endswith('\n'):
        old_lines[-1] += '\n'
    if new_lines and not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm='\n'
    )
    return ''.join(diff)


def apply_changes(lines: list[str], edits: list[LineEdit]) -> str:
    """
    Apply a Patch to a list of lines (no trailing newline characters in elements).
    Indices in the Patch refer to the ORIGINAL `lines`.

    Strategy:
      - To keep indices stable, apply all non-insert edits in DESCENDING order of line_no.
      - Apply inserts in ASCENDING order of line_no (grouped), inserting BEFORE that index.
        Inserts at the same index are applied in the order they appear in `patch.edits`.
    """
    # Separate edits
    inserts: list[InsertLine] = [e for e in edits if e.op == "insert"]
    replaces: list[ReplaceLine] = [e for e in edits if e.op == "replace"]
    deletes: list[DeleteLine] = [e for e in edits if e.op == "delete"]

    n = len(lines)

    # Validate bounds against the ORIGINAL text
    for e in replaces:
        if not (1 <= e.line_no <= n):
            raise IndexError(f"replace line_no out of range: {e.line_no} (1..{n})")
    for e in deletes:
        if not (1 <= e.line_no <= n):
            raise IndexError(f"delete line_no out of range: {e.line_no} (1..{n})")
    # For inserts, only validate >= 1; out-of-range high values will append to end
    for e in inserts:
        if e.line_no < 1:
            raise IndexError(f"insert line_no must be >= 1, got {e.line_no}")

    out = lines[:]

    # 1) Apply replace/delete in descending order to keep original indices valid
    for e in sorted([*replaces, *deletes], key=lambda x: x.line_no, reverse=True):
        if e.op == "replace":
            out[e.line_no-1] = e.line  # type: ignore[union-attr]
        else:  # delete
            del out[e.line_no-1]

    # 2) Apply inserts in ascending order; at same index keep user order
    inserts_sorted = sorted(
        enumerate(inserts), key=lambda pair: (pair[1].line_no, -pair[0])
    )
    for _, ins in inserts_sorted:
        # Cap insert position at current list length (allows sequential line numbers for appending)
        insert_pos = min(ins.line_no - 1, len(out))
        out.insert(insert_pos, ins.line)

    return "\n".join(out)

def class_name_to_llm_spec_key(class_name: str) -> str:
    """
    Convert class name to llm_spec_key using the same logic as Actor.llm_spec_key.
    Removes "Agent" suffix and converts to snake_case.
    """
    # Remove "Agent" suffix from class name
    name = class_name.replace("Agent", "")

    if not name:  # Handle case where class name is just "Agent"
        return "agent"

    result = ""
    i = 0
    while i < len(name):
        char = name[i]

        # Check if this is part of an acronym (current char is uppercase and next char is uppercase too)
        is_part_of_acronym = (
            char.isupper() and
            i + 1 < len(name) and
            name[i + 1].isupper()
        )

        # Add underscore before uppercase letters, unless it's part of an acronym
        if char.isupper() and i > 0 and not is_part_of_acronym and not name[i - 1].isupper():
            result += "_"

        # Add the lowercase character
        result += char.lower()
        i += 1

    return result


def wrap_logfire(span_name: str | None = None, extract_args: bool = True, **instrument_kwargs):
    """
    Decorator to instrument a function with logfire if llm.logfire_tags is provided.
    Expects the decorated function to have an 'llm' attribute (on self) or be passed via instrument_kwargs.
    """
    def decorator(func: Callable):
        def should_instrument(self, llm):
            """Check if the function should be instrumented with logfire."""
            return llm is not None and llm.logfire_tags is not None

        def get_instrumented_func(self, llm):
            """Create and return the logfire-instrumented function."""
            name = span_name or self.__class__.__name__
            return llm._logfire.instrument(name, extract_args=extract_args, **instrument_kwargs)(func)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                llm = instrument_kwargs.get("llm", getattr(self, "llm", None))
                if not should_instrument(self, llm):
                    return await func(self, *args, **kwargs)
                instrumented_func = get_instrumented_func(self, llm)
                return await instrumented_func(self, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                llm = instrument_kwargs.get("llm", getattr(self, "llm", None))
                if not should_instrument(self, llm):
                    return func(self, *args, **kwargs)
                instrumented_func = get_instrumented_func(self, llm)
                return instrumented_func(self, *args, **kwargs)
            return sync_wrapper

    return decorator


def wrap_logfire_on_method(cls, method_name: str):
    """
    Automatically adds logfire wrapping to a method.
    """
    original_method = getattr(cls, method_name)
    if not hasattr(original_method, '_logfire_wrapped'):
        wrapped_method = wrap_logfire()(original_method)
        wrapped_method._logfire_wrapped = True
        setattr(cls, method_name, wrapped_method)


def normalized_name(inst: param.Parameterized):
    """
    Returns the name of a Parameterized instance, stripping
    the auto-generated object count.
    """
    class_name = inst.__class__.__name__
    if re.match('^'+class_name+'[0-9]{5}$', inst.name):
        return class_name
    return inst.name


def set_nested(data, keys, value):
    """Set nested dictionary value"""
    reduce(getitem, keys[:-1], data)[keys[-1]] = value


def content_to_text(content: Any) -> str:
    """
    Extract the text portion from a message content field.

    Handles both plain string content and multimodal content lists
    (e.g. containing both text strings and image objects). Returns
    just the concatenated text parts.

    Parameters
    ----------
    content : str | list
        The message content — either a plain string or a list of
        mixed content items (strings, Image objects, etc.).

    Returns
    -------
    str
        The extracted text content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(
            item for item in content if isinstance(item, str)
        )
    # if it's neither a str nor list, it's an object type, like image bytes
    return ''


def set_content_text(new_text: str, content: str | list) -> str | list:
    """
    Return a new content value with the text part replaced by *new_text*,
    preserving any non-text items (e.g. images) in a multimodal list.

    Parameters
    ----------
    new_text : str
        The replacement text.
    content : str | list
        The original message content — either a plain string or a
        multimodal list of strings and other objects.

    Returns
    -------
    str | list
        *new_text* when content is a plain string, or
        ``[new_text, *non_text_items]`` when non-text items exist,
        or *new_text* when the list contained only strings.
    """
    if isinstance(content, list):
        non_text = [item for item in content if not isinstance(item, str)]
        return [new_text] + non_text if non_text else new_text
    return new_text


def serialize_image_content(
    filename: str, data: bytes, mime_type: str | None = None
) -> Any | None:
    """
    Build an ``instructor.Image`` payload from raw image bytes.

    Returns ``None`` if the file extension is not a recognised image
    type or if the ``instructor`` library is unavailable.

    Parameters
    ----------
    filename : str
        Original filename — used to determine the MIME type when
        *mime_type* is not given.
    data : bytes
        Raw image bytes.
    mime_type : str | None
        Explicit MIME type.  Inferred from *filename* when ``None``.

    Returns
    -------
    instructor.Image | None
        An instructor Image object ready to be included in a message
        content list, or ``None`` when the input is not an image.
    """
    ext = Path(filename).suffix.lower()
    if mime_type is None:
        mime_type = IMAGE_MIME_TYPES.get(ext)
    if mime_type is None or not mime_type.startswith('image/'):
        return None

    if Image is None:
        return None

    b64 = base64.b64encode(data).decode('utf-8')
    return Image.from_raw_base64(b64)


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize DataFrame column names for use in code execution.

    Replaces spaces with underscores and removes non-alphanumeric characters
    (except underscores) from column names.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose columns should be sanitized

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with sanitized column names
    """
    df = df.copy()
    df.columns = [
        re.sub(r'[^\w]', '', col.replace(' ', '_'))
        for col in df.columns
    ]
    return df


def result_to_dataframe(result) -> pd.DataFrame | None:
    """
    Normalize a raw function return value into a DataFrame.

    Handles the common shapes returned by Python API clients:
    DataFrame, iterator/generator of model objects, list[dict],
    single model object, dict.
    """
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, Source):
        return None

    # SourceResult from controls — extract the DataFrame from the first source
    from .controls.ingest.result import SourceResult
    if isinstance(result, SourceResult):
        if not result.sources or not result.table:
            return None
        src = result.sources[0]
        table = result.table
        try:
            return src.get(table)
        except Exception:
            return None

    # Materialise iterators / generators
    if isinstance(result, (Iterator, Generator)):
        try:
            result = list(result)
        except Exception as exc:
            log.warning(f"result_to_dataframe: failed to materialise iterator: {exc}")
            return None

    if isinstance(result, list):
        if not result:
            return pd.DataFrame()
        first = result[0]
        if isinstance(first, dict):
            return pd.json_normalize(result)
        if isinstance(first, (str, int, float, bool)):
            return pd.DataFrame({"value": result})
        # Typed model objects — try __dict__, fall back to vars()
        rows = []
        for obj in result:
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                try:
                    rows.append(vars(obj))
                except TypeError:
                    rows.append({"value": str(obj)})
        return pd.json_normalize(rows)

    if isinstance(result, dict):
        try:
            return pd.json_normalize(result)
        except Exception:
            return pd.DataFrame([result])

    # Single model object — try vars() for __slots__ support
    if not isinstance(result, (str, int, float, bool, type(None))):
        try:
            return pd.json_normalize([vars(result)])
        except TypeError:
            return None

    return None


def get_gridded_metadata(pipeline: Pipeline) -> dict[str, Any] | None:
    """Return gridded metadata for an xarray-backed pipeline, else None.

    Returns a dict with keys ``source_type``, ``dims``, ``coords``,
    ``data_vars``, and ``regular`` (True iff every coordinate dimension is
    uniformly spaced). It deliberately does not label which dims are spatial:
    a grid can be plotted many ways (a lon/lat map, a lon/time Hovmoller, a
    lat/level section), so the view spec picks the axes and the leftover dims
    are collapsed afterwards. Coord arrays are 1-D and held in memory by
    xarray even for large lazy datasets, so the regularity check is cheap; do
    not call on a hot path.
    """
    if not isinstance(pipeline.source, XArraySQLSource) or try_import_xarray() is None:
        return None

    ds = pipeline.source.dataset

    def _is_regular(values) -> bool:
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.size < 2:
            return False
        # datetime64 coords need int64 conversion before allclose
        if np.issubdtype(arr.dtype, np.datetime64):
            arr = arr.astype('int64')
        diffs = np.diff(arr)
        return bool(np.allclose(diffs, diffs[0], rtol=1e-6))

    regular = all(
        _is_regular(ds.coords[d].values) for d in ds.dims if d in ds.coords
    )

    return {
        'source_type': 'xarray',
        'dims': list(ds.dims),
        'coords': {k: list(ds.coords[k].shape) for k in ds.coords},
        'data_vars': list(ds.data_vars),
        'regular': regular,
    }


def _spec_field_references(
    spec: Any, kind: Literal['vega-lite', 'deckgl']
) -> set[str]:
    """Column names a generated view ``spec`` references, so the gridded subset
    can keep the dims a spec actually plots (an axis, color, ...) and collapse
    the rest. The two grammars reference columns differently, so the caller
    states which it is: a Vega-Lite spec names columns in encoding ``field``
    values, a deck.gl spec in ``@@=`` accessor expressions.
    """
    refs: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if kind == 'vega-lite' and key == 'field' and isinstance(val, str):
                    refs.add(val)
                elif kind == 'deckgl' and isinstance(val, str) and val.startswith('@@='):
                    refs.update(re.findall(r'[A-Za-z_]\w*', val))
                else:
                    walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(spec)
    return refs


def subset_gridded_to_2d(
    pipeline: Pipeline, spec: Any, kind: Literal['vega-lite', 'deckgl']
) -> Pipeline:
    """Pin every gridded dimension the view ``spec`` does not reference to its
    first value, so a view that cannot page a dimension (VegaLite, DeckGL)
    renders a single 2D slice instead of the full grid. ``kind`` selects the
    spec grammar (see :func:`_spec_field_references`).

    Runs after the spec is generated, so the dims the model chose for x/y/color
    are kept -- a lon/time Hovmoller works -- and only the leftover dims (e.g.
    time on a lon/lat map) collapse. Returns the pipeline unchanged when it is
    not gridded or the spec already references every dimension.
    """
    md = get_gridded_metadata(pipeline)
    if not md:
        return pipeline
    ds = pipeline.source.dataset
    referenced = _spec_field_references(spec, kind)
    collapse = [dim for dim in ds.dims if dim not in referenced]
    if not collapse:
        return pipeline
    filters = [
        ConstantFilter(field=dim, value=ds[dim].values[0]) for dim in collapse
    ]
    return pipeline.chain(filters=filters)


def normalize_vegalite_spec(
    vega_spec: dict[str, Any], *, editor_type: type[VegaLiteEditor] | None = None
) -> dict[str, Any]:
    """Normalize and validate a parsed Vega-Lite spec without an agent instance.

    Injects ``$schema``, defaults ``width``/``height`` to ``"container"``
    (skipping compound charts such as ``hconcat``/``vconcat``/``concat``/
    ``facet``/``repeat``), validates the spec via ``editor_type.validate_spec``
    and adds geographic pan/zoom interactivity for maps. The spec dict is
    mutated in place and returned wrapped in the view metadata Lumen expects.

    Pure helper: no LLM, no UI and no agent instance are required. ``editor_type``
    defaults to ``VegaLiteEditor`` when not provided.
    """
    if editor_type is None:
        # Imported lazily to avoid an editors -> utils import cycle.
        from .editors import VegaLiteEditor
        editor_type = VegaLiteEditor

    # Remove wrapper properties that aren't part of Vega-Lite spec
    for key in ['sizing_mode', 'min_height', 'type']:
        vega_spec.pop(key, None)

    if "$schema" not in vega_spec:
        vega_spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"

    # Check if this is a compound chart (hconcat, vconcat, concat, facet, repeat)
    is_compound = any(key in vega_spec for key in ['hconcat', 'vconcat', 'concat', 'facet', 'repeat'])

    if is_compound:
        # Remove invalid top-level width/height for compound charts
        # Altair's config.view.continuousWidth/Height handles sub-chart sizing
        vega_spec.pop('width', None)
        vega_spec.pop('height', None)
    else:
        if "width" not in vega_spec:
            vega_spec["width"] = "container"
        if "height" not in vega_spec:
            vega_spec["height"] = "container"

    editor_type.validate_spec(vega_spec)

    # using string comparison because these keys could be in different nested levels
    vega_spec_str = dump_yaml(vega_spec)
    # Handle different types of interactive controls based on chart type
    if "latitude:" in vega_spec_str or "longitude:" in vega_spec_str:
        vega_spec = _add_geographic_items(vega_spec, vega_spec_str)
    # elif ("point: true" not in vega_spec_str or "params" not in vega_spec) and vega_spec_str.count("encoding:") == 1:
    #     # add pan/zoom controls to all plots except geographic ones and points overlaid on line plots
    #     # because those result in an blank plot without error
    #     vega_spec["params"] = [{"bind": "scales", "name": "grid", "select": "interval"}]
    return {"spec": vega_spec, "sizing_mode": "stretch_both", "min_height": 200}


def _add_zoom_params(vega_spec: dict) -> None:
    """Add zoom parameters to vega spec."""
    if "params" not in vega_spec:
        vega_spec["params"] = []

    existing_param_names = {
        p.get("name") for p in vega_spec["params"]
        if isinstance(p, dict) and "name" in p
    }

    for p in VEGA_ZOOMABLE_MAP_ITEMS["params"]:
        if p.get("name") not in existing_param_names:
            vega_spec["params"].append(p)


def _setup_projection(vega_spec: dict) -> None:
    """Setup map projection settings."""
    if "projection" not in vega_spec:
        vega_spec["projection"] = {"type": "mercator"}
    vega_spec["projection"].update(VEGA_ZOOMABLE_MAP_ITEMS["projection"])


def _handle_map_compatibility(vega_spec: dict, vega_spec_str: str) -> None:
    """Handle map projection compatibility and add geographic outlines."""
    has_world_map = "world-110m.json" in vega_spec_str
    uses_albers_usa = vega_spec["projection"]["type"] == "albersUsa"

    if has_world_map and uses_albers_usa:
        # albersUsa incompatible with world map
        vega_spec["projection"] = "mercator"
    elif not has_world_map and not uses_albers_usa:
        # Add world map outlines if needed
        if "layer" not in vega_spec:
            vega_spec["layer"] = [{
                "mark": vega_spec.pop("mark", {})
            }]
        vega_spec["layer"].insert(0, VEGA_MAP_LAYER["world"])


def _add_geographic_items(vega_spec: dict, vega_spec_str: str) -> dict:
    """Add geographic visualization items to vega spec."""
    _add_zoom_params(vega_spec)
    _setup_projection(vega_spec)

    # Remove projection from individual layers to prevent conflicts
    # All layers must inherit the top-level projection for zoom/pan to work correctly
    if "layer" in vega_spec:
        for layer in vega_spec["layer"]:
            if isinstance(layer, dict) and "projection" in layer:
                del layer["projection"]

    _handle_map_compatibility(vega_spec, vega_spec_str)
    return vega_spec
