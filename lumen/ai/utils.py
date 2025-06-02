from __future__ import annotations

import asyncio
import html
import inspect
import json
import math
import re
import time
import traceback

from functools import wraps
from pathlib import Path
from shutil import get_terminal_size
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs

import pandas as pd

from jinja2 import (
    ChoiceLoader, DictLoader, Environment, FileSystemLoader, StrictUndefined,
)
from markupsafe import escape

from lumen.pipeline import Pipeline

from ..transforms import SQLRemoveSourceSeparator
from ..util import log
from .components import Details
from .config import (
    PROMPTS_DIR, SOURCE_TABLE_SEPARATOR, UNRECOVERABLE_ERRORS,
    RetriesExceededError,
)

if TYPE_CHECKING:
    from panel.chat.step import ChatStep

    from lumen.ai.models import LineChange
    from lumen.sources.base import Source


def format_float(num):
    if pd.isna(num) or math.isinf(num):
        return num
    # if is integer, round to 0 decimals
    if num == int(num):
        return f"{int(num)}"
    elif 0.01 <= abs(num) < 100:
        return f"{num:.1f}"  # Regular floating-point notation with one decimal
    else:
        return f"{num:.1e}"  # Exponential notation with one decimal


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
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_messages if msg['content'].strip()
    )
    system_prompt = {
        "role": "system",
        "content": f"<Chat History>\n{formatted_history}\n<\\Chat History>"
    }
    return [system_prompt] if last_user_index == -1 else [system_prompt, last_user_message]


def render_template(template_path: Path | str, overrides: dict | None = None, relative_to: Path = PROMPTS_DIR, **context):
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

    # Create a loader that can search in multiple directories
    fs_loader = FileSystemLoader(search_paths)

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

    template = env.get_template(template_name)
    return template.render(**context)


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
                        if isinstance(e, UNRECOVERABLE_ERRORS):
                            log_debug(f"LLM encountered unrecoverable error: {e}")
                            raise e
                        elif i == retries - 1:
                            raise RetriesExceededError("Maximum number of retries exceeded.") from e
                        errors.append(str(e))
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
                        if isinstance(e, UNRECOVERABLE_ERRORS):
                            log_debug(f"LLM encountered unrecoverable error: {e}")
                            raise e
                        elif i == retries - 1:
                            raise RetriesExceededError("Maximum number of retries exceeded.") from e
                        errors.append(str(e))
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

    # scale the number of enums based on the number of columns
    if reduce_enums:
        max_enums = max(2, min(10, int(10 * math.exp(-0.1 * max(0, num_cols - 10)))))
    else:
        max_enums = 6
    truncate_limit = min(limit or 5, max_enums)

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
        for field, spec in schema.items():
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
    for field, spec in schema.items():
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


async def describe_data(df: pd.DataFrame) -> str:
    def describe_data_sync(df):
        size = df.size
        shape = df.shape
        if size < 250:
            return df

        is_summarized = False
        if shape[0] > 5000:
            is_summarized = True
            df = df.sample(5000)

        df = df.sort_index()

        for col in df.columns:
            if isinstance(df[col].iloc[0], pd.Timestamp):
                df[col] = pd.to_datetime(df[col])

        sampled_columns = False
        if len(df.columns) > 10:
            df = df[df.columns[:10]]
            sampled_columns = True

        describe_df = df.describe(percentiles=[])
        columns_to_drop = ["min", "max"] # present if any numeric
        columns_to_drop = [col for col in columns_to_drop if col in describe_df.columns]
        df_describe_dict = describe_df.drop(columns=columns_to_drop).to_dict()

        for col in df.select_dtypes(include=["object"]).columns:
            if col not in df_describe_dict:
                df_describe_dict[col] = {}
            try:
                # Get unique values for enum processing
                unique_values = df[col].dropna().unique().tolist()
                if unique_values:
                    # Process enums just like in get_schema
                    temp_spec = {"enum": unique_values}
                    updated_spec, _ = process_enums(
                        temp_spec,
                        num_cols=len(df.columns),
                        limit=5000,  # Using the sample limit
                        include_enum=True,
                        reduce_enums=True
                    )
                    if "enum" in updated_spec:
                        df_describe_dict[col]["enum"] = updated_spec["enum"]

                df_describe_dict[col]["nunique"] = df[col].nunique()
            except Exception:
                df_describe_dict[col]["nunique"] = 'unknown'
            try:
                df_describe_dict[col]["lengths"] = {
                    "max": df[col].str.len().max(),
                    "min": df[col].str.len().min(),
                    "mean": format_float(float(df[col].str.len().mean())),
                }
            except AttributeError:
                pass

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
                df_describe_dict[col][key] = format_float(df_describe_dict[col][key])

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].apply(format_float)

        return {
            "summary": {
                "total_table_cells": size,
                "total_shape": shape,
                "sampled_columns": sampled_columns,
                "is_summarized": is_summarized,
            },
            "stats": df_describe_dict,
        }

    return await asyncio.to_thread(describe_data_sync, df)


def clean_sql(sql_expr: str, dialect: str | None = None) -> str:
    """
    Cleans up a SQL expression generated by an LLM by removing
    backticks, fencing and extraneous space and semi-colons.
    """
    sql_expr = SQLRemoveSourceSeparator.apply_to(sql_expr)
    cleaned_sql = sql_expr.replace("```sql", "").replace("```", "")
    if dialect != 'bigquery':
        cleaned_sql = cleaned_sql.replace('`', '"')
    return cleaned_sql.strip().rstrip(";")


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
            f"```{language}\n\n{code}\n\n```",
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


def log_debug(msg: str | list, offset: int = 24, prefix: str = "", suffix: str = "", show_sep: Literal["above", "below"] | None = None, show_length: bool = False):
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

    if isinstance(msg, list):
        for m in msg:
            log.debug(m)
    else:
        log.debug(msg)
    if show_length:
        log.debug(f"Characters: \033[94m{len(msg)}\033[0m")
    if suffix:
        log.debug(suffix)
    if show_sep == "below":
        log.debug(f"\033[90m{delimiter}\033[0m")


def mutate_user_message(content: str, messages: list[dict[str, str]], suffix: bool = True, wrap: bool | str = False, inplace: bool = True):
    """
    Helper to mutate the last user message in a list of messages. Suffixes the content by default, else prefixes.
    """
    mutated = False
    new = []
    for message in messages[::-1]:
        if message["role"] == "user" and not mutated:
            user_message = message["content"]
            if isinstance(wrap, bool):
                user_message = f"{user_message!r}"
            elif isinstance(wrap, str):
                user_message = f"{wrap}{user_message}{wrap}"

            if suffix:
                user_message += " " + content
            else:
                user_message = content + user_message
            mutated = True
            if inplace:
                message["content"] = user_message
                break
            else:
                message = dict(message, content=user_message)
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


def parse_table_slug(table: str, sources: dict[str, Source], normalize: bool = True) -> tuple[Source | None, str]:
    if SOURCE_TABLE_SEPARATOR in table:
        a_source_name, a_table = table.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)
        a_source_obj = sources.get(a_source_name)
    else:
        a_source_obj = next(iter(sources.values()))
        a_table = table
    if normalize:
        a_table = a_source_obj.normalize_table(a_table)
    return a_source_obj, a_table


def truncate_string(s, max_length=20, ellipsis="..."):
    if len(s) <= max_length:
        return s
    part_length = (max_length - len(ellipsis)) // 2
    return f"{s[:part_length]}{ellipsis}{s[-part_length:]}"


def truncate_iterable(iterable, max_length=20) -> tuple[list, list, bool]:
    iterable_list = list(iterable)
    if len(iterable_list) > max_length:
        half = max_length // 2
        first_half_indices = list(range(half))
        second_half_indices = list(range(len(iterable_list) - half, len(iterable_list)))

        first_half_items = iterable_list[:half]
        second_half_items = iterable_list[-half:]

        cols_to_show = first_half_items + second_half_items
        original_indices = first_half_indices + second_half_indices
        show_ellipsis = True
    else:
        cols_to_show = iterable_list
        original_indices = list(range(len(iterable_list)))
        show_ellipsis = False
    return cols_to_show, original_indices, show_ellipsis


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
    except asyncio.TimeoutError:
        if error_message:
            log_debug(error_message)
        return default_value


def apply_changes(original_lines: list[str], changes: list[LineChange]) -> str:
    """Apply line changes to text."""
    for change in changes:
        original_lines[change.line_no - 1] = change.replacement
    return "\n".join(original_lines)
