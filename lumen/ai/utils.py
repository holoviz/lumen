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
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

import pandas as pd
import yaml

from jinja2 import (
    ChoiceLoader, DictLoader, Environment, FileSystemLoader, StrictUndefined,
)
from markupsafe import escape

from lumen.pipeline import Pipeline
from lumen.sources.base import Source
from lumen.sources.duckdb import DuckDBSource

from ..util import log
from .config import (
    PROMPTS_DIR, PROVIDED_SOURCE_NAME, SOURCE_TABLE_SEPARATOR,
    UNRECOVERABLE_ERRORS, RetriesExceededError,
)

if TYPE_CHECKING:
    from panel.chat.step import ChatStep


def render_template(template_path: Path, overrides: dict | None = None, relative_to: Path = PROMPTS_DIR, **context):
    try:
        template_path = template_path.relative_to(relative_to).as_posix()
    except ValueError:
        pass
    fs_loader = FileSystemLoader(relative_to)

    if overrides:
        # Dynamically create block definitions based on dictionary keys with proper escaping
        block_definitions = "\n".join(
            f"{{% block {escape(key)} %}}{escape(value)}{{% endblock %}}"
            for key, value in overrides.items()
        )
        # Create the dynamic template content by extending the base template and adding blocks
        dynamic_template_content = dedent(
            f"""
            {{% extends "{template_path}" %}}
            {block_definitions}
            """
        )
        dynamic_loader = DictLoader({"dynamic_template": dynamic_template_content})
        choice_loader = ChoiceLoader([dynamic_loader, fs_loader])
        env = Environment(loader=choice_loader, undefined=StrictUndefined)
        template_name = "dynamic_template"
    else:
        env = Environment(loader=fs_loader, undefined=StrictUndefined)
        template_name = str(template_path)

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
                    if errors:
                        kwargs["errors"] = errors
                    try:
                        output = await func(*args, **kwargs)
                        if not output:
                            raise Exception("No valid output from LLM.")
                        return output
                    except Exception as e:
                        if isinstance(e, UNRECOVERABLE_ERRORS):
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
                    if errors:
                        kwargs["errors"] = errors
                    try:
                        output = func(*args, **kwargs)
                        if not output:
                            raise Exception("No valid output from LLM.")
                        return output
                    except Exception as e:
                        if isinstance(e, UNRECOVERABLE_ERRORS):
                            raise e
                        elif i == retries - 1:
                            raise RetriesExceededError("Maximum number of retries exceeded.") from e
                        errors.append(str(e))
                        if sleep:
                            time.sleep(sleep)

            return sync_wrapper

    return decorator


async def get_schema(
    source: Source | Pipeline,
    table: str | None = None,
    include_min_max: bool = True,
    include_enum: bool = True,
    include_count: bool = False,
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

    if include_min_max:
        for field, spec in schema.items():
            if "inclusiveMinimum" in spec:
                spec["min"] = spec.pop("inclusiveMinimum")
            if "inclusiveMaximum" in spec:
                spec["max"] = spec.pop("inclusiveMaximum")
    else:
        for field, spec in schema.items():
            if "inclusiveMinimum" in spec:
                spec.pop("inclusiveMinimum")
            if "inclusiveMaximum" in spec:
                spec.pop("inclusiveMaximum")
            if "min" in spec:
                spec.pop("min")
            if "max" in spec:
                spec.pop("max")

    for field, spec in schema.items():
        if "type" in spec:
            if spec["type"] == "string":
                spec["type"] = "str"
            elif spec["type"] == "integer":
                spec["type"] = "int"
            elif spec["type"] == "number":
                spec["type"] = "num"
            elif spec["type"] == "boolean":
                spec["type"] = "bool"

        if "enum" not in spec:
            continue

        limit = get_kwargs.get("limit")
        max_enums = 10
        truncate_limit = min(limit or 5, 5)
        if not include_enum:
            spec.pop("enum")
            continue
        elif len(spec["enum"]) > max_enums:
            # if there are only 10 enums, fine; let's keep them all
            # else, that assumes this column is like glasbey 100+ categories vs category10
            # so truncate to 5 and add "..."
            spec["enum"] = spec["enum"][:truncate_limit] + ["..."]
        elif limit and len(spec["enum"]) == 1 and spec["enum"][0] is None:
            spec["enum"] = [f"(unknown; truncated to {limit} rows)"]
            continue

        # truncate each enum to 100 characters
        spec["enum"] = [
            enum if enum is None or not isinstance(enum, str) or len(enum) < 100 else f"{enum[:100]} ..."
            for enum in spec["enum"]
        ]

    if count and include_count:
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
    def format_float(num):
        if pd.isna(num) or math.isinf(num):
            return num
        # if is integer, round to 0 decimals
        if num == int(num):
            return f"{int(num)}"
        elif 0.01 <= abs(num) < 100:
            return f"{num:.1f}"  # Regular floating-point notation with two decimals
        else:
            return f"{num:.1e}"  # Exponential notation with two decimals

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

        describe_df = df.describe(percentiles=[])
        columns_to_drop = ["min", "max"] # present if any numeric
        columns_to_drop = [col for col in columns_to_drop if col in describe_df.columns]
        df_describe_dict = describe_df.drop(columns=columns_to_drop).to_dict()

        for col in df.select_dtypes(include=["object"]).columns:
            if col not in df_describe_dict:
                df_describe_dict[col] = {}
            try:
                df_describe_dict[col]["nunique"] = df[col].nunique()
            except Exception:
                df_describe_dict[col]["nunique"] = 'unknown'
            try:
                df_describe_dict[col]["lengths"] = {
                    "max": df[col].str.len().max(),
                    "min": df[col].str.len().min(),
                    "mean": float(df[col].str.len().mean()),
                }
            except AttributeError:
                pass

        for col in df.columns:
            if col not in df_describe_dict:
                df_describe_dict[col] = {}
            df_describe_dict[col]["nulls"] = int(df[col].isnull().sum())

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
                "is_summarized": is_summarized,
            },
            "stats": df_describe_dict,
        }

    return await asyncio.to_thread(describe_data_sync, df)


def clean_sql(sql_expr):
    """
    Cleans up a SQL expression generated by an LLM by removing
    backticks, fencing and extraneous space and semi-colons.
    """
    return sql_expr.replace("```sql", "").replace("```", "").replace('`', '"').strip().rstrip(";")


def report_error(exc: Exception, step: ChatStep, language: str = "python", context: str = ""):
    error_msg = str(exc)
    step.stream(f'\n```{language}\n{error_msg}\n```\n')
    if context:
        step.stream(context)
    if len(error_msg) > 50:
        error_msg = error_msg[:50] + "..."
    step.failed_title = error_msg
    step.status = "failed"


async def gather_table_sources(sources: list[Source], include_provided: bool = True, include_sep: bool = False) -> tuple[dict[str, Source], str]:
    """
    Get a dictionary of tables to their respective sources
    and a markdown string of the tables and their schemas.

    Parameters
    ----------
    sources : list[Source]
        A list of sources to gather tables from.
    include_provided : bool
        Whether to include the provided source in the string; will always be included in the dictionary.
    include_sep : bool
        Whether to include the source separator in the string.
    """
    tables_to_source = {}
    tables_schema_str = ""
    for source in sources:
        for table in source.get_tables():
            tables_to_source[table] = source
            if source.name == PROVIDED_SOURCE_NAME and not include_provided:
                continue
            label = f"{source}{SOURCE_TABLE_SEPARATOR}{table}" if include_sep else table
            if isinstance(source, DuckDBSource) and source.ephemeral or "Provided" in source.name:
                sql = source.get_sql_expr(table)
                schema = await get_schema(source, table, include_enum=True, limit=5)
                tables_schema_str += f"- {label}\nSchema:\n```yaml\n{yaml.dump(schema)}```\nSQL:\n```sql\n{sql}\n```\n\n"
            else:
                tables_schema_str += f"- {label}\n\n"
    return tables_to_source, tables_schema_str.strip()


def log_debug(msg: str | list, offset: int = 24, prefix: str = "", suffix: str = "", show_sep: bool = False, show_length: bool = False):
    """
    Log a debug message with a separator line above and below.
    """
    terminal_cols, _ = get_terminal_size()
    terminal_cols -= offset  # Account for the timestamp and log level
    delimiter = "*" * terminal_cols
    if show_sep:
        log.debug(f"\033[91m{delimiter}\033[0m")
    if prefix:
        log.debug(prefix)

    if isinstance(msg, list):
        for m in msg:
            log.debug(m)
    else:
        log.debug(msg)
    if show_length:
        log.debug(f"Length is \033[94m{len(msg)}\033[0m")
    if suffix:
        log.debug(suffix)


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
