from __future__ import annotations

import asyncio
import inspect
import math
import time

from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import jinja2
import pandas as pd
import yaml

from lumen.pipeline import Pipeline
from lumen.sources.base import Source
from lumen.sources.duckdb import DuckDBSource

from .config import THIS_DIR, UNRECOVERABLE_ERRORS

if TYPE_CHECKING:
    from panel.chat.step import ChatStep


def render_template(template, **context):
    template_path = Path(template)
    if not template_path.exists():
        template_path = THIS_DIR / "prompts" / template
    template_contents = template_path.read_text()
    template = jinja2.Template(template_contents, undefined=jinja2.StrictUndefined)
    # raise if missing keys in context for template
    return template.render(context).strip()


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
                        if isinstance(e, UNRECOVERABLE_ERRORS) or i == retries - 1:
                            raise
                        errors.append(str(e))
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
                        if isinstance(e, UNRECOVERABLE_ERRORS) or i == retries - 1:
                            raise
                        errors.append(str(e))
                        if sleep:
                            time.sleep(sleep)

            return sync_wrapper

    return decorator


def format_schema(schema):
    formatted = {}
    for field, spec in schema.items():
        if "enum" in spec and len(spec["enum"]) > 5:
            spec["enum"] = spec["enum"][:5] + ["..."]
        if "type" in spec:
            if spec["type"] == "string":
                spec["type"] = "str"
            elif spec["type"] == "integer":
                spec["type"] = "int"
            elif spec["type"] == "boolean":
                spec["type"] = "bool"
        formatted[field] = spec
    return formatted


async def get_schema(
    source: Source | Pipeline,
    table: str | None = None,
    include_min_max: bool = True,
    include_enum: bool = True,
    include_count: bool = False,
    **get_kwargs
):
    if isinstance(source, Pipeline):
        schema = await asyncio.to_thread(source.get_schema)
    else:
        if "limit" not in get_kwargs:
            get_kwargs["limit"] = 100
        schema = await asyncio.to_thread(source.get_schema, table, **get_kwargs)
    schema = dict(schema)

    # first pop regardless to prevent
    # argument of type 'numpy.int64' is not iterable
    count = schema.pop("count", None)

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
        if "enum" not in spec:
            continue
        elif not include_enum:
            spec.pop("enum")
        elif "limit" in get_kwargs and len(spec["enum"]) > get_kwargs["limit"]:
            spec["enum"].append("...")

    if count and include_count:
        spec["count"] = count

    schema = format_schema(schema)
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


def report_error(exc: Exception, step: ChatStep):
    error_msg = str(exc)
    step.stream(f'\n```python\n{error_msg}\n```')
    if len(error_msg) > 50:
        error_msg = error_msg[:50] + "..."
    step.failed_title = error_msg
    step.status = "failed"


async def gather_table_sources(available_sources: list[Source]) -> tuple[dict[str, Source], str]:
    """
    Get a dictionary of tables to their respective sources
    and a markdown string of the tables and their schemas.
    """
    tables_to_source = {}
    tables_schema_str = "\nHere are the tables\n"
    for source in available_sources:
        for table in source.get_tables():
            tables_to_source[table] = source
            if isinstance(source, DuckDBSource) and source.ephemeral:
                schema = await get_schema(source, table, include_min_max=False, include_enum=True, limit=1)
                tables_schema_str += f"### {table}\nSchema:\n```yaml\n{yaml.dump(schema)}```\n"
            else:
                tables_schema_str += f"### {table}\n"
    return tables_to_source, tables_schema_str
