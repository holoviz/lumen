import asyncio
import inspect
import time

from functools import wraps
from pathlib import Path

import jinja2
import pandas as pd

from lumen.pipeline import Pipeline
from lumen.sources.base import Source

THIS_DIR = Path(__file__).parent


class LlmSetupError(Exception):
    """
    Raised when an error occurs during the setup of the LLM.
    """

    pass


UNRECOVERABLE_ERRORS = (
    ImportError,
    LlmSetupError,
    RecursionError,
)


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
                        print(errors)
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
        formatted[field] = spec
    return formatted


def get_schema(
    source: Source | Pipeline, table: str = None, include_min_max: bool = True
):
    if isinstance(source, Pipeline):
        schema = source.get_schema()
    else:
        schema = source.get_schema(table, limit=100)
    schema = dict(schema)

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
    schema = format_schema(schema)
    return schema


def describe_data(df: pd.DataFrame) -> str:
    def format_float(num):
        if pd.isna(num):
            return num
        # if is integer, round to 0 decimals
        if num == int(num):
            return f"{int(num)}"
        elif 0.01 <= abs(num) < 100:
            return f"{num:.1f}"  # Regular floating-point notation with two decimals
        else:
            return f"{num:.1e}"  # Exponential notation with two decimals

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
        df_describe_dict[col]["nunique"] = df[col].nunique()
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

    data = {
        "summary": {
            "total_table_cells": size,
            "total_shape": shape,
            "is_summarized": is_summarized,
        },
        "stats": df_describe_dict,
    }
    return data
