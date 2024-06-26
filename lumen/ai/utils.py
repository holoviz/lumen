import asyncio
import inspect
import time

from functools import wraps
from pathlib import Path

import jinja2

from lumen.pipeline import Pipeline
from lumen.sources.base import Source

THIS_DIR = Path(__file__).parent


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
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if i == retries - 1:
                            return "" # do not re-raise due to outer exception handler
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
                        return func(*args, **kwargs)
                    except Exception as e:
                        if i == retries - 1:
                            return "" # do not re-raise due to outer exception handler
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
