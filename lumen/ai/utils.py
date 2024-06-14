import asyncio
import inspect
import time

from functools import wraps
from pathlib import Path

import jinja2

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
