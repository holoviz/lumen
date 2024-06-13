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
    return template.render(context)


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
