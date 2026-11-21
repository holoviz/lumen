"""
The ``config_choices=`` hooks of the nodes and the column reader of
pipelines, see ``panel_flowdash.register`` and
``panel_flowdash.register_port_type``. The inspector offers what they
return as dropdowns, so a source ref, a table or a column is picked rather
than typed from memory.
"""
from __future__ import annotations

from typing import Any

from panel_flowdash.util import is_failure

from ..sources.base import Source
from .checks import pipeline_columns

_TYPES = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean"}


def pipeline_column_types(pipeline: Any) -> dict[str, str | None] | None:
    """
    The columns of a Lumen pipeline with short type names, from its schema,
    so no rows are fetched: ``integer``, ``number``, ``boolean``,
    ``datetime``, ``date`` or ``string``.
    """
    if pipeline is None or is_failure(pipeline):
        return None
    schema = pipeline.schema
    types: dict[str, str | None] = {}
    for column in pipeline_columns(pipeline):
        spec = schema.get(column) or {}
        kind = spec.get("format") if spec.get("format") in ("datetime", "date") else spec.get("type")
        types[column] = "datetime" if kind == "datetime" else "date" if kind == "date" else _TYPES.get(kind)
    return types


def source_choices(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, list[str]]:
    """
    Offer the refs of the sources the host provides for a ``Lumen/source``
    node, and the alias it names one by, if it does.
    """
    from panel_flowdash import dashboard_context

    from .nodes import SOURCE_ALIASES_KEY, host_sources

    sources = host_sources()
    if not sources:
        return {}
    ref = config.get("ref")
    target = (dashboard_context().get(SOURCE_ALIASES_KEY) or {}).get(ref)
    if ref and ref not in sources and target in sources:
        return {"ref": {**{r: r for r in sorted(sources)}, f"{ref} (old name of {target})": ref}}
    return {"ref": sorted(sources)}


def table_choices(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, list[str]]:
    """Offer the tables of the connected source for a ``Lumen/table`` node."""
    source = inputs.get("source")
    if not isinstance(source, Source):
        return {}
    return {"table": sorted(source.get_tables())}
