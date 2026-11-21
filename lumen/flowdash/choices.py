"""
The ``config_choices=`` hooks of the nodes and the column reader of
pipelines, see ``panel_flowdash.register`` and
``panel_flowdash.register_port_type``. The inspector offers what they
return as dropdowns, so a source ref, a table or a column is picked rather
than typed from memory. The ``palette_presets=`` hooks list each source the
host provides, and each of its tables, in the component palette.
"""
from __future__ import annotations

import inspect
import weakref

from collections.abc import Callable, Mapping
from typing import Any

from panel_flowdash import register
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


# Tables per source, read once: the palette asks for presets whenever it is drawn.
_TABLES: weakref.WeakKeyDictionary[Source, list[str]] = weakref.WeakKeyDictionary()


def _tables(source: Source) -> list[str]:
    if source not in _TABLES:
        try:
            _TABLES[source] = sorted(source.get_tables())
        except Exception:
            return []
    return _TABLES[source]


def _host_sources(context: Mapping[str, Any]) -> dict[str, Source]:
    from .nodes import SOURCES_KEY

    return {ref: source for ref, source in (context.get(SOURCES_KEY) or {}).items() if isinstance(source, Source)}


def source_presets(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A ``Lumen/source`` palette entry per source the host provides, with its ref filled in."""
    return [
        {
            "title": f"{ref} (source)",
            "description": f"The source {ref} the app provides.",
            "tooltip": f"Source {ref}: {', '.join(tables) or 'no tables'}",
            "tags": [ref, *tables],
            "config": {"ref": ref},
        }
        for ref, source in _host_sources(context).items()
        for tables in [_tables(source)]
    ]


def table_presets(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    A ``Lumen/table`` palette entry per table of each source the host
    provides, with the table filled in; dropped on the source node's output
    it is wired to it.
    """
    return [
        {
            "title": f"{table} ({ref})",
            "description": f"Table {table} of the source {ref}; drop it on the {ref} source node's output.",
            "tooltip": f"Source {ref}, table {table}",
            "tags": [ref, table],
            "config": {"table": table},
        }
        for ref, source in _host_sources(context).items()
        for table in _tables(source)
    ]


def palette_presets(hook: Callable[[Mapping[str, Any]], list[dict[str, Any]]]) -> dict[str, Any]:
    """``register`` arguments offering ``hook``'s presets, where panel-flowdash supports them."""
    return {"palette_presets": hook} if "palette_presets" in inspect.signature(register).parameters else {}
