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

IGNORED_REF = "ignored while a source is connected"

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
    """
    Offer the refs of the host's sources for a ``Lumen/table`` node, and the
    tables of its source: the connected one, or else the one its ref names.
    With a source connected, a ref it still holds is only shown as ignored.
    """
    from .nodes import listed_tables, lookup_source

    source, ref = inputs.get("source"), config.get("ref")
    if source is None:
        choices: dict[str, Any] = dict(source_choices(config, inputs))
        source = lookup_source(ref) if ref else None
    else:
        choices = {"ref": {f"{ref} ({IGNORED_REF})": ref}} if ref else {}
    if isinstance(source, Source):
        choices["table"] = sorted(listed_tables(source))
    return choices


# Tables per source, read once per reload: the palette asks for presets whenever it
# is drawn, and listing a remote database's tables is a query.
_TABLES: weakref.WeakKeyDictionary[Source, tuple[Any, Any, list[str]]] = weakref.WeakKeyDictionary()


def _tables(source: Source) -> list[str]:
    """
    The tables of ``source``, cached until it reloads: ``Source.clear_cache``,
    which a change of its reload params calls, replaces its schema cache, and
    a change of its ``tables`` changes the declared tables.
    """
    declared = getattr(source, "tables", None)
    declared = sorted(declared) if isinstance(declared, (dict, list)) else declared
    cached = _TABLES.get(source)
    token = getattr(source, "_schema_cache", None)
    if cached is not None and cached[0] is token and cached[1] == declared:
        return cached[2]
    from .nodes import listed_tables

    try:
        tables = sorted(listed_tables(source))
    except Exception:
        return []
    _TABLES[source] = (token, declared, tables)
    return tables


def _host_sources(context: Mapping[str, Any]) -> dict[str, Source]:
    from .nodes import SOURCES_KEY

    return {ref: source for ref, source in (context.get(SOURCES_KEY) or {}).items() if isinstance(source, Source)}


def source_presets(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A ``Lumen/source`` palette entry per source the host provides, with its ref filled in."""
    return [
        {
            "title": f"{ref} (source)",
            "description": f"The source {ref} the app provides, for SQL nodes and tables that share it.",
            "tooltip": f"Source {ref}: {', '.join(tables) or 'no tables'}",
            "tags": [ref, *tables],
            "config": {"ref": ref},
        }
        for ref, source in _host_sources(context).items()
        for tables in [_tables(source)]
    ]


def _served_file(source: Source, table: str) -> str | None:
    from .nodes import served_file

    return served_file(source, table)


def table_presets(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    A ``Lumen/table`` palette entry per table of each source the host
    provides, with its ref and the table filled in, so the node reads the
    table by itself.
    """
    return [
        {
            "title": f"{table} ({ref})",
            "description": f"Table {table} of the source {ref}.",
            "tooltip": f"Source {ref}, table {table}" + (f", file {path}" if (path := _served_file(source, table)) else ""),
            "tags": [ref, table],
            "config": {"ref": ref, "table": table},
        }
        for ref, source in _host_sources(context).items()
        for table in _tables(source)
    ]


def palette_presets(hook: Callable[[Mapping[str, Any]], list[dict[str, Any]]]) -> dict[str, Any]:
    """``register`` arguments offering ``hook``'s presets, where panel-flowdash supports them."""
    return {"palette_presets": hook} if "palette_presets" in inspect.signature(register).parameters else {}
