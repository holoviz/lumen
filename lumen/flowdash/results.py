"""
A SQL node's result as a source that later SQL nodes query by table name.

The source a SQL node outputs on ``result`` is its input source plus one
table holding the node's result, defined by the node's full query rather
than the capped statement it fetches. Not every engine can resolve such a
table by name, so a query on a derived source always runs as one statement
on the root source, with the results it reads inlined as CTEs (``expand``).
"""
from __future__ import annotations

import weakref

from typing import Any

from ..data.sql import inline_tables
from ..sources.base import BaseSQLSource, Source
from ..sources.duckdb import DuckDBSource

# Derived source -> (root source, {result table: query written against the parent source}).
_DERIVED: weakref.WeakKeyDictionary[Source, tuple[BaseSQLSource, dict[str, str]]] = weakref.WeakKeyDictionary()


def is_derived(source: Any) -> bool:
    """Whether ``source`` is a SQL node's result rather than a source the host provides."""
    try:
        return source in _DERIVED
    except TypeError:
        return False


def root_of(source: BaseSQLSource) -> BaseSQLSource:
    """The source the host provides that ``source`` derives from, or ``source`` itself."""
    return _DERIVED[source][0] if is_derived(source) else source


def results_of(source: BaseSQLSource) -> dict[str, str]:
    """The result tables of ``source`` by name, each with the query that defines it."""
    return dict(_DERIVED[source][1]) if is_derived(source) else {}


def expand(sql: str, source: BaseSQLSource) -> tuple[BaseSQLSource, str]:
    """
    The root source of ``source`` and ``sql`` as one statement on it, with
    every result table it reads, directly or through another result, inlined
    as a CTE. On a root source ``sql`` comes back unchanged.
    """
    if not is_derived(source):
        return source, sql
    root, definitions = _DERIVED[source]
    return root, inline_tables(sql, definitions, root.dialect)


def derive(source: BaseSQLSource, table: str, sql: str, schema: dict[str, Any] | None = None) -> BaseSQLSource:
    """
    ``source`` plus a table ``table`` defined by ``sql``, a query on
    ``source``. Each result table is declared by its expanded statement, so
    reading one through the source runs on any engine. ``schema``, the
    result's column schemas, saves asking the engine for its columns and
    their types; value ranges and counts are left out, since the schema may
    come from a capped fetch.
    """
    root, definitions = _DERIVED.get(source, (source, {}))
    stored = {**(source._get_schema_cache() or {})}
    if schema is not None:
        stored[table] = {
            column: {key: spec[key] for key in ("type", "format") if key in spec}
            for column, spec in schema.items() if isinstance(spec, dict)
        }
    return _derived(root, {**definitions, table: sql}, stored)


def _derived(root: BaseSQLSource, definitions: dict[str, str], stored: dict[str, Any]) -> BaseSQLSource:
    results = {name: inline_tables(text, definitions, root.dialect) for name, text in definitions.items()}
    if isinstance(root, DuckDBSource):
        derived = root.create_sql_expr_source(results, materialize=False)
    else:
        tables = dict(root.tables) if isinstance(root.tables, dict) else {name: name for name in root.get_tables()}
        derived = root.create_sql_expr_source({**tables, **results})
    if stored:
        derived._schema_cache = stored
    _DERIVED[derived] = (root, definitions)
    return derived


def not_a_source(value: Any, edge: str | None = None) -> str:
    """Why a SQL node's ``source`` input refuses ``value``, coming in over ``edge`` (``node.port``) if known."""
    kind = "a Lumen pipeline" if type(value).__name__ == "Pipeline" else f"a {type(value).__name__}"
    given = f"{edge} gives {kind}" if edge else f"one input gives {kind}"
    return f"source takes a Lumen SQL source or a SQL node's result; {given}."


def merge(sources: Any) -> Any:
    """
    One source for the inputs of a SQL node, which may be several when it
    joins earlier results: their root source plus every result table of
    each. A single input comes back as it is. Raises ``ValueError`` when
    the inputs derive from different root sources, which a query cannot
    join, or when two of them hold different results under one table name.
    """
    if not isinstance(sources, (list, tuple)):
        return sources
    sources = [source for source in sources if source is not None]
    if len(sources) < 2:
        return sources[0] if sources else None
    if wrong := [source for source in sources if not isinstance(source, BaseSQLSource)]:
        raise ValueError(not_a_source(wrong[0]))
    roots = {id(root_of(source)): root_of(source) for source in sources}
    if len(roots) > 1:
        raise ValueError(
            "Joining results of different sources is not supported: every input must derive from the same source."
        )
    (root,) = roots.values()
    definitions: dict[str, str] = {}
    stored: dict[str, Any] = {}
    for source in sources:
        for table, sql in results_of(source).items():
            clash = next((name for name in definitions if name.lower() == table.lower()), None)
            if clash is not None and definitions[clash] != sql:
                raise ValueError(
                    f"Two inputs hold different results named {table!r}; give one of their SQL nodes another table name."
                )
            definitions[table] = sql
        stored.update(source._get_schema_cache() or {})
    return _derived(root, definitions, stored) if definitions else root
