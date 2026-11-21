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
    result's column schemas, saves asking the engine for them.
    """
    root, definitions = _DERIVED.get(source, (source, {}))
    definitions = {**definitions, table: sql}
    results = {name: inline_tables(text, definitions, root.dialect) for name, text in definitions.items()}
    if isinstance(root, DuckDBSource):
        derived = root.create_sql_expr_source(results, materialize=False)
    else:
        tables = dict(root.tables) if isinstance(root.tables, dict) else {name: name for name in root.get_tables()}
        derived = root.create_sql_expr_source({**tables, **results})
    stored = {**(source._get_schema_cache() or {})}
    if schema is not None:
        stored[table] = schema
    if stored:
        derived._schema_cache = stored
    _DERIVED[derived] = (root, definitions)
    return derived
