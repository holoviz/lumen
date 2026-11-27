"""
The project file's ``catalog:`` section: what Lumen's assistant may use of
each source the app serves, by the source's ref and table.

    catalog:
      turbines:
        tables:
          raw_readings:
            assistant: false

A table turned off is left out of what the assistant lists and describes,
and the SQL it writes with Lumen's tools may not read it; dashboards, their
nodes and the component palette still read it. The admin view's Catalog
section (``lumen.flowdash.catalog_admin``) lists the sources and their
tables and writes this section.

The host puts a ``ProjectCatalog`` (or a ``Catalog``) in the editor's
context under ``CATALOG_KEY``; ``hidden_tables`` reads it from there.
"""
from __future__ import annotations

import copy
import logging
import re
import threading

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot

from sqlglot import exp

from ..sources.base import Source
from .project import (
    Problem, Project, ProjectError, ProjectSection, Section, register_section,
)

logger = logging.getLogger(__name__)

SECTION = "catalog"

# The key of the editor context the host puts its ``ProjectCatalog`` in.
CATALOG_KEY = "lumen.catalog"

SOURCE_KEYS = frozenset({"tables"})
TABLE_KEYS = frozenset({"assistant"})

EXAMPLE = "`turbines: {tables: {raw_readings: {assistant: false}}}`"

# The table a legacy name of a served file reads, see ``nodes.add_legacy_tables``.
_LEGACY_TARGET = re.compile(r'FROM "((?:[^"]|"")+)"')


@dataclass(frozen=True)
class Catalog:
    """The ``catalog:`` section as the app uses it: the tables turned off for the assistant, by source ref."""

    hidden: Mapping[str, frozenset[str]] = field(default_factory=dict)


def load_section(section: Section) -> Catalog:
    """
    The loader of the ``catalog:`` section. Raises ``ProjectError`` with
    every problem of its shape; a ref the app does not serve, or a table its
    source does not have, is reported by ``Section.warn`` and kept.
    """
    value = section.value
    if value is None:
        return Catalog()
    if not isinstance(value, Mapping):
        raise ProjectError([section.problem((), f"The catalog maps each source to its tables' settings, e.g. {EXAMPLE}.")])
    problems: list[Problem] = []
    hidden: dict[str, set[str]] = {}
    for ref, spec in value.items():
        if spec is None:
            continue
        if not isinstance(spec, Mapping):
            problems.append(section.problem((ref,), f"Source {ref!r} takes `tables:`, e.g. {EXAMPLE}."))
            continue
        for key in spec:
            if key not in SOURCE_KEYS:
                problems.append(section.problem((ref, key), f"Unknown key {key!r} of source {ref!r}; it takes `tables:`.", key=True))
        tables = spec.get("tables")
        if tables is None:
            continue
        if not isinstance(tables, Mapping):
            problems.append(section.problem((ref, "tables"), f"The tables of source {ref!r} map each table to its settings, e.g. {EXAMPLE}."))
            continue
        for table, settings in tables.items():
            if settings is None:
                continue
            path = (ref, "tables", table)
            if not isinstance(settings, Mapping):
                problems.append(section.problem(path, f"Table {table!r} takes `assistant: false` to turn it off for the assistant."))
                continue
            for key in settings:
                if key not in TABLE_KEYS:
                    problems.append(section.problem((*path, key), f"Unknown key {key!r} of table {table!r}; it takes `assistant:`.", key=True))
            assistant = settings.get("assistant", True)
            if not isinstance(assistant, bool):
                problems.append(section.problem((*path, "assistant"), "`assistant` is true or false."))
            elif not assistant:
                hidden.setdefault(str(ref), set()).add(str(table))
    if problems:
        raise ProjectError(problems)
    _warn_unserved(section, value)
    return Catalog({ref: frozenset(tables) for ref, tables in hidden.items()})


def _warn_unserved(section: Section, value: Mapping[str, Any]) -> None:
    from .nodes import (
        SOURCES_KEY, host_sources, listed_tables, lookup_source,
    )

    if SOURCES_KEY not in section.context:
        return
    # Section.warn logs the warning and keeps it in the project's warnings.
    report = section.warn
    for ref, spec in value.items():
        source = lookup_source(str(ref), section.context)
        if source is None:
            refs = ", ".join(sorted(host_sources(section.context))) or "none"
            report((ref,), f"The catalog names source {ref!r}, which the app does not serve; its sources are: {refs}.", key=True)
            continue
        tables = spec.get("tables") if isinstance(spec, Mapping) else None
        if not isinstance(tables, Mapping):
            continue
        try:
            known = set(listed_tables(source))
        except Exception:
            continue
        for table in tables:
            if table not in known:
                report((ref, "tables", table), f"Source {ref!r} has no table {table!r}.", key=True)


class ProjectCatalog:
    """
    The ``catalog:`` section of ``project`` as it is now, for an app's
    context: what an admin saves with ``Project.set`` is used from then on,
    by every session. A section that fails to load is logged once and the
    catalog read before it is kept.
    """

    def __init__(self, project: Project):
        self.project = project
        self._lock = threading.Lock()
        self._data: Any = None
        self._catalog = Catalog()

    def catalog(self) -> Catalog:
        """The catalog as the project has it now."""
        data = self.project.data
        if data is self._data:
            return self._catalog
        with self._lock:
            if data is not self._data:
                self._data = data
                try:
                    value = self.project.get(SECTION)
                except ProjectError as exc:
                    from panel_flowdash.value_text import error_text

                    logger.warning("Not using the project file's catalog: %s", error_text(exc))
                else:
                    self._catalog = value if isinstance(value, Catalog) else Catalog()
            return self._catalog

    @property
    def hidden(self) -> Mapping[str, frozenset[str]]:
        return self.catalog().hidden


def raw_section(project: Project) -> dict[str, Any]:
    """The ``catalog:`` section as plain YAML data, as the project file has it now."""
    from panel_flowdash.project import _FILE_LOCK, _parse

    path = Path(project.path).resolve() if project.path else None
    if path is not None and path.is_file():
        with _FILE_LOCK:
            text = path.read_text(encoding="utf-8")
        data, _ = _parse(text, project.path)
    else:
        data = project.data
    value = data.get(SECTION)
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _root(source: Source) -> Source:
    from .results import is_derived, root_of

    return root_of(source) if is_derived(source) else source


def _with_legacy_names(source: Source, hidden: set[str]) -> frozenset[str]:
    """``hidden`` with the unlisted names that only keep an earlier name of one of its tables."""
    from .nodes import LEGACY_TABLE

    declared = getattr(source, "tables", None)
    if isinstance(declared, Mapping):
        for name, sql in declared.items():
            if isinstance(sql, str) and LEGACY_TABLE in sql and (match := _LEGACY_TARGET.search(sql)):
                if match.group(1).replace('""', '"') in hidden:
                    hidden.add(name)
    return frozenset(hidden)


def hidden_tables(source: Source, context: Mapping[str, Any] | None = None) -> frozenset[str]:
    """
    The tables of ``source``, a host source or a SQL node's result over
    one, that the catalog in ``context`` (or ``dashboard_context()``) turns
    off for the assistant, under any ref or alias that names the source.
    """
    from .nodes import lookup_source

    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    catalog = context.get(CATALOG_KEY)
    hidden_by_ref = getattr(catalog, "hidden", None)
    if not hidden_by_ref:
        return frozenset()
    root = _root(source)
    hidden: set[str] = set()
    for ref, tables in hidden_by_ref.items():
        if lookup_source(ref, context) is root:
            hidden |= tables
    return _with_legacy_names(root, hidden) if hidden else frozenset()


def assistant_tables(source: Source, context: Mapping[str, Any] | None = None) -> list[str]:
    """The tables of ``source`` the assistant may use: its listed tables without those turned off."""
    from .nodes import listed_tables

    hidden = hidden_tables(source, context)
    return [table for table in listed_tables(source) if table not in hidden]


def hidden_reads(sql: str, source: Source, hidden: frozenset[str]) -> list[str]:
    """The tables of ``hidden`` that ``sql``, a query on ``source``, reads by name."""
    from .results import results_of

    if not hidden:
        return []
    parsed = sqlglot.parse_one(sql, read=getattr(source, "dialect", None))
    ctes = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    results = {name.lower() for name in results_of(source)}
    by_name = {table.lower(): table for table in hidden}
    found: dict[str, None] = {}
    for table in parsed.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue
        name = table.name.lower()
        if not table.db and (name in ctes or name in results):
            continue
        qualified = ".".join(part.name for part in table.parts).lower()
        match = by_name.get(qualified) or (by_name.get(name) if table.db in ("", "main") else None)
        if match is not None:
            found[match] = None
    return list(found)


def turned_off(tables: list[str] | frozenset[str], ref: str | None = None) -> str:
    """Why the assistant may not use ``tables``, to tell the model."""
    names = ", ".join(repr(table) for table in sorted(tables))
    of = f" of source {ref!r}" if ref else ""
    plural = len(tables) != 1
    return (f"Table{'s' * plural} {names}{of} {'are' if plural else 'is'} turned off for the assistant in the "
            "app's catalog, so you cannot read or describe "
            f"{'them' if plural else 'it'}; tell the user an admin can turn {'them' if plural else 'it'} on.")


def _settings_view(project: Project, editable: bool) -> Any:
    from .catalog_admin import catalog_settings

    return catalog_settings(project, editable)


def _register() -> None:
    from .metrics import _LumenSection

    admin = {
        key: value for key, value in {"settings_view": _settings_view, "group": "Data"}.items()
        if key in getattr(ProjectSection, "__dataclass_fields__", {})
    }
    register_section((_LumenSection if admin else ProjectSection)(
        SECTION, load_section, title="Catalog", icon="storage",
        description="The sources the app serves and their tables. Turn a table off to keep the assistant from "
        "using it; dashboards still read it.",
        **admin,
    ))


_register()
