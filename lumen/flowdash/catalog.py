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
nodes and the component palette still read it.

The host puts a ``ProjectCatalog`` (or a ``Catalog``) in the editor's
context under ``CATALOG_KEY``; ``hidden_tables`` reads it from there.

This module also reads what the app's sources say of their tables
(``catalog_tables``, ``table_info``, ``table_detail``) and writes the
section (``set_assistant``), for any view that shows them: the Data
section (``lumen.flowdash.data_section``), or the admin view's Catalog on a
panel-flowdash without sidebar sections (``lumen.flowdash.catalog_admin``).
"""
from __future__ import annotations

import copy
import datetime as dt
import logging
import math
import re
import threading

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot

from sqlglot import exp

from ..sources.base import BaseSQLSource, Source
from ..transforms.sql import SQLLimit
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

# The rows ``table_detail`` reads of a table, and how long reading them may take.
SAMPLE_ROWS = 10
SAMPLE_TIMEOUT = 5
# The values of a column ``table_detail`` lists before saying how many more there are.
VALUES_SHOWN = 6

NOT_ADMIN = "Only admins can change what the assistant uses, so this was not saved."

# Metadata ``table_detail`` reads into its own fields, or that only Lumen uses.
_SHOWN_KEYS = frozenset({"description", "columns", "rows", "docs", "source_name", "data_type"})


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


# ---------------------------------------------------------------------------
# The sources' tables


@dataclass
class CatalogTable:
    """A table of a source the app serves: its source's ref, whether the assistant may use it, and its upload."""

    ref: str
    table: str
    source: Source
    assistant: bool = True
    upload: Any = None
    """The ``lumen.flowdash.uploads.Upload`` the source was read from, if it is an upload."""


def catalog_tables(
    context: Mapping[str, Any], section: Mapping[str, Any] | None = None,
) -> tuple[list[CatalogTable], dict[str, str]]:
    """
    Every table of every source in ``context``, uploads included, in the
    order the sources are served, with whether ``section`` (the raw
    ``catalog:`` data, read from the project file in ``context`` when not
    given) leaves it to the assistant; and why the tables of a source that
    failed to list them are missing, by its ref. Lists the tables, which
    may query a source; run it off the event loop for a slow one.
    """
    from .nodes import host_sources, listed_tables
    from .uploads import UPLOADS_KEY

    if section is None:
        project = getattr(context.get(CATALOG_KEY), "project", None)
        section = raw_section(project) if project is not None else {}
    uploads = context.get(UPLOADS_KEY)
    hidden = hidden_by_source(section, context)
    tables, errors = [], {}
    for ref, source in host_sources(context).items():
        try:
            names = listed_tables(source)
        except Exception as exc:
            errors[ref] = safe_error(exc)
            continue
        off = hidden.get(id(source), set())
        upload = uploads.get(ref) if uploads is not None else None
        tables.extend(CatalogTable(ref, table, source, table not in off, upload) for table in names)
    return tables, errors


@dataclass
class TableInfo:
    """What a list shows of a table: its rows, columns and description, or why they could not be read."""

    rows: int | None = None
    columns: list[str] | None = None
    description: str = ""
    error: str | None = None
    loading: bool = False


@dataclass
class ColumnInfo:
    name: str
    type: str = ""
    values: str = ""
    description: str = ""
    nulls: str = ""
    distinct: str = ""
    sql_type: str = ""
    stats_summary: str = ""
    primary_key: bool = False
    references: str = ""


@dataclass
class TableDetail:
    """All that is known of a table: its columns with their types, statistics and values, its metadata and its first rows."""

    rows: int | None = None
    description: str = ""
    columns: list[ColumnInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    sample: Any = None
    error: str | None = None
    sample_error: str | None = None
    stats: Any = None
    stats_note: str = ""


def safe_error(exc: BaseException) -> str:
    """The first line of ``exc``'s message, with any credentials masked, to show."""
    from panel_flowdash.value_text import error_text, redact_secrets

    lines = redact_secrets(error_text(exc)).strip().splitlines()
    return lines[0] if lines else type(exc).__name__


def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number == value else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def known_info(source: Source, table: str) -> TableInfo | None:
    """What is known of ``table`` without a query: the schema the source stored or the frame it holds."""
    from .nodes import _table_metadata

    try:
        rows, columns = _table_metadata(source, table, source._get_schema_cache() or {})
    except Exception:
        return None
    if columns is None:
        return None
    described = (source.descriptions or {}).get(table) or (source.metadata or {}).get(table) or {}
    return TableInfo(rows=rows, columns=columns, description=_text(described.get("description")))


def table_info(source: Source, table: str) -> TableInfo:
    """The rows, columns and description of ``table`` from the source's metadata, which may query it."""
    try:
        metadata = source.get_metadata(table)
    except Exception as exc:
        return TableInfo(error=safe_error(exc))
    columns = metadata.get("columns")
    return TableInfo(
        rows=_count(metadata.get("rows")),
        columns=[str(c) for c in columns] if isinstance(columns, Mapping) else None,
        description=_text(metadata.get("description")),
    )


def plain_text(value: Any) -> str:
    """``value``, a cell or a bound of a column, as text: empty for a missing value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat(sep=" ") if isinstance(value, dt.datetime) else value.isoformat()
    return str(value)


def _values(spec: Mapping[str, Any]) -> str:
    if isinstance(enum := spec.get("enum"), list) and enum:
        shown = ", ".join(plain_text(v) for v in enum[:VALUES_SHOWN] if v is not None)
        return shown + (f" +{len(enum) - VALUES_SHOWN} more" if len(enum) > VALUES_SHOWN else "")
    low, high = spec.get("inclusiveMinimum"), spec.get("inclusiveMaximum")
    if low is not None or high is not None:
        return f"{plain_text(low)} to {plain_text(high)}"
    return ""


def _type(spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    kind = spec.get("type") or metadata.get("data_type") or ""
    kind = "/".join(map(str, kind)) if isinstance(kind, list) else str(kind)
    if not (fmt := spec.get("format")):
        return kind
    return str(fmt) if kind in ("string", "") else f"{kind} ({fmt})"


def sample_sql(source: BaseSQLSource, table: str, rows: int = SAMPLE_ROWS) -> str:
    """The statement reading the first ``rows`` rows of ``table``."""
    return SQLLimit(limit=rows, write=source.dialect).apply(source.get_sql_expr(source.normalize_table(table)))


def _format_null_share(nulls: float | None) -> str:
    if nulls is None:
        return ""
    if nulls <= 0:
        return "0%"
    if nulls >= 1:
        return "100%"
    pct = nulls * 100
    if pct < 1:
        return "<1%"
    if pct > 99:
        return ">99%"
    return f"{pct:.0f}%"


def _format_distinct(col: Any, stats: Any) -> str:
    from ..data.stats import EXACT

    if col is None or col.distinct is None:
        return ""
    exact = getattr(stats, "method", None) == EXACT
    rows = getattr(stats, "rows", None)
    nonnull = round(rows * (1 - col.nulls)) if exact and rows is not None and col.nulls is not None else None
    if exact and nonnull and nonnull > 1 and col.distinct == nonnull:
        return f"{col.distinct:,} (unique)"
    return f"{col.distinct:,}"


def _format_stats_summary(col: Any, stats: Any) -> str:
    from ..data.stats import EXACT

    if col is None:
        return ""
    if col.nulls is not None and col.nulls >= 1:
        return "All null"
    if col.values:
        parts = []
        for value, count in col.values[:VALUES_SHOWN]:
            text = plain_text(value) or "NULL"
            if isinstance(count, float):
                pct = count * 100
                suffix = " (<1%)" if 0 < pct < 1 else f" ({pct:.0f}%)"
            elif count is not None:
                suffix = f" ({int(count):,})"
            else:
                suffix = ""
            parts.append(f"{text}{suffix}")
        shown = ", ".join(parts)
        exact = getattr(stats, "method", None) == EXACT
        if col.distinct and col.distinct > len(parts) and (exact or col.values_complete):
            shown += f" +{col.distinct - len(parts)} more"
        elif len(col.values) > len(parts):
            shown += f" +{len(col.values) - len(parts)} more"
        elif not col.values_complete:
            shown += ", ..."
        return shown
    if col.kind in ("numeric", "temporal") and (col.min is not None or col.max is not None):
        lo, hi = plain_text(col.min), plain_text(col.max)
        return lo if lo == hi else f"{lo} to {hi}"
    if col.examples:
        shown = ", ".join(plain_text(v) for v in col.examples[:VALUES_SHOWN] if v is not None)
        return f"e.g. {shown}" if shown else ""
    return ""


def _format_stats_note(stats: Any) -> str:
    from ..data.stats import (
        ESTIMATED, EXACT, SAMPLED, SCHEMA, TYPES_ONLY,
    )

    if stats is None or not getattr(stats, "columns", None):
        return ""
    method = getattr(stats, "method", None)
    if method == EXACT:
        return "Exact"
    if method == SAMPLED:
        size = getattr(stats, "sample_size", None)
        if size:
            kind = "sample" if getattr(stats, "sample_random", True) else "first rows"
            return f"{size:,}-row {kind}"
        return "Sampled"
    if method == ESTIMATED:
        return "Engine-estimated (exact ranges)" if getattr(stats, "ranges_exact", False) else "Engine-estimated"
    if method == SCHEMA:
        return "From schema"
    if method == TYPES_ONLY:
        return "Types only"
    return ""


def table_detail(
    source: Source,
    table: str,
    timeout: float = SAMPLE_TIMEOUT,
    *,
    refresh_stats: bool = False,
    include_sample: bool = True,
) -> TableDetail:
    """
    The columns of ``table`` with their types, statistics, value ranges and
    descriptions, its other metadata and, for a SQL source, its first rows.
    Reading the schema and computing ``TableStatsStore`` statistics query
    the table; run this off the event loop.
    """
    from ..data.stats import compact_geometry_type, get_stats_store
    from .nodes import execute

    detail = TableDetail()
    try:
        metadata = source.get_metadata(table)
    except Exception as exc:
        metadata = {}
        detail.error = safe_error(exc)
    try:
        schema = source.get_schema(table)
    except Exception as exc:
        schema = {}
        detail.error = detail.error or safe_error(exc)
    stats = None
    try:
        store = get_stats_store()
        if refresh_stats:
            store.invalidate(source, table)
        stats = store.get(source, table)
        if stats is None:
            stats = store.compute(source, table)
    except Exception as exc:
        logger.debug("Could not compute TableStatsStore stats for %r: %s", table, type(exc).__name__)
    if stats is not None and stats.columns:
        detail.stats = stats
        detail.stats_note = _format_stats_note(stats)
    columns_meta = metadata.get("columns") if isinstance(metadata.get("columns"), Mapping) else {}
    detail.rows = _count(schema.get("__len__")) if isinstance(schema, Mapping) else None
    if detail.rows is None:
        detail.rows = _count(metadata.get("rows"))
    if detail.rows is None and stats is not None:
        detail.rows = _count(stats.rows)
    detail.description = _text(metadata.get("description"))
    stats_by_name = {col.name: col for col in stats.columns} if stats is not None else {}
    names = (
        [name for name in (schema or {}) if name != "__len__"]
        or ([col.name for col in stats.columns] if stats is not None and stats.columns else [])
        or list(columns_meta)
    )
    for name in names:
        spec = schema.get(name) if isinstance(schema.get(name), Mapping) else {}
        column = columns_meta.get(name)
        column = column if isinstance(column, Mapping) else {"description": column} if isinstance(column, str) else {}
        scol = stats_by_name.get(str(name))
        sql_type = compact_geometry_type(scol.type) if scol is not None and scol.type else ""
        stats_summary = _format_stats_summary(scol, stats) if scol is not None else ""
        values = _values(spec) or stats_summary
        detail.columns.append(ColumnInfo(
            str(name),
            _type(spec, column) or sql_type,
            values,
            _text(column.get("description")),
            nulls=_format_null_share(scol.nulls) if scol is not None else "",
            distinct=_format_distinct(scol, stats) if scol is not None else "",
            sql_type=sql_type,
            stats_summary=stats_summary,
            primary_key=bool(scol.primary_key) if scol is not None else False,
            references=str(scol.references or "") if scol is not None else "",
        ))
    detail.metadata = {
        str(key): value for key, value in metadata.items()
        if key not in _SHOWN_KEYS and value not in (None, "", [], {})
    }
    if include_sample and isinstance(source, BaseSQLSource):
        try:
            detail.sample = execute(source, sample_sql(source, table), timeout)
        except Exception as exc:
            detail.sample_error = safe_error(exc)
    return detail


# ---------------------------------------------------------------------------
# Writing the section


def _admin(app: Any) -> bool:
    try:
        return app is not None and bool(app._still_admin())
    except Exception as exc:
        from panel_flowdash.value_text import error_text

        logger.warning("Could not check whether this session is still an admin: %s", error_text(exc))
        return False


def _prune(section: dict[str, Any]) -> dict[str, Any]:
    """``section`` without the tables, sources and ``tables:`` left with no settings."""
    result = {}
    for ref, spec in section.items():
        if not isinstance(spec, dict):
            result[ref] = spec
            continue
        tables = spec.get("tables")
        if isinstance(tables, dict):
            kept = {table: settings for table, settings in tables.items() if settings}
            spec = {**spec, "tables": kept} if kept else {k: v for k, v in spec.items() if k != "tables"}
        if spec:
            result[ref] = spec
    return result


def set_assistant(
    project: Project, ref: str, table: str, on: bool, *, app: Any,
) -> list[Problem]:
    """
    Turn ``table`` of the source ``ref`` on or off for the assistant,
    writing the project file's ``catalog:`` section and keeping the rest of
    it as the file has it now. Turning it on also clears the setting under
    an alias of the source, e.g. one written by hand. Only an admin
    (``app._still_admin()``) may. Returns the problems that stopped it.
    """
    from panel_flowdash.project import _FILE_LOCK
    from panel_flowdash.value_text import error_text

    from .nodes import lookup_source

    with _FILE_LOCK:
        if not _admin(app):
            return [Problem(NOT_ADMIN)]
        try:
            current = raw_section(project)
        except ProjectError as exc:
            return list(exc.problems)
        except OSError as exc:
            return [Problem(f"Could not read the project file: {error_text(exc)}")]
        section = copy.deepcopy(current)
        context = project.context or {}
        source = lookup_source(ref, context)
        if on:
            for name, spec in section.items():
                if name != ref and (source is None or lookup_source(str(name), context) is not source):
                    continue
                settings = ((spec or {}).get("tables") or {}).get(table) if isinstance(spec, dict) else None
                if isinstance(settings, dict):
                    settings.pop("assistant", None)
        else:
            spec = section.get(ref)
            spec = section[ref] = spec if isinstance(spec, dict) else {}
            tables = spec.get("tables")
            tables = spec["tables"] = tables if isinstance(tables, dict) else {}
            settings = tables.get(table)
            settings = tables[table] = settings if isinstance(settings, dict) else {}
            settings["assistant"] = False
        try:
            project.set(SECTION, _prune(section))
        except ProjectError as exc:
            return list(exc.problems)
        except OSError as exc:
            return [Problem(f"Could not write the project file: {error_text(exc)}")]
    who = getattr(getattr(app, "_identity", None), "user", None) or "someone"
    logger.info("Admin: %s turned %s table %r of %r for the assistant.", who, "on" if on else "off", table, ref)
    return []


def hidden_by_source(section: Mapping[str, Any], context: Mapping[str, Any]) -> dict[int, set[str]]:
    """The tables ``section``, the raw ``catalog:`` data, turns off, by the ``id`` of the source each ref names."""
    from .nodes import lookup_source

    hidden: dict[int, set[str]] = {}
    for ref, spec in section.items():
        source = lookup_source(str(ref), context)
        tables = spec.get("tables") if isinstance(spec, Mapping) else None
        if source is None or not isinstance(tables, Mapping):
            continue
        for table, settings in tables.items():
            if isinstance(settings, Mapping) and settings.get("assistant") is False:
                hidden.setdefault(id(source), set()).add(str(table))
    return hidden


def _settings_view(project: Project, editable: bool) -> Any:
    from .catalog_admin import catalog_settings

    return catalog_settings(project, editable)


def _register() -> None:
    from .metrics import _DataSection

    admin = {
        key: value for key, value in {"settings_view": _settings_view, "group": "Data"}.items()
        if key in getattr(ProjectSection, "__dataclass_fields__", {})
    }
    register_section((_DataSection if admin else ProjectSection)(
        SECTION, load_section, title="Catalog", icon="storage",
        description="The sources the app serves and their tables. Turn a table off to keep the assistant from "
        "using it; dashboards still read it.",
        **admin,
    ))


_register()
