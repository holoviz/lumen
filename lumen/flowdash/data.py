"""
The app's data as the Data section shows it, for any view: each source
with how it got there (``source_summaries``), each metric with whether it
runs (``metric_statuses``), and what uses them: the dashboards whose graphs
read each table and each metric, and the tables each metric is computed
from.

It reads the saved dashboards, not the open ones, and only the dashboards
the session may open, so it never names one someone could not see.
"""
from __future__ import annotations

import logging
import typing as t

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import sqlglot

from sqlglot import exp

from .contracts import (
    METRIC_ID, SOURCE_ID, SQL_ID, TABLE_ID,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardRef:
    """A saved dashboard, by id and title."""

    dashboard_id: str
    title: str


def visible_dashboards(app: t.Any) -> list[t.Any]:
    """
    The saved dashboards ``app``'s session may open: every one for an
    admin, else those it owns or was given access to. None when they
    cannot be read.
    """
    from panel_flowdash.value_text import error_text

    store = getattr(app, "store", None)
    if store is None:
        return []
    try:
        if getattr(app, "is_admin", getattr(app, "_is_admin", False)):
            return list(store._all_dashboards())
        identity = getattr(app, "_identity", None)
        if identity is None:
            return []
        default_allow = app._default_allow() if hasattr(app, "_default_allow") else True
        return list(store.list_accessible(identity, default_allow=default_allow))
    except Exception as exc:
        logger.warning("Could not read the dashboards that use the app's data: %s", error_text(exc))
        return []


def metric_usage(dashboards: Sequence[t.Any]) -> dict[str, list[DashboardRef]]:
    """The dashboards whose Lumen Metric nodes read each metric, by metric name."""
    usage: dict[str, list[DashboardRef]] = {}
    for model in dashboards:
        names = {
            (item.config or {}).get("metric") for item in model.items if item.component_id == METRIC_ID
        }
        for name in sorted(n for n in names if isinstance(n, str) and n):
            usage.setdefault(name, []).append(DashboardRef(model.dashboard_id, model.title))
    return usage


def _canonical(context: Mapping[str, t.Any]) -> t.Callable[[str], str | None]:
    """A function naming a source by the ref the app serves it under, for a ref or an alias of it."""
    from .nodes import host_sources, lookup_source

    refs = {id(source): ref for ref, source in host_sources(context).items()}

    def ref_of(name: str) -> str | None:
        source = lookup_source(name, context) if isinstance(name, str) and name else None
        return refs.get(id(source)) if source is not None else None

    return ref_of


def _dialect(ref: str, context: Mapping[str, t.Any]) -> str | None:
    from .nodes import lookup_source

    dialect = getattr(lookup_source(ref, context), "dialect", None)
    return None if dialect in (None, "any") else dialect


def sql_tables(sql: str, dialect: str | None = None) -> list[str]:
    """The tables ``sql`` reads by name, without the ones its own CTEs define; none if it does not parse."""
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except (sqlglot.errors.SqlglotError, ValueError):
        return []
    if parsed is None:
        return []
    ctes = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    found: dict[str, None] = {}
    for table in parsed.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue
        if not table.db and table.name.lower() in ctes:
            continue
        found[table.name] = None
    return list(found)


def _tables_read(model: t.Any, ref_of: t.Callable[[str], str | None], context: Mapping[str, t.Any]) -> set[tuple[str, str]]:
    """The tables the graph of ``model``, a saved dashboard, reads with Lumen Table and SQL nodes."""
    items = {item.instance_id: item for item in model.items}
    upstream: dict[str, list[str]] = {}
    for edge in model.edges:
        if edge.target_port == "source":
            upstream.setdefault(edge.target, []).append(edge.source)

    def refs(instance_id: str, seen: frozenset[str] = frozenset()) -> set[str]:
        item = items.get(instance_id)
        if item is None or instance_id in seen:
            return set()
        if item.component_id == SOURCE_ID:
            ref = ref_of((item.config or {}).get("ref"))
            return {ref} if ref else set()
        found = set().union(*(refs(up, seen | {instance_id}) for up in upstream.get(instance_id, ())))
        if not found and item.component_id == TABLE_ID and (ref := ref_of((item.config or {}).get("ref"))):
            found.add(ref)
        return found

    read: set[tuple[str, str]] = set()
    for item in model.items:
        config = item.config or {}
        if item.component_id == TABLE_ID and isinstance(table := config.get("table"), str) and table:
            read |= {(ref, table) for ref in refs(item.instance_id)}
        elif item.component_id == SQL_ID and isinstance(sql := config.get("sql"), str) and sql.strip():
            for ref in refs(item.instance_id):
                read |= {(ref, table) for table in sql_tables(sql, _dialect(ref, context))}
    return read


def table_usage(dashboards: Sequence[t.Any], context: Mapping[str, t.Any]) -> dict[tuple[str, str], list[DashboardRef]]:
    """
    The dashboards whose graphs read each table, by the ref of its source
    in ``context`` and its name: a Lumen Table node on it, or a Lumen SQL
    node whose query names it, reading the source directly or through
    earlier SQL nodes. A name a SQL node reads that is no table, e.g. an
    earlier node's result, is kept too; look up only real tables.
    """
    ref_of = _canonical(context)
    usage: dict[tuple[str, str], list[DashboardRef]] = {}
    for model in dashboards:
        for key in sorted(_tables_read(model, ref_of, context)):
            usage.setdefault(key, []).append(DashboardRef(model.dashboard_id, model.title))
    return usage


def metric_tables(spec: t.Any, context: Mapping[str, t.Any]) -> list[tuple[str, str]]:
    """
    The tables a metric defined as ``spec``, its raw ``metrics:`` entry or
    a ``metrics.Metric``, is computed from, by the ref of their source in
    ``context``: its table, or the tables its SQL names. None for a source
    the app does not serve.
    """
    if hasattr(spec, "source") and hasattr(spec, "form"):
        spec = {"source": spec.source, "table": spec.table, "sql": spec.sql}
    if not isinstance(spec, Mapping):
        return []
    ref = _canonical(context)(spec.get("source"))
    if ref is None:
        return []
    if isinstance(table := spec.get("table"), str) and table:
        return [(ref, table)]
    if isinstance(sql := spec.get("sql"), str):
        return [(ref, name) for name in sql_tables(sql, _dialect(ref, context))]
    return []


def hidden_reads_of(spec: t.Any, context: Mapping[str, t.Any]) -> list[str]:
    """
    The tables a metric defined as ``spec`` (see ``metric_tables``) reads that the catalog in
    ``context`` turns off for the assistant, which then leaves the metric
    out too.
    """
    from .catalog import hidden_tables
    from .nodes import lookup_source

    tables = metric_tables(spec, context)
    if not tables:
        return []
    source = lookup_source(tables[0][0], context)
    hidden = hidden_tables(source, context) if source is not None else frozenset()
    return [table for _, table in tables if table in hidden]


# ---------------------------------------------------------------------------
# Sources


def display_path(text: str, roots: t.Sequence[t.Any] = ()) -> str:
    """
    ``text``, a file a source reads, as a page shows it: relative to one of
    ``roots`` (e.g. the project's folder or the uploads folder) when it is
    in one, else its file name, so the server's folders are not shown; a URL
    with its credentials masked.
    """
    from pathlib import Path

    from panel_flowdash.value_text import redact_secrets

    if "://" in text:
        return redact_secrets(text)
    path = Path(text)
    if not path.is_absolute():
        return text.replace("\\", "/")
    for root in roots:
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except (ValueError, OSError):
            continue
    return path.name


def data_roots(project: t.Any) -> list[t.Any]:
    """The folders a page shows files relative to: the project's and its uploads'."""
    from pathlib import Path

    from .uploads import UPLOADS_KEY

    roots = []
    if getattr(project, "path", None):
        roots.append(Path(project.path).parent)
    uploads = (getattr(project, "context", None) or {}).get(UPLOADS_KEY)
    if uploads is not None:
        roots.append(uploads.directory)
    return roots


@dataclass(frozen=True)
class SourceSummary:
    """A source the app serves, as a card shows it: what it is, what it reads and how it got there."""

    ref: str
    kind: str
    """The source's type, e.g. ``DuckDB``, or ``Upload``."""
    files: tuple[str, ...]
    """The files, database or connection it reads, credentials masked."""
    tables: int
    origin: str
    """How it got there, e.g. "From the command line" or "Uploaded by ana, 24 Nov"."""
    upload: t.Any = None
    connection: t.Any = None
    problem: str | None = None
    """Why a source an admin connected is listed but not served, e.g. its credentials can't be read."""


# How people name the databases of connection strings, by scheme.
_SCHEMES = {
    "duckdb": "DuckDB", "mssql": "SQL Server", "mysql": "MySQL", "oracle": "Oracle", "postgresql": "PostgreSQL",
    "snowflake": "Snowflake", "sqlite": "SQLite",
}


def source_kind(source: t.Any) -> str:
    """The type of ``source`` as people know it, e.g. ``DuckDB`` for a ``DuckDBSource``."""
    name = type(source).__name__
    return name.removesuffix("Source") or name


def _location(source: t.Any) -> str:
    location = getattr(source, "uri", None) or getattr(source, "url", None)
    if not isinstance(location, str) or not location or location == ":memory:":
        return ""
    return display_path(location)


def _day(when: t.Any) -> str:
    return f"{when.astimezone():%-d %b}" if when is not None else ""


def source_summaries(
    context: Mapping[str, t.Any], tables: Sequence[t.Any] = (), roots: Sequence[t.Any] = (),
) -> list[SourceSummary]:
    """
    Each source in ``context``, uploads included, in the order they are
    served, with ``tables`` (``catalog.CatalogTable`` entries) counted per
    source, and then each source an admin connected that is not served,
    with why. Files are shown relative to ``roots`` (see ``display_path``).
    """
    from .app import _BUILT
    from .nodes import host_sources, served_file
    from .uploads import UPLOADS_KEY

    uploads = context.get(UPLOADS_KEY)
    counts: dict[str, int] = {}
    files: dict[str, list[str]] = {}
    for entry in tables:
        counts[entry.ref] = counts.get(entry.ref, 0) + 1
        if (path := served_file(entry.source, entry.table)) is not None:
            files.setdefault(entry.ref, []).append(display_path(path).rsplit("/", 1)[-1])
    summaries = []
    for ref, source in host_sources(context).items():
        upload = uploads.get(ref) if uploads is not None else None
        connection = uploads.connection(ref) if uploads is not None else None
        if upload is not None:
            kind, shown = "Upload", (upload.filename,)
            origin = f"Uploaded by {upload.uploaded_by}" + (f", {_day(upload.when)}" if upload.when else "")
        else:
            kind = source_kind(source)
            shown = tuple(dict.fromkeys(files.get(ref) or [location for location in [_location(source)] if location]))
            if connection is not None:
                origin = f"Added by {connection.added_by}" + (f", {_day(connection.when)}" if connection.when else "")
            else:
                origin = "From the command line" if source in _BUILT else "Served by the app"
        summaries.append(SourceSummary(ref, kind, shown, counts.get(ref, 0), origin, upload, connection))
    served = host_sources(context)
    for connection in uploads.connections() if uploads is not None else []:
        if connection.ref in served:
            continue
        scheme = connection.shown.split("://", 1)[0].split("+", 1)[0] if "://" in connection.shown else ""
        summaries.append(SourceSummary(
            connection.ref, _SCHEMES.get(scheme, scheme.capitalize() or "Source"),
            (display_path(connection.shown, roots),), 0,
            f"Added by {connection.added_by}" + (f", {_day(connection.when)}" if connection.when else ""),
            connection=connection, problem=uploads.connection_problem(connection.ref),
        ))
    return summaries


# ---------------------------------------------------------------------------
# Metrics


# What each kind of failure says in a metric's Check, see ``metrics.MetricFailure.kind``.
CHECKS = {
    "valid": "Valid", "source": "Source missing", "table": "Table missing", "column": "Column missing",
    "query": "Can't run", "invalid": "Invalid",
}


@dataclass
class MetricStatus:
    """A metric of the project file as the Metrics page lists it: its definition and whether the app uses it."""

    name: str
    spec: dict[str, t.Any]
    metric: t.Any = None
    """The ``metrics.Metric`` it defines, if it reads as one."""
    failure: t.Any = None
    """Why it does not run on its source, a ``metrics.MetricFailure``."""
    problems: list[str] | None = None
    """Why the section, with this metric in it, does not load."""

    @property
    def check(self) -> str:
        """``valid``, or what is wrong: ``source``, ``table``, ``column``, ``query`` or ``invalid``."""
        if self.problems:
            return "invalid"
        if self.failure is not None:
            return self.failure.kind
        return "valid"

    @property
    def message(self) -> str:
        """Why it does not run, or nothing when it does."""
        if self.problems:
            return " ".join(self.problems)
        return self.failure.message if self.failure is not None else ""

    @property
    def description(self) -> str:
        value = self.spec.get("description")
        return value.strip() if isinstance(value, str) else ""


def metric_statuses(project: t.Any) -> tuple[list[MetricStatus], list[str]]:
    """
    Every metric of ``project``'s ``metrics:`` section as the file has it
    now, with whether the app uses it, and the section's problems that
    concern no one metric. It reads what the app's ``ProjectMetrics`` (in
    the project's context) found, so a page listing them runs no query;
    without one it checks them, which queries their sources.
    """
    from .metrics import METRICS_KEY, MetricSet, _Reader
    from .metrics_edit import load_metrics, masked, raw_section
    from .project import ProjectError, Section

    try:
        section = raw_section(project)
    except ProjectError as exc:
        return [], [p.message for p in masked(exc.problems)]
    current = (project.context or {}).get(METRICS_KEY)
    if current is not None and hasattr(current, "snapshot"):
        snapshot = current.snapshot()
        problems = list(getattr(current, "problems", []))
    else:
        snapshot, problems = load_metrics(project, section)
        problems = [p for p in problems if not (p.keys and p.keys[0] in snapshot.failures)]
    by_metric: dict[str | None, list[str]] = {}
    for problem in masked(problems):
        keys = tuple(problem.keys or ())
        name = keys[0] if keys and keys[0] in section else None
        by_metric.setdefault(name, []).append(problem.message)
    try:
        parsed: t.Mapping[str, t.Any] = _Reader(Section("metrics", section, project)).read()
    except ProjectError:
        parsed = MetricSet()
    failures = getattr(snapshot, "failures", {})
    statuses = []
    for name, spec in section.items():
        failure = failures.get(name)
        metric = parsed.get(name) or (failure.metric if failure is not None else None)
        statuses.append(MetricStatus(
            str(name), spec if isinstance(spec, dict) else {}, metric, failure, by_metric.get(name),
        ))
        # A metric the snapshot neither uses nor knows failing was not read yet, e.g. after a section problem.
        if not by_metric.get(name) and failure is None and name not in snapshot and problems:
            statuses[-1].problems = ["The project file's metrics have problems, so this one is not used yet."]
    return statuses, by_metric.get(None, [])
