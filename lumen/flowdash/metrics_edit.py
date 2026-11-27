"""
Reading, checking, previewing and writing the project file's ``metrics:``
section (see ``lumen.flowdash.metrics``), for any view that edits it: the
Data section, the admin view's Metrics on a panel-flowdash without sidebar
sections, the setup wizard and Save as metric.

``raw_section`` reads the section as the file has it now; ``check_metric``
says why a definition would be refused, by the field each problem
concerns, with the same loader the app reads the file with;
``preview_metric`` runs a definition read-only on its source, capped at
``PREVIEW_ROWS`` rows and ``PREVIEW_TIMEOUT`` seconds. ``save_metric`` and
``delete_metric`` check again that the caller is an admin and write the
section with ``Project.set``, refusing an invalid definition and a metric
another admin changed since it was read.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import typing as t

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml.scalarstring import LiteralScalarString

from ..sources.base import BaseSQLSource
from .access import (
    access_to_dict, is_admin, lockout_reason, normalize_access,
    session_identity,
)
from .metrics import (
    _NAME, SECTION, Metric, MetricSet, Query, _Reader, aggregate, bind_sql,
    check_metrics, coerce_dimension_values, prepare, table_sql,
)
from .project import (
    Problem, Project, ProjectError, Section,
)

logger = logging.getLogger(__name__)

# The rows a preview query returns at most, and how long it may run.
PREVIEW_ROWS = 10
PREVIEW_TIMEOUT = 5

# The pseudo key of a metric's name, for problems with the name itself.
NAME_KEY = "name"

NOT_ADMIN = "Only admins can change metrics, so this was not saved."
NOT_ADMIN_ACCESS = "Only admins can change access rules, so this was not saved."
NOT_ADMIN_PREVIEW = "Only admins can preview metrics."

# The definition a writer read before changing it, when it did not say.
UNSET: t.Any = object()


# ---------------------------------------------------------------------------
# Reading and checking the section


def raw_section(project: Project) -> dict[str, t.Any]:
    """The ``metrics:`` section as plain YAML data, as the project file has it now."""
    from panel_flowdash.project import _FILE_LOCK, _parse
    from panel_flowdash.value_text import error_text

    path = Path(project.path).resolve() if project.path else None
    if path is not None and path.is_file():
        with _FILE_LOCK:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ProjectError([Problem(f"Cannot read the project file: {error_text(exc)}", str(path))]) from None
        data, _ = _parse(text, project.path)
    else:
        data = project.data
    value = data.get(SECTION)
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def replace_metric(current: Mapping[str, t.Any], original: str | None, name: str, spec: t.Any) -> dict[str, t.Any]:
    """``current`` with ``original`` replaced by ``name``, in its place, or ``name`` added at the end."""
    result: dict[str, t.Any] = {}
    placed = False
    for key, value in current.items():
        if key == original:
            result[name] = spec
            placed = True
        elif key != name:
            result[key] = value
    if not placed:
        result[name] = spec
    return result


def _problem(name: str, keys: Sequence[str], message: str) -> Problem:
    return Problem(message, section=SECTION, keys=(name, *keys))


def masked(problems: Sequence[Problem]) -> list[Problem]:
    """``problems`` with any credentials an engine's error quotes masked, to show."""
    from panel_flowdash.value_text import redact_secrets

    return [dataclasses.replace(p, message=redact_secrets(p.message)) for p in problems]


def load_metrics(project: Project, section: Mapping[str, t.Any]) -> tuple[MetricSet, list[Problem]]:
    """
    ``section`` read as the app reads the ``metrics:`` section of
    ``project``, and checked against the project's sources when its context
    has them; a metric over a source the app does not serve is a problem
    here, not a warning, since saving it would leave it out.
    """
    from .nodes import SOURCES_KEY

    candidate = Project(project.path, {**project.data, SECTION: dict(section)}, {}, project.context)
    try:
        metrics = _Reader(Section(SECTION, dict(section), candidate)).read()
    except ProjectError as exc:
        return MetricSet(), list(exc.problems)
    if SOURCES_KEY not in project.context:
        return metrics, []
    return check_metrics(metrics, project.context)


def metric_problems(project: Project, section: Mapping[str, t.Any]) -> dict[str | None, list[str]]:
    """
    Why the app does not use each metric of ``section``, e.g. after a hand
    edit of the file, by metric name; None for problems of no one metric.
    Checking runs each metric's query on its source.
    """
    if not section:
        return {}
    _, problems = load_metrics(project, section)
    by_metric: dict[str | None, list[str]] = {}
    for problem in masked(problems):
        keys = tuple(problem.keys or ())
        name = keys[0] if problem.section in (None, SECTION) and keys and keys[0] in section else None
        by_metric.setdefault(name, []).append(problem.message)
    return by_metric


def check_metric(
    project: Project, name: str, spec: t.Any, original: str | None = None,
    current: Mapping[str, t.Any] | None = None,
) -> list[Problem]:
    """
    Why the ``metrics:`` section with ``spec`` written as metric ``name``
    (in place of ``original``) would be refused, by the field each problem
    concerns; none when it would be saved. ``current`` is the section as
    the file has it now, read when not given.
    """
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        return [_problem(str(name), (NAME_KEY,), "A metric's name is letters, digits and underscores, "
                         "starting with a letter, e.g. capacity_mw.")]
    if not isinstance(spec, dict):
        return [_problem(name, (), "A metric is a mapping of its settings.")]
    current = raw_section(project) if current is None else current
    if name != original and name in current:
        return [_problem(name, (NAME_KEY,), f"A metric named {name!r} exists already.")]
    metrics, problems = load_metrics(project, replace_metric(current, original, name, spec))
    # Another metric that does not run is left out on its own, so it does not stop this one.
    others = set(metrics.failures) - {name}
    return masked([p for p in problems if not (p.keys and p.keys[0] in others)])


def literal_blocks(value: t.Any) -> t.Any:
    """``value`` with its strings of several lines written as YAML literal blocks, e.g. a metric's SQL."""
    if isinstance(value, dict):
        return {key: literal_blocks(item) for key, item in value.items()}
    if isinstance(value, list):
        return [literal_blocks(item) for item in value]
    if isinstance(value, str) and "\n" in value.strip():
        return LiteralScalarString(value if value.endswith("\n") else value + "\n")
    return value


# ---------------------------------------------------------------------------
# Writing the section


def _who(app: t.Any) -> str:
    return getattr(getattr(app, "_identity", None), "user", None) or "someone"


def _write(project: Project, section: dict[str, t.Any]) -> list[Problem]:
    from panel_flowdash.value_text import error_text

    try:
        project.set(SECTION, section)
    except ProjectError as exc:
        return masked(exc.problems)
    except OSError as exc:
        return [Problem(f"Could not write the project file: {error_text(exc)}")]
    return []


def save_metric(
    project: Project, name: str, spec: dict[str, t.Any], *, app: t.Any, original: str | None = None,
    base: t.Any = UNSET,
) -> list[Problem]:
    """
    Write ``spec`` as metric ``name`` of the project file, in place of
    ``original`` when it renames or edits one, keeping every other metric as
    the file has it now. ``base`` is the definition the editor opened, so
    a metric another admin changed or deleted since is not overwritten.
    Only an admin (``app._still_admin()``) may save. Returns the problems
    that stopped it; none when it was written.
    """
    from panel_flowdash.project import _FILE_LOCK

    with _FILE_LOCK:
        if not is_admin(app):
            return [Problem(NOT_ADMIN)]
        try:
            current = raw_section(project)
        except ProjectError as exc:
            return masked(exc.problems)
        if original is not None and base is not UNSET:
            if original not in current:
                return [_problem(name, (), f"Metric {original!r} was deleted since you opened it; "
                                 "save it as a new metric to keep it.")]
            if current[original] != base:
                return [_problem(name, (), f"Metric {original!r} was changed since you opened it. "
                                 "Close the editor and open it again to see the change.")]
        problems = check_metric(project, name, spec, original, current)
        if problems:
            return problems
        if "access" in spec:
            perm, access_problems = normalize_access(spec.get("access"), section=SECTION, keys=(name, "access"))
            if access_problems:
                return access_problems
            if reason := lockout_reason(perm, session_identity(project.context or {}, app=app)):
                return [_problem(name, ("access",), reason)]
        problems = _write(project, replace_metric(current, original, name, spec))
    if not problems:
        logger.info("Admin: %s saved metric %r.", _who(app), name)
        from .nodes import metrics_changed

        metrics_changed()
    return problems


def delete_metric(project: Project, name: str, *, app: t.Any, base: t.Any = UNSET) -> list[Problem]:
    """Delete metric ``name`` from the project file; only an admin may. Returns the problems that stopped it."""
    from panel_flowdash.project import _FILE_LOCK

    with _FILE_LOCK:
        if not is_admin(app):
            return [Problem(NOT_ADMIN.replace("saved", "deleted"))]
        try:
            current = raw_section(project)
        except ProjectError as exc:
            return masked(exc.problems)
        if name not in current:
            return [_problem(name, (), f"Metric {name!r} is not in the project file any more.")]
        if base is not UNSET and current[name] != base:
            return [_problem(name, (), f"Metric {name!r} was changed since you opened it; look again before "
                             "deleting it.")]
        rest = {key: value for key, value in current.items() if key != name}
        problems = _write(project, rest)
    if not problems:
        logger.info("Admin: %s deleted metric %r.", _who(app), name)
        from .nodes import metrics_changed

        metrics_changed()
    return problems


def set_metric_access(
    project: Project,
    name: str,
    access: t.Any,
    *,
    app: t.Any,
    base: t.Any = UNSET,
) -> list[Problem]:
    """
    Set or clear the ``access:`` rule on metric ``name`` in the project file's
    ``metrics:`` section. Re-checks live admin rights and refuses any rule that
    would lock out the saving admin. ``base`` is the metric's ``access:`` value
    when the editor opened, so another admin's concurrent edit is not overwritten.
    """
    from panel_flowdash.project import _FILE_LOCK

    with _FILE_LOCK:
        if not is_admin(app):
            return [Problem(NOT_ADMIN_ACCESS)]
        perm, problems = normalize_access(access, section=SECTION, keys=(name, "access"))
        if problems:
            return problems
        identity = session_identity(project.context or {}, app=app)
        if reason := lockout_reason(perm, identity):
            return [_problem(name, ("access",), reason)]
        try:
            current = raw_section(project)
        except ProjectError as exc:
            return masked(exc.problems)
        if name not in current or not isinstance(current[name], dict):
            return [_problem(name, (), f"Metric {name!r} is not in the project file any more.")]
        if base is not UNSET and current[name].get("access") != base:
            return [_problem(name, ("access",), f"Access rules for metric {name!r} were changed since you opened "
                             "them. Reopen the editor to see the change.")]
        updated = copy.deepcopy(current[name])
        access_dict = access_to_dict(perm)
        if access_dict:
            updated["access"] = access_dict
        else:
            updated.pop("access", None)
        problems = _write(project, replace_metric(current, name, name, updated))
    if not problems:
        logger.info("Admin: %s updated access rules for metric %r.", _who(app), name)
        from .nodes import metrics_changed

        metrics_changed()
    return problems


# ---------------------------------------------------------------------------
# The preview


@dataclass
class PreviewResult:
    """One query of a preview: what it asks, its SQL and its rows, or why it failed."""

    label: str
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[t.Any]] = field(default_factory=list)
    error: str | None = None
    seconds: float | None = None


def _safe(exc: BaseException) -> str:
    from panel_flowdash.value_text import error_text

    text = error_text(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def _run(source: BaseSQLSource, result: PreviewResult, sql: str, limit: int, timeout: float) -> None:
    """Run ``sql`` read-only on ``source`` with ``limit`` rows at most, keeping the statement run in ``result``."""
    from .checks import check_query
    from .nodes import execute, result_sql

    check_query(sql, source)
    result.sql = result_sql(source, sql, limit)
    frame = execute(source, result.sql, timeout).head(limit)
    result.columns = [str(c) for c in frame.columns]
    result.rows = frame.astype(object).where(frame.notna(), None).values.tolist()


def preview_queries(metric: Metric, dimension: str | None = None) -> list[tuple[str, Query]]:
    """
    What the preview of a structured metric runs: every measure with no
    dimension, and every measure by ``dimension`` (its first one by
    default) at its coarsest grain, largest first.
    """
    measures = list(metric.measures)
    queries = [("All rows", Query(measures))]
    dimension = dimension if dimension in metric.dimensions else next(iter(metric.dimensions), None)
    if dimension is not None:
        spec = metric.dimensions[dimension]
        grain = spec.grains[0] if spec.time and spec.grains else None
        order = [dimension] if spec.time else [f"-{measures[0]}"]
        queries.append((f"By {dimension}", Query(measures, [dimension], grain=grain, order_by=order)))
    return queries


def preview_metric(
    project: Project, name: str, spec: t.Any, dimension: str | None = None, *,
    limit: int = PREVIEW_ROWS, timeout: float = PREVIEW_TIMEOUT,
) -> tuple[list[Problem], list[PreviewResult]]:
    """
    Run metric ``name`` defined as ``spec`` on its source as the app would,
    read-only, with at most ``limit`` rows in ``timeout`` seconds: a
    structured metric's measures alone and by one dimension, a SQL metric's
    query with its defaults. Returns the problems with the definition, or
    the results; an error a query hits is kept in its result, with any
    credentials masked.
    """
    from .nodes import lookup_source

    if not isinstance(name, str) or not _NAME.fullmatch(name):
        name = "preview"
    if not isinstance(spec, dict):
        return [_problem(name, (), "A metric is a mapping of its settings.")], []
    candidate = Project(project.path, {SECTION: {name: spec}}, {}, project.context)
    try:
        metric = _Reader(Section(SECTION, {name: spec}, candidate)).read()[name]
    except ProjectError as exc:
        return masked(exc.problems), []
    source = lookup_source(metric.source, project.context)
    if source is None:
        from .nodes import host_sources

        refs = ", ".join(sorted(host_sources(project.context))) or "none"
        return [_problem(name, ("source",), f"The app serves no source {metric.source!r}; its sources are: {refs}.")], []
    if not isinstance(source, BaseSQLSource):
        return [_problem(name, ("source",), f"{type(source).__name__} does not run SQL.")], []
    results = []
    if metric.form == "sql":
        result = PreviewResult("The query, with its defaults")
        try:
            _run(source, result, bind_sql(metric, metric.values({}), source.dialect), limit, timeout)
        except Exception as exc:
            result.error = _safe(exc)
        return [], [result]
    try:
        tables = source.get_tables()
    except Exception as exc:
        return [_problem(name, ("source",), f"Could not list the source's tables: {_safe(exc)}")], []
    if metric.table not in tables:
        return [_problem(name, ("table",), f"The source {metric.source!r} has no table {metric.table!r}.")], []
    columns = {str(c): s for c, s in source.get_schema(metric.table).items() if c != "__len__"}
    for label, query in preview_queries(metric, dimension):
        results.append(_run_query(metric, source, query, label, limit, timeout, columns))
    return [], results


def _run_query(
    metric: Metric, source: BaseSQLSource, query: Query, label: str, limit: int, timeout: float,
    columns: Mapping[str, t.Any] | None = None,
) -> PreviewResult:
    """``query`` of ``metric`` run read-only on ``source``, as a metric node would run it, at most ``limit`` rows."""
    dialect = None if source.dialect == "any" else source.dialect
    result = PreviewResult(label)
    try:
        query, values = prepare(metric, query)
        if metric.form == "sql":
            sql = bind_sql(metric, values, source.dialect)
        else:
            if columns is None:
                columns = {str(c): s for c, s in source.get_schema(metric.table).items() if c != "__len__"}
            values = coerce_dimension_values(metric, values, columns)
            transform = aggregate(metric, query, values, source.dialect)
            transform.columns = list(columns)
            sql = transform.apply(table_sql(metric.table, dialect))
        _run(source, result, sql, min(query.limit or limit, limit), timeout)
    except Exception as exc:
        result.error = _safe(exc)
    return result


def try_metric(
    metric: Metric, context: Mapping[str, t.Any], query: Query, *, limit: int = PREVIEW_ROWS,
    timeout: float = PREVIEW_TIMEOUT,
) -> PreviewResult:
    """
    Run ``query`` of ``metric``, a metric the app uses, on its source in
    ``context``, read-only, with at most ``limit`` rows in ``timeout``
    seconds, as the metric page's Try it does for anyone who can open the
    app: the values are checked and bound as a metric node binds them.
    The result keeps the statement run, or the error, credentials masked.
    """
    import time

    from .nodes import lookup_source

    source = lookup_source(metric.source, context)
    if not isinstance(source, BaseSQLSource):
        return PreviewResult("Try it", error=f"The app serves no SQL source {metric.source!r}.")
    start = time.monotonic()
    result = _run_query(metric, source, query, "Try it", limit, timeout)
    result.seconds = time.monotonic() - start
    if result.error:
        from panel_flowdash.value_text import redact_secrets

        result.error = redact_secrets(result.error)
    return result
