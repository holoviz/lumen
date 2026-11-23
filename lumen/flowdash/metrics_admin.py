"""
The Metrics section of panel-flowdash's admin view and its setup wizard
step: the agreed definitions of the app's key numbers, which dashboards
and the assistant use instead of writing their own SQL.

It edits the project file's ``metrics:`` section (see
``lumen.flowdash.metrics``). The table lists each metric; its editor has a
form for the name, description, source and table and the rest of the
definition as YAML, checked by the same loader the app reads the file
with, every problem shown at its field. The preview runs the metric on its
source, read-only, capped at ``PREVIEW_ROWS`` rows and ``PREVIEW_TIMEOUT``
seconds. Saving and deleting check again that the caller is an admin and
write the section with ``Project.set``, refusing an invalid definition and
a metric another admin changed since the editor opened.
"""
from __future__ import annotations

import copy
import logging
import math
import typing as t

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import panel as pn
import panel_material_ui as pmui
import param

from panel.util import edit_readonly
from panel.viewable import Viewer
from panel_flowdash.admin.form import current_app
from panel_flowdash.admin.kit import (
    Table, caption, cell, chip, esc, heading,
)
from panel_flowdash.admin.wizard import WizardStep
from panel_flowdash.project import (
    _FILE_LOCK, Problem, Project, ProjectError, Section, _dump, _parse, _read,
)
from panel_flowdash.value_text import error_text
from ruamel.yaml.error import YAMLError
from ruamel.yaml.scalarstring import LiteralScalarString

from ..sources.base import BaseSQLSource
from .contracts import METRIC_ID
from .metrics import (
    _NAME, SECTION, Metric, MetricSet, Query, _Reader, aggregate, bind_sql,
    check_metrics, coerce_dimension_values, prepare, table_sql,
)

logger = logging.getLogger(__name__)

HELP = "Numbers defined here are the agreed definitions; the assistant uses them instead of writing its own SQL."

# The rows a preview query returns at most, and how long it may run.
PREVIEW_ROWS = 10
PREVIEW_TIMEOUT = 5

# The keys of a metric the form edits; the YAML field holds the others.
FORM_KEYS = ("description", "source", "table")
# The pseudo key of a metric's name, for problems with the name itself.
NAME_KEY = "name"

NOT_ADMIN = "Only admins can change metrics, so this was not saved."

_UNSET = object()

_MENU_SX = {
    "minWidth": 0,
    "padding": "6px",
    "borderRadius": "50%",
    "color": "text.secondary",
    "& .MuiButton-startIcon": {"margin": 0},
    "& .MuiButton-endIcon": {"display": "none"},
}

_ROW_CSS = """
.k .metric { cursor: pointer; min-width: 0; }
.k .fields { display: flex; flex-wrap: wrap; gap: 2px 0; }
.k a { color: var(--mui-palette-primary-text, var(--mui-palette-primary-main)); text-decoration: none; }
"""

_PREVIEW_CSS = """
.k .result { margin: 0 0 12px 0; }
.k .label { font-size: 0.75rem; letter-spacing: 0.4px; color: var(--mui-palette-text-secondary);
  margin: 0 0 4px 0; }
.k .numbers { display: flex; flex-wrap: wrap; gap: 8px 24px; margin: 0 0 6px 0; }
.k .number small { display: block; font-size: 0.75rem; color: var(--mui-palette-text-secondary); }
.k .number span { font-size: 1.375rem; line-height: 1.3; font-variant-numeric: tabular-nums; }
.k table { border-collapse: collapse; width: 100%; font-size: 0.8125rem; margin: 0 0 6px 0; }
.k th { text-align: left; font-weight: 500; color: var(--mui-palette-text-secondary); }
.k th, .k td { padding: 4px 8px 4px 0; border-bottom: 1px solid var(--mui-palette-divider); }
.k td.n { text-align: right; font-variant-numeric: tabular-nums; }
.k pre { margin: 0; padding: 8px 10px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap;
  font-family: "Roboto Mono", ui-monospace, monospace; font-size: 0.75rem; line-height: 1.5;
  background: rgba(var(--mui-palette-text-primaryChannel) / 0.06); color: var(--mui-palette-text-primary); }
"""

_MONO_SX = {
    "& textarea": {
        "fontFamily": '"Roboto Mono", ui-monospace, monospace',
        "fontSize": "0.8125rem",
        "lineHeight": 1.5,
    },
}

_OVERLINE_SX = {
    "color": "text.secondary",
    "fontSize": "0.75rem",
    "fontWeight": 500,
    "letterSpacing": "1.2px",
    "lineHeight": 2,
}


# ---------------------------------------------------------------------------
# Reading, checking and writing the section


def raw_section(project: Project) -> dict[str, t.Any]:
    """The ``metrics:`` section as plain YAML data, as the project file has it now."""
    path = Path(project.path) if project.path else None
    if path is not None and path.is_file():
        data, _ = _parse(_read(path.resolve()), project.path)
    else:
        data = project.data
    value = data.get(SECTION)
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _replaced(current: Mapping[str, t.Any], original: str | None, name: str, spec: t.Any) -> dict[str, t.Any]:
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
    try:
        kept, warnings = check_metrics(metrics, project.context)
    except ProjectError as exc:
        return metrics, list(exc.problems)
    return kept, warnings


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
    return load_metrics(project, _replaced(current, original, name, spec))[1]


def _admin(app: t.Any) -> bool:
    try:
        return app is not None and bool(app._still_admin())
    except Exception:
        logger.exception("Could not check whether this session is still an admin.")
        return False


def _who(app: t.Any) -> str:
    return getattr(getattr(app, "_identity", None), "user", None) or "someone"


def _write(project: Project, section: dict[str, t.Any]) -> list[Problem]:
    try:
        project.set(SECTION, section)
    except ProjectError as exc:
        return list(exc.problems)
    except OSError as exc:
        return [Problem(f"Could not write the project file: {error_text(exc)}")]
    return []


def save_metric(
    project: Project, name: str, spec: dict[str, t.Any], *, app: t.Any, original: str | None = None,
    base: t.Any = _UNSET,
) -> list[Problem]:
    """
    Write ``spec`` as metric ``name`` of the project file, in place of
    ``original`` when it renames or edits one, keeping every other metric as
    the file has it now. ``base`` is the definition the editor opened, so
    a metric another admin changed or deleted since is not overwritten.
    Only an admin (``app._still_admin()``) may save. Returns the problems
    that stopped it; none when it was written.
    """
    if not _admin(app):
        return [Problem(NOT_ADMIN)]
    with _FILE_LOCK:
        try:
            current = raw_section(project)
        except ProjectError as exc:
            return list(exc.problems)
        if original is not None and base is not _UNSET:
            if original not in current:
                return [_problem(name, (), f"Metric {original!r} was deleted since you opened it; "
                                 "save it as a new metric to keep it.")]
            if current[original] != base:
                return [_problem(name, (), f"Metric {original!r} was changed since you opened it. "
                                 "Close the editor and open it again to see the change.")]
        problems = check_metric(project, name, spec, original, current)
        if problems:
            return problems
        problems = _write(project, _replaced(current, original, name, spec))
    if not problems:
        logger.info("Admin: %s saved metric %r.", _who(app), name)
    return problems


def delete_metric(project: Project, name: str, *, app: t.Any, base: t.Any = _UNSET) -> list[Problem]:
    """Delete metric ``name`` from the project file; only an admin may. Returns the problems that stopped it."""
    if not _admin(app):
        return [Problem(NOT_ADMIN.replace("saved", "deleted"))]
    with _FILE_LOCK:
        try:
            current = raw_section(project)
        except ProjectError as exc:
            return list(exc.problems)
        if name not in current:
            return [_problem(name, (), f"Metric {name!r} is not in the project file any more.")]
        if base is not _UNSET and current[name] != base:
            return [_problem(name, (), f"Metric {name!r} was changed since you opened it; look again before "
                             "deleting it.")]
        rest = {key: value for key, value in current.items() if key != name}
        problems = _write(project, rest)
    if not problems:
        logger.info("Admin: %s deleted metric %r.", _who(app), name)
    return problems


@dataclass(frozen=True)
class DashboardRef:
    dashboard_id: str
    title: str


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


# ---------------------------------------------------------------------------
# Splitting a definition into the form and the YAML


def split_spec(spec: Mapping[str, t.Any]) -> tuple[dict[str, t.Any], str]:
    """The form's values of a metric's definition, and the rest of it as YAML text."""
    form = {key: spec.get(key) for key in FORM_KEYS}
    rest = {key: value for key, value in spec.items() if key not in FORM_KEYS}
    return form, (_dump(_literal(rest)) if rest else "")


def join_spec(form: Mapping[str, t.Any], text: str) -> tuple[dict[str, t.Any], list[Problem], dict]:
    """
    The definition the form's values and the YAML ``text`` make, the
    problems with the YAML itself, and where each of its values is written
    (``positions`` by path, lines counted from 1).
    """
    spec: dict[str, t.Any] = {}
    for key in FORM_KEYS:
        value = form.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value:
            spec[key] = value
    if not text.strip():
        return spec, [], {}
    try:
        data, positions = _parse(text, None)
    except ProjectError as exc:
        return spec, [Problem(f"Line {p.line}: {p.message}" if p.line else p.message, section=SECTION,
                              keys=("", "yaml")) for p in exc.problems], {}
    except YAMLError as exc:
        return spec, [Problem(f"Not valid YAML: {exc}", section=SECTION, keys=("", "yaml"))], {}
    problems = []
    for key in data:
        if key in FORM_KEYS:
            problems.append(Problem(f"Set `{key}` in the form above, not in the YAML.", section=SECTION,
                                    keys=("", "yaml")))
    spec.update({key: value for key, value in data.items() if key not in FORM_KEYS})
    return spec, problems, positions


def template(columns: Sequence[str] = (), sql_table: str | None = None) -> str:
    """
    The YAML a new metric starts from: a row count, and a dimension on the
    first of ``columns`` commented out, or a query of ``sql_table``.
    """
    if sql_table is not None:
        return f"sql: |\n  SELECT COUNT(*) AS rows\n  FROM {sql_table}\n"
    text = "measures:\n  rows: COUNT(*)\n"
    if columns:
        text += f"# dimensions:\n#   {columns[0]}: {columns[0]}\n"
    return text


def _set_text(widget: t.Any, text: str) -> None:
    """Show ``text`` in a text field, replacing what the browser last reported typing too."""
    with edit_readonly(widget):
        widget.param.update(value=text, value_input=text)


def _literal(value: t.Any) -> t.Any:
    if isinstance(value, dict):
        return {key: _literal(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_literal(item) for item in value]
    if isinstance(value, str) and "\n" in value.strip():
        return LiteralScalarString(value if value.endswith("\n") else value + "\n")
    return value


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


def _safe(exc: BaseException) -> str:
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
        return list(exc.problems), []
    source = lookup_source(metric.source, project.context)
    if source is None:
        from .nodes import host_sources

        refs = ", ".join(sorted(host_sources(project.context))) or "none"
        return [_problem(name, ("source",), f"The app serves no source {metric.source!r}; its sources are: {refs}.")], []
    if not isinstance(source, BaseSQLSource):
        return [_problem(name, ("source",), f"{type(source).__name__} does not run SQL.")], []
    dialect = None if source.dialect == "any" else source.dialect
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
        result = PreviewResult(label)
        try:
            query, values = prepare(metric, query)
            values = coerce_dimension_values(metric, values, columns)
            transform = aggregate(metric, query, values, source.dialect)
            transform.columns = list(columns)
            _run(source, result, transform.apply(table_sql(metric.table, dialect)), limit, timeout)
        except Exception as exc:
            result.error = _safe(exc)
        results.append(result)
    return [], results


def _number(value: t.Any) -> str:
    if isinstance(value, bool) or value is None:
        return "-" if value is None else str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    return str(value)


def preview_html(results: Sequence[PreviewResult]) -> str:
    """The preview's results as escaped HTML: one row as numbers, more as a table, and each query's SQL."""
    parts = []
    for result in results:
        body = f'<div class="label">{esc(result.label)}</div>'
        if result.error is not None:
            body += f'<div class="error-text">{esc(result.error)}</div>'
        elif len(result.rows) == 1 and all(isinstance(v, (int, float)) for v in result.rows[0]):
            body += '<div class="numbers">' + "".join(
                f'<div class="number"><small>{esc(c)}</small><span>{esc(_number(v))}</span></div>'
                for c, v in zip(result.columns, result.rows[0], strict=False)
            ) + "</div>"
        elif not result.rows:
            body += '<div class="muted">No rows.</div>'
        else:
            head = "".join(f"<th>{esc(c)}</th>" for c in result.columns)
            rows = "".join(
                "<tr>" + "".join(
                    f'<td class="{"n" if isinstance(v, (int, float)) and not isinstance(v, bool) else ""}">'
                    f"{esc(_number(v))}</td>" for v in row
                ) + "</tr>"
                for row in result.rows
            )
            body += f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
            if len(result.rows) >= PREVIEW_ROWS:
                body += f'<div class="muted small">The first {PREVIEW_ROWS} rows.</div>'
        if result.sql:
            body += f"<pre>{esc(result.sql)}</pre>"
        parts.append(f'<div class="result">{body}</div>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# The editor: the form, the YAML and the preview


def _fields(spec: Mapping[str, t.Any]) -> tuple[list[str], list[str]]:
    measures = spec.get("measures")
    dimensions = spec.get("dimensions")
    return (
        [str(m) for m in measures] if isinstance(measures, dict) else [],
        [str(d) for d in dimensions] if isinstance(dimensions, dict) else [],
    )


class MetricEditor(Viewer):
    """
    The form of one metric: its name, description, source and table, the
    rest of its definition as YAML, the problems at their fields and a
    preview. ``spec`` makes the definition; it is checked with ``check``.
    """

    def __init__(self, project: Project, **params):
        super().__init__(**params)
        self.project = project
        self._name = pmui.TextInput(
            label="Name", placeholder="capacity", sizing_mode="stretch_width", margin=0,
            css_classes=["flowdash-metric-name"],
        )
        self._description = pmui.TextInput(
            label="Description", placeholder="What the number means and how it is counted",
            sizing_mode="stretch_width", margin=0, css_classes=["flowdash-metric-description"],
        )
        self._source = pmui.Select(
            label="Source", options=self._sources(), sizing_mode="stretch_width", margin=0,
            css_classes=["flowdash-metric-source"],
        )
        self._table = pmui.Select(
            label="Table", options={}, sizing_mode="stretch_width", margin=0,
            css_classes=["flowdash-metric-table"],
        )
        self._yaml = pmui.TextAreaInput(
            label="Measures, dimensions and parameters (YAML)", rows=9, max_rows=24, auto_grow=True,
            sizing_mode="stretch_width", margin=0, sx=_MONO_SX, css_classes=["flowdash-metric-yaml"],
            helper_text="Or a `sql:` query with :name parameters in place of a table.",
        )
        self._source.param.watch(lambda _event: self._sync_tables(), "value")
        self._table.param.watch(lambda _event: self._on_table(), "value")
        self._dimension = pmui.Select(
            label="By", options=[], width=200, margin=0, visible=False,
            css_classes=["flowdash-metric-preview-by"],
        )
        self._run = pmui.Button(
            label="Run preview", icon="play_arrow", variant="outlined", margin=0,
            css_classes=["flowdash-metric-preview-run"],
        )
        self._run.on_click(lambda _event: self.preview())
        self._preview = cell(
            '<span class="muted">Run the preview to see the numbers this definition gives.</span>',
            stylesheets=[_PREVIEW_CSS], css_classes=["flowdash-metric-preview"],
        )
        self._template = template()
        self._layout = pn.Column(
            self._name,
            self._description,
            pn.Row(
                self._source, self._table, margin=0, sizing_mode="stretch_width",
                styles={"gap": "12px", "flex-wrap": "wrap"},
            ),
            self._yaml,
            pmui.Typography("Preview", variant="overline", sx=_OVERLINE_SX, margin=(8, 0, 0, 0)),
            caption(f"Runs read-only on the source, at most {PREVIEW_ROWS} rows in {PREVIEW_TIMEOUT} seconds."),
            pn.Row(
                self._dimension, self._run, margin=0,
                styles={"gap": "12px", "align-items": "center", "flex-wrap": "wrap"},
            ),
            self._preview,
            margin=0,
            sizing_mode="stretch_width",
            styles={"gap": "14px"},
            css_classes=["flowdash-metric-editor"],
        )
        self.load()

    def _context(self) -> Mapping[str, t.Any]:
        return self.project.context or {}

    def _sources(self) -> list[str]:
        from .nodes import host_sources

        return sorted(host_sources(self._context()))

    def _tables(self, ref: str | None) -> list[str]:
        from .nodes import lookup_source

        source = lookup_source(ref, self._context()) if ref else None
        if source is None:
            return []
        try:
            return sorted(source.get_tables())
        except Exception:
            logger.exception("Could not list the tables of source %r", ref)
            return []

    def _columns(self, ref: str | None, table: str | None) -> list[str]:
        from .nodes import lookup_source

        source = lookup_source(ref, self._context()) if ref and table else None
        if source is None:
            return []
        try:
            schema = {str(c): spec for c, spec in source.get_schema(table).items() if c != "__len__"}
        except Exception:
            return []
        # Text columns first: the template suggests the first column as a dimension.
        return sorted(schema, key=lambda c: (schema[c] or {}).get("type") != "string")

    def _sync_tables(self, keep: str | None = None):
        tables = self._tables(self._source.value)
        options = {"No table: a SQL query": "", **{name: name for name in tables}}
        if keep and keep not in tables:
            options[keep] = keep
        value = keep if keep is not None else (tables[0] if tables else "")
        self._table.param.update(options=options, value=value)

    def _on_table(self):
        if self._template is not None and self._text() in ("", self._template):
            self._template = self._template_for(self._source.value, self._table.value)
            _set_text(self._yaml, self._template)

    def _template_for(self, ref: str | None, table: str | None) -> str:
        if table == "":
            tables = self._tables(ref)
            return template(sql_table=tables[0] if tables else "my_table")
        return template(self._columns(ref, table))

    def load(self, name: str = "", spec: Mapping[str, t.Any] | None = None):
        """Show metric ``name`` defined as ``spec``, or a blank one."""
        spec = dict(spec or {})
        form, text = split_spec(spec)
        sources = self._sources()
        source = form.get("source") or (sources[0] if sources else None)
        options = list(dict.fromkeys([*sources, *([source] if source else [])]))
        with pn.io.hold():
            _set_text(self._name, name)
            _set_text(self._description, form.get("description") or "")
            self._source.param.update(options=options, value=source)
            # The template is set last, so changing the table above does not replace the text.
            _set_text(self._yaml, "")
            self._template = None
            self._sync_tables(keep=(form.get("table") or "") if spec else None)
            self._template = self._template_for(self._source.value, self._table.value)
            _set_text(self._yaml, text if spec else self._template)
            self._preview.object = (
                '<div class="k"><span class="muted">Run the preview to see the numbers this '
                "definition gives.</span></div>"
            )
            self._dimension.visible = False
            self.show_problems([])

    @property
    def metric_name(self) -> str:
        return (self._name.value_input if self._name.value_input is not None else self._name.value or "").strip()

    def _form(self) -> dict[str, t.Any]:
        description = self._description.value_input
        return {
            "description": description if description is not None else self._description.value,
            "source": self._source.value,
            "table": self._table.value,
        }

    def _text(self) -> str:
        value = self._yaml.value_input
        return value if value is not None else (self._yaml.value or "")

    def spec(self) -> tuple[dict[str, t.Any], list[Problem]]:
        """The definition the form makes, and the problems with its YAML."""
        spec, problems, self._positions = join_spec(self._form(), self._text())
        return spec, problems

    def is_blank(self) -> bool:
        """Whether nothing was entered beyond the template."""
        return not self.metric_name and self._text() in ("", self._template) and not self._form()["description"]

    def show_problems(self, problems: Sequence[Problem]) -> list[str]:
        """Show ``problems`` at the fields they concern; return the ones no field shows."""
        by_field: dict[str, list[str]] = {}
        general: list[str] = []
        positions = getattr(self, "_positions", {}) or {}
        for problem in problems:
            keys = tuple(problem.keys)
            if problem.section not in (None, SECTION) or not keys:
                general.append(problem.message)
                continue
            rest = keys[1:]
            if rest[:1] == (NAME_KEY,):
                by_field.setdefault("name", []).append(problem.message)
            elif rest[:1] == ("yaml",):
                by_field.setdefault("yaml", []).append(problem.message)
            elif rest and rest[0] in FORM_KEYS:
                by_field.setdefault(rest[0], []).append(problem.message)
            elif keys[0] in ("", self.metric_name) and rest:
                line = next((positions[rest[:i]][0] for i in range(len(rest), 0, -1) if rest[:i] in positions), None)
                where = f"Line {line}: " if line else ""
                by_field.setdefault("yaml", []).append(f"{where}{problem.message}")
            elif keys[0] in ("", self.metric_name):
                by_field.setdefault("yaml" if self._text().strip() else "name", []).append(problem.message)
            else:
                general.append(problem.message)
        widgets = {
            "name": self._name, "description": self._description, "source": self._source,
            "table": self._table, "yaml": self._yaml,
        }
        defaults = {"yaml": "Or a `sql:` query with :name parameters in place of a table."}
        with pn.io.hold():
            for key, widget in widgets.items():
                messages = by_field.get(key, [])
                widget.param.update(
                    error_state=bool(messages), helper_text=" ".join(messages) or defaults.get(key, ""),
                )
        return general

    def preview(self) -> list[PreviewResult]:
        """Run the preview of the form's definition and show it."""
        spec, problems = self.spec()
        if problems:
            self.show_problems(problems)
            return []
        problems, results = preview_metric(self.project, self.metric_name or "preview", spec, self._dimension.value)
        general = self.show_problems(problems)
        _, dimensions = _fields(spec)
        with pn.io.hold():
            self._dimension.param.update(
                options=dimensions, visible=bool(dimensions) and "sql" not in spec,
                value=self._dimension.value if self._dimension.value in dimensions else (
                    dimensions[0] if dimensions else None),
            )
            if problems:
                self._preview.object = (
                    '<div class="k"><span class="error-text">Fix the definition to preview it'
                    + (f": {esc(' '.join(general))}" if general else ".") + "</span></div>"
                )
            else:
                self._preview.object = f'<div class="k">{preview_html(results)}</div>'
        return results

    def __panel__(self):
        return self._layout


# ---------------------------------------------------------------------------
# The admin section

# Below this width the table shows only a metric, where it is used and its menu.
_NARROW = 760

_NARROW_CSS = f"""
@media (max-width: {_NARROW}px) {{
  :host > div:nth-of-type(2), :host > div:nth-of-type(3), :host > div:nth-of-type(4) {{ display: none !important; }}
}}
"""


class _MetricsTable(Table):
    """The kit's table, dropping the source, measures and dimensions columns at phone width."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        header = self._layout.objects[1]
        header.stylesheets = [*header.stylesheets, _NARROW_CSS]

    def set_rows(self, rows):
        super().set_rows(rows)
        if rows:
            for row in self._rows:
                row.stylesheets = [*row.stylesheets, _NARROW_CSS]


class MetricsSettings(Viewer):
    """The Metrics section: the table of the project's metrics and a drawer that edits one."""

    editable = param.Boolean(default=True)

    selected = param.String(default=None, allow_None=True, doc="The metric the drawer edits; None for a new one.")

    def __init__(self, project: Project, editable: bool = True, **params):
        super().__init__(editable=editable, **params)
        self.project = project
        self._app = current_app()
        self._base: t.Any = _UNSET
        self._alert = pmui.Alert(
            object="", severity="error", visible=False, sizing_mode="stretch_width", margin=0
        )
        self._new = pmui.Button(
            label="New metric", icon="add", variant="contained", margin=0,
            disabled=not editable, css_classes=["flowdash-metrics-new"],
        )
        self._new.on_click(lambda _event: self.open_metric(None))
        self._count = caption("", styles={"flex": "1 1 160px", "min-width": "0"})
        toolbar = pn.Row(
            self._count,
            self._new,
            margin=0,
            sizing_mode="stretch_width",
            styles={
                "padding": "10px 12px", "align-items": "center", "gap": "12px", "flex-wrap": "wrap",
                "border-bottom": "1px solid var(--mui-palette-divider)",
            },
        )
        self._table = _MetricsTable(
            [
                ("Metric", "2.2fr"),
                ("Source", "1.1fr"),
                ("Measures", "1.4fr"),
                ("Dimensions", "1.4fr"),
                ("Used in", "0.7fr"),
                ("", "40px"),
            ],
            toolbar=toolbar,
            footer=caption(
                "Dashboards and the assistant read a metric by its name, so renaming or deleting one "
                "breaks the dashboards that use it.",
                margin=(12, 12),
            ),
            empty="No metrics yet. Define the numbers people ask for most, e.g. installed capacity.",
            css_classes=["flowdash-admin-metrics"],
        )
        self._editor = MetricEditor(project)
        self._build_drawer()
        self._build_dialog()
        self._layout = pn.Column(
            self._alert,
            self._table,
            self._drawer,
            self._dialog,
            margin=0,
            sizing_mode="stretch_width",
            css_classes=["flowdash-admin-metrics-section"],
        )
        self.refresh()

    # ------------------------------------------------------------------
    # Data

    def section(self) -> dict[str, t.Any]:
        """The ``metrics:`` section as the file has it now."""
        try:
            return raw_section(self.project)
        except ProjectError as exc:
            self._show_problems(exc.problems)
            return {}

    def usage(self) -> dict[str, list[DashboardRef]]:
        store = getattr(self._app, "store", None)
        if store is None:
            return {}
        try:
            return metric_usage(store._all_dashboards())
        except Exception:
            logger.exception("Could not read the dashboards that use the metrics.")
            return {}

    def _show_problems(self, problems: Sequence[Problem]):
        self._alert.param.update(object="\n\n".join(p.message for p in problems), visible=bool(problems))

    def _notify(self, message: str):
        if self._app is not None:
            self._app._notify("success", message)

    # ------------------------------------------------------------------
    # The table

    def refresh(self):
        """Draw the table from the project file."""
        section = self.section()
        usage = self.usage()
        where = Path(self.project.path).name if self.project.path else "the project file"
        n = len(section)
        self._count.object = f"{n} metric{'' if n == 1 else 's'} in {where}"
        self._table.set_rows([(name, self._cells(name, spec, usage.get(name, []))) for name, spec in section.items()])

    def _cells(self, name: str, spec: t.Any, used: Sequence[DashboardRef]) -> list:
        spec = spec if isinstance(spec, dict) else {}
        measures, dimensions = _fields(spec)
        description = spec.get("description") if isinstance(spec.get("description"), str) else ""
        title = cell(
            f'<div class="metric"><span class="main">{esc(name)}</span>'
            f'<span class="main muted small">{esc(description) or "&nbsp;"}</span></div>',
            stylesheets=[_ROW_CSS],
        )
        open_editor = pmui.Clickable(object=title, sizing_mode="stretch_width", margin=0)
        open_editor.on_click(lambda *_args, name=name: self.open_metric(name))
        where = spec.get("table") if "table" in spec else "SQL query"
        source = cell(
            f'<span class="main">{esc(spec.get("source", ""))}</span>'
            f'<span class="mono">{esc(where or "")}</span>'
        )
        none = '<span class="muted">None</span>'
        fields = lambda names: f'<div class="fields">{"".join(chip(n) for n in names)}</div>' if names else none
        used_text = (
            f'<span class="main" title="{esc(", ".join(r.title for r in used))}">{len(used)}</span>'
            if used else '<span class="main">-</span>'
        )
        menu = pmui.MenuButton(
            items=[
                {"label": "Edit", "icon": "edit", "action": "edit"},
                {"label": "Duplicate", "icon": "content_copy", "action": "duplicate",
                 "disabled": not self.editable},
                {"label": "Delete", "icon": "delete", "action": "delete", "disabled": not self.editable},
            ],
            icon="more_vert", label="", variant="text", color="default", size="small", icon_size="20px",
            margin=0, sx=_MENU_SX, description=f"More actions for {name}",
            css_classes=["flowdash-admin-metric-menu"],
        )
        menu.on_click(lambda item, name=name: self.run(item.get("action"), name))
        sql_note = '<span class="muted">Its query\'s columns</span>'
        return [
            open_editor,
            source,
            cell(fields(measures) if "sql" not in spec else sql_note),
            cell(fields(dimensions) if "sql" not in spec else none),
            cell(used_text),
            menu,
        ]

    def run(self, action: str | None, name: str):
        """Carry out the row menu's ``action`` on metric ``name``."""
        if action == "edit":
            self.open_metric(name)
        elif action == "duplicate":
            self.duplicate(name)
        elif action == "delete":
            self.ask_delete(name)

    # ------------------------------------------------------------------
    # The drawer

    def _build_drawer(self):
        self._drawer_title = pmui.Typography("", variant="h6", margin=0, sx={"fontWeight": 400})
        self._drawer_subtitle = caption(HELP)
        close = pmui.IconButton(icon="close", description="Close", size="small", margin=0)
        close.on_click(lambda _event: self.close_metric())
        self._drawer_alert = pmui.Alert(
            object="", severity="error", visible=False, sizing_mode="stretch_width", margin=0
        )
        cancel = pmui.Button(label="Cancel", variant="outlined", color="default", margin=0)
        cancel.on_click(lambda _event: self.close_metric())
        self._save = pmui.Button(
            label="Save", icon="save", variant="contained", margin=0, disabled=not self.editable,
            css_classes=["flowdash-metric-save"],
        )
        self._save.on_click(lambda _event: self.save())
        head = pn.Row(
            pn.Column(self._drawer_title, self._drawer_subtitle, margin=0, sizing_mode="stretch_width"),
            close,
            margin=0,
            sizing_mode="stretch_width",
            styles={
                "align-items": "flex-start", "gap": "8px", "padding": "20px 20px 12px 20px",
                "box-sizing": "border-box", "flex": "0 0 auto",
            },
        )
        fields = pn.Column(
            self._drawer_alert,
            self._editor,
            margin=0,
            sizing_mode="stretch_width",
            styles={
                "gap": "12px", "padding": "8px 20px 20px 20px", "overflow-y": "auto", "flex": "1 1 auto",
                "min-height": "0", "box-sizing": "border-box",
            },
        )
        foot = pn.Row(
            pn.layout.HSpacer(),
            cancel,
            self._save,
            margin=0,
            sizing_mode="stretch_width",
            styles={
                "gap": "8px", "border-top": "1px solid var(--mui-palette-divider)", "padding": "16px 20px",
                "box-sizing": "border-box", "flex": "0 0 auto",
            },
        )
        body = pn.Column(
            head, fields, foot,
            margin=0,
            sizing_mode="stretch_width",
            styles={"height": "100vh", "display": "flex", "flex-direction": "column", "box-sizing": "border-box"},
            css_classes=["flowdash-admin-metric-drawer"],
        )
        self._drawer = pmui.Drawer(
            body, anchor="right", variant="temporary", size=600, open=False,
            sx={"maxWidth": "100vw"},
        )

    def open_metric(self, name: str | None, spec: Mapping[str, t.Any] | None = None, title: str | None = None):
        """Open the editor on metric ``name``, or on a new one (defined as ``spec``, if given)."""
        section = self.section()
        if name is not None and name not in section:
            return
        self.selected = name
        self._base = copy.deepcopy(section[name]) if name is not None else _UNSET
        with pn.io.hold():
            self._drawer_alert.visible = False
            self._drawer_title.object = title or ("New metric" if name is None else f"Edit {name}")
            self._editor.load(name or "", section[name] if name is not None else spec)
            self._drawer.open = True

    def close_metric(self):
        """Close the editor without saving."""
        self._drawer.open = False
        self.selected = None
        self._base = _UNSET

    def save(self) -> bool:
        """Save the editor's metric; show the problems at their fields if it is refused."""
        spec, problems = self._editor.spec()
        name = self._editor.metric_name
        if not problems:
            problems = save_metric(
                self.project, name, spec, app=self._app if self.editable else None,
                original=self.selected, base=self._base,
            )
        if problems:
            general = self._editor.show_problems(problems)
            self._drawer_alert.param.update(
                object="\n\n".join(general) if general else "Fix the fields marked below, then save again.",
                visible=True,
            )
            return False
        self.close_metric()
        self.refresh()
        self._notify(f"Metric {name} saved.")
        return True

    def duplicate(self, name: str):
        """Open the editor on a new metric defined as ``name`` is, named after it."""
        section = self.section()
        if name not in section:
            return
        copy_name, n = f"{name}_copy", 1
        while copy_name in section:
            n += 1
            copy_name = f"{name}_copy_{n}"
        self.open_metric(None, copy.deepcopy(section[name]), title=f"Duplicate {name}")
        self._editor._name.value = copy_name

    # ------------------------------------------------------------------
    # Deleting

    def _build_dialog(self):
        self._pending: tuple[str, t.Any] | None = None
        self._dialog_text = cell("", stylesheets=[_ROW_CSS])
        self._dialog_alert = pmui.Alert(
            object="", severity="error", visible=False, sizing_mode="stretch_width", margin=0
        )
        cancel = pmui.Button(label="Cancel", variant="text", color="default", margin=0)
        cancel.on_click(lambda _event: self.cancel_delete())
        self._confirm = pmui.Button(label="Delete metric", variant="contained", color="error", margin=0)
        self._confirm.on_click(lambda _event: self.confirm_delete())
        self._dialog = pmui.Dialog(
            pn.Column(self._dialog_alert, self._dialog_text, margin=0, sizing_mode="stretch_width",
                      styles={"gap": "12px"}),
            actions=[cancel, self._confirm],
            title="",
            open=False,
            close_on_click=True,
            css_classes=["flowdash-admin-confirm"],
        )

    def ask_delete(self, name: str):
        """Ask before deleting ``name``, listing the dashboards that use it."""
        section = self.section()
        if name not in section:
            return
        used = self.usage().get(name, [])
        self._pending = (name, copy.deepcopy(section[name]))
        if used:
            links = ", ".join(
                f'<a href="/dash/{esc(ref.dashboard_id)}">{esc(ref.title)}</a>' for ref in used[:10]
            ) + (f" and {len(used) - 10} more" if len(used) > 10 else "")
            n = len(used)
            text = (
                f'<p class="warn"><b>{n} dashboard{"" if n == 1 else "s"} use{"s" if n == 1 else ""} '
                f"{esc(name)}</b>: {links}. Their metric tiles will show an error until a metric of "
                "this name is defined again.</p>"
            )
        else:
            text = f"<p>No dashboard uses <b>{esc(name)}</b>. The assistant stops offering it.</p>"
        with pn.io.hold():
            self._dialog_alert.visible = False
            self._dialog_text.object = f'<div class="k">{text}<p>This can\'t be undone.</p></div>'
            self._dialog.param.update(title=f"Delete metric {name}?", open=True)

    def cancel_delete(self):
        self._pending = None
        self._dialog.open = False

    def confirm_delete(self) -> bool:
        """Delete the metric the dialog asks about."""
        if self._pending is None:
            return False
        name, base = self._pending
        problems = delete_metric(self.project, name, app=self._app if self.editable else None, base=base)
        if problems:
            self._dialog_alert.param.update(object="\n\n".join(p.message for p in problems), visible=True)
            return False
        self._pending = None
        self._dialog.open = False
        if self.selected == name:
            self.close_metric()
        self.refresh()
        self._notify(f"Metric {name} deleted.")
        return True

    def __panel__(self):
        return self._layout


def metrics_settings(project: Project, editable: bool) -> MetricsSettings:
    """Build the Metrics section of the admin view."""
    return MetricsSettings(project, editable)


# ---------------------------------------------------------------------------
# The setup wizard's step


class KeyNumbersStep(WizardStep):
    """
    The wizard's Key numbers step: the data the app serves, and a form for
    a first metric with its preview. It writes nothing until Review's
    Finish, and nothing at all when the form is left blank or skipped.
    """

    def __init__(self, project: Project, **params):
        params.setdefault("title", "Key numbers")
        params.setdefault("caption", "Agreed metrics")
        params.setdefault("heading", "Define your key numbers")
        params.setdefault(
            "description",
            f"{HELP} Start with the three to five numbers people ask for most; you can add more "
            "later in Admin, Metrics.",
        )
        params.setdefault("icon", "functions")
        super().__init__(project, **params)
        self.editor = MetricEditor(project)

    def _sources_html(self) -> str:
        from .nodes import host_sources

        items = []
        for ref, source in sorted(host_sources(self.project.context or {}).items()):
            try:
                tables = sorted(source.get_tables())
            except Exception:
                tables = []
            shown = ", ".join(f"<code>{esc(t)}</code>" for t in tables[:8]) + (" and more" if len(tables) > 8 else "")
            items.append(f"<li><b>{esc(ref)}</b>: {shown or '<span class=muted>no tables</span>'}</li>")
        if not items:
            return '<span class="muted">The app serves no data yet.</span>'
        return f'<ul style="margin:0;padding-left:18px">{"".join(items)}</ul>'

    def changes(self) -> dict[str, t.Any]:
        """The ``metrics:`` section with the form's metric added, or nothing for a blank form."""
        if self.editor.is_blank():
            return {}
        spec, problems = self.editor.spec()
        name = self.editor.metric_name
        if not problems:
            problems = check_metric(self.project, name, spec)
        if problems:
            self.editor.show_problems(problems)
            raise ValueError(" ".join(p.message for p in problems))
        return {SECTION: _replaced(raw_section(self.project), None, name, spec)}

    def summary(self) -> list[str]:
        if self.editor.is_blank():
            return ["No metric defined"]
        spec, _ = self.editor.spec()
        measures, dimensions = _fields(spec)
        if "sql" in spec:
            return [f"{self.editor.metric_name}: a SQL query on {spec.get('source', '')}"]
        return [f"{self.editor.metric_name}: {len(measures)} measure{'' if len(measures) == 1 else 's'}, "
                f"{len(dimensions)} dimension{'' if len(dimensions) == 1 else 's'}"]

    def view(self):
        return pn.Column(
            heading("Your data", margin=(0, 0, 4, 0)),
            cell(self._sources_html(), margin=(0, 0, 16, 0)),
            heading("A first metric", margin=(0, 0, 8, 0)),
            self.editor,
            margin=0,
            sizing_mode="stretch_width",
            styles={"max-width": "720px"},
            css_classes=["flowdash-wizard-metrics"],
        )


def metrics_wizard_step(project: Project) -> KeyNumbersStep:
    """Build the setup wizard's Key numbers step."""
    return KeyNumbersStep(project)
