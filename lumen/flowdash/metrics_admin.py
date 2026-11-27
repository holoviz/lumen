"""
The metrics' view and the setup wizard's step: the agreed definitions of
the app's key numbers, which dashboards and the assistant use instead of
writing their own SQL. The Data section shows the view, and so does
panel-flowdash's admin view as its Metrics section where there are no
sidebar sections.

It edits the project file's ``metrics:`` section (see
``lumen.flowdash.metrics``) through ``lumen.flowdash.metrics_edit``. The
table lists each metric; its editor has a form for the name, description,
source and table and the rest of the definition as YAML, checked by the
same loader the app reads the file with, every problem shown at its field.
The preview runs the metric on its source, read-only, capped at
``PREVIEW_ROWS`` rows and ``PREVIEW_TIMEOUT`` seconds. Saving and deleting
check again that the caller is an admin and write the section with
``Project.set``, refusing an invalid definition and a metric another admin
changed since the editor opened. Someone who is not an admin sees the
metrics and their definitions read-only.
"""
from __future__ import annotations

import copy
import io
import logging
import math
import typing as t

from collections.abc import Mapping, Sequence
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
    Problem, Project, ProjectError, _parse,
)
from panel_flowdash.value_text import error_text
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.nodes import MappingNode, SequenceNode

from .access import is_admin
from .catalog_admin import _USED_CSS, used_html
from .data import (
    CHECKS, DashboardRef, MetricStatus, metric_statuses, metric_usage,
    visible_dashboards,
)
from .metrics import (
    DBT_KEY, DBT_ORIGIN, SECTION, metrics_page,
)
from .metrics_edit import (
    NAME_KEY, NOT_ADMIN_PREVIEW, PREVIEW_ROWS, PREVIEW_TIMEOUT, UNSET,
    PreviewResult, check_metric, delete_metric, literal_blocks, masked,
    metric_problems, preview_metric, raw_section, replace_metric, save_metric,
)

logger = logging.getLogger(__name__)

HELP = "Numbers defined here are the agreed definitions; the assistant uses them instead of writing its own SQL."

# The longest YAML the editor reads, and how deeply its values may nest.
YAML_LIMIT = 20_000
YAML_DEPTH = 20
TOO_DEEP = f"The YAML nests deeper than {YAML_DEPTH} levels; a metric needs a few."

# The keys of a metric the form edits; the YAML field holds the others.
FORM_KEYS = ("description", "source", "table")

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
.k .mono-name { font-family: "Roboto Mono", ui-monospace, monospace; font-size: 0.875rem; }
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
# Splitting a definition into the form and the YAML


def split_spec(spec: Mapping[str, t.Any]) -> tuple[dict[str, t.Any], str]:
    """The form's values of a metric's definition, and the rest of it as YAML text."""
    form = {key: spec.get(key) for key in FORM_KEYS}
    rest = {key: value for key, value in spec.items() if key not in FORM_KEYS}
    return form, (_yaml_text(literal_blocks(rest)) if rest else "")


def _yaml_text(value: Mapping[str, t.Any]) -> str:
    yaml = YAML()
    yaml.width = 4096
    buffer = io.StringIO()
    yaml.dump(value, buffer)
    return buffer.getvalue()


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
    if (refusal := _yaml_refusal(text)) is not None:
        return spec, [Problem(refusal, section=SECTION, keys=("", "yaml"))], {}
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


def _yaml_refusal(text: str) -> str | None:
    """
    Why the YAML field's ``text`` is not read: too long, or using anchors and
    aliases, which a metric never needs and which can expand to far more
    than the text, e.g. a billion laughs. Composing does not expand them.
    """
    if len(text) > YAML_LIMIT:
        return f"The YAML is longer than {YAML_LIMIT:,} characters."
    try:
        root = YAML(typ="safe", pure=True).compose(text)
    except YAMLError:
        return None
    except RecursionError:
        return TOO_DEEP
    seen: set[int] = set()
    stack = [(root, 0)] if root is not None else []
    while stack:
        node, depth = stack.pop()
        if id(node) in seen or node.anchor:
            return "Anchors and aliases (&name, *name) are not supported; write the values out."
        if depth > YAML_DEPTH:
            return TOO_DEEP
        seen.add(id(node))
        if isinstance(node, MappingNode):
            stack.extend((item, depth + 1) for pair in node.value for item in pair)
        elif isinstance(node, SequenceNode):
            stack.extend((item, depth + 1) for item in node.value)
    return None


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


# ---------------------------------------------------------------------------
# The preview


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
    Unless ``editable`` its fields only show the definition, and there is
    no preview, which runs what the fields say.
    """

    def __init__(self, project: Project, editable: bool = True, app: t.Any = None, **params):
        super().__init__(**params)
        self.project = project
        self.editable = editable
        self._app = app if app is not None else current_app()
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
        self._preview_parts = list(self._layout[4:])
        if not editable:
            self.set_readonly(True)
        self.load()

    def set_readonly(self, readonly: bool):
        """Show the definition without letting it be edited or previewed, or editable again."""
        for widget in (self._name, self._description, self._source, self._table, self._yaml):
            widget.disabled = readonly
        for item in self._preview_parts:
            item.visible = not readonly

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
        except Exception as exc:
            logger.warning("Could not list the tables of source %r: %s", ref, error_text(exc))
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
        """Run the preview of the form's definition and show it; only for an admin."""
        if not is_admin(self._app):
            self._preview.object = f'<div class="k"><span class="error-text">{esc(NOT_ADMIN_PREVIEW)}</span></div>'
            return []
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

    def focus(self):
        """Focus the name field."""
        self._name.focus()

    def __panel__(self):
        return self._layout


# ---------------------------------------------------------------------------
# The admin section

# Below this width the table shows only a metric, what it reads, its check, where it is used and its menu.
_NARROW = 760

_NARROW_CSS = f"""
@media (max-width: {_NARROW}px) {{
  :host > div:nth-of-type(3), :host > div:nth-of-type(4), :host > div:nth-of-type(5) {{ display: none !important; }}
}}
"""


class _MetricsTable(Table):
    """The kit's table, dropping the measures, dimensions and parameters columns at phone width."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        header = self._layout.objects[1]
        header.stylesheets = [*header.stylesheets, _NARROW_CSS]

    def set_rows(self, rows):
        super().set_rows(rows)
        if rows:
            for row in self._rows:
                row.stylesheets = [*row.stylesheets, _NARROW_CSS]


# How long after the drawer or dialog closes focus moves: its exit transition hands focus back to
# what had it when it opened, and a row the table just redrew mounts only after this.
_FOCUS_DELAY_MS = 400


# Armed when the drawer or dialog closes, it marks itself moved when the user presses a key or the
# pointer before focus is put back, so focus stays where they put it in the meantime.
_FOCUS_GUARD_JS = """
const token = cb_obj.value
if (!token.startsWith("armed")) { return }
const seen = () => {
  document.removeEventListener("keydown", seen, true)
  document.removeEventListener("pointerdown", seen, true)
  if (cb_obj.value === token) { cb_obj.value = "moved" + token.slice(5) }
}
document.addEventListener("keydown", seen, true)
document.addEventListener("pointerdown", seen, true)
"""


def _focus_guard() -> pn.widgets.TextInput:
    guard = pn.widgets.TextInput(value="", visible=False, margin=0)
    guard.jscallback(value=_FOCUS_GUARD_JS)
    return guard


def _focus_after_close(widget: t.Any, guard: t.Any = None):
    doc = pn.state.curdoc
    if doc is None or doc.session_context is None:
        widget.focus()
        return
    token = ""
    if guard is not None:
        token = str(int(guard.value[5:] or 0) + 1 if guard.value[:5] in ("armed", "moved") else 1)
        guard.value = f"armed{token}"

    def focus():
        if guard is None or guard.value != f"moved{token}":
            widget.focus()

    pn.state.add_periodic_callback(focus, period=_FOCUS_DELAY_MS, count=1)


class MetricsSettings(Viewer):
    """
    The metrics: the table of the project's metrics and a drawer that edits
    one, or only shows it unless ``editable``. ``app`` is the session's
    app, by default the one whose admin view builds it.
    """

    editable = param.Boolean(default=True)

    selected = param.String(default=None, allow_None=True, doc="The metric the drawer edits; None for a new one.")

    filter = param.String(default="all", doc="""
        The chip filtering the table: ``all``, ``problems``, or ``source:<ref>`` for one source.""")

    query = param.String(default="")

    def __init__(
        self, project: Project, editable: bool = True, app: t.Any = None, show_new: bool = True,
        on_open: t.Callable[[str], t.Any] | None = None, on_change: t.Callable[[str | None], t.Any] | None = None,
        **params,
    ):
        super().__init__(editable=editable, **params)
        self.project = project
        self._app = app if app is not None else current_app()
        # Opens a metric's page when a row is clicked; without it the row opens the editor.
        self._on_open = on_open
        # Told the metric saved (by its name) or deleted (None).
        self._on_change = on_change
        self._base: t.Any = UNSET
        # The metric whose row menu takes the focus when the drawer or the dialog closes; None for New metric.
        self._return_to: str | None = None
        self._alert = pmui.Alert(
            object="", severity="error", visible=False, sizing_mode="stretch_width", margin=0
        )
        self._new = pmui.Button(
            label="New metric", icon="add", variant="contained", margin=0,
            disabled=not editable, visible=editable and show_new, css_classes=["flowdash-metrics-new"],
        )
        self._new.on_click(lambda _event: self.open_metric(None))
        self._banner = pn.Column(margin=0, sizing_mode="stretch_width", visible=False,
                                 css_classes=["flowdash-metrics-banner"])
        self._search = pmui.TextInput(placeholder="Search metrics", size="small", width=240, margin=0)
        self._search.param.watch(lambda event: setattr(self, "query", event.new or ""), "value_input")
        self._chips: dict[str, pmui.Chip] = {}
        self._chip_row = pn.Row(margin=0, styles={"gap": "8px", "flex-wrap": "wrap", "align-items": "center"})
        self._count = caption("", styles={"flex": "1 1 160px", "min-width": "0", "text-align": "right"})
        toolbar = pn.Row(
            self._search,
            self._chip_row,
            self._count,
            self._new,
            margin=0,
            sizing_mode="stretch_width",
            styles={
                "padding": "10px 12px", "align-items": "center", "gap": "8px", "flex-wrap": "wrap",
                "border-bottom": "1px solid var(--mui-palette-divider)",
            },
        )
        self._table = _MetricsTable(
            [
                ("Metric", "2fr"),
                ("Reads", "1.3fr"),
                ("Measures", "1.4fr"),
                ("Dimensions", "1.4fr"),
                ("Parameters", "1fr"),
                ("Check", "1.1fr"),
                ("Used by", "0.6fr"),
                *([("", "40px")] if editable else []),
            ],
            toolbar=toolbar,
            footer=caption(
                "Dashboards and the assistant read a metric by its name, so renaming or deleting one "
                "breaks the dashboards that use it.",
                margin=(12, 12),
            ),
            empty=(
                "No metrics yet. Define the numbers people ask for most, e.g. installed capacity." if editable
                else "No metrics yet. Admins define them."
            ),
            css_classes=["flowdash-admin-metrics"],
        )
        self._menus: dict[str, pmui.MenuButton] = {}
        self._editor = MetricEditor(project, editable=editable, app=self._app)
        self._build_drawer()
        self._build_dialog()
        self._focus_guard = _focus_guard()
        self._layout = pn.Column(
            self._alert,
            self._banner,
            self._table,
            self._drawer,
            self._dialog,
            self._focus_guard,
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
        """The dashboards this session may open that use each metric, by metric name."""
        return metric_usage(visible_dashboards(self._app))

    def _show_problems(self, problems: Sequence[Problem]):
        self._alert.param.update(object="\n\n".join(p.message for p in masked(problems)), visible=bool(problems))

    def _notify(self, message: str):
        if self._app is not None:
            self._app._notify("success", message)

    # ------------------------------------------------------------------
    # The table

    def statuses(self) -> list[MetricStatus]:
        """Each metric of the file as it is now, with whether the app uses it."""
        from .access import can_access_metric

        statuses, general = metric_statuses(self.project)
        self._general = general
        context = self.project.context or {}
        return [status for status in statuses if can_access_metric(status, context, app=self._app)]

    def _passes(self, status: MetricStatus) -> bool:
        if self.filter == "problems" and status.check == "valid":
            return False
        if self.filter == f"origin:{DBT_ORIGIN}" and status.origin != DBT_ORIGIN:
            return False
        if self.filter.startswith("source:") and status.spec.get("source") != self.filter[7:]:
            return False
        measures, dimensions = _fields(status.spec)
        haystack = " ".join([status.name, status.description, str(status.spec.get("table") or ""),
                             status.origin or "", *measures, *dimensions]).lower()
        return all(word in haystack for word in self.query.lower().split())

    def _chip(self, key: str) -> pmui.Chip:
        if key not in self._chips:
            css = "".join(c if c.isalnum() else "-" for c in key).lower()
            widget = pmui.Chip(label=key, size="medium", margin=0, sx={"fontSize": "0.875rem"},
                               css_classes=[f"filter-{css}"])
            widget.on_click(lambda *_args, key=key: setattr(self, "filter", key))
            self._chips[key] = widget
        return self._chips[key]

    @param.depends("filter", "query", watch=True)
    def refresh(self):
        """Draw the banner, the chips and the table from the project file."""
        self._menus = {}
        self._alert.visible = False
        statuses = self.statuses()
        usage = self.usage()
        where = Path(self.project.path).name if self.project.path else "the project file"
        n = len(statuses)
        n_dbt = sum(s.origin == DBT_ORIGIN for s in statuses)
        dbt_note = f" ({n_dbt} from dbt)" if n_dbt else ""
        self._count.object = f"{n} metric{'' if n == 1 else 's'}{dbt_note} in {where}"
        refs = list(dict.fromkeys(str(s.spec.get("source")) for s in statuses if s.spec.get("source")))
        failing = [s for s in statuses if s.check != "valid"]
        counts = {"all": n, f"origin:{DBT_ORIGIN}": n_dbt, "problems": len(failing)}
        for ref in refs:
            counts[f"source:{ref}"] = sum(s.spec.get("source") == ref for s in statuses)
        labels = {
            "all": "All",
            **({f"origin:{DBT_ORIGIN}": "From dbt"} if n_dbt else {}),
            **({f"source:{ref}": ref for ref in refs} if len(refs) > 1 else {}),
            "problems": "Has problems",
        }
        if self.filter not in labels:
            self.filter = "all"
            return
        with pn.io.hold():
            chips = []
            for key, label in labels.items():
                active = key == self.filter
                if key == "problems" and not counts[key] and not active:
                    continue
                widget = self._chip(key)
                widget.param.update(
                    label=f"{label} {counts.get(key, 0)}", icon="check" if active else None,
                    variant="filled" if active else "outlined", color="primary" if active else "default",
                )
                chips.append(widget)
            self._chip_row[:] = chips
            self._table.set_rows([
                (status.name, self._cells(status, usage.get(status.name, [])))
                for status in statuses if self._passes(status)
            ])
            self._render_banner(failing, usage)
            if self._general:
                self._alert.param.update(
                    object="The project file's metrics have problems: " + " ".join(self._general), visible=True,
                )

    def _render_banner(self, failing: Sequence[MetricStatus], usage: Mapping[str, Sequence[DashboardRef]]):
        """Say which metric cannot run and what that does, for an admin to fix it."""
        if not failing or not self.editable:
            self._banner.visible = False
            return
        status = failing[0]
        used = usage.get(status.name, [])
        where = (
            f"{', '.join(d.title for d in used[:3])} {'shows' if len(used) == 1 else 'show'} this error where "
            "it uses the metric, and the" if used else "The"
        )
        more = f" {len(failing) - 1} more metric{'s' if len(failing) > 2 else ''} cannot run either." if len(failing) > 1 else ""
        from .metric_page import problem_banner

        open_it = pmui.Button(
            label=f"Open {status.name}", variant="text", size="small", margin=0,
            sx={"textTransform": "none"}, css_classes=["flowdash-metrics-banner-open"],
        )
        open_it.on_click(lambda _event, name=status.name: self.open(name))
        self._banner[:] = [problem_banner(
            f"{status.name} can't run",
            f"{status.message} {where} assistant leaves the metric out until it is fixed.{more}", [open_it],
        )]
        self._banner.param.update(visible=True, margin=(0, 0, 16, 0))

    def open(self, name: str):
        """Open metric ``name``: its page, where there is one, else the editor."""
        if self._on_open is not None:
            self._on_open(name)
        else:
            self.open_metric(name)

    def problems(self, section: Mapping[str, t.Any]) -> dict[str | None, list[str]]:
        """
        Why the app does not use each metric of ``section``, e.g. after a hand
        edit of the file, by metric name; None for problems of no one metric.
        """
        return metric_problems(self.project, section)

    def _cells(self, status: MetricStatus, used: Sequence[DashboardRef]) -> list:
        name, spec = status.name, status.spec
        measures, dimensions = _fields(spec)
        parameters = [str(p) for p in spec.get("parameters") or {}] if isinstance(spec.get("parameters"), dict) else []
        origin_chip = f" {chip(status.origin, 'outlined')}" if status.origin else ""
        title = cell(
            f'<div class="metric"><span class="main mono-name">Σ&nbsp; {esc(name)}{origin_chip}</span>'
            f'<span class="main muted small">{esc(status.description) or "&nbsp;"}</span></div>',
            stylesheets=[_ROW_CSS],
        )
        row_classes = []
        if status.check != "valid":
            row_classes.append("flowdash-admin-metric-invalid")
        if status.origin == DBT_ORIGIN:
            row_classes.append("flowdash-metric-dbt")
        open_it = pmui.Clickable(
            object=title, sizing_mode="stretch_width", margin=0,
            css_classes=row_classes,
        )
        open_it.on_click(lambda *_args, name=name: self.open(name))
        table = spec.get("table") if "table" in spec else None
        missing = status.check == "table"
        where = (
            f'<span class="mono{" error-text" if missing else ""}">{esc(table)}</span>' if table
            else '<span class="mono">SQL query</span>'
        )
        reads = cell(f'<span class="main mono-name">{esc(spec.get("source", ""))}</span>{where}', stylesheets=[_ROW_CSS])
        none = '<span class="muted">-</span>'
        fields = lambda names: f'<div class="fields">{"".join(chip(n) for n in names)}</div>' if names else none
        sql_note = '<span class="muted">its query\'s columns</span>'
        check = (
            chip(CHECKS["valid"], "success") if status.check == "valid"
            else f'<span title="{esc(status.message)}">{chip(CHECKS[status.check], "error")}</span>'
        )
        cells = [
            open_it,
            reads,
            cell(fields(measures) if "sql" not in spec else sql_note),
            cell(fields(dimensions) if "sql" not in spec else none),
            cell(fields(parameters)),
            cell(check, css_classes=["flowdash-metric-check"]),
            cell(used_html(used, []), stylesheets=[_USED_CSS]),
        ]
        if self.editable:
            if status.read_only:
                cells.append(cell("", css_classes=["flowdash-metric-readonly"]))
            else:
                menu = pmui.MenuButton(
                    items=[
                        {"label": "Edit", "icon": "edit", "action": "edit"},
                        {"label": "Duplicate", "icon": "content_copy", "action": "duplicate"},
                        {"label": "Delete", "icon": "delete", "action": "delete"},
                    ],
                    icon="more_vert", label="", variant="text", color="default", size="small", icon_size="20px",
                    margin=0, sx=_MENU_SX, description=f"More actions for {name}",
                    css_classes=["flowdash-admin-metric-menu"],
                )
                menu.on_click(lambda item, name=name: self.run(item.get("action"), name))
                self._menus[name] = menu
                cells.append(menu)
        return cells

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
        self._cancel = cancel = pmui.Button(
            label="Cancel" if self.editable else "Done", variant="outlined", color="default", margin=0,
        )
        cancel.on_click(lambda _event: self.close_metric())
        self._save = pmui.Button(
            label="Save", icon="save", variant="contained", margin=0, disabled=not self.editable,
            visible=self.editable, css_classes=["flowdash-metric-save"],
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
        self._drawer.param.watch(self._on_drawer_open, "open")

    def _on_drawer_open(self, event):
        if event.new:
            return
        # Closed with Escape, the backdrop or a button: focus goes back to what opened it.
        self.selected = None
        self._base = UNSET
        if not self._dialog.open:
            self._restore_focus()

    def _restore_focus(self):
        """Focus the row menu of the metric last acted on, or New metric when it is gone."""
        opener = self._menus.get(self._return_to) if self._return_to is not None else None
        _focus_after_close(opener or self._new, self._focus_guard)

    def _set_editor_readonly(self, readonly: bool):
        disabled = readonly or not self.editable
        self._editor.set_readonly(disabled)
        self._save.param.update(disabled=disabled, visible=not disabled)
        self._cancel.label = "Done" if disabled else "Cancel"

    def open_metric(self, name: str | None, spec: Mapping[str, t.Any] | None = None, title: str | None = None):
        """Open the editor on metric ``name``, or on a new one (defined as ``spec``, if given)."""
        section = self.section()
        readonly_status = None
        if name is not None and (name == DBT_KEY or name not in section):
            readonly_status = next((s for s in self.statuses() if s.name == name and s.read_only), None)
            if readonly_status is None:
                return
        self.selected = None if readonly_status is not None else name
        self._return_to = name
        self._base = copy.deepcopy(section[name]) if name is not None and readonly_status is None else UNSET
        with pn.io.hold():
            self._drawer_alert.visible = False
            if readonly_status is not None:
                self._drawer_title.object = title or f"{name} (from {readonly_status.origin})"
                self._drawer_subtitle.object = f"Inherited from {readonly_status.origin} (read-only)."
                self._set_editor_readonly(True)
                self._editor.load(name or "", readonly_status.spec)
            else:
                self._drawer_title.object = title or (
                    "New metric" if name is None else f"Edit {name}" if self.editable else name
                )
                self._drawer_subtitle.object = HELP
                self._set_editor_readonly(False)
                self._editor.load(name or "", section[name] if name is not None else spec)
            self._drawer.open = True

    def close_metric(self):
        """Close the editor without saving."""
        if self._drawer.open:
            self._drawer.open = False
        self.selected = None
        self._base = UNSET

    def save(self, confirmed: bool = False) -> bool:
        """
        Save the editor's metric; show the problems at their fields if it is
        refused. Renaming a metric dashboards use asks first, as deleting
        it does, unless ``confirmed``.
        """
        spec, problems = self._editor.spec()
        name = self._editor.metric_name
        original = self.selected
        renamed = original is not None and name != original
        if not problems and renamed and not confirmed and (used := self.usage().get(original)):
            problems = check_metric(self.project, name, spec, original)
            if not problems:
                self.ask_rename(original, name, used)
                return False
        if not problems:
            problems = save_metric(
                self.project, name, spec, app=self._app if self.editable else None,
                original=original, base=self._base,
            )
        if problems:
            general = self._editor.show_problems(problems)
            self._drawer_alert.param.update(
                object="\n\n".join(general) if general else "Fix the fields marked below, then save again.",
                visible=True,
            )
            return False
        self._return_to = name
        self.refresh()
        self.close_metric()
        self._notify(f"Metric {name} saved.")
        if self._on_change is not None:
            self._on_change(name)
        return True

    def duplicate(self, name: str):
        """Open the editor on a new metric defined as ``name`` is, named after it."""
        section = self.section()
        if name == DBT_KEY or name not in section:
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
        # What the dialog asks about: ("delete", name, its definition) or ("rename", name, new name).
        self._pending: tuple[str, str, t.Any] | None = None
        self._dialog_text = cell("", stylesheets=[_ROW_CSS])
        self._dialog_alert = pmui.Alert(
            object="", severity="error", visible=False, sizing_mode="stretch_width", margin=0
        )
        cancel = pmui.Button(label="Cancel", variant="text", color="default", margin=0)
        cancel.on_click(lambda _event: self.cancel_delete())
        self._confirm = pmui.Button(label="Delete metric", variant="contained", color="error", margin=0)
        self._confirm.on_click(lambda _event: self.confirm())
        self._dialog = pmui.Dialog(
            pn.Column(self._dialog_alert, self._dialog_text, margin=0, sizing_mode="stretch_width",
                      styles={"gap": "12px"}),
            actions=[cancel, self._confirm],
            title="",
            open=False,
            close_on_click=True,
            css_classes=["flowdash-admin-confirm"],
        )
        self._dialog.param.watch(self._on_dialog_open, "open")

    def _on_dialog_open(self, event):
        if event.new:
            return
        self._pending = None
        if self._drawer.open:
            _focus_after_close(self._editor, self._focus_guard)
        else:
            self._restore_focus()

    @staticmethod
    def _used_html(name: str, used: Sequence[DashboardRef], effect: str) -> str:
        links = ", ".join(
            f'<a href="/dash/{esc(ref.dashboard_id)}">{esc(ref.title)}</a>' for ref in used[:10]
        ) + (f" and {len(used) - 10} more" if len(used) > 10 else "")
        n = len(used)
        return (
            f'<p class="warn"><b>{n} dashboard{"" if n == 1 else "s"} use{"s" if n == 1 else ""} '
            f"{esc(name)}</b>: {links}. {effect}</p>"
        )

    def ask_rename(self, original: str, name: str, used: Sequence[DashboardRef]):
        """Ask before renaming ``original``, which the dashboards ``used`` read, to ``name``."""
        self._pending = ("rename", original, name)
        text = self._used_html(
            original, used,
            f"Their metric tiles will show an error until a metric named {esc(original)} is defined again.",
        )
        with pn.io.hold():
            self._dialog_alert.visible = False
            self._dialog_text.object = f'<div class="k">{text}<p>Rename it to <b>{esc(name)}</b> anyway?</p></div>'
            self._confirm.label = "Rename metric"
            self._dialog.param.update(title=f"Rename metric {original}?", open=True)

    def ask_delete(self, name: str):
        """Ask before deleting ``name``, listing the dashboards that use it."""
        section = self.section()
        if name == DBT_KEY or name not in section:
            return
        used = self.usage().get(name, [])
        self._pending = ("delete", name, copy.deepcopy(section[name]))
        self._return_to = name
        if used:
            text = self._used_html(
                name, used, "Their metric tiles will show an error until a metric of this name is defined again."
            )
        else:
            text = f"<p>No dashboard uses <b>{esc(name)}</b>. The assistant stops offering it.</p>"
        with pn.io.hold():
            self._dialog_alert.visible = False
            self._dialog_text.object = f'<div class="k">{text}<p>This can\'t be undone.</p></div>'
            self._confirm.label = "Delete metric"
            self._dialog.param.update(title=f"Delete metric {name}?", open=True)

    def cancel_delete(self):
        """Close the dialog without deleting or renaming."""
        self._dialog.open = False

    def confirm(self) -> bool:
        """Carry out what the dialog asks about."""
        if self._pending is None:
            return False
        if self._pending[0] == "rename":
            self._dialog.open = False
            return self.save(confirmed=True)
        return self.confirm_delete()

    def confirm_delete(self) -> bool:
        """Delete the metric the dialog asks about."""
        if self._pending is None or self._pending[0] != "delete":
            return False
        _, name, base = self._pending
        problems = delete_metric(self.project, name, app=self._app if self.editable else None, base=base)
        if problems:
            self._dialog_alert.param.update(object="\n\n".join(p.message for p in problems), visible=True)
            return False
        if self.selected == name:
            self.close_metric()
        self.refresh()
        self._dialog.open = False
        self._notify(f"Metric {name} deleted.")
        if self._on_change is not None:
            self._on_change(None)
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
            f"later in {metrics_page()}.",
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
        return {SECTION: replace_metric(raw_section(self.project), None, name, spec)}

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
