"""
A metric's page in the Data section: its definition as the project file
has it, what it reads and who uses it, why it does not run when it does
not, and Try it, which runs it with the parameters and filters picked.

Try it is for everyone who can open the app, as a dashboard runs the
metric for them: it runs the saved definition, never one typed in, with
the values checked and bound as a metric node binds them, read-only, at
most ``PREVIEW_ROWS`` rows in ``PREVIEW_TIMEOUT`` seconds. An admin also
gets Edit, Duplicate and Delete, through the metrics' editor.
"""
from __future__ import annotations

import typing as t

from pathlib import PureWindowsPath
from urllib.parse import quote

import panel as pn
import panel_material_ui as pmui

from panel.viewable import Viewer
from panel_flowdash.admin.kit import (
    caption, cell, chip, esc,
)

from .metrics import DBT_ORIGIN
from .metrics_edit import PREVIEW_ROWS, PREVIEW_TIMEOUT

_MONO_SX = {"fontFamily": '"Roboto Mono", ui-monospace, monospace', "fontSize": "0.8125rem"}

_BANNER_CSS = """
.banner { display: flex; gap: 12px; font-family: inherit; color: var(--mui-palette-text-primary); }
.banner svg { flex: 0 0 20px; width: 20px; height: 20px; fill: var(--mui-palette-error-main, #d32f2f); margin-top: 1px; }
.banner b { display: block; font-weight: 500; margin-bottom: 4px; }
.banner p { margin: 0; font-size: 0.875rem; line-height: 1.5; }
"""

_ERROR_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M11 15h2v2h-2zm0-8h2v6h-2zm.99-5C6.47 2 2 6.48 2 12s4.47 10'
    ' 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2M12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8"/></svg>'
)


def problem_banner(title: str, text: str, actions: t.Sequence[t.Any] = (), **params) -> pn.Column:
    """A tinted banner saying what cannot run and why, with ``actions`` under the text."""
    body = cell(f'<div class="banner">{_ERROR_SVG}<div><b>{esc(title)}</b><p>{esc(text)}</p></div></div>',
                stylesheets=[_BANNER_CSS])
    return pn.Column(
        body, *([pn.Row(*actions, margin=(4, 0, 0, 28), styles={"gap": "8px"})] if actions else []),
        margin=0, sizing_mode="stretch_width",
        styles={"background": "rgba(var(--mui-palette-error-mainChannel) / 0.08)", "border-radius": "8px",
                "padding": "14px 16px 10px 16px", "box-sizing": "border-box"},
        **params,
    )


_OVERLINE_SX = {
    "color": "text.secondary", "fontSize": "0.75rem", "fontWeight": 500, "letterSpacing": "1.2px", "lineHeight": 2,
}

_PAGE_CSS = """
.k { font-size: 0.875rem; color: var(--mui-palette-text-primary); }
.k dl { display: grid; grid-template-columns: 120px 1fr; gap: 8px 16px; margin: 0; }
.k dt { color: var(--mui-palette-text-secondary); font-size: 0.8125rem; }
.k dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.k .scroll { overflow-x: auto; border: 1px solid var(--mui-palette-divider); border-radius: 4px; }
.k table { border-collapse: collapse; width: 100%; font-size: 0.8125rem; }
.k th { text-align: left; font-weight: 400; color: var(--mui-palette-text-secondary); white-space: nowrap; }
.k th, .k td { padding: 6px 10px; border-bottom: 1px solid var(--mui-palette-divider); vertical-align: top; }
.k tr:last-child td { border-bottom: none; }
.k td.n { text-align: right; font-variant-numeric: tabular-nums; }
.k code, .k .mono { font-family: "Roboto Mono", ui-monospace, monospace; font-size: 0.8125rem; }
.k pre { margin: 0; padding: 10px 12px; border-radius: 4px; white-space: pre-wrap; overflow-x: auto;
  font-family: "Roboto Mono", ui-monospace, monospace; font-size: 0.75rem; line-height: 1.55;
  background: rgba(var(--mui-palette-text-primaryChannel) / 0.06); color: var(--mui-palette-text-primary); }
.k .muted { color: var(--mui-palette-text-secondary); }
.k .error-text { color: var(--mui-palette-error-main, #d32f2f); }
.k .note { color: var(--mui-palette-text-secondary); font-size: 0.8125rem; }
.k .blocked { border: 1px dashed var(--mui-palette-divider); border-radius: 6px; padding: 14px 16px;
  color: var(--mui-palette-text-secondary); }
"""

_HEAD_CSS = """
.head { display: flex; gap: 16px; align-items: flex-start; font-family: inherit; }
.icon { flex: 0 0 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
  background: rgba(var(--mui-palette-primary-mainChannel) / 0.1); color: var(--mui-palette-primary-main);
  font-size: 1.25rem; font-weight: 500; }
.name { font-family: "Roboto Mono", ui-monospace, monospace; font-size: 1.625rem; line-height: 1.25;
  color: var(--mui-palette-text-primary); }
.label { color: var(--mui-palette-text-primary); font-size: 1rem; margin-top: 4px; }
.chips { margin-top: 8px; }
"""

# Below this width Try it goes under the definition.
_STACKED = 1180

# How each kind of failure stops Try it.
_BLOCKED = {
    "source": "Not run: the source it reads is not served.",
    "table": "Not run: the table it reads is missing.",
    "column": "Not run: a column it reads is missing.",
    "query": "Not run: its query does not run.",
    "invalid": "Not run: its definition has problems.",
}


def _table(head: t.Sequence[str], rows: t.Sequence[t.Sequence[str]]) -> str:
    """An escaped HTML table; each cell of ``rows`` is markup already."""
    header = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def _code(text: t.Any) -> str:
    return f'<span class="mono">{esc(text)}</span>' if text not in (None, "") else '<span class="muted">-</span>'


def what_it_does(parameter: t.Any) -> str:
    """What a metric's parameter does, e.g. "Keeps rows with p_year >= since. Optional."."""
    from .metrics import json_value

    if parameter.column and parameter.op:
        text = f"Keeps rows with {parameter.column} {parameter.op} {parameter.name}."
    else:
        text = parameter.description.rstrip(".") + "." if parameter.description else ""
    if parameter.required:
        text += " Required."
    elif parameter.default is not None:
        text += f" Default {json_value(parameter.default)}."
    elif parameter.column:
        text += " Optional."
    return text.strip()


def formatted(value: t.Any, spec: str | None) -> str:
    """``value`` in a measure's d3 ``format``, where Python's format language reads it the same; else nothing."""
    if not spec:
        return ""
    try:
        return format(value, spec.replace("~", ""))
    except (ValueError, TypeError):
        return ""


def _values(text: str, kind: str) -> t.Any:
    """What someone typed for a parameter or a filter: a list for comma-separated values or a list type."""
    text = text.strip()
    if not text:
        return None
    if kind in ("list", "range") or "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


class MetricPage(Viewer):
    """
    The page of one metric, a ``data.MetricStatus``. ``controller`` is the
    metrics' editor (``MetricsSettings``) an admin edits it with;
    ``navigate`` opens a URL in the app.
    """

    def __init__(
        self, app: t.Any, project: t.Any, status: t.Any, *, admin: bool = False, controller: t.Any = None,
        navigate: t.Callable[[str], t.Any] | None = None, **params,
    ):
        super().__init__(**params)
        from .data import metric_usage, visible_dashboards

        self.app = app
        self.project = project
        self.status = status
        self.admin = admin
        self.controller = controller
        self._navigate = navigate or (lambda url: None)
        self.context = project.context or {}
        self.metric = status.metric
        self.used = metric_usage(visible_dashboards(app)).get(status.name, [])
        main = pn.Column(
            self._head(),
            *self._problem(),
            self._facts(),
            *self._definition(),
            margin=0, sizing_mode="stretch_width", min_width=0,
            styles={"padding": "24px 32px 48px 32px", "box-sizing": "border-box", "min-width": "0", "gap": "12px",
                    "flex": "1 1 auto"},
            stylesheets=[f"@media (max-width: {_STACKED}px) {{ :host {{ padding: 16px !important; }} }}"],
            css_classes=["lumen-metric-definition"],
        )
        side = pn.Column(
            *self._try_it(),
            margin=0, sizing_mode="stretch_width", min_width=0,
            styles={"flex": "0 0 380px", "max-width": "380px", "padding": "24px 20px 48px 20px",
                    "box-sizing": "border-box", "border-left": "1px solid var(--mui-palette-divider)",
                    "min-height": "calc(100vh - 56px)", "gap": "12px"},
            stylesheets=[f"@media (max-width: {_STACKED}px) {{ :host {{ max-width: none !important; "
                         "border-left: none !important; min-height: 0 !important; }} }}"],
            css_classes=["lumen-metric-try"],
        )
        parts: list[t.Any] = [main, side]
        if controller is not None and not status.read_only:
            parts += [controller._drawer, controller._dialog, controller._focus_guard]
        # An explicit least width, so Panel does not add up its children's into one a phone cannot fit.
        self._layout = pn.Row(
            *parts, margin=0, sizing_mode="stretch_width", min_width=0, css_classes=["lumen-metric-page"],
            styles={"align-items": "stretch"},
            stylesheets=[f"@media (max-width: {_STACKED}px) {{ :host {{ flex-direction: column !important; }} }}"],
        )

    # ------------------------------------------------------------------
    # The definition

    def _head(self) -> pn.Row:
        from .data import CHECKS

        status = self.status
        check = (chip(CHECKS["valid"], "success") if status.check == "valid"
                 else chip(CHECKS[status.check], "error"))
        # PureWindowsPath(...).name splits on both "/" and "\\" regardless of host OS,
        # so a Windows project path still shows just its file name in the breadcrumb.
        where = PureWindowsPath(self.project.path).name if self.project.path else "the project file"
        description = status.description
        if status.origin:
            origin_chips = f'{chip(f"From {status.origin}", "outlined")}{chip("Read-only", "outlined")}'
        else:
            origin_chips = chip(where, "outlined")
        head = cell(
            f'<div class="head"><div class="icon">Σ</div><div><div class="name">{esc(status.name)}</div>'
            + (f'<div class="label">{esc(description)}</div>' if description else "")
            + f'<div class="chips">{check}{origin_chips}</div></div></div>',
            stylesheets=[_HEAD_CSS], css_classes=["lumen-metric-head"],
        )
        actions = []
        if self.admin and self.controller is not None and not status.read_only:
            edit = pmui.Button(label="Edit", icon="edit", variant="outlined", margin=0,
                               css_classes=["lumen-metric-edit"])
            edit.on_click(lambda _event: self.controller.open_metric(status.name))
            menu = pmui.MenuButton(
                items=[{"label": "Duplicate", "icon": "content_copy", "action": "duplicate"},
                       {"label": "Delete", "icon": "delete", "action": "delete"}],
                icon="more_vert", label="", variant="text", color="default", size="small", margin=0,
                description=f"More actions for {status.name}", css_classes=["lumen-metric-menu"],
                sx={"minWidth": 0, "& .MuiButton-endIcon": {"display": "none"}},
            )
            menu.on_click(lambda item: self.controller.run(item.get("action"), status.name))
            actions = [edit, menu]
        return pn.Row(head, pn.layout.HSpacer(), *actions, margin=0, sizing_mode="stretch_width",
                      styles={"align-items": "flex-start", "gap": "8px"})

    def _where_it_shows(self) -> str:
        if not self.used:
            return "No dashboard uses it yet."
        titles = ", ".join(d.title for d in self.used[:3]) + (f" and {len(self.used) - 3} more" if len(self.used) > 3 else "")
        return f"{titles} {'shows' if len(self.used) == 1 else 'show'} this error where {'it uses' if len(self.used) == 1 else 'they use'} the metric."

    def _problem(self) -> list[t.Any]:
        status = self.status
        if status.check == "valid":
            return []
        buttons = []
        if self.admin and self.controller is not None and not status.read_only:
            edit = pmui.Button(label="Edit definition", variant="text", size="small", margin=0,
                               sx={"textTransform": "none"})
            edit.on_click(lambda _event: self.controller.open_metric(status.name))
            buttons.append(edit)
        ref = status.spec.get("source")
        if status.check in ("table", "column") and isinstance(ref, str):
            show = pmui.Button(label=f"Show {ref}'s tables", variant="text", size="small", margin=0,
                               sx={"textTransform": "none"})
            show.on_click(lambda _event: self._navigate(f"/data?source={quote(ref, safe='')}"))
            buttons.append(show)
        text = f"{status.message} {self._where_it_shows()} The assistant leaves the metric out until it is fixed."
        return [problem_banner("This metric can't run", text, buttons, css_classes=["lumen-metric-problem"])]

    def _link(self, label: str, url: str, icon: str | None = None, mono: bool = False) -> pmui.Button:
        button = pmui.Button(
            label=label, icon=icon, variant="text", size="small", margin=0,
            sx={"textTransform": "none", "minWidth": 0,
                **({"fontFamily": '"Roboto Mono", ui-monospace, monospace'} if mono else {})},
        )
        button.on_click(lambda _event: self._navigate(url))
        return button

    def _fact(self, label: str, *value: t.Any) -> pn.Row:
        return pn.Row(
            pmui.Typography(label, variant="body2", margin=0,
                            sx={"color": "text.secondary"}, styles={"flex": "0 0 120px", "width": "120px"}),
            pn.Row(*value, margin=0, sizing_mode="stretch_width",
                   styles={"flex-wrap": "wrap", "align-items": "center", "gap": "0 6px", "min-height": "30px"}),
            margin=0, sizing_mode="stretch_width", styles={"gap": "16px", "align-items": "center"},
        )

    def _facts(self) -> pn.Column:
        from .data import hidden_reads_of, metric_tables

        status, spec = self.status, self.status.spec
        ref = str(spec.get("source") or "")
        rows = []
        reads: list[t.Any] = [pmui.Typography(f"{ref} ·", variant="body2", margin=0, sx=_MONO_SX)]
        if status.check == "table" and spec.get("table"):
            reads.append(pmui.Typography(str(spec["table"]), variant="body2", margin=0,
                                         sx={**_MONO_SX, "color": "error.main"}))
            reads.append(caption("not found"))
        else:
            tables = metric_tables(spec, self.context)
            for table_ref, table in tables[:6]:
                reads.append(self._link(table, f"/data?table={quote(f'{table_ref}.{table}', safe='')}", mono=True))
            note = "opens the table" if spec.get("table") else "its SQL query reads it" if len(tables) == 1 else "its SQL query reads them"
            reads.append(caption(note))
        rows.append(self._fact("Reads", *reads))
        if status.origin == DBT_ORIGIN:
            detail = "dbt Semantic Layer (read-only · compiled SQL)" if self.metric and self.metric.form == "sql" else "dbt Semantic Layer (read-only)"
            rows.append(self._fact("Origin", caption(detail)))
        if status.check == "valid":
            rows.append(self._fact("On dashboards", caption(f"Σ  Metric: {status.name}")))
        used = [self._link(d.title, f"/dash/{d.dashboard_id}", icon="space_dashboard") for d in self.used[:8]]
        rows.append(self._fact("Used by", *(used or [caption("No dashboard yet")])))
        if hidden := hidden_reads_of(spec, self.context):
            names = ", ".join(hidden)
            rows.append(self._fact("Assistant", caption(
                f"Leaves this metric out: {names} {'is' if len(hidden) == 1 else 'are'} turned off for the assistant.")))
        return pn.Column(*rows, margin=(4, 0, 4, 0), sizing_mode="stretch_width", styles={"gap": "4px"},
                         css_classes=["lumen-metric-facts"])

    def _definition(self) -> list[t.Any]:
        metric, spec = self.metric, self.status.spec
        parts: list[t.Any] = []

        def section(title: str, html: str):
            parts.append(pmui.Typography(title, variant="overline", sx=_OVERLINE_SX, margin=(8, 0, 0, 0)))
            parts.append(cell(f'<div class="k">{html}</div>', stylesheets=[_PAGE_CSS]))

        if metric is None:
            import yaml

            section("Definition", f"<pre>{esc(yaml.safe_dump(spec, sort_keys=False))}</pre>")
            return parts
        if metric.form == "sql":
            section("Query", f"<pre>{esc(metric.sql.strip())}</pre>")
        else:
            section("Measures", _table(["Measure", "Expression", "Format"], [
                [_code(m.name), _code(m.expr), _code(m.format)] for m in metric.measures.values()
            ]))
            if metric.dimensions:
                section("Dimensions", _table(["Dimension", "Column", "Kind"], [
                    [_code(d.name), _code(d.column or d.expr),
                     esc(f"time, by {' or '.join(d.grains)}" if d.time and d.grains else "time" if d.time else "value")]
                    for d in metric.dimensions.values()
                ]))
        # A metric's limit and order_by are listed with the parameters it declares, as a metric node offers them.
        parameters = {**metric.parameters, **metric.knobs()} if metric.parameters else {}
        if parameters:
            section("Parameters", _table(["Parameter", "Type", "What it does"], [
                [_code(p.name), esc("list of columns" if p.name == "order_by" and p.kind == "list" else p.type),
                 esc(what_it_does(p))]
                for p in parameters.values()
            ]))
        filters = metric._filterable()
        if filters:
            first = metric.dimensions[filters[0]]
            parts.append(caption(
                f"{'No parameters. ' if not metric.parameters else ''}Every dimension also filters, e.g. "
                f"{filters[0]} = a value or a list of values of {first.column}.", margin=(4, 0, 0, 0)))
        return parts

    # ------------------------------------------------------------------
    # Try it

    def _try_it(self) -> list[t.Any]:
        status, metric = self.status, self.metric
        parts: list[t.Any] = [pmui.Typography("Try it", variant="overline", sx=_OVERLINE_SX, margin=0)]
        if status.check != "valid" or metric is None:
            parts.append(cell(f'<div class="blocked">{esc(_BLOCKED.get(status.check, _BLOCKED["query"]))}</div>',
                              stylesheets=[_PAGE_CSS], css_classes=["lumen-metric-blocked"]))
            parts.append(pmui.Typography("Where it shows", variant="overline", sx=_OVERLINE_SX, margin=(12, 0, 0, 0)))
            parts.append(caption(self._where_it_shows()))
            return parts
        self._inputs: dict[str, t.Any] = {}
        dimensions = list(metric.dimensions)
        if metric.form == "structured" and dimensions:
            self._group = pmui.Select(label="Group by", options={"Nothing": "", **{d: d for d in dimensions}},
                                      value=dimensions[0], sizing_mode="stretch_width", margin=0,
                                      css_classes=["lumen-try-group"])
            self._group.param.watch(lambda _event: self._sync_filter(), "value")
            parts.append(self._group)
        else:
            self._group = None
        for parameter in metric.parameters.values():
            if parameter.name in metric.knobs():
                continue
            default = parameter.default
            placeholder = "Required" if parameter.required else (
                f"Default {default}" if default is not None else "Any")
            widget = pmui.TextInput(label=parameter.name, placeholder=placeholder, sizing_mode="stretch_width",
                                    margin=0, css_classes=[f"lumen-try-{parameter.name}"])
            self._inputs[parameter.name] = (widget, parameter.kind)
            parts.append(widget)
        self._filter = pmui.TextInput(label="", placeholder="Any", sizing_mode="stretch_width", margin=0,
                                      visible=False, css_classes=["lumen-try-filter"])
        parts.append(self._filter)
        knobs = metric.knobs()
        if knobs:
            first = next(iter(metric.measures), "")
            self._order = pmui.TextInput(label="order_by", value=f"-{first}" if first else "", margin=0,
                                         sizing_mode="stretch_width", css_classes=["lumen-try-order"])
            self._limit = pmui.IntInput(label="limit", value=5, start=1, end=PREVIEW_ROWS, margin=0, width=110,
                                        css_classes=["lumen-try-limit"])
            parts.append(pn.Row(self._order, self._limit, margin=0, sizing_mode="stretch_width",
                                styles={"gap": "12px"}))
        else:
            self._order = self._limit = None
        self._run_button = pmui.Button(label="Run", icon="play_arrow", variant="outlined", margin=0,
                                       css_classes=["lumen-try-run"])
        self._run_button.on_click(lambda _event: self.run())
        self._result = cell("", stylesheets=[_PAGE_CSS], css_classes=["lumen-try-result"])
        self._took = caption("", visible=False)
        self._sql = cell("", stylesheets=[_PAGE_CSS])
        self._sql_card = pmui.Card(self._sql, title="The SQL it ran", collapsible=True, collapsed=True, margin=0,
                                   title_variant="body1",
                                   sizing_mode="stretch_width", visible=False, css_classes=["lumen-try-sql"])
        parts += [self._run_button, self._result, self._took, self._sql_card]
        self._sync_filter()
        assistant = self._assistant_reads()
        if assistant:
            parts.append(pmui.Typography("The assistant reads", variant="overline", sx=_OVERLINE_SX,
                                         margin=(12, 0, 0, 0)))
            parts.append(cell(f'<div class="k"><pre>{esc(assistant)}</pre></div>', stylesheets=[_PAGE_CSS],
                              css_classes=["lumen-try-assistant"]))
        return parts

    def _assistant_reads(self) -> str:
        """The metric as the assistant's list_tables describes it, or nothing where Lumen's agent is not installed."""
        try:
            from ..agent.data import metric_line
        except Exception:
            return ""
        try:
            return metric_line(self.metric).removeprefix("- ")
        except Exception:
            return ""

    def _sync_filter(self):
        dimension = self._group.value if self._group is not None else ""
        spec = self.metric.dimensions.get(dimension) if dimension else None
        filterable = spec is not None and spec.column is not None and not spec.time
        self._filter.param.update(label=dimension or "", visible=filterable)

    def query(self) -> t.Any:
        """The ``metrics.Query`` the inputs ask for."""
        from .metrics import Query

        metric = self.metric
        values = {name: _values(widget.value_input or widget.value or "", kind)
                  for name, (widget, kind) in self._inputs.items()}
        dimension = self._group.value if self._group is not None else ""
        if self._filter.visible and (given := _values(self._filter.value_input or self._filter.value or "", "")):
            values[dimension] = given
        if metric.form == "sql":
            return Query(parameters={k: v for k, v in values.items() if v is not None})
        order = [p.strip() for p in (self._order.value_input or self._order.value or "").split(",") if p.strip()] \
            if self._order is not None else []
        dimensions = [dimension] if dimension else []
        grain = None
        if dimension and metric.dimensions[dimension].time and metric.dimensions[dimension].grains:
            grain = metric.dimensions[dimension].grains[0]
        outputs = {*dimensions, *metric.measures}
        return Query(
            measures=list(metric.measures), dimensions=dimensions, grain=grain,
            order_by=[o for o in order if o.lstrip("-") in outputs],
            limit=self._limit.value if self._limit is not None else None,
            parameters={k: v for k, v in values.items() if v is not None},
        )

    def run(self) -> t.Any:
        """Run the metric with the inputs, off the event loop, and show the result."""
        from .metrics_edit import try_metric
        from .nodes import in_background

        query = self.query()
        result: list[t.Any] = []
        self._run_button.loading = True

        def work():
            result.append(try_metric(self.metric, self.context, query))

        def done():
            self._run_button.loading = False
            if result:
                self._show(result[0])

        in_background(work, done)
        return result[0] if result else None

    def _show(self, result: t.Any):
        from .metrics_admin import _number

        with pn.io.hold():
            if result.error:
                self._result.object = f'<div class="k"><span class="error-text">{esc(result.error)}</span></div>'
                self._took.visible = False
            elif not result.rows:
                self._result.object = '<div class="k"><span class="muted">No rows.</span></div>'
            else:
                formats = [self.metric.measures[c].format if c in self.metric.measures else None for c in result.columns]
                rows = [
                    [f'<span style="display:block;text-align:right">{esc(formatted(v, f) or _number(v))}</span>'
                     if isinstance(v, (int, float)) and not isinstance(v, bool) else esc(_number(v))
                     for v, f in zip(row, formats, strict=False)]
                    for row in result.rows
                ]
                self._result.object = f'<div class="k">{_table(result.columns, rows)}</div>'
            if not result.error:
                took = f" · {result.seconds:.1f} s" if result.seconds is not None else ""
                self._took.param.update(
                    object=f"Runs read-only, at most {PREVIEW_ROWS} rows in {PREVIEW_TIMEOUT} seconds{took}",
                    visible=True,
                )
            self._sql.object = f'<div class="k"><pre>{esc(result.sql)}</pre></div>' if result.sql else ""
            self._sql_card.visible = bool(result.sql)

    def __panel__(self):
        return self._layout
