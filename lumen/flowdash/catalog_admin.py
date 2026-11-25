"""
The catalog's view: the sources the app serves and their tables, and which
of them Lumen's assistant may use. The Data section shows it, and so does
panel-flowdash's admin view as its Catalog section where there are no
sidebar sections.

The table lists every table of every source with its row and column
counts, read off the event loop; its switch turns a table off or on for
the assistant at once, writing the project file's ``catalog:`` section (see
``lumen.flowdash.catalog``, which reads and writes what this shows). A
table's drawer shows its columns with their types, value ranges and
descriptions, its other metadata and its first rows. Turning a table off or
on checks again that the caller is an admin; someone who is not sees it
all read-only, and can add data where uploads let them.
"""
from __future__ import annotations

import logging
import numbers
import typing as t

from collections.abc import Sequence
from functools import partial

import panel as pn
import panel_material_ui as pmui
import param

from panel.viewable import Viewer
from panel_flowdash.admin.form import current_app
from panel_flowdash.admin.kit import (
    Table, caption, cell, chip, esc,
)
from panel_flowdash.project import Problem, Project, ProjectError
from panel_flowdash.value_text import error_text

from ..sources.base import Source
from .add_data import AddDataDialog
from .catalog import (
    ColumnInfo, TableDetail, TableInfo, catalog_tables, known_info, plain_text,
    raw_section, set_assistant, table_detail, table_info,
)
from .uploads import UPLOADS_KEY, Upload

logger = logging.getLogger(__name__)

# The tables whose row and column counts the list reads by itself; the others when their drawer opens.
LIST_TABLES = 50
# The column names a row lists.
NAMES_SHOWN = 6
# The rows the table draws at most; a search narrows down the others.
ROWS_SHOWN = 100
# The sources the filter chips name each, at most.
SOURCE_CHIPS = 6
# The dashboards and metrics a drawer's Used by names before saying how many more.
USED_SHOWN = 8
# The columns past which a drawer folds the table's first rows.
FOLD_COLUMNS = 8

_CHIP_SX = {"fontSize": "0.875rem"}

_OVERLINE_SX = {
    "color": "text.secondary",
    "fontSize": "0.75rem",
    "fontWeight": 500,
    "letterSpacing": "1.2px",
    "lineHeight": 2,
}

_ROW_CSS = """
.k .table { cursor: pointer; min-width: 0; }
.k .off { opacity: 0.6; }
"""

_DETAILS_CSS = """
.k dl { display: grid; grid-template-columns: 120px 1fr; gap: 10px 16px; margin: 0; }
.k dt { color: var(--mui-palette-text-secondary); font-size: 0.8125rem; letter-spacing: 0.25px; }
.k dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.k .scroll { overflow-x: auto; border: 1px solid var(--mui-palette-divider); border-radius: 4px; }
.k table { border-collapse: collapse; width: 100%; font-size: 0.8125rem; }
.k th { text-align: left; font-weight: 500; color: var(--mui-palette-text-secondary); white-space: nowrap; }
.k th, .k td { padding: 6px 10px; border-bottom: 1px solid var(--mui-palette-divider); vertical-align: top; }
.k tr:last-child td { border-bottom: none; }
.k td.n { text-align: right; font-variant-numeric: tabular-nums; }
.k .sample td { white-space: nowrap; max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
.k .col { font-family: "Roboto Mono", ui-monospace, monospace; font-size: 0.75rem; white-space: nowrap; }
"""


# ---------------------------------------------------------------------------
# The view


def uploaded_text(upload: Upload) -> str:
    """Who uploaded ``upload`` and when, e.g. "Uploaded by ana on 24 Nov 2026"."""
    when = upload.when
    on = f" on {when.astimezone():%d %b %Y}" if when is not None else ""
    return f"Uploaded by {upload.uploaded_by}{on}"


def _key(ref: str, table: str) -> str:
    return f"{ref}--{table}"


def _type_name(source: Source) -> str:
    return type(source).__name__.removesuffix("Source") or type(source).__name__


def _column_names(names: Sequence[str]) -> str:
    if len(names) <= NAMES_SHOWN:
        return ", ".join(names)
    return f"{', '.join(names[:NAMES_SHOWN])} +{len(names) - NAMES_SHOWN} more"


def _sample_html(frame: t.Any) -> str:
    from ..util import as_pandas

    frame = as_pandas(frame)
    if frame is None or not len(frame.columns):
        return '<span class="muted">No rows</span>'
    head = "".join(f"<th>{esc(c)}</th>" for c in frame.columns)
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"n" if isinstance(v, numbers.Number) and not isinstance(v, bool) else ""}" '
            f'title="{esc(plain_text(v))}">{esc(plain_text(v))}</td>'
            for v in row
        ) + "</tr>"
        for row in frame.itertuples(index=False)
    ) or f'<tr><td colspan="{len(frame.columns)}" class="muted">No rows</td></tr>'
    return f'<div class="scroll sample"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _columns_html(columns: Sequence[ColumnInfo]) -> str:
    if not columns:
        return '<span class="muted">No columns known</span>'
    described = any(c.description for c in columns)
    none = '<span class="muted">-</span>'
    rows = "".join(
        f'<tr><td class="col">{esc(c.name)}</td><td>{esc(c.type) or "&nbsp;"}</td>'
        f"<td>{esc(c.values) or none}</td>"
        + (f"<td>{esc(c.description) or none}</td>" if described else "")
        + "</tr>"
        for c in columns
    )
    return (
        '<div class="scroll"><table><thead><tr><th>Column</th><th>Type</th><th>Values</th>'
        + ("<th>Description</th>" if described else "")
        + f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


# Material's dashboard icon, beside a count of dashboards.
DASHBOARD_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 13h8V3H3zm0 8h8v-6H3zm10 0h8V11h-8zm0-18v6h8V3z"/></svg>'
)

_USED_CSS = """
.k .used { display: inline-flex; align-items: center; gap: 4px; margin-right: 12px; white-space: nowrap;
  color: var(--mui-palette-text-secondary); font-size: 0.8125rem; }
.k .used svg { width: 14px; height: 14px; fill: currentColor; }
.k .used b { font-weight: 400; font-family: "Roboto", sans-serif; }
"""


def used_html(dashboards: Sequence[t.Any], metrics: Sequence[str]) -> str:
    """How many dashboards and metrics use a table, with their names as tooltips."""
    parts = []
    if dashboards:
        titles = esc(", ".join(d.title for d in dashboards))
        parts.append(f'<span class="used" title="Dashboards: {titles}">{DASHBOARD_SVG}{len(dashboards)}</span>')
    if metrics:
        parts.append(f'<span class="used" title="Metrics: {esc(", ".join(metrics))}"><b>Σ</b>{len(metrics)}</span>')
    return "".join(parts) or '<span class="muted">-</span>'


# Below this width the table drops its Source and Columns columns.
_NARROW = 760

_NARROW_CSS = f"""
@media (max-width: {_NARROW}px) {{
  :host > div:nth-of-type(3), :host > div:nth-of-type(5) {{ display: none !important; }}
}}
"""


class _CatalogTable(Table):
    """The kit's table, dropping the source and columns columns at phone width."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        header = self._layout.objects[1]
        header.stylesheets = [*header.stylesheets, _NARROW_CSS]

    def set_rows(self, rows):
        super().set_rows(rows)
        if rows:
            for row in self._rows:
                row.stylesheets = [*row.stylesheets, _NARROW_CSS]


class CatalogSettings(Viewer):
    """
    The catalog: the table of the sources' tables and a drawer showing one
    of them. Only an ``editable`` one turns tables off for the assistant
    and deletes uploads; anyone the uploads settings let upload may add
    data. ``app`` is the session's app, by default the one whose admin view
    builds it.
    """

    editable = param.Boolean(default=True)

    filter = param.String(default="all", doc="""
        The chip filtering the table: ``all``, ``off``, or ``source:<ref>`` for one source.""")

    query = param.String(default="")

    def __init__(
        self, project: Project, editable: bool = True, app: t.Any = None, show_add: bool = True,
        on_select: t.Callable[[tuple[str, str] | None], t.Any] | None = None,
        on_open: t.Callable[[str], t.Any] | None = None, on_change: t.Callable[[], t.Any] | None = None,
        **params,
    ):
        super().__init__(editable=editable, **params)
        self.project = project
        self._app = app if app is not None else current_app()
        # Told the table the drawer shows, or None when it closes; asked to open a dashboard or metric URL;
        # and told a source was added or deleted.
        self._on_select = on_select
        self._on_open = on_open
        self._on_change = on_change
        self._usage: tuple[dict, dict] | None = None
        self._rendering = False
        self._infos: dict[tuple[str, str], TableInfo] = {}
        self._listing_errors: dict[str, str] = {}
        self.selected: tuple[str, str] | None = None
        self._alert = pmui.Alert(
            object="", severity="error", visible=False, sizing_mode="stretch_width", margin=0
        )
        self._search = pmui.TextInput(
            placeholder="Search tables and columns", size="small", width=280, margin=0,
        )
        self._search.param.watch(lambda event: setattr(self, "query", event.new or ""), "value_input")
        self._chips: dict[str, pmui.Chip] = {}
        self._chip_row = pn.Row(margin=0, styles={"gap": "8px", "flex-wrap": "wrap", "align-items": "center"})
        self.uploads = (project.context or {}).get(UPLOADS_KEY)
        self._upload_dialog = (
            AddDataDialog(self.uploads, self._app, on_added=self._on_uploaded) if self.uploads is not None else None
        )
        self._upload = pmui.Button(
            label="Add data", icon="add", variant="contained", margin=0,
            visible=show_add and self.may_add(), css_classes=["flowdash-catalog-upload"],
        )
        self._upload.on_click(lambda _event: self.add_data())
        toolbar = pn.Row(
            self._search,
            self._chip_row,
            pn.layout.HSpacer(),
            self._upload,
            margin=0,
            sizing_mode="stretch_width",
            styles={
                "padding": "10px 12px", "align-items": "center", "gap": "8px", "flex-wrap": "wrap",
                "border-bottom": "1px solid var(--mui-palette-divider)",
            },
        )
        self._table = _CatalogTable(
            [
                ("Assistant", "88px"),
                ("Table", "2fr"),
                ("Source", "1fr"),
                ("Rows", "0.7fr"),
                ("Columns", "2.2fr"),
                ("Used by", "0.9fr"),
            ],
            toolbar=toolbar,
            footer=caption(
                "Off keeps the assistant from listing, describing or querying a table. Dashboards, the "
                "component list and SQL people write still read it.",
                margin=(12, 12),
            ),
            empty="No tables match.",
            css_classes=["flowdash-admin-catalog"],
        )
        self._build_drawer()
        self._more = caption("", visible=False, margin=(8, 0, 0, 0))
        self._layout = pn.Column(
            self._alert,
            self._table,
            self._more,
            self._drawer,
            self._confirm_dialog,
            *([self._upload_dialog] if self._upload_dialog is not None else []),
            margin=0,
            sizing_mode="stretch_width",
            css_classes=["flowdash-admin-catalog-section"],
        )
        self.refresh()
        self._read_infos()

    # ------------------------------------------------------------------
    # Data

    def may_add(self) -> bool:
        """Whether this session may add data: an admin always, others where the uploads settings let them."""
        if self.uploads is None:
            return False
        return self.editable or self.uploads.offered(self._app)

    def add_data(self, mode: str = "upload"):
        """Open the Add data dialog."""
        if self._upload_dialog is not None and self.may_add():
            self._upload_dialog.open(mode)

    def usage(self) -> tuple[dict[tuple[str, str], list[t.Any]], dict[tuple[str, str], list[str]]]:
        """
        The dashboards this session may open that read each table, and the
        metrics computed from it, by source ref and table.
        """
        if self._usage is None:
            from .data import metric_tables, table_usage, visible_dashboards
            from .metrics_edit import raw_section as metrics_section

            context = self.project.context or {}
            metrics: dict[tuple[str, str], list[str]] = {}
            try:
                section = metrics_section(self.project)
            except ProjectError:
                section = {}
            for name, spec in section.items():
                for key in metric_tables(spec, context):
                    metrics.setdefault(key, []).append(str(name))
            self._usage = (table_usage(visible_dashboards(self._app), context), metrics)
        return self._usage

    def sources(self) -> dict[str, Source]:
        """The sources the app serves, by ref, in the order given."""
        from .nodes import host_sources

        return host_sources(self.project.context or {})

    def section(self) -> dict[str, t.Any]:
        """The ``catalog:`` section as the file has it now."""
        try:
            return raw_section(self.project)
        except (ProjectError, OSError) as exc:
            problems = exc.problems if isinstance(exc, ProjectError) else [Problem(error_text(exc))]
            self._show_problems(problems)
            return {}

    def rows(self) -> list[dict[str, t.Any]]:
        """What the table shows for each table of each source, in the order the sources are served."""
        tables, self._listing_errors = catalog_tables(self.project.context or {}, self.section())
        rows = []
        for entry in tables:
            key = (entry.ref, entry.table)
            info = self._infos.get(key)
            if info is None and (known := known_info(entry.source, entry.table)) is not None:
                info = self._infos[key] = known
            rows.append({
                "ref": entry.ref, "source": entry.source, "table": entry.table, "assistant": entry.assistant,
                "info": info or TableInfo(), "upload": entry.upload,
            })
        return rows

    def _passes(self, row: dict[str, t.Any]) -> bool:
        if self.filter == "off" and row["assistant"]:
            return False
        if self.filter.startswith("source:") and row["ref"] != self.filter[7:]:
            return False
        info = row["info"]
        haystack = " ".join([row["ref"], row["table"], info.description, *(info.columns or [])]).lower()
        return all(word in haystack for word in self.query.lower().split())

    def visible_rows(self) -> list[dict[str, t.Any]]:
        """The rows passing the search and the filter chip."""
        return [row for row in self.rows() if self._passes(row)]

    def _show_problems(self, problems: Sequence[Problem]):
        self._alert.param.update(object="\n\n".join(p.message for p in problems), visible=bool(problems))

    # ------------------------------------------------------------------
    # Reading the tables' metadata

    def _schedule(self, work: t.Callable[[], t.Any], done: t.Callable[[], t.Any]) -> None:
        from .nodes import in_background

        in_background(work, done)

    def _read_infos(self, rows: Sequence[dict[str, t.Any]] | None = None):
        """Read the rows and columns of the first ``LIST_TABLES`` tables not known yet, then redraw."""
        rows = self.rows() if rows is None else rows
        pending = [
            (row["ref"], row["source"], row["table"]) for row in rows if (row["ref"], row["table"]) not in self._infos
        ][:LIST_TABLES]
        if not pending:
            return
        for ref, _, table in pending:
            self._infos[(ref, table)] = TableInfo(loading=True)

        def work():
            for ref, source, table in pending:
                self._infos[(ref, table)] = table_info(source, table)

        self._schedule(work, self.refresh)

    # ------------------------------------------------------------------
    # The table

    def _set_filter(self, key: str, *_args):
        self.filter = key

    def _chip(self, key: str) -> pmui.Chip:
        if key not in self._chips:
            css = "".join(c if c.isalnum() else "-" for c in key).lower()
            widget = pmui.Chip(label=key, size="medium", margin=0, sx=_CHIP_SX, css_classes=[f"filter-{css}"])
            widget.on_click(partial(self._set_filter, key))
            self._chips[key] = widget
        return self._chips[key]

    @param.depends("filter", "query", watch=True)
    def refresh(self):
        """Draw the chips and the table from the sources and the project file."""
        self._alert.visible = False
        rows = self.rows()
        refs = list(dict.fromkeys(row["ref"] for row in rows))
        counts = {"all": len(rows), "off": sum(not row["assistant"] for row in rows)}
        for ref in refs:
            counts[f"source:{ref}"] = sum(row["ref"] == ref for row in rows)
        # Past a handful of sources the chips would crowd the search; the Data section's nav filters by source too.
        by_source = 1 < len(refs) <= SOURCE_CHIPS or self.filter.startswith("source:")
        labels = {
            "all": "All", **({f"source:{ref}": ref for ref in refs} if by_source else {}),
            "off": "Assistant off",
        }
        if self.filter not in labels:
            self.filter = "all"
            return
        with pn.io.hold():
            chips = []
            for key, label in labels.items():
                active = key == self.filter
                if key == "off" and not counts[key] and not active:
                    continue
                widget = self._chip(key)
                widget.param.update(
                    label=f"{label} {counts.get(key, 0)}",
                    icon="check" if active else None,
                    variant="filled" if active else "outlined",
                    color="primary" if active else "default",
                )
                chips.append(widget)
            self._chip_row[:] = chips
            shown = [row for row in rows if self._passes(row)]
            table_rows = [(_key(row["ref"], row["table"]), self._cells(row)) for row in shown[:ROWS_SHOWN]]
            table_rows += [
                (f"{ref}--error", [
                    cell(""),
                    cell(f'<span class="main">Could not list its tables</span><span class="mono error-text">{esc(message)}</span>'),
                    cell(f'<span class="main">{esc(ref)}</span>'), cell(""), cell(""), cell(""),
                ])
                for ref, message in self._listing_errors.items()
                if self.filter in ("all", f"source:{ref}")
            ]
            self._table.set_rows(table_rows)
            more = len(shown) - ROWS_SHOWN
            self._more.param.update(
                object=f"Showing {ROWS_SHOWN} of {len(shown)} tables; search to find the others.", visible=more > 0,
            )

    def _cells(self, row: dict[str, t.Any]) -> list:
        from .nodes import served_file

        ref, table, info, source = row["ref"], row["table"], row["info"], row["source"]
        if self.editable:
            switch = pmui.Switch(value=row["assistant"], margin=0, css_classes=["flowdash-catalog-assistant"])
            switch.param.watch(lambda event, ref=ref, table=table: self.set_assistant(ref, table, event.new), "value")
        else:
            switch = cell(
                chip("On", "success") if row["assistant"] else chip("Off", "warning"),
                css_classes=["flowdash-catalog-assistant-status"],
            )
        upload = row["upload"]
        if info.description:
            second = f'<span class="main muted small">{esc(info.description)}</span>'
        elif upload is not None:
            second = f'<span class="main muted small">{esc(uploaded_text(upload))}</span>'
        else:
            second = f'<span class="mono">{esc(served_file(source, table) or "") or "&nbsp;"}</span>'
        title = cell(
            f'<div class="table{"" if row["assistant"] else " off"}"><span class="main">{esc(table)}</span>{second}</div>',
            stylesheets=[_ROW_CSS],
        )
        open_drawer = pmui.Clickable(object=title, sizing_mode="stretch_width", margin=0)
        open_drawer.on_click(lambda *_args, ref=ref, table=table: self.open_table(ref, table))
        if info.error:
            rows_text = f'<span class="main muted" title="{esc(info.error)}">Unknown</span>'
            columns_text = f'<span class="main error-text small">{esc(info.error)}</span>'
        elif info.loading:
            rows_text = columns_text = '<span class="main muted">Reading…</span>'
        else:
            rows_text = f'<span class="main">{info.rows:,}</span>' if info.rows is not None else '<span class="muted">-</span>'
            if info.columns is None:
                columns_text = '<span class="muted">-</span>'
            else:
                n = len(info.columns)
                columns_text = (
                    f'<span class="main">{n} column{"" if n == 1 else "s"}</span>'
                    f'<span class="mono">{esc(_column_names(info.columns))}</span>'
                )
        dashboards, metrics = self.usage()
        used = used_html(dashboards.get((ref, table), []), metrics.get((ref, table), []))
        return [
            switch,
            open_drawer,
            cell(f'<span class="main">{esc(ref)}</span><span class="mono">'
                 f'{"Upload, " if upload is not None else ""}{esc(_type_name(source))}</span>'),
            cell(f'<div style="text-align: right">{rows_text}</div>'),
            cell(columns_text),
            cell(used, stylesheets=[_USED_CSS]),
        ]

    def set_assistant(self, ref: str, table: str, on: bool) -> bool:
        """Turn ``table`` of ``ref`` on or off for the assistant, for everyone, writing the project file at once."""
        if not self.editable:
            self.refresh()
            return False
        problems = set_assistant(self.project, ref, table, on, app=self._app)
        self.refresh()
        if problems:
            self._show_problems(problems)
            return False
        if self.selected == (ref, table):
            self._render_head()
        return True

    # ------------------------------------------------------------------
    # The drawer

    def _build_drawer(self):
        self._drawer_title = pmui.Typography("", variant="h6", margin=0, sx={"fontWeight": 400})
        self._drawer_subtitle = caption("")
        close = pmui.IconButton(icon="close", description="Close", size="small", margin=0)
        close.on_click(lambda _event: self.close_table())
        self._details = cell("", stylesheets=[_DETAILS_CSS])
        self._columns = cell("", stylesheets=[_DETAILS_CSS])
        self._sample = cell("", stylesheets=[_DETAILS_CSS])
        self._drawer_switch = pmui.Switch(value=True, margin=0, css_classes=["flowdash-catalog-drawer-assistant"])
        self._drawer_switch.param.watch(self._on_drawer_switch, "value")
        self._switch_text = pmui.Typography("", variant="body2", margin=0)
        self._switch_row = pn.Row(
            pmui.Typography("Assistant", variant="body2", margin=0, sx={"color": "text.secondary"}, width=120),
            self._drawer_switch, self._switch_text,
            margin=0, sizing_mode="stretch_width", visible=self.editable,
            styles={"align-items": "center", "gap": "16px"},
        )
        self._used_links = pn.Row(margin=0, sizing_mode="stretch_width", styles={"flex-wrap": "wrap", "gap": "0 4px"})
        self._used_row = pn.Row(
            pmui.Typography("Used by", variant="body2", margin=(6, 0, 0, 0), sx={"color": "text.secondary"},
                            width=120, styles={"flex": "0 0 120px"}),
            self._used_links,
            margin=0, sizing_mode="stretch_width", styles={"gap": "16px", "align-items": "flex-start"},
            css_classes=["flowdash-catalog-used-by"],
        )
        self._first_rows = pmui.Card(
            self._sample, title="First rows", collapsible=True, collapsed=False, margin=0, title_variant="body1",
            sizing_mode="stretch_width", css_classes=["flowdash-catalog-first-rows"],
        )
        self._detail_body = pn.Column(
            pmui.Typography("Columns", variant="overline", sx=_OVERLINE_SX, margin=(12, 0, 0, 0)),
            self._columns,
            self._first_rows,
            margin=0,
            sizing_mode="stretch_width",
            styles={"gap": "8px"},
            css_classes=["flowdash-catalog-detail"],
        )
        self._loading = pmui.LoadingSpinner(value=True, size=28, margin=(24, 0), visible=False)
        done = pmui.Button(label="Done", variant="contained", margin=0)
        done.on_click(lambda _event: self.close_table())
        self._delete = pmui.Button(
            label="Delete upload", icon="delete", variant="outlined", color="error", margin=0, visible=False,
            css_classes=["flowdash-catalog-delete"],
        )
        self._delete.on_click(lambda _event: self.ask_delete())
        self._build_confirm()
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
            self._switch_row,
            self._details,
            self._used_row,
            self._loading,
            self._detail_body,
            margin=0,
            sizing_mode="stretch_width",
            styles={
                "gap": "12px", "padding": "4px 20px 20px 20px", "overflow-y": "auto", "flex": "1 1 auto",
                "min-height": "0", "box-sizing": "border-box",
            },
        )
        foot = pn.Row(
            self._delete,
            pn.layout.HSpacer(),
            done,
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
            css_classes=["flowdash-admin-catalog-drawer"],
        )
        self._drawer = pmui.Drawer(
            body, anchor="right", variant="temporary", size=680, open=False, sx={"maxWidth": "100vw"},
        )
        self._drawer.param.watch(self._on_drawer_open, "open")
        self._detail: TableDetail | None = None

    def _build_confirm(self):
        self._confirm_text = cell("")
        cancel = pmui.Button(label="Cancel", variant="outlined", color="default", margin=0)
        cancel.on_click(lambda _event: setattr(self._confirm_dialog, "open", False))
        confirm = pmui.Button(
            label="Delete upload", variant="contained", color="error", margin=0,
            css_classes=["flowdash-catalog-confirm-delete"],
        )
        confirm.on_click(lambda _event: self.delete_upload())
        self._confirm_dialog = pmui.Dialog(
            pn.Column(
                self._confirm_text,
                pn.Row(pn.layout.HSpacer(), cancel, confirm, margin=0, sizing_mode="stretch_width",
                       styles={"gap": "8px"}),
                margin=0, sizing_mode="stretch_width", styles={"gap": "16px"},
            ),
            title="Delete upload?", open=False, show_close_button=True, width_option="sm",
        )

    def _on_uploaded(self, upload: t.Any):
        self._usage = None
        if self._on_change is not None:
            self._on_change()
            return
        self.refresh()
        self._read_infos()

    def ask_delete(self):
        """Ask before deleting the upload whose table the drawer shows."""
        upload = self.uploads.get(self.selected[0]) if self.uploads is not None and self.selected else None
        if upload is None:
            return
        tables = ", ".join(f"<code>{esc(t)}</code>" for t in upload.tables)
        self._confirm_text.object = (
            f'<div class="k">{esc(upload.filename)} and its tables {tables} are deleted for everyone. '
            "Dashboards that read them stop working.</div>"
        )
        self._confirm_dialog.open = True

    def delete_upload(self) -> bool:
        """Delete the upload whose table the drawer shows, for everyone."""
        if self.uploads is None or self.selected is None:
            return False
        ref = self.selected[0]
        problems = self.uploads.delete(ref, app=self._app if self.editable else None)
        self._confirm_dialog.open = False
        if problems:
            self._show_problems(problems)
            return False
        self.close_table()
        for key in [key for key in self._infos if key[0] == ref]:
            del self._infos[key]
        self.refresh()
        if self._app is not None and hasattr(self._app, "_notify"):
            self._app._notify("success", f"Deleted the upload {ref}.")
        if self._on_change is not None:
            self._on_change()
        return True

    def _on_drawer_open(self, event):
        if not event.new:
            self.selected = None
            if self._on_select is not None:
                self._on_select(None)

    def _on_drawer_switch(self, event):
        if self.selected is None or self._rendering:
            return
        ref, table = self.selected
        self.set_assistant(ref, table, event.new)

    def _row(self, ref: str, table: str) -> dict[str, t.Any] | None:
        return next((r for r in self.rows() if r["ref"] == ref and r["table"] == table), None)

    def open_table(self, ref: str, table: str):
        """Open the drawer on ``table`` of the source ``ref`` and read its columns and first rows."""
        row = self._row(ref, table)
        if row is None:
            return
        self.selected = (ref, table)
        self._detail = None
        with pn.io.hold():
            self._render_head()
            self._loading.visible = True
            self._detail_body.visible = False
            self._drawer.open = True
        if self._on_select is not None:
            self._on_select((ref, table))
        source = row["source"]
        result: list[TableDetail] = []

        def work():
            result.append(table_detail(source, table))

        def done():
            if self.selected != (ref, table) or not result:
                return
            detail = result[0]
            self._detail = detail
            info = self._infos.get((ref, table))
            if info is None or info.loading or info.rows is None:
                self._infos[(ref, table)] = TableInfo(
                    rows=detail.rows, columns=[c.name for c in detail.columns],
                    description=detail.description, error=detail.error if not detail.columns else None,
                )
                self.refresh()
            self._render_detail()

        self._schedule(work, done)

    def _render_head(self):
        from .nodes import served_file

        if self.selected is None:
            return
        ref, table = self.selected
        row = self._row(ref, table)
        if row is None:
            return
        source, detail = row["source"], self._detail
        info = row["info"]
        self._drawer_title.object = table
        kind = "an uploaded" if row["upload"] is not None else f"a {_type_name(source)}"
        self._drawer_subtitle.object = f"Table of {ref}, {kind} source"
        self._delete.visible = row["upload"] is not None and self.editable
        self._rendering = True
        try:
            self._drawer_switch.value = row["assistant"]
        finally:
            self._rendering = False
        self._switch_text.object = "Uses it" if row["assistant"] else "Turned off"
        assistant = chip("Uses it", "success") if row["assistant"] else chip("Turned off", "warning")
        rows = detail.rows if detail is not None else info.rows
        description = (detail.description if detail is not None else "") or info.description
        details = [] if self.editable else [("Assistant", assistant)]
        details.append(("Rows", f"{rows:,}" if rows is not None else '<span class="muted">Reading…</span>'))
        upload = row["upload"]
        if upload is not None:
            when = upload.when
            details.append(("Uploaded by", esc(upload.uploaded_by)))
            if when is not None:
                details.append(("Uploaded", f"{when.astimezone():%d %b %Y, %H:%M}"))
            details.append(("File", f"<code>{esc(upload.filename)}</code>" + (
                f" (sheet {esc(upload.sheet)})" if upload.sheet else ""
            )))
        else:
            if where := served_file(source, table):
                details.append(("File", f"<code>{esc(where)}</code>"))
            details.append(("Source", f"{esc(ref)}, {esc(_type_name(source))}"))
        if description:
            details.append(("Description", esc(description)))
        if detail is not None:
            details += [(esc(key.replace("_", " ").capitalize()), esc(plain_text(value))) for key, value in detail.metadata.items()]
            if detail.error:
                details.append(("Problem", f'<span class="error-text">{esc(detail.error)}</span>'))
        self._details.object = '<div class="k"><dl>' + "".join(
            f"<dt>{label}</dt><dd>{value}</dd>" for label, value in details
        ) + "</dl></div>"
        self._render_used(ref, table)

    def _render_used(self, ref: str, table: str):
        """The dashboards and metrics using the table, each a link."""
        dashboards, metrics = self.usage()
        links = [
            *((f"/dash/{d.dashboard_id}", d.title, "space_dashboard") for d in dashboards.get((ref, table), [])),
            *((f"/data/metrics/{name}", name, "functions") for name in metrics.get((ref, table), [])),
        ]
        buttons = []
        for url, label, icon in links[:USED_SHOWN]:
            button = pmui.Button(
                label=label, icon=icon, variant="text", size="small", margin=0,
                disabled=self._on_open is None, sx={"textTransform": "none", "minWidth": 0},
            )
            button.on_click(lambda _event, url=url: self._on_open(url) if self._on_open is not None else None)
            buttons.append(button)
        if len(links) > USED_SHOWN:
            buttons.append(caption(f"and {len(links) - USED_SHOWN} more", margin=(6, 4, 0, 4)))
        self._used_links[:] = buttons or [caption("Nothing yet", margin=(6, 0, 0, 0))]

    def _render_detail(self):
        detail = self._detail
        if detail is None:
            return
        with pn.io.hold():
            self._render_head()
            self._columns.object = f'<div class="k">{_columns_html(detail.columns)}</div>'
            if detail.sample_error:
                sample = f'<span class="error-text">{esc(detail.sample_error)}</span>'
            elif detail.sample is None:
                sample = '<span class="muted">Not a SQL source; open it in a dashboard to see its rows.</span>'
            else:
                sample = _sample_html(detail.sample)
            self._sample.object = f'<div class="k">{sample}</div>'
            n = len(detail.sample) if detail.sample is not None else 0
            self._first_rows.param.update(
                title=f"First rows ({n})" if n else "First rows",
                # A wide table's columns already fill the drawer, so its rows wait to be unfolded.
                collapsed=len(detail.columns) > FOLD_COLUMNS,
            )
            self._loading.visible = False
            self._detail_body.visible = True

    def close_table(self):
        """Close the drawer."""
        if self._drawer.open:
            self._drawer.open = False
        self.selected = None

    def __panel__(self):
        return self._layout


def catalog_settings(project: Project, editable: bool) -> CatalogSettings:
    """Build the admin view's Catalog section."""
    return CatalogSettings(project, editable)
