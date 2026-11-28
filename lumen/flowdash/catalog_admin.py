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
from .access import effective_table_access
from .access_editor import (
    ACCESS_CSS, build_access_editor, effective_access_html, load_access_editor,
)
from .add_data import AddDataDialog
from .catalog import (
    NOT_ADMIN_ACCESS, ColumnInfo, TableDetail, TableInfo, catalog_tables,
    known_info, plain_text, raw_section, safe_error, set_assistant,
    set_table_access, table_detail, table_info,
)
from .metadata import (
    NO_MODEL, NO_STORE, NOT_ADMIN_ENRICH, NOT_ADMIN_REFRESH, EnrichmentModel,
    TableEnrichment, _is_admin, configured_model, description_store,
    enrich_table_sync, refresh_source as _refresh_source,
    refresh_table as _refresh_table,
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
.k .badge { display: inline-block; margin-left: 6px; padding: 0 5px; border-radius: 4px; font-size: 0.6875rem;
  font-weight: 500; background: rgba(var(--mui-palette-primary-mainChannel) / 0.12);
  color: var(--mui-palette-primary-main); vertical-align: middle; }
.k .sub { font-size: 0.75rem; margin-top: 2px; }
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
    has_nulls = any(c.nulls for c in columns)
    has_distinct = any(c.distinct for c in columns)
    described = any(c.description for c in columns)
    none = '<span class="muted">-</span>'
    rows = []
    for c in columns:
        name_html = esc(c.name)
        if c.primary_key:
            name_html += '<span class="badge">PK</span>'
        if c.references:
            name_html += f'<div class="sub muted">-&gt; {esc(c.references)}</div>'
        type_title = f' title="{esc(c.sql_type)}"' if c.sql_type and c.sql_type.lower() != c.type.lower() else ""
        values_html = esc(c.values) if c.values else none
        if c.stats_summary and c.stats_summary != c.values:
            values_html += f'<div class="sub muted">{esc(c.stats_summary)}</div>'
        row = f'<tr><td class="col">{name_html}</td><td{type_title}>{esc(c.type) or "&nbsp;"}</td>'
        if has_nulls:
            row += f'<td class="n">{esc(c.nulls) or none}</td>'
        if has_distinct:
            row += f'<td class="n">{esc(c.distinct) or none}</td>'
        row += f"<td>{values_html}</td>"
        if described:
            row += f"<td>{esc(c.description) or none}</td>"
        row += "</tr>"
        rows.append(row)
    return (
        '<div class="scroll"><table><thead><tr><th>Column</th><th>Type</th>'
        + ("<th>Nulls</th>" if has_nulls else "")
        + ("<th>Distinct</th>" if has_distinct else "")
        + "<th>Values</th>"
        + ("<th>Description</th>" if described else "")
        + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
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
    of them. Only an ``editable`` one turns tables off for the assistant;
    anyone the uploads settings let upload may add data, and may delete
    their own upload, an admin anyone's. ``app`` is the session's app, by
    default the one whose admin view builds it.
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
        self._model_known: bool | None = None
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
                ("", "44px"),
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
        from .access import can_access_table

        context = self.project.context or {}
        tables, self._listing_errors = catalog_tables(context, self.section())
        rows = []
        for entry in tables:
            if not can_access_table(entry.source, entry.table, context, app=self._app):
                continue
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

    def _may_refresh(self) -> bool:
        """Whether this session may refresh metadata, which drops caches every session shares (admins only)."""
        if self.editable and _is_admin(self._app):
            return True
        self._show_problems([Problem(NOT_ADMIN_REFRESH)])
        return False

    def refresh_source(self, ref: str, *, on_done: t.Callable[[], t.Any] | None = None) -> None:
        """
        Invalidate cached schema, metadata and stats for ``ref`` and read its
        tables fresh in the background, profiling only the first few.
        """
        source = self.sources().get(ref)
        if source is None or not self._may_refresh():
            return
        for row in self.rows():
            if row["ref"] == ref:
                self._infos[(ref, row["table"])] = TableInfo(loading=True)
        self.refresh()
        result: list[tuple[dict[str, TableInfo], str | None]] = []

        def work():
            result.append(_refresh_source(source))

        def done():
            if not result:
                return
            infos, error = result[0]
            for key in [k for k in self._infos if k[0] == ref and k[1] not in infos]:
                del self._infos[key]
            for table, info in infos.items():
                self._infos[(ref, table)] = info
            self._usage = None
            if error:
                self._listing_errors[ref] = error
                self._show_problems([Problem(f"Could not refresh {ref}: {error}")])
            else:
                self._listing_errors.pop(ref, None)
                if self._app is not None and hasattr(self._app, "_notify"):
                    n = len(infos)
                    self._app._notify("success", f"Refreshed metadata for {ref} ({n} table{'' if n == 1 else 's'}).")
            self.refresh()
            if self.selected is not None and self.selected[0] == ref:
                self.open_table(*self.selected)
            if on_done is not None:
                on_done()

        self._schedule(work, done)

    def refresh_table(self, ref: str, table: str) -> None:
        """Invalidate cached schema, metadata and stats for one table and read them fresh."""
        source = self.sources().get(ref)
        if source is None or not self._may_refresh():
            return
        self._infos[(ref, table)] = TableInfo(loading=True)
        is_open = self.selected == (ref, table)
        if is_open:
            self._loading.visible = True
            self._drawer_refresh.disabled = True
        self.refresh()
        result: list[TableDetail] = []

        def work():
            result.append(_refresh_table(source, table))

        def done():
            if not result:
                if self.selected == (ref, table):
                    self._loading.visible = False
                    self._drawer_refresh.disabled = False
                return
            detail = result[0]
            self._infos[(ref, table)] = TableInfo(
                rows=detail.rows,
                columns=[c.name for c in detail.columns],
                description=detail.description,
                error=detail.error if not detail.columns else None,
            )
            self.refresh()
            if self.selected == (ref, table):
                self._detail = detail
                self._drawer_refresh.disabled = False
                self._render_detail()
            if detail.error and not detail.columns:
                self._show_problems([Problem(f"Could not refresh {table}: {detail.error}")])
            elif self._app is not None and hasattr(self._app, "_notify"):
                self._app._notify("success", f"Refreshed metadata for {table}.")

        self._schedule(work, done)

    def refresh_selected(self) -> None:
        """Refresh schema, metadata and stats for the table currently shown in the drawer."""
        if self.selected is not None:
            self.refresh_table(*self.selected)

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
                    cell(f'<span class="main">{esc(ref)}</span>'), cell(""), cell(""), cell(""), cell(""),
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
            second = self._file_html(served_file(source, table), "mono")
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
        if self.editable:
            refresh_btn = pmui.IconButton(
                icon="refresh", size="small", margin=0, disabled=info.loading,
                description=f"Read schema and fetch metadata for {table}",
                css_classes=["flowdash-catalog-row-refresh"],
            )
            refresh_btn.on_click(lambda _event, ref=ref, table=table: self.refresh_table(ref, table))
        else:
            refresh_btn = cell("")
        return [
            switch,
            open_drawer,
            cell(f'<span class="main">{esc(ref)}</span><span class="mono">'
                 f'{"Upload, " if upload is not None else ""}{esc(_type_name(source))}</span>'),
            cell(f'<div style="text-align: right">{rows_text}</div>'),
            cell(columns_text),
            cell(used, stylesheets=[_USED_CSS]),
            refresh_btn,
        ]

    def _file_html(self, path: str | None, css: str) -> str:
        """
        The file a table reads, relative to the project's or the uploads'
        folder, else by its name, so the server's folders are not shown; an
        admin gets the whole path as a tooltip.
        """
        from .data import data_roots, display_path

        if not path:
            return '<span class="mono">&nbsp;</span>'
        shown = display_path(path, data_roots(self.project))
        title = f' title="{esc(path)}"' if self.editable and shown != path else ""
        if css == "code":
            return f"<code{title}>{esc(shown)}</code>"
        return f'<span class="mono"{title}>{esc(shown)}</span>'

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
        self._drawer_refresh = pmui.Button(
            label="Refresh metadata", icon="refresh", variant="outlined", size="small", margin=0,
            visible=self.editable, css_classes=["flowdash-catalog-refresh"],
        )
        self._drawer_refresh.on_click(lambda _event: self.refresh_selected())
        self._enrich_btn = pmui.Button(
            label="Enrich with LLM", icon="auto_awesome", variant="outlined", size="small", margin=0,
            visible=self.editable, css_classes=["flowdash-catalog-enrich"],
        )
        self._enrich_btn.on_click(lambda _event: self.start_enrich())
        self._edit_desc_btn = pmui.Button(
            label="Edit descriptions", icon="edit", variant="text", size="small", margin=0,
            visible=self.editable, css_classes=["flowdash-catalog-edit-descriptions"],
        )
        self._edit_desc_btn.on_click(lambda _event: self.open_descriptions_editor())
        self._access_editor = (
            build_access_editor(
                title="Table access rules", framed=True, app=self._app, project=self.project,
                on_save=self._save_table_access, on_cancel=self.close_access_editor,
            )
            if self.editable
            else None
        )
        self._access_btn = pmui.Button(
            label="Access", icon="lock_open", variant="outlined", size="small", margin=0,
            visible=self.editable and self._access_editor is not None, css_classes=["flowdash-catalog-access"],
        )
        self._access_btn.on_click(lambda _event: self.open_access_editor())
        self._drawer_actions = pn.Row(
            self._drawer_refresh,
            self._enrich_btn,
            self._edit_desc_btn,
            self._access_btn,
            margin=0, sizing_mode="stretch_width",
            styles={"gap": "8px", "flex-wrap": "wrap", "align-items": "center"},
            css_classes=["flowdash-catalog-drawer-actions"],
        )
        self._build_enrich_panel()
        self._access_summary = cell("", stylesheets=[ACCESS_CSS], css_classes=["flowdash-catalog-access-text"])
        self._access_row = pn.Row(
            pmui.Typography("Access", variant="body2", margin=(2, 0, 0, 0), sx={"color": "text.secondary"},
                            width=120, styles={"flex": "0 0 120px"}),
            self._access_summary,
            margin=0, sizing_mode="stretch_width", visible=self.editable,
            styles={"gap": "16px", "align-items": "flex-start"},
            css_classes=["flowdash-catalog-access-summary"],
        )
        self._access_panel = pn.Column(
            *([self._access_editor] if self._access_editor is not None else []),
            margin=0, sizing_mode="stretch_width", visible=False,
            css_classes=["flowdash-catalog-access-editor"],
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
            self._drawer_actions,
            self._enrich_panel,
            self._access_panel,
            self._switch_row,
            self._access_row,
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

    def _build_enrich_panel(self):
        self._enrich_origin = "manual"
        self._enrich_running = False
        self._enrich_col_inputs: dict[str, pmui.TextInput] = {}
        self._enrich_title = pmui.Typography("Review descriptions", variant="subtitle2", margin=0)
        self._enrich_caption = caption(
            "Review and edit before saving. Saved descriptions are shared with dashboards and the assistant.",
            margin=(2, 0, 8, 0),
        )
        self._enrich_alert = pmui.Alert(
            object="", severity="error", visible=False, sizing_mode="stretch_width", margin=(0, 0, 8, 0),
        )
        self._enrich_spinner = pn.Row(
            pmui.LoadingSpinner(value=True, size=20, margin=0),
            pmui.Typography("Generating descriptions with the app's model…", variant="body2", margin=0),
            visible=False, margin=(4, 0), styles={"align-items": "center", "gap": "8px"},
        )
        self._enrich_table_input = pmui.TextAreaInput(
            label="Table description", value="", rows=2, sizing_mode="stretch_width", margin=0,
            css_classes=["flowdash-enrich-table-desc"],
        )
        self._enrich_cols_box = pn.Column(
            margin=0, sizing_mode="stretch_width",
            styles={"gap": "8px", "max-height": "280px", "overflow-y": "auto", "padding-top": "4px"},
        )
        self._enrich_cancel = pmui.Button(
            label="Cancel", variant="outlined", color="default", size="small", margin=0,
            css_classes=["flowdash-enrich-cancel"],
        )
        self._enrich_cancel.on_click(lambda _event: self.cancel_enrich())
        self._enrich_regenerate = pmui.Button(
            label="Regenerate", icon="auto_awesome", variant="outlined", size="small", margin=0,
            css_classes=["flowdash-enrich-regenerate"],
        )
        self._enrich_regenerate.on_click(lambda _event: self.start_enrich())
        self._enrich_save = pmui.Button(
            label="Save descriptions", icon="save", variant="contained", color="primary", size="small", margin=0,
            css_classes=["flowdash-enrich-save"],
        )
        self._enrich_save.on_click(lambda _event: self.save_descriptions())
        self._enrich_form = pn.Column(
            self._enrich_table_input,
            pmui.Typography("Column descriptions", variant="overline", sx=_OVERLINE_SX, margin=(8, 0, 0, 0)),
            self._enrich_cols_box,
            pn.Row(
                pn.layout.HSpacer(), self._enrich_cancel, self._enrich_regenerate, self._enrich_save,
                margin=(12, 0, 0, 0), sizing_mode="stretch_width", styles={"gap": "8px"},
            ),
            margin=0, sizing_mode="stretch_width", styles={"gap": "4px"},
        )
        self._enrich_panel = pmui.Paper(
            pn.Column(
                self._enrich_title,
                self._enrich_caption,
                self._enrich_alert,
                self._enrich_spinner,
                self._enrich_form,
                margin=0, sizing_mode="stretch_width",
            ),
            variant="outlined", visible=False, margin=0, sizing_mode="stretch_width",
            sx={"p": "14px 16px", "borderRadius": "8px"},
            css_classes=["flowdash-catalog-enrich-review"],
        )

    def open_descriptions_editor(
        self, draft: TableEnrichment | None = None, *, origin: str = "manual",
    ) -> bool:
        """Open the inline review and edit panel for the selected table's descriptions (admins only)."""
        if not self.editable or self.selected is None:
            return False
        if not _is_admin(self._app):
            self._enrich_alert.param.update(object=NOT_ADMIN_ENRICH, visible=True)
            self._enrich_panel.visible = True
            self._enrich_form.visible = False
            return False
        ref, table = self.selected
        row = self._row(ref, table)
        detail = self._detail
        # Without the loaded columns, saving would replace their descriptions with none.
        if row is None or detail is None:
            return False
        current_desc = detail.description or row["info"].description
        current_cols = {c.name: c.description for c in detail.columns}
        col_names = [c.name for c in detail.columns]
        if draft is not None:
            table_desc = draft.description or current_desc
            col_values = {name: draft.columns.get(name, current_cols.get(name, "")) for name in col_names}
            self._enrich_title.object = "Review suggested descriptions"
        else:
            table_desc = current_desc
            col_values = {name: current_cols.get(name, "") for name in col_names}
            self._enrich_title.object = "Edit descriptions"
        self._enrich_origin = origin
        self._enrich_running = False
        self._enrich_table_input.value = table_desc
        inputs: dict[str, pmui.TextInput] = {}
        widgets = []
        for name in col_names:
            css = "".join(c if c.isalnum() else "-" for c in name).lower()
            field_input = pmui.TextInput(
                label=name, value=col_values.get(name, ""), placeholder=f"Description for {name}",
                size="small", sizing_mode="stretch_width", margin=0,
                css_classes=["flowdash-enrich-col-desc", f"flowdash-enrich-col-{css}"],
            )
            inputs[name] = field_input
            widgets.append(field_input)
        self._enrich_col_inputs = inputs
        self._enrich_cols_box[:] = widgets or [caption("No columns loaded yet.", margin=(4, 0))]
        has_model = self._has_model()
        with pn.io.hold():
            self._enrich_alert.visible = False
            self._enrich_spinner.visible = False
            self._enrich_regenerate.visible = has_model
            self._enrich_regenerate.disabled = False
            self._enrich_save.disabled = False
            self._enrich_btn.disabled = not has_model
            self._enrich_form.visible = True
            self._enrich_panel.visible = True
        return True

    def _has_model(self) -> bool:
        """Whether the session's assistant has a model enrichment can run on, checked once per drawer."""
        if self._model_known is None:
            self._model_known = configured_model(self._app) is not None
        return self._model_known

    def start_enrich(self, *, model: EnrichmentModel | None = None) -> bool:
        """Generate table and column descriptions with the assistant's model for review before saving."""
        if not self.editable or self.selected is None:
            return False
        if not _is_admin(self._app):
            self._enrich_alert.param.update(object=NOT_ADMIN_ENRICH, visible=True)
            self._enrich_panel.visible = True
            self._enrich_form.visible = False
            return False
        if model is None and not self._has_model():
            self._enrich_alert.param.update(object=NO_MODEL, visible=True)
            self._enrich_panel.visible = True
            self._enrich_form.visible = False
            return False
        ref, table = self.selected
        row = self._row(ref, table)
        if row is None or self._detail is None:
            return False
        source = row["source"]
        detail = self._detail
        self._enrich_running = True
        with pn.io.hold():
            self._enrich_title.object = "Enrich with LLM"
            self._enrich_alert.visible = False
            self._enrich_form.visible = False
            self._enrich_spinner.visible = True
            self._enrich_panel.visible = True
            self._enrich_btn.disabled = True
            self._enrich_regenerate.disabled = True
            self._enrich_save.disabled = True
        result: list[TableEnrichment] = []
        errors: list[str] = []

        def work():
            try:
                draft = enrich_table_sync(source, table, ref=ref, app=self._app, model=model, detail=detail)
            except Exception as exc:
                errors.append(safe_error(exc))
            else:
                result.append(draft)

        def done():
            self._enrich_running = False
            if self.selected != (ref, table):
                return
            self._enrich_btn.disabled = not self._has_model()
            self._enrich_spinner.visible = False
            if errors:
                self._enrich_alert.param.update(object=errors[0], visible=True)
                self._enrich_form.visible = False
                self._enrich_panel.visible = True
                return
            if result:
                self.open_descriptions_editor(result[0], origin="llm")

        self._schedule(work, done)
        return True

    def save_descriptions(self) -> bool:
        """Persist the reviewed table and column descriptions and update the source and assistant."""
        # Only from an open editor, whose inputs hold every loaded column's description.
        if not self.editable or self.selected is None or self._detail is None or not self._enrich_form.visible:
            return False
        ref, table = self.selected
        row = self._row(ref, table)
        if row is None:
            return False
        source = row["source"]
        description = (self._enrich_table_input.value or "").strip()
        columns = {name: (widget.value or "").strip() for name, widget in self._enrich_col_inputs.items()}
        store = description_store(self.project)
        if store is None:
            problems = [Problem(NO_STORE)]
        else:
            problems = store.save(
                ref, table, description, columns, source=source, app=self._app, origin=self._enrich_origin,
            )
        if problems:
            self._enrich_alert.param.update(object="\n\n".join(p.message for p in problems), visible=True)
            return False
        if self._detail is not None:
            self._detail.description = description
            for col in self._detail.columns:
                if col.name in columns:
                    col.description = columns[col.name]
        info = self._infos.get((ref, table))
        if info is not None:
            info.description = description
        self.cancel_enrich()
        self._render_detail()
        self.refresh()
        if self._app is not None and hasattr(self._app, "_notify"):
            self._app._notify("success", f"Saved descriptions for {table}.")
        return True

    def cancel_enrich(self) -> None:
        """Close the inline descriptions editor without saving."""
        self._enrich_running = False
        with pn.io.hold():
            self._enrich_alert.visible = False
            self._enrich_spinner.visible = False
            self._enrich_panel.visible = False

    def open_access_editor(self) -> bool:
        """Open the inline access rules editor for the selected table (admins only)."""
        if not self.editable or self.selected is None or self._access_editor is None:
            return False
        ref, table = self.selected
        row = self._row(ref, table)
        if row is None:
            return False
        effective = effective_table_access(ref, row["source"], table, self.project.context or {}, section=self.section())
        load_access_editor(
            self._access_editor, effective, title=f"Access rules for {table}", app=self._app, project=self.project,
        )
        self._access_panel.visible = True
        return True

    def close_access_editor(self) -> None:
        """Close the inline access rules editor without saving."""
        self._access_panel.visible = False
        if self._access_editor is not None:
            self._access_editor.show_problems([])

    def _save_table_access(self, perm: t.Any) -> list[Problem]:
        if not self.editable or self.selected is None:
            return [Problem(NOT_ADMIN_ACCESS)]
        ref, table = self.selected
        problems = set_table_access(self.project, ref, table, perm, app=self._app)
        if problems:
            return problems
        self.close_access_editor()
        self._render_head()
        self.refresh()
        if self._app is not None and hasattr(self._app, "_notify"):
            self._app._notify("success", f"Saved access rules for {table}.")
        return []

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
        problems = self.uploads.delete(ref, app=self._app)
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
            self.cancel_enrich()
            self.close_access_editor()
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
        self.cancel_enrich()
        self.close_access_editor()
        self.selected = (ref, table)
        self._detail = None
        self._model_known = None
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
            self._render_head()
            self._render_detail()
            if detail.stats_pending is not None:
                self._await_stats(ref, table, source, detail.stats_pending)

        self._schedule(work, done)

    def _await_stats(self, ref: str, table: str, source: Source, pending: t.Any) -> None:
        """Show the drawer's table again once statistics that took longer than the drawer waited arrive."""
        result: list[TableDetail] = []

        def work():
            try:
                pending.result()
            except Exception:
                return
            result.append(table_detail(source, table))

        def done():
            if result and self.selected == (ref, table) and self._detail is not None and not self._enrich_panel.visible:
                self._detail = result[0]
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
        self._delete.visible = self.uploads is not None and self.uploads.may_delete(row["upload"], self._app)
        has_model = self._has_model()
        loaded = self._detail is not None
        self._enrich_btn.param.update(
            visible=self.editable,
            disabled=self._enrich_running or not has_model or not loaded,
            description="" if has_model else NO_MODEL,
        )
        self._edit_desc_btn.param.update(visible=self.editable, disabled=not loaded)
        if self.editable:
            effective = effective_table_access(ref, source, table, self.project.context or {}, section=self.section())
            self._access_summary.object = effective_access_html(effective, compact=True)
            self._access_row.visible = True
            self._access_btn.param.update(
                visible=self._access_editor is not None, icon="lock" if not effective.is_empty else "lock_open",
            )
        else:
            self._access_row.visible = False
            self._access_btn.visible = False
            self._access_panel.visible = False
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
        if detail is not None and detail.stats_note:
            details.append(("Stats", esc(detail.stats_note)))
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
                details.append(("File", self._file_html(where, "code")))
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
        self.cancel_enrich()
        self.close_access_editor()
        if self._drawer.open:
            self._drawer.open = False
        self.selected = None

    def __panel__(self):
        return self._layout


def catalog_settings(project: Project, editable: bool) -> CatalogSettings:
    """Build the admin view's Catalog section."""
    return CatalogSettings(project, editable)
