"""
The Catalog section of panel-flowdash's admin view: the sources the app
serves and their tables, and which of them Lumen's assistant may use.

The table lists every table of every source with its row and column
counts, read off the event loop; its switch turns a table off or on for
the assistant at once, writing the project file's ``catalog:`` section (see
``lumen.flowdash.catalog``). A table's drawer shows its columns with their
types, value ranges and descriptions, its other metadata and its first
rows. Turning a table off or on checks again that the caller is an admin.
"""
from __future__ import annotations

import asyncio
import copy
import datetime as dt
import logging
import math
import numbers
import typing as t

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial

import panel as pn
import panel_material_ui as pmui
import param

from panel.viewable import Viewer
from panel_flowdash.admin.form import current_app
from panel_flowdash.admin.kit import (
    Table, caption, cell, chip, esc,
)
from panel_flowdash.project import (
    _FILE_LOCK, Problem, Project, ProjectError,
)
from panel_flowdash.value_text import error_text, redact_secrets

from ..sources.base import BaseSQLSource, Source
from ..transforms.sql import SQLLimit
from .catalog import SECTION, raw_section

logger = logging.getLogger(__name__)

# The tables whose row and column counts the list reads by itself; the others when their drawer opens.
LIST_TABLES = 50
# The rows a table's drawer shows, and how long reading them may take.
SAMPLE_ROWS = 10
SAMPLE_TIMEOUT = 5
# The values of a column the drawer lists before saying how many more there are.
VALUES_SHOWN = 6
# The column names a row lists.
NAMES_SHOWN = 6
# The rows the table draws at most; a search narrows down the others.
ROWS_SHOWN = 100

NOT_ADMIN = "Only admins can change what the assistant uses, so this was not saved."

# Metadata the drawer shows elsewhere, or that only Lumen uses.
_SHOWN_KEYS = frozenset({"description", "columns", "rows", "docs", "source_name", "data_type"})

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
# Reading the sources


@dataclass
class TableInfo:
    """What the list shows of a table: its rows, columns and description, or why they could not be read."""

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


@dataclass
class TableDetail:
    """What a table's drawer shows."""

    rows: int | None = None
    description: str = ""
    columns: list[ColumnInfo] = field(default_factory=list)
    metadata: dict[str, t.Any] = field(default_factory=dict)
    sample: t.Any = None
    error: str | None = None
    sample_error: str | None = None


def _safe(exc: BaseException) -> str:
    lines = redact_secrets(error_text(exc)).strip().splitlines()
    return lines[0] if lines else type(exc).__name__


def _count(value: t.Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number == value else None


def _text(value: t.Any) -> str:
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
        return TableInfo(error=_safe(exc))
    columns = metadata.get("columns")
    return TableInfo(
        rows=_count(metadata.get("rows")),
        columns=[str(c) for c in columns] if isinstance(columns, Mapping) else None,
        description=_text(metadata.get("description")),
    )


def _plain(value: t.Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat(sep=" ") if isinstance(value, dt.datetime) else value.isoformat()
    return str(value)


def _values(spec: Mapping[str, t.Any]) -> str:
    if isinstance(enum := spec.get("enum"), list) and enum:
        shown = ", ".join(_plain(v) for v in enum[:VALUES_SHOWN] if v is not None)
        return shown + (f" +{len(enum) - VALUES_SHOWN} more" if len(enum) > VALUES_SHOWN else "")
    low, high = spec.get("inclusiveMinimum"), spec.get("inclusiveMaximum")
    if low is not None or high is not None:
        return f"{_plain(low)} to {_plain(high)}"
    return ""


def _type(spec: Mapping[str, t.Any], metadata: Mapping[str, t.Any]) -> str:
    kind = spec.get("type") or metadata.get("data_type") or ""
    kind = "/".join(map(str, kind)) if isinstance(kind, list) else str(kind)
    if not (fmt := spec.get("format")):
        return kind
    return str(fmt) if kind in ("string", "") else f"{kind} ({fmt})"


def sample_sql(source: BaseSQLSource, table: str, rows: int = SAMPLE_ROWS) -> str:
    """The statement reading the first ``rows`` rows of ``table``."""
    return SQLLimit(limit=rows, write=source.dialect).apply(source.get_sql_expr(source.normalize_table(table)))


def table_detail(source: Source, table: str, timeout: float = SAMPLE_TIMEOUT) -> TableDetail:
    """
    The columns of ``table`` with their types, value ranges and
    descriptions, its other metadata and, for a SQL source, its first rows.
    Reading the schema counts the rows and the value ranges, so this queries
    the table; run it off the event loop.
    """
    from .nodes import execute

    detail = TableDetail()
    try:
        metadata = source.get_metadata(table)
    except Exception as exc:
        metadata = {}
        detail.error = _safe(exc)
    try:
        schema = source.get_schema(table)
    except Exception as exc:
        schema = {}
        detail.error = detail.error or _safe(exc)
    columns_meta = metadata.get("columns") if isinstance(metadata.get("columns"), Mapping) else {}
    detail.rows = _count(schema.get("__len__")) if isinstance(schema, Mapping) else None
    if detail.rows is None:
        detail.rows = _count(metadata.get("rows"))
    detail.description = _text(metadata.get("description"))
    names = [name for name in (schema or {}) if name != "__len__"] or list(columns_meta)
    for name in names:
        spec = schema.get(name) if isinstance(schema.get(name), Mapping) else {}
        column = columns_meta.get(name)
        column = column if isinstance(column, Mapping) else {"description": column} if isinstance(column, str) else {}
        detail.columns.append(ColumnInfo(
            str(name), _type(spec, column), _values(spec), _text(column.get("description")),
        ))
    detail.metadata = {
        str(key): value for key, value in metadata.items()
        if key not in _SHOWN_KEYS and value not in (None, "", [], {})
    }
    if isinstance(source, BaseSQLSource):
        try:
            detail.sample = execute(source, sample_sql(source, table), timeout)
        except Exception as exc:
            detail.sample_error = _safe(exc)
    return detail


# ---------------------------------------------------------------------------
# Writing the section


def _admin(app: t.Any) -> bool:
    try:
        return app is not None and bool(app._still_admin())
    except Exception as exc:
        logger.warning("Could not check whether this session is still an admin: %s", error_text(exc))
        return False


def _prune(section: dict[str, t.Any]) -> dict[str, t.Any]:
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
    project: Project, ref: str, table: str, on: bool, *, app: t.Any,
) -> list[Problem]:
    """
    Turn ``table`` of the source ``ref`` on or off for the assistant,
    writing the project file's ``catalog:`` section and keeping the rest of
    it as the file has it now. Turning it on also clears the setting under
    an alias of the source, e.g. one written by hand. Only an admin
    (``app._still_admin()``) may. Returns the problems that stopped it.
    """
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


def hidden_by_source(section: Mapping[str, t.Any], context: Mapping[str, t.Any]) -> dict[int, set[str]]:
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


# ---------------------------------------------------------------------------
# The view


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
            f'title="{esc(_plain(v))}">{esc(_plain(v))}</td>'
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
    """The Catalog section: the table of the sources' tables and a drawer showing one of them."""

    editable = param.Boolean(default=True)

    filter = param.String(default="all", doc="""
        The chip filtering the table: ``all``, ``off``, or ``source:<ref>`` for one source.""")

    query = param.String(default="")

    def __init__(self, project: Project, editable: bool = True, **params):
        super().__init__(editable=editable, **params)
        self.project = project
        self._app = current_app()
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
        toolbar = pn.Row(
            self._search,
            self._chip_row,
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
                ("Rows", "0.8fr"),
                ("Columns", "2.6fr"),
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
            margin=0,
            sizing_mode="stretch_width",
            css_classes=["flowdash-admin-catalog-section"],
        )
        self.refresh()
        self._read_infos()

    # ------------------------------------------------------------------
    # Data

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
        from .nodes import listed_tables

        hidden = hidden_by_source(self.section(), self.project.context or {})
        rows = []
        for ref, source in self.sources().items():
            try:
                tables = listed_tables(source)
            except Exception as exc:
                self._listing_errors[ref] = _safe(exc)
                continue
            self._listing_errors.pop(ref, None)
            off = hidden.get(id(source), set())
            for table in tables:
                info = self._infos.get((ref, table))
                if info is None and (known := known_info(source, table)) is not None:
                    info = self._infos[(ref, table)] = known
                rows.append({
                    "ref": ref, "source": source, "table": table, "assistant": table not in off,
                    "info": info or TableInfo(),
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
        """Run ``work`` on a worker and then ``done`` on the session's event loop; both at once outside a server."""
        from .nodes import _in_served_session, _preview_pool, _schedule

        if not _in_served_session():
            work()
            done()
            return
        doc = pn.state.curdoc
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def finished(future):
            if not future.cancelled():
                if (exc := future.exception()) is not None:
                    logger.warning("Could not read the catalog: %s", error_text(exc))
                _schedule(done, doc, loop)

        _preview_pool().submit(work).add_done_callback(finished)

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
        refs = list(self.sources())
        counts = {"all": len(rows), "off": sum(not row["assistant"] for row in rows)}
        for ref in refs:
            counts[f"source:{ref}"] = sum(row["ref"] == ref for row in rows)
        labels = {"all": "All", **({f"source:{ref}": ref for ref in refs} if len(refs) > 1 else {}), "off": "Off"}
        if self.filter not in labels:
            self.filter = "all"
            return
        with pn.io.hold():
            chips = []
            for key, label in labels.items():
                active = key == self.filter
                if key == "off" and not counts["off"] and not active:
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
                    cell(f'<span class="main">{esc(ref)}</span>'), cell(""), cell(""),
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
        switch = pmui.Switch(
            value=row["assistant"], margin=0, disabled=not self.editable, css_classes=["flowdash-catalog-assistant"],
        )
        switch.param.watch(lambda event, ref=ref, table=table: self.set_assistant(ref, table, event.new), "value")
        where = served_file(source, table)
        second = info.description or where or ""
        title = cell(
            f'<div class="table{"" if row["assistant"] else " off"}"><span class="main">{esc(table)}</span>'
            f'<span class="{"main muted small" if info.description else "mono"}">{esc(second) or "&nbsp;"}</span></div>',
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
        return [
            switch,
            open_drawer,
            cell(f'<span class="main">{esc(ref)}</span><span class="mono">{esc(_type_name(source))}</span>'),
            cell(rows_text),
            cell(columns_text),
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
        self._detail_body = pn.Column(
            pmui.Typography("Columns", variant="overline", sx=_OVERLINE_SX, margin=(12, 0, 0, 0)),
            self._columns,
            pmui.Typography("First rows", variant="overline", sx=_OVERLINE_SX, margin=(12, 0, 0, 0)),
            self._sample,
            margin=0,
            sizing_mode="stretch_width",
            styles={"gap": "8px"},
            css_classes=["flowdash-catalog-detail"],
        )
        self._loading = pmui.LoadingSpinner(value=True, size=28, margin=(24, 0), visible=False)
        done = pmui.Button(label="Done", variant="contained", margin=0)
        done.on_click(lambda _event: self.close_table())
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
            self._details,
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

    def _on_drawer_open(self, event):
        if not event.new:
            self.selected = None

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
        self._drawer_subtitle.object = f"Table of {ref}, a {_type_name(source)} source"
        assistant = chip("Uses it", "success") if row["assistant"] else chip("Turned off", "warning")
        rows = detail.rows if detail is not None else info.rows
        description = (detail.description if detail is not None else "") or info.description
        details = [
            ("Assistant", assistant),
            ("Rows", f"{rows:,}" if rows is not None else '<span class="muted">Reading…</span>'),
        ]
        if where := served_file(source, table):
            details.append(("File", f"<code>{esc(where)}</code>"))
        if description:
            details.append(("Description", esc(description)))
        if detail is not None:
            details += [(esc(key.replace("_", " ").capitalize()), esc(_plain(value))) for key, value in detail.metadata.items()]
            if detail.error:
                details.append(("Problem", f'<span class="error-text">{esc(detail.error)}</span>'))
        self._details.object = '<div class="k"><dl>' + "".join(
            f"<dt>{label}</dt><dd>{value}</dd>" for label, value in details
        ) + "</dl></div>"

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
