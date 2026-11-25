"""
The Data section of a Lumen FlowDash app's sidebar: the tables of the
sources the app serves, the sources people uploaded or an admin connected,
and the app's metrics, for everyone who can open the app.

``lumen serve --flowdash`` registers it on a panel-flowdash with sidebar
sections (``panel_flowdash.sidebar``), served at ``/data`` (the tables,
``?table=<ref>.<table>`` opening one's drawer and ``?source=<ref>``
filtering by source), ``/data/metrics`` and ``/data/metrics/<name>``. It
shows the catalog (``catalog_admin.CatalogSettings``) and the metrics
(``metrics_admin.MetricsSettings``, ``metric_page.MetricPage``), which then
leave the admin view, whose old routes lead here. Everyone sees it all;
only an admin turns tables off for the assistant, connects and removes
sources, deletes uploads and defines metrics, and anyone the uploads
settings let upload may add files. Where panel-flowdash has no sidebar
sections, ``data_section`` gives None and the admin view keeps its
Catalog and Metrics.
"""
from __future__ import annotations

import logging
import typing as t

from urllib.parse import parse_qs, quote

import panel as pn
import panel_material_ui as pmui

from panel.viewable import Viewer
from panel_flowdash.admin.kit import caption, cell, esc

logger = logging.getLogger(__name__)

SECTION_ID = "data"
ROUTE = "data"

# Below this width the section's nav gives way to buttons above the page, as the admin view's does.
_WIDE = 900

# The admin view's sections that moved here, by the Data page each leads to.
MOVED = {"catalog": f"/{ROUTE}", "metrics": f"/{ROUTE}/metrics"}

_NAV_SX = {
    "py": 0,
    "& .MuiListItemButton-root": {"pl": "21px", "pr": 2, "py": 0.75, "borderLeft": "3px solid transparent"},
    "& .MuiListItemIcon-root": {"minWidth": "34px"},
    "& .MuiListItemText-root": {"display": "flex", "justifyContent": "space-between", "alignItems": "baseline",
                                "gap": "8px", "my": 0},
    "& .MuiListItemText-primary": {"fontSize": "0.9375rem", "letterSpacing": "0.15px", "overflow": "hidden",
                                   "textOverflow": "ellipsis", "whiteSpace": "nowrap"},
    "& .MuiListItemText-secondary": {"fontSize": "0.8125rem", "flex": "0 0 auto"},
    "& .MuiListItemButton-root.Mui-selected": {
        "borderLeftColor": "primary.main", "bgcolor": "rgba(var(--mui-palette-primary-mainChannel) / 0.12)",
    },
    "& .MuiListItemButton-root.Mui-selected .MuiListItemText-primary": {"fontWeight": 500},
}

_SOURCES_SX = {
    **_NAV_SX,
    "& .MuiListItemButton-root": {"pl": "21px", "pr": 2, "py": 0.5, "borderLeft": "3px solid transparent"},
    "& .MuiListItemText-primary": {"fontSize": "0.875rem", "overflow": "hidden", "textOverflow": "ellipsis",
                                   "whiteSpace": "nowrap"},
}

_CAPTION_SX = {"color": "text.secondary", "letterSpacing": "0.25px"}

_TITLE_SX = {"fontWeight": 400, "fontSize": "1.75rem", "lineHeight": 1.3, "letterSpacing": 0}

_CARD_CSS = """
:host { min-width: 0; }
.card { display: flex; gap: 14px; align-items: flex-start; min-width: 0; }
.icon { flex: 0 0 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
  background: rgba(var(--mui-palette-primary-mainChannel) / 0.1); color: var(--mui-palette-primary-main); }
.icon svg { width: 18px; height: 18px; fill: currentColor; }
.text { min-width: 0; }
.name { font-family: "Roboto Mono", ui-monospace, monospace; font-size: 0.9375rem; font-weight: 500; }
.line { font-size: 0.8125rem; color: var(--mui-palette-text-secondary); overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; margin-top: 2px; }
"""

_EMPTY_CSS = """
.empty { text-align: center; padding: 48px 24px 8px 24px; font-family: inherit; color: var(--mui-palette-text-primary); }
.empty svg { width: 40px; height: 40px; fill: var(--mui-palette-primary-main); }
.empty h3 { font-size: 1.25rem; font-weight: 500; margin: 8px 0; }
.empty p { color: var(--mui-palette-text-secondary); max-width: 440px; margin: 0 auto; line-height: 1.5; }
.empty code { font-family: "Roboto Mono", ui-monospace, monospace; font-size: 0.8125rem; }
"""

DATABASE_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3C7.58 3 4 4.79 4 7v10c0 2.21 3.59 4 8 4s8-1.79 8-4V7c0'
    "-2.21-3.58-4-8-4m6 14c0 .5-2.13 2-6 2s-6-1.5-6-2v-2.23c1.61.78 3.72 1.23 6 1.23s4.39-.45 6-1.23zm0-4.55c-1.3.95-"
    "3.58 1.55-6 1.55s-4.7-.6-6-1.55V9.64c1.47.83 3.61 1.36 6 1.36s4.53-.53 6-1.36zM12 9C8.13 9 6 7.5 6 7s2.13-2 6-2"
    ' 6 1.5 6 2-2.13 2-6 2"/></svg>'
)

UPLOAD_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.89 2 1.99 2H18c1.1 0 2-.9'
    " 2-2V8zm4 18H6V4h7v5h5zM8 15.01l1.41 1.41L11 14.84V19h2v-4.16l1.59 1.59L16 15.01 12.01 11z\"/></svg>"
)


def data_section() -> t.Any:
    """The Data section, a ``panel_flowdash.SidebarSection``; None on a panel-flowdash without sidebar sections."""
    try:
        from panel_flowdash.sidebar import SidebarSection
    except ImportError:
        return None
    return SidebarSection(
        id=SECTION_ID, label="Data", icon="storage", order=50, route=ROUTE, view=data_view,
    )


def table_url(ref: str, table: str) -> str:
    """The URL opening the drawer of ``table`` of the source ``ref``."""
    return f"/{ROUTE}?table={quote(f'{ref}.{table}', safe='')}"


def _query(search: str) -> dict[str, str]:
    return {key: values[-1] for key, values in parse_qs(search.lstrip("?")).items() if values}


def _heading(title: str, description: str, *actions: t.Any) -> pn.Column:
    return pn.Column(
        pn.Row(
            pmui.Typography(title, variant="h4", margin=0, sx=_TITLE_SX, css_classes=["lumen-data-title"]),
            pn.layout.HSpacer(),
            *actions,
            margin=0, sizing_mode="stretch_width", styles={"align-items": "center", "gap": "8px"},
        ),
        pmui.Typography(description, variant="body2", margin=(6, 0, 0, 0),
                        sx={"color": "text.secondary", "maxWidth": "760px", "letterSpacing": "0.25px"}),
        margin=(0, 0, 16, 0), sizing_mode="stretch_width",
    )


class DataPage(Viewer):
    """
    The Data section's page at ``/data/<path>?<search>``: the section's
    nav, with Tables, Metrics and the sources as filters, beside the
    tables, the metrics or one metric. ``navigate`` opens a URL in the app.
    """

    def __init__(
        self, app: t.Any, project: t.Any, path: str = "", search: str = "", *, admin: bool = False,
        navigate: t.Callable[[str], t.Any] | None = None, **params,
    ):
        super().__init__(**params)
        from .catalog import catalog_tables, raw_section
        from .data import metric_statuses, source_summaries
        from .uploads import UPLOADS_KEY

        self.app = app
        self.project = project
        self.admin = admin
        self._navigate = navigate
        self.context = project.context or {}
        self.uploads = self.context.get(UPLOADS_KEY)
        parts = [part for part in path.strip("/").split("/") if part]
        self.page = "metric" if parts[:1] == ["metrics"] and len(parts) > 1 else (parts[0] if parts else "tables")
        query = _query(search)
        try:
            section = raw_section(project)
        except Exception:
            section = {}
        tables, _ = catalog_tables(self.context, section)
        self.sources = source_summaries(self.context, tables)
        self.statuses, _ = metric_statuses(project)
        self.n_tables = len(tables)
        if self.page == "metrics":
            content = self._metrics_page()
        elif self.page == "metric":
            content = self._metric_page("/".join(parts[1:]))
        else:
            self.page = "tables"
            content = self._tables_page(query)
        self._layout = pn.Row(
            self._nav(query.get("source")),
            pn.Column(
                self._narrow_nav(), content, margin=0, sizing_mode="stretch_width", min_width=0,
                styles={"flex": "1 1 auto"},
            ),
            sizing_mode="stretch_both", margin=0, min_width=0,
            styles={"min-height": "100%", "align-items": "stretch"},
            css_classes=["lumen-data-page", f"lumen-data-{self.page}"],
        )

    def navigate(self, url: str):
        if self._navigate is not None:
            self._navigate(url)

    # ------------------------------------------------------------------
    # The nav

    def _summary(self) -> str:
        if not self.sources and not self.statuses:
            return "No data yet"
        parts = [(len(self.sources), "source"), (self.n_tables, "table"), (len(self.statuses), "metric")]
        return " · ".join(f"{n} {noun}{'' if n == 1 else 's'}" for n, noun in parts)

    def _footer(self) -> str:
        if self.admin:
            return "Who may upload, file types and size are in Admin, Uploads."
        settings = self.uploads.settings() if self.uploads is not None else None
        if settings is None or not settings.enabled:
            return "Admins add sources and define metrics. Adding files is off in this app."
        if self.uploads.offered(self.app):
            return "Admins add sources and define metrics. You can add files from Tables."
        return "Admins add sources and define metrics. Adding files is open to some people in this app."

    def _nav(self, source: str | None) -> pn.Column:
        pages = pmui.MenuList(
            items=[
                {"label": "Tables", "icon": "table_chart", "secondary": str(self.n_tables), "url": f"/{ROUTE}"},
                {"label": "Metrics", "icon": "functions", "secondary": str(len(self.statuses)),
                 "url": f"/{ROUTE}/metrics"},
            ],
            active=(1,) if self.page in ("metrics", "metric") else ((0,) if not source else None),
            dense=False, margin=0, sizing_mode="stretch_width", sx=_NAV_SX, css_classes=["lumen-data-nav"],
        )
        pages.on_click(lambda item: self.navigate(item["url"]))
        if self.sources:
            refs = [s.ref for s in self.sources]
            sources = pmui.MenuList(
                items=[
                    {"label": s.ref, "icon": "upload_file" if s.upload is not None else "storage",
                     "secondary": s.kind, "url": f"/{ROUTE}?source={quote(s.ref, safe='')}"}
                    for s in self.sources
                ],
                active=(refs.index(source),) if source in refs and self.page == "tables" else None,
                dense=True, margin=0, sizing_mode="stretch_width", sx=_SOURCES_SX,
                css_classes=["lumen-data-nav-sources"],
            )
            sources.on_click(lambda item: self.navigate(item["url"]))
        else:
            sources = pmui.Typography("No sources yet", variant="body2", margin=(4, 24))
        return pn.Column(
            pmui.Typography("Data", variant="h6", margin=(20, 24, 0, 24), sx={"fontWeight": 400}),
            pmui.Typography(self._summary(), variant="caption", margin=(2, 24, 16, 24), sx=_CAPTION_SX,
                            css_classes=["lumen-data-summary"]),
            pages,
            pmui.Typography("Sources", variant="body2", margin=(20, 24, 4, 24),
                            sx={"color": "text.secondary", "fontSize": "0.8125rem", "letterSpacing": "0.4px"}),
            sources,
            pn.layout.VSpacer(),
            pmui.Typography(
                self._footer(), variant="caption", margin=0, sizing_mode="stretch_width",
                sx={**_CAPTION_SX, "borderTop": 1, "borderColor": "divider", "display": "block", "px": 3, "py": 2},
                css_classes=["lumen-data-footer"],
            ),
            # Its width is CSS's alone: a width Panel knows would count towards the page's least width on phones.
            margin=0, sizing_mode="stretch_height",
            styles={"flex": "0 0 240px", "width": "240px", "border-right": "1px solid var(--mui-palette-divider)",
                    "min-height": "calc(100vh - 56px)"},
            stylesheets=[f"@media (max-width: {_WIDE - 0.05}px) {{ :host {{ display: none !important; }} }}"],
        )

    def _narrow_nav(self) -> pn.Row:
        """Tables and Metrics as buttons above the page, where the screen has no room for the section's nav."""
        buttons = []
        for label, url, current in (("Tables", f"/{ROUTE}", self.page == "tables"),
                                    ("Metrics", f"/{ROUTE}/metrics", self.page in ("metrics", "metric"))):
            button = pmui.Button(label=label, variant="contained" if current else "outlined", size="small", margin=0)
            button.on_click(lambda _event, url=url: self.navigate(url))
            buttons.append(button)
        return pn.Row(
            *buttons, margin=(16, 16, 0, 16), styles={"gap": "8px"}, css_classes=["lumen-data-narrow-nav"],
            stylesheets=[f"@media (min-width: {_WIDE}px) {{ :host {{ display: none !important; }} }}"],
        )

    # ------------------------------------------------------------------
    # Tables

    def _tables_page(self, query: dict[str, str]) -> pn.Column:
        from .catalog_admin import CatalogSettings

        self.catalog = CatalogSettings(
            self.project, editable=self.admin, app=self.app, show_add=False,
            on_select=self._on_table, on_open=self.navigate, on_change=lambda: self.navigate(f"/{ROUTE}"),
        )
        add = pmui.Button(
            label="Add data", icon="add", variant="contained", margin=0, visible=self.catalog.may_add(),
            css_classes=["lumen-data-add"],
        )
        add.on_click(lambda _event: self.catalog.add_data())
        description = (
            "The tables the app serves, from every source. Dashboards read them, and the assistant reads the ones "
            "that are on." if self.admin else
            "The tables the app serves. Dashboards read them, and the assistant reads the ones an admin turned on."
        )
        heading = _heading("Tables", description, add)
        if not self.sources:
            return self._content(heading, self._empty(), self.catalog._upload_dialog or pn.Spacer(height=0))
        if (source := query.get("source")) and source in {s.ref for s in self.sources}:
            self.catalog.filter = f"source:{source}"
        if (table := query.get("table")) and "." in table:
            ref, name = table.split(".", 1)
            self.catalog.open_table(ref, name)
        return self._content(heading, self._cards(), self.catalog)

    def _content(self, *objects: t.Any) -> pn.Column:
        return pn.Column(
            *objects, margin=0, sizing_mode="stretch_width", min_width=0,
            styles={"padding": "24px 32px 48px 32px", "box-sizing": "border-box", "gap": "0"},
            stylesheets=[f"@media (max-width: {_WIDE - 0.05}px) {{ :host {{ padding: 16px !important; }} }}"],
        )

    def _empty(self) -> pmui.Paper:
        if self.catalog.may_add():
            text = ("Upload a file or connect a database. Its tables show up here, and dashboards and the "
                    "assistant can read them.") if self.admin else (
                    "Upload a file. Its tables show up here, and dashboards and the assistant can read them.")
            add = pmui.Button(label="Add data", icon="add", variant="contained", margin=(16, 0, 0, 0), align="center",
                              css_classes=["lumen-data-empty-add"])
            add.on_click(lambda _event: self.catalog.add_data())
            actions = [pn.Row(pn.layout.HSpacer(), add, pn.layout.HSpacer(), margin=0, sizing_mode="stretch_width")]
        else:
            text, actions = "An admin can add files or connect a database.", []
        note = (
            '<p style="font-size: 0.8125rem; margin-top: 20px">Files given on the command line, like '
            "<code>lumen serve turbines.csv</code>, show up here too.</p>" if self.admin else ""
        )
        return pmui.Paper(
            cell(f'<div class="empty">{DATABASE_SVG}<h3>No data yet</h3><p>{esc(text)}</p></div>',
                 stylesheets=[_EMPTY_CSS]),
            *actions,
            cell(f'<div class="empty" style="padding: 0 24px 40px 24px">{note}</div>', stylesheets=[_EMPTY_CSS]),
            variant="outlined", margin=0, sizing_mode="stretch_width", css_classes=["lumen-data-empty"],
        )

    def _cards(self) -> pn.FlexBox:
        cards = []
        for summary in self.sources:
            files = ", ".join(summary.files)
            line = f"{summary.kind} · {files}" if files else summary.kind
            noun = "table" if summary.tables == 1 else "tables"
            body = cell(
                f'<div class="card"><div class="icon">{UPLOAD_SVG if summary.upload is not None else DATABASE_SVG}'
                f'</div><div class="text"><div class="name">{esc(summary.ref)}</div>'
                f'<div class="line" title="{esc(line)}">{esc(line)}</div>'
                f'<div class="line">{summary.tables} {noun} · {esc(summary.origin)}</div></div></div>',
                stylesheets=[_CARD_CSS],
            )
            open_it = pmui.Clickable(object=body, sizing_mode="stretch_width", margin=0)
            open_it.on_click(lambda *_args, ref=summary.ref: self.navigate(f"/{ROUTE}?source={quote(ref, safe='')}"))
            parts = [open_it]
            if self.admin and (summary.upload is not None or summary.connection is not None):
                menu = pmui.MenuButton(
                    items=[{"label": "Delete upload" if summary.upload is not None else "Remove source",
                            "icon": "delete", "action": "remove"}],
                    icon="more_vert", label="", variant="text", color="default", size="small", margin=0,
                    description=f"More actions for {summary.ref}", css_classes=["lumen-data-source-menu"],
                    sx={"minWidth": 0, "& .MuiButton-endIcon": {"display": "none"}},
                )
                menu.on_click(lambda _item, ref=summary.ref: self.remove_source(ref))
                parts.append(menu)
            cards.append(pmui.Paper(
                pn.Row(*parts, margin=0, sizing_mode="stretch_width", styles={"align-items": "flex-start"}),
                variant="outlined", margin=0, sizing_mode="stretch_width",
                styles={"flex": "1 1 280px", "min-width": "0", "max-width": "100%"},
                sx={"p": "16px 12px 16px 16px", "borderRadius": "8px"},
                css_classes=["lumen-data-source", f"lumen-data-source-{summary.ref}"],
            ))
        return pn.FlexBox(
            *cards, gap="12px", flex_wrap="wrap", margin=(0, 0, 16, 0), sizing_mode="stretch_width",
            css_classes=["lumen-data-sources"],
        )

    def remove_source(self, ref: str) -> None:
        """Remove the source an admin connected, or ask before deleting an upload, from its card's menu."""
        if self.uploads is None:
            return
        if self.uploads.get(ref) is not None:
            upload = self.uploads.get(ref)
            self.catalog.selected = (ref, upload.tables[0] if upload.tables else "")
            self.catalog.ask_delete()
            return
        problems = self.uploads.disconnect(ref, app=self.app)
        if problems:
            self.catalog._show_problems(problems)
            return
        if hasattr(self.app, "_notify"):
            self.app._notify("success", f"Removed the source {ref}.")
        self.navigate(f"/{ROUTE}")

    def _on_table(self, selected: tuple[str, str] | None):
        # The URL follows the drawer, so a reload or a shared link opens it again.
        location = pn.state.location
        if location is None:
            return
        search = f"?table={quote(f'{selected[0]}.{selected[1]}', safe='')}" if selected else ""
        if (location.search or "") != search:
            location.search = search

    # ------------------------------------------------------------------
    # Metrics

    def _metrics_controller(self, **params) -> t.Any:
        from .metrics_admin import MetricsSettings

        return MetricsSettings(
            self.project, editable=self.admin, app=self.app, show_new=False,
            on_open=lambda name: self.navigate(f"/{ROUTE}/metrics/{quote(name, safe='')}"), **params,
        )

    def _metrics_page(self) -> pn.Column:
        self.metrics = self._metrics_controller(on_change=lambda _name: self.navigate(f"/{ROUTE}/metrics"))
        new = pmui.Button(label="New metric", icon="add", variant="contained", margin=0, visible=self.admin,
                          css_classes=["lumen-data-new-metric"])
        new.on_click(lambda _event: self.metrics.open_metric(None))
        heading = _heading(
            "Metrics",
            "The agreed definitions of the app's key numbers. Dashboards and the assistant use them by name instead "
            "of writing their own SQL.",
            new,
        )
        return self._content(heading, self.metrics)

    def _metric_page(self, name: str) -> t.Any:
        from .metric_page import MetricPage

        status = next((s for s in self.statuses if s.name == name), None)
        if status is None:
            back = pmui.Button(label="All metrics", variant="outlined", margin=0)
            back.on_click(lambda _event: self.navigate(f"/{ROUTE}/metrics"))
            return self._content(_heading(name, "The project defines no metric of this name, or no longer does.", back))

        def changed(saved: str | None):
            self.navigate(f"/{ROUTE}/metrics/{quote(saved, safe='')}" if saved else f"/{ROUTE}/metrics")

        controller = self._metrics_controller(on_change=changed) if self.admin else None
        self.metric = MetricPage(
            self.app, self.project, status, admin=self.admin, controller=controller, navigate=self.navigate,
        )
        return self.metric

    def __panel__(self):
        return self._layout


def data_view(ctx: t.Any, path: str) -> DataPage:
    """The page of ``/data/<path>`` for the session of ``ctx``, a ``panel_flowdash.SidebarContext``."""
    location = pn.state.location
    search = (location.search or "") if location is not None else ""
    return DataPage(
        ctx.app, ctx.app._settings_project(), path, search, admin=ctx.is_admin, navigate=ctx.navigate,
    )


def redirect_moved_admin_pages(app: t.Any) -> None:
    """
    Have ``app``, a session's ``FlowDashApp``, open the Data page of an
    admin view section that moved there, e.g. ``/admin/catalog`` from a
    bookmark, instead of the admin view's first section.
    """
    show_admin = getattr(app, "_admin_view", None)
    if show_admin is None:
        return

    def admin_view(pathname: str) -> t.Any:
        target = MOVED.get(pathname.strip("/").split("/", 1)[-1]) if pathname.strip("/").count("/") == 1 else None
        if target is None:
            return show_admin(pathname)
        doc = pn.state.curdoc
        if doc is not None and doc.session_context is not None:
            doc.add_next_tick_callback(lambda: app._navigate_url(target))
        return caption(f"This moved to Data. Opening {target}…", margin=(24, 32))

    app._admin_view = admin_view
