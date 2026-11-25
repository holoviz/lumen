"""The Data section of the sidebar and where the catalog and the metrics move from the admin view, without a browser."""
import getpass
import sys

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("panel_flowdash.admin.wizard")

from bokeh.document import Document
from panel.io.state import set_curdoc
from panel_flowdash import MemoryDashboardStore
from panel_flowdash.admin.page import admin_sections
from panel_flowdash.auth import Identity
from panel_flowdash.dashboard_store import DashboardItem

from lumen.flowdash import METRICS_KEY, SOURCES_KEY
from lumen.flowdash.app import build_routes
from lumen.flowdash.catalog import raw_section
from lumen.flowdash.contracts import METRIC_ID
from lumen.flowdash.data_section import (
    SECTION_ID, DataPage, data_section, data_view, redirect_moved_admin_pages,
)
from lumen.flowdash.metrics import ProjectMetrics
from lumen.flowdash.project import PROJECT_FILE, load_project
from lumen.flowdash.uploads import UPLOADS_KEY, Uploads
from lumen.sources.duckdb import DuckDBSource

TABLES = {"orders": "SELECT * FROM (VALUES (1, 'paid', 30), (2, 'open', 20), (3, 'paid', 5)) AS t(order_id, status, amount)"}

YAML = """\
metrics:
  paid:
    description: Value of the orders
    source: shop
    table: orders
    measures:
      amount: SUM(amount)
    dimensions:
      status: status
    parameters:
      since: {type: integer, column: order_id, op: ">="}
  hourly:
    description: Output by hour
    source: shop
    table: hourly_output
    measures:
      output: SUM(kw)
"""

needs_sidebar = pytest.mark.skipif(data_section() is None, reason="panel-flowdash has no sidebar sections")


class Session:
    """The parts of a FlowDash app's session the Data section's pages use."""

    def __init__(self, project, admin=False, user="bo", store=None):
        self.project = project
        self.is_admin = self.admin = admin
        self.store = store or MemoryDashboardStore()
        self._identity = Identity(user)
        self.notified = []

    def _still_admin(self):
        return self.admin

    def _settings_project(self):
        return self.project

    def _default_allow(self):
        return False

    def _notify(self, severity, message, duration=3000):
        self.notified.append((severity, message))


class Context:
    def __init__(self, app):
        self.app = app
        self.is_admin = app.is_admin
        self.opened = []

    def navigate(self, url):
        self.opened.append(url)


@pytest.fixture
def project(tmp_path):
    source = DuckDBSource(uri=":memory:", tables=TABLES, name="shop")
    sources = {"shop": source}
    path = tmp_path / PROJECT_FILE
    path.write_text(YAML)
    project = load_project(path, {SOURCES_KEY: sources})
    project.context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", sources, project=project)
    project.context[METRICS_KEY] = ProjectMetrics(project)
    return project


def page(project, path="", search="", admin=False, **params):
    opened = []
    session = Session(project, admin=admin, **params)
    view = DataPage(session, project, path, search, admin=admin, navigate=opened.append)
    view.opened = opened
    return view


def _text(view):
    return "".join(str(o.object) for o in view.__panel__().select() if isinstance(getattr(o, "object", None), str))


def _admin_names():
    return [section.name for section in admin_sections()]


def _serve(tmp_path, monkeypatch, **params):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,p_cap\nAlta,6.0\n")
    (tmp_path / PROJECT_FILE).write_text(f"general:\n  admins: [{getpass.getuser()}]\n")
    return build_routes(["turbines.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE, **params)


# ---------------------------------------------------------------------------
# The section and the admin view


@needs_sidebar
def test_the_data_section_takes_the_catalog_and_metrics_out_of_the_admin_view(tmp_path, monkeypatch):
    routes = _serve(tmp_path, monkeypatch)
    assert "/data" in routes and any(path.startswith("/data/") for path in routes)
    with set_curdoc(Document()):
        app = routes["/"]()
        assert [s.id for s in app.sections] == ["dashboards", SECTION_ID]
        names = _admin_names()
        assert "catalog" not in names and "metrics" not in names
        assert "uploads" in names


def test_without_sidebar_sections_the_admin_view_keeps_them(tmp_path, monkeypatch):
    # A panel-flowdash without the sidebar module, as before sidebar sections.
    monkeypatch.setitem(sys.modules, "panel_flowdash.sidebar", None)
    assert data_section() is None
    routes = _serve(tmp_path, monkeypatch)
    assert not any(path.startswith("/data") for path in routes)
    with set_curdoc(Document()):
        app = routes["/"]()
        assert SECTION_ID not in [s.id for s in getattr(app, "sections", [])]
        assert {"catalog", "metrics", "uploads"} <= set(_admin_names())


@needs_sidebar
@pytest.mark.parametrize("params", [{"nav_variant": "left"}, {"sections": "own"}])
def test_the_admin_view_keeps_them_where_the_data_section_is_not_shown(tmp_path, monkeypatch, params):
    from panel_flowdash.sidebar import SidebarSection

    if params.get("sections") == "own":
        own = SidebarSection(id=SECTION_ID, label="Our data", icon="storage", route="ours", view=lambda ctx, path: "")
        params = {"sections": lambda: [own]}
    routes = _serve(tmp_path, monkeypatch, **params)
    with set_curdoc(Document()):
        app = routes["/"]()
        assert all(s.view is not data_view for s in app.sections)
        assert {"catalog", "metrics"} <= set(_admin_names())


def test_the_admin_views_old_routes_lead_to_data():
    shown = []

    class App:
        def _admin_view(self, pathname):
            shown.append(pathname)
            return "admin"

    app = App()
    redirect_moved_admin_pages(app)
    assert app._admin_view("/admin/general") == "admin"
    assert "moved to Data" in app._admin_view("/admin/catalog").object
    assert "/data/metrics" in app._admin_view("/admin/metrics").object
    assert shown == ["/admin/general"]


@needs_sidebar
def test_the_section_opens_the_page_its_path_names(project):
    context = Context(Session(project, admin=True))
    assert data_view(context, "").page == "tables"
    assert data_view(context, "metrics").page == "metrics"
    assert data_view(context, "metrics/paid").page == "metric"
    assert data_view(context, "nothing/here").page == "tables"


# ---------------------------------------------------------------------------
# Tables


def _switches(view):
    return [c for row in view._table._rows for c in row.select() if "flowdash-catalog-assistant" in getattr(c, "css_classes", [])]


def test_the_nav_counts_the_data_and_filters_by_source(project):
    view = page(project, admin=True)
    text = _text(view)
    assert "1 source · 1 table · 2 metrics" in text and "Admin, Uploads" in text
    nav = [o for o in view.__panel__().select() if "lumen-data-nav-sources" in getattr(o, "css_classes", [])][0]
    nav._process_click({}, (0,), nav.items[0])
    assert view.opened == ["/data?source=shop"]
    assert page(project, search="?source=shop", admin=True).catalog.filter == "source:shop"


def test_an_admin_turns_tables_off_and_adds_data(project):
    view = page(project, admin=True)
    catalog = view.catalog
    assert [r["table"] for r in catalog.rows()] == ["orders"]
    assert _switches(catalog) and not any(s.disabled for s in _switches(catalog))
    assert catalog.set_assistant("shop", "orders", False)
    assert raw_section(project) == {"shop": {"tables": {"orders": {"assistant": False}}}}
    assert "From the command line" not in _text(view) and "Served by the app" in _text(view)
    assert catalog.may_add()
    catalog.add_data("connect")
    assert catalog._upload_dialog._mode.visible and catalog._upload_dialog.connecting


def test_a_viewer_sees_status_and_no_add_data_where_uploads_are_off(project):
    view = page(project)
    catalog = view.catalog
    assert not _switches(catalog)
    assert not catalog.set_assistant("shop", "orders", False) and raw_section(project) == {}
    assert not catalog.may_add()
    assert "Adding files is off in this app." in _text(view)


def test_a_viewer_allowed_to_upload_adds_a_file_without_the_connect_tab(project):
    project.set("uploads", {"enabled": True})
    view = page(project)
    catalog = view.catalog
    assert catalog.may_add() and "You can add files" in _text(view)
    dialog = catalog._upload_dialog
    catalog.add_data("connect")
    assert not dialog._mode.visible and not dialog.connecting
    dialog.choose(b"region,revenue\nNorth,1\n", "sales.csv")
    assert dialog._preview.tables == ["sales"] and "Added as the source sales" in dialog._note.object
    dialog._assistant.value = False
    assert dialog.submit() is not None
    assert view.opened == ["/data"]
    assert raw_section(project)["sales"] == {"tables": {"sales": {"assistant": False}}}
    catalog.open_table("sales", "sales")
    assert not catalog._delete.visible and not catalog.delete_upload()
    assert "sales" in project.context[SOURCES_KEY]


def test_the_table_drawer_opens_from_the_url_and_names_what_uses_the_table(project):
    store = MemoryDashboardStore()
    model = store.create_dashboard("bo", "Orders")
    model.items = [DashboardItem("t", "Lumen/table", config={"ref": "shop", "table": "orders"})]
    store.save_dashboard(model)
    view = page(project, search="?table=shop.orders", store=store)
    catalog = view.catalog
    assert catalog._drawer.open and catalog.selected == ("shop", "orders")
    assert [b.label for b in catalog._used_links] == ["Orders", "paid"]
    catalog._used_links[1].clicks += 1
    assert view.opened[-1] == "/data/metrics/paid"


def test_an_empty_app_says_how_data_gets_there(tmp_path):
    path = tmp_path / PROJECT_FILE
    path.write_text("")
    empty = load_project(path, {SOURCES_KEY: {}})
    empty.context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", empty.context[SOURCES_KEY], project=empty)
    admin = page(empty, admin=True)
    assert "No data yet" in _text(admin) and "lumen serve turbines.csv" in _text(admin)
    assert any("lumen-data-empty-add" in getattr(o, "css_classes", []) for o in admin.__panel__().select())
    viewer = page(empty)
    assert "An admin can add files or connect a database." in _text(viewer)
    assert not any("lumen-data-empty-add" in getattr(o, "css_classes", []) for o in viewer.__panel__().select())


# ---------------------------------------------------------------------------
# Metrics


def test_the_metrics_list_their_checks_and_an_admin_gets_the_banner(project):
    view = page(project, "metrics", admin=True)
    metrics = view.metrics
    assert [s.check for s in metrics.statuses()] == ["valid", "table"]
    assert metrics._banner.visible and "hourly can" in metrics._banner[0][0].object
    assert "hourly_output" in metrics._banner[0][0].object
    metrics.open("paid")
    assert view.opened == ["/data/metrics/paid"]
    assert set(metrics._menus) == {"paid", "hourly"}


def test_a_viewer_gets_the_metrics_without_editing_or_the_banner(project):
    metrics = page(project, "metrics").metrics
    assert not metrics._banner.visible and not metrics._menus and not metrics._new.visible


def test_the_metric_page_shows_the_definition_and_runs_try_it(project):
    view = page(project, "metrics/paid")
    metric = view.metric
    text = _text(view)
    assert "Keeps rows with order_id &gt;= since." in text and "SUM(amount)" in text
    assert metric.controller is None
    metric._inputs["since"][0].value = "2"
    metric._filter.value = "paid"
    result = metric.run()
    assert result.error is None and result.rows == [["paid", 5]]
    assert "LIMIT" in result.sql and metric._sql_card.visible


def test_a_failing_metric_page_says_why_and_try_it_waits(project):
    store = MemoryDashboardStore()
    model = store.create_dashboard("bo", "Power")
    model.items = [DashboardItem("m", METRIC_ID, config={"metric": "hourly"})]
    store.save_dashboard(model)
    view = page(project, "metrics/hourly", admin=True, store=store)
    text = _text(view)
    assert "This metric can&#x27;t run" in text and "has no table &#x27;hourly_output&#x27;" in text
    assert "Power shows this error where it uses the metric." in text
    assert "the table it reads is missing" in text
    assert view.metric.controller is not None


def test_an_admin_edits_a_metric_from_its_page(project):
    view = page(project, "metrics/paid", admin=True)
    controller = view.metric.controller
    controller.open_metric("paid")
    assert controller._drawer.open
    controller.run("delete", "paid")
    assert controller.confirm()
    assert view.opened == ["/data/metrics"] and "paid" not in load_project(project.path).raw("metrics")


def test_the_metrics_count_only_the_dashboards_the_session_may_open(project):
    store = MemoryDashboardStore()
    for owner in ("ana", "bo"):
        model = store.create_dashboard(owner, f"{owner}'s")
        model.items = [DashboardItem("m", METRIC_ID, config={"metric": "paid"})]
        store.save_dashboard(model)
    assert [r.title for r in page(project, "metrics", store=store).metrics.usage()["paid"]] == ["bo's"]
    admin = page(project, "metrics", admin=True, store=store)
    assert sorted(r.title for r in admin.metrics.usage()["paid"]) == ["ana's", "bo's"]


def test_a_source_listed_without_its_credentials_offers_to_enter_them_again(project, monkeypatch):
    from panel_flowdash.secret_store import (
        KEY_ENV, encryption_available, fernet_key,
    )

    if not encryption_available():
        pytest.skip("needs cryptography")
    monkeypatch.setenv(KEY_ENV, fernet_key())
    monkeypatch.setattr(Uploads, "_open", lambda self, location, ref: DuckDBSource(
        uri=":memory:", tables={"readings": "SELECT 1 AS kw"}, name=ref))
    uploads = project.context[UPLOADS_KEY]
    uploads.connect("postgresql://ana:hunter2@db.example.com/plant", "plant", app=Session(project, admin=True))
    monkeypatch.setenv(KEY_ENV, fernet_key())
    sources = project.context[SOURCES_KEY]
    sources.pop("plant")
    restored = project.context[UPLOADS_KEY] = Uploads(uploads.directory, sources, project=project)
    restored.restore()

    view = page(project, admin=True)
    text = _text(view)
    assert "Credentials can&#x27;t be read; enter them again" in text and "hunter2" not in text
    menus = [o for o in view.__panel__().select() if "lumen-data-source-menu" in getattr(o, "css_classes", [])]
    assert [item["label"] for item in menus[0].items] == ["Enter credentials again", "Remove source"]
    view.reconnect(next(s for s in view.sources if s.ref == "plant"))
    dialog = view.catalog._upload_dialog
    assert dialog.connecting and dialog._ref.value == "plant"
    assert dialog._location.value == "postgresql://ana:***@db.example.com/plant"


def test_a_file_outside_the_project_is_shown_by_its_name(project, tmp_path):
    from lumen.flowdash.nodes import set_served_files

    set_served_files(project.context[SOURCES_KEY]["shop"], {"orders": "/srv/data/orders.parquet"})
    admin, viewer = page(project, admin=True).catalog, page(project).catalog
    assert 'title="/srv/data/orders.parquet">orders.parquet<' in "".join(
        str(o.object) for o in admin._table._rows[0].select() if isinstance(getattr(o, "object", None), str))
    viewer_text = "".join(str(o.object) for o in viewer._table._rows[0].select() if isinstance(getattr(o, "object", None), str))
    assert ">orders.parquet<" in viewer_text and "/srv/data" not in viewer_text
