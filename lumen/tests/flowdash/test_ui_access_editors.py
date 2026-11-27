"""
Playwright browser tests for admin-only access rule editors on source cards,
in the table drawer and on the metric page, and confirming viewers see nothing new.
"""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel.tests.util import serve_component
from panel_flowdash import MemoryDashboardStore
from panel_flowdash.auth import Identity
from playwright.sync_api import expect

from lumen.flowdash import (
    CATALOG_KEY, METRICS_KEY, SOURCES_KEY, ProjectCatalog,
)
from lumen.flowdash.catalog import raw_section as raw_catalog
from lumen.flowdash.data_section import DataPage
from lumen.flowdash.metrics import ProjectMetrics
from lumen.flowdash.metrics_edit import raw_section as raw_metrics
from lumen.flowdash.project import PROJECT_FILE, load_project
from lumen.flowdash.uploads import UPLOADS_KEY, Uploads
from lumen.sources.duckdb import DuckDBSource

pytestmark = pytest.mark.filterwarnings("ignore:reference already known")

PROJECT_YAML = """\
general:
  admins: [ana]
permissions:
  people:
    alice:
      groups: [hr, finance]
    bob:
      groups: [finance]
metrics:
  fleet:
    description: Turbines and rated capacity
    source: shop
    table: turbines
    measures:
      turbines: COUNT(*)
  payroll:
    description: Readings total
    source: shop
    table: readings
    measures:
      total: SUM(power_kw)
"""


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


class _Session:
    def __init__(self, project, *, admin: bool = True, user: str = "ana", groups: tuple[str, ...] = ()):
        self.project = project
        self.is_admin = self.admin = admin
        self.store = MemoryDashboardStore()
        self._identity = Identity(user=user, groups=frozenset(groups))

    def still_admin(self) -> bool:
        return self.admin

    def _still_admin(self) -> bool:
        return self.admin

    def _authorize_admin(self, *, fresh: bool = False) -> bool:
        return self.admin

    def _settings_project(self):
        return self.project

    def _default_allow(self):
        return False

    def _known_users(self):
        return ["alice", "ana", "bob"]

    def _known_groups(self):
        return ["admins", "finance", "hr"]

    def _notify(self, severity, message, duration=3000):
        pass


def _make_project(tmp_path, yaml_text: str = PROJECT_YAML):
    source = DuckDBSource(
        uri=":memory:",
        tables={
            "turbines": "SELECT * FROM (VALUES ('Alta', 'CA', 6.0), ('Roscoe', 'TX', 4.5)) AS t(p_name, t_state, p_cap)",
            "readings": "SELECT * FROM (VALUES ('Alta', 120.5), ('Roscoe', 80.25)) AS t(p_name, power_kw)",
        },
        name="shop",
    )
    sources = {"shop": source}
    path = tmp_path / PROJECT_FILE
    path.write_text(yaml_text, encoding="utf-8")
    project = load_project(path, {SOURCES_KEY: sources})
    project.context[CATALOG_KEY] = ProjectCatalog(project)
    project.context[METRICS_KEY] = ProjectMetrics(project)
    project.context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", sources, project=project)
    return project


def test_admin_edits_source_table_and_metric_access_rules(page, tmp_path):
    project = _make_project(tmp_path)
    session = _Session(project, admin=True, user="ana", groups=("admins",))
    tables_view = DataPage(session, project, admin=True, navigate=lambda _url: None)

    page.set_viewport_size({"width": 1440, "height": 900})
    serve_component(page, tables_view)

    # 1. Source card: open access dialog, verify lockout refusal, then save a valid rule.
    expect(page.locator(".lumen-data-title")).to_have_text("Tables", timeout=20000)
    card = page.locator(".lumen-data-source-shop")
    expect(card).to_be_visible(timeout=15000)
    card.locator(".lumen-data-source-access button").click()

    dialog = page.locator(".lumen-access-dialog")
    expect(dialog).to_be_visible(timeout=15000)
    expect(dialog.locator(".lumen-access-effective")).to_contain_text("Open")

    deny_users = dialog.locator(".lumen-access-deny-users input").first
    deny_users.fill("ana")
    expect(dialog.locator(".lumen-access-lockout-warning")).to_contain_text("lock you out", timeout=10000)
    dialog.locator(".lumen-access-save button").click()
    expect(dialog.locator(".lumen-access-error")).to_contain_text("lock you out", timeout=10000)
    assert raw_catalog(project) == {}

    deny_users.fill("")
    dialog.locator(".lumen-access-allow-groups input").first.fill("hr, finance")
    dialog.locator(".lumen-access-save button").click()
    expect(dialog).to_be_hidden(timeout=15000)
    expect(card).to_contain_text("Access: Allow groups finance, hr", timeout=15000)
    assert raw_catalog(project)["shop"]["access"] == {"allow_groups": ["finance", "hr"]}

    # 2. Table drawer: shows inherited source rule, lets admin add table rule.
    readings_row = page.locator(".flowdash-admin-row-shop--readings")
    expect(readings_row).to_be_visible(timeout=15000)
    readings_row.get_by_text("readings").first.click()

    drawer = page.locator(".flowdash-admin-catalog-drawer")
    expect(drawer).to_be_visible(timeout=15000)
    expect(drawer.locator(".flowdash-catalog-access-summary")).to_contain_text("Inherited from Source 'shop'", timeout=15000)
    expect(drawer.locator(".flowdash-catalog-access-summary")).to_contain_text("Allow groups finance, hr")

    drawer.locator(".flowdash-catalog-access button").click()
    table_editor = drawer.locator(".flowdash-catalog-access-editor")
    expect(table_editor).to_be_visible(timeout=15000)
    table_editor.locator(".lumen-access-allow-groups input").first.fill("hr")
    table_editor.locator(".lumen-access-deny-users input").first.fill("contractor")
    table_editor.locator(".lumen-access-save button").click()
    expect(table_editor).to_be_hidden(timeout=15000)

    expect(drawer.locator(".flowdash-catalog-access-summary")).to_contain_text("Restricted (own + inherited)", timeout=15000)
    expect(drawer.locator(".flowdash-catalog-access-summary")).to_contain_text("Allow group hr; deny user contractor")
    assert raw_catalog(project)["shop"]["tables"]["readings"]["access"] == {
        "allow_groups": ["hr"],
        "deny_users": ["contractor"],
    }

    # 3. Metric page: shows inherited source + table rules, refuses lockout, saves metric rule.
    metric_view = DataPage(session, project, path="metrics/payroll", admin=True, navigate=lambda _url: None)
    serve_component(page, metric_view)
    expect(page.locator(".lumen-metric-head")).to_contain_text("payroll", timeout=20000)
    access_row = page.locator(".lumen-metric-access-row")
    expect(access_row).to_contain_text("Inherited from Source 'shop'", timeout=15000)
    expect(access_row).to_contain_text("Inherited from Table shop.readings")

    page.locator(".lumen-metric-access-edit button").click()
    metric_editor = page.locator(".lumen-metric-access-editor")
    expect(metric_editor).to_be_visible(timeout=15000)
    metric_editor.locator(".lumen-access-deny-groups input").first.fill("admins")
    metric_editor.locator(".lumen-access-save button").click()
    expect(metric_editor.locator(".lumen-access-error")).to_contain_text("lock you out", timeout=10000)
    assert "access" not in raw_metrics(project)["payroll"]

    metric_editor.locator(".lumen-access-deny-groups input").first.fill("")
    metric_editor.locator(".lumen-access-deny-users input").first.fill("intern")
    metric_editor.locator(".lumen-access-save button").click()
    expect(metric_editor).to_be_hidden(timeout=15000)

    expect(access_row).to_contain_text("Deny user intern", timeout=15000)
    assert raw_metrics(project)["payroll"]["access"] == {"deny_users": ["intern"]}


def test_viewer_sees_no_access_editors_in_data_section(page, tmp_path):
    project = _make_project(tmp_path)
    session = _Session(project, admin=False, user="alice", groups=("hr",))

    tables_view = DataPage(session, project, admin=False, navigate=lambda _url: None)
    page.set_viewport_size({"width": 1440, "height": 900})
    serve_component(page, tables_view)

    expect(page.locator(".lumen-data-title")).to_have_text("Tables", timeout=20000)
    expect(page.locator(".lumen-data-source-shop")).to_be_visible(timeout=15000)
    expect(page.locator(".lumen-data-source-access")).to_have_count(0)
    expect(page.locator(".lumen-access-dialog")).to_have_count(0)

    turbines_row = page.locator(".flowdash-admin-row-shop--turbines")
    expect(turbines_row).to_be_visible(timeout=15000)
    turbines_row.get_by_text("turbines").first.click()
    drawer = page.locator(".flowdash-admin-catalog-drawer")
    expect(drawer).to_be_visible(timeout=15000)
    expect(drawer.locator(".flowdash-catalog-access")).to_be_hidden()
    expect(drawer.locator(".flowdash-catalog-access-summary")).to_be_hidden()
    expect(drawer.locator(".flowdash-catalog-access-editor")).to_be_hidden()

    metric_view = DataPage(session, project, path="metrics/fleet", admin=False, navigate=lambda _url: None)
    serve_component(page, metric_view)
    expect(page.locator(".lumen-metric-head")).to_contain_text("fleet", timeout=20000)
    expect(page.locator(".lumen-metric-access-row")).to_have_count(0)
    expect(page.locator(".lumen-metric-access-edit")).to_have_count(0)
    expect(page.locator(".lumen-metric-access-editor")).to_have_count(0)
