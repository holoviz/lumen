"""Browser UI tests for dbt Semantic Layer inherited metrics on the Metrics pages."""

from __future__ import annotations

import getpass

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel.tests.util import serve_component
from panel_flowdash import MemoryDashboardStore
from panel_flowdash.auth import Identity
from playwright.sync_api import expect

from lumen.flowdash import METRICS_KEY, SOURCES_KEY
from lumen.flowdash.app import build_routes
from lumen.flowdash.data_section import DataPage
from lumen.flowdash.dbt_metrics import DBT_CLIENT_KEY, clear_dbt_cache
from lumen.flowdash.metrics import ProjectMetrics
from lumen.flowdash.project import PROJECT_FILE, load_project
from lumen.flowdash.uploads import UPLOADS_KEY, Uploads
from lumen.sources.duckdb import DuckDBSource

from .dbt_fakes import (
    FakeSemanticLayer, dimension, measure, metric, semantic_sql,
)

pytestmark = pytest.mark.filterwarnings("ignore:reference already known")


def _FakeClient():
    return FakeSemanticLayer(
        [
            metric("dbt_revenue", measures=[measure("amount", "SUM", "amount")], dimensions=[dimension("order__region")],
                   description="Revenue from dbt semantic layer"),
            metric("dbt_cumulative", "CUMULATIVE", description="Cumulative revenue via compile_sql"),
        ],
        {
            "dbt_revenue": semantic_sql("orders", "SUM(amount) AS dbt_revenue", {"order__region": "region"}),
            "dbt_cumulative": "SELECT region, SUM(amount) AS dbt_cumulative FROM orders GROUP BY region",
        },
    )


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_dbt_cache()
    yield
    clear_dbt_cache()


class _AdminSession:
    def __init__(self, project):
        self.project = project
        self.is_admin = True
        self.store = MemoryDashboardStore()
        self._identity = Identity("ana")

    def _still_admin(self):
        return True

    def _notify(self, severity, message, duration=3000):
        pass


PROJECT = """\
general:
  admins: [{user}]
metrics:
  dbt:
    source: orders
    environment_id: 12345
    tables:
      dbt_revenue: orders
  local_orders:
    description: Hand-authored order count
    source: orders
    table: orders
    measures:
      order_count: COUNT(*)
    dimensions:
      region: region
"""


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


def test_dbt_metrics_appear_marked_and_readonly_in_app(page, tmp_path, monkeypatch):
    from lumen.flowdash import dbt_metrics

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dbt_metrics, "_make_client", lambda _config: _FakeClient())

    (tmp_path / "orders.csv").write_text("region,amount\nNorth,120\nNorth,80\nSouth,50\n")
    (tmp_path / PROJECT_FILE).write_text(PROJECT.format(user=getpass.getuser()))

    routes = build_routes(["orders.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE)
    page.set_viewport_size({"width": 1440, "height": 900})
    suffix = "/data/metrics" if "/data" in routes else "/admin/metrics"
    serve_component(page, routes, suffix=suffix)

    local_row = page.locator(".flowdash-admin-row-local-orders")
    expect(local_row).to_contain_text("Valid", timeout=20000)
    expect(local_row.locator(".flowdash-admin-metric-menu")).to_be_visible()

    dbt_row = page.locator(".flowdash-admin-row-dbt-revenue")
    expect(dbt_row).to_contain_text("dbt", timeout=15000)
    expect(dbt_row).to_contain_text("Valid")
    expect(dbt_row.locator(".flowdash-admin-metric-menu")).to_have_count(0)

    cum_row = page.locator(".flowdash-admin-row-dbt-cumulative")
    expect(cum_row).to_contain_text("dbt")
    expect(cum_row).to_contain_text("Valid")
    expect(cum_row.locator(".flowdash-admin-metric-menu")).to_have_count(0)

    # "From dbt" filter chip filters down to only dbt metrics.
    page.locator(".filter-origin-dbt").click()
    expect(dbt_row).to_be_visible()
    expect(local_row).to_have_count(0)

    # Opening a dbt metric shows its definition read-only, closed with Done rather than Cancel.
    dbt_row.locator(".flowdash-metric-dbt").click()
    expect(page.get_by_role("button", name="Done")).to_be_visible(timeout=10000)
    expect(page.locator(".flowdash-metric-save")).to_be_hidden()
    expect(page.locator(".flowdash-metric-preview-run")).to_be_hidden()


def test_dbt_metric_page_is_readonly_and_runs_try_it(page, tmp_path):
    orders = DuckDBSource(
        uri=":memory:",
        tables={"orders": "SELECT * FROM (VALUES ('North', 120), ('North', 80), ('South', 50)) AS t(region, amount)"},
        name="orders",
    )
    path = tmp_path / PROJECT_FILE
    path.write_text(PROJECT.format(user="ana"))
    sources = {"orders": orders}
    project = load_project(path, {SOURCES_KEY: sources, DBT_CLIENT_KEY: _FakeClient()})
    project.context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", sources, project=project)
    project.context[METRICS_KEY] = ProjectMetrics(project)

    view = DataPage(_AdminSession(project), project, "metrics/dbt_revenue", admin=True, navigate=lambda _url: None)
    page.set_viewport_size({"width": 1440, "height": 900})
    serve_component(page, view)

    expect(page.locator(".lumen-metric-head")).to_contain_text("dbt_revenue", timeout=20000)
    expect(page.locator(".lumen-metric-head")).to_contain_text("From dbt")
    expect(page.locator(".lumen-metric-head")).to_contain_text("Read-only")
    expect(page.locator(".lumen-metric-facts")).to_contain_text("dbt Semantic Layer (read-only)")
    expect(page.locator(".lumen-metric-edit")).to_have_count(0)
    expect(page.locator(".lumen-metric-menu")).to_have_count(0)

    page.locator(".lumen-try-run button").click()
    result = page.locator(".lumen-try-result")
    expect(result).to_contain_text("North", timeout=15000)
    expect(result).to_contain_text("200")
    expect(result).to_contain_text("South")
