"""
The Data section driven through the browser: the tables, a table's
drawer, the metrics and a metric's page. Runs where pytest-playwright is
installed and panel-flowdash has sidebar sections.
"""
import getpass
import re

from urllib.parse import urlsplit

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")
pytest.importorskip("panel_flowdash.sidebar")

from panel.tests.util import serve_component
from playwright.sync_api import expect

from lumen.flowdash.app import build_routes
from lumen.flowdash.project import PROJECT_FILE

ROWS = [
    ("Alta", "CA", 2010, 6.0, 3000),
    ("Alta", "CA", 2010, 6.0, 3000),
    ("Roscoe", "TX", 2012, 4.5, 1500),
]

PROJECT = """\
general:
  admins: [{user}]
metrics:
  fleet:
    description: Turbines and rated capacity
    source: local
    table: turbines
    measures:
      turbines: COUNT(*)
    dimensions:
      state: t_state
  output:
    description: Output by hour
    source: local
    table: hourly_output
    measures:
      kw: SUM(kw)
"""

# Panel warns of a model it already sent, which the test's warning filters would turn into an error.
pytestmark = pytest.mark.filterwarnings("ignore:reference already known")


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


def _path(page):
    return urlsplit(page.url).path


def test_the_data_section_shows_the_tables_and_the_metrics(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text(
        "p_name,t_state,p_year,p_cap,t_cap\n" + "".join(f"{n},{s},{y},{p},{c}\n" for n, s, y, p, c in ROWS)
    )
    (tmp_path / "readings.csv").write_text("p_name,power_kw\nAlta,120.5\nRoscoe,80.25\n")
    (tmp_path / PROJECT_FILE).write_text(PROJECT.format(user=getpass.getuser()))
    routes = build_routes(
        ["turbines.csv", "readings.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE
    )
    page.set_viewport_size({"width": 1440, "height": 900})
    serve_component(page, routes, suffix="/admin/catalog")
    base = "{0.scheme}://{0.netloc}".format(urlsplit(page.url))

    # The admin view's Catalog moved: its old route opens the Data section's tables.
    expect(page.locator(".lumen-data-title")).to_have_text("Tables", timeout=20000)
    assert _path(page) == "/data"
    expect(page.locator(".lumen-data-source-local")).to_contain_text("2 tables")
    turbines = page.locator(".flowdash-admin-row-local--turbines")
    expect(turbines).to_contain_text("5 columns", timeout=15000)

    turbines.get_by_text("turbines.csv").click()
    drawer = page.locator(".flowdash-admin-catalog-drawer")
    expect(drawer).to_contain_text("t_state", timeout=15000)
    expect(drawer.locator(".flowdash-catalog-used-by")).to_contain_text("fleet")
    expect(page).to_have_url(re.compile(r"/data\?table=local\.turbines$"))
    drawer.get_by_role("button", name="Done").click()
    expect(drawer).to_be_hidden()

    page.locator(".lumen-data-nav").get_by_text("Metrics").click()
    expect(page.locator(".lumen-data-title")).to_have_text("Metrics", timeout=20000)
    expect(page.locator(".flowdash-metrics-banner")).to_contain_text("output can't run")
    expect(page.locator(".flowdash-admin-row-output")).to_contain_text("Table missing")
    expect(page.locator(".flowdash-admin-row-fleet")).to_contain_text("Valid")

    page.locator(".flowdash-admin-row-fleet").get_by_text("Turbines and rated capacity").click()
    expect(page.locator(".lumen-metric-head")).to_contain_text("fleet", timeout=20000)
    assert _path(page) == "/data/metrics/fleet"
    page.locator(".lumen-try-run button").click()
    result = page.locator(".lumen-try-result")
    expect(result).to_contain_text("CA", timeout=15000)
    expect(result).to_contain_text("TX")

    page.goto(f"{base}/data/metrics/output")
    expect(page.locator(".lumen-metric-problem")).to_contain_text("has no table 'hourly_output'", timeout=20000)
    expect(page.locator(".lumen-metric-blocked")).to_contain_text("the table it reads is missing")

    page.goto(f"{base}/admin/uploads")
    expect(page.locator(".flowdash-admin-content")).to_contain_text("Let people upload data", timeout=20000)
    expect(page.locator(".flowdash-admin-nav-data")).not_to_contain_text("Catalog")
    expect(page.locator(".flowdash-admin-nav-data")).not_to_contain_text("Metrics")


RESTRICTED = """\
general:
  admins: [someone_else]
catalog:
  local:
    tables:
      readings:
        access:
          allow_users: [someone_else]
metrics:
  fleet:
    description: Turbines and rated capacity
    source: local
    table: turbines
    measures:
      turbines: COUNT(*)
  payroll:
    description: Restricted readings
    source: local
    table: readings
    measures:
      total: SUM(power_kw)
"""


def test_the_data_section_hides_what_the_session_may_not_access(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text(
        "p_name,t_state,p_year,p_cap,t_cap\n" + "".join(f"{n},{s},{y},{p},{c}\n" for n, s, y, p, c in ROWS)
    )
    (tmp_path / "readings.csv").write_text("p_name,power_kw\nAlta,120.5\nRoscoe,80.25\n")
    (tmp_path / PROJECT_FILE).write_text(RESTRICTED)
    routes = build_routes(
        ["turbines.csv", "readings.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE
    )
    page.set_viewport_size({"width": 1440, "height": 900})
    serve_component(page, routes, suffix="/data")
    base = "{0.scheme}://{0.netloc}".format(urlsplit(page.url))

    expect(page.locator(".lumen-data-title")).to_have_text("Tables", timeout=20000)
    expect(page.locator(".flowdash-admin-row-local--turbines")).to_be_visible(timeout=15000)
    expect(page.locator(".lumen-data-source-local")).to_contain_text("1 table")
    expect(page.locator(".flowdash-admin-row-local--readings")).to_have_count(0)

    page.goto(f"{base}/data?table=local.readings")
    expect(page.locator(".lumen-data-title")).to_have_text("Tables", timeout=20000)
    expect(page.locator(".flowdash-admin-catalog-drawer")).to_be_hidden()
    expect(page.locator("body")).not_to_contain_text("power_kw")

    page.goto(f"{base}/data/metrics")
    expect(page.locator(".lumen-data-title")).to_have_text("Metrics", timeout=20000)
    expect(page.locator(".flowdash-admin-row-fleet")).to_be_visible()
    expect(page.locator(".flowdash-admin-row-payroll")).to_have_count(0)

    page.goto(f"{base}/data/metrics/payroll")
    expect(page.locator(".lumen-data-title, .lumen-metric-head").first).to_be_visible(timeout=20000)
    expect(page.locator(".lumen-metric-head")).to_have_count(0)
    expect(page.locator("body")).not_to_contain_text("Restricted readings")
