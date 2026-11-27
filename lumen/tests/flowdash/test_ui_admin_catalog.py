"""
The admin view's Catalog section driven through the browser. Runs where
pytest-playwright is installed.
"""
import getpass

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel.tests.util import serve_component
from playwright.sync_api import expect

from lumen.flowdash.app import build_routes
from lumen.flowdash.project import PROJECT_FILE, load_project

ROWS = [
    ("Alta", "CA", 2010, 6.0, 3000),
    ("Alta", "CA", 2010, 6.0, 3000),
    ("Roscoe", "TX", 2012, 4.5, 1500),
]

# Panel warns of a model it already sent, which the test's warning filters would turn into an error.
pytestmark = pytest.mark.filterwarnings("ignore:reference already known")


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


def _serve(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text(
        "p_name,t_state,p_year,p_cap,t_cap\n" + "".join(f"{n},{s},{y},{p},{c}\n" for n, s, y, p, c in ROWS)
    )
    (tmp_path / "readings.csv").write_text("p_name,power_kw\nAlta,120.5\nRoscoe,80.25\n")
    (tmp_path / PROJECT_FILE).write_text(f"general:\n  admins: [{getpass.getuser()}]\n")
    routes = build_routes(
        ["turbines.csv", "readings.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE
    )
    page.set_viewport_size({"width": 1400, "height": 900})
    serve_component(page, routes, suffix="/admin/catalog")


def test_an_admin_browses_the_catalog_and_turns_a_table_off(page, tmp_path, monkeypatch):
    _serve(page, tmp_path, monkeypatch)
    expect(page.locator(".flowdash-admin-content")).to_contain_text("keep the assistant from using it")
    turbines = page.locator(".flowdash-admin-row-local--turbines")
    expect(turbines).to_contain_text("5 columns", timeout=15000)
    expect(turbines).to_contain_text("p_name, t_state, p_year")
    expect(page.locator(".flowdash-admin-row-local--readings")).to_contain_text("2 columns")

    page.locator(".flowdash-admin-row-local--readings .flowdash-catalog-assistant input").click()
    expect(page.locator(".filter-off")).to_contain_text("Off 1")
    assert load_project(tmp_path / PROJECT_FILE).raw("catalog") == {
        "local": {"tables": {"readings": {"assistant": False}}}
    }

    page.locator(".flowdash-admin-row-local--readings").get_by_text("readings.csv").click()
    drawer = page.locator(".flowdash-admin-catalog-drawer")
    expect(drawer).to_contain_text("Turned off", timeout=15000)
    expect(drawer).to_contain_text("power_kw")
    expect(drawer).to_contain_text("80.25 to 120.5")
    expect(drawer.locator(".flowdash-catalog-detail")).to_contain_text("Roscoe")
    drawer.get_by_role("button", name="Done").click()
    expect(drawer).to_be_hidden()

    page.locator(".flowdash-admin-catalog-section input[placeholder='Search tables and columns']").fill("t_cap")
    expect(turbines).to_be_visible()
    expect(page.locator(".flowdash-admin-row-local--readings")).to_have_count(0)
