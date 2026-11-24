"""
Turning uploads on in the admin view and uploading a file from a
dashboard, driven through the browser. Runs where pytest-playwright is
installed.
"""
import getpass
import json

from urllib.parse import urlsplit

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel.tests.util import serve_component
from panel_flowdash.dashboard_store import DashboardStore
from playwright.sync_api import expect

from lumen.flowdash.app import build_routes
from lumen.flowdash.project import PROJECT_FILE, load_project

# Panel warns of a model it already sent, which the test's warning filters would turn into an error.
pytestmark = pytest.mark.filterwarnings("ignore:reference already known")


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


def test_an_admin_turns_uploads_on_and_a_file_becomes_a_table(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,p_cap\nAlta,6.0\n")
    (tmp_path / PROJECT_FILE).write_text(f"general:\n  admins: [{getpass.getuser()}]\n")
    (tmp_path / "regional sales.csv").write_text("region,revenue\nNorth,120.5\nSouth,98\n")
    store = DashboardStore(str(tmp_path / "dash.db"))
    dashboard = store.create_dashboard(getpass.getuser(), "Sales")
    routes = build_routes(["turbines.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE)
    page.set_viewport_size({"width": 1400, "height": 900})
    serve_component(page, routes, suffix="/admin/uploads")
    base = "{0.scheme}://{0.netloc}".format(urlsplit(page.url))

    expect(page.locator(".flowdash-admin-content")).to_contain_text("Let people upload data")
    page.locator(".flowdash-setting-enabled input").first.click()
    page.get_by_role("button", name="Save").click()
    expect(page.get_by_text("All changes saved")).to_be_visible()
    assert load_project(tmp_path / PROJECT_FILE).raw("uploads") == {"enabled": True}

    page.goto(f"{base}/dash/{dashboard.dashboard_id}")
    add = page.locator(".flowdash-add-data button").first
    expect(add).to_be_visible(timeout=20000)
    add.click()
    page.locator(".flowdash-upload-file input[type=file]").set_input_files(str(tmp_path / "regional sales.csv"))
    expect(page.locator(".flowdash-upload-name input")).to_have_value("regional_sales", timeout=15000)
    page.locator(".flowdash-upload-submit button").click()
    expect(page.get_by_text("Added the table regional_sales as source regional_sales")).to_be_visible(timeout=20000)
    manifest = json.loads((tmp_path / "uploads" / "manifest.json").read_text())
    assert [u["ref"] for u in manifest["uploads"]] == ["regional_sales"]

    page.goto(f"{base}/dash/{dashboard.dashboard_id}?edit=true")
    expect(page.get_by_text("regional_sales (source)")).to_be_visible(timeout=20000)

    page.goto(f"{base}/admin/catalog")
    row = page.locator(".flowdash-admin-row-regional-sales--regional-sales")
    expect(row).to_contain_text(f"Uploaded by {getpass.getuser()}", timeout=20000)
    expect(row).to_contain_text("2 columns")
