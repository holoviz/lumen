"""
The admin view's Metrics section and the setup wizard's Key numbers step,
driven through the browser. Runs where pytest-playwright is installed.
"""
import getpass
import re

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel.tests.util import serve_component, wait_until
from panel_flowdash.dashboard_store import DashboardItem, DashboardStore
from playwright.sync_api import expect

from lumen.flowdash.app import build_routes
from lumen.flowdash.contracts import METRIC_ID
from lumen.flowdash.project import PROJECT_FILE, load_project

ROWS = [
    ("Alta", "CA", 2010, 6.0, 3000),
    ("Alta", "CA", 2010, 6.0, 3000),
    ("Roscoe", "TX", 2012, 4.5, 1500),
    ("Roscoe", "TX", 2012, 4.5, 1500),
    ("Roscoe", "TX", 2012, 4.5, 1500),
]

# Panel warns of a model it already sent, which the test's warning filters would turn into an error.
pytestmark = pytest.mark.filterwarnings("ignore:reference already known")

FLEET = """\
metrics:
  fleet:
    description: Turbines and their rated capacity.
    source: turbines
    table: turbines
    measures:
      turbines: COUNT(*)
    dimensions:
      state: t_state
"""


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


def _serve(page, tmp_path, monkeypatch, suffix="/admin/metrics", metrics=FLEET, dashboards=()):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text(
        "p_name,t_state,p_year,p_cap,t_cap\n" + "".join(f"{n},{s},{y},{p},{c}\n" for n, s, y, p, c in ROWS)
    )
    (tmp_path / PROJECT_FILE).write_text(f"general:\n  admins: [{getpass.getuser()}]\n{metrics}")
    store = DashboardStore(str(tmp_path / "dash.db"))
    for title, metric in dashboards:
        model = store.create_dashboard(getpass.getuser(), title)
        model.items = [DashboardItem("m", METRIC_ID, config={"metric": metric})]
        store.save_dashboard(model)
    routes = build_routes(["turbines.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE)
    page.set_viewport_size({"width": 1400, "height": 900})
    serve_component(page, routes, suffix=suffix)


def _metrics(tmp_path):
    return load_project(tmp_path / PROJECT_FILE).raw("metrics") or {}


def _drawer(page):
    return page.locator(".flowdash-admin-metric-drawer")


def _yaml(page):
    return _drawer(page).locator(".flowdash-metric-yaml textarea").first


def test_an_admin_defines_a_metric_previews_and_saves_it(page, tmp_path, monkeypatch):
    _serve(page, tmp_path, monkeypatch)
    expect(page.locator(".flowdash-admin-content")).to_contain_text("the assistant uses them instead")
    expect(page.locator(".flowdash-admin-row-fleet")).to_contain_text("Turbines and their rated capacity.")

    page.get_by_role("button", name="New metric").click()
    drawer = _drawer(page)
    expect(drawer).to_contain_text("New metric", timeout=15000)
    drawer.locator(".flowdash-metric-name input").fill("turbine_capacity")
    drawer.locator(".flowdash-metric-description input").fill("Rated capacity of the turbines in MW.")
    _yaml(page).fill("measures:\n  capacity_mw: SUM(t_cap) / 1000.0\ndimensions:\n  state: t_state\n")
    drawer.get_by_role("button", name="Run preview").click()
    preview = drawer.locator(".flowdash-metric-preview")
    expect(preview).to_contain_text("All rows")
    expect(preview).to_contain_text("10.50")
    expect(preview).to_contain_text("By state")
    expect(preview).to_contain_text('GROUP BY "t_state"')
    assert "turbine_capacity" not in _metrics(tmp_path)

    drawer.get_by_role("button", name="Save").click()
    expect(drawer).to_be_hidden()
    row = page.locator(".flowdash-admin-row-turbine-capacity")
    expect(row).to_contain_text("Rated capacity of the turbines in MW.")
    expect(row).to_contain_text("capacity_mw")
    assert _metrics(tmp_path)["turbine_capacity"] == {
        "description": "Rated capacity of the turbines in MW.", "source": "turbines", "table": "turbines",
        "measures": {"capacity_mw": "SUM(t_cap) / 1000.0"}, "dimensions": {"state": "t_state"},
    }


def test_an_edit_the_engine_refuses_is_shown_at_its_field_and_not_saved(page, tmp_path, monkeypatch):
    _serve(page, tmp_path, monkeypatch)
    before = (tmp_path / PROJECT_FILE).read_text()
    page.locator(".flowdash-admin-row-fleet").get_by_text("fleet", exact=True).click()
    drawer = _drawer(page)
    expect(drawer).to_contain_text("Edit fleet", timeout=15000)
    _yaml(page).fill("measures:\n  turbines: COUNT(*)\n  capacity: SUM(t_capacity)\n")
    drawer.get_by_role("button", name="Save").click()
    field = drawer.locator(".flowdash-metric-yaml")
    expect(field).to_contain_text("Line 3: Unknown column 't_capacity' in measure 'capacity'. Did you mean 't_cap'?")
    expect(drawer.get_by_role("alert").first).to_contain_text("Fix the fields marked below")
    assert (tmp_path / PROJECT_FILE).read_text() == before

    drawer.locator(".flowdash-metric-name input").fill("fleet size")
    _yaml(page).fill("measures:\n  turbines: COUNT(*)\n  capacity: SUM(t_cap)\n")
    drawer.get_by_role("button", name="Save").click()
    expect(drawer.locator(".flowdash-metric-name")).to_contain_text("letters, digits and underscores")
    expect(field).not_to_contain_text("Unknown column")

    drawer.locator(".flowdash-metric-name input").fill("fleet")
    drawer.get_by_role("button", name="Save").click()
    expect(drawer).to_be_hidden()
    assert _metrics(tmp_path)["fleet"]["measures"] == {"turbines": "COUNT(*)", "capacity": "SUM(t_cap)"}


def test_deleting_a_metric_dashboards_use_lists_them_first(page, tmp_path, monkeypatch):
    _serve(page, tmp_path, monkeypatch, dashboards=[("Fleet overview", "fleet"), ("Other", "other")])
    row = page.locator(".flowdash-admin-row-fleet")
    expect(row.locator(".k").filter(has_text="1").first).to_be_visible()
    row.locator(".flowdash-admin-metric-menu button").click()
    page.get_by_role("menuitem", name="Delete").click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text("Delete metric fleet?", timeout=15000)
    expect(dialog).to_contain_text("1 dashboard uses fleet")
    expect(dialog).to_contain_text("Fleet overview")
    expect(dialog).not_to_contain_text("Other")
    dialog.get_by_role("button", name="Cancel").click()
    expect(dialog).to_be_hidden()
    assert "fleet" in _metrics(tmp_path)

    row.locator(".flowdash-admin-metric-menu button").click()
    page.get_by_role("menuitem", name="Delete").click()
    dialog.get_by_role("button", name="Delete metric").click()
    expect(row).to_have_count(0)
    assert "fleet" not in _metrics(tmp_path)


def test_the_wizard_suggests_key_numbers_and_writes_one_on_finish(page, tmp_path, monkeypatch):
    _serve(page, tmp_path, monkeypatch, suffix="/admin/setup", metrics="")
    footer = page.locator(".flowdash-wizard-footer")
    footer.get_by_role("button", name="Start").click()
    content = page.locator(".flowdash-wizard-content")
    expect(content).to_contain_text("Define your key numbers")
    expect(content).to_contain_text("the assistant uses them instead of writing its own SQL")
    expect(content).to_contain_text("turbines")

    editor = page.locator(".flowdash-wizard-metrics")
    editor.locator(".flowdash-metric-name input").fill("capacity")
    editor.locator(".flowdash-metric-table [role=combobox]").click()
    page.locator(".MuiMenuItem-root", has_text="No table: a SQL query").click()
    expect(editor.locator(".flowdash-metric-yaml textarea").first).to_have_value(re.compile("^sql: "))
    editor.locator(".flowdash-metric-yaml textarea").first.fill(
        "sql: SELECT SUM(p_cap) AS capacity_mw FROM (SELECT p_name, MAX(p_cap) AS p_cap FROM turbines GROUP BY p_name)\n"
    )
    editor.get_by_role("button", name="Run preview").click()
    expect(editor.locator(".flowdash-metric-preview")).to_contain_text("10.50")

    for heading in ("Connect an AI provider", "Make it yours", "Who can do what", "Review and finish"):
        footer.get_by_role("button", name="Continue").click()
        expect(page.locator(".flowdash-wizard-content")).to_contain_text(heading)
    expect(page.locator(".flowdash-wizard-diff")).to_contain_text("capacity:")
    footer.get_by_role("button", name="Finish and open Lumen").click()
    wait_until(lambda: "capacity" in _metrics(tmp_path), page)
    assert _metrics(tmp_path)["capacity"]["source"] == "turbines"


def test_a_skipped_key_numbers_step_writes_no_metric(page, tmp_path, monkeypatch):
    _serve(page, tmp_path, monkeypatch, suffix="/admin/setup", metrics="")
    footer = page.locator(".flowdash-wizard-footer")
    footer.get_by_role("button", name="Start").click()
    expect(page.locator(".flowdash-wizard-content")).to_contain_text("Define your key numbers")
    page.locator(".flowdash-wizard-metrics .flowdash-metric-name input").fill("capacity")
    footer.get_by_role("button", name="Skip this step").click()
    expect(page.locator(".flowdash-wizard-content")).not_to_contain_text("Define your key numbers")
