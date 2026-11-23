"""
"Save as metric…" from a KPI's node menu, driven through the browser.
Runs where pytest-playwright is installed.
"""
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")
node_actions = pytest.importorskip("panel_flowdash.node_actions")
if not hasattr(node_actions, "NodeActionContext"):
    pytest.skip("panel-flowdash without node action contexts", allow_module_level=True)

import yaml

from panel.tests.util import serve_component, wait_until
from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps
from playwright.sync_api import expect

from lumen.flowdash import COMPONENTS, METRICS_KEY, bind_source
from lumen.flowdash.metrics import ProjectMetrics
from lumen.flowdash.project import load_project
from lumen.sources.duckdb import DuckDBSource

from .test_save_metric import CAPACITY, ad_hoc_kpi, turbines_sql

# A menu built after the page loaded sends its model in full with its click, which Bokeh warns about.
pytestmark = pytest.mark.filterwarnings("ignore:reference already known:UserWarning")

CORRECTED = """source: data
sql: |
  SELECT SUM(p_cap) AS capacity
  FROM (SELECT p_name, MAX(p_cap) AS p_cap FROM turbines GROUP BY p_name) AS p
"""


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


def _serve(page, tmp_path, admin: bool):
    """Serve a dashboard with an ad hoc KPI, built in the session as an app builds its editor."""
    project_file = tmp_path / "flowdash.yaml"
    project_file.write_text("metrics: {}\n")
    built = {}

    def app():
        editor = FlowDash(components=COMPONENTS, notifications=False, authorize_admin=lambda fresh=False: admin)
        ref = bind_source(editor, DuckDBSource(uri=":memory:", tables={"turbines": turbines_sql()}))
        project = load_project(project_file, editor.context)
        editor.context[METRICS_KEY] = ProjectMetrics(project)
        editor.project = project
        editor.new_dashboard("Wind")
        ops = DashboardOps(editor)
        ops.ref = ref
        built.update(editor=editor, ops=ops, kpi=ad_hoc_kpi(ops))
        return editor

    page.set_viewport_size({"width": 1400, "height": 950})
    serve_component(page, app)
    wait_until(lambda: "kpi" in built, page, timeout=20000)
    return built["editor"], built["ops"], built["kpi"], project_file


def _open_node_menu(page):
    node = page.locator(".react-flow__node", has_text="total_capacity").filter(has_text="KPI").first
    expect(node).to_be_visible(timeout=20000)
    node.click()
    expect(page.locator(".fd-inspector-provenance")).to_contain_text("Ad hoc", timeout=10000)
    page.locator(".fd-inspector-menu button").first.click()


def test_an_admin_saves_a_corrected_metric_and_points_the_kpi_at_it(page, tmp_path):
    editor, ops, kpi, project_file = _serve(page, tmp_path, admin=True)
    _open_node_menu(page)
    page.get_by_role("menuitem", name="Save as metric…").click()

    dialog = page.locator(".lumen-save-metric")
    expect(dialog.get_by_text("They agree.")).to_be_visible(timeout=15000)
    expect(dialog.locator(".lumen-save-metric-now")).to_contain_text("14")
    dialog.locator(".lumen-save-metric-name input").fill("capacity")
    dialog.locator(".ace_editor").first.click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.press("Backspace")
    page.keyboard.insert_text(CORRECTED)
    dialog.locator(".lumen-save-metric-preview button, button.lumen-save-metric-preview").first.click()
    expect(dialog.get_by_text("They differ", exact=False)).to_be_visible(timeout=15000)
    expect(dialog.locator(".lumen-save-metric-value")).to_contain_text("6")

    dialog.get_by_role("button", name="Save metric").click()
    expect(dialog.get_by_text("Saved the metric capacity")).to_be_visible(timeout=15000)
    assert yaml.safe_load(project_file.read_text())["metrics"]["capacity"]["source"] == "data"

    dialog.get_by_role("button", name="Point the node at it").click()
    expect(dialog).to_be_hidden(timeout=15000)
    wait_until(lambda: editor.graph.get_state(kpi).value == CAPACITY, page, timeout=10000)
    expect(page.locator(".fd-inspector-provenance")).to_contain_text("Metric: capacity", timeout=10000)
    assert "capacity_metric" in {n["name"] for n in ops.nodes()}


def test_a_viewer_is_not_offered_the_action(page, tmp_path):
    _editor, _ops, _kpi, project_file = _serve(page, tmp_path, admin=False)
    _open_node_menu(page)
    expect(page.get_by_role("menuitem", name="Duplicate")).to_be_visible(timeout=10000)
    expect(page.get_by_role("menuitem", name="Save as metric…")).to_have_count(0)
    assert yaml.safe_load(project_file.read_text()) == {"metrics": {}}
