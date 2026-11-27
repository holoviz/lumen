"""
Playwright UI tests for MCP server integrations in Lumen's Data section:
admin configuration at ``/data/integrations``, redirect from ``/admin/mcp``,
and read-only per-user access filtering for non-admins.
"""
import getpass

from urllib.parse import urlsplit

import pytest
import yaml

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")
pytest.importorskip("panel_flowdash.sidebar")
pytest.importorskip("panel_flowdash.mcp")
pytest.importorskip("panel_flowdash.admin.mcp")

from panel.tests.util import serve_component
from playwright.sync_api import expect

from lumen.flowdash.app import build_routes
from lumen.flowdash.project import PROJECT_FILE

pytestmark = pytest.mark.filterwarnings("ignore:reference already known")


@pytest.fixture(autouse=True)
def stop_servers():
    yield
    from panel.io.state import state

    state.kill_all_servers()


def _path(page):
    return urlsplit(page.url).path


def test_admin_manages_mcp_servers_in_data_integrations(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,p_cap\nAlta,6.0\n")
    project_path = tmp_path / PROJECT_FILE
    project_path.write_text(f"general:\n  admins: [{getpass.getuser()}]\n")
    routes = build_routes(
        ["turbines.csv"], db_path=tmp_path / "dash.db", project=project_path,
    )
    page.set_viewport_size({"width": 1440, "height": 900})
    serve_component(page, routes, suffix="/admin/mcp")
    base = "{0.scheme}://{0.netloc}".format(urlsplit(page.url))

    # /admin/mcp moved to Data -> Integrations.
    expect(page.locator(".lumen-data-title")).to_have_text("Integrations", timeout=20000)
    assert _path(page) == "/data/integrations"

    # Admin nav no longer duplicates MCP.
    page.goto(f"{base}/admin/uploads")
    expect(page.locator(".flowdash-admin-content")).to_contain_text("Let people upload data", timeout=20000)
    expect(page.locator(".flowdash-admin")).not_to_contain_text("MCP servers")

    # Open Data -> Integrations via the section nav.
    page.goto(f"{base}/data")
    expect(page.locator(".lumen-data-title")).to_have_text("Tables", timeout=20000)
    page.locator(".lumen-data-nav").get_by_text("Integrations").click()
    expect(page.locator(".lumen-data-title")).to_have_text("Integrations", timeout=20000)
    assert _path(page) == "/data/integrations"

    # Add an HTTP MCP server through the hosted MCPSettings drawer.
    page.locator(".lumen-data-add-mcp button").click()
    drawer = page.locator(".flowdash-admin-mcp-drawer")
    expect(drawer).to_be_visible(timeout=15000)
    drawer.get_by_label("Server id").fill("docs")
    drawer.get_by_label("Description").fill("External documentation search")
    drawer.get_by_label("Server URL").fill("https://mcp.example.com/docs")
    drawer.locator(".flowdash-mcp-save button").click()

    card = page.locator(".flowdash-mcp-card-docs")
    expect(card).to_be_visible(timeout=20000)
    expect(card).to_contain_text("HTTP")
    expect(card).to_contain_text("External documentation search")
    expect(card).to_contain_text("Not tested")

    saved = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    assert saved["mcp"]["servers"]["docs"]["url"] == "https://mcp.example.com/docs"


VIEWER_PROJECT = """\
general:
  admins: [someone_else]
mcp:
  servers:
    public_docs:
      transport: http
      url: https://secret-internal.example.com/docs
      description: Public documentation tools
      auto_approve: [search_docs]
      tool_access:
        admin_*:
          allow_users: [someone_else]
    finance_only:
      transport: http
      url: https://secret-internal.example.com/finance
      description: Restricted finance server
      access:
        allow_users: [someone_else]
    disabled_srv:
      transport: http
      url: https://secret-internal.example.com/disabled
      enabled: false
"""


def test_viewer_sees_read_only_integrations_with_access_filtering(page, tmp_path, monkeypatch):
    from panel_flowdash.mcp import MCP_CHECKS, MCPServerCheck, MCPToolInfo

    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,p_cap\nAlta,6.0\n")
    project_path = tmp_path / PROJECT_FILE
    project_path.write_text(VIEWER_PROJECT)
    MCP_CHECKS["public_docs"] = MCPServerCheck(
        True,
        "Connected (2 tools)",
        tools=(
            MCPToolInfo("search_docs", "Search public docs"),
            MCPToolInfo("admin_reindex", "Reindex corpus"),
        ),
    )
    try:
        routes = build_routes(
            ["turbines.csv"], db_path=tmp_path / "dash.db", project=project_path,
        )
        page.set_viewport_size({"width": 1440, "height": 900})
        serve_component(page, routes, suffix="/data/integrations")

        expect(page.locator(".lumen-data-title")).to_have_text("Integrations", timeout=20000)
        expect(page.locator(".lumen-data-add-mcp")).to_be_hidden()
        expect(page.locator(".flowdash-mcp-edit")).to_have_count(0)
        expect(page.locator(".flowdash-mcp-test")).to_have_count(0)
        expect(page.locator(".flowdash-mcp-toggle")).to_have_count(0)

        card = page.locator(".flowdash-mcp-card-public_docs")
        expect(card).to_be_visible(timeout=15000)
        expect(card).to_contain_text("Connected")
        expect(card).to_contain_text("search_docs")
        expect(card).not_to_contain_text("admin_reindex")
        expect(card).not_to_contain_text("secret-internal.example.com")

        expect(page.locator(".flowdash-mcp-card-finance_only")).to_have_count(0)
        expect(page.locator(".flowdash-mcp-card-disabled_srv")).to_have_count(0)
    finally:
        MCP_CHECKS.pop("public_docs", None)
