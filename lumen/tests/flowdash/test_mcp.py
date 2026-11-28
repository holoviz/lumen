"""Unit tests for MCP integrations in Lumen's Data section."""
import getpass
import sys

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("panel_flowdash.admin.wizard")

from bokeh.document import Document
from panel.io.state import set_curdoc
from panel_flowdash import MemoryDashboardStore
from panel_flowdash.auth import Identity

from lumen.flowdash import METRICS_KEY, SOURCES_KEY
from lumen.flowdash.app import build_routes
from lumen.flowdash.data_section import (
    MOVED, DataPage, data_section, data_view,
)
from lumen.flowdash.mcp import mcp_supported, visible_mcp_servers
from lumen.flowdash.metrics import ProjectMetrics
from lumen.flowdash.project import PROJECT_FILE, load_project
from lumen.flowdash.uploads import UPLOADS_KEY, Uploads
from lumen.sources.duckdb import DuckDBSource

TABLES = {"orders": "SELECT 1 AS order_id"}

needs_mcp = pytest.mark.skipif(
    not mcp_supported() or data_section() is None,
    reason="panel-flowdash has no MCP or sidebar section support",
)


class Session:
    """Minimal FlowDashApp session stub matching ``test_data_section.py``."""

    def __init__(self, project, admin=False, user="bo", groups=(), mcp_servers=None, tool_authorize=None):
        self.project = project
        self.is_admin = self.admin = admin
        self.store = MemoryDashboardStore()
        self._identity = Identity(user, groups=frozenset(groups)) if user is not None else None
        self.mcp_servers = dict(mcp_servers or {})
        self.tool_authorize = tool_authorize
        self.notified = []

    def _still_admin(self):
        return self.admin

    def _settings_project(self):
        return self.project

    def _tool_authorized(self, tool_name):
        if self.tool_authorize is None:
            return True
        return bool(self.tool_authorize(self._identity, tool_name))

    def _notify(self, severity, message, duration=3000):
        self.notified.append((severity, message))


class Context:
    def __init__(self, app):
        self.app = app
        self.is_admin = app.is_admin
        self.opened = []

    def navigate(self, url):
        self.opened.append(url)


def _text(view):
    return "".join(str(o.object) for o in view.__panel__().select() if isinstance(getattr(o, "object", None), str))


def _card_classes(view):
    return {
        cls
        for o in view.__panel__().select()
        for cls in getattr(o, "css_classes", [])
        if cls.startswith("flowdash-mcp-card-")
    }


def _make_project(tmp_path, yaml_text=""):
    source = DuckDBSource(uri=":memory:", tables=TABLES, name="shop")
    sources = {"shop": source}
    path = tmp_path / PROJECT_FILE
    path.write_text(yaml_text)
    project = load_project(path, {SOURCES_KEY: sources})
    project.context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", sources, project=project)
    project.context[METRICS_KEY] = ProjectMetrics(project)
    return project


def _serve(tmp_path, monkeypatch, yaml_text=None, **params):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,p_cap\nAlta,6.0\n")
    text = yaml_text if yaml_text is not None else f"general:\n  admins: [{getpass.getuser()}]\n"
    (tmp_path / PROJECT_FILE).write_text(text)
    return build_routes(["turbines.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE, **params)


# ---------------------------------------------------------------------------
# Feature detection without panel-flowdash MCP support


def test_without_mcp_support_lumen_behaves_like_branch_2_0(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "panel_flowdash.mcp", None)
    monkeypatch.setitem(sys.modules, "panel_flowdash.admin.mcp", None)
    assert not mcp_supported()
    assert "mcp" not in MOVED

    project = _make_project(tmp_path)
    assert visible_mcp_servers(project) == []

    opened = []
    session = Session(project, admin=True)
    view = DataPage(session, project, "integrations", admin=True, navigate=opened.append)
    assert view.page == "tables"
    nav = next(o for o in view.__panel__().select() if "lumen-data-nav" in getattr(o, "css_classes", []))
    assert [item["label"] for item in nav.items] == ["Tables", "Metrics"]

    routes = _serve(tmp_path, monkeypatch)
    with set_curdoc(Document()):
        app = routes["/"]()
        assert "mcp" not in getattr(app, "admin_section_redirects", {})


# ---------------------------------------------------------------------------
# Admin relocation, Data section nav, and Integrations page


@needs_mcp
def test_data_section_relocates_mcp_out_of_admin_view(tmp_path, monkeypatch):
    from panel_flowdash.admin.page import AdminPage

    assert "mcp" not in MOVED
    routes = _serve(tmp_path, monkeypatch)
    with set_curdoc(Document()):
        app = routes["/"]()
        assert app.admin_section_redirects.get("mcp") == "/data/integrations"
        admin_page = AdminPage(app, app._settings_project())
        assert "mcp" not in admin_page._sections
        assert "uploads" in admin_page._sections


@needs_mcp
@pytest.mark.parametrize("params", [{"nav_variant": "left"}, {"sections": "own"}])
def test_without_data_section_mcp_stays_in_admin_view(tmp_path, monkeypatch, params):
    from panel_flowdash.admin.page import AdminPage
    from panel_flowdash.sidebar import SidebarSection

    if params.get("sections") == "own":
        own = SidebarSection(id="data", label="Our data", icon="storage", route="ours", view=lambda ctx, path: "")
        params = {"sections": lambda: [own]}
    routes = _serve(tmp_path, monkeypatch, **params)
    with set_curdoc(Document()):
        app = routes["/"]()
        assert "mcp" not in app.admin_section_redirects
        admin_page = AdminPage(app, app._settings_project())
        assert "mcp" in admin_page._sections


MCP_YAML = """\
general:
  admins: [alice, mallory]
mcp:
  servers:
    github:
      transport: http
      url: https://mcp.example.com/github
      description: GitHub issues and PRs
      auto_approve: [list_issues]
      access:
        allow_groups: [engineering]
        deny_users: [mallory]
      tool_access:
        create_*:
          allow_groups: [maintainers]
    public_docs:
      transport: http
      url: https://mcp.example.com/docs
      description: Public docs
    disabled_srv:
      transport: http
      url: https://mcp.example.com/off
      enabled: false
"""


@needs_mcp
def test_admin_sees_integrations_nav_all_servers_and_add_drawer(tmp_path):
    from panel_flowdash.mcp import (
        MCPServerCheck, MCPToolInfo, clear_mcp_check, set_mcp_check,
    )

    project = _make_project(tmp_path, MCP_YAML)
    set_mcp_check(
        "github",
        MCPServerCheck(
            True, "Connected (2 tools)",
            tools=(MCPToolInfo("list_issues", "List issues"), MCPToolInfo("create_issue", "Create issue")),
        ),
        project=project,
    )
    try:
        ctx = Context(Session(project, admin=True, user="alice"))
        view = data_view(ctx, "integrations")
        assert view.page == "integrations"

        nav = next(o for o in view.__panel__().select() if "lumen-data-nav" in getattr(o, "css_classes", []))
        assert [item["label"] for item in nav.items] == ["Tables", "Metrics", "Integrations"]
        assert nav.items[2]["secondary"] == "3"
        assert nav.active == (2,)
        assert "3 integrations" in _text(view)

        add_btns = [o for o in view.__panel__().select() if "lumen-data-add-mcp" in getattr(o, "css_classes", [])]
        assert len(add_btns) == 1 and add_btns[0].visible
        add_btns[0].clicks += 1
        assert view.integrations_view._drawer.open

        assert _card_classes(view.integrations_view) == {
            "flowdash-mcp-card-github",
            "flowdash-mcp-card-public_docs",
            "flowdash-mcp-card-disabled_srv",
        }
        text = _text(view.integrations_view)
        assert "list_issues" in text and "create_issue" in text
    finally:
        clear_mcp_check("github", project=project)


@needs_mcp
def test_denied_admin_still_sees_all_servers_in_editable_view(tmp_path):
    project = _make_project(tmp_path, MCP_YAML)
    # mallory is an admin explicitly listed in github's deny_users; the editable
    # admin UI and nav count must still show github so mallory can manage/fix it.
    denied_admin = Session(project, admin=True, user="mallory", groups=("engineering",))
    assert [s.name for s in visible_mcp_servers(project, denied_admin, admin=True)] == [
        "github", "public_docs", "disabled_srv",
    ]

    view = data_view(Context(denied_admin), "integrations")
    nav = next(o for o in view.__panel__().select() if "lumen-data-nav" in getattr(o, "css_classes", []))
    assert nav.items[2]["secondary"] == "3"
    assert _card_classes(view.integrations_view) == {
        "flowdash-mcp-card-github",
        "flowdash-mcp-card-public_docs",
        "flowdash-mcp-card-disabled_srv",
    }


@needs_mcp
def test_viewer_sees_read_only_integrations_filtered_by_access(tmp_path):
    from panel_flowdash.mcp import (
        MCPServerCheck, MCPToolInfo, clear_mcp_check, set_mcp_check,
    )

    project = _make_project(tmp_path, MCP_YAML)
    set_mcp_check(
        "github",
        MCPServerCheck(
            True, "Connected (2 tools)",
            tools=(MCPToolInfo("list_issues", "List issues"), MCPToolInfo("create_issue", "Create issue")),
        ),
        project=project,
    )
    try:
        # Engineer without maintainer role: sees github and public_docs, not disabled_srv;
        # sees list_issues on github, not create_issue.
        eng_session = Session(project, admin=False, user="bob", groups=("engineering",))
        eng_view = DataPage(eng_session, project, "integrations", admin=False, navigate=lambda _u: None)
        assert [s.name for s in eng_view.integrations] == ["github", "public_docs"]
        add_btns = [o for o in eng_view.__panel__().select() if "lumen-data-add-mcp" in getattr(o, "css_classes", [])]
        assert len(add_btns) == 1 and not add_btns[0].visible
        assert eng_view.integrations_view._drawer is None
        eng_text = _text(eng_view.integrations_view)
        assert "list_issues" in eng_text
        assert "create_issue" not in eng_text
        assert "mcp.example.com" not in eng_text

        # Outsider: only public_docs is visible; github is hidden.
        outsider = Session(project, admin=False, user="carol", groups=("sales",))
        out_view = DataPage(outsider, project, "integrations", admin=False, navigate=lambda _u: None)
        assert [s.name for s in out_view.integrations] == ["public_docs"]

        # Explicitly denied user in engineering: github is hidden even though group matches.
        denied = Session(project, admin=False, user="mallory", groups=("engineering", "maintainers"))
        denied_view = DataPage(denied, project, "integrations", admin=False, navigate=lambda _u: None)
        assert [s.name for s in denied_view.integrations] == ["public_docs"]

        # Missing identity fails closed on restricted servers while keeping open servers visible.
        anon = Session(project, admin=False, user=None)
        anon_view = DataPage(anon, project, "integrations", admin=False, navigate=lambda _u: None)
        assert [s.name for s in anon_view.integrations] == ["public_docs"]
    finally:
        clear_mcp_check("github", project=project)
