"""Unit tests for MCP integrations in Lumen's Data section and ``tool_authorize``."""
import asyncio
import getpass
import sys

from dataclasses import replace

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("panel_flowdash.admin.wizard")

from bokeh.document import Document
from panel.io.state import set_curdoc
from panel_flowdash import MemoryDashboardStore
from panel_flowdash.auth import Identity, Permission

from lumen.flowdash import METRICS_KEY, SOURCES_KEY
from lumen.flowdash.app import build_routes
from lumen.flowdash.data_section import (
    DataPage, data_section, data_view, moved_admin_pages,
    redirect_moved_admin_pages,
)
from lumen.flowdash.mcp import (
    attach_mcp_access, can_access_mcp_server, can_access_mcp_tool,
    make_tool_authorize, mcp_supported, visible_mcp_servers,
)
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

    def _authorize_admin(self):
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
    assert "mcp" not in moved_admin_pages()

    project = _make_project(tmp_path)
    assert visible_mcp_servers(project) == []

    opened = []
    session = Session(project, admin=True)
    assert not attach_mcp_access(session)

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

    routes = _serve(tmp_path, monkeypatch)
    with set_curdoc(Document()):
        app = routes["/"]()
        assert app.admin_section_redirects.get("mcp") == "/data/integrations"
        admin_page = AdminPage(app, app._settings_project())
        assert "mcp" not in admin_page._sections
        assert "uploads" in admin_page._sections

    shown = []

    class StubApp:
        def _admin_view(self, pathname):
            shown.append(pathname)
            return "admin"

    stub = StubApp()
    redirect_moved_admin_pages(stub)
    assert "/data/integrations" in stub._admin_view("/admin/mcp").object
    assert stub._admin_view("/admin/general") == "admin"
    assert shown == ["/admin/general"]


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
  admins: [alice]
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
    from panel_flowdash.mcp import MCP_CHECKS, MCPServerCheck, MCPToolInfo

    project = _make_project(tmp_path, MCP_YAML)
    MCP_CHECKS["github"] = MCPServerCheck(
        True, "Connected (2 tools)",
        tools=(MCPToolInfo("list_issues", "List issues"), MCPToolInfo("create_issue", "Create issue")),
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

        cards = [
            cls
            for o in view.integrations_view.__panel__().select()
            for cls in getattr(o, "css_classes", [])
            if cls.startswith("flowdash-mcp-card-")
        ]
        assert set(cards) == {
            "flowdash-mcp-card-github",
            "flowdash-mcp-card-public_docs",
            "flowdash-mcp-card-disabled_srv",
        }
        text = _text(view.integrations_view)
        assert "list_issues" in text and "create_issue" in text
    finally:
        MCP_CHECKS.pop("github", None)


@needs_mcp
def test_viewer_sees_read_only_integrations_filtered_by_access(tmp_path):
    from panel_flowdash.mcp import MCP_CHECKS, MCPServerCheck, MCPToolInfo

    project = _make_project(tmp_path, MCP_YAML)
    MCP_CHECKS["github"] = MCPServerCheck(
        True, "Connected (2 tools)",
        tools=(MCPToolInfo("list_issues", "List issues"), MCPToolInfo("create_issue", "Create issue")),
    )
    try:
        # Engineer without maintainer role: sees github and public_docs, not disabled_srv;
        # sees list_issues on github, not create_issue.
        eng_session = Session(project, admin=False, user="bob", groups=("engineering",))
        attach_mcp_access(eng_session)
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
        attach_mcp_access(outsider)
        out_view = DataPage(outsider, project, "integrations", admin=False, navigate=lambda _u: None)
        assert [s.name for s in out_view.integrations] == ["public_docs"]

        # Explicitly denied user in engineering: github is hidden even though group matches.
        denied = Session(project, admin=False, user="mallory", groups=("engineering", "maintainers"))
        attach_mcp_access(denied)
        denied_view = DataPage(denied, project, "integrations", admin=False, navigate=lambda _u: None)
        assert [s.name for s in denied_view.integrations] == ["public_docs"]
    finally:
        MCP_CHECKS.pop("github", None)


# ---------------------------------------------------------------------------
# Per-user access semantics and FlowDashApp.tool_authorize


@needs_mcp
def test_mcp_access_semantics_match_lumen_access():
    from panel_flowdash.mcp import MCPServerConfig

    server = MCPServerConfig(
        name="github",
        url="https://mcp.example.com/github",
        access=Permission.from_spec(allow_groups=["engineering"], deny_users=["bad_admin"]),
        tool_access={
            "create_*": Permission.from_spec(allow_groups=["maintainers"]),
            "delete_*": Permission.from_spec(deny_groups=["interns"]),
        },
    )
    eng = Identity("bob", groups=frozenset({"engineering"}))
    maint = Identity("dana", groups=frozenset({"engineering", "maintainers"}))
    intern = Identity("ivy", groups=frozenset({"engineering", "interns"}))
    outsider = Identity("carol", groups=frozenset({"sales"}))
    admin_id = Identity("alice", groups=frozenset())
    denied_admin = Identity("bad_admin", groups=frozenset({"engineering", "maintainers"}))

    # Missing identity fails closed on restricted servers.
    assert not can_access_mcp_server(server, identity=None, admin=False)
    assert not can_access_mcp_tool(server, "list_issues", identity=None, admin=False)

    # Unrestricted server allows even without identity.
    open_server = MCPServerConfig(name="open", url="https://mcp.example.com/open")
    assert can_access_mcp_server(open_server, identity=None, admin=False)
    assert can_access_mcp_tool(open_server, "anything", identity=None, admin=False)

    # Allowlist + admin bypass + explicit deny precedence.
    assert can_access_mcp_server(server, identity=eng, admin=False)
    assert not can_access_mcp_server(server, identity=outsider, admin=False)
    assert can_access_mcp_server(server, identity=admin_id, admin=True)
    assert not can_access_mcp_server(server, identity=denied_admin, admin=True)

    # Tool-level rules intersect with server access.
    assert can_access_mcp_tool(server, "list_issues", identity=eng, admin=False)
    assert not can_access_mcp_tool(server, "create_issue", identity=eng, admin=False)
    assert can_access_mcp_tool(server, "create_issue", identity=maint, admin=False)
    assert can_access_mcp_tool(server, "create_issue", identity=admin_id, admin=True)
    assert not can_access_mcp_tool(server, "create_issue", identity=denied_admin, admin=True)
    assert can_access_mcp_tool(server, "delete_comment", identity=eng, admin=False)
    assert not can_access_mcp_tool(server, "delete_comment", identity=intern, admin=False)


@needs_mcp
def test_tool_authorize_hook_enforces_rules_live_and_composes(tmp_path):
    project = _make_project(tmp_path, MCP_YAML)
    eng = Identity("bob", groups=frozenset({"engineering"}))
    maint = Identity("dana", groups=frozenset({"engineering", "maintainers"}))
    admin_id = Identity("alice")

    # Session as engineering viewer.
    viewer = Session(project, admin=False, user="bob", groups=("engineering",))
    attach_mcp_access(viewer)
    assert viewer.tool_authorize(eng, "list_tables")
    assert viewer.tool_authorize(eng, "mcp_public_docs__search")
    assert viewer.tool_authorize(eng, "mcp_github__list_issues")
    assert not viewer.tool_authorize(eng, "mcp_github__create_issue")
    assert not viewer.tool_authorize(eng, "mcp_disabled_srv__ping")
    assert not viewer.tool_authorize(eng, "mcp_unknown__ping")

    # Admin session bypasses allowlist for its own identity, but does not leak admin to other identities.
    admin_session = Session(project, admin=True, user="alice")
    attach_mcp_access(admin_session)
    assert admin_session.tool_authorize(admin_id, "mcp_github__create_issue")
    assert not admin_session.tool_authorize(eng, "mcp_github__create_issue")
    assert admin_session.tool_authorize(maint, "mcp_github__create_issue")

    # Live reload: editing flowdash.yaml updates authorization immediately.
    project.path = str(tmp_path / PROJECT_FILE)
    (tmp_path / PROJECT_FILE).write_text(
        MCP_YAML.replace("deny_users: [mallory]", "deny_users: [mallory, alice]")
    )
    assert not admin_session.tool_authorize(admin_id, "mcp_github__list_issues")

    # Composition with caller-provided tool_authorize (both must allow; exceptions fail closed).
    custom_denied = make_tool_authorize(
        viewer, inner=lambda _id, name: name != "mcp_public_docs__search",
    )
    assert not custom_denied(eng, "mcp_public_docs__search")
    assert custom_denied(eng, "mcp_github__list_issues")
    assert not custom_denied(eng, "mcp_github__create_issue")

    def exploding(_id, _name):
        raise RuntimeError("boom")

    failing_closed = make_tool_authorize(viewer, inner=exploding)
    assert not failing_closed(eng, "mcp_public_docs__search")
    assert not failing_closed(eng, "list_tables")


@needs_mcp
def test_build_routes_wires_tool_authorize_into_resilient_mcp_toolset(tmp_path, monkeypatch):
    from panel_flowdash.agent.mcp import build_mcp_toolset
    from panel_flowdash.mcp import MCPServerConfig
    from pydantic_ai import RunContext
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
    from pydantic_ai.usage import RunUsage

    class FakeToolset(AbstractToolset):
        def __init__(self):
            self.called = []

        @property
        def id(self):
            return "fake"

        async def get_tools(self, ctx):
            return {
                name: ToolsetTool(
                    toolset=self,
                    tool_def=ToolDefinition(name=name, description=f"Run {name}"),
                    max_retries=0,
                    args_validator=None,
                )
                for name in ("list_issues", "create_issue")
            }

        async def call_tool(self, name, tool_args, ctx, tool):
            self.called.append((name, tool_args))
            return f"ok:{name}"

    fake = FakeToolset()
    server = MCPServerConfig(
        name="github",
        url="https://mcp.example.com/github",
        tool_access={"create_*": Permission.from_spec(allow_users=["someone_else"])},
        toolset=fake,
    )
    yaml_text = "general:\n  admins: [someone_else]\n"
    routes = _serve(tmp_path, monkeypatch, yaml_text=yaml_text, mcp_servers={"github": server})
    with set_curdoc(Document()):
        app = routes["/"]()
        assert app.tool_authorize is not None
        assert app._tool_authorized("mcp_github__list_issues")
        assert not app._tool_authorized("mcp_github__create_issue")

        toolset = build_mcp_toolset(
            server,
            identity=app._identity,
            is_admin=app.is_admin,
            tool_authorize=app.tool_authorize,
            record_check=False,
        )

        async def exercise():
            ctx = RunContext(deps=None, model=None, usage=RunUsage())
            async with toolset:
                tools = await toolset.get_tools(ctx)
                assert list(tools) == ["mcp_github__list_issues"]
                allowed_result = await toolset.call_tool(
                    "mcp_github__list_issues", {}, ctx, tools["mcp_github__list_issues"],
                )
                assert allowed_result == "ok:list_issues"
                denied_tool = replace(
                    tools["mcp_github__list_issues"],
                    tool_def=ToolDefinition(name="mcp_github__create_issue", description=""),
                )
                denied_result = await toolset.call_tool(
                    "mcp_github__create_issue", {}, ctx, denied_tool,
                )
                assert "not authorized" in denied_result.lower()

        asyncio.run(exercise())
        assert fake.called == [("list_issues", {})]
