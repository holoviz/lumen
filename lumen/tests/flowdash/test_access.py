"""Per-resource authorization for FlowDash sources, tables, metrics, nodes and the Data section."""

from __future__ import annotations

import weakref

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("panel_flowdash")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, OperationError
from panel_flowdash.auth import Identity, Permission
from panel_flowdash.editor import _current

from lumen.flowdash import (
    ADMIN_KEY, CATALOG_KEY, COMPONENTS, IDENTITY_KEY, METRICS_KEY,
    SOURCE_ALIASES_KEY, SOURCES_KEY, Catalog, ProjectCatalog, attach_access,
    authorized_metrics, authorized_sources, authorized_tables, bind_identity,
    can_access, can_access_metric, can_access_source, can_access_table,
    check_source_config, check_sql_config, check_table_config, is_admin,
)
from lumen.flowdash.app import build_routes
from lumen.flowdash.catalog import set_assistant
from lumen.flowdash.catalog_admin import CatalogSettings
from lumen.flowdash.choices import (
    metric_choices, metric_presets, parameter_choices, source_choices,
    source_presets, table_choices, table_presets,
)
from lumen.flowdash.data_section import DataPage
from lumen.flowdash.metrics import ProjectMetrics
from lumen.flowdash.metrics_admin import MetricsSettings
from lumen.flowdash.nodes import add_legacy_tables
from lumen.flowdash.project import (
    PROJECT_FILE, ProjectError, load_project, parse_project,
)
from lumen.sources.duckdb import DuckDBSource

HR_TABLES = {
    "employees": "SELECT * FROM (VALUES (1, 'Ada', 'eng'), (2, 'Bo', 'sales')) AS t(id, name, dept)",
    "salaries": "SELECT * FROM (VALUES (1, 120000), (2, 95000)) AS t(id, salary)",
}

SALES_TABLES = {
    "orders": "SELECT * FROM (VALUES (101, 50), (102, 75)) AS t(order_id, amount)",
}

PROJECT_YAML = """\
catalog:
  hr:
    access:
      allow_groups: [hr, finance]
      deny_users: [contractor]
    tables:
      salaries:
        assistant: false
        access:
          allow_groups: [hr]
          deny_users: [blocked_admin]
metrics:
  headcount:
    description: Number of employees.
    source: hr
    table: employees
    measures:
      total: COUNT(*)
  payroll:
    description: Total payroll.
    source: hr
    table: salaries
    measures:
      total_pay: SUM(salary)
    parameters:
      min_salary:
        type: integer
        default: 0
        column: salary
        op: ">="
  orders_total:
    description: Total order value.
    source: sales
    table: orders
    access:
      allow_groups: [sales]
    measures:
      revenue: SUM(amount)
  public_orders:
    description: Public order count.
    source: sales
    table: orders
    measures:
      n: COUNT(*)
"""


class FakeApp:
    def __init__(self, user: str = "ana", groups: tuple[str, ...] = (), *, admin: bool = False):
        self._identity = Identity(user=user, groups=frozenset(groups))
        self.admin = admin
        self.is_admin = admin
        self.store = None
        self._editor = None

    def still_admin(self) -> bool:
        return self.admin

    def _still_admin(self) -> bool:
        return self.admin


def _make_context(project_yaml: str = PROJECT_YAML, tmp_path=None):
    hr = DuckDBSource(uri=":memory:", tables=HR_TABLES)
    sales = DuckDBSource(uri=":memory:", tables=SALES_TABLES)
    sources = {"hr": hr, "sales": sales}
    context = {SOURCES_KEY: sources, SOURCE_ALIASES_KEY: {"people": "hr"}}
    if tmp_path is not None:
        path = tmp_path / PROJECT_FILE
        path.write_text(project_yaml, encoding="utf-8")
        project = load_project(path, context)
    else:
        project = parse_project(project_yaml, PROJECT_FILE, context)
    context[CATALOG_KEY] = ProjectCatalog(project)
    context[METRICS_KEY] = ProjectMetrics(project)
    project.context.setdefault(CATALOG_KEY, context[CATALOG_KEY])
    project.context.setdefault(METRICS_KEY, context[METRICS_KEY])
    return project, context, hr, sales


# ---------------------------------------------------------------------------
# Core permission evaluation and session helpers


def test_can_access_precedence_and_fail_closed():
    empty = Permission()
    assert can_access(empty, None)
    assert can_access(None, None)

    hr_only = Permission.from_spec(allow_groups=["hr"])
    assert not can_access(hr_only, None)
    assert not can_access(hr_only, None, admin=True)
    assert can_access(hr_only, Identity("alice", groups=frozenset({"hr"})))
    assert not can_access(hr_only, Identity("bob", groups=frozenset({"sales"})))
    # Admins bypass allowlists unless explicitly denied.
    assert can_access(hr_only, Identity("root"), admin=True)

    denied_admin = Permission.from_spec(allow_groups=["hr"], deny_users=["root"])
    assert not can_access(denied_admin, Identity("root", groups=frozenset({"hr"})), admin=True)

    deny_only = Permission.from_spec(deny_groups=["contractors"])
    assert can_access(deny_only, Identity("alice", groups=frozenset({"sales"})))
    assert not can_access(deny_only, Identity("carl", groups=frozenset({"contractors"})))
    assert not can_access(deny_only, None)


def test_is_admin_fails_closed():
    assert not is_admin(None)
    assert not is_admin(object())

    class BrokenApp:
        def still_admin(self):
            raise RuntimeError("cannot read flowdash.yaml")

    assert not is_admin(BrokenApp())
    app = FakeApp(admin=True)
    assert is_admin(app)
    app.admin = False
    assert not is_admin(app)


def test_attach_access_binds_identity_and_live_admin():
    app = FakeApp(user="ana", groups=("hr",), admin=True)
    editor = FlowDash(components=COMPONENTS, notifications=False, history=False)
    app._editor = editor
    try:
        attach_access(app)
        assert editor.context[IDENTITY_KEY] == app._identity
        assert editor.context[ADMIN_KEY]() is True
        app.admin = False
        assert editor.context[ADMIN_KEY]() is False
    finally:
        editor.dispose()


# ---------------------------------------------------------------------------
# YAML parsing and preservation


def test_catalog_and_metrics_access_parse_and_validate(tmp_path):
    project, context, _, _ = _make_context(tmp_path=tmp_path)
    catalog = project.get("catalog")
    assert isinstance(catalog, Catalog)
    assert catalog.source_access["hr"] == Permission.from_spec(
        allow_groups=["hr", "finance"],
        deny_users=["contractor"],
    )
    assert catalog.table_access["hr"]["salaries"] == Permission.from_spec(
        allow_groups=["hr"],
        deny_users=["blocked_admin"],
    )
    assert catalog.hidden["hr"] == frozenset({"salaries"})

    metrics = project.get("metrics")
    assert metrics["orders_total"].access == Permission.from_spec(allow_groups=["sales"])
    assert metrics["public_orders"].access.is_empty

    # Toggling assistant back on preserves access rules in flowdash.yaml.
    assert set_assistant(project, "hr", "salaries", True, app=FakeApp(admin=True)) == []
    reloaded = load_project(tmp_path / PROJECT_FILE, context)
    reloaded_cat = reloaded.get("catalog")
    assert reloaded_cat.hidden == {}
    assert reloaded_cat.source_access["hr"] == catalog.source_access["hr"]
    assert reloaded_cat.table_access["hr"]["salaries"] == catalog.table_access["hr"]["salaries"]


@pytest.mark.parametrize(
    "bad_yaml,match",
    [
        ("catalog:\n  hr:\n    access: hr\n", "must be a mapping"),
        ("catalog:\n  hr:\n    access:\n      allow_roles: [hr]\n", "Unknown key 'allow_roles'"),
        ("catalog:\n  hr:\n    tables:\n      salaries:\n        access:\n          allow_groups: [123]\n", "list of non-empty names"),
        ("metrics:\n  m:\n    source: sales\n    table: orders\n    measures: {n: COUNT(*)}\n    access:\n      bad_key: [x]\n", "Unknown key 'bad_key'"),
    ],
)
def test_invalid_access_yaml_raises_project_error(bad_yaml, match):
    with pytest.raises(ProjectError, match=match):
        parse_project(bad_yaml, PROJECT_FILE)


# ---------------------------------------------------------------------------
# Resource evaluators and inheritance


def test_resource_access_inheritance_and_aliases():
    _, context, hr, sales = _make_context()
    add_legacy_tables(hr, {"salaries_csv": "salaries"})

    hr_user = Identity("alice", groups=frozenset({"hr"}))
    finance_user = Identity("bob", groups=frozenset({"finance"}))
    sales_user = Identity("sam", groups=frozenset({"sales"}))
    contractor = Identity("contractor", groups=frozenset({"hr"}))
    blocked_admin = Identity("blocked_admin", groups=frozenset({"admins"}))
    admin = Identity("root", groups=frozenset({"admins"}))

    # Source access (including via alias "people").
    assert can_access_source("hr", context, identity=hr_user)
    assert can_access_source("people", context, identity=hr_user)
    assert not can_access_source("hr", context, identity=sales_user)
    assert not can_access_source("people", context, identity=sales_user)
    assert not can_access_source("hr", context, identity=contractor)
    assert can_access_source("hr", context, identity=admin, admin=True)

    # Table access inherits source access; salaries further restricts to hr.
    assert can_access_table(hr, "employees", context, identity=finance_user)
    assert not can_access_table(hr, "salaries", context, identity=finance_user)
    assert not can_access_table(hr, "salaries_csv", context, identity=finance_user)
    assert can_access_table(hr, "salaries", context, identity=hr_user)
    assert not can_access_table(hr, "employees", context, identity=sales_user)
    # Explicit deny on salaries beats admin.
    assert can_access_table(hr, "employees", context, identity=blocked_admin, admin=True)
    assert not can_access_table(hr, "salaries", context, identity=blocked_admin, admin=True)

    # Metric access inherits source/table access and applies metric's own access.
    assert set(authorized_metrics(context, identity=hr_user)) == {"headcount", "payroll", "public_orders"}
    assert set(authorized_metrics(context, identity=finance_user)) == {"headcount", "public_orders"}
    assert set(authorized_metrics(context, identity=sales_user)) == {"orders_total", "public_orders"}
    assert set(authorized_metrics(context, identity=admin, admin=True)) == {
        "headcount",
        "payroll",
        "orders_total",
        "public_orders",
    }
    assert set(authorized_metrics(context, identity=blocked_admin, admin=True)) == {
        "headcount",
        "orders_total",
        "public_orders",
    }
    assert set(authorized_sources(context, identity=sales_user)) == {"sales"}
    assert authorized_tables(hr, context, identity=finance_user) == ["employees"]
    assert authorized_tables(sales, context, identity=sales_user) == ["orders"]
    # Malformed raw metric access mapping fails closed.
    assert not can_access_metric({"source": "sales", "table": "orders", "access": "invalid"}, context, identity=sales_user)


# ---------------------------------------------------------------------------
# Runtime nodes, checks, and choices/presets


async def test_nodes_and_checks_enforce_per_resource_access():
    _, context, hr, _ = _make_context()
    finance = Identity("bob", groups=frozenset({"finance"}))
    sales_user = Identity("sam", groups=frozenset({"sales"}))
    admin = Identity("root", groups=frozenset({"admins"}))

    # Build a saved dashboard as an admin referencing hr, salaries, employees, and payroll.
    author = FlowDash(components=COMPONENTS, notifications=False, history=False, context=dict(context))
    bind_identity(author, admin, admin=True)
    try:
        author_ops = DashboardOps(author)
        author_ops.add_node("Lumen/source", {"ref": "hr"}, name="hr_src")
        author_ops.add_node("Lumen/table", {"ref": "hr", "table": "employees"}, name="emp_tbl")
        author_ops.add_node("Lumen/table", {"ref": "hr", "table": "salaries"}, name="sal_tbl")
        author_ops.add_node("Lumen/sql", {"sql": "SELECT * FROM main.salaries"}, name="bad_sql")
        author_ops.connect("hr_src.source", "bad_sql.source")
        author_ops.add_node("Lumen/sql", {"sql": "SELECT COUNT(*) AS n FROM employees"}, name="ok_sql")
        author_ops.connect("hr_src.source", "ok_sql.source")
        author_ops.add_node("Lumen/metric", {"metric": "payroll", "measures": ["total_pay"]}, name="pay_m")
        saved = author.export_dashboard()
    finally:
        author.dispose()

    editor = FlowDash(components=COMPONENTS, notifications=False, history=False, context=dict(context))
    bind_identity(editor, sales_user)
    try:
        ops = DashboardOps(editor)
        # Sales user is refused at placement time for hr SourceNode and TableNode.
        with pytest.raises(OperationError, match="permission"):
            ops.add_node("Lumen/source", {"ref": "hr"}, name="hr_src")
        with pytest.raises(OperationError, match="permission"):
            ops.add_node("Lumen/table", {"ref": "hr", "table": "employees"}, name="emp_tbl")

        # Opening the admin-saved dashboard referencing hr fails closed at runtime.
        editor.import_dashboard(saved)
        nodes = ops.preview()["nodes"]
        for blocked in ("hr_src", "emp_tbl", "pay_m"):
            assert nodes[blocked]["status"] in ("invalid", "failed")
            assert "permission" in nodes[blocked]["message"].lower()

        # Sales user cannot resolve headcount metric (inherits hr source) but can resolve orders_total.
        ops.add_node("Lumen/metric", {"metric": "headcount", "measures": ["total"]}, name="hc")
        ops.add_node("Lumen/metric", {"metric": "orders_total", "measures": ["revenue"]}, name="rev")
        nodes = ops.preview()["nodes"]
        assert nodes["hc"]["status"] in ("invalid", "failed")
        assert "permission" in nodes["hc"]["message"].lower()
        assert nodes["rev"]["status"] == "ok"
        assert ops.node_output("rev.data")["revenue"].iloc[0] == 125

        # Config checks refuse unauthorized sources and tables.
        with _current(weakref.ref(editor)):
            assert "permission" in (check_source_config({"ref": "hr"}, {}) or "").lower()
            assert check_source_config({"ref": "sales"}, {}) is None
            assert "permission" in (check_table_config({"ref": "hr", "table": "employees"}, {}) or "").lower()
    finally:
        editor.dispose()

    # Finance user can read hr.employees, but NOT hr.salaries (via TableNode, SQLNode, or MetricNode).
    editor2 = FlowDash(components=COMPONENTS, notifications=False, history=False, context=dict(context))
    bind_identity(editor2, finance)
    try:
        ops = DashboardOps(editor2)
        with pytest.raises(OperationError, match="permission"):
            ops.add_node("Lumen/table", {"ref": "hr", "table": "salaries"}, name="sal")

        editor2.import_dashboard(saved)
        nodes = ops.preview()["nodes"]
        assert nodes["hr_src"]["status"] == "ok"
        assert nodes["emp_tbl"]["status"] == "ok"
        for blocked in ("sal_tbl", "bad_sql", "pay_m"):
            assert nodes[blocked]["status"] in ("invalid", "failed")
            assert "permission" in nodes[blocked]["message"].lower()
        assert nodes["ok_sql"]["status"] == "ok"
        assert ops.node_output("ok_sql.value") == 2

        with _current(weakref.ref(editor2)):
            diag = await check_sql_config({"sql": "SELECT salary FROM salaries"}, {"source": hr})
            assert diag is not None and "permission" in str(diag).lower()
            assert await check_sql_config({"sql": "SELECT name FROM employees"}, {"source": hr}) is None
    finally:
        editor2.dispose()


def test_choices_and_palette_presets_filter_by_access():
    _, context, _, _ = _make_context()
    finance = Identity("bob", groups=frozenset({"finance"}))
    sales_user = Identity("sam", groups=frozenset({"sales"}))

    editor = FlowDash(components=COMPONENTS, notifications=False, history=False, context=dict(context))
    try:
        bind_identity(editor, finance)
        with _current(weakref.ref(editor)):
            assert source_choices({}, {}) == {"ref": ["hr", "sales"]}
            assert table_choices({"ref": "hr"}, {})["table"] == ["employees"]
            assert metric_choices({}, {})["metric"] == ["headcount", "public_orders"]
            assert "min_salary" not in parameter_choices({}, {}).get("parameter", [])
        assert [p["config"]["table"] for p in table_presets(editor.context)] == ["employees", "orders"]
        assert [p["config"]["metric"] for p in metric_presets(editor.context)] == ["headcount", "public_orders"]

        bind_identity(editor, Identity("alice", groups=frozenset({"hr"})))
        with _current(weakref.ref(editor)):
            assert "min_salary" in parameter_choices({}, {})["parameter"]

        bind_identity(editor, sales_user)
        with _current(weakref.ref(editor)):
            assert source_choices({}, {}) == {"ref": ["sales"]}
            assert "table" not in table_choices({"ref": "hr"}, {})
            assert metric_choices({}, {})["metric"] == ["orders_total", "public_orders"]
        assert [p["config"]["ref"] for p in source_presets(editor.context)] == ["sales"]
        assert [p["config"]["table"] for p in table_presets(editor.context)] == ["orders"]
    finally:
        editor.dispose()


# ---------------------------------------------------------------------------
# Data section and build_routes integration


def test_data_section_and_admin_views_filter_by_access(tmp_path):
    project, _, _, _ = _make_context(tmp_path=tmp_path)

    sales_app = FakeApp(user="sam", groups=("sales",), admin=False)
    page = DataPage(sales_app, project, path="", search="", admin=False)
    assert [s.ref for s in page.sources] == ["sales"]
    assert page.n_tables == 1
    assert [m.name for m in page.statuses] == ["orders_total", "public_orders"]
    # Direct URL drawer open for an unauthorized table is ignored.
    blocked_page = DataPage(sales_app, project, path="", search="?table=hr.salaries", admin=False)
    assert blocked_page.catalog.selected is None
    # Direct URL for an unauthorized metric shows not-found instead of the metric page.
    blocked_metric = DataPage(sales_app, project, path="metrics/payroll", search="", admin=False)
    assert not hasattr(blocked_metric, "metric")

    finance_app = FakeApp(user="bob", groups=("finance",), admin=False)
    catalog_view = CatalogSettings(project, editable=False, app=finance_app)
    assert [(r["ref"], r["table"]) for r in catalog_view.rows()] == [("hr", "employees"), ("sales", "orders")]
    metrics_view = MetricsSettings(project, editable=False, app=finance_app)
    assert [m.name for m in metrics_view.statuses()] == ["headcount", "public_orders"]

    admin_app = FakeApp(user="root", groups=("admins",), admin=True)
    admin_page = DataPage(admin_app, project, path="", search="", admin=True)
    assert [s.ref for s in admin_page.sources] == ["hr", "sales"]
    assert admin_page.n_tables == 3
    assert [m.name for m in admin_page.statuses] == ["headcount", "payroll", "orders_total", "public_orders"]


def test_build_routes_attaches_session_access(tmp_path):
    csv = tmp_path / "orders.csv"
    csv.write_text("order_id,amount\n1,50\n2,75\n", encoding="utf-8")
    routes = build_routes([str(csv)], db_path=tmp_path / "dash.db")
    app = routes["/"]()
    try:
        assert app._editor.context[IDENTITY_KEY] == app._identity
        assert callable(app._editor.context[ADMIN_KEY])
    finally:
        app._editor.dispose()
