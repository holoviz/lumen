"""Unit tests for access rule helpers, project writers and admin-only Data editors."""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel_flowdash import MemoryDashboardStore
from panel_flowdash.auth import Identity, Permission

from lumen.flowdash import (
    CATALOG_KEY, METRICS_KEY, SOURCE_ALIASES_KEY, SOURCES_KEY, ProjectCatalog,
)
from lumen.flowdash.access import (
    EffectiveAccess, InheritedRule, access_badge, access_to_dict,
    can_access_metric, can_access_source, can_access_table,
    describe_permission, effective_metric_access, effective_source_access,
    effective_table_access, lockout_reason, normalize_access,
)
from lumen.flowdash.access_editor import (
    AccessRuleEditor, effective_access_html, known_principals, parse_name_list,
)
from lumen.flowdash.catalog import (
    NOT_ADMIN_ACCESS, raw_section as raw_catalog, set_assistant,
    set_source_access, set_table_access,
)
from lumen.flowdash.data_section import DataPage
from lumen.flowdash.metrics import ProjectMetrics
from lumen.flowdash.metrics_edit import (
    NOT_ADMIN_ACCESS as METRIC_NOT_ADMIN_ACCESS, raw_section as raw_metrics,
    save_metric, set_metric_access,
)
from lumen.flowdash.project import PROJECT_FILE, load_project
from lumen.flowdash.uploads import UPLOADS_KEY, Uploads
from lumen.sources.duckdb import DuckDBSource

HR_TABLES = {
    "employees": "SELECT * FROM (VALUES (1, 'Ada', 'eng'), (2, 'Bo', 'sales')) AS t(id, name, dept)",
    "salaries": "SELECT * FROM (VALUES (1, 120000), (2, 95000)) AS t(id, salary)",
}

BASE_YAML = """\
general:
  admins: [ana]
permissions:
  admin_groups: [admins]
  people:
    alice:
      groups: [hr, finance]
    bob:
      groups: [finance]
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
  joined_report:
    description: Employees with salary.
    source: hr
    sql: |
      SELECT e.dept, SUM(s.salary) AS total_pay
      FROM employees e
      JOIN salaries s ON e.id = s.id
      GROUP BY e.dept
  broken:
    description: Missing table metric.
    source: hr
    table: missing_table
    measures:
      n: COUNT(*)
"""


class FakeSession:
    def __init__(
        self,
        project,
        *,
        user: str = "ana",
        groups: tuple[str, ...] = ("admins",),
        admin: bool = True,
        oauth_user: str | None = None,
        system_user: str | None = None,
        identity: Identity | None = ...,
    ):
        self.project = project
        self.is_admin = self.admin = admin
        self.store = MemoryDashboardStore()
        self._identity = (
            Identity(user=user, oauth_user=oauth_user, system_user=system_user, groups=frozenset(groups))
            if identity is ...
            else identity
        )
        self.notified: list[tuple[str, str]] = []

    def still_admin(self) -> bool:
        return self.admin

    def _still_admin(self) -> bool:
        return self.admin

    def _authorize_admin(self, *, fresh: bool = False) -> bool:
        return self.admin

    def _settings_project(self):
        return self.project

    def _default_allow(self):
        return False

    def _known_users(self):
        return ["alice", "ana", "bob"]

    def _known_groups(self):
        return ["admins", "finance", "hr"]

    def _notify(self, severity, message, duration=3000):
        self.notified.append((severity, message))


def _make_project(tmp_path, yaml_text: str = BASE_YAML):
    hr = DuckDBSource(uri=":memory:", tables=HR_TABLES, name="hr")
    sources = {"hr": hr}
    context = {SOURCES_KEY: sources, SOURCE_ALIASES_KEY: {"people": "hr"}}
    path = tmp_path / PROJECT_FILE
    path.write_text(yaml_text, encoding="utf-8")
    project = load_project(path, context)
    project.context[CATALOG_KEY] = ProjectCatalog(project)
    project.context[METRICS_KEY] = ProjectMetrics(project)
    project.context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", sources, project=project)
    return project, hr


def _text(view) -> str:
    return "".join(str(o.object) for o in view.__panel__().select() if isinstance(getattr(o, "object", None), str))


# ---------------------------------------------------------------------------
# Serialization, summaries, lockout checks and inheritance


def test_access_helpers_and_lockout_checks():
    perm = Permission.from_spec(allow_groups=["hr", "finance"], deny_users=["contractor"])
    assert access_to_dict(perm) == {"allow_groups": ["finance", "hr"], "deny_users": ["contractor"]}
    assert access_to_dict(Permission()) == {}
    assert describe_permission(Permission()) == "Open to everyone"
    assert describe_permission(perm) == "Allow groups finance, hr; deny user contractor"
    assert describe_permission(Permission.from_spec(deny_groups=["guests"])) == "Deny group guests"

    norm, problems = normalize_access({"allow_users": [" alice ", "bob"], "deny_groups": "guests"})
    assert problems == []
    assert norm == Permission.from_spec(allow_users=["alice", "bob"], deny_groups=["guests"])

    _, bad = normalize_access({"allow_roles": ["hr"]})
    assert bad and "Unknown key 'allow_roles'" in bad[0].message

    admin_id = Identity(user="ana", oauth_user="ana@corp", groups=frozenset({"admins", "ops"}))
    # Admins bypass allowlists, so restricting to another group never locks them out.
    assert lockout_reason(Permission.from_spec(allow_groups=["hr"]), admin_id) is None
    # Clearing a rule never locks anyone out.
    assert lockout_reason(Permission(), None) is None
    # Explicit user deny (primary or alias name) and group deny refuse to save.
    assert "denies your user ('ana')" in (lockout_reason(Permission.from_spec(deny_users=["ana"]), admin_id) or "")
    assert "denies your user ('ana@corp')" in (lockout_reason(Permission.from_spec(deny_users=["ana@corp"]), admin_id) or "")
    assert "denies your group ('admins')" in (lockout_reason(Permission.from_spec(deny_groups=["admins"]), admin_id) or "")
    # Non-empty rule with no session identity fails closed.
    assert "no identity" in (lockout_reason(Permission.from_spec(allow_groups=["hr"]), None) or "")

    assert parse_name_list(" hr, finance ,\nhr , ") == ["hr", "finance"]


def test_effective_access_and_inheritance(tmp_path):
    project, hr = _make_project(tmp_path)
    ctx = project.context

    src_eff = effective_source_access("hr", ctx)
    assert src_eff.has_own and not src_eff.has_inherited
    assert access_badge(src_eff) == ("Restricted", "warning")

    emp_eff = effective_table_access("hr", hr, "employees", ctx)
    assert not emp_eff.has_own and emp_eff.has_inherited
    assert [r.label for r in emp_eff.inherited] == ["Source 'hr'"]
    assert access_badge(emp_eff) == ("Inherited", "primary")

    sal_eff = effective_table_access("hr", hr, "salaries", ctx)
    assert sal_eff.has_own and sal_eff.has_inherited
    assert [r.label for r in sal_eff.inherited] == ["Source 'hr'"]
    assert access_badge(sal_eff) == ("Restricted (own + inherited)", "warning")

    hr_user = Identity("alice", groups=frozenset({"hr"}))
    fin_user = Identity("bob", groups=frozenset({"finance"}))
    assert emp_eff.allows(fin_user) is True
    assert sal_eff.allows(fin_user) is False
    assert sal_eff.allows(hr_user) is True

    # Metrics inherit from source and every table they read (including SQL joins).
    metrics = project.get("metrics")
    hc_eff = effective_metric_access(metrics["headcount"], ctx)
    assert [r.label for r in hc_eff.inherited] == ["Source 'hr'"]

    pay_eff = effective_metric_access(metrics["payroll"], ctx)
    assert [r.label for r in pay_eff.inherited] == ["Source 'hr'", "Table hr.salaries"]

    join_eff = effective_metric_access(metrics["joined_report"], ctx)
    assert [r.label for r in join_eff.inherited] == ["Source 'hr'", "Table hr.salaries"]
    assert join_eff.allows(fin_user) is False
    assert join_eff.allows(hr_user) is True

    html = effective_access_html(pay_eff)
    assert "Inherited from Source &#x27;hr&#x27;" in html
    assert "Inherited from Table hr.salaries" in html
    assert "No own rule (inherits above)" in html


# ---------------------------------------------------------------------------
# Project writers: catalog.py & metrics_edit.py


def test_set_source_and_table_access_writers(tmp_path):
    project, hr = _make_project(tmp_path)
    admin = FakeSession(project, user="ana", groups=("admins",), admin=True)
    viewer = FakeSession(project, user="bob", groups=("finance",), admin=False)

    # Non-admin refused.
    assert set_source_access(project, "hr", {"allow_groups": ["sales"]}, app=viewer)[0].message == NOT_ADMIN_ACCESS
    assert set_table_access(project, "hr", "employees", {"allow_groups": ["sales"]}, app=viewer)[0].message == NOT_ADMIN_ACCESS

    # Self-lockout refused (by user, group, or missing identity).
    assert "lock you out" in set_source_access(project, "hr", {"deny_users": ["ana"]}, app=admin)[0].message
    assert "lock you out" in set_source_access(project, "hr", {"deny_groups": ["admins"]}, app=admin)[0].message
    no_id = FakeSession(project, admin=True, identity=None)
    assert "lock you out" in set_source_access(project, "hr", {"allow_groups": ["hr"]}, app=no_id)[0].message
    assert "lock you out" in set_table_access(project, "hr", "employees", {"deny_users": ["ana"]}, app=admin)[0].message

    # Live admin re-check: revoking admin rights refuses even on an admin session.
    admin.admin = False
    assert set_source_access(project, "hr", {"allow_groups": ["hr"]}, app=admin)[0].message == NOT_ADMIN_ACCESS
    assert set_table_access(project, "hr", "employees", {"allow_groups": ["hr"]}, app=admin)[0].message == NOT_ADMIN_ACCESS
    admin.admin = True

    # Valid save updates flowdash.yaml and reloads ProjectCatalog live.
    assert set_source_access(project, "hr", {"allow_groups": ["ops", "hr"], "deny_users": ["guest"]}, app=admin) == []
    assert raw_catalog(project)["hr"]["access"] == {"allow_groups": ["hr", "ops"], "deny_users": ["guest"]}
    assert not can_access_source("hr", project.context, identity=Identity("bob", groups=frozenset({"finance"})))
    assert can_access_source("hr", project.context, identity=Identity("sam", groups=frozenset({"ops"})))

    # Saving table access preserves assistant: false and source access.
    assert set_table_access(project, "hr", "salaries", {"allow_users": ["alice"]}, app=admin) == []
    assert raw_catalog(project)["hr"]["tables"]["salaries"] == {
        "assistant": False,
        "access": {"allow_users": ["alice"]},
    }
    assert can_access_table(hr, "salaries", project.context, identity=Identity("alice", groups=frozenset({"hr"})))
    assert not can_access_table(hr, "salaries", project.context, identity=Identity("carol", groups=frozenset({"hr"})))

    # Clearing table access keeps assistant: false; clearing source access keeps tables.
    assert set_table_access(project, "hr", "salaries", Permission(), app=admin) == []
    assert raw_catalog(project)["hr"]["tables"]["salaries"] == {"assistant": False}
    assert set_source_access(project, "hr", None, app=admin) == []
    assert "access" not in raw_catalog(project)["hr"]
    assert raw_catalog(project)["hr"]["tables"]["salaries"] == {"assistant": False}

    # Turning assistant back on prunes the empty catalog section completely.
    assert set_assistant(project, "hr", "salaries", True, app=admin) == []
    assert raw_catalog(project) == {}


def test_set_metric_access_and_save_metric_lockout(tmp_path):
    project, _ = _make_project(tmp_path)
    admin = FakeSession(project, user="ana", groups=("admins",), admin=True)
    viewer = FakeSession(project, user="bob", groups=("finance",), admin=False)

    assert set_metric_access(project, "headcount", {"allow_groups": ["hr"]}, app=viewer)[0].message == METRIC_NOT_ADMIN_ACCESS
    assert "lock you out" in set_metric_access(project, "headcount", {"deny_users": ["ana"]}, app=admin)[0].message
    assert "lock you out" in set_metric_access(project, "headcount", {"deny_groups": ["admins"]}, app=admin)[0].message

    # save_metric also refuses self-lockout in YAML edits.
    spec = dict(raw_metrics(project)["headcount"])
    spec["access"] = {"deny_users": ["ana"]}
    assert "lock you out" in save_metric(project, "headcount", spec, app=admin, original="headcount")[0].message

    # Live admin revocation refuses save.
    admin.admin = False
    assert set_metric_access(project, "headcount", {"allow_groups": ["hr"]}, app=admin)[0].message == METRIC_NOT_ADMIN_ACCESS
    admin.admin = True

    # Valid save updates flowdash.yaml and reloads ProjectMetrics.
    assert set_metric_access(project, "headcount", {"allow_groups": ["hr"], "deny_users": ["temp"]}, app=admin, base=None) == []
    assert raw_metrics(project)["headcount"]["access"] == {"allow_groups": ["hr"], "deny_users": ["temp"]}
    assert not can_access_metric(project.get("metrics")["headcount"], project.context, identity=Identity("bob", groups=frozenset({"finance"})))
    assert can_access_metric(project.get("metrics")["headcount"], project.context, identity=Identity("alice", groups=frozenset({"hr"})))

    # Concurrent edit conflict via base is caught.
    conflict = set_metric_access(project, "headcount", {"allow_groups": ["finance"]}, app=admin, base=None)
    assert conflict and "changed since you opened" in conflict[0].message

    # Works on a failing metric too, and clearing removes the access key.
    assert set_metric_access(project, "broken", {"allow_groups": ["hr"]}, app=admin) == []
    assert raw_metrics(project)["broken"]["access"] == {"allow_groups": ["hr"]}
    assert set_metric_access(project, "headcount", Permission(), app=admin) == []
    assert "access" not in raw_metrics(project)["headcount"]


# ---------------------------------------------------------------------------
# UI components (headless): DataPage, CatalogSettings drawer, MetricPage


def test_access_rule_editor_live_preview_and_known_principals(tmp_path):
    project, _ = _make_project(tmp_path)
    admin = FakeSession(project, user="ana", groups=("admins",), admin=True)
    users, groups = known_principals(admin, project)
    assert {"alice", "ana", "bob"} <= set(users)
    assert {"admins", "finance", "hr"} <= set(groups)

    saved: list[Permission] = []
    editor = AccessRuleEditor(app=admin, project=project, on_save=lambda perm: (saved.append(perm), [])[1])
    inherited = (InheritedRule("Source 'hr'", Permission.from_spec(allow_groups=["hr", "finance"])),)
    editor.load(EffectiveAccess(own=Permission(), inherited=inherited))
    assert "Inherited from Source &#x27;hr&#x27;" in editor._effective_view.object
    assert not editor._lockout_warn.visible

    # Typing own username in deny_users triggers immediate lockout warning.
    editor._deny_users.value = "ana"
    assert editor._lockout_warn.visible and "lock you out" in editor._lockout_warn.object

    # Clear removes warning and resets all fields.
    editor.clear()
    assert not editor._lockout_warn.visible
    assert editor.permission().is_empty

    editor._allow_groups.value = "hr"
    assert "Allow group hr" in editor._effective_view.object
    assert editor.save()
    assert saved == [Permission.from_spec(allow_groups=["hr"])]


def test_source_card_and_table_drawer_access_editors_admin_vs_viewer(tmp_path):
    project, _ = _make_project(tmp_path)

    # Admin sees source card access button and can edit via dialog.
    admin_session = FakeSession(project, user="ana", groups=("admins",), admin=True)
    admin_view = DataPage(admin_session, project, admin=True, navigate=lambda _url: None)
    assert any("lumen-data-source-access" in getattr(o, "css_classes", []) for o in admin_view.__panel__().select())
    assert "Access: Allow groups finance, hr; deny user contractor" in _text(admin_view)

    assert admin_view.edit_source_access("hr")
    assert admin_view._access_dialog.is_open
    # Refuses lockout and stays open with error alert.
    admin_view._access_dialog.editor._deny_users.value = "ana"
    assert not admin_view._access_dialog.editor.save()
    assert admin_view._access_dialog.is_open
    assert admin_view._access_dialog.editor._alert.visible

    # Valid edit closes dialog and updates flowdash.yaml & card text.
    admin_view._access_dialog.editor._deny_users.value = "guest"
    admin_view._access_dialog.editor._allow_groups.value = "hr"
    assert admin_view._access_dialog.editor.save()
    assert not admin_view._access_dialog.is_open
    assert raw_catalog(project)["hr"]["access"] == {"allow_groups": ["hr"], "deny_users": ["guest"]}
    assert "Access: Allow group hr; deny user guest" in _text(admin_view)

    # Admin opens table drawer and sees inherited source rule + table rule.
    catalog = admin_view.catalog
    catalog.open_table("hr", "employees")
    assert catalog._access_row.visible and catalog._access_btn.visible
    assert "Inherited from Source &#x27;hr&#x27;" in catalog._access_summary.object

    assert catalog.open_access_editor()
    assert catalog._access_panel.visible
    # Live admin revocation refuses save from the drawer too.
    admin_session.admin = False
    catalog._access_editor._allow_users.value = "alice"
    assert not catalog._access_editor.save()
    assert catalog._access_panel.visible and catalog._access_editor._alert.visible
    admin_session.admin = True

    assert catalog._access_editor.save()
    assert not catalog._access_panel.visible
    assert raw_catalog(project)["hr"]["tables"]["employees"]["access"] == {"allow_users": ["alice"]}
    assert "Allow user alice" in catalog._access_summary.object

    # Viewer sees nothing new on source cards or in table drawer.
    viewer_session = FakeSession(project, user="alice", groups=("hr",), admin=False)
    viewer_view = DataPage(viewer_session, project, admin=False, navigate=lambda _url: None)
    assert viewer_view._access_dialog is None
    assert not viewer_view.edit_source_access("hr")
    assert not any("lumen-data-source-access" in getattr(o, "css_classes", []) for o in viewer_view.__panel__().select())
    assert "Access:" not in _text(viewer_view)

    viewer_view.catalog.open_table("hr", "employees")
    assert not viewer_view.catalog._access_row.visible
    assert not viewer_view.catalog._access_btn.visible
    assert not viewer_view.catalog._access_panel.visible
    assert not viewer_view.catalog.open_access_editor()


def test_metric_page_access_editor_admin_vs_viewer(tmp_path):
    project, _ = _make_project(tmp_path)

    # Admin opens metric page and sees inherited source + table rules.
    admin_session = FakeSession(project, user="ana", groups=("admins",), admin=True)
    admin_page = DataPage(admin_session, project, "metrics/payroll", admin=True, navigate=lambda _url: None)
    metric_view = admin_page.metric
    assert metric_view._access_summary is not None
    assert "Inherited from Source &#x27;hr&#x27;" in metric_view._access_summary.object
    assert "Inherited from Table hr.salaries" in metric_view._access_summary.object

    assert metric_view.open_access_editor()
    assert metric_view._access_panel.visible

    # Refuses lockout.
    metric_view._access_editor._deny_groups.value = "admins"
    assert not metric_view._access_editor.save()
    assert metric_view._access_panel.visible and metric_view._access_editor._alert.visible

    # Refuses when live admin rights are revoked.
    metric_view._access_editor._deny_groups.value = ""
    metric_view._access_editor._deny_users.value = "intern"
    admin_session.admin = False
    assert not metric_view._access_editor.save()
    assert metric_view._access_panel.visible and metric_view._access_editor._alert.visible
    admin_session.admin = True

    # Saves cleanly and updates summary in place.
    assert metric_view._access_editor.save()
    assert not metric_view._access_panel.visible
    assert raw_metrics(project)["payroll"]["access"] == {"deny_users": ["intern"]}
    assert "Deny user intern" in metric_view._access_summary.object
    assert "Inherited from Table hr.salaries" in metric_view._access_summary.object

    # Viewer sees nothing new on metric page.
    viewer_session = FakeSession(project, user="alice", groups=("hr",), admin=False)
    viewer_page = DataPage(viewer_session, project, "metrics/payroll", admin=False, navigate=lambda _url: None)
    viewer_metric = viewer_page.metric
    assert viewer_metric._access_summary is None
    assert viewer_metric._access_btn is None
    assert viewer_metric._access_editor is None
    assert viewer_metric._access_panel is None
    assert not viewer_metric.open_access_editor()
    assert not any("lumen-metric-access" in c for o in viewer_metric.__panel__().select() for c in getattr(o, "css_classes", []))
