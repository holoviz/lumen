"""Per-resource authorization enforcement in Lumen's FlowDash agent tools."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("panel_flowdash")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, FlowDashDeps
from panel_flowdash.auth import Identity
from pydantic_ai import ModelRetry

from lumen.agent import _graph
from lumen.agent.authoring import add_metric, add_sql
from lumen.agent.data import (
    describe_table, distinct_values, list_tables, run_sql,
)
from lumen.flowdash import (
    CATALOG_KEY, COMPONENTS, METRICS_KEY, SOURCES_KEY, ProjectCatalog,
    bind_identity,
)
from lumen.flowdash.metrics import ProjectMetrics
from lumen.flowdash.project import PROJECT_FILE, parse_project
from lumen.sources.duckdb import DuckDBSource

HR_TABLES = {
    "employees": "SELECT * FROM (VALUES (1, 'Ada', 'eng'), (2, 'Bo', 'sales')) AS t(id, name, dept)",
    "salaries": "SELECT * FROM (VALUES (1, 120000), (2, 95000)) AS t(id, salary)",
    "notes": "SELECT * FROM (VALUES (1, 'ok')) AS t(id, note)",
}

SALES_TABLES = {
    "orders": "SELECT * FROM (VALUES (101, 50), (102, 75)) AS t(order_id, amount)",
}

PROJECT_YAML = """\
catalog:
  hr:
    access:
      allow_groups: [hr, finance]
    tables:
      salaries:
        access:
          allow_groups: [hr]
      notes:
        assistant: false
metrics:
  headcount:
    description: Employee count.
    source: hr
    table: employees
    measures: {total: COUNT(*)}
  payroll:
    description: Total payroll.
    source: hr
    table: salaries
    measures: {total_pay: SUM(salary)}
  sales_rev:
    description: Sales revenue.
    source: sales
    table: orders
    access:
      allow_groups: [sales]
    measures: {revenue: SUM(amount)}
"""


def _session(identity: Identity | None, *, admin: bool = False):
    hr = DuckDBSource(uri=":memory:", tables=HR_TABLES)
    sales = DuckDBSource(uri=":memory:", tables=SALES_TABLES)
    context = {SOURCES_KEY: {"hr": hr, "sales": sales}}
    project = parse_project(PROJECT_YAML, PROJECT_FILE, context)
    context[CATALOG_KEY] = ProjectCatalog(project)
    context[METRICS_KEY] = ProjectMetrics(project)
    editor = FlowDash(components=COMPONENTS, notifications=False, history=False, context=context)
    bind_identity(editor, identity, admin=admin)
    ops = DashboardOps(editor)
    deps = FlowDashDeps(ops)
    session = SimpleNamespace(editor=editor, ops=ops, deps=deps, close=editor.dispose)
    return session, SimpleNamespace(deps=deps)


async def test_agent_list_tables_filters_sources_tables_and_metrics():
    finance = Identity("bob", groups=frozenset({"finance"}))
    session, ctx = _session(finance)
    try:
        overview = await list_tables(ctx)
        assert "headcount" in overview
        assert "payroll" not in overview
        assert "sales_rev" not in overview
        assert "hr" in overview and "sales" in overview
        assert "employees" in overview
        assert "salaries" not in overview
        assert "notes" not in overview

        # Single-source listing for hr must NOT leak the restricted salaries table
        # in the "turned off for the assistant" count (only notes counts as turned off).
        hr_listing = await list_tables(ctx, source="hr")
        assert "1 table(s) in source 'hr'" in hr_listing
        assert "employees" in hr_listing
        assert "salaries" not in hr_listing
        assert "1 more table is turned off for the assistant" in hr_listing
    finally:
        session.close()


async def test_agent_refuses_unauthorized_source_for_sales_user():
    sales_user = Identity("sam", groups=frozenset({"sales"}))
    session, ctx = _session(sales_user)
    try:
        listing = await list_tables(ctx)
        assert "orders" in listing
        assert "sales_rev" in listing
        assert "source 'hr'" not in listing and "source hr" not in listing
        assert "employees" not in listing and "headcount" not in listing

        with pytest.raises(ModelRetry, match="permission"):
            await list_tables(ctx, source="hr")
        with pytest.raises(ModelRetry, match="permission"):
            await run_sql(ctx, "SELECT * FROM employees", source="hr")
        with pytest.raises(ModelRetry, match="permission"):
            await add_sql(ctx, "hr", "emp_q", "SELECT * FROM employees")
    finally:
        session.close()


async def test_agent_refuses_unauthorized_table_queries_and_metrics():
    finance = Identity("bob", groups=frozenset({"finance"}))
    session, ctx = _session(finance)
    try:
        for text in (
            await describe_table(ctx, "salaries", source="hr"),
            await distinct_values(ctx, "salaries", "salary", source="hr"),
            await run_sql(ctx, "SELECT SUM(salary) FROM salaries", source="hr"),
        ):
            assert "permission" in text.lower()
            assert "120000" not in text

        with pytest.raises(ModelRetry, match="permission"):
            await add_sql(ctx, "hr", "pay_q", "SELECT SUM(salary) AS pay FROM salaries")
        assert "pay_q" not in session.ops.nodes()

        # Finance user CAN query employees and place headcount metric.
        assert "2 rows" in await run_sql(ctx, "SELECT * FROM employees", source="hr")
        placed = await add_sql(ctx, "hr", "emp_q", "SELECT * FROM employees")
        assert placed.startswith("Placed SQL node 'emp_q'")

        # Cannot place unauthorized metrics (payroll or sales_rev).
        assert list(_graph.metrics(session.ops)) == ["headcount"]
        with pytest.raises(ModelRetry, match="No metric 'payroll'"):
            await add_metric(ctx, "payroll", "pay_m", measures=["total_pay"])
        with pytest.raises(ModelRetry, match="No metric 'sales_rev'"):
            await add_metric(ctx, "sales_rev", "rev_m", measures=["revenue"])

        placed_metric = await add_metric(ctx, "headcount", "hc_m", measures=["total"])
        assert placed_metric.startswith("Placed metric node 'hc_m'")
    finally:
        session.close()


async def test_agent_allows_authorized_user_and_admin():
    hr_user = Identity("alice", groups=frozenset({"hr"}))
    session, ctx = _session(hr_user)
    try:
        assert "salary" in await describe_table(ctx, "salaries", source="hr")
        assert "1 rows" in await run_sql(ctx, "SELECT SUM(salary) FROM salaries", source="hr")
        placed = await add_metric(ctx, "payroll", "pay_m", measures=["total_pay"])
        assert placed.startswith("Placed metric node 'pay_m'")
    finally:
        session.close()

    admin_session, admin_ctx = _session(Identity("root", groups=frozenset({"admins"})), admin=True)
    try:
        assert set(_graph.metrics(admin_session.ops)) == {"headcount", "payroll", "sales_rev"}
        placed = await add_metric(admin_ctx, "sales_rev", "rev_m", measures=["revenue"])
        assert placed.startswith("Placed metric node 'rev_m'")
    finally:
        admin_session.close()
