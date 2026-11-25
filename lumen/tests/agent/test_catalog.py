"""What Lumen's agent may use of a source when the app's catalog turns tables off."""
from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession
from lumen.agent.authoring import add_sql
from lumen.agent.data import (
    describe_table, distinct_values, list_tables, run_sql,
)
from lumen.flowdash import CATALOG_KEY, Catalog, ProjectCatalog
from lumen.flowdash.catalog_admin import set_assistant
from lumen.flowdash.project import PROJECT_FILE, load_project
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "orders": "SELECT * FROM (VALUES (101, 1, 30, 'paid'), (102, 2, 40, 'open')) AS t(order_id, customer_id, amount, status)",
    "customers": "SELECT * FROM (VALUES (1, 'Ada', 'east'), (2, 'Bo', 'west')) AS t(customer_id, customer, region)",
    "salaries": "SELECT * FROM (VALUES (1, 100000)) AS t(customer_id, salary)",
}


class Admin:
    def _still_admin(self):
        return True


@pytest.fixture
def session():
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(lambda m, i: ModelResponse([TextPart("")])))
    session.editor.context[CATALOG_KEY] = Catalog({"data": frozenset({"salaries"})})
    yield session
    session.close()


@pytest.fixture
def ctx(session):
    return SimpleNamespace(deps=session.deps)


async def test_list_tables_leaves_out_the_tables_turned_off(ctx):
    text = await list_tables(ctx)
    assert text.startswith("2 table(s) in source 'data' (duckdb SQL):")
    assert "orders" in text and "customers" in text and "salary" not in text
    assert "1 more table is turned off for the assistant" in text


async def test_list_tables_of_several_sources_leaves_them_out(ctx, session):
    session.ops.add_node("Lumen/source", {"ref": "data"}, name="other")
    text = await list_tables(ctx)
    assert "orders, customers" in text and "salaries" not in text


async def test_describing_or_reading_values_of_a_table_turned_off_is_refused(ctx):
    for text in (await describe_table(ctx, "salaries"), await distinct_values(ctx, "SALARIES", "salary")):
        assert "'salaries' of source 'data' is turned off for the assistant" in text
        assert "100000" not in text
    with pytest.raises(ModelRetry, match="Tables: orders, customers"):
        await describe_table(ctx, "salary")
    assert "amount" in await describe_table(ctx, "orders")


async def test_queries_reading_a_table_turned_off_are_refused(ctx, session):
    assert "turned off for the assistant" in await run_sql(ctx, "SELECT AVG(salary) FROM salaries")
    joined = "SELECT c.customer, s.salary FROM customers c JOIN salaries s USING (customer_id)"
    assert "turned off for the assistant" in await run_sql(ctx, joined)
    assert "2 rows" in await run_sql(ctx, "SELECT * FROM customers")
    with pytest.raises(ModelRetry, match="turned off for the assistant"):
        await add_sql(ctx, "data", "pay", joined)
    assert "pay" not in session.ops.nodes()


async def test_a_follow_up_query_cannot_reach_a_table_turned_off(ctx, session):
    await add_sql(ctx, "data", "paid", "SELECT * FROM orders WHERE status = 'paid'")
    with pytest.raises(ModelRetry, match="turned off for the assistant"):
        await add_sql(ctx, "paid", "pay", "SELECT p.amount, s.salary FROM paid p JOIN salaries s USING (customer_id)")
    text = await add_sql(ctx, "paid", "paid_total", "SELECT SUM(amount) AS total FROM paid")
    assert text.startswith("Placed SQL node 'paid_total'")


async def test_turning_a_table_on_in_the_admin_view_lets_the_agent_use_it(ctx, session, tmp_path):
    path = tmp_path / PROJECT_FILE
    path.write_text("catalog:\n  data:\n    tables:\n      salaries: {assistant: false}\n")
    project = load_project(path, dict(session.editor.context))
    session.editor.context[CATALOG_KEY] = ProjectCatalog(project)
    assert "salaries" not in await list_tables(ctx)
    assert set_assistant(project, "data", "salaries", True, app=Admin()) == []
    text = await list_tables(ctx)
    assert text.startswith("3 table(s)") and "salary" in text
    assert "1 rows" in await run_sql(ctx, "SELECT AVG(salary) FROM salaries")


async def test_a_metric_over_a_table_turned_off_is_left_out(ctx, session):
    from lumen.agent import _graph
    from lumen.flowdash import METRICS_KEY
    from lumen.flowdash.project import parse_project

    text = (
        "metrics:\n"
        "  revenue:\n    source: data\n    table: orders\n    measures: {total: SUM(amount)}\n"
        "  payroll:\n    source: data\n    sql: SELECT SUM(salary) AS total FROM salaries\n"
    )
    project = parse_project(text, PROJECT_FILE, dict(session.editor.context))
    session.editor.context[METRICS_KEY] = project.get("metrics")
    assert list(_graph.metrics(session.ops)) == ["revenue"]
    listing = await list_tables(ctx)
    assert "revenue" in listing and "payroll" not in listing
