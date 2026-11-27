from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

from panel_flowdash.agent import OperationError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession, assistant
from lumen.agent.authoring import (
    add_chart, add_metric, add_sql, add_table,
)
from lumen.agent.report import (
    HAS_REPORT_HOOKS, MAX_CONTEXT, REPORT_INSTRUCTIONS, report_context,
)
from lumen.flowdash import nodes
from lumen.flowdash.metrics import read_metrics
from lumen.flowdash.project import parse_project
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "orders": (
        "SELECT * FROM (VALUES (101, 'east', 30, 'paid'), (102, 'east', 20, 'paid'), "
        "(103, 'west', 40, 'paid'), (104, 'west', 15, 'cancelled')) AS t(order_id, region, amount, status)"
    ),
}
BY_REGION = "SELECT region, SUM(amount) AS total FROM orders WHERE status = 'paid' GROUP BY region ORDER BY total DESC"
BAR = {
    "mark": "bar",
    "encoding": {"x": {"field": "region", "type": "nominal"}, "y": {"field": "total", "type": "quantitative"}},
}
METRICS_YAML = """\
metrics:
  revenue:
    source: data
    table: orders
    description: Paid and cancelled order amounts
    measures:
      total_amount: "SUM(amount)"
    dimensions:
      status: status
"""

needs_hooks = pytest.mark.skipif(not HAS_REPORT_HOOKS, reason="panel-flowdash without report hooks")


@pytest.fixture
def session():
    model = FunctionModel(lambda m, i: ModelResponse([TextPart("")]))
    sess = AgentSession(
        DuckDBSource(uri=":memory:", tables=TABLES), model, metrics=read_metrics(parse_project(METRICS_YAML, {})),
    )
    yield sess
    sess.close()


@pytest.fixture
def ctx(session):
    return SimpleNamespace(deps=session.deps)


async def test_report_context_says_what_views_show_and_what_metrics_mean(ctx, session):
    await add_sql(ctx, "data", "paid_by_region", BY_REGION)
    add_chart(ctx, "paid_by_region", "paid_chart", BAR)
    add_table(ctx, "paid_by_region", "paid_table")
    await add_metric(ctx, "revenue", "rev_by_status", measures=["total_amount"], dimensions=["status"])

    text = report_context(session.ops)

    assert "- paid_chart (chart): bar x=region, y=total" in text
    assert "- paid_table (table): " in text
    assert (
        "- rev_by_status: metric 'revenue'. Paid and cancelled order amounts. total_amount = SUM(amount). by status"
    ) in text
    assert "paid_by_region" not in text


def test_report_context_adds_nothing_to_an_empty_dashboard(session):
    assert report_context(session.ops) is None


async def test_report_context_is_bounded(ctx, session):
    await add_sql(ctx, "data", "paid_by_region", BY_REGION)
    for i in range(40):
        add_chart(ctx, "paid_by_region", f"chart_{i}", BAR)

    text = report_context(session.ops)

    assert len(text) <= MAX_CONTEXT + 30
    assert text.endswith("more.")


@needs_hooks
async def test_view_hint_points_report_text_at_the_sql_node(ctx, session):
    await add_sql(ctx, "data", "paid_by_region", BY_REGION)
    add_chart(ctx, "paid_by_region", "paid_chart", BAR)

    hint = session.ops.node_placeholder_hint("paid_chart")

    assert hint.upstream == "pipeline"
    assert "SQL or metric node" in hint.note
    with pytest.raises(OperationError, match=r"paid_by_region\.data"):
        session.ops.edit_report([{"op": "insert", "block": {"text": "Top: {{ paid_chart.pipeline }}"}}])


@needs_hooks
async def test_get_report_appends_lumens_context(ctx, session):
    await add_sql(ctx, "data", "paid_by_region", BY_REGION)
    add_chart(ctx, "paid_by_region", "paid_chart", BAR)
    session.ops.report_context = report_context

    catalogue = session.ops.report_catalogue()

    assert "- paid_chart (chart): bar x=region, y=total" in catalogue


def test_view_gives_no_hint_without_report_hooks(monkeypatch):
    monkeypatch.setattr(nodes, "PlaceholderHint", None)
    assert nodes.LumenViewNode.__flowdash_placeholder_hint__(object()) is None


def test_layout_agent_gets_lumens_report_wording_with_report_hooks(monkeypatch):
    given = []
    monkeypatch.setattr(
        assistant, "create_flowdash_agent",
        lambda *args, **kwargs: given.append(kwargs["instructions"]) or assistant.Agent("test"),
    )
    assistant.create_agent("test")
    assistant.create_agent("test", dashboard_tools="layout")
    monkeypatch.setattr(assistant, "HAS_REPORT_HOOKS", False)
    assistant.create_agent("test", dashboard_tools="layout")
    data, layout, old = given
    assert REPORT_INSTRUCTIONS not in data and REPORT_INSTRUCTIONS not in old
    has_report = "edit_report" in (assistant.FLOWDASH_LAYOUT_TOOLS or ())
    assert (REPORT_INSTRUCTIONS in layout) == (HAS_REPORT_HOOKS and has_report)
