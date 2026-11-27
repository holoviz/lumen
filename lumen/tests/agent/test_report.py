from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, FlowDashDeps
from pydantic_ai import ModelRetry
from pydantic_ai.messages import (
    ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from lumen.agent import (
    AgentSession, create_agent, create_explore_agent, report_toolset,
)
from lumen.agent.assistant import REPORT_INSTRUCTIONS
from lumen.agent.authoring import (
    add_chart, add_metric, add_sql, add_table,
)
from lumen.agent.controls import add_kpi
from lumen.agent.report import (
    HAS_REPORTS, PROSE_LIMIT, add_report_prose, check_report_text,
    describe_report, edit_report_prose, write_report,
)
from lumen.flowdash import COMPONENTS, bind_source
from lumen.flowdash.metrics import read_metrics
from lumen.flowdash.project import parse_project
from lumen.sources.duckdb import DuckDBSource

pytestmark = pytest.mark.skipif(not HAS_REPORTS, reason="panel-flowdash without report ops")

TABLES = {
    "customers": ("SELECT * FROM (VALUES (1, 'Ada', 'east'), (2, 'Bo', 'west'), (3, 'Cy', 'east')) AS t(customer_id, customer, region)"),
    "orders": (
        "SELECT * FROM (VALUES (101, 1, 30, 'paid'), (102, 1, 20, 'paid'), "
        "(103, 2, 40, 'paid'), (104, 3, 15, 'cancelled')) AS t(order_id, customer_id, amount, status)"
    ),
}
BY_REGION = (
    "SELECT c.region, SUM(o.amount) AS total FROM orders o JOIN customers c "
    "ON o.customer_id = c.customer_id WHERE o.status = 'paid' GROUP BY c.region ORDER BY total DESC"
)
TOTAL_PAID = "SELECT SUM(amount) AS paid_total FROM orders WHERE status = 'paid'"
BAR = {
    "mark": "bar",
    "encoding": {
        "x": {"field": "region", "type": "nominal"},
        "y": {"field": "total", "type": "quantitative"},
    },
}
METRICS_YAML = """\
metrics:
  revenue:
    source: data
    table: orders
    measures:
      total_amount: "SUM(amount)"
    dimensions:
      status: status
"""


def _quiet_model():
    return FunctionModel(lambda m, i: ModelResponse([TextPart("")]))


def _script(*responses):
    requests = []

    def respond(messages, info):
        requests.append(messages[-1])
        return ModelResponse(parts=list(responses[len(requests) - 1]))

    return FunctionModel(respond, model_name="scripted"), requests


def _tools(agent):
    return {tool for toolset in agent.toolsets for tool in getattr(toolset, "tools", {})}


@pytest.fixture
def session():
    sess = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), _quiet_model())
    yield sess
    sess.close()


@pytest.fixture
def ctx(session):
    return SimpleNamespace(deps=session.deps)


async def _seed_dashboard(ctx):
    await add_sql(ctx, "data", "paid_by_region", BY_REGION)
    await add_sql(ctx, "data", "paid_total", TOTAL_PAID)
    add_kpi(ctx, "paid_total", "paid_kpi", label="Paid total", unit="$")
    add_chart(ctx, "paid_by_region", "paid_chart", BAR)
    add_table(ctx, "paid_by_region", "paid_table")


async def test_describe_report_lists_blocks_and_grounded_lumen_catalog(ctx):
    await _seed_dashboard(ctx)
    first = describe_report(ctx)
    second = describe_report(ctx)

    # Deterministic content-hashed fencing so replays stay identical.
    assert first == second
    assert "derived from dashboard tiles and groups" in first
    assert "[0] node: paid_kpi" in first
    assert "[1] node: paid_chart" in first
    assert "[2] node: paid_table" in first
    # Grounded catalog connects visuals to upstream data nodes and gives valid placeholders.
    assert "paid_chart (chart: bar x=region, y=total; data from 'paid_by_region')" in first
    assert "paid_kpi (Widgets/KPI: KPI 'Paid total'" in first
    assert "data from 'paid_total'" in first
    assert "{{ paid_total.value }}" in first
    assert "{{ paid_by_region.data.region }}" in first
    assert "{{ paid_by_region.data.total }}" in first
    assert "{{#each paid_by_region.data limit=5}}...{{/each}}" in first
    assert "east" in first and "50" in first


async def test_describe_report_flags_unaddressable_column_names(ctx):
    await add_sql(ctx, "data", "spaced", 'SELECT region AS "Sales Region", SUM(amount) AS total FROM orders JOIN customers USING (customer_id) GROUP BY 1')
    out = describe_report(ctx)
    assert "unaddressable columns" in out and "'Sales Region'" in out
    assert "{{ spaced.data.total }}" in out


async def test_add_report_prose_inserts_with_live_placeholders_and_anchors(ctx, session):
    await _seed_dashboard(ctx)
    rev = session.ops.report_revision()

    top = add_report_prose(
        ctx,
        text="Total paid revenue reached **${{ paid_total.value:,.0f }}**, led by {{ paid_by_region.data.region }} at ${{ paid_by_region.data.total:,.0f }}.",
        heading="Executive summary",
        level=2,
        position="top",
        title="Paid Orders Review",
        revision=rev,
    )
    assert "Inserted heading and prose at blocks 0-1 and titled the report 'Paid Orders Review'." in top
    assert "Total paid revenue reached **$90**, led by east at $50." in top

    before_chart = add_report_prose(
        ctx,
        text="Regional breakdown:\n{{#each paid_by_region.data limit=5}}\n- {{ .region }}: ${{ .total:,.0f }}\n{{/each}}",
        before="paid_chart",
    )
    assert "- east: $50" in before_chart and "- west: $40" in before_chart

    after_table = add_report_prose(ctx, text="End of report.", after="paid_table")
    assert "End of report." in after_table

    report = session.ops.report()
    assert report["title"] == "Paid Orders Review"
    kinds = [next(iter(b)) for b in report["blocks"]]
    assert kinds == ["heading", "text", "node", "text", "node", "node", "text"]
    assert report["blocks"][3]["text"].startswith("Regional breakdown:")
    assert report["blocks"][4]["node"] == "paid_chart"
    assert report["blocks"][6]["text"] == "End of report."


async def test_add_report_prose_supports_standalone_heading_and_index_anchor(ctx, session):
    await _seed_dashboard(ctx)
    add_report_prose(ctx, heading="Overview", level=1, position="top")
    add_report_prose(ctx, text="Short intro.", after=0)

    blocks = session.ops.report()["blocks"]
    assert blocks[0] == {"heading": "Overview", "level": 1}
    assert blocks[1] == {"text": "Short intro."}


async def test_add_report_prose_hints_when_anchored_on_hidden_sql_node(ctx):
    await _seed_dashboard(ctx)
    with pytest.raises(ModelRetry, match=r"'paid_by_region' is not a block in the report.*'paid_chart'"):
        add_report_prose(ctx, text="Breakdown.", before="paid_by_region")


async def test_placeholder_validation_catches_lumen_mistakes_before_saving(ctx, session):
    await _seed_dashboard(ctx)
    before = session.ops.report()

    # Chart node has no data output: hint points directly to upstream SQL node.
    with pytest.raises(ModelRetry, match=r"'paid_chart' is a chart node.*upstream node 'paid_by_region'"):
        add_report_prose(ctx, text="Top region: {{ paid_chart.data.region }}")

    # KPI card has no value output: hint points directly to upstream SQL node.
    with pytest.raises(ModelRetry, match=r"'paid_kpi'.*upstream node 'paid_total'.*\{\{ paid_total\.value \}\}"):
        add_report_prose(ctx, text="Total: {{ paid_kpi.value }}")

    # Pipeline port is not a plain table/value.
    with pytest.raises(ModelRetry, match=r"'pipeline' is not a plain value or table"):
        add_report_prose(ctx, text="Pipeline: {{ paid_by_region.pipeline }}")

    # Bare {{ value }} copied from Text/Markdown tiles.
    with pytest.raises(ModelRetry, match=r"must name both a node and an output port"):
        add_report_prose(ctx, text="Total: {{ value }}")

    # Unknown node, unknown column, out-of-range row, unclosed #each.
    with pytest.raises(ModelRetry, match=r"names no node on the dashboard"):
        add_report_prose(ctx, text="Total: {{ missing.value }}")
    with pytest.raises(ModelRetry, match=r"no column revenue"):
        add_report_prose(ctx, text="Total: {{ paid_by_region.data.revenue }}")
    with pytest.raises(ModelRetry, match=r"no row 9"):
        add_report_prose(ctx, text="Row 10: {{ paid_by_region.data.region[9] }}")
    with pytest.raises(ModelRetry, match=r"has no '\{\{/each\}\}'"):
        add_report_prose(ctx, text="{{#each paid_by_region.data}}- {{ .region }}")

    # Nothing was mutated by any failed validation.
    assert session.ops.report() == before


async def test_code_spans_with_braces_are_allowed_in_report_prose(ctx):
    await _seed_dashboard(ctx)
    check_report_text(ctx.deps.ops, "Use `{{ value }}` in Markdown tiles; report prose uses {{ paid_total.value }}.")


async def test_placeholder_validation_works_with_metric_nodes():
    metrics = read_metrics(parse_project(METRICS_YAML, {}))
    sess = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), _quiet_model(), metrics=metrics)
    try:
        ctx = SimpleNamespace(deps=sess.deps)
        await add_metric(ctx, "revenue", "rev_by_status", measures=["total_amount"], dimensions=["status"], order_by=["-total_amount"])
        result = add_report_prose(ctx, text="Top status is {{ rev_by_status.data.status }} (${{ rev_by_status.data.total_amount:,.0f }}).")
        assert "Top status is paid ($90)." in result
    finally:
        sess.close()


async def test_edit_report_prose_updates_text_heading_caption_title_and_removes(ctx, session):
    await _seed_dashboard(ctx)
    add_report_prose(ctx, text="Draft intro.", heading="Intro", level=2, position="top", title="Draft")
    rev = session.ops.report_revision()

    # Rewrite prose in place.
    edited = edit_report_prose(ctx, index=1, text="Paid orders total ${{ paid_total.value:,.0f }}.", revision=rev)
    assert "Paid orders total $90." in edited

    # Stale revision is refused cleanly.
    with pytest.raises(ModelRetry, match="report changed since you last read it"):
        edit_report_prose(ctx, index=1, text="Stale.", revision=rev)

    # Update heading text and level.
    edit_report_prose(ctx, index=0, heading="Summary", level=1)
    assert session.ops.report()["blocks"][0] == {"heading": "Summary", "level": 1}

    # Add and clear caption on a node block.
    edit_report_prose(ctx, index=3, caption="Paid revenue by region")
    assert session.ops.report()["blocks"][3]["caption"] == "Paid revenue by region"
    edit_report_prose(ctx, index=3, caption="")
    assert "caption" not in session.ops.report()["blocks"][3]

    # Rename title alone and reset to dashboard title.
    edit_report_prose(ctx, title="Final Review")
    assert session.ops.report()["title"] == "Final Review"
    edit_report_prose(ctx, title="")
    assert "title" not in session.ops.report()

    # Cannot remove a node block via edit_report_prose.
    with pytest.raises(ModelRetry, match="only removes text and heading blocks"):
        edit_report_prose(ctx, index=3, remove=True)

    # Can remove a text block.
    edit_report_prose(ctx, index=1, remove=True)
    assert session.ops.report()["blocks"][0] == {"heading": "Summary", "level": 1}
    assert session.ops.report()["blocks"][1]["node"] == "paid_kpi"


async def test_edit_report_prose_rejects_mismatched_block_kinds_and_bad_args(ctx):
    await _seed_dashboard(ctx)
    add_report_prose(ctx, text="Intro.", heading="Header", position="top")

    with pytest.raises(ModelRetry, match="not a text block"):
        edit_report_prose(ctx, index=0, text="Wrong.")
    with pytest.raises(ModelRetry, match="not a heading block"):
        edit_report_prose(ctx, index=1, heading="Wrong")
    with pytest.raises(ModelRetry, match="only node blocks take a caption"):
        edit_report_prose(ctx, index=1, caption="Nope")
    with pytest.raises(ModelRetry, match="Pass the block index"):
        edit_report_prose(ctx)
    with pytest.raises(ModelRetry, match=f"at most {PROSE_LIMIT}"):
        edit_report_prose(ctx, index=1, text="x" * (PROSE_LIMIT + 1))


async def test_write_report_builds_interleaved_narrative_and_preserves_unreferenced_views(ctx, session):
    await _seed_dashboard(ctx)

    # Omit paid_table on purpose: keep_views=True appends it rather than dropping a visual.
    out = write_report(
        ctx,
        title="Regional Paid Orders",
        blocks=[
            {"heading": "Overview", "level": 1},
            {"text": "Paid orders total ${{ paid_total.value:,.0f }} across two regions."},
            {"node": "paid_kpi"},
            {"text": "The chart below compares regional totals, led by {{ paid_by_region.data.region }}."},
            {"node": "paid_chart", "caption": "Paid revenue by region"},
        ],
    )
    assert "Kept unreferenced visual(s) at the end: paid_table" in out
    assert "Paid orders total $90 across two regions." in out
    blocks = session.ops.report()["blocks"]
    assert [b.get("node") for b in blocks if "node" in b] == ["paid_kpi", "paid_chart", "paid_table"]

    # Explicit keep_views=False omits unreferenced views.
    write_report(
        ctx,
        blocks=[
            {"text": "Summary only: ${{ paid_total.value:,.0f }}."},
            {"node": "paid_chart"},
        ],
        keep_views=False,
    )
    assert [b.get("node") for b in session.ops.report()["blocks"] if "node" in b] == ["paid_chart"]


async def test_write_report_rejects_invalid_blocks_without_mutating(ctx, session):
    await _seed_dashboard(ctx)
    before = session.ops.report()

    with pytest.raises(ModelRetry, match="no node 'nope'"):
        write_report(ctx, blocks=[{"node": "nope"}])
    with pytest.raises(ModelRetry, match="upstream node 'paid_by_region'"):
        write_report(ctx, blocks=[{"text": "{{ paid_chart.data.total }}"}, {"node": "paid_chart"}])

    assert session.ops.report() == before


async def test_add_report_prose_anchors_on_node_inside_group(ctx, session):
    await _seed_dashboard(ctx)
    session.ops.group_nodes(["paid_chart", "paid_table"], label="Regional views", name="regional")
    add_report_prose(ctx, text="Regional section intro.", before="paid_chart")
    blocks = session.ops.report()["blocks"]
    group_idx = next(i for i, b in enumerate(blocks) if b.get("group") == "regional")
    assert blocks[group_idx - 1] == {"text": "Regional section intro."}


async def test_report_edits_are_refused_in_explore(session):
    flowdash_agent = pytest.importorskip("panel_flowdash.agent")
    if not hasattr(flowdash_agent, "ExplorationOps"):
        pytest.skip("panel-flowdash without ExplorationOps")
    session.editor.new_dashboard("Sales")
    explored = flowdash_agent.ExplorationOps(session.ops)
    try:
        explore_ctx = SimpleNamespace(deps=FlowDashDeps(explored))
        with pytest.raises(PermissionError, match="cannot edit the dashboard's report"):
            add_report_prose(explore_ctx, text="Not allowed.")
        with pytest.raises(PermissionError, match="cannot edit the dashboard's report"):
            edit_report_prose(explore_ctx, title="Nope")
        with pytest.raises(PermissionError, match="cannot edit the dashboard's report"):
            write_report(explore_ctx, blocks=[{"text": "Nope"}])
    finally:
        explored.dispose()


def test_report_toolset_wiring_in_create_agent():
    report_names = set(report_toolset().tools)
    assert report_names == {"describe_report", "add_report_prose", "edit_report_prose", "write_report"}
    assert all(tool.sequential for tool in report_toolset().tools.values())

    # Layout and full agents get report tools; data-only and Explore agents do not by default.
    assert report_names <= _tools(create_agent("test", dashboard_tools="layout"))
    assert report_names <= _tools(create_agent("test", dashboard_tools=None))
    assert not (report_names & _tools(create_agent("test", dashboard_tools="data")))
    assert not (report_names & _tools(create_explore_agent("test")))

    # Explicit opt-in / opt-out via report_tools=.
    assert report_names <= _tools(create_agent("test", dashboard_tools="data", report_tools=True))
    assert not (report_names & _tools(create_agent("test", dashboard_tools="layout", report_tools=False)))


async def test_end_to_end_agent_writes_and_edits_report_prose_and_persists_round_trip():
    model, requests = _script(
        [ToolCallPart("add_sql", {"source": "data", "name": "paid_by_region", "sql": BY_REGION}, tool_call_id="s1")],
        [ToolCallPart("add_chart", {"data": "paid_by_region", "name": "paid_chart", "spec": BAR}, tool_call_id="c1")],
        [ToolCallPart("describe_report", {}, tool_call_id="d1")],
        # First attempt mistakenly reads from chart node; tool retries with upstream hint.
        [ToolCallPart("add_report_prose", {"text": "Top total: ${{ paid_chart.data.total:,.0f }}.", "before": "paid_chart"}, tool_call_id="p1")],
        [
            ToolCallPart(
                "add_report_prose",
                {
                    "text": "Top region is {{ paid_by_region.data.region }} at ${{ paid_by_region.data.total:,.0f }}.",
                    "heading": "Regional summary",
                    "before": "paid_chart",
                    "title": "Orders Report",
                },
                tool_call_id="p2",
            )
        ],
        [
            ToolCallPart(
                "edit_report_prose",
                {"index": 2, "text": "**{{ paid_by_region.data.region }}** leads paid orders with ${{ paid_by_region.data.total:,.0f }}."},
                tool_call_id="e1",
            )
        ],
        [TextPart("Wrote the regional summary in the report.")],
    )
    events = []
    session = AgentSession(
        DuckDBSource(uri=":memory:", tables=TABLES),
        model,
        agent=create_agent(model, dashboard_tools="layout"),
    )
    session.deps.edit_listeners.append(lambda before, after: events.append((before, after)))
    try:
        turn = await session.ask("Chart paid orders by region and write a short report summary")
        rendered = session.ops.report_text()
        saved_model = session.editor.to_model()
    finally:
        session.close()

    assert turn.error is None
    assert REPORT_INSTRUCTIONS.strip() in str(requests[0].instructions)
    # The chart-node mistake was retried with the upstream SQL node hint.
    retry_part = requests[4].parts[0]
    assert getattr(retry_part, "outcome", None) == "retried" or isinstance(retry_part, RetryPromptPart)
    assert "paid_by_region" in str(retry_part.content)
    # The describe_report call succeeded and was read-only (not an edit listener event).
    assert isinstance(requests[3].parts[0], ToolReturnPart)
    # 4 editing calls fired edit_listeners: add_sql, add_chart, add_report_prose (p2), edit_report_prose (e1).
    assert len(events) == 4
    assert "# Orders Report" in rendered
    assert "## Regional summary" in rendered
    assert "**east** leads paid orders with $50." in rendered

    # Report survives a save/load round trip.
    reloaded = FlowDash(components=COMPONENTS)
    try:
        bind_source(reloaded, DuckDBSource(uri=":memory:", tables=TABLES), "data")
        reloaded.load_model(saved_model)
        reloaded_text = DashboardOps(reloaded).report_text()
        assert "# Orders Report" in reloaded_text
        assert "**east** leads paid orders with $50." in reloaded_text
    finally:
        reloaded.dispose()
