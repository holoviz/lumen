from types import SimpleNamespace

import panel as pn
import param
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("instructor")

from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession, _graph
from lumen.agent.authoring import add_analysis, add_chart
from lumen.agent.data import list_tables
from lumen.agent.edits import dashboard_state
from lumen.ai.analysis import Analysis
from lumen.flowdash import ANALYSIS_ID
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "sales": ("SELECT * FROM (VALUES ('A', 10), ('A', 20), ('B', 30)) AS t(category, amount)"),
    "notes": "SELECT * FROM (VALUES ('x'), ('y')) AS t(note)",
}


class NormalizeAmount(Analysis):
    """Compute share of total amount per row."""

    columns = ["amount"]

    scale = param.Number(default=100.0, bounds=(1.0, 1000.0), doc="Percentage scale.")

    def __call__(self, pipeline, context):
        df = pipeline.data.copy()
        df["share"] = df["amount"] / df["amount"].sum() * self.scale
        return Pipeline(source=InMemorySource(tables={"normalized": df}), table="normalized")


class TopCategories(Analysis):
    """Return top rows by a chosen numeric column."""

    columns = ["amount"]

    column = param.Selector(default=None, objects=[], doc="Metric column.")

    n = param.Integer(default=2, bounds=(1, 10), doc="Rows to keep.")

    _field_params = ["column"]

    def __call__(self, pipeline, context):
        col = self.column or "amount"
        return pipeline.data.nlargest(self.n, col).reset_index(drop=True)


class ExecutiveCard(Analysis):
    """Render a summary Markdown card of total amount."""

    columns = ["amount"]

    title = param.String(default="Total", doc="Card title.")

    def __call__(self, pipeline, context):
        total = pipeline.data["amount"].sum()
        return pn.pane.Markdown(f"### {self.title}: {total}")


class InternalOnly(Analysis):
    """Should never be visible to or callable by the agent."""

    _callable_by_llm = False

    def __call__(self, pipeline, context):
        return pipeline


class HighVolumeOnly(Analysis):
    """Applies only to datasets with at least 100 rows."""

    @classmethod
    async def applies(cls, pipeline) -> bool:
        return len(pipeline.data) >= 100

    def __call__(self, pipeline, context):
        return pipeline


@pytest.fixture
def session():
    sess = AgentSession(
        DuckDBSource(uri=":memory:", tables=TABLES),
        FunctionModel(lambda m, i: ModelResponse([TextPart("")])),
        analyses=[NormalizeAmount, TopCategories, ExecutiveCard, InternalOnly, HighVolumeOnly],
    )
    sess.add_sql("sales_rows", "SELECT category, amount FROM sales")
    sess.add_sql("note_rows", "SELECT note FROM notes")
    yield sess
    sess.close()


@pytest.fixture
def ctx(session):
    return SimpleNamespace(deps=session.deps)


async def test_list_tables_and_dashboard_state_list_only_llm_callable_analyses(ctx, session):
    text = await list_tables(ctx)
    assert "Analyses, custom analyses the app provides; run one on a data node with add_analysis:" in text
    assert "- NormalizeAmount: Compute share of total amount per row." in text
    assert "  required columns: amount" in text
    assert "scale" in text and "default 100.0" in text
    assert "InternalOnly" not in text

    state = dashboard_state(session.ops)
    assert "The app's analyses, which list_tables describes:" in state
    assert "NormalizeAmount" in state
    assert "InternalOnly" not in state


def test_dashboard_state_omits_analyses_when_none_registered():
    sess = AgentSession(
        DuckDBSource(uri=":memory:", tables=TABLES),
        FunctionModel(lambda m, i: ModelResponse([TextPart("")])),
    )
    try:
        assert "analyses" not in dashboard_state(sess.ops)
    finally:
        sess.close()


async def test_add_analysis_pipeline_and_dataframe_outputs_can_be_charted(ctx, session):
    result = await add_analysis(ctx, "sales_rows", "NormalizeAmount", "amount_shares", {"scale": 1.0})
    assert result.startswith("Placed analysis node 'amount_shares' (NormalizeAmount) on 'sales_rows'")
    ops = session.ops
    assert _graph.component_of(ops, "amount_shares") == ANALYSIS_ID
    frame = _graph.output(ops, "amount_shares", "data")
    assert pytest.approx(frame["share"].sum()) == 1.0

    # Downstream chart reads the analysis node's result pipeline.
    charted = add_chart(
        ctx,
        "amount_shares",
        "share_chart",
        {
            "mark": "bar",
            "encoding": {"x": {"field": "category"}, "y": {"field": "share"}},
        },
    )
    assert charted.startswith("Placed chart 'share_chart'")

    # In-place update keeps the node and recomputes with new params.
    changed = await add_analysis(ctx, "sales_rows", "NormalizeAmount", "amount_shares", {"scale": 100.0})
    assert changed.startswith("Changed analysis node 'amount_shares'")
    assert pytest.approx(_graph.output(ops, "amount_shares", "data")["share"].sum()) == 100.0

    # DataFrame-returning analysis with _field_params works and is listed in list_tables.
    top = await add_analysis(ctx, "sales_rows", "TopCategories", "top_sales", {"column": "amount", "n": 1})
    assert top.startswith("Placed analysis node 'top_sales' (TopCategories)")
    assert _graph.output(ops, "top_sales", "data")["amount"].tolist() == [30]
    listing = await list_tables(ctx)
    assert "amount_shares (analysis 'NormalizeAmount'" in listing


async def test_add_analysis_view_output_shows_tile_and_refuses_charting(ctx, session):
    result = await add_analysis(ctx, "sales_rows", "ExecutiveCard", "total_card", {"title": "Revenue"})
    assert result == "Placed analysis node 'total_card' (ExecutiveCard) on 'sales_rows': view (Markdown)."
    view = _graph.output(session.ops, "total_card", "view")
    assert isinstance(view, pn.pane.Markdown) and "Revenue: 60" in view.object

    # Tile is made visible on the dashboard so the user sees the view.
    state = dashboard_state(session.ops)
    assert "total_card (analysis: ExecutiveCard" in state
    assert "grid row" in state

    with pytest.raises(ModelRetry, match="no Lumen pipeline output"):
        add_chart(
            ctx,
            "total_card",
            "bad_chart",
            {
                "mark": "bar",
                "encoding": {"x": {"field": "category"}, "y": {"field": "amount"}},
            },
        )


async def test_add_analysis_refusals_and_retries(ctx, session):
    # Unknown analysis or non-LLM analysis.
    with pytest.raises(ModelRetry, match="No analysis 'Unknown'"):
        await add_analysis(ctx, "sales_rows", "Unknown", "a")
    with pytest.raises(ModelRetry, match="No analysis 'InternalOnly'"):
        await add_analysis(ctx, "sales_rows", "InternalOnly", "a")

    # Missing required columns.
    with pytest.raises(ModelRetry, match=r"requires column.*'amount'"):
        await add_analysis(ctx, "note_rows", "NormalizeAmount", "a")

    # Custom applies() returning False.
    with pytest.raises(ModelRetry, match="does not apply"):
        await add_analysis(ctx, "sales_rows", "HighVolumeOnly", "a")

    # Invalid param name or value.
    with pytest.raises(ModelRetry, match="no parameter 'bogus'"):
        await add_analysis(ctx, "sales_rows", "NormalizeAmount", "a", {"bogus": 1})
    with pytest.raises(ModelRetry, match="Invalid value for parameter 'scale'"):
        await add_analysis(ctx, "sales_rows", "NormalizeAmount", "a", {"scale": 0.0})

    # Name collision with another component kind.
    with pytest.raises(ModelRetry, match="already a SQL node"):
        await add_analysis(ctx, "sales_rows", "NormalizeAmount", "sales_rows")

    assert "a" not in _graph.nodes(session.ops)


async def test_add_analysis_refuses_when_app_has_no_analyses():
    sess = AgentSession(
        DuckDBSource(uri=":memory:", tables=TABLES),
        FunctionModel(lambda m, i: ModelResponse([TextPart("")])),
    )
    try:
        sess.add_sql("sales_rows", "SELECT category, amount FROM sales")
        ctx = SimpleNamespace(deps=sess.deps)
        result = await add_analysis(ctx, "sales_rows", "NormalizeAmount", "a")
        assert result.startswith("Refused: this app defines no analyses")
    finally:
        sess.close()


async def test_add_analysis_runs_in_agent_session_without_approval():
    calls = iter(
        [
            [
                ToolCallPart(
                    "add_analysis",
                    {
                        "data": "sales_rows",
                        "analysis": "NormalizeAmount",
                        "name": "shares",
                        "params": {"scale": 100.0},
                    },
                )
            ],
            [TextPart("Computed shares.")],
        ]
    )
    sess = AgentSession(
        DuckDBSource(uri=":memory:", tables=TABLES),
        FunctionModel(lambda m, i: ModelResponse(next(calls))),
        analyses=[NormalizeAmount],
    )
    try:
        sess.add_sql("sales_rows", "SELECT category, amount FROM sales")
        turn = await sess.ask("Normalize the sales amounts")
        assert turn.error is None
        assert "shares" in turn.nodes
        assert turn.output == "Computed shares."
    finally:
        sess.close()


async def test_add_analysis_in_explore_copies_earlier_answer_and_protects_dashboard(session, monkeypatch):
    import panel_agents

    from panel_flowdash import agent_edits

    flowdash_agent = pytest.importorskip("panel_flowdash.agent")
    if not hasattr(flowdash_agent, "ExplorationOps"):
        pytest.skip("panel-flowdash without ExplorationOps")
    session.editor.new_dashboard("Sales")
    session.ops.add_node("Lumen/source", {"ref": session.source_node}, name=session.source_node)
    session.add_sql("sales_rows", "SELECT category, amount FROM sales")
    await add_analysis(SimpleNamespace(deps=session.deps), "sales_rows", "NormalizeAmount", "dash_norm", {"scale": 10.0})
    explored = flowdash_agent.ExplorationOps(session.ops)
    ctx = SimpleNamespace(deps=flowdash_agent.FlowDashDeps(explored))
    run = {"id": "r1"}
    monkeypatch.setattr(panel_agents, "current_run_id", lambda: run["id"])
    try:
        with pytest.raises(PermissionError, match="under another name"):
            await add_analysis(ctx, "sales_rows", "NormalizeAmount", "dash_norm", {"scale": 50.0})

        with agent_edits("r1"):
            placed = await add_analysis(ctx, "sales_rows", "NormalizeAmount", "mine", {"scale": 10.0})
        assert placed.startswith("Placed analysis node 'mine'")

        run["id"] = "r2"
        with agent_edits("r2"):
            changed = await add_analysis(ctx, "sales_rows", "NormalizeAmount", "mine", {"scale": 50.0})
        assert changed.startswith("Changed analysis node 'mine_2'")
        assert explored.node_config("mine")["params"] == {"scale": 10.0}
        assert explored.node_config("mine_2")["params"] == {"scale": 50.0}
    finally:
        explored.dispose()
