import pandas as pd
import panel as pn
import param
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("instructor")

from panel_flowdash import FlowDash
from panel_flowdash.agent import (
    DashboardOps, OperationError, dashboard_from_yaml, dashboard_to_yaml,
)

from lumen.ai.analysis import Analysis
from lumen.flowdash import (
    ANALYSES_KEY, ANALYSIS_ID, COMPONENTS, TABLE_ID, VIEW_ID, AnalysisNode,
    bind_analysis, bind_source, bound_analyses, check_analysis_config,
    unbind_analysis,
)
from lumen.flowdash.analysis import analysis_choices, analysis_presets
from lumen.flowdash.app import build_routes
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.sources.duckdb import DuckDBSource
from lumen.transforms.base import Filter as FilterTransform
from lumen.views import Panel as LumenPanelView


class ScaleAmount(Analysis):
    """Multiply the amount column by a factor and return a Pipeline."""

    columns = ["amount"]

    factor = param.Number(default=2.0, bounds=(0.1, 100.0), doc="Scale multiplier.")

    _hidden = param.String(default="secret", precedence=-1)

    def __call__(self, pipeline, context):
        df = pipeline.data.copy()
        df["scaled"] = df["amount"] * self.factor
        src = InMemorySource(tables={"scaled": df})
        return Pipeline(source=src, table="scaled")


class TopByColumn(Analysis):
    """Return top-N rows sorted by a chosen column as a DataFrame."""

    columns = [("amount", "score")]

    column = param.Selector(default=None, objects=[], doc="Column to rank by.")

    top_n = param.Integer(default=2, bounds=(1, 50), doc="Number of rows to keep.")

    _field_params = ["column"]

    def __call__(self, pipeline, context):
        df = pipeline.data
        col = self.column or "amount"
        return df.nlargest(self.top_n, col).reset_index(drop=True)


class SummaryView(Analysis):
    """Render a Panel Markdown summary of total amount."""

    columns = ["amount"]

    heading = param.String(default="Summary", doc="Heading text.")

    def __call__(self, pipeline, context):
        total = pipeline.data["amount"].sum()
        return pn.pane.Markdown(f"## {self.heading}\nTotal: {total}")


class LumenViewAnalysis(Analysis):
    """Return a Lumen Panel View wrapping a Markdown pane."""

    def __call__(self, pipeline, context):
        return LumenPanelView(object=pn.pane.Markdown(f"Rows: {len(pipeline.data)}"), pipeline=pipeline)


class AsyncAnalysis(Analysis):
    """Async applies and async __call__ returning a filtered Pipeline."""

    min_rows = param.Integer(default=1, bounds=(1, 100))

    @classmethod
    async def applies(cls, pipeline) -> bool:
        return len(pipeline.data) >= 2

    async def __call__(self, pipeline, context):
        return pipeline.chain(transforms=[FilterTransform(conditions=[("category", "A")])])


def _output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


def _component(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


@pytest.fixture
def duck():
    return DuckDBSource(
        uri=":memory:",
        tables={
            "sales": ("SELECT * FROM (VALUES ('A', 10), ('A', 5), ('B', 20)) AS t(category, amount)"),
            "names": "SELECT * FROM (VALUES ('alice'), ('bob')) AS t(label)",
        },
    )


def _setup(duck, analyses=None):
    context = {ANALYSES_KEY: analyses} if analyses is not None else {}
    editor = FlowDash(components=COMPONENTS, notifications=False, context=context)
    bind_source(editor, duck, "data")
    ops = DashboardOps(editor)
    ops.add_node(TABLE_ID, {"ref": "data", "table": "sales"}, name="sales")
    return editor, ops


def test_analysis_node_registered_and_wraps_pipeline_output(duck):
    assert COMPONENTS[ANALYSIS_ID] is AnalysisNode
    editor, ops = _setup(duck, [ScaleAmount])
    try:
        ops.add_node(ANALYSIS_ID, {"analysis": "ScaleAmount", "params": {"factor": 3.0}}, name="scaled")
        ops.connect("sales.pipeline", "scaled.pipeline")

        result_pipe = _output(ops, "scaled", "result")
        data = _output(ops, "scaled", "data")
        view = _output(ops, "scaled", "view")

        assert isinstance(result_pipe, Pipeline)
        assert view is None
        assert sorted(data["scaled"].tolist()) == [15.0, 30.0, 60.0]

        # Downstream chart reads the analysis's result pipeline.
        spec = {
            "mark": "bar",
            "encoding": {
                "x": {"field": "category", "type": "nominal"},
                "y": {"field": "scaled", "type": "quantitative"},
            },
        }
        ops.add_node(VIEW_ID, {"spec": spec}, name="chart")
        ops.connect("scaled.result", "chart.pipeline")
        assert _component(ops, "chart")._error is None
    finally:
        editor.dispose()


def test_analysis_returning_dataframe_populates_data_and_pipeline(duck):
    editor, ops = _setup(duck, {"top": TopByColumn})
    try:
        ops.add_node(
            ANALYSIS_ID,
            {"analysis": "top", "params": {"column": "amount", "top_n": 2}},
            name="top_sales",
        )
        ops.connect("sales.pipeline", "top_sales.pipeline")

        data = _output(ops, "top_sales", "data")
        pipe = _output(ops, "top_sales", "result")
        assert isinstance(data, pd.DataFrame)
        assert data["amount"].tolist() == [20, 10]
        assert isinstance(pipe, Pipeline)
        assert _output(ops, "top_sales", "view") is None
    finally:
        editor.dispose()


def test_analysis_returning_viewable_and_lumen_view(duck):
    editor, ops = _setup(duck, [SummaryView, LumenViewAnalysis])
    try:
        ops.add_node(ANALYSIS_ID, {"analysis": "SummaryView", "params": {"heading": "Sales"}}, name="summary")
        ops.connect("sales.pipeline", "summary.pipeline")

        view = _output(ops, "summary", "view")
        assert isinstance(view, pn.pane.Markdown)
        assert "Sales" in view.object and "35" in view.object
        assert _output(ops, "summary", "result") is None
        assert _output(ops, "summary", "data") is None
        assert _component(ops, "summary")._body() is view
        assert "SummaryView view" in _component(ops, "summary").__flowdash_preview__()

        ops.add_node(ANALYSIS_ID, {"analysis": "LumenViewAnalysis"}, name="lumen_view")
        ops.connect("sales.pipeline", "lumen_view.pipeline")
        lv = _output(ops, "lumen_view", "view")
        assert isinstance(lv, pn.viewable.Viewable)
    finally:
        editor.dispose()


def test_async_analysis_applies_and_call(duck):
    editor, ops = _setup(duck, [AsyncAnalysis])
    try:
        ops.add_node(ANALYSIS_ID, {"analysis": "AsyncAnalysis"}, name="only_a")
        ops.connect("sales.pipeline", "only_a.pipeline")

        data = _output(ops, "only_a", "data")
        assert sorted(data["amount"].tolist()) == [5, 10]
    finally:
        editor.dispose()


def test_preconfigured_analysis_instance_preserves_defaults(duck):
    configured = ScaleAmount.instance(name="triple", factor=3.0)
    editor, ops = _setup(duck, [configured])
    try:
        ops.add_node(ANALYSIS_ID, {"analysis": "triple"}, name="tripled")
        ops.connect("sales.pipeline", "tripled.pipeline")
        assert sorted(_output(ops, "tripled", "data")["scaled"].tolist()) == [15.0, 30.0, 60.0]

        # Overriding in node params does not mutate the registered instance.
        ops.set_config("tripled", {"params": {"factor": 5.0}})
        assert sorted(_output(ops, "tripled", "data")["scaled"].tolist()) == [25.0, 50.0, 100.0]
        assert configured.factor == 3.0
    finally:
        editor.dispose()


def test_bind_and_unbind_analysis_refreshes_waiting_node(duck):
    editor, ops = _setup(duck)
    try:
        ops.add_node(ANALYSIS_ID, {}, name="late")
        ops.editor._apply_config_patch(ops._instance("late"), {"analysis": "ScaleAmount"})
        ops.connect("sales.pipeline", "late.pipeline")
        assert _component(ops, "late")._error is not None

        bind_analysis(editor, ScaleAmount)
        assert "ScaleAmount" in bound_analyses(editor)
        assert _component(ops, "late")._error is None
        assert sorted(_output(ops, "late", "data")["scaled"].tolist()) == [10.0, 20.0, 40.0]

        unbind_analysis(editor, "ScaleAmount")
        assert "ScaleAmount" not in bound_analyses(editor)
    finally:
        editor.dispose()


def test_check_analysis_config_and_validation(duck, monkeypatch):
    editor, ops = _setup(duck, [ScaleAmount, TopByColumn])
    try:
        ops.add_node(TABLE_ID, {"ref": "data", "table": "names"}, name="names")

        # Unknown analysis rejected.
        with pytest.raises(OperationError, match="no analysis `Missing`"):
            ops.add_node(ANALYSIS_ID, {"analysis": "Missing"}, name="bad")

        # Internal Analysis params and negative-precedence params rejected.
        for forbidden in ("agents", "interface", "_run_button", "_dynamic_provides", "_hidden"):
            with pytest.raises(OperationError, match=f"no parameter '{forbidden}'"):
                ops.add_node(
                    ANALYSIS_ID,
                    {"analysis": "ScaleAmount", "params": {forbidden: "x"}},
                    name="bad_param",
                )

        # Out-of-bounds param rejected.
        with pytest.raises(OperationError, match="Invalid value for parameter 'factor'"):
            ops.add_node(
                ANALYSIS_ID,
                {"analysis": "ScaleAmount", "params": {"factor": -5.0}},
                name="oob",
            )

        # Missing required column fails on check_analysis_config and on resolve.
        names_pipe = _output(ops, "names", "pipeline")
        monkeypatch.setattr("lumen.flowdash.analysis.dashboard_context", lambda: editor.context)
        diag = check_analysis_config({"analysis": "ScaleAmount", "params": {}}, {"pipeline": names_pipe})
        assert diag is not None and "'amount'" in str(diag)
        assert analysis_choices({}, {}) == {"analysis": ["ScaleAmount", "TopByColumn"]}

        presets = analysis_presets(editor.context)
        assert [p["config"] for p in presets] == [{"analysis": "ScaleAmount"}, {"analysis": "TopByColumn"}]
    finally:
        editor.dispose()


def test_global_filters_affect_body_not_outputs(duck):
    from panel_flowdash.selection import Predicate

    editor, ops = _setup(duck, [SummaryView])
    try:
        ops.add_node(ANALYSIS_ID, {"analysis": "SummaryView"}, name="summary")
        ops.connect("sales.pipeline", "summary.pipeline")
        node = _component(ops, "summary")
        assert "35" in node._body().object

        node.__flowdash_apply_filters__({"category": Predicate(field="category", op="in", values=["A"])})
        assert "15" in node._body().object
        # Unfiltered output stays unchanged per GlobalFilterTarget contract.
        assert "35" in _output(ops, "summary", "view").object
    finally:
        editor.dispose()


def test_yaml_round_trip(duck):
    editor, ops = _setup(duck, [ScaleAmount])
    try:
        ops.add_node(ANALYSIS_ID, {"analysis": "ScaleAmount", "params": {"factor": 4.0}}, name="quad")
        ops.connect("sales.pipeline", "quad.pipeline")
        text = dashboard_to_yaml(editor)
        assert "Lumen/analysis" in text and "ScaleAmount" in text
    finally:
        editor.dispose()

    reloaded = FlowDash(components=COMPONENTS, notifications=False, context={ANALYSES_KEY: [ScaleAmount]})
    bind_source(reloaded, duck, "data")
    try:
        reloaded.import_dashboard(dashboard_from_yaml(text))
        reloaded_ops = DashboardOps(reloaded)
        assert sorted(_output(reloaded_ops, "quad", "data")["scaled"].tolist()) == [20.0, 40.0, 80.0]
    finally:
        reloaded.dispose()


def test_build_routes_accepts_analyses_without_marking_sensitive(tmp_path):
    csv = tmp_path / "sales.csv"
    csv.write_text("category,amount\nA,10\nB,20\n")
    routes = build_routes([str(csv)], db_path=tmp_path / "dash.db", analyses=[ScaleAmount])
    app = routes["/"]()
    try:
        assert ANALYSIS_ID not in app.agent_sensitive_components
        assert "ScaleAmount" in bound_analyses(app._editor)
    finally:
        app._editor.dispose()
