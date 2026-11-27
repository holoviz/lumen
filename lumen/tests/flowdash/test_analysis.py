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
from lumen.flowdash.analysis import (
    analysis_choices, analysis_presets, build_analysis, run_analysis,
)
from lumen.flowdash.app import build_routes
from lumen.flowdash.nodes import set_row_ceiling
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


class RankByName(Analysis):
    """Rank rows by a column named in a String field parameter."""

    column = param.String(default="amount")

    _field_params = ["column"]

    def __call__(self, pipeline, context):
        return pipeline.data.sort_values(self.column).reset_index(drop=True)


class Broken(Analysis):
    """Fails with an error that is not a ValueError."""

    def __call__(self, pipeline, context):
        raise RuntimeError("analysis exploded")


def test_a_field_param_that_is_not_a_selector_is_checked_against_the_columns():
    columns = ["category", "amount"]
    assert build_analysis(RankByName, {"column": "category"}, columns=columns).column == "category"
    with pytest.raises(ValueError, match="names column\\(s\\) 'nope', not in pipeline columns"):
        build_analysis(RankByName, {"column": "nope"}, columns=columns)
    # Without the columns there is nothing to check against.
    assert build_analysis(RankByName, {"column": "nope"}).column == "nope"
    # A Selector field still gets the columns as its objects.
    assert build_analysis(TopByColumn, {"column": "amount"}, columns=columns).param["column"].objects == columns


def test_a_bad_string_field_param_is_refused_in_the_config(duck):
    editor, ops = _setup(duck, [RankByName])
    try:
        ops.add_node(ANALYSIS_ID, {"analysis": "RankByName"}, name="ranked")
        ops.connect("sales.pipeline", "ranked.pipeline")
        with pytest.raises(OperationError, match="names column\\(s\\) 'nope'"):
            ops.set_config("ranked", {"params": {"column": "nope"}})
        ops.set_config("ranked", {"params": {"column": "category"}})
        assert _output(ops, "ranked", "data")["category"].tolist() == ["A", "A", "B"]
    finally:
        editor.dispose()


def test_a_host_instance_keeps_its_columns():
    configured = RankByName.instance(name="by_label")
    with param.edit_constant(configured):
        configured.columns = ["label"]
    assert build_analysis(configured).columns == ["label"]


def test_an_analysis_on_a_capped_input_says_so(duck):
    editor, ops = _setup(duck, [SummaryView])
    try:
        sales = _output(ops, "sales", "pipeline")
        assert run_analysis(SummaryView, sales).capped_at is None
        assert run_analysis(SummaryView, sales, limit=2).capped_at == 2

        set_row_ceiling(editor, 2)
        ops.add_node(ANALYSIS_ID, {"analysis": "SummaryView"}, name="summary")
        ops.connect("sales.pipeline", "summary.pipeline")
        node = _component(ops, "summary")
        body = node._body()
        assert isinstance(body, pn.Column) and "first 2 rows" in body[0].object
        assert "ran on the first 2 rows" in node.__flowdash_preview__()
    finally:
        editor.dispose()


def test_an_analysis_raising_another_error_shows_it_on_the_node(duck):
    editor, ops = _setup(duck, [Broken])
    try:
        ops.add_node(ANALYSIS_ID, {"analysis": "Broken"}, name="broken")
        ops.connect("sales.pipeline", "broken.pipeline")
        node = _component(ops, "broken")
        assert node._object is None and "analysis exploded" in node._error
    finally:
        editor.dispose()


def test_global_filters_run_the_analysis_once_per_change(duck, monkeypatch):
    from panel_flowdash.selection import Predicate

    from lumen.flowdash import analysis as module

    editor, ops = _setup(duck, [SummaryView])
    try:
        ops.add_node(ANALYSIS_ID, {"analysis": "SummaryView"}, name="summary")
        ops.connect("sales.pipeline", "summary.pipeline")
        node = _component(ops, "summary")
        runs = []
        real = module.run_analysis
        monkeypatch.setattr(module, "run_analysis", lambda *a, **k: runs.append(1) or real(*a, **k))

        node.__flowdash_apply_filters__({"category": Predicate(field="category", op="in", values=["A"])})
        assert "15" in node._body().object
        assert "15" in node._body().object
        assert len(runs) == 1

        node.__flowdash_apply_filters__({"category": Predicate(field="category", op="in", values=["B"])})
        assert "20" in node._body().object
        assert len(runs) == 2
    finally:
        editor.dispose()


def test_unbinding_an_analysis_refreshes_the_nodes_naming_it(duck):
    editor, ops = _setup(duck, [])
    try:
        bind_analysis(editor, ScaleAmount)
        ops.add_node(ANALYSIS_ID, {"analysis": "ScaleAmount"}, name="scaled")
        ops.connect("sales.pipeline", "scaled.pipeline")
        node = _component(ops, "scaled")
        assert node._error is None

        unbind_analysis(editor, "ScaleAmount")
        assert node._object is None and "no Lumen analyses" in node._error
    finally:
        editor.dispose()


def test_analysis_access_rules_and_enforcement(duck, tmp_path):
    import weakref

    from panel_flowdash.auth import Identity, Permission
    from panel_flowdash.editor import _current

    from lumen.flowdash import (
        ANALYSES_ACCESS_KEY, authorized_analyses, can_access_analysis,
    )
    from lumen.flowdash.access import bind_identity

    hr_user = Identity("alice", groups=frozenset({"hr"}))
    sales_user = Identity("sam", groups=frozenset({"sales"}))
    denied_admin = Identity("blocked", groups=frozenset({"admins", "hr"}))
    admin = Identity("root", groups=frozenset({"admins"}))

    # Build a saved dashboard as admin with both a public and a restricted analysis.
    rules = {
        "ScaleAmount": {"allow_groups": ["hr"], "deny_users": ["blocked"]},
    }
    author = FlowDash(
        components=COMPONENTS,
        notifications=False,
        context={ANALYSES_KEY: [ScaleAmount, SummaryView], ANALYSES_ACCESS_KEY: rules},
    )
    bind_source(author, duck, "data")
    bind_identity(author, admin, admin=True)
    try:
        a_ops = DashboardOps(author)
        a_ops.add_node(TABLE_ID, {"ref": "data", "table": "sales"}, name="sales")
        a_ops.add_node(ANALYSIS_ID, {"analysis": "ScaleAmount"}, name="restricted")
        a_ops.connect("sales.pipeline", "restricted.pipeline")
        a_ops.add_node(ANALYSIS_ID, {"analysis": "SummaryView"}, name="public")
        a_ops.connect("sales.pipeline", "public.pipeline")
        saved = author.export_dashboard()
    finally:
        author.dispose()

    # Unauthorized sales user: choices/presets/authorized_analyses omit ScaleAmount,
    # config check and runtime node report permission denied, missing hint hides restricted names.
    editor = FlowDash(
        components=COMPONENTS,
        notifications=False,
        context={ANALYSES_KEY: [ScaleAmount, SummaryView], ANALYSES_ACCESS_KEY: rules},
    )
    bind_source(editor, duck, "data")
    bind_identity(editor, sales_user)
    try:
        ops = DashboardOps(editor)
        assert not can_access_analysis("ScaleAmount", editor.context)
        assert can_access_analysis("SummaryView", editor.context)
        assert set(authorized_analyses(editor.context)) == {"SummaryView"}

        with _current(weakref.ref(editor)):
            assert analysis_choices({}, {}) == {"analysis": ["SummaryView"]}
            assert [p["config"]["analysis"] for p in analysis_presets(editor.context)] == ["SummaryView"]
            diag = check_analysis_config({"analysis": "ScaleAmount"}, {})
            assert diag is not None and "permission" in str(diag).lower()
            missing = check_analysis_config({"analysis": "DoesNotExist"}, {})
            assert missing is not None and "ScaleAmount" not in str(missing) and "SummaryView" in str(missing)

        with pytest.raises(OperationError, match="permission"):
            ops.add_node(ANALYSIS_ID, {"analysis": "ScaleAmount"}, name="nope")

        # Opening an admin-saved dashboard fails closed on the restricted analysis node.
        editor.import_dashboard(saved)
        nodes = ops.preview()["nodes"]
        assert nodes["public"]["status"] == "ok"
        assert nodes["restricted"]["status"] in ("invalid", "failed")
        assert "permission" in nodes["restricted"]["message"].lower()

        # Explicit deny beats admin; unrestricted admin passes allowlist; no identity fails closed.
        assert can_access_analysis("ScaleAmount", editor.context, identity=hr_user)
        assert can_access_analysis("ScaleAmount", editor.context, identity=admin, admin=True)
        assert not can_access_analysis("ScaleAmount", editor.context, identity=denied_admin, admin=True)
        assert not can_access_analysis("ScaleAmount", editor.context, identity=None, admin=True)
        assert can_access_analysis("SummaryView", editor.context, identity=None)
    finally:
        editor.dispose()

    # Malformed raw context rules fail closed to admins-only.
    bad_ctx = {ANALYSES_KEY: [ScaleAmount], ANALYSES_ACCESS_KEY: {"ScaleAmount": {"allow_roles": ["hr"]}}}
    assert not can_access_analysis("ScaleAmount", bad_ctx, identity=hr_user)
    assert can_access_analysis("ScaleAmount", bad_ctx, identity=admin, admin=True)

    # bind_analysis(..., access=...) validates, attaches, and refreshes nodes; unbind clears it.
    editor2, ops2 = _setup(duck, [])
    bind_identity(editor2, sales_user)
    try:
        with pytest.raises(ValueError, match="Unknown key 'bad_key'"):
            bind_analysis(editor2, ScaleAmount, access={"bad_key": ["hr"]})
        bind_analysis(editor2, ScaleAmount, access=Permission.from_spec(allow_groups=["hr"]))
        assert not can_access_analysis("ScaleAmount", editor2.context)
        bind_analysis(editor2, ScaleAmount, access={"allow_groups": ["sales"]})
        assert can_access_analysis("ScaleAmount", editor2.context)
        ops2.add_node(ANALYSIS_ID, {"analysis": "ScaleAmount"}, name="bound")
        ops2.connect("sales.pipeline", "bound.pipeline")
        assert _component(ops2, "bound")._error is None
        bind_analysis(editor2, ScaleAmount, access={"allow_groups": ["hr"]})
        assert "permission" in (_component(ops2, "bound")._error or "").lower()
        unbind_analysis(editor2, "ScaleAmount")
        assert "ScaleAmount" not in editor2.context[ANALYSES_ACCESS_KEY]
    finally:
        editor2.dispose()

    # build_routes validates and wires analysis_access into editor context.
    csv = tmp_path / "sales.csv"
    csv.write_text("category,amount\nA,10\nB,20\n")
    with pytest.raises(ValueError, match="Invalid access rule for analysis 'ScaleAmount'"):
        build_routes(
            [str(csv)],
            db_path=tmp_path / "bad.db",
            analyses=[ScaleAmount],
            analysis_access={"ScaleAmount": {"unknown": ["hr"]}},
        )
    routes = build_routes(
        [str(csv)],
        db_path=tmp_path / "ok.db",
        analyses=[ScaleAmount, SummaryView],
        analysis_access={"ScaleAmount": {"allow_groups": ["hr"]}},
    )
    app = routes["/"]()
    try:
        assert app._editor.context[ANALYSES_ACCESS_KEY]["ScaleAmount"] == Permission.from_spec(allow_groups=["hr"])
        assert not can_access_analysis("ScaleAmount", app._editor.context, identity=sales_user)
        assert can_access_analysis("ScaleAmount", app._editor.context, identity=hr_user)
    finally:
        app._editor.dispose()
