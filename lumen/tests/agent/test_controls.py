import re

from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import OperationError
from panel_flowdash.agent.tools import TOOLS as FLOWDASH_TOOLS
from pydantic_ai import ModelRetry
from pydantic_ai.messages import (
    ModelResponse, RetryPromptPart, TextPart, ToolCallPart,
)
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession, _graph, assistant
from lumen.agent.assistant import create_agent, create_explore_agent
from lumen.agent.authoring import (
    add_chart, add_metric, add_table, authoring_toolset,
)
from lumen.agent.controls import (
    KPI_ID, add_filter_control, add_kpi, controls_toolset,
)
from lumen.agent.data import data_toolset
from lumen.flowdash import FILTER_ID, TABLE_ID
from lumen.flowdash.metrics import read_metrics
from lumen.flowdash.project import parse_project
from lumen.sources.duckdb import DuckDBSource

# Ad hoc captions read the provenance panel-flowdash gives a node, which 0.1.0rc8 does not have yet.
needs_provenance = pytest.mark.skipif(
    not hasattr(FlowDash, "node_provenance"), reason="panel-flowdash without node provenance"
)

# Turbines as in the wind data: p_cap is the project's capacity, repeated on each of its turbines' rows.
TURBINES = (
    "SELECT * FROM (VALUES "
    "('Alta', 'CA', 1999, 300.0, 1500, 80.0, DATE '1999-05-01'), "
    "('Alta', 'CA', 1999, 300.0, 1500, 80.0, DATE '1999-05-01'), "
    "('Roscoe', 'TX', 2010, 200.0, 2000, 90.5, DATE '2010-03-15'), "
    "('Roscoe', 'TX', 2010, 200.0, 2500, 95.0, DATE '2010-03-15'), "
    "('Story', 'IA', 2015, 100.0, 1000, 70.0, DATE '2015-08-30')"
    ") AS t(p_name, t_state, p_year, p_cap, t_cap, t_hh, t_date)"
)
PENGUINS = (
    "SELECT * FROM (VALUES ('Adelie', 'Torgersen', 3750, 181), ('Adelie', 'Biscoe', 3800, 186), "
    "('Gentoo', 'Biscoe', 5400, 217), ('Chinstrap', 'Dream', 3500, 192)) AS t(species, island, body_mass_g, flipper_length_mm)"
)
# 40 projects of 2 to 11 turbines each: p_cap, the project's capacity, repeats on each of its
# turbines' rows (one project in 17 has a stray value, as in the wind data); t_cap, a turbine's
# capacity, is fixed by its model but varies within a project; t_hh varies per turbine; one is a
# constant. In COMPOSITE, v depends on a and b together, neither alone.
PROJECTS = (
    "SELECT 'P' || p.i AS p_name, 'M' || ((p.i + t.j) % 4) AS t_model, ['TX', 'IA', 'PA'][p.i % 3 + 1] AS t_state, "
    "1500 + 500 * ((p.i + t.j) % 4) AS t_cap, "
    "(2 + p.i % 10) * 2.25 + CASE WHEN p.i % 17 = 0 AND t.j = 0 THEN 1 ELSE 0 END AS p_cap, "
    "80.0 + (p.i * 7 + t.j * 3) % 50 AS t_hh, 1.0 AS one "
    "FROM range(40) AS p(i), range(12) AS t(j) WHERE t.j < 2 + p.i % 10"
)
COMPOSITE = (
    "SELECT a.i AS a, b.i AS b, 10.0 * a.i + b.i AS v, r.i AS r "
    "FROM range(5) AS a(i), range(5) AS b(i), range(4) AS r(i)"
)
BIG = "SELECT range AS i, range % 7 AS g, 1.5 AS fare, range % 1000 AS k, (range % 1000) * 1.0 AS v FROM range(1000000)"
TABLES = {"turbines": TURBINES, "penguins": PENGUINS, "projects": PROJECTS, "composite": COMPOSITE}
BAR = {"mark": "bar", "encoding": {"x": {"field": "t_state", "type": "nominal"},
                                   "y": {"aggregate": "sum", "field": "t_cap", "type": "quantitative"}}}
PROJECT = """\
metrics:
  capacity:
    source: data
    table: turbines
    measures:
      capacity_mw: {expr: "SUM(t_cap) / 1000.0", format: ",.1f"}
      turbines: "COUNT(*)"
    dimensions:
      state: t_state
"""


def _model():
    return FunctionModel(lambda m, i: ModelResponse([TextPart("")]))


@pytest.fixture
def session():
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), _model())
    yield session
    session.close()


@pytest.fixture
def ctx(session):
    return SimpleNamespace(deps=session.deps)


def instance(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def graph(session):
    return yaml.safe_load(session.graph())


def structure(session):
    """The nodes, edge set and layout of the dashboard, without its edit history."""
    shown = graph(session)
    return shown["nodes"], sorted(map(str, shown.get("edges") or [])), shown.get("layout")


def retried(part, text):
    """Whether a request part asks the model to retry a tool call, saying ``text``."""
    return getattr(part, "outcome", None) == "retried" or (isinstance(part, RetryPromptPart) and text in str(part.content))


def top_bar(session):
    return [t["node"] for t in graph(session)["layout"]["tiles"] if t.get("area") == "top"]


def edges(session):
    return {(e["from"], e["to"]) for e in graph(session)["edges"]}


# add_kpi

@needs_provenance
def test_add_kpi_reuses_the_value_of_a_one_cell_sql_node(ctx, session):
    session.add_sql("capacity", "SELECT SUM(t_cap) / 1000.0 AS capacity_mw FROM turbines")
    result = add_kpi(ctx, "capacity", "capacity_kpi", label="Installed capacity", unit="MW")

    assert result.startswith("Placed 1 KPI card(s) in the top bar")
    assert "'capacity_kpi': Installed capacity = 8.50 MW." in result
    assert ("capacity.value", "capacity_kpi.value") in edges(session)
    assert top_bar(session) == ["capacity_kpi"]
    config = _graph.config_of(session.ops, "capacity_kpi")
    assert config == {"label": "Installed capacity", "format": ",.2f", "prefix": "", "suffix": " MW",
                      "caption": ("Ad hoc: SUM(t_cap) / 1000.0 over all rows of turbines, computed by its own SQL "
                                  "query; no metric was used.")}
    assert instance(session.ops, "capacity_kpi").value == 8.5


def test_add_kpi_splits_a_row_of_several_measures_into_one_kpi_each(ctx, session):
    session.add_sql("headline", "SELECT COUNT(*) AS turbines, SUM(t_cap) AS capacity_kw, "
                                "AVG(t_hh) AS avg_hub_height, 0.4 AS tx_share FROM turbines")
    result = add_kpi(ctx, "headline", "kpi")

    names = ["kpi_turbines", "kpi_capacity_kw", "kpi_avg_hub_height", "kpi_tx_share"]
    assert top_bar(session) == names
    assert "Turbines = 5." in result and "Capacity kw = 8,500." in result
    assert "Avg hub height = 83.1." in result and "Tx share = 40.0%." in result
    for name, column in zip(names, ["turbines", "capacity_kw", "avg_hub_height", "tx_share"], strict=True):
        assert _graph.config_of(session.ops, name)["label"]
        edge = next(e for e in graph(session)["edges"] if e["to"] == f"{name}.value")
        assert edge["from"] == "headline.data"
        assert edge["transform"] == [{"op": "column", "column": column}, {"op": "aggregate", "how": "last"}]
    assert instance(session.ops, "kpi_tx_share").value == pytest.approx(0.4)


def test_add_kpi_aggregates_a_column_over_the_rows_of_a_node(ctx, session):
    session.add_sql("turbine_rows", "SELECT * FROM turbines")
    result = add_kpi(ctx, "turbine_rows", "capacity", column="t_cap", aggregate="sum", label="Capacity", unit="kW")
    counted = add_kpi(ctx, "turbine_rows", "turbines", aggregate="count")

    assert "the sum of 't_cap' of 'turbine_rows' through 'capacity_agg', 'capacity_value'" in result
    assert "Capacity = 8,500 kW." in result
    assert "'turbines': Rows = 5." in counted
    assert _graph.component_of(session.ops, "capacity_agg") == "Lumen/aggregate"
    assert instance(session.ops, "capacity").value == 8500
    assert instance(session.ops, "turbines").value == 5
    hidden = {t["node"] for t in graph(session)["layout"]["tiles"] if t.get("visible") is False}
    assert {"capacity_agg", "capacity_value", "turbines_agg", "turbines_value"} <= hidden


def test_add_kpi_aggregates_every_row_of_a_sql_node_with_a_limit_of_its_own(ctx, session):
    session.add_sql("turbine_rows", "SELECT * FROM turbines ORDER BY t_cap DESC")
    session.ops.set_config("turbine_rows", {"limit": 2})
    add_kpi(ctx, "turbine_rows", "capacity", column="t_cap", aggregate="sum")
    add_kpi(ctx, "turbine_rows", "turbines", aggregate="count")

    assert len(_graph.output(session.ops, "turbine_rows", "data")) == 2
    assert instance(session.ops, "capacity").value == 8500
    assert instance(session.ops, "turbines").value == 5
    assert _graph.config_of(session.ops, "capacity")["caption"] == (
        "Ad hoc: Sum of t_cap over all rows of turbine_rows; no metric was used.")


def test_add_kpi_on_table_and_metric_nodes():
    metrics = read_metrics(parse_project(PROJECT, "flowdash.yaml"))
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), _model(), metrics=metrics)
    ctx = SimpleNamespace(deps=session.deps)
    try:
        session.ops.add_node(TABLE_ID, {"ref": "data", "table": "penguins"}, name="penguins_table")
        mass = add_kpi(ctx, "penguins_table", "mass", column="body_mass_g", aggregate="mean", unit="g")
        session.ops.add_node("Lumen/metric", {"metric": "capacity", "measures": ["capacity_mw"]}, name="total")
        total = add_kpi(ctx, "total", "total_kpi", column="capacity_mw", unit="MW")
    finally:
        session.close()
    assert "Average body mass g = 4,112 g." in mass
    assert "through 'total_kpi'" not in total and "Capacity mw = 8.50 MW." in total


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"data": "rows", "column": "t_state", "aggregate": "sum"}, "'t_state' holds string values; a KPI shows a number"),
    ({"data": "rows", "column": "t_cap"}, "'rows' has more than one row; pass aggregate"),
    ({"data": "rows", "column": "missing", "aggregate": "sum"}, "'rows' has no column 'missing'"),
    ({"data": "rows", "aggregate": "sum"}, "Name the column to sum"),
    ({"data": "turbines", "column": "t_cap", "aggregate": "sum"}, "'turbines' is a source table"),
    ({"data": "rows", "name": "rows", "column": "t_cap", "aggregate": "sum"}, "'rows' is taken"),
    ({"data": "two", "label": "Both"}, "label names one KPI"),
])
def test_add_kpi_refusals(ctx, session, kwargs, message):
    session.add_sql("rows", "SELECT * FROM turbines")
    session.add_sql("two", "SELECT SUM(t_cap) AS a, COUNT(*) AS b FROM turbines")
    before = structure(session)
    with pytest.raises(ModelRetry, match=re.escape(message)):
        add_kpi(ctx, **{"name": "kpi", **kwargs})
    assert structure(session) == before


def _projects(session, sql="SELECT * FROM projects", name="project_rows"):
    frame = session.source.execute(PROJECTS)
    session.add_sql(name, sql)
    return frame


def test_add_kpi_aggregates_the_rows_as_they_are_and_says_how(ctx, session):
    frame = _projects(session)
    result = add_kpi(ctx, "project_rows", "capacity", column="p_cap", aggregate="sum")
    assert instance(session.ops, "capacity").value == pytest.approx(frame.p_cap.sum())
    assert _graph.config_of(session.ops, "capacity")["caption"] == (
        "Ad hoc: Sum of p_cap over all rows of project_rows; no metric was used.")
    assert "The number is computed ad hoc, not an agreed metric; each card's caption says so." in result
    assert "asking an admin" not in result


def test_add_kpi_distinct_by_takes_the_largest_value_per_key(ctx, session):
    frame = _projects(session)
    result = add_kpi(ctx, "project_rows", "capacity", column="p_cap", aggregate="sum", distinct_by="p_name")
    expected = frame.groupby("p_name").p_cap.max().sum()
    assert instance(session.ops, "capacity").value == pytest.approx(expected)
    assert f"Total p cap = {expected:,.0f}." in result
    assert ("of 'project_rows', taking the largest 'p_cap' of each p_name once through 'capacity_per_key', "
            "'capacity_agg', 'capacity_value'") in result
    caption = _graph.config_of(session.ops, "capacity")["caption"]
    assert caption == "Ad hoc: Sum of p_cap over project_rows, taking the largest per p_name; no metric was used."


def test_a_count_with_distinct_by_counts_the_keys(ctx, session):
    frame = _projects(session)
    add_kpi(ctx, "project_rows", "projects", column="p_cap", aggregate="count", distinct_by="p_name")
    assert instance(session.ops, "projects").value == frame.p_name.nunique()
    assert _graph.config_of(session.ops, "projects")["caption"] == (
        "Ad hoc: Count of p_name values with a p_cap in project_rows; no metric was used.")


def test_an_ad_hoc_kpi_lists_at_most_ten_of_the_apps_metrics(monkeypatch):
    from lumen.agent import controls

    names = [f"m{i}" for i in range(12)]
    monkeypatch.setattr(controls, "host_metrics", lambda context: dict.fromkeys(names))
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), _model())
    try:
        session.add_sql("total", "SELECT SUM(t_cap) AS total FROM turbines")
        result = add_kpi(SimpleNamespace(deps=session.deps), "total", "kpi")
    finally:
        session.close()
    assert f"This app defines metrics: {', '.join(names[:10])} and 2 more. None was used" in result


def test_add_kpi_distinct_by_takes_a_composite_key(ctx, session):
    session.add_sql("composite_rows", "SELECT * FROM composite")
    add_kpi(ctx, "composite_rows", "total", column="v", aggregate="sum", distinct_by=["a", "b"])
    assert instance(session.ops, "total").value == sum(10 * a + b for a in range(5) for b in range(5))


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"distinct_by": "nope"}, "distinct_by names other columns of 'project_rows', not 'nope'"),
    ({"distinct_by": "p_cap"}, "not 'p_cap'"),
])
def test_add_kpi_refuses_distinct_by_other_than_a_column_of_the_node(ctx, session, kwargs, message):
    _projects(session)
    with pytest.raises(ModelRetry, match=re.escape(message)):
        add_kpi(ctx, "project_rows", "capacity", **{"column": "p_cap", "aggregate": "sum", **kwargs})


def test_a_kpi_of_a_metric_says_which_metric_and_others_point_to_an_admin():
    metrics = read_metrics(parse_project(PROJECT, "flowdash.yaml"))
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), _model(), metrics=metrics)
    ctx = SimpleNamespace(deps=session.deps)
    try:
        session.ops.add_node("Lumen/metric", {"metric": "capacity", "measures": ["capacity_mw"]}, name="total")
        from_metric = add_kpi(ctx, "total", "total_kpi", column="capacity_mw")
        session.add_sql("rows", "SELECT * FROM turbines")
        ad_hoc = add_kpi(ctx, "rows", "rows_kpi", column="t_cap", aggregate="sum")
        captions = [_graph.config_of(session.ops, n)["caption"] for n in ("total_kpi", "rows_kpi")]
    finally:
        session.close()
    assert captions == ["Metric capacity", "Ad hoc: Sum of t_cap over all rows of rows; no metric was used."]
    assert "ad hoc" not in from_metric
    assert ("This app defines metrics: capacity. None was used for this number; use add_metric if one fits, or ask "
            "an admin to define one.") in ad_hoc


def test_add_kpi_aggregates_every_row_of_a_capped_sql_node():
    session = AgentSession(DuckDBSource(uri=":memory:", tables={"big": BIG}), _model())
    try:
        session.add_sql("rows", "SELECT * FROM big")
        assert len(_graph.output(session.ops, "rows", "data")) < 1_000_000
        result = add_kpi(SimpleNamespace(deps=session.deps), "rows", "fares", column="fare", aggregate="sum")
        value = instance(session.ops, "fares").value
    finally:
        session.close()
    assert value == 1_500_000 and "Total fare = 1,500,000." in result
    assert "through 'fares_agg', 'fares_value'" in result


@pytest.mark.parametrize("node", ["filtered", "child", "table", "metric"])
def test_add_kpi_aggregates_every_row_of_large_nodes(node):
    metrics = read_metrics(parse_project(
        "metrics:\n  fares:\n    source: data\n    table: big\n    measures:\n      fare: SUM(fare)\n", "flowdash.yaml"))
    session = AgentSession(DuckDBSource(uri=":memory:", tables={"big": BIG}), _model(), metrics=metrics)
    ctx = SimpleNamespace(deps=session.deps)
    expected = {"filtered": 1.5 * sum(1 for i in range(1_000_000) if i % 7 < 3), "child": 1_500_000,
                "table": 1_500_000, "metric": 1_500_000}[node]
    try:
        if node == "filtered":
            session.add_sql("rows", "SELECT * FROM big WHERE g < 3")
        elif node == "child":
            session.add_sql("parent", "SELECT * FROM big")
            session.add_sql("rows", "SELECT * FROM parent", source="parent")
        elif node == "table":
            session.ops.add_node(TABLE_ID, {"ref": "data", "table": "big"}, name="rows")
        else:
            session.ops.add_node("Lumen/metric", {"metric": "fares", "measures": ["fare"]}, name="rows")
        add_kpi(ctx, "rows", "fares", column="fare", aggregate=None if node == "metric" else "sum")
        value = instance(session.ops, "fares").value
    finally:
        session.close()
    assert value == pytest.approx(expected)


def test_add_kpi_takes_one_value_per_key_over_every_row_of_a_capped_sql_node():
    session = AgentSession(DuckDBSource(uri=":memory:", tables={"big": BIG}), _model())
    ctx = SimpleNamespace(deps=session.deps)
    try:
        session.add_sql("rows", "SELECT * FROM big")
        add_kpi(ctx, "rows", "total", column="v", aggregate="sum", distinct_by="k")
        value = instance(session.ops, "total").value
    finally:
        session.close()
    assert value == sum(range(1000))


@pytest.fixture(scope="module")
def big_frame():
    import duckdb

    with duckdb.connect() as connection:
        return connection.execute(BIG).df()


def test_a_dropdown_filter_narrows_a_kpi_over_every_row_of_a_capped_sql_node(big_frame):
    session = AgentSession(DuckDBSource(uri=":memory:", tables={"big": BIG}), _model())
    ctx = SimpleNamespace(deps=session.deps)
    try:
        session.add_sql("rows", "SELECT * FROM big")
        assert len(_graph.output(session.ops, "rows", "data")) < len(big_frame)
        add_kpi(ctx, "rows", "fares", column="fare", aggregate="sum")
        add_kpi(ctx, "rows", "count", aggregate="count")
        add_kpi(ctx, "rows", "top", column="i", aggregate="max")
        add_filter_control(ctx, "rows", "g", "pick", control="multiselect", views=["fares", "count", "top"])
        widget = instance(session.ops, "pick")._widget
        assert sorted(widget.options) == list(range(7))
        seen = []
        for picked in [[], [0], [3, 5], [6], []]:
            widget.value = picked
            kept = big_frame[big_frame.g.isin(picked)] if picked else big_frame
            values = [instance(session.ops, n).value for n in ("fares", "count", "top")]
            seen.append(values)
            assert values == [pytest.approx(kept.fare.sum()), len(kept), kept.i.max()]
    finally:
        session.close()
    assert seen[0] == seen[-1] and len({v[1] for v in seen}) == 4


def _big_session():
    session = AgentSession(DuckDBSource(uri=":memory:", tables={"big": BIG}), _model())
    return session, SimpleNamespace(deps=session.deps)


def test_add_kpi_on_a_sql_node_named_like_its_source_table_aggregates_the_nodes_rows():
    session, ctx = _big_session()
    try:
        session.add_sql("big", "SELECT * FROM big WHERE g = 0")
        assert _graph.output(session.ops, "big", "pipeline").table == "big_result"
        result = add_kpi(ctx, "big", "fares", column="fare", aggregate="sum")
        value = instance(session.ops, "fares").value
    finally:
        session.close()
    expected = 1.5 * len(range(0, 1_000_000, 7))
    assert value == pytest.approx(expected)
    assert f"Total fare = {expected:,.0f}." in result


def test_a_kpi_of_a_sql_node_follows_its_result_past_the_row_ceiling():
    session, ctx = _big_session()
    try:
        session.add_sql("rows", "SELECT * FROM big WHERE i < 10")
        add_kpi(ctx, "rows", "fares", column="fare", aggregate="sum")
        add_kpi(ctx, "rows", "count", aggregate="count")
        assert instance(session.ops, "fares").value == pytest.approx(15)
        session.ops.set_config("rows", {"sql": "SELECT * FROM big"})
        assert len(_graph.output(session.ops, "rows", "data")) < 1_000_000
        values = instance(session.ops, "fares").value, instance(session.ops, "count").value
        caption = _graph.config_of(session.ops, "fares")["caption"]
    finally:
        session.close()
    assert values == (pytest.approx(1_500_000), 1_000_000)
    assert caption == "Ad hoc: Sum of fare over all rows of rows; no metric was used."


@pytest.mark.parametrize("limit", [None, 70])
def test_add_kpi_aggregates_a_filter_node_over_every_row_of_a_sql_node(limit):
    session, ctx = _big_session()
    try:
        session.add_sql("rows", "SELECT * FROM big ORDER BY i")
        if limit:
            session.ops.set_config("rows", {"limit": limit})
        session.ops.add_node(FILTER_ID, {"field": "g"}, name="g0")
        session.ops.set_inputs("g0", {"value": 0})
        session.ops.connect("rows.pipeline", "g0.pipeline")
        add_kpi(ctx, "g0", "fares", column="fare", aggregate="sum")
        value = instance(session.ops, "fares").value
    finally:
        session.close()
    assert value == pytest.approx(1.5 * len(range(0, 1_000_000, 7)))


async def test_add_kpi_reads_its_number_at_once_in_a_served_session(monkeypatch):
    from lumen.flowdash import nodes as flowdash_nodes

    # A served session runs a SQL node's query in the background and applies its result on the event loop.

    session, ctx = _big_session()
    try:
        session.add_sql("rows", "SELECT * FROM big")
        monkeypatch.setattr(flowdash_nodes, "_in_served_session", lambda: True)
        result = add_kpi(ctx, "rows", "fares", column="fare", aggregate="sum")
    finally:
        session.close()
    assert "Total fare = 1,500,000." in result


@needs_provenance
def test_a_kpi_caption_of_a_sql_value_says_how_its_query_computes_it(ctx, session):
    session.add_sql("capacity_total", "SELECT SUM(t_cap) AS capacity_kw FROM turbines")
    add_kpi(ctx, "capacity_total", "kpi")
    caption = _graph.config_of(session.ops, "kpi")["caption"]
    assert caption == "Ad hoc: Sum of t_cap over all rows of turbines, computed by its own SQL query; no metric was used."
    provenance = session.ops.editor.node_provenance(session.ops._instance("kpi"))[0]
    assert caption == f"{provenance.label}: {provenance.detail}"


def test_add_kpi_refuses_absurd_formats(ctx, session):
    session.add_sql("total", "SELECT SUM(t_cap) AS total FROM turbines")
    for kwargs, message in [({"format": ".999999999f"}, "at most 10 decimals"), ({"format": "x" * 30}, "short format spec"),
                            ({"format": "{}"}, "not a Python format spec")]:
        with pytest.raises(ModelRetry, match=re.escape(message)):
            add_kpi(ctx, "total", "kpi", **kwargs)


def test_add_kpi_says_when_the_value_is_empty(ctx, session):
    session.add_sql("none", "SELECT SUM(t_cap) AS total FROM turbines WHERE t_state = 'NY'")
    result = add_kpi(ctx, "none", "ny_total")
    assert "Total = empty. Its value is empty" in result
    assert top_bar(session) == ["ny_total"]


def test_add_kpi_rolls_back_when_placing_fails(ctx, session, monkeypatch):
    session.add_sql("rows", "SELECT * FROM turbines")
    before = structure(session)

    def refuse(name, **kwargs):
        raise OperationError("The top bar is full.")

    monkeypatch.setattr(session.ops, "place_tile", refuse)
    with pytest.raises(ModelRetry, match="The top bar is full"):
        add_kpi(ctx, "rows", "capacity", column="t_cap", aggregate="sum")
    assert structure(session) == before


# add_filter_control

def test_add_filter_control_filters_only_the_listed_views(ctx, session):
    session.add_sql("rows", "SELECT * FROM turbines")
    add_chart(ctx, "rows", "by_state", BAR)
    add_table(ctx, "rows", "all_rows")
    add_kpi(ctx, "rows", "capacity", column="t_cap", aggregate="sum")
    result = add_filter_control(ctx, "rows", "t_state", "state", control="multiselect", views=["by_state", "capacity"])

    assert result == (
        "Placed multi-select 'state' (3 options from 't_state', starting with nothing picked, so every row shows) "
        "and filter node 'state_filter' on 'rows'.t_state, "
        "fed by 'state'. Reading 'state_filter' now: 'by_state', 'capacity' (through 'capacity_agg'). "
        "Unfiltered: 'all_rows'."
    )
    wired = edges(session)
    assert {("rows.pipeline", "state_filter.pipeline"), ("state.selected", "state_filter.value"),
            ("state_filter.filtered", "by_state.pipeline"), ("state_filter.filtered", "capacity_agg.pipeline"),
            ("rows.pipeline", "all_rows.pipeline")} <= wired
    assert _graph.component_of(session.ops, "state_filter") == FILTER_ID
    assert _graph.config_of(session.ops, "state")["default_options"] == ["CA", "IA", "TX"]
    tiles = [t["node"] for t in graph(session)["layout"]["tiles"] if t.get("visible", True) and "area" not in t]
    assert tiles.index("state") == tiles.index("by_state") - 1

    assert instance(session.ops, "capacity").value == 8500
    instance(session.ops, "state")._widget.value = ["TX"]
    assert instance(session.ops, "capacity").value == 4500
    assert len(_graph.output(session.ops, "rows", "data")) == 5


def test_add_filter_control_select_range_and_date_range(ctx, session):
    session.add_sql("rows", "SELECT * FROM turbines")
    add_table(ctx, "rows", "table_a")
    add_table(ctx, "rows", "table_b")
    add_table(ctx, "rows", "table_c")
    picked = add_filter_control(ctx, "rows", "t_state", "state", views=["table_a"])
    years = add_filter_control(ctx, "rows", "p_year", "years", control="range", views=["table_b"])
    dates = add_filter_control(ctx, "rows", "t_date", "dates", control="date_range", views=["table_c"])

    full = "starting on the full span, which leaves out only rows without a value"
    assert ("Placed multi-select of one pick 'state' (3 options from 't_state', starting with nothing picked, "
            "so every row shows)") in picked
    assert f"Placed range 'years' (1999 to 2015 from 'p_year', {full})" in years
    assert f"Placed date range 'dates' (1999-05-01 to 2015-08-30 from 't_date', {full})" in dates
    assert _graph.component_of(session.ops, "state") == "Widgets/MultiChoice"
    assert _graph.config_of(session.ops, "state")["max_items"] == 1
    assert _graph.config_of(session.ops, "years")["default_start"] == 1999
    assert _graph.config_of(session.ops, "dates")["default_period"] == "all"
    assert len(_graph.output(session.ops, "state_filter", "filtered").data) == 5
    instance(session.ops, "state")._widget.value = ["TX"]
    assert len(_graph.output(session.ops, "state_filter", "filtered").data) == 2
    assert len(_graph.output(session.ops, "years_filter", "filtered").data) == 5
    assert len(_graph.output(session.ops, "dates_filter", "filtered").data) == 5


def test_add_filter_control_select_starts_on_an_explicit_value(ctx, session):
    session.add_sql("rows", "SELECT * FROM turbines")
    add_table(ctx, "rows", "table_a")
    result = add_filter_control(ctx, "rows", "t_state", "state", value="TX")
    assert "(3 options from 't_state', starting on 'TX', so it filters from the start)" in result
    assert _graph.component_of(session.ops, "state") == "Widgets/Select"
    assert _graph.config_of(session.ops, "state")["default_options"] == ["TX", "CA", "IA"]
    assert len(_graph.output(session.ops, "state_filter", "filtered").data) == 2


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"column": "t_state", "value": "NY"}, "'NY' is not a value of 't_state'"),
    ({"column": "t_state", "control": "multiselect", "value": "TX"}, "a multiselect starts with nothing picked"),
    ({"column": "p_year", "control": "range", "value": 2010}, "a range starts on the full span"),
    ({"column": "t_state", "control": "range"}, "'t_state' holds string values; a range control takes a numeric column"),
    ({"column": "p_year", "control": "date_range"}, "a date_range control takes a date column"),
    ({"column": "t_state", "views": ["other_table"]}, "'other_table' does not read 'rows''s rows"),
    ({"column": "t_state", "views": ["missing"]}, "No node 'missing'"),
    ({"column": "nope"}, "'rows' has no column 'nope'"),
    ({"column": "t_state", "name": "rows"}, "'rows' is taken"),
])
def test_add_filter_control_refusals(ctx, session, kwargs, message):
    session.add_sql("rows", "SELECT * FROM turbines")
    session.add_sql("other", "SELECT * FROM penguins")
    add_table(ctx, "rows", "rows_table")
    add_table(ctx, "other", "other_table")
    before = structure(session)
    with pytest.raises(ModelRetry, match=re.escape(message)):
        add_filter_control(ctx, "rows", **{"name": "pick", **kwargs})
    assert structure(session) == before


def test_add_filter_control_refuses_a_column_of_too_many_values(ctx, session, monkeypatch):
    from lumen.agent import controls

    monkeypatch.setattr(controls, "MAX_OPTIONS", 2)
    session.add_sql("rows", "SELECT * FROM turbines")
    add_table(ctx, "rows", "rows_table")
    with pytest.raises(ModelRetry, match="more than 2 values"):
        add_filter_control(ctx, "rows", "t_state", "pick")


def test_add_filter_control_rolls_back_when_rewiring_fails(ctx, session, monkeypatch):
    session.add_sql("rows", "SELECT * FROM turbines")
    add_chart(ctx, "rows", "by_state", BAR)
    add_table(ctx, "rows", "all_rows")
    before = structure(session)
    connect = session.ops.connect

    def failing(source, target, transform=None):
        if target == "all_rows.pipeline" and source.startswith("pick_filter"):
            raise OperationError("The input is taken.")
        return connect(source, target, transform)

    monkeypatch.setattr(session.ops, "connect", failing)
    with pytest.raises(ModelRetry, match="The input is taken"):
        add_filter_control(ctx, "rows", "t_state", "pick", views=["by_state", "all_rows"])
    assert structure(session)[:2] == before[:2]


def test_a_second_local_filter_filters_only_its_own_views(ctx, session):
    session.add_sql("rows", "SELECT * FROM turbines")
    add_table(ctx, "rows", "table_a")
    add_table(ctx, "rows", "table_b")
    add_filter_control(ctx, "rows", "p_year", "years", control="range", views=["table_a", "table_b"])
    result = add_filter_control(ctx, "rows", "t_state", "state", control="multiselect", views=["table_b"])

    assert "Reading 'state_filter' now: 'table_b'." in result
    wired = edges(session)
    assert ("rows.pipeline", "years_filter.pipeline") in wired
    assert {("years_filter.filtered", "state_filter.pipeline"), ("state_filter.filtered", "table_b.pipeline"),
            ("years_filter.filtered", "table_a.pipeline")} <= wired
    instance(session.ops, "state")._widget.value = ["TX"]
    assert len(_graph.output(session.ops, "years_filter", "filtered").data) == 5
    assert len(_graph.output(session.ops, "state_filter", "filtered").data) == 2


def test_a_local_filter_on_views_that_read_different_inputs_gets_a_filter_node_for_each(ctx, session):
    session.add_sql("rows", "SELECT * FROM turbines")
    add_table(ctx, "rows", "table_a")
    add_table(ctx, "rows", "table_b")
    add_table(ctx, "rows", "table_c")
    add_filter_control(ctx, "rows", "p_year", "years", control="range", views=["table_a", "table_b"])
    result = add_filter_control(ctx, "rows", "t_state", "state", control="multiselect", views=["table_b", "table_c"])

    assert "filter nodes 'state_filter', 'state_filter_2' (one per input the views read)" in result
    wired = edges(session)
    assert {("years_filter.filtered", "state_filter.pipeline"), ("rows.pipeline", "state_filter_2.pipeline"),
            ("state_filter.filtered", "table_b.pipeline"), ("state_filter_2.filtered", "table_c.pipeline"),
            ("years_filter.filtered", "table_a.pipeline"), ("state.selected", "state_filter_2.value")} <= wired


def test_add_filter_control_refuses_a_second_filter_on_the_same_column(ctx, session):
    session.add_sql("rows", "SELECT * FROM turbines")
    add_table(ctx, "rows", "table_a")
    add_filter_control(ctx, "rows", "t_state", "state", control="multiselect")
    before = structure(session)
    with pytest.raises(ModelRetry, match=re.escape("'table_a' is already filtered on 't_state' by 'state_filter'")):
        add_filter_control(ctx, "rows", "t_state", "state_2", views=["table_a"])
    assert structure(session) == before


def test_add_filter_control_refuses_a_column_the_global_filter_bar_filters(ctx, session):
    if not callable(getattr(session.ops, "add_filter", None)):
        pytest.skip("panel-flowdash without global filters in DashboardOps")
    session.add_sql("rows", "SELECT * FROM turbines")
    add_table(ctx, "rows", "table_a")
    session.ops.add_filter("t_state")
    before = structure(session)
    with pytest.raises(ModelRetry, match=re.escape("The global filter bar already filters on 't_state'")):
        add_filter_control(ctx, "rows", "t_state", "state", views=["table_a"])
    assert structure(session) == before
    assert "years" in add_filter_control(ctx, "rows", "p_year", "years", control="range")


def test_add_kpi_says_it_does_not_follow_the_global_filters(ctx, session):
    session.add_sql("capacity", "SELECT SUM(t_cap) AS capacity FROM turbines")
    assert add_kpi(ctx, "capacity", "capacity_kpi").endswith("KPI cards do not follow the global filter bar's filters; "
                                                             "tell the user if the dashboard has one.")
    doc = controls_toolset().tools["add_kpi"].tool_def
    assert "does not follow the global filter bar" in doc.description
    decimals = doc.parameters_json_schema["properties"]["decimals"]["anyOf"][0]
    assert (decimals["minimum"], decimals["maximum"]) == (0, 10)


# add_metric and add_table

async def test_add_metric_is_offered_with_or_without_metrics():
    seen = []

    def respond(messages, info):
        seen.append({tool.name for tool in info.function_tools})
        return ModelResponse(parts=[TextPart("ok")])

    metrics = read_metrics(parse_project(PROJECT, "flowdash.yaml"))
    for kwargs in ({}, {"metrics": metrics}):
        model = FunctionModel(respond)
        session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model, **kwargs)
        try:
            await session.ask("Hi")
        finally:
            session.close()
    assert "add_metric" in seen[0] and seen[0] == seen[1]
    doc = authoring_toolset().tools["add_metric"].tool_def.description
    assert "cannot create one" in doc and "add_sql" in doc


async def test_add_metric_refuses_in_an_app_without_metrics(ctx, session):
    result = await add_metric(ctx, "capacity", "capacity")
    assert result == "Refused: this app defines no metrics, so there is none to place. Compute the number with add_sql."
    assert "capacity" not in _graph.nodes(session.ops)


def test_the_agent_fingerprint_does_not_depend_on_the_apps_metrics(monkeypatch):
    from lumen.agent import evals

    metrics = read_metrics(parse_project(PROJECT, "flowdash.yaml"))
    plain = evals.agent_fingerprint.__wrapped__()
    monkeypatch.setattr(evals, "_reference_source", lambda: DuckDBSource(uri=":memory:", tables=TABLES))
    real = evals.AgentSession
    monkeypatch.setattr(evals, "AgentSession", lambda *a, **k: real(*a, metrics=metrics, **k))
    assert evals.agent_fingerprint.__wrapped__() == plain


def test_add_table_takes_number_formatters(ctx, session):
    session.add_sql("rows", "SELECT p_name, t_cap, t_hh FROM turbines")
    placed = add_table(ctx, "rows", "table", formatters={"t_cap": "number", "t_hh": {"type": "number", "decimals": 1}})
    assert placed == "Placed table 'table' of 'rows' (formatters)."
    assert _graph.config_of(session.ops, "table")["spec"]["formatters"] == {
        "t_cap": "number", "t_hh": {"type": "number", "decimals": 1},
    }


@pytest.mark.parametrize(("formatters", "message"), [
    ({"t_cap": "decimal"}, "The formatter of 't_cap' is 'decimal'; use one of number, money"),
    ({"t_cap": {"type": "plaintext", "thousandSeparator": True}},
     'The plaintext formatter of \'t_cap\' takes no params, not \'thousandSeparator\'; for thousands separators and '
     'decimals use {"type": "number", "decimals": 1}.'),
    ({"t_cap": {"type": "money", "precision": 1}}, "takes symbol, symbolAfter, decimals"),
    ({"t_cap": {"type": "number", "decimals": 1.5}}, "decimals of 't_cap' is 1.5"),
])
def test_add_table_refuses_formatters_it_cannot_apply(ctx, session, formatters, message):
    session.add_sql("rows", "SELECT p_name, t_cap FROM turbines")
    with pytest.raises(ModelRetry, match=re.escape(message)):
        add_table(ctx, "rows", "table", spec={"formatters": formatters})


def test_add_table_formatters_are_a_closed_vocabulary():
    schema = authoring_toolset().tools["add_table"].tool_def.parameters_json_schema
    formatter = schema["properties"]["formatters"]["anyOf"][0]["additionalProperties"]["anyOf"][0]
    assert formatter["enum"] == ["number", "money", "plaintext", "textarea", "progress", "star", "tickCross", "traffic", "color"]
    data = schema["properties"]["data"]["description"]
    assert "a node name, not a source table" in data


# The prompt names only tools the agent has.

def _surface(monkeypatch, **kwargs):
    given = {}
    real = assistant.create_flowdash_agent
    monkeypatch.setattr(assistant, "create_flowdash_agent", lambda *a, **k: given.update(k) or real(*a, **k))
    (create_explore_agent if kwargs.pop("explore", False) else create_agent)("test", **kwargs)
    offered = {name for toolset in given["toolsets"] for name in toolset.tools}
    fd = given["tools"]
    offered |= {tool.__name__ for tool in FLOWDASH_TOOLS} if fd is None else set(fd)
    return given["instructions"], offered


TOOL_NAMES = {tool.__name__ for tool in FLOWDASH_TOOLS} | {
    name for toolset in (data_toolset(), authoring_toolset(), controls_toolset()) for name in toolset.tools
} | {"add_filter", "set_title", "arrange_tiles", "add_node"}


# The command palette actions a served FlowDash app adds to its agent, which FlowDash's own
# instructions for an agent with every dashboard tool name; a headless run has no palette.
APP_ACTIONS = {"list_commands", "run_command"}
# Words shaped like a tool name: a verb the tools start with, then snake_case.
TOOL_LIKE = re.compile(
    r"\b(?:add|set|remove|place|arrange|list|describe|distinct|run|get|edit|report|connect|disconnect|"
    r"declare|bind|open|ask|group|ungroup|replace|validate|preview|save)_[a-z_]*[a-z]\b"
)


def prompt_tool_problems(instructions: str, tools: set[str]) -> list[str]:
    """What the instructions name that the agent cannot call: unknown tool-like names and tools it lacks."""
    problems = []
    for word in sorted(set(TOOL_LIKE.findall(instructions)) | (set(re.findall(r"\b\w+\b", instructions)) & TOOL_NAMES)):
        if word in APP_ACTIONS:
            continue
        if word not in TOOL_NAMES:
            problems.append(f"{word} (no such tool)")
        elif word not in tools:
            problems.append(f"{word} (not offered)")
    return problems


def test_prompt_lint_catches_typos_and_tools_the_agent_lacks():
    tools = {"add_sql", "add_kpi"}
    assert prompt_tool_problems("Show it with add_kpis.", tools) == ["add_kpis (no such tool)"]
    assert prompt_tool_problems("Show it with add_kpi.", {"add_sql"}) == ["add_kpi (not offered)"]
    assert prompt_tool_problems("Write SQL with add_sql; use snake_case names.", tools) == []


@pytest.mark.parametrize("kwargs", [
    {}, {"dashboard_tools": "data"}, {"dashboard_tools": "layout"}, {"dashboard_tools": None},
    {"explore": True},
], ids=["default", "data", "layout", "all", "explore"])
def test_instructions_name_only_tools_the_agent_has(monkeypatch, kwargs):
    text, offered = _surface(monkeypatch, **kwargs)
    problems = prompt_tool_problems(text, offered)
    assert not problems, f"The instructions name tools the agent cannot call: {', '.join(problems)}"


def test_layout_agent_gets_kpi_and_filter_tools_and_a_shorter_prompt(monkeypatch):
    layout, offered = _surface(monkeypatch, dashboard_tools="layout")
    _, data_tools = _surface(monkeypatch, dashboard_tools="data")
    assert {"add_kpi", "add_filter_control"} <= offered
    assert not {"add_kpi", "add_filter_control"} & data_tools
    assert "add_kpi" in layout and "add_filter_control" in layout
    assert ("add_filter" in offered) == assistant.HAS_ADD_FILTER
    assert (assistant.GLOBAL_FILTER in layout) == assistant.HAS_ADD_FILTER
    assert (assistant.LOCAL_FILTER in layout) != assistant.HAS_ADD_FILTER
    # FlowDash's own instructions send filters to add_filter; Lumen's only add the local case.
    assert layout.count("filter bar") <= 1
    if not assistant.HAS_ADD_FILTER:
        assert "filter bar" not in layout and "global" not in layout.lower()
    assert assistant.SELECTION_INSTRUCTIONS not in layout
    assert len(layout[layout.index("## Laying out the dashboard"):]) <= 450


# Playtest turns, replayed with a scripted model on the layout agent.

def _scripted(*steps):
    requests = []

    def respond(messages, info):
        requests.append(messages[-1])
        step = steps[len(requests) - 1]
        if isinstance(step, str):
            return ModelResponse(parts=[TextPart(step)])
        return ModelResponse(parts=[ToolCallPart(step[0], step[1], tool_call_id=f"call-{len(requests)}")])

    return FunctionModel(respond, model_name="scripted"), requests


def _layout_session(*steps):
    model, requests = _scripted(*steps)
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model,
                           agent=create_agent(model, dashboard_tools="layout"))
    return session, requests


async def test_playtest_kpis_requested_by_name_show_in_the_top_bar():
    # Playtest A s3 and C s3: "KPIs for total turbines, capacity in MW and avg hub height", which ended
    # as hidden SQL nodes or Vega text charts, never as cards in the top bar.
    sql = ("SELECT COUNT(*) AS total_turbines, SUM(t_cap) / 1000.0 AS capacity_mw, AVG(t_hh) AS avg_hub_height "
           "FROM turbines")
    session, _ = _layout_session(
        ("add_sql", {"source": "data", "name": "headline", "sql": sql}),
        ("add_kpi", {"data": "headline", "name": "kpi"}),
        "Added three KPI cards in the top bar.",
    )
    try:
        turn = await session.ask("KPIs for total turbines, capacity in MW and avg hub height")
    finally:
        session.close()
    assert turn.error is None
    shown = yaml.safe_load(turn.graph)
    top = [t["node"] for t in shown["layout"]["tiles"] if t.get("area") == "top"]
    assert top == ["kpi_total_turbines", "kpi_capacity_mw", "kpi_avg_hub_height"]
    assert all(shown["nodes"][name]["component"] == KPI_ID for name in top)
    assert not any(node["component"] == "Lumen/view" for node in shown["nodes"].values())


async def test_playtest_project_capacity_once_per_project():
    # Playtest A s6 and B s5: SUM(p_cap) over turbine rows counted each project once per turbine; told
    # that p_cap is a project's capacity, the model takes one value per project.
    session, _ = _layout_session(
        ("add_sql", {"source": "data", "name": "turbine_rows", "sql": "SELECT * FROM projects"}),
        ("add_kpi", {"data": "turbine_rows", "name": "capacity", "column": "p_cap", "aggregate": "sum", "unit": "MW",
                     "distinct_by": "p_name"}),
        "Done.",
    )
    try:
        turn = await session.ask("Add a KPI of total project capacity; p_cap is the project's capacity on each turbine")
    finally:
        session.close()
    import duckdb

    expected = duckdb.sql(f"SELECT SUM(c) FROM (SELECT MAX(p_cap) AS c FROM ({PROJECTS}) GROUP BY p_name)").fetchall()[0][0]
    assert f"Total p cap = {expected:,.0f} MW." in turn.tool_results["call-2"]


async def test_playtest_local_filter_for_one_chart():
    # Playtest A s5 and s7: "add a filter for island" and "a dropdown so I can pick the year" were impossible.
    session, _ = _layout_session(
        ("add_sql", {"source": "data", "name": "penguin_rows", "sql": "SELECT * FROM penguins"}),
        ("add_chart", {"data": "penguin_rows", "name": "mass_by_species", "spec": {
            "mark": "bar", "encoding": {"x": {"field": "species", "type": "nominal"},
                                        "y": {"aggregate": "mean", "field": "body_mass_g", "type": "quantitative"}}}}),
        ("add_table", {"data": "penguin_rows", "name": "penguin_table"}),
        ("add_filter_control", {"data": "penguin_rows", "column": "island", "name": "island",
                                "views": ["mass_by_species"]}),
        "Added an island dropdown that filters the chart only.",
    )
    try:
        turn = await session.ask("Add an island dropdown, only for the mass chart")
        island = session.ops.node_config("island")
    finally:
        session.close()
    assert "Reading 'island_filter' now: 'mass_by_species'. Unfiltered: 'penguin_table'." in turn.tool_results["call-4"]
    assert island["default_options"] == ["Biscoe", "Dream", "Torgersen"]


async def test_playtest_metric_is_refused_without_metrics():
    # Playtest C s1 t2 and s3 t3: add_metric was called with invented metrics in an app that has none.
    session, _ = _layout_session(
        ("add_metric", {"metric": "Turbines", "name": "turbines"}),
        ("add_sql", {"source": "data", "name": "turbine_count", "sql": "SELECT COUNT(*) AS turbines FROM turbines"}),
        ("add_kpi", {"data": "turbine_count", "name": "turbines_kpi", "label": "Turbines"}),
        "Added the turbine count.",
    )
    try:
        turn = await session.ask("Show the number of turbines")
    finally:
        session.close()
    assert turn.tool_results["call-1"].startswith("Refused: this app defines no metrics")
    assert "Turbines = 5." in turn.tool_results["call-3"]


async def test_playtest_number_formatter_retry_is_meaningful():
    # Playtest B s7: 'number' was refused and the retry, plaintext with thousandSeparator, did nothing.
    session, requests = _layout_session(
        ("add_sql", {"source": "data", "name": "rows", "sql": "SELECT p_name, t_cap FROM turbines"}),
        ("add_table", {"data": "rows", "name": "table", "spec": {
            "formatters": {"t_cap": {"type": "plaintext", "thousandSeparator": True}}}}),
        ("add_table", {"data": "rows", "name": "table", "formatters": {"t_cap": {"type": "number", "decimals": 0}}}),
        "Added the table.",
    )
    try:
        await session.ask("A table of turbine capacity with thousands separators")
        spec = session.ops.node_config("table")["spec"]
    finally:
        session.close()
    assert retried(requests[2].parts[0], '{"type": "number", "decimals": 1}')
    assert spec == {"formatters": {"t_cap": {"type": "number", "decimals": 0}}}


def test_layout_role_takes_flowdash_layout_tools_with_report_tools():
    from panel_flowdash.agent import tools as fd_tools

    if not hasattr(fd_tools, "LAYOUT_TOOLS"):
        pytest.skip("panel-flowdash without LAYOUT_TOOLS")
    layout = assistant.DASHBOARD_TOOLS["layout"]
    assert set(fd_tools.REPORT_TOOLS) <= set(layout)
    assert {"add_filter", "set_title", "place_tile", "arrange_tiles"} <= set(layout)


async def test_local_add_filter_points_to_add_filter_control():
    from panel_flowdash.agent import tools as fd_tools

    if not hasattr(fd_tools, "register_local_filter"):
        pytest.skip("panel-flowdash without register_local_filter")
    layout = create_agent("test", dashboard_tools="layout")
    data = create_agent("test")
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), _model())
    try:
        local = await fd_tools.add_filter(SimpleNamespace(deps=session.deps, agent=layout), "t_state", scope="local")
        other = await fd_tools.add_filter(SimpleNamespace(deps=session.deps, agent=data), "t_state", scope="local")
    finally:
        session.close()
    assert "Build a local filter with add_filter_control instead." in str(local)
    # Another agent in the same process, e.g. FlowDash's own, has no add_filter_control.
    assert "add_filter_control" not in str(other) and "No tool here builds one" in str(other)


async def test_playtest_local_add_filter_in_a_run_points_to_add_filter_control():
    if not assistant.HAS_ADD_FILTER:
        pytest.skip("panel-flowdash without add_filter")
    session, _ = _layout_session(
        ("add_filter", {"column": "t_state", "scope": "local"}),
        "Use add_filter_control.",
    )
    try:
        turn = await session.ask("A state filter for one chart")
    finally:
        session.close()
    assert "Build a local filter with add_filter_control instead." in str(turn.tool_results["call-1"])


def test_kpi_and_filter_tools_run_alone():
    assert all(tool.sequential for tool in controls_toolset().tools.values())
