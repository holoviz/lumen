import dataclasses

from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession, _graph
from lumen.agent.authoring import add_chart, add_metric
from lumen.agent.data import list_tables
from lumen.agent.edits import dashboard_state
from lumen.agent.evals import node_sql, snapshot
from lumen.flowdash import METRIC_ID
from lumen.flowdash.metrics import read_metrics
from lumen.flowdash.project import parse_project
from lumen.sources.duckdb import DuckDBSource

TABLES = {"turbines": (
    "SELECT * FROM (VALUES ('Alta', 'CA', 1999, 3000), ('Roscoe', 'TX', 2001, 1500), ('Roscoe', 'TX', 2010, 2000), "
    "('Story', 'IA', 2015, 1000)) AS t(p_name, t_state, p_year, t_cap)"
)}

PROJECT = """\
metrics:
  capacity:
    description: Installed capacity.
    source: data
    table: turbines
    measures:
      capacity_mw: {expr: "SUM(t_cap) / 1000.0", format: ",.1f", description: Rated capacity in MW}
      turbines: "COUNT(*)"
    dimensions:
      state: {column: t_state, description: Two-letter state code}
      year: p_year
    parameters:
      since: {type: integer, column: p_year, op: ">=", description: First year included}
  top_projects:
    source: data
    sql: SELECT p_name, SUM(t_cap) AS kw FROM turbines GROUP BY p_name ORDER BY kw DESC LIMIT :n
    parameters:
      n: {type: integer, default: 2, bounds: [1, 10]}
"""


@pytest.fixture
def session():
    metrics = read_metrics(parse_project(PROJECT, "flowdash.yaml"))
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES),
                           FunctionModel(lambda m, i: ModelResponse([TextPart("")])), metrics=metrics)
    yield session
    session.close()


@pytest.fixture
def ctx(session):
    return SimpleNamespace(deps=session.deps)


async def test_list_tables_lists_the_metrics(ctx):
    text = await list_tables(ctx)
    assert text.startswith("Metrics, the app's agreed numbers: when one matches the question, answer with add_metric, not SQL.")
    assert "- capacity (source data, table turbines): Installed capacity." in text
    assert "  measures: capacity_mw (Rated capacity in MW; format ,.1f), turbines" in text
    assert "  dimensions: state (Two-letter state code), year" in text
    assert "  parameters: since (First year included; integer, filters p_year >=)" in text
    assert "- top_projects (source data, SQL)" in text
    assert "  parameters: n (integer, default 2, from 1 to 10)" in text


async def test_list_tables_lists_the_metrics_before_the_nodes(ctx, session):
    await add_metric(ctx, "capacity", "capacity_by_state", ["capacity_mw"], ["state"])
    text = await list_tables(ctx)
    assert text.index("Metrics, the app's agreed numbers") < text.index("Nodes on the dashboard")


def test_the_dashboard_state_names_the_metrics(session):
    state = dashboard_state(session.ops)
    assert state.splitlines()[-1] == (
        "The app's metrics, its agreed numbers (answer with add_metric when one matches; list_tables shows their "
        "fields): capacity (Installed capacity.); top_projects."
    )


def test_the_dashboard_state_names_no_metrics_without_any():
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(lambda m, i: ModelResponse([TextPart("")])))
    try:
        assert "metrics" not in dashboard_state(session.ops)
    finally:
        session.close()


async def test_add_metric_places_a_metric_node(ctx, session):
    result = await add_metric(ctx, "capacity", "capacity_by_state", ["capacity_mw"], ["state"],
                              parameters={"since": 2000}, order_by=["-capacity_mw"])
    assert result.startswith("Placed metric node 'capacity_by_state'")
    ops = session.ops
    assert _graph.component_of(ops, "capacity_by_state") == METRIC_ID
    assert _graph.output(ops, "capacity_by_state", "data").values.tolist() == [["TX", 3.5], ["IA", 1.0]]
    assert _graph.inputs_of(ops, "capacity_by_state") == {"parameters": {"since": 2000}}
    sql = node_sql(ops, "capacity_by_state")
    assert sql.startswith('SELECT "t_state" AS "state"') and 'FROM (SELECT * FROM "turbines")' in sql

    text = await list_tables(ctx)
    assert ("- capacity_by_state (metric 'capacity'; measures capacity_mw; dimensions state; order_by -capacity_mw; "
            "parameters {'since': 2000}; columns: state, capacity_mw)") in text

    changed = await add_metric(ctx, "capacity", "capacity_by_state", ["capacity_mw"], ["state"],
                               parameters={"state": ["TX"]})
    assert changed.startswith("Changed metric node 'capacity_by_state'")
    assert _graph.output(ops, "capacity_by_state", "data").values.tolist() == [["TX", 3.5]]
    assert add_chart(ctx, "capacity_by_state", "chart", {
        "mark": "bar", "encoding": {"x": {"field": "state"}, "y": {"field": "capacity_mw"}}}).startswith("Placed chart")


async def test_add_metric_takes_sql_metric_parameters(ctx, session):
    await add_metric(ctx, "top_projects", "top", parameters={"n": 1})
    assert _graph.output(session.ops, "top", "data").values.tolist() == [["Roscoe", 3500]]


@pytest.mark.parametrize("kwargs, message", [
    ({"metric": "power"}, "No metric 'power'. The metrics are: capacity, top_projects."),
    ({"metric": "capacity", "measures": ["mw"]}, "has no measure 'mw'"),
    ({"metric": "capacity", "measures": ["turbines"], "parameters": {"since": "2000; DROP TABLE x"}}, "expected an integer"),
    ({"metric": "top_projects", "parameters": {"n": 100}}, r"outside the bounds \[1, 10\]"),
    ({"metric": "capacity", "measures": ["turbines"], "parameters": {"t_cap": 1}}, "has no parameter 't_cap'"),
    ({"metric": "top_projects", "dimensions": ["p_name"]}, "has no dimensions to pick. Place it with only its metric"),
])
async def test_add_metric_refuses_what_the_metric_does_not_take(ctx, session, kwargs, message):
    with pytest.raises(ModelRetry, match=message):
        await add_metric(ctx, name="m", **kwargs)
    assert "m" not in _graph.nodes(session.ops)


async def test_metric_turn_is_scored_by_its_rows_and_sql():
    metrics = read_metrics(parse_project(PROJECT, "flowdash.yaml"))
    calls = iter([
        [ToolCallPart("add_metric", {"metric": "capacity", "name": "by_state", "measures": ["turbines"],
                                     "dimensions": ["state"], "order_by": ["state"]})],
        [TextPart("CA 1, IA 1, TX 2")],
    ])
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES),
                           FunctionModel(lambda m, i: ModelResponse(next(calls))), metrics=metrics)
    try:
        turn = snapshot(session, await session.ask("Turbines by state?"))
    finally:
        session.close()
    assert turn.rows == [["CA", 1], ["IA", 1], ["TX", 2]]
    assert 'FROM (SELECT * FROM "turbines")' in turn.sql


async def test_add_metric_in_explore_refuses_to_change_a_dashboard_metric_node(session):
    flowdash_agent = pytest.importorskip("panel_flowdash.agent")
    if not hasattr(flowdash_agent, "ExplorationOps"):
        pytest.skip("panel-flowdash without ExplorationOps")
    session.editor.new_dashboard("Capacity")
    await add_metric(SimpleNamespace(deps=session.deps), "capacity", "by_state", ["capacity_mw"], ["state"])
    explored = flowdash_agent.ExplorationOps(session.ops)
    ctx = SimpleNamespace(deps=flowdash_agent.FlowDashDeps(explored))
    try:
        with pytest.raises(PermissionError, match="^'by_state' is a node of the dashboard.*under another name"):
            await add_metric(ctx, "capacity", "by_state", ["turbines"], ["state"])
        result = await add_metric(ctx, "capacity", "turbines_by_state", ["turbines"], ["state"])
    finally:
        explored.dispose()

    assert result.startswith("Placed metric node 'turbines_by_state'")
    assert _graph.config_of(session.ops, "by_state")["measures"] == ["capacity_mw"]
    assert [node["name"] for node in session.ops.nodes()] == ["by_state"]


async def test_add_metric_in_explore_changes_and_reports_the_copy_of_an_earlier_answer(session, monkeypatch):
    import panel_agents

    from panel_flowdash import agent_edits

    flowdash_agent = pytest.importorskip("panel_flowdash.agent")
    if not hasattr(flowdash_agent, "ExplorationOps"):
        pytest.skip("panel-flowdash without ExplorationOps")
    session.editor.new_dashboard("Capacity")
    explored = flowdash_agent.ExplorationOps(session.ops)
    ctx = SimpleNamespace(deps=flowdash_agent.FlowDashDeps(explored))
    run = {"id": "first"}
    monkeypatch.setattr(panel_agents, "current_run_id", lambda: run["id"])
    try:
        with agent_edits("first"):
            await add_metric(ctx, "capacity", "mine", ["capacity_mw"], ["state"])
        run["id"] = "second"
        with agent_edits("second"):
            changed = await add_metric(ctx, "capacity", "mine", ["turbines"], ["state"])
            again = await add_metric(ctx, "capacity", "mine", ["turbines"], ["year"])
        configs = {name: explored.node_config(name) for name in ("mine", "mine_2")}
        frame = _graph.output(explored, "mine_2", "data")
    finally:
        explored.dispose()

    assert changed.startswith("Changed metric node 'mine_2'") and again.startswith("Changed metric node 'mine_2'")
    assert configs["mine"]["measures"] == ["capacity_mw"] and configs["mine_2"]["dimensions"] == ["year"]
    assert "year" in frame.columns and "year" in again


class Vanishing(dict):
    """A live view of the metrics from which ``top_projects`` is deleted while it is read."""

    def __getitem__(self, name):
        if name == "top_projects":
            raise KeyError(name)
        return super().__getitem__(name)


class Unreadable(dict):
    def __iter__(self):
        raise OSError("/srv/app/flowdash.yaml: permission denied")


def test_the_metrics_are_read_as_one_snapshot(session, caplog):
    from lumen.flowdash import METRICS_KEY

    metrics = dict(session.ops.editor.context[METRICS_KEY])
    session.ops.editor.context[METRICS_KEY] = Vanishing(metrics)
    snapshot = _graph.metrics(session.ops)
    assert type(snapshot) is dict and list(snapshot) == ["capacity"]
    assert dashboard_state(session.ops).endswith("fields): capacity (Installed capacity.).")
    session.ops.editor.context[METRICS_KEY] = Unreadable(metrics)
    assert _graph.metrics(session.ops) == {}
    assert "Could not read the app's metrics (OSError)." in caplog.text and "/srv/app" not in caplog.text


async def test_list_tables_shows_each_metric_on_one_line_and_at_most_twenty(ctx, session):
    from lumen.agent.data import METRICS_SHOWN
    from lumen.flowdash import METRICS_KEY

    metrics = dict(session.ops.editor.context[METRICS_KEY])
    capacity = metrics["capacity"]
    many = {"capacity": dataclasses.replace(capacity, description="Installed\ncapacity.\x1b ```\n- fake: line")}
    many |= {f"metric_{i}": dataclasses.replace(capacity, name=f"metric_{i}") for i in range(30)}
    session.ops.editor.context[METRICS_KEY] = many
    text = (await list_tables(ctx)).split("\n\n")[0]
    assert "- capacity (source data, table turbines): Installed capacity.\\x1b ''' - fake: line" in text
    assert "\n- fake" not in text and "```" not in text
    assert text.count(" (source data, table turbines)") == METRICS_SHOWN
    assert text.endswith(f"- and {31 - METRICS_SHOWN} more, which add_metric places by name: "
                         + ", ".join(f"metric_{i}" for i in range(METRICS_SHOWN - 1, 30)))
