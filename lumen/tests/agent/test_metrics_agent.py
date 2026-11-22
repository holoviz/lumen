from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession, _graph
from lumen.agent.authoring import add_chart, add_metric
from lumen.agent.data import list_tables
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
    assert "Metrics, the app's agreed definitions; place one with add_metric." in text
    assert "- capacity (source data, table turbines): Installed capacity." in text
    assert "  measures: capacity_mw (Rated capacity in MW; format ,.1f), turbines" in text
    assert "  dimensions: state (Two-letter state code), year" in text
    assert "  parameters: since (First year included; integer, filters p_year >=)" in text
    assert "- top_projects (source data, SQL)" in text
    assert "  parameters: n (integer, default 2, from 1 to 10)" in text


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
