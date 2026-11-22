import re

from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("duckdb")

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
TABLES = {"turbines": TURBINES, "penguins": PENGUINS}
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

def test_add_kpi_reuses_the_value_of_a_one_cell_sql_node(ctx, session):
    session.add_sql("capacity", "SELECT SUM(t_cap) / 1000.0 AS capacity_mw FROM turbines")
    result = add_kpi(ctx, "capacity", "capacity_kpi", label="Installed capacity", unit="MW")

    assert result.startswith("Placed 1 KPI card(s) in the top bar")
    assert "'capacity_kpi': Installed capacity = 8.50 MW." in result
    assert ("capacity.value", "capacity_kpi.value") in edges(session)
    assert top_bar(session) == ["capacity_kpi"]
    config = _graph.config_of(session.ops, "capacity_kpi")
    assert config == {"label": "Installed capacity", "format": ",.2f", "prefix": "", "suffix": " MW"}
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
    ({"data": "rows", "column": "p_cap", "aggregate": "sum"},
     "sum of 'p_cap' over 'rows' would double count: 'p_cap' repeats on every row of each 'p_name' (5 rows, 3"),
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


def test_add_kpi_sums_a_project_column_once_per_project(ctx, session):
    session.add_sql("projects", "SELECT DISTINCT p_name, p_cap FROM turbines")
    assert "Total p cap = 600." in add_kpi(ctx, "projects", "project_capacity", column="p_cap", aggregate="sum")


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
        "Placed multiselect 'state' (3 options from 't_state') and filter node 'state_filter' on 'rows'.t_state, "
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

    assert "Placed select 'state' (3 options from 't_state', starting on 'CA')" in picked
    assert "Placed range 'years' (1999 to 2015 from 'p_year', all selected)" in years
    assert "Placed date_range 'dates' (1999-05-01 to 2015-08-30 from 't_date', all selected)" in dates
    assert _graph.component_of(session.ops, "state") == "Widgets/Select"
    assert _graph.config_of(session.ops, "years")["default_start"] == 1999
    assert _graph.config_of(session.ops, "dates")["default_period"] == "all"
    assert len(_graph.output(session.ops, "state_filter", "filtered").data) == 2
    assert len(_graph.output(session.ops, "years_filter", "filtered").data) == 5
    assert len(_graph.output(session.ops, "dates_filter", "filtered").data) == 5


@pytest.mark.parametrize(("kwargs", "message"), [
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


# add_metric and add_table

async def test_add_metric_is_offered_only_in_an_app_with_metrics():
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
    assert "add_metric" not in seen[0] and "add_metric" in seen[1]
    doc = authoring_toolset().tools["add_metric"].tool_def.description
    assert "cannot create one" in doc and "add_sql" in doc


async def test_add_metric_refuses_in_an_app_without_metrics(ctx):
    with pytest.raises(ModelRetry, match=re.escape("This app defines no metrics; compute it with add_sql.")):
        await add_metric(ctx, "capacity", "capacity")


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


@pytest.mark.parametrize("kwargs", [
    {}, {"dashboard_tools": "data"}, {"dashboard_tools": "layout"}, {"dashboard_tools": None},
    {"explore": True},
], ids=["default", "data", "layout", "all", "explore"])
def test_instructions_name_only_tools_the_agent_has(monkeypatch, kwargs):
    text, offered = _surface(monkeypatch, **kwargs)
    named = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", text)) & TOOL_NAMES
    assert named <= offered, f"named but not offered: {sorted(named - offered)}"


def test_layout_agent_gets_kpi_and_filter_tools_and_a_shorter_prompt(monkeypatch):
    layout, offered = _surface(monkeypatch, dashboard_tools="layout")
    data, data_tools = _surface(monkeypatch, dashboard_tools="data")
    assert {"add_kpi", "add_filter_control"} <= offered
    assert not {"add_kpi", "add_filter_control"} & data_tools
    assert "add_kpi" in layout and "add_filter_control" in layout
    assert ("add_filter" in offered) == assistant.HAS_ADD_FILTER
    assert ("global filter bar with add_filter." in layout) == assistant.HAS_ADD_FILTER
    assert assistant.SELECTION_INSTRUCTIONS not in layout
    # The layout prompt was 5,111 characters before add_kpi and add_filter_control.
    assert len(layout) <= 5111 and len(data) <= 4066


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


async def test_playtest_summed_project_capacity_is_refused_and_fixed():
    # Playtest A s6 and B s5: SUM(p_cap) over turbine rows counted each project once per turbine.
    session, requests = _layout_session(
        ("add_sql", {"source": "data", "name": "turbine_rows", "sql": "SELECT * FROM turbines"}),
        ("add_kpi", {"data": "turbine_rows", "name": "capacity", "column": "p_cap", "aggregate": "sum", "unit": "MW"}),
        ("add_sql", {"source": "data", "name": "projects", "sql": "SELECT DISTINCT p_name, p_cap FROM turbines"}),
        ("add_kpi", {"data": "projects", "name": "capacity", "column": "p_cap", "aggregate": "sum", "unit": "MW"}),
        "Total capacity is 600 MW.",
    )
    try:
        turn = await session.ask("Add a KPI of total project capacity")
    finally:
        session.close()
    assert retried(requests[2].parts[0], "would double count")
    assert "Total p cap = 600 MW." in turn.tool_results["call-4"]


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


async def test_playtest_metric_is_not_offered_without_metrics():
    # Playtest C s1 t2 and s3 t3: add_metric was called with invented metrics in an app that has none.
    session, requests = _layout_session(
        ("add_metric", {"metric": "Turbines", "name": "turbines"}),
        ("add_sql", {"source": "data", "name": "turbine_count", "sql": "SELECT COUNT(*) AS turbines FROM turbines"}),
        ("add_kpi", {"data": "turbine_count", "name": "turbines_kpi", "label": "Turbines"}),
        "Added the turbine count.",
    )
    try:
        turn = await session.ask("Show the number of turbines")
    finally:
        session.close()
    assert retried(requests[1].parts[0], "Unknown tool name: 'add_metric'")
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
    create_agent("test", dashboard_tools="layout")
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), _model())
    try:
        result = await fd_tools.add_filter(SimpleNamespace(deps=session.deps), "t_state", scope="local")
    finally:
        session.close()
    assert "Build a local filter with add_filter_control instead." in str(result)
