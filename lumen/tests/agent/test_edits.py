from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

import yaml

from pydantic_ai import ModelRetry
from pydantic_ai.messages import (
    ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession, _graph
from lumen.agent.authoring import (
    add_chart, add_sql, add_table, add_text, update_chart,
)
from lumen.agent.data import list_tables
from lumen.agent.edits import (
    STATE_HEADER, STATE_NODES, dashboard_state, merge_chart, merge_patch,
    name_notes, similar,
)
from lumen.flowdash import VIEW_ID
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "quakes": (
        "SELECT * FROM (VALUES (DATE '2024-01-01', 10, 4.1), (DATE '2024-02-01', 14, 4.3), (DATE '2024-03-01', 9, 4.0), "
        "(DATE '2024-04-01', 16, 4.6)) AS t(month, quakes, magnitude)"
    ),
}
MONTHLY = "SELECT month, quakes FROM quakes ORDER BY month"
ROLLING = (
    "SELECT month, quakes, AVG(quakes) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) "
    "AS quakes_3_month_avg FROM quakes ORDER BY month"
)
BAR = {"mark": {"type": "bar", "tooltip": True}, "encoding": {
    "x": {"field": "month", "type": "temporal", "title": "Month"},
    "y": {"field": "quakes", "type": "quantitative", "title": "Quakes"},
    "color": {"value": "#4c78a8"},
}}


def script(*responses):
    """A model answering each request with the next response, recording the messages of each request."""
    requests = []

    def respond(messages, info):
        requests.append(list(messages))
        return ModelResponse(parts=list(responses[len(requests) - 1]))

    return FunctionModel(respond, model_name="scripted"), requests


@pytest.fixture
def session():
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), FunctionModel(lambda m, i: ModelResponse([TextPart("")])))
    yield session
    session.close()


@pytest.fixture
def ctx(session):
    return SimpleNamespace(deps=session.deps)


def tiles(session):
    return {tile["node"]: tile for tile in yaml.safe_load(session.graph())["layout"]["tiles"]}


def grid(session):
    return [tile["node"] for tile in yaml.safe_load(session.graph())["layout"]["tiles"] if tile.get("visible") is not False]


async def test_state_lists_each_node_with_its_contents_and_tile(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    add_chart(ctx, "monthly_quakes", "monthly_quakes_chart", {**BAR, "title": "Quakes per month"})
    add_table(ctx, "monthly_quakes", "monthly_quakes_table")
    add_text(ctx, "heading", "quakes_title", "Earthquakes", size="Title")
    await add_sql(ctx, "data", "strongest_quake", "SELECT MAX(magnitude) AS magnitude FROM quakes")
    session.ops.place_tile("monthly_quakes_table", visible=False)

    state = dashboard_state(session.ops)

    assert state.splitlines() == [
        f"{STATE_HEADER} Title 'Untitled'; 6 nodes in reading order (top bar, grid rows, sidebar, then nodes without a visible tile):",
        "- monthly_quakes_chart (chart: bar x=month, y=quakes; title 'Quakes per month'; reads monthly_quakes); grid row 1, 100%",
        "- quakes_title (heading: 'Earthquakes'); grid row 2, 100%, 96px",
        "- monthly_quakes_table (table: default options; reads monthly_quakes); HIDDEN tile",
        "- data (source); no tile, read by monthly_quakes, strongest_quake",
        "- monthly_quakes (SQL: SELECT month, quakes FROM quakes ORDER BY month; reads data); no tile, read by "
        "monthly_quakes_chart, monthly_quakes_table",
        "- strongest_quake (SQL: SELECT MAX(magnitude) AS magnitude FROM quakes; reads data); no tile, nothing reads it",
    ]


async def test_state_is_deterministic_and_bounded(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    for index in range(STATE_NODES + 10):
        add_chart(ctx, "monthly_quakes", f"quakes_chart_{index}", BAR)

    first, second = dashboard_state(session.ops), dashboard_state(session.ops)

    assert first == second
    lines = first.splitlines()
    assert len(lines) == STATE_NODES + 2
    assert lines[-1] == "- and 12 more nodes; list_tables lists them."
    assert max(map(len, lines)) < 200


async def test_state_costs_less_than_listing_the_nodes(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    add_chart(ctx, "monthly_quakes", "monthly_quakes_chart", BAR)
    assert len(dashboard_state(session.ops)) < len(await list_tables(ctx))


async def test_each_user_turn_comes_with_the_state_and_keeps_the_cached_prefix():
    model, requests = script(
        [ToolCallPart("add_sql", {"source": "data", "name": "monthly_quakes", "sql": MONTHLY}, tool_call_id="a1")],
        [TextPart("Done.")],
        [TextPart("Still done.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    try:
        await session.ask("Quakes per month")
        await session.ask("And now?")
    finally:
        session.close()

    def states(messages):
        return [part.content for m in messages if isinstance(m, ModelRequest) for part in m.parts
                if isinstance(part, UserPromptPart) and part.content.startswith(STATE_HEADER)]

    first, tool_turn, second = requests
    (prompt_parts,) = [m.parts for m in first if isinstance(m, ModelRequest)]
    assert [type(p) for p in prompt_parts] == [UserPromptPart, UserPromptPart]
    assert prompt_parts[0].content == "Quakes per month" and prompt_parts[1].content.startswith(STATE_HEADER)
    assert "monthly_quakes" not in prompt_parts[1].content
    # A tool result gets no state; the next turn does, after the history as it was sent.
    assert states(tool_turn) == states(first)
    assert second[:len(tool_turn)] == tool_turn
    assert len(states(second)) == 2 and "- monthly_quakes (SQL" in states(second)[1]


async def test_add_chart_with_a_taken_name_changes_the_chart_in_its_tile(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    add_chart(ctx, "monthly_quakes", "quakes_chart", BAR)
    add_chart(ctx, "monthly_quakes", "magnitude_chart", {"mark": "bar", "encoding": {"x": {"field": "month", "type": "temporal"}, "y": {"field": "quakes", "type": "quantitative"}}})
    session.ops.place_tile("quakes_chart", width=70)
    session.ops.place_tile("magnitude_chart", width=30)
    before = grid(session)

    text = add_chart(ctx, "monthly_quakes", "quakes_chart", {**BAR, "mark": "line"})

    assert text == "Changed chart 'quakes_chart' (line) of 'monthly_quakes'."
    assert grid(session) == before and tiles(session)["quakes_chart"]["width"] == 70
    assert _graph.config_of(session.ops, "quakes_chart")["spec"]["mark"] == "line"
    assert len(_graph.nodes(session.ops, VIEW_ID)) == 2


async def test_add_chart_with_a_taken_name_can_show_other_data(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    await add_sql(ctx, "data", "monthly_quakes_rolling", ROLLING)
    add_chart(ctx, "monthly_quakes", "quakes_chart", BAR)

    # The new spec charts a column only the new data has.
    layered = {"encoding": BAR["encoding"], "layer": [
        {"mark": "bar"}, {"mark": "line", "encoding": {"y": {"field": "quakes_3_month_avg", "type": "quantitative"}}}]}
    text = add_chart(ctx, "monthly_quakes_rolling", "quakes_chart", layered)

    assert text == "Changed chart 'quakes_chart' (bar, line) of 'monthly_quakes_rolling'."
    assert _graph.feeding_all(session.ops, "quakes_chart.pipeline") == ["monthly_quakes_rolling.pipeline"]
    with pytest.raises(ModelRetry, match="quakes_3_month_avg"):
        add_chart(ctx, "monthly_quakes", "quakes_chart", layered)
    assert _graph.feeding_all(session.ops, "quakes_chart.pipeline") == ["monthly_quakes_rolling.pipeline"]


async def test_add_tools_refuse_a_name_taken_by_another_kind(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    with pytest.raises(ModelRetry, match="'monthly_quakes' is already a SQL node, so no chart can take its name"):
        add_chart(ctx, "monthly_quakes", "monthly_quakes", BAR)
    add_chart(ctx, "monthly_quakes", "quakes_chart", BAR)
    with pytest.raises(ModelRetry, match="already a chart node, so no table"):
        add_table(ctx, "monthly_quakes", "quakes_chart")
    with pytest.raises(ModelRetry, match="already a chart node, so no heading"):
        add_text(ctx, "heading", "quakes_chart", "Quakes")


async def test_add_sql_with_a_taken_name_changes_its_query_and_keeps_its_readers(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    add_chart(ctx, "monthly_quakes", "quakes_chart", BAR)

    text = await add_sql(ctx, "data", "monthly_quakes", ROLLING)

    assert text.startswith("Changed SQL node 'monthly_quakes': 4 rows x 3 columns")
    assert "quakes_3_month_avg" in _graph.config_of(session.ops, "monthly_quakes")["sql"]
    assert _graph.reading(session.ops, "monthly_quakes.pipeline") == ["quakes_chart"]
    assert _graph.nodes(session.ops) == ["data", "monthly_quakes", "quakes_chart"]


async def test_add_sql_with_a_taken_name_leaves_the_node_as_it_was_on_an_error(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    with pytest.raises(ModelRetry, match="Binder Error"):
        await add_sql(ctx, "data", "monthly_quakes", "SELECT nope FROM quakes")
    assert _graph.config_of(session.ops, "monthly_quakes")["sql"].split() == ["SELECT", "month,", "quakes", "FROM", "quakes", "ORDER", "BY", "month"]


async def test_add_sql_with_a_taken_name_can_read_another_result(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    await add_sql(ctx, "data", "busy_months", "SELECT month FROM quakes WHERE quakes > 12")

    text = await add_sql(ctx, "monthly_quakes", "busy_months", "SELECT month FROM monthly_quakes WHERE quakes > 12")

    assert text.startswith("Changed SQL node 'busy_months'")
    assert _graph.feeding_all(session.ops, "busy_months.source") == ["monthly_quakes.result"]
    with pytest.raises(ModelRetry, match="cannot read its own result"):
        await add_sql(ctx, "busy_months", "busy_months", "SELECT * FROM busy_months")


async def test_add_table_and_add_text_with_a_taken_name_change_the_node(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    add_table(ctx, "monthly_quakes", "quakes_table")
    add_text(ctx, "heading", "quakes_title", "Quakes")

    assert add_table(ctx, "monthly_quakes", "quakes_table", {"page_size": 5}) == (
        "Changed table 'quakes_table' of 'monthly_quakes' (page_size).")
    assert add_text(ctx, "heading", "quakes_title", "Earthquakes per month") == "Changed heading 'quakes_title'."
    assert _graph.config_of(session.ops, "quakes_title")["text"] == "Earthquakes per month"
    assert len(_graph.nodes(session.ops)) == 4


def test_merge_patch_merges_objects_removes_nulls_and_keeps_mark_properties():
    assert merge_patch({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"b": 5, "c": None}, "e": [1]}) == {"a": {"b": 5}, "d": 3, "e": [1]}
    assert merge_chart(BAR, {"mark": "line"})["mark"] == {"type": "line", "tooltip": True}
    assert merge_chart(BAR, {"mark": {"point": True}})["mark"] == {"type": "bar", "tooltip": True, "point": True}
    assert "color" not in merge_chart(BAR, {"encoding": {"color": None}})["encoding"]
    assert merge_chart(BAR, {"encoding": {"y": {"title": "Count"}}})["encoding"]["y"] == {
        "field": "quakes", "type": "quantitative", "title": "Count"}


def test_merge_chart_layers_a_single_chart_and_shares_its_axes():
    line = {"mark": {"type": "line", "strokeDash": [4, 2]}, "encoding": {"y": {"field": "quakes_3_month_avg", "type": "quantitative"}}}

    spec = merge_chart(BAR, None, [line])

    assert "mark" not in spec
    assert spec["encoding"] == {"x": BAR["encoding"]["x"], "y": BAR["encoding"]["y"]}
    assert spec["layer"] == [{"mark": {"type": "bar", "tooltip": True}, "encoding": {"color": {"value": "#4c78a8"}}}, line]
    # Layers merge by position, a single chart counting as the first.
    assert merge_chart(BAR, {"layer": [{}, line]}) == spec


def test_merge_chart_changes_a_layered_chart_where_it_defines_a_channel():
    layered = merge_chart(BAR, None, [{"mark": "rule", "encoding": {"y": {"datum": 12}}},
                                      {"mark": "line", "encoding": {"color": {"value": "orange"}}}])

    spec = merge_chart(layered, {"mark": "area", "encoding": {"color": {"value": "red"}, "x": {"title": "When"}}})

    assert [layer["mark"] for layer in spec["layer"]] == [{"type": "area", "tooltip": True}, "rule", "area"]
    assert spec["layer"][0]["encoding"]["color"] == {"value": "red"} == spec["layer"][2]["encoding"]["color"]
    assert spec["encoding"]["x"]["title"] == "When"


async def test_update_chart_changes_the_mark_in_one_call_and_returns_the_spec(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    add_chart(ctx, "monthly_quakes", "quakes_chart", BAR)

    text = update_chart(ctx, "quakes_chart", {"mark": "line", "title": "Quakes per month"})

    assert text.startswith("Changed chart 'quakes_chart' of 'monthly_quakes': line x=month, y=quakes; title 'Quakes per month'.")
    assert '"mark":{"type":"line","tooltip":true}' in text
    assert _graph.config_of(session.ops, "quakes_chart")["spec"]["encoding"] == BAR["encoding"]


async def test_update_chart_checks_the_merged_spec_and_leaves_the_chart_on_a_mistake(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    add_chart(ctx, "monthly_quakes", "quakes_chart", BAR)
    with pytest.raises(ModelRetry, match="fields the data does not have: rolling"):
        update_chart(ctx, "quakes_chart", add_layers=[{"mark": "line", "encoding": {"y": {"field": "rolling", "type": "quantitative"}}}])
    with pytest.raises(ModelRetry, match="'monthly_quakes' is a SQL node; update_chart changes charts, tables and maps"):
        update_chart(ctx, "monthly_quakes", {"mark": "line"})
    with pytest.raises(ModelRetry, match="No node 'quake_chart'. The charts, tables and maps are: quakes_chart"):
        update_chart(ctx, "quake_chart", {"mark": "line"})
    assert _graph.config_of(session.ops, "quakes_chart")["spec"] == BAR


async def test_update_chart_merges_table_options(ctx, session):
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    add_table(ctx, "monthly_quakes", "quakes_table", {"page_size": 5, "titles": {"quakes": "Quakes"}})

    text = update_chart(ctx, "quakes_table", {"titles": {"month": "Month"}})

    assert text.startswith("Changed table 'quakes_table' of 'monthly_quakes': options page_size, titles.")
    assert _graph.config_of(session.ops, "quakes_table")["spec"] == {"page_size": 5, "titles": {"quakes": "Quakes", "month": "Month"}}
    with pytest.raises(ModelRetry, match="Only a chart takes add_layers"):
        update_chart(ctx, "quakes_table", add_layers=[{"mark": "rule"}])


def test_name_notes_flag_names_that_say_what_the_node_does_not_hold():
    assert name_notes("quakes_by_month", ["month", "quakes"], MONTHLY) == []
    (note,) = name_notes("electricity_output", ["t_state", "capacity_kw"], "SELECT t_state, SUM(t_cap) AS capacity_kw FROM turbines GROUP BY 1")
    assert note.startswith("the name 'electricity_output' shares no word with its columns")
    (note,) = name_notes("iowa_turbines", ["electricity_output"], "SELECT SUM(p_cap) AS electricity_output FROM iowa_turbines")
    assert note.startswith("column 'electricity_output' is computed from p_cap")
    assert name_notes("capacity_gw", ["capacity_gw"], "SELECT SUM(t_cap) / 1000 AS capacity_gw FROM turbines") == []
    assert name_notes("turbine_count", ["turbines"], "SELECT COUNT(*) AS turbines FROM t") == []


async def test_add_sql_and_add_chart_note_a_misleading_name(ctx, session):
    text = await add_sql(ctx, "data", "electricity_output", "SELECT SUM(quakes) AS quakes FROM quakes")
    assert "\nNote: the name 'electricity_output' shares no word" in text
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    text = add_chart(ctx, "monthly_quakes", "revenue_forecast", BAR)
    assert "\nNote: the name 'revenue_forecast' shares no word with its data 'monthly_quakes' or fields (month, quakes)" in text


def test_similar_names():
    assert similar("wind_largest_projects_table", "wind_largest_projects_table_gw")
    assert similar("capacity_kpi", "kpi_capacity_gw")
    assert not similar("quakes_chart", "magnitude_table")


async def test_a_node_removed_and_placed_again_keeps_its_tile():
    model, requests = script(
        [ToolCallPart("remove_node", {"name": "quakes_chart"}, tool_call_id="r1")],
        [ToolCallPart("add_chart", {"data": "monthly_quakes", "name": "quakes_chart_line", "spec": {**BAR, "mark": "line"}}, tool_call_id="a1")],
        [TextPart("Done.")],
    )
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    ctx = SimpleNamespace(deps=session.deps)
    try:
        await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
        add_chart(ctx, "monthly_quakes", "quakes_chart", BAR)
        add_table(ctx, "monthly_quakes", "quakes_table")
        session.ops.place_tile("quakes_chart", width=60)
        turn = await session.ask("Make it a line chart")
        order = grid(session)
        width = tiles(session)["quakes_chart_line"]["width"]
    finally:
        session.close()

    removed = next(p for p in requests[1][-1].parts if isinstance(p, ToolReturnPart))
    assert "Note: its tile (grid position 0, 60% wide) goes to the next chart with a similar name" in removed.model_response_str()
    placed = next(p for p in requests[2][-1].parts if isinstance(p, ToolReturnPart))
    assert placed.content.endswith("It took the tile of the removed 'quakes_chart' (grid position 0, 60% wide).")
    assert turn.error is None and order[:2] == ["quakes_chart_line", "quakes_table"] and width == 60


async def _rolling_dashboard(model):
    session = AgentSession(DuckDBSource(uri=":memory:", tables=TABLES), model)
    ctx = SimpleNamespace(deps=session.deps)
    await add_sql(ctx, "data", "monthly_quakes", MONTHLY)
    add_chart(ctx, "monthly_quakes", "quakes_per_month_chart", {**BAR, "mark": {"type": "line", "tooltip": True}})
    return session


async def test_adding_a_rolling_average_to_that_chart_edits_its_nodes():
    # Playtest B, s1 turn 2: "Add a 3-month rolling average line to that chart." placed a second chart.
    model, requests = script(
        [ToolCallPart("add_sql", {"source": "data", "name": "monthly_quakes", "sql": ROLLING}, tool_call_id="s1"),
         ToolCallPart("update_chart", {"name": "quakes_per_month_chart", "add_layers": [
             {"mark": {"type": "line", "strokeDash": [4, 2], "color": "orange"},
              "encoding": {"y": {"field": "quakes_3_month_avg", "type": "quantitative"}}}]}, tool_call_id="u1")],
        [TextPart("Added the 3-month rolling average to the chart.")],
    )
    session = await _rolling_dashboard(model)
    try:
        turn = await session.ask("Add a 3-month rolling average line to that chart.")
        spec = _graph.config_of(session.ops, "quakes_per_month_chart")["spec"]
    finally:
        session.close()

    # In one request the query gets the column and the chart a layer showing it; the first
    # call's change lands before the second is checked.
    results = {p.tool_call_id: p.content for p in requests[1][-1].parts if isinstance(p, ToolReturnPart)}
    assert results["s1"].startswith("Changed SQL node 'monthly_quakes'")
    assert results["u1"].startswith("Changed chart 'quakes_per_month_chart' of 'monthly_quakes': layers line + line y=quakes_3_month_avg")
    assert turn.error is None and turn.nodes == []
    assert sorted(turn.changed) == ["monthly_quakes", "quakes_per_month_chart"]
    assert [layer["mark"]["type"] for layer in spec["layer"]] == ["line", "line"]
    assert spec["encoding"]["x"]["field"] == "month"


async def test_make_it_a_line_chart_is_one_call():
    # Playtest A, s6: "change the trend chart to a line chart" took two refused set_config calls.
    model, requests = script(
        [ToolCallPart("update_chart", {"name": "quakes_per_month_chart", "changes": {"mark": "bar"}}, tool_call_id="u1")],
        [TextPart("It is a bar chart now.")],
    )
    session = await _rolling_dashboard(model)
    try:
        turn = await session.ask("Make it a bar chart.")
        spec = _graph.config_of(session.ops, "quakes_per_month_chart")["spec"]
    finally:
        session.close()

    assert len(requests) == 2 and turn.error is None
    assert turn.nodes == [] and turn.changed == ["quakes_per_month_chart"]
    assert spec["mark"] == {"type": "bar", "tooltip": True} and spec["encoding"] == BAR["encoding"]


async def test_explore_cannot_change_a_dashboard_node_through_an_add_tool():
    from lumen.agent.explore import CHANGING_TOOLS, PLACING_TOOLS

    assert {"add_sql", "add_chart", "add_table", "add_map", "add_text", "add_metric", "update_chart"} <= set(CHANGING_TOOLS)
    assert "update_chart" not in PLACING_TOOLS
