"""
The scenarios, drawn from the assistant playtests A, B and C (Nov 2026) and
grouped by the failure themes of the improvement plan. Most start from a
seeded dashboard so that a turn measures one behaviour; the builds start
from the bare source.
"""
from __future__ import annotations

from typing import Any

import yaml

from ...flowdash import SQL_ID, VIEW_ID
from ...flowdash.table_view import TABLE_VIEW_ID
from . import Scenario, Turn
from .checks import (
    KPI_ID, PERCENT_SCALES, adhoc_label, answer_names, axes, before_views,
    chart_mark, chart_title, charts_titled, color, dashboard_title, declines,
    edited_in_place, filter_on, has_view, heading_on_top, kept, kpi_units,
    kpi_values, kpis, legend, max_bars, max_requests, multi_series,
    no_duplicates, no_hidden_leftovers, no_new_nodes, no_unused, removed,
    same_height, same_row, selection_link, shows_row, shows_series,
    shows_value, tables_rounded, titled_node, unchanged, uses_metric, width,
    years_unformatted,
)
from .fixtures import STATES as STATE_ROWS

# -- reference queries on the fixtures --------------------------------------------------------------

TURBINES = "SELECT COUNT(*) FROM turbines"
PROJECTS = "SELECT COUNT(DISTINCT p_name) FROM turbines"
STATES = "SELECT COUNT(DISTINCT t_state) FROM turbines"
CAPACITY_MW = "SELECT SUM(t_cap) / 1000.0 FROM turbines"
HUB_HEIGHT = "SELECT AVG(t_hh) FROM turbines"
CAPACITY_BY_YEAR = "SELECT p_year, SUM(t_cap) / 1000.0 FROM turbines GROUP BY 1"
CAPACITY_BY_STATE_GW = "SELECT t_state, SUM(t_cap) / 1e6 FROM turbines GROUP BY 1"
CAPACITY_BY_REGION = (
    "SELECT region, SUM(t_cap) / 1000.0 FROM turbines JOIN states ON turbines.t_state = states.state GROUP BY 1"
)
TEXAS_SHARE = "SELECT SUM(CASE WHEN t_state = 'TX' THEN t_cap ELSE 0 END) / SUM(t_cap) FROM turbines"
LARGEST_PROJECTS = "SELECT p_name, MAX(p_cap) FROM turbines GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
CALIFORNIA = "SELECT COUNT(*) FROM quakes WHERE place LIKE '%, CA' OR place ILIKE '%california%'"
CALIFORNIA_STRONGEST = (
    "SELECT place, mag FROM quakes WHERE place LIKE '%, CA' OR place ILIKE '%california%' ORDER BY mag DESC LIMIT 1"
)
TOP_STATE = "SELECT t_state FROM turbines GROUP BY 1 ORDER BY SUM(t_cap) DESC LIMIT 1"
STATE_NAMES = {code: name for code, name, _ in STATE_ROWS}
LAST_MONTH = "SELECT COUNT(*) FROM quakes WHERE time >= TIMESTAMP '2025-12-01' AND time < TIMESTAMP '2026-01-01'"
PENGUINS = "SELECT COUNT(*) FROM penguins"
BODY_MASS = "SELECT AVG(body_mass_g) FROM penguins"
FLIPPER = "SELECT AVG(flipper_length_mm) FROM penguins"
# Of the penguins whose sex is recorded, or of all of them: either reading of "share of females" counts.
FEMALE_SHARE = (
    "SELECT COUNT(*) FILTER (WHERE sex = 'female') / COUNT(sex) FROM penguins",
    "SELECT COUNT(*) FILTER (WHERE sex = 'female') / COUNT(*) FROM penguins",
)

COUNT = (1.0,)

# The questions of the numbers scenarios that the metrics scenarios ask again.
TEXAS_SHARE_PROMPT = "What percentage of national installed capacity is in Texas?"
CAPACITY_BY_YEAR_PROMPT = "Chart installed wind capacity in MW by project year."
CAPACITY_BY_REGION_PROMPT = "Chart total installed capacity by US census region, using the states table."

# -- seeded dashboards ------------------------------------------------------------------------------


def bar(x: str, y: str, **spec: Any) -> dict[str, Any]:
    return {"spec": {"mark": {"type": "bar"}, "encoding": {
        "x": {"field": x, "type": "nominal", "sort": "-y"}, "y": {"field": y, "type": "quantitative"}}, **spec}}


def dashboard(queries: dict[str, str], views: dict[str, tuple[str, str, dict]],
              tiles: list[tuple[str, float] | tuple[str, float, dict]], title: str | None = None) -> str:
    """
    A seed dashboard: SQL nodes on the source ``data``; views, each as
    ``(component, sql node, config)`` (a KPI reads the node's value, others
    its pipeline); the tiles shown, as ``(node, width[, more])``. Sources
    and SQL nodes are hidden.
    """
    nodes: dict[str, Any] = {"data": {"component": "Lumen/source", "config": {"ref": "data"}}}
    edges = []
    for name, sql in queries.items():
        nodes[name] = {"component": SQL_ID, "config": {"sql": sql, "table": name}}
        edges.append({"from": "data.source", "to": f"{name}.source"})
    for name, (component, upstream, config) in views.items():
        nodes[name] = {"component": component, "config": config}
        port = "value" if component == KPI_ID else "pipeline"
        edges.append({"from": f"{upstream}.{port}", "to": f"{name}.{port}"})
    layout = [{"node": tile[0], "width": tile[1], **(tile[2] if len(tile) > 2 else {})} for tile in tiles]
    layout += [{"node": name, "visible": False} for name in ["data", *queries]]
    doc = {"flowdash": 1, "title": title or "Untitled", "nodes": nodes, "edges": edges, "layout": {"tiles": layout}}
    return yaml.safe_dump(doc, sort_keys=False)


STATE_CAPACITY_SQL = "SELECT t_state, SUM(t_cap) / 1000.0 AS capacity_mw FROM turbines GROUP BY t_state ORDER BY capacity_mw DESC LIMIT 10"
WIND = dashboard(
    {"state_capacity": STATE_CAPACITY_SQL,
     "projects": "SELECT p_name, t_state, p_year, MAX(p_cap) AS capacity_mw FROM turbines GROUP BY ALL ORDER BY capacity_mw DESC LIMIT 20"},
    {"state_capacity_chart": (VIEW_ID, "state_capacity", bar("t_state", "capacity_mw")),
     "projects_table": (TABLE_VIEW_ID, "projects", {"spec": {}})},
    [("state_capacity_chart", 100), ("projects_table", 100)],
)

WIND_FOUR = dashboard(
    {"capacity_by_year": "SELECT p_year, SUM(t_cap) / 1000.0 AS capacity_mw FROM turbines GROUP BY p_year ORDER BY p_year",
     "state_capacity": STATE_CAPACITY_SQL,
     "manufacturers": "SELECT t_manu, COUNT(*) AS turbines FROM turbines GROUP BY t_manu ORDER BY turbines DESC LIMIT 10",
     "projects": "SELECT p_name, t_state, MAX(p_cap) AS capacity_mw FROM turbines GROUP BY ALL ORDER BY capacity_mw DESC LIMIT 10"},
    {"capacity_over_time": (VIEW_ID, "capacity_by_year", {"spec": {"title": "Capacity over time", "mark": "line", "encoding": {
        "x": {"field": "p_year", "type": "ordinal"}, "y": {"field": "capacity_mw", "type": "quantitative"}}}}),
     "top_states": (VIEW_ID, "state_capacity", bar("t_state", "capacity_mw", title="Top states")),
     "top_manufacturers": (VIEW_ID, "manufacturers", bar("t_manu", "turbines", title="Top manufacturers")),
     "largest_projects": (TABLE_VIEW_ID, "projects", {"spec": {}})},
    [("capacity_over_time", 100), ("top_states", 100), ("top_manufacturers", 100), ("largest_projects", 100)],
)

WIND_KPIS_LAST = dashboard(
    {"state_capacity": STATE_CAPACITY_SQL,
     "turbine_count": "SELECT COUNT(*) AS turbines FROM turbines",
     "project_count": "SELECT COUNT(DISTINCT p_name) AS projects FROM turbines",
     "total_capacity": "SELECT SUM(t_cap) / 1000.0 AS capacity_mw FROM turbines"},
    {"state_capacity_chart": (VIEW_ID, "state_capacity", bar("t_state", "capacity_mw", title="Capacity by state")),
     "kpi_turbines": (KPI_ID, "turbine_count", {"label": "Turbines"}),
     "kpi_projects": (KPI_ID, "project_count", {"label": "Projects"}),
     "kpi_capacity": (KPI_ID, "total_capacity", {"label": "Capacity (MW)"})},
    [("state_capacity_chart", 100), ("kpi_turbines", 100), ("kpi_projects", 100), ("kpi_capacity", 100)],
)

WIND_YEARLY = dashboard(
    {"turbines_per_year": "SELECT p_year, COUNT(*) AS turbines FROM turbines GROUP BY p_year ORDER BY p_year"},
    {"turbines_per_year_chart": (VIEW_ID, "turbines_per_year", {"spec": {"title": "Turbines per year", "mark": {"type": "bar"},
     "encoding": {"x": {"field": "p_year", "type": "ordinal"}, "y": {"field": "turbines", "type": "quantitative"}}}})},
    [("turbines_per_year_chart", 100)],
)

WIND_HELPERS = dashboard(
    {"state_capacity": STATE_CAPACITY_SQL,
     **{f"helper_{i}": sql for i, sql in enumerate([
         "SELECT COUNT(*) AS n FROM turbines", "SELECT MAX(p_year) AS latest FROM turbines",
         "SELECT t_manu, COUNT(*) AS n FROM turbines GROUP BY 1", "SELECT AVG(t_hh) AS hub FROM turbines",
         "SELECT DISTINCT t_state FROM turbines", "SELECT p_name, COUNT(*) AS n FROM turbines GROUP BY 1"], 1)}},
    {"state_capacity_chart": (VIEW_ID, "state_capacity", bar("t_state", "capacity_mw", title="Capacity by state"))},
    [("state_capacity_chart", 100)],
)

PENGUIN_CHARTS = dashboard(
    {"species_counts": "SELECT species, COUNT(*) AS penguins FROM penguins GROUP BY species",
     "measurements": "SELECT species, island, bill_length_mm, bill_depth_mm, body_mass_g FROM penguins WHERE bill_length_mm IS NOT NULL",
     "mass_by_species": "SELECT species, AVG(body_mass_g) AS body_mass_g FROM penguins GROUP BY species"},
    {"species_counts_chart": (VIEW_ID, "species_counts", bar("species", "penguins", title="Penguins per species")),
     "bill_scatter": (VIEW_ID, "measurements", {"spec": {"title": "Bill length vs depth", "mark": {"type": "point"}, "encoding": {
         "x": {"field": "bill_length_mm", "type": "quantitative"}, "y": {"field": "bill_depth_mm", "type": "quantitative"},
         "color": {"field": "species", "type": "nominal"}}}}),
     "mass_chart": (VIEW_ID, "mass_by_species", bar("species", "body_mass_g", title="Body mass by species"))},
    [("species_counts_chart", 50), ("bill_scatter", 50), ("mass_chart", 100)],
)

PENGUIN_YEARS = dashboard(
    {"penguin_mass": "SELECT species, year, AVG(body_mass_g) AS body_mass_g FROM penguins GROUP BY species, year"},
    {"mass_chart": (VIEW_ID, "penguin_mass", bar("species", "body_mass_g", title="Body mass by species"))},
    [("mass_chart", 100)],
)

QUAKES_MONTHLY = dashboard(
    {"quakes_monthly": "SELECT DATE_TRUNC('month', time) AS month, COUNT(*) AS quakes FROM quakes GROUP BY 1 ORDER BY 1"},
    {"quakes_monthly_chart": (VIEW_ID, "quakes_monthly", {"spec": {"title": "Earthquakes per month", "mark": {"type": "line"},
     "encoding": {"x": {"field": "month", "type": "temporal"}, "y": {"field": "quakes", "type": "quantitative"}}}})},
    [("quakes_monthly_chart", 100)],
)

NO_NOTE = no_new_nodes(kinds=["markdown", "heading"], id="no_note")

SCENARIOS: list[Scenario] = [
    # -- KPIs ----------------------------------------------------------------------------------------
    Scenario("kpi_turbine_row", "kpis", "turbines", playtest="A s3 t1, C s1 t2", turns=[Turn(
        "Show KPIs for total turbines, total installed capacity in MW and average hub height in a row at the top.",
        [kpis(3), kpi_values(TURBINES, CAPACITY_MW, HUB_HEIGHT), no_hidden_leftovers(), max_requests()])]),
    Scenario("kpi_penguin_units", "kpis", "penguins", playtest="A s5 t1-t2", turns=[
        Turn("Show KPIs for the number of penguins, average body mass and average flipper length.",
             [kpis(3), kpi_values(PENGUINS, BODY_MASS, FLIPPER, scales=COUNT), max_requests()]),
        Turn("Make the KPIs show their units: grams for body mass and millimetres for flipper length.",
             [kpi_units("g", "mm"), edited_in_place(kind="kpi", workstream="W2"), no_duplicates()]),
    ]),
    Scenario("kpi_quakes_row", "kpis", "quakes", playtest="B s3 t1", turns=[Turn(
        "Give me a row of KPIs: total quakes, average magnitude and largest magnitude.",
        [kpis(3), kpi_values("SELECT COUNT(*) FROM quakes", "SELECT AVG(mag) FROM quakes", "SELECT MAX(mag) FROM quakes",
                             scales=COUNT), max_requests()])]),
    Scenario("kpi_top_bar", "kpis", "turbines", seed=WIND, playtest="C s3 t3", turns=[Turn(
        "Add three number cards in the top bar: the number of turbines, of projects and of states.",
        [kpis(3), kpi_values(TURBINES, PROJECTS, STATES, scales=COUNT), no_hidden_leftovers(), no_duplicates(),
         max_requests()])]),
    Scenario("wind_build", "kpis", "turbines", playtest="C s1 t1", turns=[Turn(
        "Build me a dashboard about US wind turbines: a few headline numbers, installed capacity over time, "
        "the top states and the top manufacturers.",
        [kpis(2), before_views("kpi"), has_view("chart", 3, workstream="W1"),
         shows_series(CAPACITY_BY_YEAR), max_bars(20), charts_titled(), max_requests()])]),
    # -- filters -------------------------------------------------------------------------------------
    Scenario("filter_state", "filters", "turbines", seed=WIND, playtest="C s1 t5, A s2 t3", turns=[Turn(
        "Add a filter by state that applies to the chart and the table.", [filter_on("t_state"), NO_NOTE])]),
    Scenario("filter_island", "filters", "penguins", seed=PENGUIN_CHARTS, playtest="A s5 t3", turns=[Turn(
        "Add a filter for island so I can pick one.", [filter_on("island"), NO_NOTE])]),
    Scenario("dropdown_year", "filters", "penguins", seed=PENGUIN_YEARS, playtest="A s7 t2", turns=[Turn(
        "Add a dropdown so I can pick the year.", [filter_on("year"), NO_NOTE])]),
    Scenario("click_to_filter", "filters", "turbines", seed=WIND, playtest="C s4 t2", turns=[Turn(
        "When I click a state bar in the chart, filter the table to that state.",
        [selection_link("state_capacity_chart", "projects_table"), NO_NOTE])]),
    # -- titles --------------------------------------------------------------------------------------
    Scenario("rename_dashboard", "titles", "turbines", seed=WIND, playtest="A s3 t4", turns=[Turn(
        "Rename the dashboard to Wind Fleet Overview.", [dashboard_title("Wind Fleet Overview"), NO_NOTE])]),
    Scenario("title_on_top", "titles", "penguins", seed=PENGUIN_CHARTS, playtest="C s2 t2", turns=[Turn(
        "Add a title and a short explanation of what the dashboard shows at the top.",
        [dashboard_title(), heading_on_top()])]),
    Scenario("retitle_chart", "titles", "turbines", seed=WIND, playtest="C s3 t7", turns=[Turn(
        "Retitle the capacity chart to 'Top states by capacity'.",
        [chart_title("state_capacity_chart", "Top states by capacity"), edited_in_place("state_capacity_chart")])]),
    Scenario("rename_table", "titles", "turbines", seed=WIND, playtest="C s4 t5", turns=[Turn(
        "Rename the table to 'Largest projects'.",
        [titled_node("projects_table", "Largest projects"), edited_in_place("projects_table", kind="table")])]),
    # -- layout --------------------------------------------------------------------------------------
    Scenario("two_columns", "layout", "turbines", seed=WIND_FOUR, playtest="C s1 t6", turns=[Turn(
        "Make the layout two columns: capacity over time next to the top states, and top manufacturers "
        "next to the largest projects table.",
        [same_row("capacity_over_time", "top_states"), same_row("top_manufacturers", "largest_projects"),
         width("capacity_over_time", 50), width("largest_projects", 50)])]),
    Scenario("kpis_to_top", "layout", "turbines", seed=WIND_KPIS_LAST, playtest="C s1 t3", turns=[Turn(
        "The KPIs are below the chart at full width. Put them in one row at the top.",
        [before_views("kpi"), same_row("kpi_turbines", "kpi_projects", "kpi_capacity"),
         no_new_nodes(kinds=["kpi", "chart"], workstream="W2")])]),
    Scenario("table_left_chart_right", "layout", "turbines", seed=WIND, playtest="C s4 t3", turns=[Turn(
        "Put the table on the left and the chart on the right, same height.",
        [same_row("projects_table", "state_capacity_chart"), same_height("projects_table", "state_capacity_chart")])]),
    Scenario("wider_scatter", "layout", "penguins", seed=PENGUIN_CHARTS, playtest="C s2 t4, C s3 t5", turns=[Turn(
        "Make the scatter plot twice as wide as the species count chart, keeping them side by side.",
        [same_row("species_counts_chart", "bill_scatter", ordered=False, guard=True), width("bill_scatter", 66.67),
         width("species_counts_chart", 33.33)])]),
    # -- edits ---------------------------------------------------------------------------------------
    Scenario("line_chart", "edits", "turbines", seed=WIND_YEARLY, playtest="A s3 t2, A s6 t2", turns=[Turn(
        "Change the turbines per year chart to a line chart.",
        [chart_mark("line", "turbines_per_year_chart"), edited_in_place("turbines_per_year_chart"), max_requests(4, workstream="W2")])]),
    Scenario("swap_axes", "edits", "penguins", seed=PENGUIN_CHARTS, playtest="C s2 t5", turns=[Turn(
        "On the scatter, swap the axes.",
        [axes("bill_scatter", x="bill_depth_mm", y="bill_length_mm"), edited_in_place("bill_scatter"), max_requests(4, workstream="W2")])]),
    Scenario("rolling_average", "edits", "quakes", seed=QUAKES_MONTHLY, playtest="B s1 t2", turns=[Turn(
        "Add a 3-month rolling average line to that chart.",
        [multi_series("quakes_monthly_chart"), legend(), edited_in_place("quakes_monthly_chart")])]),
    Scenario("units_gw", "edits", "turbines", seed=WIND, playtest="C s1 t8", turns=[Turn(
        "Show capacity in GW instead of MW in the chart.",
        [shows_series(CAPACITY_BY_STATE_GW, scales=COUNT, workstream="W2"), edited_in_place("state_capacity_chart"),
         no_new_nodes(kinds=["chart", "table"], workstream="W2")])]),
    Scenario("make_green", "edits", "turbines", seed=WIND, playtest="C s3 t8", turns=[Turn(
        "Make the capacity chart green.", [color("state_capacity_chart", "green"), edited_in_place("state_capacity_chart"),
                                           max_requests(4, workstream="W2")])]),
    # -- wrong-number traps --------------------------------------------------------------------------
    Scenario("capacity_by_year", "numbers", "turbines", playtest="A s6 t1, C s1 t1/t4", turns=[Turn(
        "Chart installed wind capacity in MW by project year.",
        [shows_series(CAPACITY_BY_YEAR), years_unformatted(), charts_titled()])]),
    Scenario("capacity_by_region", "numbers", "turbines_states", playtest="B s5 t1", turns=[Turn(
        "Chart total installed capacity by US census region, using the states table.",
        [shows_series(CAPACITY_BY_REGION), has_view("chart", workstream="W4")])]),
    Scenario("texas_share", "numbers", "turbines", metrics="turbines", playtest="B s5 t2", turns=[Turn(
        "What percentage of national installed capacity is in Texas?",
        [shows_value(TEXAS_SHARE, scales=PERCENT_SCALES)])]),
    Scenario("largest_projects", "numbers", "turbines", playtest="A s7, A s8", turns=[Turn(
        "Show a table of the 10 largest wind projects with their state and capacity in MW.",
        [shows_series(LARGEST_PROJECTS, where="shown"), has_view("table", workstream="W4"), tables_rounded()])]),
    Scenario("california_quakes", "numbers", "quakes", playtest="B s2 t3", turns=[Turn(
        "How many earthquakes were in California? Show the 10 strongest in a table.",
        [shows_value(CALIFORNIA, scales=COUNT), shows_row(CALIFORNIA_STRONGEST, id="strongest"),
         has_view("table", workstream="W4")])]),
    Scenario("last_month", "numbers", "quakes", playtest="B s8 t1", turns=[Turn(
        "How many earthquakes happened last month?", [shows_value(LAST_MONTH, scales=COUNT)])]),
    # -- chart design --------------------------------------------------------------------------------
    Scenario("capacity_by_state", "charts", "turbines", playtest="A s1, A s2", turns=[Turn(
        "Chart installed capacity by state.",
        [has_view("chart", workstream="W5"), max_bars(20), charts_titled(), years_unformatted()])]),
    Scenario("quakes_by_type", "charts", "quakes", playtest="B s1 t1, B s3 t2", turns=[Turn(
        "Chart the number of earthquakes per quarter by event type.",
        [has_view("chart", workstream="W5"), charts_titled(), legend()])]),
    # -- claims vs graph -----------------------------------------------------------------------------
    Scenario("email_schedule", "claims", "penguins", seed=PENGUIN_CHARTS, playtest="C s2 t10", turns=[Turn(
        "Email this dashboard as a PDF to my team every Monday morning.", [declines(), unchanged()])]),
    Scenario("why_question", "claims", "turbines", seed=WIND, playtest="C s3 t6, C s1 t4", turns=[Turn(
        "Why is Texas at the top of the capacity chart? Is it really the state with the most capacity?",
        [answer_names(TOP_STATE, STATE_NAMES), no_new_nodes()])]),
    # -- runaway turns -------------------------------------------------------------------------------
    Scenario("penguin_batch", "runaway", "penguins", playtest="A s9", turns=[Turn(
        "Build a complete penguin dashboard in one go: four KPI cards (count, mean body mass in kg, mean flipper "
        "length, share of females), four charts (species counts, body mass distribution by species, flipper vs "
        "body mass scatter, counts per island per year) and a table of all records.",
        [kpis(4), kpi_values(PENGUINS, FLIPPER, FEMALE_SHARE, scales=(1.0, 100.0)), has_view("chart", 4, workstream="W1"),
         has_view("table", workstream="W1"), max_requests()])]),
    Scenario("remove_helpers", "runaway", "turbines", seed=WIND_HELPERS, playtest="C s3 t4", turns=[Turn(
        "Remove the helper queries nothing uses anymore.",
        [no_unused(), kept("state_capacity_chart"), removed("helper_1", workstream="W6"), max_requests(4)])]),
    # -- metrics first -------------------------------------------------------------------------------
    # The numbers questions again in an app whose project file defines a metric for each: the turn
    # must answer with that metric, and a hand-written query with the same numbers earns nothing.
    Scenario("texas_share_metric", "metrics", "turbines_states", metrics="wind", playtest="B s5 t2", turns=[Turn(
        TEXAS_SHARE_PROMPT, [uses_metric("capacity_share", TEXAS_SHARE, key="TX", scales=PERCENT_SCALES)])]),
    Scenario("capacity_by_year_metric", "metrics", "turbines_states", metrics="wind", playtest="A s6 t1", turns=[Turn(
        CAPACITY_BY_YEAR_PROMPT,
        [uses_metric("capacity", CAPACITY_BY_YEAR, where="shown"), has_view("chart", workstream="W4"),
         years_unformatted(), charts_titled()])]),
    Scenario("capacity_by_region_metric", "metrics", "turbines_states", metrics="wind", playtest="B s5 t1", turns=[Turn(
        CAPACITY_BY_REGION_PROMPT,
        [uses_metric("regional_capacity", CAPACITY_BY_REGION, where="shown"), has_view("chart", workstream="W4")])]),
    # And in an app without metrics: the answer is ad hoc, and says how it was computed and that no
    # metric was used.
    Scenario("texas_share_adhoc", "metrics", "turbines", playtest="B s5 t2", turns=[Turn(
        TEXAS_SHARE_PROMPT, [shows_value(TEXAS_SHARE, scales=PERCENT_SCALES), adhoc_label()])]),
    Scenario("capacity_by_year_adhoc", "metrics", "turbines", playtest="A s6 t1", turns=[Turn(
        CAPACITY_BY_YEAR_PROMPT, [shows_series(CAPACITY_BY_YEAR), has_view("chart", workstream="W4"), adhoc_label()])]),
    Scenario("capacity_by_region_adhoc", "metrics", "turbines_states", playtest="B s5 t1", turns=[Turn(
        "Chart total installed capacity by US census region, using the states table.",
        [shows_series(CAPACITY_BY_REGION), has_view("chart", workstream="W4"), adhoc_label()])]),
]


# The checks that failed in the recorded baseline (gpt-6-luna on the integration branch with the
# assistant plan's W1, W2 and W3), each with the workstream of the improvement plan expected to
# fix it. Replaying the cassettes reports them as xfail, and as XPASS once a change makes them pass.
EXPECTED_FAILURES: dict[str, dict[str, str]] = {
    "kpi_turbine_row": {"1/kpis": "W1"},
    "kpi_penguin_units": {"1/kpis": "W1", "2/units": "W1"},
    "kpi_quakes_row": {"1/kpis": "W1"},
    "kpi_top_bar": {"1/kpis": "W1"},
    "wind_build": {"1/kpis": "W1", "1/chart_titles": "W5"},
    "click_to_filter": {"1/selection": "W1"},
    "rename_table": {"1/node_title": "W1", "1/in_place": "W2"},
    "rolling_average": {"1/legend": "W5"},
    "capacity_by_year": {"1/chart_titles": "W5"},
    "largest_projects": {"1/numbers": "W4"},
    "california_quakes": {"1/value": "W4", "1/strongest": "W4"},
    "last_month": {"1/value": "W4"},
    "capacity_by_state": {"1/max_bars": "W5", "1/chart_titles": "W5"},
    "quakes_by_type": {"1/chart_titles": "W5"},
    "why_question": {"1/no_new_nodes": "W6"},
    "penguin_batch": {"1/kpis": "W1", "1/requests": "W6"},
    "kpis_to_top": {"1/row_kpi_turbines": "W3"},
    # The ad hoc label is lumen#2416's and M2's.
    "texas_share_adhoc": {"1/adhoc_label": "M2/#2416"},
    "capacity_by_year_adhoc": {"1/adhoc_label": "M2/#2416"},
    "capacity_by_region_adhoc": {"1/adhoc_label": "M2/#2416"},
}

for _scenario in SCENARIOS:
    _scenario.xfail = dict(EXPECTED_FAILURES.get(_scenario.name, {}))

BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
