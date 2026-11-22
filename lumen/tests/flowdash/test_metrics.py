import datetime as dt
import logging
import sqlite3

import pytest

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, OperationError
from panel_flowdash.agent.ops import OutputUnavailable

from lumen.flowdash import (
    COMPONENTS, DATA_ID, FILTER_ID, METRIC_ID, METRICS_KEY, SOURCES_KEY,
    VIEW_ID, bind_source, fetch, filtered_pipeline, metric_pipeline,
    pipeline_sql,
)
from lumen.flowdash.app import build_routes
from lumen.flowdash.choices import metric_presets
from lumen.flowdash.metrics import (
    MetricError, Query, aggregate, bind_sql, check_metrics, read_metrics,
)
from lumen.flowdash.project import (
    PROJECT_FILE, ProjectError, find_project, load_project, parse_project,
)
from lumen.sources.duckdb import DuckDBSource
from lumen.sources.sqlalchemy import SQLAlchemySource

ROWS = [
    ("Alta", "CA", 1999, "1999-02-01", 3000),
    ("Roscoe", "TX", 2001, "2001-05-17", 1500),
    ("Roscoe", "TX", 2010, "2010-11-02", 2000),
    ("Horse Hollow", "TX", 2010, "2010-03-30", 2500),
    ("Story", "IA", 2015, "2015-08-09", 1000),
]

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
      online: {column: online, type: time, grains: [year, quarter, month, week, day]}
    parameters:
      since: {type: integer, default: 2000, column: p_year, op: ">="}
  top_projects:
    description: Projects ranked by capacity.
    source: data
    sql: |
      SELECT p_name, SUM(t_cap) AS capacity_kw FROM turbines
      WHERE t_state IN (:states) GROUP BY p_name ORDER BY capacity_kw DESC, p_name LIMIT :n
    parameters:
      n: {type: integer, default: 2, bounds: [1, 50]}
      states: {type: "list[string]", default: [CA, TX, IA]}
"""


def values_sql():
    rows = ", ".join(f"('{n}', '{s}', {y}, DATE '{d}', {c})" for n, s, y, d, c in ROWS)
    return f"SELECT * FROM (VALUES {rows}) AS t(p_name, t_state, p_year, online, t_cap)"


@pytest.fixture
def duck():
    return DuckDBSource(uri=":memory:", tables={"turbines": values_sql()})


@pytest.fixture
def lite(tmp_path):
    path = tmp_path / "turbines.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE turbines (p_name TEXT, t_state TEXT, p_year INTEGER, online TEXT, t_cap INTEGER)")
        conn.executemany("INSERT INTO turbines VALUES (?, ?, ?, ?, ?)", ROWS)
    return SQLAlchemySource(url=f"sqlite:///{path}")


def metrics_for(source, text=PROJECT):
    metrics, warnings = check_metrics(read_metrics(parse_project(text, PROJECT_FILE)), {SOURCES_KEY: {"data": source}})
    assert warnings == []
    return metrics


def problems(text):
    with pytest.raises(ProjectError) as info:
        read_metrics(parse_project(text, PROJECT_FILE))
    return [str(p) for p in info.value.problems]


# Loading

@pytest.mark.parametrize("text, expected", [
    ("metrics:\n  m:\n    source: data\n    table: t\n    measures:\n      x: {expr: \"t_cap +\"}\n",
     "flowdash.yaml:6:17: Measure 'x' does not parse"),
    ("metrics:\n  m:\n    source: data\n    table: t\n    measures:\n      x: t_cap\n",
     "flowdash.yaml:6:10: Measure 'x' must aggregate the rows"),
    ("metrics:\n  m:\n    source: data\n    table: t\n    measures:\n      x: \"(SELECT MAX(a) FROM other)\"\n",
     "flowdash.yaml:6:10: Measure 'x' must be one expression over the table's columns, not a query"),
    ("metrics:\n  m:\n    source: data\n    table: t\n    measures:\n      x: \"SUM(CAST(current_setting('x') AS INT))\"\n",
     "flowdash.yaml:6:10: Measure 'x' calls current_setting()"),
    ("metrics:\n  m:\n    source: data\n    table: t\n    measures: {x: \"COUNT(*)\"}\n    dimensions:\n      d: {expr: \"SUM(a)\"}\n",
     "flowdash.yaml:7:17: Dimension 'd' must not aggregate"),
    ("metrics:\n  m:\n    source: data\n    table: t\n    measures: {x: \"COUNT(*)\"}\n    parameters:\n      n: {type: integr}\n",
     "flowdash.yaml:7:17: Parameter 'n': type must be one of"),
    ("metrics:\n  m:\n    source: data\n    table: t\n    measures: {x: \"COUNT(*)\"}\n    parameters:\n      n: {type: integer, column: a, default: two}\n",
     "flowdash.yaml:7:46: Parameter 'n': the default is wrong: expected an integer, got 'two'"),
    ("metrics:\n  m:\n    source: data\n    table: t\n    measures: {x: \"COUNT(*)\"}\n    parameters:\n      n: {type: integer}\n",
     "flowdash.yaml:7:7: Parameter 'n' needs a `column` to filter by"),
    ("metrics:\n  m:\n    source: data\n    table: t\n    measures: {x: {expr: \"COUNT(*)\", format: \"%Y\"}}\n",
     "flowdash.yaml:5:46: format '%Y' is not a d3-format string"),
    ("metrics:\n  m:\n    source: data\n    table: t\n    measures: {x: \"COUNT(*)\"}\n    dimensions:\n      d: {column: a, grains: [year]}\n",
     "flowdash.yaml:7:22: Dimension 'd' has grains but is not `type: time`"),
    ("metrics:\n  m:\n    source: data\n    sql: SELECT a FROM t LIMIT :n\n",
     "flowdash.yaml:4:10: The query uses :n, which is not a parameter of the metric."),
    ("metrics:\n  m:\n    source: data\n    sql: SELECT a FROM t\n    parameters:\n      n: {type: integer, default: 1}\n",
     "flowdash.yaml:6:7: Parameter 'n' is not used as :n in the query."),
    ("metrics:\n  m:\n    source: data\n    sql: SELECT a FROM t WHERE a = :s\n    parameters:\n      s: {type: \"list[string]\", required: true}\n",
     "flowdash.yaml:4:10: List parameter :s may only be used in IN (:s)."),
    ("metrics:\n  m:\n    source: data\n    tabel: t\n", "flowdash.yaml:4:5: Unknown key 'tabel' in Metric 'm'. Did you mean 'table'?"),
    ("metrics:\n  m:\n    source: data\n    table: t\n    table: u\n", "flowdash.yaml:5:5: 'table' is given twice."),
    ("metrics: [\n", "flowdash.yaml:2:1: Not valid YAML"),
])
def test_load_errors_carry_file_positions(text, expected):
    assert any(problem.startswith(expected) for problem in problems(text)), problems(text)


def test_load_reports_every_problem():
    text = "metrics:\n  a:\n    source: data\n  b:\n    source: data\n    table: t\n"
    assert problems(text) == [
        "flowdash.yaml:2:3: Metric 'a' needs either a `table` with measures and dimensions, or a `sql` query.",
        "flowdash.yaml:4:3: Metric 'b' needs at least one measure.",
    ]


def test_check_against_the_source_points_at_the_column(duck):
    text = PROJECT.replace("year: p_year", "year: p_yaer")
    with pytest.raises(ProjectError) as info:
        metrics_for(duck, text)
    assert str(info.value) == "flowdash.yaml:11:13: Unknown column 'p_yaer' in dimension 'year'. Did you mean 'p_year'?"


def test_check_reports_an_unknown_table_and_engine_errors(duck):
    with pytest.raises(ProjectError, match=r"flowdash.yaml:5:12: The source 'data' has no table 'turbine'. Did you mean 'turbines'"):
        metrics_for(duck, PROJECT.replace("table: turbines", "table: turbine"))
    with pytest.raises(ProjectError, match=r"flowdash.yaml:\d+:\d+: Metric 'top_projects': .*'nope'"):
        metrics_for(duck, PROJECT.replace("FROM turbines\n", "FROM nope\n"))


def test_metric_over_a_source_not_served_is_left_out_with_a_warning(duck):
    metrics, warnings = check_metrics(
        read_metrics(parse_project(PROJECT.replace("source: data\n    sql", "source: other\n    sql"), PROJECT_FILE)),
        {SOURCES_KEY: {"data": duck}},
    )
    assert list(metrics) == ["capacity"]
    assert [str(w) for w in warnings] == [
        "flowdash.yaml:17:13: Metric 'top_projects' is left out: the app serves no source 'other'; its sources are: data."
    ]


def test_unknown_sections_are_ignored_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="lumen.flowdash.project"):
        project = parse_project("theme:\n  primary: red\nmetrics: {}\n", PROJECT_FILE)
    assert project.section("theme") == {"primary": "red"}
    assert read_metrics(project) == {}
    assert "flowdash.yaml:1:1: Ignoring the unknown section 'theme'" in caplog.text


def test_find_project_next_to_the_data_then_in_the_working_directory(tmp_path):
    data, cwd = tmp_path / "data", tmp_path / "cwd"
    data.mkdir(), cwd.mkdir()
    (data / "turbines.csv").write_text("a\n1\n")
    assert find_project([str(data / "turbines.csv")], cwd=cwd) is None
    (cwd / PROJECT_FILE).write_text("metrics: {}\n")
    assert find_project([str(data / "turbines.csv")], cwd=cwd) == cwd / PROJECT_FILE
    (data / PROJECT_FILE).write_text("metrics: {}\n")
    assert find_project([str(data / "turbines.csv"), "https://example.com/x.csv"], cwd=cwd) == data / PROJECT_FILE
    with pytest.raises(ProjectError, match="Cannot read the project file"):
        load_project(tmp_path / "missing.yaml")


# Values

@pytest.mark.parametrize("kind, value, rendered", [
    ("string", "TX' OR '1'='1", "'TX'' OR ''1''=''1'"),
    ("string", "x\\'; DROP TABLE turbines; --", "'x\\''; DROP TABLE turbines; --'"),
    ("integer", "7", "7"),
    ("number", 2.5, "2.5"),
    ("boolean", "true", "TRUE"),
    ("date", "2010-01-02", "CAST('2010-01-02' AS DATE)"),
    ("datetime", "2010-01-02T03:04:05", "CAST('2010-01-02 03:04:05' AS TIMESTAMP)"),
])
def test_values_are_bound_as_typed_literals(kind, value, rendered):
    text = (
        "metrics:\n  m:\n    source: data\n    sql: SELECT * FROM turbines WHERE x = :v\n"
        f"    parameters:\n      v: {{type: {kind}, required: true}}\n"
    )
    metric = read_metrics(parse_project(text))["m"]
    sql = bind_sql(metric, metric.values({"v": value}), "duckdb")
    assert sql == f"SELECT * FROM turbines WHERE x = {rendered}"


@pytest.mark.parametrize("kind, value", [
    ("integer", "1; DROP TABLE turbines"), ("integer", True), ("integer", 1.5), ("number", "nan"),
    ("number", float("inf")), ("boolean", "yes"), ("date", "2010-01-02' OR 1=1 --"),
    ("datetime", "now()"), ("string", 1), ("string", ["a"]),
])
def test_values_of_the_wrong_type_are_refused(kind, value):
    text = (
        "metrics:\n  m:\n    source: data\n    sql: SELECT * FROM turbines WHERE x = :v\n"
        f"    parameters:\n      v: {{type: {kind}, required: true}}\n"
    )
    metric = read_metrics(parse_project(text))["m"]
    with pytest.raises(MetricError, match="Parameter 'v' of metric 'm'"):
        metric.values({"v": value})


def test_list_values_bounds_options_and_unknown_names(duck):
    metric = metrics_for(duck)["top_projects"]
    sql = bind_sql(metric, metric.values({"states": ["TX", "x') OR ('1'='1"], "n": 3}), "duckdb")
    assert "t_state IN ('TX', 'x'') OR (''1''=''1')" in sql
    assert sql.endswith("LIMIT 3")
    assert "WHERE TRUE GROUP BY" in " ".join(bind_sql(metric, metric.values({"states": []}), "duckdb").split())
    with pytest.raises(MetricError, match=r"outside the bounds \[1, 50\]"):
        metric.values({"n": 500})
    with pytest.raises(MetricError, match="has no parameter 'm'; its parameters are: n, states"):
        metric.values({"m": 1})
    capacity = metrics_for(duck)["capacity"]
    assert capacity.values({"state": ["TX"]}) == {"since": 2000, "state": ["TX"]}
    with pytest.raises(MetricError, match="takes a value or a list of values"):
        capacity.values({"state": {"$gt": 1}})


def test_structured_metric_binds_filters_as_literals(duck):
    capacity = metrics_for(duck)["capacity"]
    transform = aggregate(capacity, Query(["capacity_mw"], ["state"]), capacity.values({"since": 2005, "state": "T'X"}), "duckdb")
    assert transform.where == ['"p_year" >= 2005', "\"t_state\" = 'T''X'"]


# Pipelines

def run(pipeline, limit=1000):
    return [tuple(row) for row in fetch(pipeline, limit).values.tolist()]


def test_structured_metric_pipeline(duck):
    capacity = metrics_for(duck)["capacity"]
    pipeline = metric_pipeline(capacity, duck, Query(["capacity_mw", "turbines"], ["state"], order_by=["-capacity_mw"]))
    assert list(pipeline.schema) == ["state", "capacity_mw", "turbines"]
    assert pipeline.schema["capacity_mw"]["d3_format"] == ",.1f"
    assert pipeline.schema["state"]["description"] == "Two-letter state code"
    assert run(pipeline) == [("TX", 6.0, 3), ("IA", 1.0, 1)]


def test_filter_on_a_column_of_the_rows_is_applied_before_aggregation(duck):
    capacity = metrics_for(duck)["capacity"]
    by_year = metric_pipeline(capacity, duck, Query(["capacity_mw"], ["year"], order_by=["year"]))
    texas = filtered_pipeline(by_year, "t_state", "TX")
    sql = " ".join(pipeline_sql(texas).split())
    assert sql.index("\"t_state\" = 'TX'") < sql.index("GROUP BY")
    assert run(texas) == [(2001, 1.5), (2010, 4.5)]
    by_state = metric_pipeline(capacity, duck, Query(["capacity_mw"], ["state"], order_by=["state"]))
    via_dimension = " ".join(pipeline_sql(filtered_pipeline(by_state, "state", ["IA"])).split())
    assert via_dimension.index("\"t_state\" IN ('IA')") < via_dimension.index("GROUP BY")
    big = " ".join(pipeline_sql(filtered_pipeline(by_state, "capacity_mw", (5, None))).split())
    assert big.index("GROUP BY") < big.index("\"capacity_mw\" >= '5'")
    assert run(filtered_pipeline(by_state, "capacity_mw", (5, None))) == [("TX", 6.0)]
    with pytest.raises(ValueError, match="no column 'nope'"):
        filtered_pipeline(by_state, "nope", 1)


def test_filter_on_a_sql_metric_filters_the_tables_it_reads(duck):
    top = metrics_for(duck)["top_projects"]
    pipeline = metric_pipeline(top, duck, Query(parameters={"n": 5}))
    assert run(pipeline) == [("Roscoe", 3500), ("Alta", 3000), ("Horse Hollow", 2500), ("Story", 1000)]
    in_2010 = filtered_pipeline(pipeline, "p_year", 2010)
    sql = " ".join(pipeline_sql(in_2010).split())
    assert sql.index('"p_year" = 2010') < sql.index("GROUP BY")
    assert run(in_2010) == [("Horse Hollow", 2500), ("Roscoe", 2000)]
    assert run(filtered_pipeline(pipeline, "capacity_kw", (3000, None))) == [("Roscoe", 3500), ("Alta", 3000)]


@pytest.mark.parametrize("grain, expected", [
    ("year", ["1999-01-01", "2001-01-01", "2010-01-01", "2015-01-01"]),
    ("quarter", ["1999-01-01", "2001-04-01", "2010-01-01", "2010-10-01", "2015-07-01"]),
    ("month", ["1999-02-01", "2001-05-01", "2010-03-01", "2010-11-01", "2015-08-01"]),
    ("week", ["1999-02-01", "2001-05-14", "2010-03-29", "2010-11-01", "2015-08-03"]),
    ("day", ["1999-02-01", "2001-05-17", "2010-03-30", "2010-11-02", "2015-08-09"]),
])
@pytest.mark.parametrize("engine", ["duck", "lite"])
def test_time_grains(request, engine, grain, expected):
    source = request.getfixturevalue(engine)
    capacity = metrics_for(source)["capacity"]
    pipeline = metric_pipeline(
        capacity, source, Query(["turbines"], ["online"], grain=grain, order_by=["online"], parameters={"since": 1990}),
    )
    got = [str(online)[:10] for online, _ in run(pipeline)]
    assert got == expected


def test_query_fields_are_checked(duck):
    capacity = metrics_for(duck)["capacity"]
    with pytest.raises(MetricError, match="has no measure 'power'"):
        metric_pipeline(capacity, duck, Query(["power"]))
    with pytest.raises(MetricError, match="A grain needs a time dimension"):
        metric_pipeline(capacity, duck, Query(["turbines"], ["state"], grain="year"))
    with pytest.raises(MetricError, match="has no grain 'hour'"):
        metric_pipeline(capacity, duck, Query(["turbines"], ["online"], grain="hour"))
    with pytest.raises(MetricError, match="Cannot order by '-year'"):
        metric_pipeline(capacity, duck, Query(["turbines"], ["state"], order_by=["-year"]))


# The node in a dashboard

def output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


def instance(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


@pytest.fixture
def dashboard(duck):
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, duck)
    editor.context[METRICS_KEY] = metrics_for(duck)
    ops = DashboardOps(editor)
    yield ops
    editor.dispose()


def test_metric_node_in_a_dashboard(dashboard):
    ops = dashboard
    ops.add_node(METRIC_ID, {"metric": "capacity", "measures": ["capacity_mw"], "dimensions": ["state", "online"],
                             "grain": "year", "order_by": ["state", "online"]}, name="capacity")
    ops.add_node("Widgets/Select", {"label": "State", "default_options": ["TX", "IA"]}, name="pick")
    ops.add_node(FILTER_ID, {"field": "t_state"}, name="by_state")
    ops.connect("capacity.pipeline", "by_state.pipeline")
    ops.connect("pick.selected", "by_state.value")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("by_state.filtered", "rows.pipeline")
    ops.add_node(VIEW_ID, {"spec": {"mark": "bar", "encoding": {
        "x": {"field": "online", "type": "temporal"}, "y": {"field": "capacity_mw", "type": "quantitative"}}}},
        name="chart")
    ops.connect("by_state.filtered", "chart.pipeline")

    frame = output(ops, "rows", "data")
    assert [(s, str(o)[:4], v) for s, o, v in frame.values.tolist()] == [("TX", "2001", 1.5), ("TX", "2010", 4.5)]
    assert output(ops, "capacity", "data")["state"].tolist() == ["IA", "TX", "TX"]

    ops.set_inputs("capacity", {"parameters": {"since": 2005}})
    frame = output(ops, "rows", "data")
    assert [(s, str(o)[:4], v) for s, o, v in frame.values.tolist()] == [("TX", "2010", 4.5)]

    problems = ops.set_inputs("capacity", {"parameters": {"since": "soon"}})
    assert problems == ["Parameter 'since' of metric 'capacity': expected an integer, got 'soon'."]
    with pytest.raises(OutputUnavailable, match="expected an integer, got 'soon'"):
        ops.node_output("capacity.pipeline")
    ops.set_inputs("capacity", {"parameters": {"since": 2005}})
    with pytest.raises(OperationError, match="has no measure 'power'"):
        ops.set_config("capacity", {"measures": ["power"]})
    with pytest.raises(OperationError, match="no metric `nope`; its metrics are: `capacity`, `top_projects`"):
        ops.add_node(METRIC_ID, {"metric": "nope"}, name="nope")


def test_metric_node_takes_parameters_from_a_wired_dict(dashboard):
    ops = dashboard
    ops.add_node(METRIC_ID, {"metric": "top_projects"}, name="top")
    assert len(output(ops, "top", "data")) == 2
    ops.set_inputs("top", {"parameters": {"n": 3, "states": ["TX"]}})
    assert output(ops, "top", "data").values.tolist() == [["Roscoe", 3500], ["Horse Hollow", 2500]]


async def test_metric_choices_and_presets(dashboard):
    from panel_flowdash.config_choices import node_choices

    ops = dashboard
    presets = metric_presets(ops.editor.context)
    assert [p["title"] for p in presets] == ["capacity (metric)", "top_projects (metric)"]
    assert presets[0]["config"] == {"metric": "capacity", "measures": ["capacity_mw"]}
    ops.add_node(METRIC_ID, {"metric": "capacity", "measures": ["turbines"], "dimensions": ["online"]}, name="m")
    choices = await node_choices(ops.editor, ops._instance("m"))
    assert choices["metric"].values() == ["capacity", "top_projects"]
    assert choices["dimensions"].values() == ["state", "year", "online"]
    assert choices["grain"].values() == ["", "year", "quarter", "month", "week", "day"]
    assert choices["order_by"].labelled()["turbines (descending)"] == "-turbines"
    ops.add_node(FILTER_ID, {}, name="f")
    ops.connect("m.pipeline", "f.pipeline")
    field = (await node_choices(ops.editor, ops._instance("f")))["field"].values()
    assert field == ["online", "turbines", "p_name", "t_state", "p_year", "t_cap"]


def test_build_routes_reads_the_project_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "turbines.csv"
    data.write_text("p_name,t_state,p_year,online,t_cap\n" + "".join(f"{n},{s},{y},{d},{c}\n" for n, s, y, d, c in ROWS))
    project = tmp_path / PROJECT_FILE
    project.write_text(PROJECT.replace("source: data", "source: turbines"))
    routes = build_routes([str(data)], project=project)
    assert routes
    project.write_text(PROJECT.replace("source: data", "source: turbines").replace("p_year, op", "p_yeer, op"))
    with pytest.raises(ProjectError, match=r"flowdash.yaml:\d+:\d+: "):
        build_routes([str(data)], project=project)


def test_dates_coerce(duck):
    text = "metrics:\n  m:\n    source: data\n    table: turbines\n    measures: {n: \"COUNT(*)\"}\n" \
           "    parameters:\n      after: {type: date, column: online, op: \">=\", default: \"2010-01-01\"}\n" \
           "      span: {type: \"range[date]\", column: online}\n"
    metric = metrics_for(duck, text)["m"]
    assert metric.values({})["after"] == dt.date(2010, 1, 1)
    assert run(metric_pipeline(metric, duck, Query(["n"], parameters={"span": ["2010-01-01", None]}))) == [(3,)]
    assert run(metric_pipeline(metric, duck, Query(["n"], parameters={"span": [None, "2010-06-30"], "after": "1990-01-01"}))) == [(3,)]


# Review follow-ups

def test_null_takes_the_default_and_a_required_parameter_needs_a_value(duck):
    capacity = metrics_for(duck)["capacity"]
    assert capacity.values({"since": None}) == {"since": 2000}
    text = PROJECT.replace("default: 2000, column: p_year", "column: p_year, options: [2000, 2010]")
    optional = metrics_for(duck, text)["capacity"]
    assert optional.values({"since": None}) == {"since": None}
    with pytest.raises(MetricError, match="is not one of 2000, 2010"):
        optional.values({"since": 1990})
    top = metrics_for(duck)["top_projects"]
    assert top.values({"n": None, "states": None}) == {"n": 2, "states": ["CA", "TX", "IA"]}
    required = read_metrics(parse_project(
        "metrics:\n  m:\n    source: data\n    sql: SELECT * FROM turbines LIMIT :n\n"
        "    parameters:\n      n: {type: integer, required: true}\n"))["m"]
    with pytest.raises(MetricError, match="needs a value for 'n'"):
        required.values({"n": None})


def test_sql_parameters_need_a_default_or_required():
    assert problems(
        "metrics:\n  m:\n    source: data\n    sql: SELECT * FROM t LIMIT :n\n    parameters:\n      n: {type: integer}\n"
    ) == ["flowdash.yaml:6:7: Parameter 'n' of a SQL metric needs a `default` or `required: true`."]
    assert problems(
        "metrics:\n  m:\n    source: data\n    sql: SELECT * FROM t LIMIT :n\n    parameters:\n"
        "      n: {type: number, default: 1}\n"
    ) == ["flowdash.yaml:4:10: Parameter :n in LIMIT or OFFSET must be an integer."]


def test_a_negative_row_count_is_refused(duck):
    top = metrics_for(duck, PROJECT.replace("bounds: [1, 50]", "bounds: [null, 50]"))["top_projects"]
    with pytest.raises(MetricError, match="is a row count; it cannot be -1"):
        bind_sql(top, top.values({"n": -1}), "duckdb")


def test_limit_and_order_by_are_parameters_of_a_structured_metric(duck):
    capacity = metrics_for(duck)["capacity"]
    by_state = Query(["capacity_mw"], ["state"], order_by=["state"])
    assert [r[0] for r in run(metric_pipeline(capacity, duck, by_state))] == ["IA", "TX"]
    top = Query(["capacity_mw"], ["state"], order_by=["state"], parameters={"limit": 1, "order_by": ["-capacity_mw"]})
    assert run(metric_pipeline(capacity, duck, top)) == [("TX", 6.0)]
    with pytest.raises(MetricError, match="Parameter 'limit' of metric 'capacity': .*bounds"):
        metric_pipeline(capacity, duck, Query(["turbines"], parameters={"limit": 0}))
    declared = metrics_for(duck, PROJECT.replace(
        "      since:", "      limit: {type: integer, default: 1, bounds: [1, 5]}\n      since:"))["capacity"]
    assert len(run(metric_pipeline(declared, duck, Query(["turbines"], ["state"])))) == 1
    with pytest.raises(MetricError, match=r"outside the bounds \[1, 5\]"):
        metric_pipeline(declared, duck, Query(["turbines"], ["state"], parameters={"limit": 9}))
    assert problems(PROJECT.replace("      since:", "      limit: {type: number}\n      since:"))[0].endswith(
        "Parameter 'limit' sets the metric's limit: it is of type integer and has no column or op.")


def test_order_by_refuses_a_doubled_minus_and_fields_are_deduplicated(duck):
    capacity = metrics_for(duck)["capacity"]
    with pytest.raises(MetricError, match="Cannot order by '--turbines'"):
        metric_pipeline(capacity, duck, Query(["turbines"], ["state"], order_by=["--turbines"]))
    from lumen.flowdash.nodes import metric_query

    query = metric_query({"measures": ["turbines", "turbines"], "dimensions": ["state", "state"]})
    assert list(metric_pipeline(capacity, duck, query).schema) == ["state", "turbines"]


def test_dimension_filters_are_coerced_to_the_column_type(duck):
    capacity = metrics_for(duck)["capacity"]
    assert run(metric_pipeline(capacity, duck, Query(["turbines"], parameters={"year": "2010"}))) == [(2,)]
    with pytest.raises(MetricError, match="Filter 'year' of metric 'capacity': expected an integer, got 'soon'"):
        metric_pipeline(capacity, duck, Query(["turbines"], parameters={"year": "soon"}))
    with pytest.raises(MetricError, match="Filter 'state' of metric 'capacity': expected a string, got 1"):
        metric_pipeline(capacity, duck, Query(["turbines"], parameters={"state": [1]}))


def test_sqlite_grains_are_timestamps(lite, duck):
    for source in (lite, duck):
        capacity = metrics_for(source)["capacity"]
        pipeline = metric_pipeline(capacity, source, Query(["turbines"], ["online"], grain="year", order_by=["online"]))
        frame = fetch(pipeline, 10)
        assert str(frame["online"].dtype).startswith("datetime64"), source
        assert pipeline.schema["online"]["type"] == "string" and pipeline.schema["online"]["format"] == "datetime"


def test_sqlite_runs_queries_with_colons_in_literals_and_comments(lite):
    text = PROJECT.replace(
        "WHERE t_state IN (:states)",
        "WHERE t_state IN (:states) AND p_name != 'a :b' AND p_name != 'c \\\\:d' -- top :n\n     ",
    )
    top = metrics_for(lite, text)["top_projects"]
    assert run(metric_pipeline(top, lite, Query(parameters={"states": ["x :y", "TX"]}))) == [
        ("Roscoe", 3500), ("Horse Hollow", 2500)]


def test_sql_metric_filters_only_the_tables_its_outer_query_reads(duck):
    text = PROJECT + """\
  shares:
    source: data
    sql: |
      SELECT t_state, SUM(t_cap) * 1.0 / (SELECT SUM(t_cap) FROM turbines) AS share
      FROM turbines GROUP BY t_state ORDER BY t_state
  shadowed:
    source: data
    sql: |
      WITH turbines AS (SELECT p_name, t_cap FROM main.turbines WHERE t_cap > 1500)
      SELECT p_name, SUM(t_cap) AS kw FROM turbines GROUP BY p_name ORDER BY p_name
"""
    metrics = metrics_for(duck, text)
    texas = filtered_pipeline(metric_pipeline(metrics["shares"], duck, Query()), "p_year", 2010)
    assert run(texas) == [("TX", 4500 / 10000)]
    shadowed = metric_pipeline(metrics["shadowed"], duck, Query())
    assert run(shadowed) == [("Alta", 3000), ("Horse Hollow", 2500), ("Roscoe", 2000)]
    assert run(filtered_pipeline(shadowed, "kw", (2500, None))) == [("Alta", 3000), ("Horse Hollow", 2500)]
    with pytest.raises(ValueError, match="no column 't_state'"):
        filtered_pipeline(shadowed, "t_state", "TX")


@pytest.mark.parametrize("function", ["fts3_tokenizer", "current_query", "txid_current"])
def test_more_functions_are_denied(function):
    from lumen.flowdash.checks import DENIED_FUNCTIONS

    assert "current_role" in DENIED_FUNCTIONS
    text = f"metrics:\n  m:\n    source: data\n    table: t\n    measures:\n      x: \"COUNT({function}('a'))\"\n"
    assert problems(text)[0].endswith(f"Measure 'x' calls {function}(), which reads the engine's settings or environment.")


def test_parameter_nodes_drive_a_metric_from_widgets(dashboard):
    from lumen.flowdash import PARAMETER_ID

    ops = dashboard
    ops.add_node(METRIC_ID, {"metric": "top_projects"}, name="top")
    ops.add_node("Widgets/Slider", {"label": "N", "default_start": 1, "default_end": 10, "default_value": 1},
                 name="slider")
    ops.add_node(PARAMETER_ID, {"parameter": "n"}, name="n")
    ops.connect("slider.selected", "n.value")
    ops.connect("n.parameters", "top.parameters")
    ops.add_node(PARAMETER_ID, {"parameter": "states"}, name="states")
    ops.set_inputs("states", {"value": ["TX"]})
    ops.connect("states.parameters", "top.parameters")
    assert output(ops, "top", "data").values.tolist() == [["Roscoe", 3500]]
    instance(ops, "slider").value = 3
    assert output(ops, "n", "parameters") == {"n": 3}
    assert output(ops, "top", "data").values.tolist() == [["Roscoe", 3500], ["Horse Hollow", 2500]]
    with pytest.raises(OperationError, match="No metric takes a parameter 'm'"):
        ops.add_node(PARAMETER_ID, {"parameter": "m"}, name="m")


@pytest.mark.parametrize("where, rows", [
    ("t_state NOT IN (:ex)", 5),
    ("p_year > 2000 AND t_state NOT IN (:ex)", 4),
    ("(p_year > 2000 AND (t_state IN (:ex)))", 4),
    ("p_year > 2000 AND NOT (t_state IN (:ex))", 4),
])
def test_an_empty_list_drops_the_condition_it_is_anded_into(duck, where, rows):
    text = (
        f"metrics:\n  m:\n    source: data\n    sql: SELECT * FROM turbines WHERE {where}\n"
        "    parameters:\n      ex: {type: \"list[string]\", default: [CA]}\n"
    )
    metric = metrics_for(duck, text)["m"]
    assert len(run(metric_pipeline(metric, duck, Query(parameters={"ex": []})))) == rows


@pytest.mark.parametrize("sql", [
    "SELECT * FROM turbines WHERE p_year > 2010 OR t_state IN (:ex)",
    "SELECT * FROM turbines WHERE NOT (p_year > 2010 OR t_state NOT IN (:ex))",
    "SELECT t_state IN (:ex) AS picked FROM turbines",
    "SELECT * FROM turbines WHERE (t_state IN (:ex)) = FALSE",
])
def test_an_empty_list_is_refused_where_dropping_its_condition_changes_the_query(duck, sql):
    text = (
        f"metrics:\n  m:\n    source: data\n    sql: \"{sql}\"\n"
        "    parameters:\n      ex: {type: \"list[string]\", default: [CA]}\n"
    )
    metric = metrics_for(duck, text)["m"]
    with pytest.raises(MetricError, match="'ex' of metric 'm' is an empty list, which only drops a condition ANDed"):
        metric_pipeline(metric, duck, Query(parameters={"ex": []}))


def test_parameter_tiles_start_hidden_and_duplicates_warn(dashboard):
    from lumen.flowdash import PARAMETER_ID

    ops = dashboard
    ops.add_node(METRIC_ID, {"metric": "top_projects"}, name="top")
    for name in ("a", "b"):
        ops.add_node(PARAMETER_ID, {"parameter": "n"}, name=name)
        ops.set_inputs(name, {"value": 1 if name == "a" else 3})
        ops.connect(f"{name}.parameters", "top.parameters")
    hidden = {e["instance_id"] for e in ops.editor.layout if e.get("visible") is False}
    assert {ops._instance("a"), ops._instance("b"), ops._instance("top")} <= hidden
    assert len(output(ops, "top", "data")) == 3
    from panel_flowdash.config_check import check_node_sync

    result = check_node_sync(ops.editor, ops._instance("top"))
    assert result.warnings == ["More than one Lumen Parameter node sets 'n'; the one wired last wins."]
