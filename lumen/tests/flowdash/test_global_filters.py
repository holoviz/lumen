import datetime as dt

from importlib.util import find_spec

import pytest
import sqlglot

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps
from panel_flowdash.selection import Predicate

from lumen.flowdash import (
    COMPONENTS, FILTER_OPTIONS, SOURCE_ID, SQL_ID, TABLE_ID, VIEW_ID,
    FilterMismatch, bind_source, column_values, fetch, global_condition,
    global_conditions, globally_filtered, pipeline_sql, table_pipeline,
)
from lumen.sources.duckdb import DuckDBSource
from lumen.transforms.sql import SQLFilter

GLOBAL_FILTERS = find_spec("panel_flowdash.global_filters") is not None
needs_global_filters = pytest.mark.skipif(
    not GLOBAL_FILTERS, reason="panel-flowdash without global filters"
)

SALES = (
    "SELECT * FROM (VALUES ('A', 10, 'TX'), ('B', 20, 'OK'), ('A', 5, 'TX'), ('C', 7, 'NM')) "
    "AS t(category, amount, state)"
)

SPEC = {
    "mark": "bar",
    "encoding": {
        "x": {"field": "category", "type": "nominal"},
        "y": {"field": "amount", "type": "quantitative"},
    },
}

INJECTION = "x') OR 1=1; DROP TABLE sales; --"


@pytest.fixture
def duckdb_source():
    return DuckDBSource(uri=":memory:", tables={"sales": SALES})


@pytest.fixture
def sqlite_source(tmp_path):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from lumen.sources.sqlalchemy import SQLAlchemySource

    url = f"sqlite:///{tmp_path / 'sales.db'}"
    engine = sqlalchemy.create_engine(url)
    with engine.begin() as connection:
        connection.execute(sqlalchemy.text(
            "CREATE TABLE sales (category TEXT, amount INTEGER, state TEXT)"
        ))
        connection.execute(sqlalchemy.text(
            "INSERT INTO sales VALUES ('A', 10, 'TX'), ('B', 20, 'OK'), ('A', 5, 'TX'), ('C', 7, 'NM')"
        ))
    engine.dispose()
    source = SQLAlchemySource(url=url)
    yield source
    source.close()


def rows(frame):
    return sorted(map(tuple, frame[["category", "amount"]].values.tolist()))


class TestPushdown:

    @pytest.mark.parametrize("fixture", ["duckdb_source", "sqlite_source"])
    def test_predicates_become_part_of_the_query(self, fixture, request):
        source = request.getfixturevalue(fixture)
        pipeline = globally_filtered(table_pipeline(source, "sales"), {
            "category": Predicate("category", "in", ("A", "C")),
            "amount": Predicate("amount", "range", (6, None)),
        })
        sql = " ".join(pipeline_sql(pipeline).split())
        assert "\"category\" IN ('A', 'C')" in sql
        assert "\"amount\" >= '6'" in sql
        assert sql.index("WHERE") < sql.index("LIMIT")
        assert rows(fetch(pipeline)) == [("A", 10), ("C", 7)]

    @pytest.mark.parametrize("fixture", ["duckdb_source", "sqlite_source"])
    def test_a_closed_range_and_numeric_strings(self, fixture, request):
        source = request.getfixturevalue(fixture)
        pipeline = globally_filtered(table_pipeline(source, "sales"), {
            "amount": Predicate("amount", "range", ("5", "10")),
        })
        assert "BETWEEN '5' AND '10'" in pipeline_sql(pipeline)
        assert rows(fetch(pipeline)) == [("A", 5), ("A", 10), ("C", 7)]

    @pytest.mark.parametrize("fixture", ["duckdb_source", "sqlite_source"])
    def test_injection_in_values_is_a_quoted_literal(self, fixture, request):
        source = request.getfixturevalue(fixture)
        pipeline = globally_filtered(table_pipeline(source, "sales"), {
            "category": Predicate("category", "in", (INJECTION,)),
        })
        assert "'x'') OR 1=1; DROP TABLE sales; --'" in pipeline_sql(pipeline)
        assert fetch(pipeline).empty
        assert len(fetch(table_pipeline(source, "sales"))) == 4

    @pytest.mark.parametrize("dialect, state, literal", [
        ("postgres", '"state"', "'x'') OR 1=1; DROP TABLE sales; --'"),
        ("bigquery", "`state`", "'x\\') OR 1=1; DROP TABLE sales; --'"),
    ])
    def test_other_dialects_quote_through_sqlglot(self, dialect, state, literal):
        conditions = [
            ("state", global_condition(Predicate("state", "in", ("TX", INJECTION)), "string")),
            ("amount", global_condition(Predicate("amount", "range", (1, 10), closed="left"), "integer")),
        ]
        sql = SQLFilter(conditions=conditions, write=dialect).apply("SELECT * FROM sales")
        assert f"WHERE {state} IN ('TX', {literal}) AND" in sql
        statements = sqlglot.parse(sql, read=dialect)
        assert len(statements) == 1
        assert INJECTION in [v.this for v in statements[0].find_all(sqlglot.exp.Literal)]

    def test_a_filter_on_a_column_the_pipeline_lacks_is_left_out(self, duckdb_source):
        pipeline = table_pipeline(duckdb_source, "sales")
        conditions, mismatched = global_conditions(pipeline, {
            'state" = 1; --': Predicate('state" = 1; --', "in", ("TX",)),
        })
        assert conditions == [] and mismatched == {}
        assert globally_filtered(pipeline, {"region": Predicate("region", "in", ("X",))}) is pipeline

    def test_values_a_column_cannot_take_are_not_applied(self, duckdb_source):
        pipeline = table_pipeline(duckdb_source, "sales")
        conditions, mismatched = global_conditions(pipeline, {
            "amount": Predicate("amount", "in", ("n/a",)),
            "state": Predicate("state", "range", (1, 2)),
            "category": Predicate("category", "in", (1.0,)),
        })
        assert conditions == [("category", ["1"])]
        assert set(mismatched) == {"amount", "state"}

    def test_dates_arrive_as_iso_strings_or_datetimes(self):
        assert global_condition(Predicate("day", "in", ("2020-01-01T00:00:00",)), "date") == [
            dt.datetime(2020, 1, 1)
        ]
        assert global_condition(Predicate("day", "range", (dt.datetime(2020, 1, 1), None)), "datetime") == (
            dt.datetime(2020, 1, 1), None
        )
        with pytest.raises(FilterMismatch):
            global_condition(Predicate("day", "in", ("soon",)), "date")


class TestColumnValues:

    def test_distinct_values_are_computed_by_the_source_with_a_limit(self, executed):
        source = DuckDBSource(uri=":memory:", tables={
            "many": "SELECT i % 50 AS n, 'v' || (i % 50) AS label FROM range(1000) t(i)"
        })
        executed.clear()
        values = column_values(table_pipeline(source, "many"), "n", limit=10)
        assert values == list(range(10))
        # Besides the schema queries a pipeline makes once, one query, which fetches no rows.
        fetched = [q for q in executed if "__count" not in q and not q.endswith(" LIMIT 1")]
        assert len(fetched) == 1
        assert 'SELECT DISTINCT "n"' in fetched[0] and fetched[0].endswith("LIMIT 11")

    def test_values_follow_the_pipeline_filters_but_not_its_row_limit(self, duckdb_source):
        pipeline = globally_filtered(table_pipeline(duckdb_source, "sales", limit=1), {
            "state": Predicate("state", "in", ("TX", "NM")),
        })
        assert column_values(pipeline, "category") == ["A", "C"]

    def test_a_span(self, duckdb_source):
        assert column_values(table_pipeline(duckdb_source, "sales"), "amount", span=True) == (5, 20)

    def test_nothing_for_a_column_the_pipeline_lacks(self, duckdb_source):
        assert column_values(table_pipeline(duckdb_source, "sales"), "region") is None

    def test_values_on_sqlite(self, sqlite_source):
        assert column_values(table_pipeline(sqlite_source, "sales"), "state") == ["NM", "OK", "TX"]

    def test_the_limit_is_panel_flowdash_s(self):
        if GLOBAL_FILTERS:
            from panel_flowdash.global_filters import MAX_OPTIONS
            assert FILTER_OPTIONS == MAX_OPTIONS


def dashboard(source):
    editor = FlowDash(components=COMPONENTS, notifications=False)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node(VIEW_ID, {"spec": SPEC}, name="chart")
    ops.connect("sales.pipeline", "chart.pipeline")
    ops.add_node(VIEW_ID, {"spec": SPEC}, name="reference")
    ops.connect("sales.pipeline", "reference.pipeline")
    ops.add_node(SQL_ID, {"sql": "SELECT category, SUM(amount) AS amount FROM sales GROUP BY category"}, name="totals")
    ops.connect("data.source", "totals.source")
    ops.add_node(VIEW_ID, {"spec": SPEC}, name="by_category")
    ops.connect("totals.pipeline", "by_category.pipeline")
    return ops


def node(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def chart_rows(ops, name):
    return rows(node(ops, name)._object.pipeline.data)


class TestNodes:

    def test_nodes_declare_their_columns(self, duckdb_source):
        ops = dashboard(duckdb_source)
        assert node(ops, "chart").__flowdash_columns__() == ["category", "amount", "state"]
        assert node(ops, "by_category").__flowdash_columns__() == ["category", "amount"]
        assert node(ops, "sales").__flowdash_columns__() == ["category", "amount", "state"]
        assert node(ops, "totals").__flowdash_columns__() == ["category", "amount"]
        assert not hasattr(node(ops, "data"), "__flowdash_apply_filters__")
        ops.editor.dispose()

    def test_a_view_queries_with_its_filters_and_its_input_is_left_alone(self, duckdb_source, executed):
        ops = dashboard(duckdb_source)
        chart = node(ops, "chart")
        executed.clear()
        chart.__flowdash_apply_filters__({"state": Predicate("state", "in", ("TX",))})
        assert chart_rows(ops, "chart") == [("A", 5), ("A", 10)]
        assert any("\"state\" IN ('TX')" in q for q in executed)
        assert len(fetch(chart.pipeline)) == 4
        chart.__flowdash_apply_filters__({})
        assert len(node(ops, "chart")._object.pipeline.data) == 4
        ops.editor.dispose()

    def test_a_table_node_shows_its_filtered_rows_and_outputs_every_row(self, duckdb_source):
        ops = dashboard(duckdb_source)
        sales = node(ops, "sales")
        sales.__flowdash_apply_filters__({"category": Predicate("category", "in", ("B",))})
        assert sales._body().value["category"].tolist() == ["B"]
        assert len(fetch(sales.pipeline())) == 4
        ops.editor.dispose()

    def test_a_sql_node_filters_its_result_in_the_database(self, duckdb_source, executed):
        ops = dashboard(duckdb_source)
        totals = node(ops, "totals")
        executed.clear()
        totals.__flowdash_apply_filters__({"amount": Predicate("amount", "range", (10, None))})
        assert sorted(totals._body().value["category"]) == ["A", "B"]
        assert any("\"amount\" >= '10'" in q for q in executed)
        ops.editor.dispose()

    def test_column_values_of_a_node(self, duckdb_source):
        ops = dashboard(duckdb_source)
        assert node(ops, "chart").__flowdash_column_values__("state") == ["NM", "OK", "TX"]
        assert node(ops, "by_category").__flowdash_column_values__("state") is None
        ops.editor.dispose()

    @needs_global_filters
    def test_the_filter_bar_reaches_lumen_views(self, duckdb_source):
        ops = dashboard(duckdb_source)
        editor = ops.editor
        editor.set_filters([{"column": "state", "label": "State"}])
        assert "TX" in editor._global_filters.options("state")
        editor.set_filter_value("state", ["TX"])
        assert chart_rows(ops, "chart") == chart_rows(ops, "reference") == [("A", 5), ("A", 10)]
        editor.dispose()

    @needs_global_filters
    def test_a_view_opts_out(self, duckdb_source):
        ops = dashboard(duckdb_source)
        editor = ops.editor
        editor.set_filters([{"column": "state", "value": ["TX"]}])
        editor.exclude_from_filter("state", ops._instance("reference"))
        assert chart_rows(ops, "chart") == [("A", 5), ("A", 10)]
        assert len(chart_rows(ops, "reference")) == 4
        _, hint, detail = editor._global_filters.notes(ops._instance("reference"))
        assert hint == "Not filtered by state" and "opts out" in detail
        editor.exclude_from_filter("state", ops._instance("reference"), excluded=False)
        assert chart_rows(ops, "reference") == [("A", 5), ("A", 10)]
        editor.dispose()

    @needs_global_filters
    def test_a_view_fed_by_an_aggregate_without_the_column_says_so(self, duckdb_source):
        ops = dashboard(duckdb_source)
        editor = ops.editor
        editor.set_filters([{"column": "state", "label": "State", "value": ["TX"]}])
        assert len(chart_rows(ops, "by_category")) == 3
        _, hint, detail = editor._global_filters.notes(ops._instance("by_category"))
        assert hint == "Not filtered by State"
        assert "no column 'state'" in detail
        editor.dispose()

    @needs_global_filters
    def test_a_range_filter_offers_the_span(self, duckdb_source):
        ops = dashboard(duckdb_source)
        editor = ops.editor
        editor.set_filters([{"column": "amount", "widget": "range"}])
        assert editor._global_filters.span("amount") == (5, 20)
        editor.set_filter_value("amount", [10, 20])
        assert chart_rows(ops, "chart") == [("A", 10), ("B", 20)]
        editor.dispose()

    @needs_global_filters
    def test_filters_reach_a_view_whose_columns_arrive_later(self, duckdb_source):
        editor = FlowDash(components=COMPONENTS, notifications=False)
        bind_source(editor, duckdb_source)
        ops = DashboardOps(editor)
        ops.add_node(TABLE_ID, {"ref": "data"}, name="sales")
        ops.add_node(VIEW_ID, {"spec": SPEC}, name="chart")
        ops.connect("sales.pipeline", "chart.pipeline")
        editor.set_filters([{"column": "state", "value": ["TX"]}])
        assert node(ops, "chart").__flowdash_columns__() is None
        node(ops, "sales").table = "sales"
        assert chart_rows(ops, "chart") == [("A", 5), ("A", 10)]
        editor.dispose()
