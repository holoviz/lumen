import sqlite3

import pandas as pd
import pytest

pytest.importorskip("duckdb")
config_choices = pytest.importorskip("panel_flowdash.config_choices")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, OperationError

from lumen.flowdash import (
    AGGREGATE_ID, COMPONENTS, DATA_ID, DERIVE_ID, FILTER_ID, RESULT_LIMIT,
    SELECT_ID, SORT_ID, SOURCE_ID, TABLE_ID, VIEW_ID, bind_source, fetch,
    pipeline_sql,
)
from lumen.flowdash.choices import pipeline_column_types
from lumen.flowdash.contracts import check_unrecorded, expects
from lumen.flowdash.transforms import (
    aggregate_schema, check_aggregate_config, check_derive_config,
    expression_type, parse_expression,
)
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.sources.duckdb import DuckDBSource

ROWS = [("A", "north", 10), ("B", "south", 20), ("A", "south", 5), ("C", "north", 1)]
SALES = pd.DataFrame(ROWS, columns=["category", "region", "amount"])
VALUES = ", ".join(f"('{c}', '{r}', {a})" for c, r, a in ROWS)


def duckdb_source(tmp_path):
    return DuckDBSource(uri=":memory:", tables={"sales": f"SELECT * FROM (VALUES {VALUES}) AS t(category, region, amount)"})


def sqlite_source(tmp_path):
    from lumen.sources.sqlalchemy import SQLAlchemySource

    path = tmp_path / "sales.db"
    with sqlite3.connect(path) as connection:
        SALES.to_sql("sales", connection, index=False)
    return SQLAlchemySource(url=f"sqlite:///{path}", tables=["sales"])


def memory_source(tmp_path):
    return InMemorySource(tables={"sales": SALES.copy()})


SOURCES = [duckdb_source, sqlite_source, memory_source]


@pytest.fixture(params=SOURCES, ids=["duckdb", "sqlite", "memory"])
def any_source(request, tmp_path):
    if request.param is sqlite_source:
        pytest.importorskip("sqlalchemy")
    return request.param(tmp_path)


def chain(source, *steps, data=True):
    """
    A dashboard reading table ``sales`` of ``source`` through ``steps`` of
    (component, config), into a data node unless ``data`` is False.
    """
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")
    port = "sales.pipeline"
    for i, (component, config) in enumerate(steps):
        ops.add_node(component, config, name=f"step{i}")
        ops.connect(port, f"step{i}.pipeline")
        port = f"step{i}.result"
    if data:
        ops.add_node(DATA_ID, {}, name="rows")
        ops.connect(port, "rows.pipeline")
    return ops


def result(ops):
    last = [node["name"] for node in ops.nodes() if node["name"].startswith("step")][-1]
    return ops.node_output(f"{last}.result")


def rows(ops):
    return [tuple(row) for row in ops.node_output("rows.data").values.tolist()]


def data_queries(executed):
    return [q for q in executed if "__count" not in q and not (q.endswith(" LIMIT 1") and "subquery" not in q)]


def test_aggregate_groups_and_aggregates(any_source):
    ops = chain(any_source, (AGGREGATE_ID, {"by": ["category"], "columns": ["amount"], "method": "sum"}))
    assert rows(ops) == [("A", 15), ("B", 20), ("C", 1)]
    ops.set_config("step0", {"method": "mean", "by": ["region", "category"]})
    assert rows(ops) == [("north", "A", 10.0), ("north", "C", 1.0), ("south", "A", 5.0), ("south", "B", 20.0)]


def test_aggregate_without_groups_or_columns(any_source):
    ops = chain(any_source, (AGGREGATE_ID, {"columns": ["amount"], "method": "max"}))
    assert rows(ops) == [(20,)]
    ops.set_config("step0", {"columns": [], "method": "count", "by": ["region"]})
    assert rows(ops) == [("north", 2), ("south", 2)]
    assert list(ops.node_output("rows.data").columns) == ["region", "count"]


def test_sort_orders_rows_and_keeps_the_first(any_source):
    ops = chain(any_source, (SORT_ID, {"by": ["amount"], "descending": True}))
    assert [r[2] for r in rows(ops)] == [20, 10, 5, 1]
    ops.set_config("step0", {"limit": 2, "descending": False})
    assert [r[2] for r in rows(ops)] == [1, 5]


def test_select_keeps_columns_in_order(any_source):
    ops = chain(any_source, (SELECT_ID, {"columns": ["amount", "category"]}))
    assert rows(ops) == [(10, "A"), (20, "B"), (5, "A"), (1, "C")]


def test_derived_column_is_added_or_replaces_one(any_source):
    ops = chain(any_source, (DERIVE_ID, {"column": "double", "expression": '"amount" * 2'}))
    assert [r[-1] for r in rows(ops)] == [20, 40, 10, 2]
    assert list(ops.node_output("rows.data").columns) == ["category", "region", "amount", "double"]
    ops.set_config("step0", {"column": "amount", "expression": "CASE WHEN amount > 5 THEN 'big' ELSE 'small' END"})
    assert rows(ops)[0] == ("A", "north", "big")
    assert pipeline_column_types(result(ops)) == {"category": "string", "region": "string", "amount": "string"}


def test_a_chain_of_steps_runs_as_one_query(executed, tmp_path):
    source = duckdb_source(tmp_path)
    ops = chain(
        source,
        (DERIVE_ID, {"column": "cents", "expression": '"amount" * 100'}),
        (AGGREGATE_ID, {"by": ["category"], "columns": ["cents"], "method": "sum"}),
        (SORT_ID, {"by": ["cents"], "descending": True, "limit": 2}),
        (SELECT_ID, {"columns": ["category"]}),
    )

    assert rows(ops) == [("B",), ("A",)]
    statement = pipeline_sql(result(ops))
    assert data_queries(executed) == [" ".join(statement.split())]
    assert 'GROUP BY "category"' in statement
    assert 'ORDER BY "cents" DESC LIMIT 2' in statement
    assert statement.endswith(f"LIMIT {RESULT_LIMIT}")


def test_steps_on_a_source_without_sql_run_on_the_fetched_rows(tmp_path):
    ops = chain(memory_source(tmp_path), (AGGREGATE_ID, {"by": ["region"], "columns": ["amount"], "method": "sum"}))
    pipeline = result(ops)
    assert pipeline_sql(pipeline) is None
    assert [type(t).__name__ for t in pipeline.transforms] == ["FrameSQL", "Iloc"]


def test_a_filter_after_a_top_n_filters_the_rows_kept(tmp_path):
    ops = chain(duckdb_source(tmp_path), (SORT_ID, {"by": ["amount"], "descending": True, "limit": 2}), data=False)
    ops.add_node(DATA_ID, {}, name="rows")
    ops.add_node("Widgets/Select", {"default_options": ["A", "B"]}, name="pick")
    ops.add_node(FILTER_ID, {"field": "category"}, name="by_category")
    ops.connect("step0.result", "by_category.pipeline")
    ops.connect("pick.selected", "by_category.value")
    ops.connect("by_category.filtered", "rows.pipeline")
    ops.editor._component_instances[ops._instance("pick")]._widget.value = "A"
    assert rows(ops) == [("A", "north", 10)]


def test_a_step_on_a_sql_node_result_is_pushed_into_its_query(executed, tmp_path):
    source = duckdb_source(tmp_path)
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node("Lumen/sql", {"sql": "SELECT category, amount FROM sales", "table": "q"}, name="q")
    ops.connect("data.source", "q.source")
    ops.add_node(AGGREGATE_ID, {"by": ["category"], "columns": ["amount"], "method": "min"}, name="least")
    ops.connect("q.pipeline", "least.pipeline")
    executed.clear()
    assert fetch(ops.node_output("least.result")).values.tolist() == [["A", 5], ["B", 20], ["C", 1]]
    assert len(data_queries(executed)) == 1


def test_output_schema_is_known_without_fetching(executed, tmp_path):
    ops = chain(
        duckdb_source(tmp_path),
        (DERIVE_ID, {"column": "ratio", "expression": '"amount" / 3'}),
        (AGGREGATE_ID, {"by": ["category"], "columns": ["amount", "ratio"], "method": "count"}),
        data=False,
    )
    assert pipeline_column_types(result(ops)) == {"category": "string", "amount": "integer", "ratio": "integer"}
    ops.set_config("step1", {"method": "mean"})
    assert pipeline_column_types(result(ops)) == {"category": "string", "amount": "number", "ratio": "number"}
    assert pipeline_column_types(ops.node_output("step0.result"))["ratio"] == "number"
    assert data_queries(executed) == []


@pytest.mark.parametrize("component, config, error", [
    (AGGREGATE_ID, {"by": ["shop"], "columns": ["amount"]}, "no column 'shop'"),
    (AGGREGATE_ID, {"by": ["category"], "columns": ["region"], "method": "sum"}, "not numeric"),
    (AGGREGATE_ID, {"by": ["amount"], "columns": ["amount"]}, "grouped by and aggregated"),
    (SORT_ID, {"by": ["price"]}, "no column 'price'"),
    (SELECT_ID, {"columns": ["amount", "amount"]}, "kept once"),
    (SELECT_ID, {"columns": ["amount", "cost"]}, "no column 'cost'"),
    (DERIVE_ID, {"column": "x", "expression": '"price" * 2'}, "no column 'price'"),
    (DERIVE_ID, {"column": "x", "expression": "(SELECT 1)"}, "not a query or a table"),
    (DERIVE_ID, {"column": "x", "expression": "SUM(amount)"}, "Aggregate in an aggregate node"),
    (DERIVE_ID, {"column": "x", "expression": "getenv('HOME')"}, "getenv"),
    (DERIVE_ID, {"column": "x", "expression": "amount AS y"}, "alias"),
    (DERIVE_ID, {"column": "x", "expression": "amount, region"}, "one SQL expression"),
    (DERIVE_ID, {"column": "x", "expression": "1 FROM sales"}, "one SQL expression"),
    (DERIVE_ID, {"column": "x;y", "expression": "1"}, "not a column name"),
    (DERIVE_ID, {"column": "x", "expression": "*"}, "not a value"),
    (DERIVE_ID, {"column": "x", "expression": "COUNT(*) OVER ()"}, "not a value"),
    (DERIVE_ID, {"column": "x", "expression": 'sales."amount" * 2'}, "without a table"),
])
def test_checks_refuse_configs_before_they_apply(tmp_path, component, config, error):
    ops = chain(duckdb_source(tmp_path), (component, {}))
    with pytest.raises(OperationError, match=error):
        ops.set_config("step0", config)


def test_derived_expression_window_functions_are_allowed():
    assert parse_expression("amount - AVG(amount) OVER ()", "duckdb") is not None
    assert check_derive_config({"column": "x", "expression": "Amount + 1"}, {}) is None


def test_expression_type_falls_back_to_unknown():
    schema = {"amount": {"type": "integer"}, "sold": {"type": "string", "format": "datetime"},
              "Cap kw": {"type": "integer"}, "Price": {"type": "number"}}
    assert expression_type("amount / 2", schema, "duckdb") == {"type": "number"}
    assert expression_type('"Cap kw" * 2', schema, "duckdb") == {"type": "integer"}
    assert expression_type("price + 1", schema, "duckdb") == {"type": "number"}
    assert expression_type("CAST(sold AS DATE)", schema, "duckdb") == {"type": "string", "format": "date"}
    assert expression_type("no_such_function(amount)", schema, "duckdb") == {}


async def test_columns_are_offered_from_the_input_schema(tmp_path):
    ops = chain(duckdb_source(tmp_path), (DERIVE_ID, {"column": "double", "expression": "amount * 2"}), (AGGREGATE_ID, {}))
    choices = await config_choices.node_choices(ops.editor, ops._instance("step1"))
    assert choices["by"].values() == ["category", "region", "amount", "double"]
    assert choices["columns"].values() == ["category", "region", "amount", "double"]


def test_incomplete_config_waits_without_an_error(tmp_path):
    ops = chain(duckdb_source(tmp_path), (AGGREGATE_ID, {"method": "sum"}))
    assert ops.node_output("step0.result") is None


def test_card_preview_shows_the_transformed_rows(tmp_path):
    ops = chain(duckdb_source(tmp_path), (AGGREGATE_ID, {"by": ["region"], "columns": ["amount"]}))
    node = ops.editor._component_instances[ops._instance("step0")]
    preview = node.__flowdash_preview__("result")
    assert preview.values.tolist() == [["north", 11], ["south", 25]]


def test_a_view_charts_a_transformed_pipeline(tmp_path):
    ops = chain(duckdb_source(tmp_path), (AGGREGATE_ID, {"by": ["region"], "columns": ["amount"]}))
    spec = {"mark": "bar", "encoding": {"x": {"field": "region", "type": "nominal"}, "y": {"field": "amount", "type": "quantitative"}}}
    ops.add_node(VIEW_ID, {"spec": spec}, name="chart")
    ops.connect("step0.result", "chart.pipeline")
    with pytest.raises(OperationError, match="nope"):
        ops.set_config("chart", {"spec": {**spec, "encoding": {"x": {"field": "nope", "type": "nominal"}}}})


def _doc(*steps, view_fields=()):
    doc = {"flowdash": 4, "nodes": {
        "data": {"component": SOURCE_ID, "config": {"ref": "data"}},
        "sales": {"component": TABLE_ID, "config": {"table": "sales"}},
    }, "edges": [{"from": "data.source", "to": "sales.source"}]}
    previous = "sales.pipeline"
    for i, (component, config) in enumerate(steps):
        doc["nodes"][f"step{i}"] = {"component": component, "config": config}
        doc["edges"].append({"from": previous, "to": f"step{i}.pipeline"})
        previous = f"step{i}.result"
    encoding = {f"c{i}": {"field": field} for i, field in enumerate(view_fields)}
    doc["nodes"]["chart"] = {"component": VIEW_ID, "config": {"spec": {"mark": "bar", "encoding": encoding}}}
    doc["edges"].append({"from": previous, "to": "chart.pipeline"})
    return doc


SCHEMA = {"sales": {"category": "string", "region": "string", "amount": "integer", "unused": "string"}}


def test_contract_records_the_columns_transforms_read():
    doc = _doc(
        (DERIVE_ID, {"column": "cents", "expression": '"amount" * 100'}),
        (AGGREGATE_ID, {"by": ["category"], "columns": ["cents"], "method": "sum"}),
        (SORT_ID, {"by": ["cents"]}),
        view_fields=["category", "cents"],
    )
    assert expects(doc, "data", SCHEMA) == {
        "tables": {"sales": {"amount": "integer", "category": "string"}},
    }


def test_contract_traces_a_count_to_no_column():
    doc = _doc((AGGREGATE_ID, {"by": ["region"], "method": "count"}), view_fields=["region", "count"])
    assert expects(doc, "data", SCHEMA) == {"tables": {"sales": {"region": "string"}}}


def test_unrecorded_check_names_the_transform_reading_a_missing_column():
    doc = _doc((SELECT_ID, {"columns": ["category", "price"]}), view_fields=["category"])
    problems = check_unrecorded(doc, "data", SCHEMA)
    assert [(p.node, p.message) for p in problems] == [("step0", "Table 'sales' has no column 'price'.")]


def test_transform_output_is_a_lumen_pipeline(tmp_path):
    ops = chain(duckdb_source(tmp_path), (SELECT_ID, {"columns": ["category"]}))
    assert isinstance(result(ops), Pipeline)


def test_aggregate_sums_columns_of_unknown_type(tmp_path):
    ops = chain(
        duckdb_source(tmp_path),
        (DERIVE_ID, {"column": "Amount x2", "expression": '"amount" * 2'}),
        (DERIVE_ID, {"column": "opaque", "expression": "no_such_function(amount)"}),
        data=False,
    )
    assert pipeline_column_types(ops.node_output("step0.result"))["Amount x2"] == "integer"
    assert check_aggregate_config(
        {"by": ["category"], "columns": ["opaque"], "method": "sum"}, {"pipeline": result(ops)}
    ) is None


def test_sum_of_a_boolean_is_an_integer():
    schema = {"category": {"type": "string"}, "paid": {"type": "boolean"}}
    assert aggregate_schema(schema, ["category"], ["paid"], "sum")["paid"] == {"type": "integer"}


@pytest.mark.parametrize("make", SOURCES, ids=["duckdb", "sqlite", "memory"])
def test_aggregate_puts_null_groups_last_on_every_engine(make, tmp_path, monkeypatch):
    if make is sqlite_source:
        pytest.importorskip("sqlalchemy")
    with_null = SALES.astype({"region": object})
    with_null.loc[0, "region"] = None
    monkeypatch.setitem(globals(), "SALES", with_null)
    monkeypatch.setitem(globals(), "VALUES", VALUES.replace("'north', 10", "NULL, 10"))
    ops = chain(make(tmp_path), (AGGREGATE_ID, {"by": ["region"], "columns": ["amount"], "method": "count"}))
    assert [r[0] for r in rows(ops)][:2] == ["north", "south"]
    assert pd.isna(rows(ops)[-1][0])
