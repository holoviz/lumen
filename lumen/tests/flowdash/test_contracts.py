from pathlib import Path

import pytest
import yaml

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, OperationError
from panel_flowdash.parameters import ContractError

from lumen.data.contracts import FULL, PARTIAL, source_schema
from lumen.flowdash import COMPONENTS, bind_source
from lumen.flowdash.contracts import (
    check, check_unrecorded, expects, source_contract, usages,
)
from lumen.sources.duckdb import DuckDBSource

SESSIONS = Path(__file__).parents[3] / "tests" / "evals" / "sessions"

COMMERCE = {
    "customers": "SELECT * FROM (VALUES (1, 'Ada', 'east'), (2, 'Bo', 'west')) AS t(customer_id, customer, region)",
    "orders": "SELECT * FROM (VALUES (101, 1, 30, 'paid'), (102, 2, 20, 'cancelled')) AS t(order_id, customer_id, amount, status)",
}

SCHEMA = {
    "customers": {"customer_id": "integer", "customer": "string", "region": "string"},
    "orders": {"order_id": "integer", "customer_id": "integer", "amount": "integer", "status": "string"},
}

JOIN = """
SELECT region, SUM(amount) AS total
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE status = 'paid' GROUP BY region ORDER BY total
"""


def graph(*nodes, edges=()):
    doc = {"flowdash": 4, "nodes": {"data": {"component": "Lumen/source", "config": {"ref": "data"}}}, "edges": []}
    for name, component, config in nodes:
        doc["nodes"][name] = {"component": component, "config": config}
        doc["edges"].append({"from": "data.source", "to": f"{name}.source"})
    doc["edges"] += [{"from": a, "to": b} for a, b in edges]
    return doc


def tables(usage):
    return {table: sorted(columns) for table, columns in usage.tables.items()}


def test_views_through_a_star_and_on_a_table_node_read_table_columns():
    doc = graph(
        ("everything", "Lumen/sql", {"sql": "SELECT * FROM orders"}),
        ("customers", "Lumen/table", {"table": "customers"}),
        edges=[
            ("everything.pipeline", "chart.pipeline"),
            ("customers.pipeline", "only_east.pipeline"),
            ("only_east.filtered", "map.pipeline"),
        ],
    )
    doc["nodes"]["chart"] = {"component": "Lumen/view", "config": {"spec": {"encoding": {"x": {"field": "status"}, "y": {"field": "nope"}}}}}
    doc["nodes"]["only_east"] = {"component": "Lumen/filter", "config": {"field": "region"}}
    doc["nodes"]["map"] = {"component": "Lumen/view", "config": {"spec": {"encoding": {"color": {"field": "customer"}}}}}
    found = {u.node: (u.coverage, tables(u)) for u in usages(doc, "data", SCHEMA, "duckdb")}
    assert found == {
        "everything": (PARTIAL, {"orders": []}),
        "chart": (PARTIAL, {"orders": ["status"]}),
        "customers": (FULL, {"customers": []}),
        "only_east": (FULL, {"customers": ["region"]}),
        "map": (FULL, {"customers": ["customer"]}),
    }


class TestCheck:

    def doc(self):
        return graph(("by_region", "Lumen/sql", {"sql": JOIN}),
                     ("customers_only", "Lumen/sql", {"sql": "SELECT customer FROM customers"}))

    def contract(self):
        return expects(self.doc(), "data", SCHEMA, "duckdb")

    def changed(self, table, **columns):
        schema = {t: dict(c) for t, c in SCHEMA.items()}
        for column, kind in columns.items():
            if kind is None:
                del schema[table][column]
            else:
                schema[table][column] = kind
        return schema

    def test_the_contract_lists_what_is_read(self):
        assert self.contract() == {"tables": {
            "customers": {"customer": "string", "customer_id": "integer", "region": "string"},
            "orders": {"amount": "integer", "customer_id": "integer", "status": "string"},
        }}

    def test_the_same_schema_passes(self):
        assert check(self.doc(), "data", self.contract(), SCHEMA, "duckdb") == []

    def test_a_missing_column_names_only_the_nodes_reading_it(self):
        problems = check(self.doc(), "data", self.contract(), self.changed("customers", region=None), "duckdb")
        assert [(p.node, p.message) for p in problems] == [
            ("by_region", "Table 'customers' has no column 'region'."),
        ]

    def test_a_changed_type_fails_but_number_to_integer_does_not(self):
        problems = check(self.doc(), "data", self.contract(), self.changed("orders", amount="string"), "duckdb")
        assert [(p.node, p.message) for p in problems] == [
            ("by_region", "Column orders.amount is string now; the contract expects integer."),
        ]
        assert [p.node for p in check(self.doc(), "data", self.contract(), self.changed("orders", amount="number"), "duckdb")] == ["by_region"]
        contract = self.contract()
        contract["tables"]["orders"]["amount"] = "number"
        assert check(self.doc(), "data", contract, SCHEMA, "duckdb") == []

    def test_a_missing_table_names_its_readers(self):
        schema = {"orders": SCHEMA["orders"]}
        problems = check(self.doc(), "data", self.contract(), schema, "duckdb")
        assert {(p.node, p.message) for p in problems} == {
            ("by_region", "The data has no table 'customers'."),
            ("customers_only", "The data has no table 'customers'."),
        }

    def test_partly_covered_nodes_are_warnings(self):
        doc = graph(("all", "Lumen/sql", {"sql": "SELECT * FROM orders"}))
        problems = check(doc, "data", expects(doc, "data", SCHEMA, "duckdb"), SCHEMA, "duckdb")
        assert [(p.node, p.warning) for p in problems] == [("all", True)]

    def test_unrecorded_checks_names_against_the_new_schema(self):
        doc = self.doc()
        assert check_unrecorded(doc, "data", SCHEMA, "duckdb") == []
        problems = check_unrecorded(doc, "data", self.changed("customers", region=None), "duckdb")
        assert [(p.node, p.message) for p in problems] == [
            ("by_region", "No table it reads has a column 'region'."),
        ]
        problems = check_unrecorded(doc, "data", self.changed("customers", customer=None), "duckdb")
        assert [(p.node, p.message) for p in problems] == [
            ("customers_only", "Table 'customers' has no column 'customer'."),
        ]


@pytest.mark.parametrize("path", sorted(SESSIONS.glob("*.yaml")), ids=lambda p: p.stem)
def test_recorded_sessions_are_fully_covered(path):
    from lumen.ai.evals.harness import Inputs
    from tests.evals.cases import source

    session = yaml.safe_load(path.read_text())
    data = source(Inputs(["x"], fixture=session["fixture"]))
    for turn in session["turns"]:
        doc = yaml.safe_load(turn["dashboard"])
        found = usages(doc, "data", source_schema(data), "duckdb")
        components = {node["component"] for node in doc["nodes"].values()} - {"Lumen/source"}
        if components == {"Lumen/metric"}:
            # A metric node reads its source by the metric's ref, not through an edge, so no contract covers it yet.
            assert found == []
            continue
        assert found and all(u.coverage == FULL for u in found)
        contract = source_contract(doc, "data", data)
        assert check(doc, "data", contract, source_schema(data), "duckdb") == []


class TestParameter:

    @pytest.fixture
    def ops(self):
        editor = FlowDash(components=COMPONENTS, notifications=False, history_coalesce=0)
        bind_source(editor, DuckDBSource(uri=":memory:", tables=COMMERCE), "data")
        ops = DashboardOps(editor)
        doc = graph(("by_region", "Lumen/sql", {"sql": JOIN, "table": "by_region"}))
        ops.replace_yaml(yaml.safe_dump(doc), revision=ops.revision())
        ops.declare_parameter("data", "data", "ref", type="source")
        yield ops
        editor.dispose()

    def test_declaring_records_the_contract(self, ops):
        assert ops.parameters()["data"]["expects"] == {"tables": {
            "customers": {"customer_id": "integer", "region": "string"},
            "orders": {"amount": "integer", "customer_id": "integer", "status": "string"},
        }}

    def test_rebinding_to_data_without_a_column_is_refused(self, ops):
        renamed = {**COMMERCE, "customers": COMMERCE["customers"].replace("region", "area")}
        bind_source(ops.editor, DuckDBSource(uri=":memory:", tables=renamed), "other")
        with pytest.raises(OperationError, match="by_region: Table 'customers' has no column 'region'"):
            ops.bind_parameters({"data": "other"})
        assert ops.node_config("data")["ref"] == "data"
        with pytest.raises(ContractError) as info:
            ops.editor.bind_parameters({"data": "other"})
        assert info.value.nodes == ["by_region"]

    def test_rebinding_to_data_that_fits_recomputes(self, ops):
        more = {**COMMERCE, "orders": COMMERCE["orders"].replace("'cancelled'", "'paid'")}
        bind_source(ops.editor, DuckDBSource(uri=":memory:", tables=more), "other")
        ops.bind_parameters({"data": "other"})
        assert sorted(ops.node_output("by_region.data")["total"]) == [20, 30]

    def test_an_unknown_ref_is_refused(self, ops):
        with pytest.raises(OperationError, match="no source `nowhere`"):
            ops.bind_parameters({"data": "nowhere"})
