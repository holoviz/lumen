import argparse
import json

import pandas as pd
import pytest
import yaml

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, dashboard_to_yaml

from lumen.command.run import Run
from lumen.flowdash import COMPONENTS, bind_source
from lumen.flowdash.run import (
    CONTRACT, FAILED, OK, USAGE, format_report, resolve_source, run_dashboard,
)

JOIN = """
SELECT region, SUM(amount) AS total
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE status = 'paid' GROUP BY region ORDER BY region
"""

CUSTOMERS = pd.DataFrame({"customer_id": [1, 2], "region": ["east", "west"]})
ORDERS = pd.DataFrame({"customer_id": [1, 2, 2], "amount": [30, 20, 5], "status": ["paid", "paid", "cancelled"]})


def dashboard(sql=JOIN):
    return {
        "flowdash": 4,
        "turns": [{"id": "t1", "actor": "agent", "at": "2026-11-19T10:00:00Z", "question": "Paid by region?"}],
        "nodes": {
            "data": {"component": "Lumen/source", "config": {"ref": "data"}},
            "by_region": {"component": "Lumen/sql", "origin": "t1", "config": {"sql": sql, "table": "by_region"}},
        },
        "edges": [{"from": "data.source", "to": "by_region.source"}],
    }


@pytest.fixture
def files(tmp_path):
    def write(name, frame):
        path = tmp_path / name
        frame.to_parquet(path)
        return str(path)

    return {
        "orders": write("orders.parquet", ORDERS),
        "customers": write("customers.parquet", CUSTOMERS),
        "area": write("area.parquet", CUSTOMERS.rename(columns={"region": "area"})),
        "text": write("text.parquet", ORDERS.assign(amount=ORDERS.amount.astype(str))),
        "more": write("more.parquet", pd.concat([ORDERS, ORDERS])),
    }


def data(files, orders="orders", customers="customers"):
    return f"orders={files[orders]},customers={files[customers]}"


@pytest.fixture
def declared(files):
    """The dashboard with a source parameter, declared on the data it was built on."""
    editor = FlowDash(components=COMPONENTS, notifications=False)
    try:
        bind_source(editor, resolve_source(data(files)), "data")
        editor.import_dashboard(yaml.safe_dump(dashboard()))
        ops = DashboardOps(editor)
        ops.declare_parameter("data", "data", "ref", type="source")
        return dashboard_to_yaml(editor)
    finally:
        editor.dispose()


def rows(report, node="by_region"):
    return [[r["region"], r["total"]] for r in report.nodes[node]["outputs"]["data"]["head"]]


def test_a_declared_parameter_records_the_contract(declared):
    assert yaml.safe_load(declared)["parameters"]["data"]["expects"] == {"tables": {
        "customers": {"customer_id": "integer", "region": "string"},
        "orders": {"amount": "integer", "customer_id": "integer", "status": "string"},
    }}


def test_run_with_a_parameter(declared, files):
    report = run_dashboard(declared, {"data": data(files)})
    assert (report.status, report.exit_code) == ("ok", OK)
    assert rows(report) == [["east", 30], ["west", 20]]
    report = run_dashboard(declared, {"data": data(files, orders="more")})
    assert rows(report) == [["east", 60], ["west", 40]]


def test_a_renamed_column_fails_the_contract_naming_the_node(declared, files):
    report = run_dashboard(declared, {"data": data(files, customers="area")})
    assert (report.status, report.exit_code) == ("contract", CONTRACT)
    assert report.problems == [{
        "node": "by_region", "message": "Table 'customers' has no column 'region'.", "parameter": "data",
    }]
    assert report.questions == ["Paid by region?"]
    assert not report.nodes
    text = format_report(report)
    assert "by_region: Table 'customers' has no column 'region'." in text and "Paid by region?" in text


def test_a_changed_type_fails_the_contract(declared, files):
    report = run_dashboard(declared, {"data": data(files, orders="text")})
    assert report.exit_code == CONTRACT
    assert report.problems[0]["message"] == "Column orders.amount is string now; the contract expects integer."


def test_a_source_by_ref_is_checked_against_what_the_nodes_name(files):
    text = yaml.safe_dump(dashboard())
    assert run_dashboard(text, sources={"data": data(files)}).exit_code == OK
    report = run_dashboard(text, sources={"data": data(files, customers="area")})
    assert report.exit_code == CONTRACT
    assert report.problems == [{"node": "by_region", "message": "No table it reads has a column 'region'."}]


def test_a_single_file_takes_the_name_of_the_one_table_read(files):
    text = yaml.safe_dump(dashboard("SELECT SUM(amount) AS total FROM orders"))
    report = run_dashboard(text, sources={"data": files["more"]})
    assert report.exit_code == OK
    assert report.nodes["by_region"]["outputs"]["data"]["head"] == [{"total": 110}]


def test_usage_errors(files):
    text = yaml.safe_dump(dashboard())
    report = run_dashboard(text)
    assert (report.status, report.exit_code) == ("usage", USAGE)
    assert report.message == "Bind the data of the dashboard's sources: --source data=<data>."
    assert "is not a parameter" in run_dashboard(text, {"nope": "x"}).message
    assert run_dashboard("flowdash: 99\n").exit_code == USAGE


def test_a_failing_node_is_reported(files):
    text = yaml.safe_dump(dashboard("SELECT SUM(amount) / 0 AS total FROM orders WHERE amount > 'x'"))
    report = run_dashboard(text, sources={"data": data(files)})
    assert (report.status, report.exit_code) == ("failed", FAILED)
    assert report.nodes["by_region"]["status"] in ("failed", "invalid")


def test_the_command(declared, files, tmp_path, capsys):
    path = tmp_path / "dashboard.yaml"
    path.write_text(declared)
    parser = argparse.ArgumentParser()
    command = Run(parser=parser)

    def invoke(*argv):
        return command.invoke(parser.parse_args([str(path), *argv]))

    assert invoke("--param", f"data={data(files)}") == OK
    assert "by_region" in capsys.readouterr().out
    assert invoke("--json", "--param", f"data={data(files, customers='area')}") == CONTRACT
    assert json.loads(capsys.readouterr().out)["status"] == "contract"
    assert invoke("--param", "data") == USAGE
    assert "--param takes NAME=VALUE" in capsys.readouterr().err
