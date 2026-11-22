import argparse
import json
import sys

from pathlib import Path
from unittest.mock import patch

import bokeh.command.subcommands.serve as bokeh_serve
import pytest

pytest.importorskip("duckdb")

from panel.command import Serve
from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, OperationError

from lumen.ai.llm import OpenAI
from lumen.command.ai import LumenAIServe
from lumen.flowdash import (
    DATA_ID, SOURCE_ALIASES_KEY, SOURCE_ID, TABLE_ID, bound_sources,
    listed_tables,
)
from lumen.flowdash.app import (
    build_routes, resolve_sources, source_aliases, source_refs,
)
from lumen.sources.duckdb import DuckDBSource

FIXTURES = Path(__file__).parent / "fixtures"


def output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


@pytest.fixture
def csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "sales.csv"
    path.write_text("category,amount\nA,10\nB,20\nA,5\n")
    return path


def parse(*argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    LumenAIServe(sub.add_parser("serve"))
    return parser.parse_args(["serve", *argv])


def test_flowdash_flag_serves_flowdash_routes_without_an_llm(csv):
    args = parse(str(csv), "--flowdash", "--db-path", str(csv.parent / "dash.db"))
    command = LumenAIServe(argparse.ArgumentParser())
    original = bokeh_serve.build_single_handler_applications
    with patch.object(Serve, "invoke", return_value=True) as base_invoke, \
         patch("lumen.command.ai.get_available_llm") as get_llm:
        assert command.invoke(args) is True

    assert base_invoke.called and not get_llm.called
    assert args.files == []
    assert bokeh_serve.build_single_handler_applications is original
    assert "/" in command._flowdash_apps
    assert command.customize_applications(args, {}) is command._flowdash_apps


def test_default_serve_path_is_unchanged(csv):
    args = parse(str(csv))
    assert args.flowdash is False
    command = LumenAIServe(argparse.ArgumentParser())
    original = bokeh_serve.build_single_handler_applications
    try:
        with patch.object(Serve, "invoke", return_value=True), \
             patch("lumen.command.ai.get_available_llm", return_value=OpenAI) as get_llm:
            command.invoke(args)
        assert get_llm.called
        assert bokeh_serve.build_single_handler_applications is not original
        assert command._flowdash_apps is None
    finally:
        bokeh_serve.build_single_handler_applications = original


@pytest.mark.parametrize("argv, error", [
    (["app.yaml", "--flowdash"], "serve a Lumen YAML dashboard without it"),
    (["data.csv", "--db-path", "x.db"], "only apply with --flowdash"),
    (["data.csv", "--agent-model", "test"], "only apply with --flowdash"),
    (["data.csv", "--flowdash", "--provider", "openai"], "--provider configures the Lumen AI explorer and does not apply"),
    (["data.csv", "--flowdash", "--model", "x", "--temperature", "0.1"], "--temperature, --model configure the Lumen AI explorer and do not apply"),
])
def test_flowdash_options_are_checked(argv, error):
    with pytest.raises(ValueError, match=error):
        LumenAIServe(argparse.ArgumentParser()).invoke(parse(*argv))


def test_served_app_offers_lumen_nodes_over_the_data(csv):
    routes = build_routes([csv.name], db_path=csv.parent / "dash.db")
    app = routes["/"]()
    editor = app._editor

    assert type(editor) is FlowDash
    assert app.agent is False
    assert SOURCE_ID not in app.agent_sensitive_components
    (source,) = bound_sources(editor).values()
    assert list(bound_sources(editor)) == ["sales"]
    (table,) = listed_tables(source)

    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "sales"}, name="data")
    ops.add_node(TABLE_ID, {"table": table, "limit": 2}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("sales.pipeline", "rows.pipeline")
    assert len(editor.graph.get_state(ops._instance("rows")).data) == 2


def test_agent_model_gives_the_app_lumens_agent(csv):
    pytest.importorskip("pydantic_ai")
    routes = build_routes([str(csv)], db_path=csv.parent / "dash.db", agent_model="test")
    agent = routes["/"]().agent

    tools = {tool for toolset in agent.toolsets for tool in getattr(toolset, "tools", {})}
    assert {"list_tables", "add_sql", "add_chart"} <= tools


def test_repeated_agent_model_flags_reach_the_app(csv):
    args = parse(
        str(csv), "--flowdash", "--db-path", str(csv.parent / "dash.db"),
        "--agent-model", "test=Test", "--agent-model", "openai:gpt-6-luna=Luna",
    )
    with patch.object(Serve, "invoke", return_value=True), \
         patch("lumen.flowdash.app.build_routes", return_value={}) as routes:
        LumenAIServe(argparse.ArgumentParser()).invoke(args)

    assert routes.call_args.kwargs["agent_model"] == ["test=Test", "openai:gpt-6-luna=Luna"]


def test_one_agent_model_is_named_without_a_picker(csv):
    pytest.importorskip("pydantic_ai")
    app = build_routes([str(csv)], db_path=csv.parent / "dash.db", agent_model=["test=Test"])["/"]()
    drawer = app._agent_session.drawer
    menu = drawer.chat.model_picker._menu

    assert app.agent.model.model_name == "test"
    assert app.agent_model == "test"
    assert [(m["id"], m["label"]) for m in app.agent_models] == [("test", "Test")]
    assert [option["id"] for option in menu.options] == [menu.value] == ["test:test"]
    assert "remember_model" not in drawer.chat_params


def test_several_agent_models_offer_a_picker(csv):
    pytest.importorskip("pydantic_ai")
    routes = build_routes(
        [str(csv)], db_path=csv.parent / "dash.db", agent_model=["test=Test", "openai:gpt-6-luna=Luna"],
    )
    app = routes["/"]()
    chat_params = app._agent_session.drawer.chat_params

    assert app.agent_model == "test"
    assert [(m.id, m.label) for m in chat_params["models"]] == [("test", "Test"), ("openai:gpt-6-luna", "Luna")]
    assert chat_params["model_id"] == "test"


def test_agent_model_without_an_id_is_refused(csv):
    with pytest.raises(ValueError, match="names no model"):
        build_routes([str(csv)], db_path=csv.parent / "dash.db", agent_model=["test", "=Luna"])


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A DuckDB file with an orders table and two CSV files, as ``lumen serve`` gets them."""
    import duckdb

    monkeypatch.chdir(tmp_path)
    with duckdb.connect(str(tmp_path / "shop.db")) as conn:
        conn.execute("CREATE TABLE orders AS SELECT * FROM (VALUES ('A', 10), ('B', 20)) t(category, amount)")
    (tmp_path / "plants.csv").write_text("plant,mw\nx,1\ny,2\nz,3\n")
    (tmp_path / "turbines.csv").write_text("turbine,mw\na,1\n")
    return tmp_path


def refs(*data):
    return source_refs(resolve_sources(list(data)))


def aliases(*data):
    return source_aliases(refs(*data))


def tables(source):
    return listed_tables(source)


def sql_node(ops, source, sql, name="n"):
    ops.add_node("Lumen/sql", {"sql": sql, "table": name}, name=name)
    ops.connect(f"{source}.source", f"{name}.source")
    return output(ops, name, "data")


def test_the_data_files_source_is_named_after_the_data(served):
    (served / "folder").mkdir()
    (served / "folder" / "wind.csv").write_text("x\n1\n")
    (served / "folder" / "notes.txt").write_text("not data")
    assert list(refs("plants.csv")) == ["plants"]
    assert list(refs("folder")) == ["folder"]
    assert list(refs("plants.csv", "turbines.csv")) == ["local"]
    assert list(refs("shop.db", "plants.csv")) == ["shop", "plants"]
    assert refs("plants.csv")["plants"].name == "plants"
    assert tables(refs("folder")["folder"]) == ["wind"]


def test_tables_drop_the_file_extension(served):
    (served / "x").mkdir()
    (served / "x" / "turbines.csv").write_text("turbine,mw\nb,2\n")
    assert tables(refs("plants.csv", "turbines.csv")["local"]) == ["plants", "turbines"]
    assert tables(refs("x/turbines.csv")["turbines"]) == ["turbines"]
    assert tables(refs(str(served / "x" / "turbines.csv"))["turbines"]) == ["turbines"]
    assert tables(refs("plants.csv", "./plants.csv", str(served / "plants.csv"))["plants"]) == ["plants"]


def test_files_of_one_name_are_numbered_in_command_line_order(served):
    for year in ("2025", "2026"):
        (served / year).mkdir()
        (served / year / "sales.csv").write_text(f"year\n{year}\n")
    first = refs("2025/sales.csv", "2026/sales.csv")["local"]
    assert tables(first) == ["sales", "sales_2"]
    assert first.execute("SELECT * FROM sales_2")["year"].tolist() == [2026]
    second = refs("2026/sales.csv", "2025/sales.csv")["local"]
    assert second.execute("SELECT * FROM sales")["year"].tolist() == [2026]


def test_a_file_added_later_never_takes_a_saved_table(served):
    (served / "x").mkdir()
    (served / "x" / "plants.csv").write_text("plant,mw\nq,9\n")
    (served / "plants_2.csv").write_text("plant,mw\nr,7\n")
    source = refs("plants.csv", "x/plants.csv", "plants_2.csv")["local"]
    assert tables(source) == ["plants", "plants_3", "plants_2"]
    assert source.execute("SELECT plant FROM plants")["plant"].tolist() == ["x", "y", "z"]
    assert source.execute("SELECT plant FROM plants_2")["plant"].tolist() == ["r"]


def test_sources_of_one_name_are_numbered_in_command_line_order(served):
    (served / "shop.csv").write_text("x\n1\n")
    assert list(refs("shop.db", "shop.csv")) == ["shop", "shop_2"]
    assert list(refs("shop.csv", "shop.db")) == ["shop", "shop_2"]
    assert refs("shop.csv", "shop.db")["shop"].get_tables()[0] == "shop"


def test_source_refs_name_sources_by_their_name():
    named = DuckDBSource(uri=":memory:", tables={"t": "SELECT 1 AS x"}, name="Sales Data")
    counted = DuckDBSource(uri=":memory:", tables={"t": "SELECT 1 AS x"}, name="Sales12345")
    table = DuckDBSource(uri=":memory:", tables={"b": "SELECT 1 AS x"})
    several = DuckDBSource(uri=":memory:", tables={"a": "SELECT 1 AS x", "b": "SELECT 1 AS x"})
    assert list(source_refs([named, counted, table, several])) == ["sales_data", "sales12345", "b", "duckdb"]


def test_earlier_table_names_still_read_in_queries_but_are_not_listed(served):
    (served / "x").mkdir()
    (served / "x" / "plants.csv").write_text("plant,mw\nq,9\n")
    editor = build_routes(["plants.csv", "x/plants.csv", "turbines.csv"], db_path=served / "dash.db")["/"]()._editor
    source = bound_sources(editor)["local"]
    assert tables(source) == ["plants", "plants_2", "turbines"]
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "local"}, name="local")
    # turbines_csv: the file name with its extension; x_plants_csv: the path as given.
    assert sql_node(ops, "local", "SELECT COUNT(*) AS n FROM turbines_csv", "t")["n"].tolist() == [1]
    assert sql_node(ops, "local", "SELECT plant FROM x_plants_csv", "x")["plant"].tolist() == ["q"]
    # plants_csv named both files, so it reads neither.
    result = sql_node(ops, "local", "SELECT COUNT(*) AS n FROM plants_csv", "p")
    assert not hasattr(result, "columns")
    assert "plants_csv" in str(getattr(result, "exception", result))


def test_earlier_source_names_still_resolve_unless_two_sources_had_them(served):
    (served / "shop.csv").write_text("x\n1\n")
    assert aliases("shop.db", "plants.csv", "turbines.csv") == {
        "data": "shop", "data_2": "local", "shop_db": "shop", "files": "local",
        "plants": "local", "turbines": "local", "plants_files": "local", "turbines_files": "local",
    }
    clash = aliases("shop.db", "plants.csv", "shop.csv")
    assert "shop_db" in clash and "shop_files" in clash and "files" in clash
    assert "shop" not in clash


@pytest.mark.parametrize("fixture", ["positional_refs.json", "positional_refs_v3.json"])
def test_dashboard_saved_with_positional_refs_still_loads(served, fixture):
    """
    The fixtures were saved when ``lumen serve shop.db plants.csv`` named
    its sources ``data`` (shop.db, read first) and ``data_2`` (the CSV
    files) and the CSV's table ``plants_csv``, as stored model versions 4 and 3.
    """
    routes = build_routes(["plants.csv", "shop.db"], db_path=served / "dash.db")
    editor = routes["/"]()._editor
    assert list(bound_sources(editor)) == ["plants", "shop"]
    assert editor.context[SOURCE_ALIASES_KEY] == {
        "data": "shop", "data_2": "plants", "files": "plants", "local": "plants", "plants_files": "plants",
        "shop_db": "shop",
    }

    editor.import_dashboard(json.loads((FIXTURES / fixture).read_text()))
    ops = DashboardOps(editor)

    assert ops._item(ops._instance("first"))["config"]["ref"] == "data"
    assert output(ops, "first", "source") is bound_sources(editor)["shop"]
    assert output(ops, "second", "source") is bound_sources(editor)["plants"]
    assert output(ops, "order_count", "data")["n"].tolist() == [2]
    assert output(ops, "plant_count", "data")["n"].tolist() == [3]
    assert ops.validate() == []


def test_dashboard_saved_with_one_file_keeps_its_source_when_files_are_added(served):
    editor = build_routes(["turbines.csv", "plants.csv", "shop.db"], db_path=served / "dash.db")["/"]()._editor
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "plants"}, name="saved")
    assert output(ops, "saved", "source") is bound_sources(editor)["local"]
    assert sql_node(ops, "saved", "SELECT COUNT(*) AS n FROM plants")["n"].tolist() == [3]


def test_every_served_source_is_named_by_its_ref(served):
    (served / "shop.csv").write_text("x\n1\n")
    editor = build_routes(["shop.db", "shop.csv"], db_path=served / "dash.db")["/"]()._editor
    sources = bound_sources(editor)
    assert {ref: source.name for ref, source in sources.items()} == {"shop": "shop", "shop_2": "shop_2"}
    assert editor.context[SOURCE_ALIASES_KEY]["shop_db"] == "shop"


def test_a_served_folder_keeps_resolving_when_files_are_served_beside_it(served):
    (served / "f").mkdir()
    (served / "f" / "wind.csv").write_text("x\n1\n")
    assert list(refs("f")) == ["f"]
    assert aliases("f", "plants.csv")["f"] == "local"
    assert "f" not in aliases("f/wind.csv", "plants.csv")


def test_local_resolves_with_one_file(served):
    assert aliases("plants.csv")["local"] == "plants"


def test_clashing_file_names_are_logged_with_their_tables(served, caplog):
    for folder in ("x", "y"):
        (served / folder).mkdir()
        (served / folder / "sales.csv").write_text("x\n1\n")
    with caplog.at_level("WARNING", logger="lumen.flowdash.app"):
        refs("y/sales.csv", "plants.csv", "x/sales.csv")
    assert "sales: y/sales.csv, sales_2: x/sales.csv" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="lumen.flowdash.app"):
        refs("plants.csv", "turbines.csv")
    assert "share a name" not in caplog.text


def test_earlier_table_names_never_come_from_the_working_directory(served):
    source = refs("plants.csv")["plants"]
    legacy = set(source.get_tables()) - set(tables(source))
    assert legacy == {"plants_csv"}


def test_sources_keep_resolve_data_order_when_it_makes_sources_of_its_own(served):
    import pandas as pd

    frame = pd.DataFrame({"x": [1]})
    # The frame joins the data files' source, so there is no source per position.
    sources = resolve_sources([frame, "plants.csv", "shop.db"])
    assert [sorted(listed_tables(s)) for s in sources] == [["orders"], ["data", "plants"]]


def test_a_source_the_host_passes_keeps_its_name(served):
    warehouse = DuckDBSource(uri=":memory:", tables={"t": "SELECT 1 AS x"}, name="Warehouse DB")
    first = build_routes([warehouse, "shop.db"], db_path=served / "one.db")["/"]()._editor
    other = DuckDBSource(uri=":memory:", tables={"u": "SELECT 1 AS x"}, name="warehouse_db")
    second = build_routes([other, warehouse], db_path=served / "two.db")["/"]()._editor
    assert list(bound_sources(first)) == ["warehouse_db", "shop"]
    assert list(bound_sources(second)) == ["warehouse_db", "warehouse_db_2"]
    assert warehouse.name == "Warehouse DB" and other.name == "warehouse_db"
    assert bound_sources(first)["shop"].name == "shop"
    ops = DashboardOps(first)
    ops.add_node(SOURCE_ID, {"ref": "warehouse_db"}, name="w")
    node = first._component_instances[ops._instance("w")]
    assert node._body().object == "warehouse_db: t"


METRICS = """\
metrics:
  total:
    source: sales
    table: sales
    measures: {amount: "SUM(amount)"}
    dimensions: {category: category}
"""


@pytest.mark.parametrize("given", [True, False])
def test_flowdash_reads_the_project_file_given_or_found(csv, given):
    project = csv.parent / ("metrics.yaml" if given else "flowdash.yaml")
    project.write_text(METRICS)
    argv = [str(csv), "--flowdash", *(["--project", str(project)] if given else [])]
    command = LumenAIServe(argparse.ArgumentParser())
    with patch.object(Serve, "invoke", return_value=True), \
         patch("lumen.flowdash.app.build_routes", wraps=build_routes) as routes:
        command.invoke(parse(*argv))
    assert Path(routes.call_args.kwargs["project"]) == project


def test_flowdash_refuses_a_project_file_with_problems(csv):
    project = csv.parent / "flowdash.yaml"
    project.write_text(METRICS.replace("SUM(amount)", "SUM(amout)"))
    with patch.object(Serve, "invoke", return_value=True), pytest.raises(ValueError, match=(
        r"flowdash.yaml:5:24: Unknown column 'amout' in measure 'amount'. Did you mean 'amount'\?"
    )):
        LumenAIServe(argparse.ArgumentParser()).invoke(parse(str(csv), "--flowdash"))


def test_project_only_applies_with_flowdash(csv):
    with pytest.raises(ValueError, match="--project only apply with --flowdash"):
        LumenAIServe(argparse.ArgumentParser()).invoke(parse(str(csv), "--project", "x.yaml"))


@pytest.mark.parametrize("given", [True, False])
def test_the_project_file_read_is_logged(csv, caplog, given):
    project = csv.parent / "flowdash.yaml"
    project.write_text(METRICS)
    argv = [str(csv), "--flowdash", *(["--project", str(project)] if given else [])]
    with patch.object(Serve, "invoke", return_value=True), caplog.at_level("INFO", logger="lumen.flowdash"):
        LumenAIServe(argparse.ArgumentParser()).invoke(parse(*argv))
    assert f"Reading the project file {project}" in caplog.text


def test_a_failing_command_reports_its_error_once(csv, monkeypatch, capsys):
    from lumen.command import ai

    (csv.parent / "flowdash.yaml").write_text(METRICS.replace("SUM(amount)", "SUM(amout)"))
    monkeypatch.setattr(sys, "argv", ["lumen", "serve", str(csv), "--flowdash"])
    with patch.object(Serve, "invoke", return_value=True), pytest.raises(SystemExit):
        ai.main()
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ") and not err.startswith("ERROR: ERROR")
