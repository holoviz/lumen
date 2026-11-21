import argparse
import json

from pathlib import Path
from unittest.mock import patch

import bokeh.command.subcommands.serve as bokeh_serve
import pytest

pytest.importorskip("duckdb")

from panel.command import Serve
from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps

from lumen.ai.llm import OpenAI
from lumen.command.ai import LumenAIServe
from lumen.flowdash import (
    DATA_ID, SOURCE_ALIASES_KEY, SOURCE_ID, TABLE_ID, bound_sources,
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
    assert SOURCE_ID in app.agent_sensitive_components
    (source,) = bound_sources(editor).values()
    assert list(bound_sources(editor)) == ["sales"]
    (table,) = source.get_tables()

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


def test_one_agent_model_has_no_picker(csv):
    pytest.importorskip("pydantic_ai")
    app = build_routes([str(csv)], db_path=csv.parent / "dash.db", agent_model=["test=Test"])["/"]()

    assert app.agent.model.model_name == "test"
    assert app.agent_model == "test"
    assert app.agent_models == []
    assert "models" not in app._agent_session.drawer.chat_params


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


def test_source_refs_come_from_file_and_database_names(served):
    assert list(source_refs(resolve_sources(["plants.csv"]))) == ["plants"]
    assert list(source_refs(resolve_sources(["shop.db", "plants.csv"]))) == ["shop", "plants"]
    assert list(source_refs(resolve_sources(["plants.csv", "turbines.csv"]))) == ["files"]


def test_source_refs_do_not_depend_on_the_order_of_the_data(served):
    first = source_refs(resolve_sources(["shop.db", "plants.csv", "turbines.csv"]))
    second = source_refs(resolve_sources(["turbines.csv", "plants.csv", "shop.db"]))
    assert sorted(first) == sorted(second) == ["files", "shop"]
    assert first["files"].get_tables() == ["plants_csv", "turbines_csv"]


def test_source_refs_name_sources_by_their_name_and_deduplicate():
    named = DuckDBSource(uri=":memory:", tables={"t": "SELECT 1 AS x"}, name="Sales Data")
    one = DuckDBSource(uri=":memory:", tables={"b": "SELECT 1 AS x"})
    other = DuckDBSource(uri=":memory:", tables={"a": "SELECT 1 AS x"})
    clash = DuckDBSource(uri=":memory:", tables={"a": "SELECT 2 AS x"}, name="a")
    several = DuckDBSource(uri=":memory:", tables={"a": "SELECT 1 AS x", "b": "SELECT 1 AS x"})

    assert list(source_refs([named, one, other, several])) == ["sales_data", "b", "a", "duckdb"]
    # Deduplicated by location, type and tables, not by position.
    assert source_refs([clash, other]) == {"a": clash, "a_2": other}
    assert source_refs([other, clash]) == {"a_2": other, "a": clash}


def test_source_aliases_keep_positional_refs_and_file_stems(served):
    refs = source_refs(resolve_sources(["shop.db", "plants.csv", "turbines.csv"]))
    assert source_aliases(refs) == {
        "data": "shop", "data_2": "files", "plants": "files", "turbines": "files",
    }
    single = source_refs(resolve_sources(["plants.csv"]))
    assert source_aliases(single) == {"data": "plants", "files": "plants"}


def test_dashboard_saved_with_positional_refs_still_loads(served):
    """
    The fixture was saved when ``lumen serve shop.db plants.csv`` named its
    sources ``data`` (shop.db, read first) and ``data_2`` (the CSV files).
    """
    routes = build_routes(["plants.csv", "shop.db"], db_path=served / "dash.db")
    editor = routes["/"]()._editor
    assert list(bound_sources(editor)) == ["shop", "plants"]
    assert editor.context[SOURCE_ALIASES_KEY] == {"data": "shop", "data_2": "plants", "files": "plants"}

    editor.import_dashboard(json.loads((FIXTURES / "positional_refs.json").read_text()))
    ops = DashboardOps(editor)

    assert ops._item(ops._instance("first"))["config"]["ref"] == "data"
    assert output(ops, "first", "source") is bound_sources(editor)["shop"]
    assert output(ops, "second", "source") is bound_sources(editor)["plants"]
    assert output(ops, "order_count", "data")["n"].tolist() == [2]
    assert output(ops, "plant_count", "data")["n"].tolist() == [3]
    assert ops.validate() == []


def test_stable_ref_wins_over_a_positional_alias(served):
    (served / "data.csv").write_text("x\n1\n")
    editor = build_routes(["shop.db", "data.csv"], db_path=served / "dash.db")["/"]()._editor
    assert list(bound_sources(editor)) == ["shop", "data"]
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="first")
    assert output(ops, "first", "source").get_tables() == ["data_csv"]
