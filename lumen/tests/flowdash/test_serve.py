import argparse

from unittest.mock import patch

import bokeh.command.subcommands.serve as bokeh_serve
import pytest

pytest.importorskip("duckdb")

from panel.command import Serve
from panel_flowdash.agent import DashboardOps

from lumen.command.ai import LumenAIServe
from lumen.flowdash import (
    DATA_ID, SOURCE_ID, TABLE_ID, bound_sources,
)
from lumen.flowdash.app import LumenFlowDash, build_routes, source_refs


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
             patch("lumen.command.ai.get_available_llm") as get_llm:
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

    assert isinstance(editor, LumenFlowDash)
    assert app.agent is False
    assert SOURCE_ID in app.agent_sensitive_components
    (source,) = bound_sources(editor).values()
    assert list(bound_sources(editor)) == ["data"]
    (table,) = source.get_tables()

    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
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


def test_source_refs():
    assert list(source_refs(["a", "b", "c"])) == ["data", "data_2", "data_3"]
