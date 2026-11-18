"""
How the host hands Lumen nodes their sources (the editor's context) and
how the nodes' config checks refuse a config before it is applied.
"""
import asyncio
import time

import panel as pn
import pytest

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import (
    DashboardOps, OperationError, dashboard_to_yaml,
)
from panel_flowdash.app import FlowDashApp
from panel_flowdash.dashboard_store import DashboardStore
from panel_flowdash.ui.status_bar import select_on_canvas

from lumen.flowdash import (
    COMPONENTS, DATA_ID, SOURCE_ID, SOURCES_KEY, SQL_ID, TABLE_ID, bind_source,
    bound_sources, check_sql_config, host_sources, unbind_source,
)
from lumen.flowdash.app import build_routes
from lumen.pipeline import Pipeline
from lumen.sources.base import QueryTimeoutError
from lumen.sources.duckdb import DuckDBSource

QUERY = "SELECT category, SUM(amount) AS total FROM sales GROUP BY category"
BROKEN = "SELECT category, SUM(price) AS total FROM sales GROUP BY category"


def output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


def failure(ops, name, port):
    value = output(ops, name, port)
    assert not isinstance(value, Pipeline)
    return str(getattr(value, "exception", value))


def query_dashboard(editor):
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(SQL_ID, {"sql": QUERY, "table": "by_category"}, name="by_category")
    ops.connect("data.source", "by_category.source")
    return ops


async def settle(seconds=0.5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        await asyncio.sleep(0.01)


def test_source_nodes_read_the_editor_context(source):
    editor = FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"data": source}})
    ops = query_dashboard(editor)

    assert output(ops, "data", "source") is source
    assert len(output(ops, "by_category", "data")) == 2
    assert bound_sources(editor) == {"data": source}
    assert host_sources() == {}
    assert host_sources(editor.context) == {"data": source}


def test_binding_after_placing_reaches_the_node(source):
    editor = FlowDash(components=COMPONENTS)
    ops = query_dashboard(editor)
    assert "provides no Lumen sources" in failure(ops, "data", "source")

    bind_source(editor, source)
    assert editor.context[SOURCES_KEY] == {"data": source}
    assert len(output(ops, "by_category", "data")) == 2


def test_preview_scratch_editor_resolves_the_hosts_sources(source):
    ops = query_dashboard(FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"data": source}}))
    changed = dashboard_to_yaml(ops.editor).replace("SUM(amount)", "MAX(amount)")

    preview = ops.preview(changed)

    assert preview["status"] == "ok", preview
    assert preview["nodes"]["by_category"]["outputs"]["data"]["type"] != "failure"
    assert "MAX" not in ops._item(ops._instance("by_category"))["config"]["sql"]


def test_preview_does_not_change_the_editors_sources(source):
    editor = FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"data": source}})
    scratch = DashboardOps(editor)._scratch_editor()
    try:
        bind_source(scratch, DuckDBSource(uri=":memory:", tables={"t": "SELECT 1 AS x"}), "other")
        unbind_source(scratch, "data")
    finally:
        scratch.dispose()

    assert bound_sources(editor) == {"data": source}


def test_sessions_keep_their_own_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sales.csv").write_text("category,amount\nA,10\nB,20\nA,5\n")
    routes = build_routes(["sales.csv"], db_path=tmp_path / "dash.db")
    first, second = routes["/"]()._editor, routes["/"]()._editor

    assert first.context[SOURCES_KEY] is not second.context[SOURCES_KEY]
    extra = DuckDBSource(uri=":memory:", tables={"t": "SELECT 1 AS x"})
    bind_source(first, extra, "extra")
    unbind_source(first, "data")
    assert list(bound_sources(first)) == ["extra"]
    assert list(bound_sources(second)) == ["data"]

    ops = DashboardOps(second)
    ops.add_node(SOURCE_ID, {"ref": "extra"}, name="extra")
    assert "no source 'extra'; its sources are: data" in failure(ops, "extra", "source")
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales_csv"}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("sales.pipeline", "rows.pipeline")
    assert len(output(ops, "rows", "data")) == 3


def test_flowdash_app_without_sources_says_how_to_provide_them(tmp_path, source):
    store = DashboardStore(str(tmp_path / "dashboards.db"))
    plain = FlowDashApp(project_dir=str(tmp_path), registry={}, store=store, component_libraries=["lumen.flowdash"])
    ops = DashboardOps(plain._editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    assert 'editor_options = {"context": {"lumen.sources"' in failure(ops, "data", "source")

    configured = FlowDashApp(
        project_dir=str(tmp_path), registry={}, store=store, component_libraries=["lumen.flowdash"],
        editor_options={"context": {SOURCES_KEY: {"data": source}}},
    )
    assert len(output(query_dashboard(configured._editor), "by_category", "data")) == 2


def test_set_config_refuses_sql_the_engine_rejects(source, executed):
    ops = query_dashboard(FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"data": source}}))
    executed.clear()

    with pytest.raises(OperationError, match="price"):
        ops.set_config("by_category", {"sql": BROKEN})

    assert ops._item(ops._instance("by_category"))["config"]["sql"] == QUERY
    assert len(output(ops, "by_category", "data")) == 2
    assert executed == [" ".join(f"{BROKEN} LIMIT 0".split())]


async def test_sql_check_passes_when_the_probe_times_out(source, monkeypatch):
    async def slow(sql, timeout, fetch=False):
        raise QueryTimeoutError("slow")

    monkeypatch.setattr(source, "execute_with_timeout", slow)
    assert await check_sql_config({"sql": BROKEN}, {"source": source}) is None


async def test_flowdash_agent_set_config_refuses_broken_sql(source):
    pytest.importorskip("pydantic_ai")
    from panel_flowdash.agent import FlowDashDeps, create_agent
    from pydantic_ai.messages import (
        ModelResponse, TextPart, ToolCallPart, ToolReturnPart,
    )
    from pydantic_ai.models.function import FunctionModel

    ops = query_dashboard(FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"data": source}}))
    returned = []

    def respond(messages, info):
        if len(returned) == 0:
            returned.append(None)
            return ModelResponse([ToolCallPart("set_config", {"name": "by_category", "config": {"sql": BROKEN}})])
        returned.append(messages[-1].parts[0])
        return ModelResponse([TextPart("Done.")])

    await create_agent(FunctionModel(respond)).run("Sum the price instead", deps=FlowDashDeps(ops))

    result = returned[-1]
    assert isinstance(result, ToolReturnPart) and result.outcome == "retried"
    assert "was not changed" in result.content and "price" in result.content
    assert "Updated the config" not in result.content
    assert ops._item(ops._instance("by_category"))["config"]["sql"] == QUERY


async def test_inspector_holds_back_sql_the_engine_rejects(source):
    editor = FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"data": source}}, notifications=False)
    ops = query_dashboard(editor)
    node = ops._instance("by_category")
    select_on_canvas(editor._flow, node_id=node)
    form = next(iter(pn.Column(*editor._inspector._body).select(pn.Param)))

    def alerts():
        return [
            a.object for a in pn.Column(*editor._inspector._body).select()
            if "fd-config-check-alert" in getattr(a, "css_classes", [])
        ]

    form._widgets["sql"].value = BROKEN
    await settle()
    assert editor._component_instances[node].sql == QUERY
    assert len(alerts()) == 1 and "price" in alerts()[0]

    fixed = QUERY.replace("SUM", "MAX")
    form._widgets["sql"].value = fixed
    await settle()
    assert editor._component_instances[node].sql == fixed
    assert alerts() == []
