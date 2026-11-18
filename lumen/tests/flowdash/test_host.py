"""
How the host hands Lumen nodes their sources (the editor's context) and
how the nodes' config checks refuse a config before it is applied.
"""
import asyncio
import time

from types import SimpleNamespace

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
    bound_sources, check_columns, check_sql_config, checks as checks_module,
    forget_probes, host_sources, unbind_source,
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


def place_unchecked(ops, component, config, name):
    """Place a node with a config its check would refuse, as a dashboard saved before the check existed has."""
    ops.add_node(component, {}, name=name)
    ops.editor._apply_config_patch(ops._instance(name), config)


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
    ops = DashboardOps(editor)
    with pytest.raises(OperationError, match="provides no Lumen sources"):
        ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    place_unchecked(ops, SOURCE_ID, {"ref": "data"}, "data")
    ops.add_node(SQL_ID, {"sql": QUERY, "table": "by_category"}, name="by_category")
    ops.connect("data.source", "by_category.source")
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
    with pytest.raises(OperationError, match="no source `extra`; its sources are: `data`"):
        ops.add_node(SOURCE_ID, {"ref": "extra"}, name="extra")
    place_unchecked(ops, SOURCE_ID, {"ref": "extra"}, "extra")
    assert "no source `extra`; its sources are: `data`" in failure(ops, "extra", "source")
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
    place_unchecked(ops, SOURCE_ID, {"ref": "data"}, "data")
    message = failure(ops, "data", "source")
    assert "`lumen serve <data> --flowdash`" in message
    assert '`editor_options = {"context": {"lumen.sources": {"data": source}}}`' in message
    assert "`__init__.py`" in message

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


def test_add_node_refuses_unsafe_sql_before_a_source_is_connected(source):
    ops = DashboardOps(FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"data": source}}))
    with pytest.raises(OperationError, match="read-only"):
        ops.add_node(SQL_ID, {"sql": "DROP TABLE sales"}, name="drop")
    with pytest.raises(OperationError, match="does not parse"):
        ops.add_node(SQL_ID, {"sql": "SELECT FROM WHERE ("}, name="garbled")
    ops.add_node(SQL_ID, {"sql": "SELECT * FROM glob('/etc/*')"}, name="duckdb_syntax")


def test_add_node_then_connect_runs_the_check_on_the_node(source):
    ops = DashboardOps(FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"data": source}}))
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(SQL_ID, {"sql": BROKEN, "table": "q"}, name="q")

    result = ops.connect("data.source", "q.source")

    if hasattr(ops, "last_check"):
        assert any("price" in problem for problem in result)
    assert "price" in failure(ops, "q", "data")
    assert any("price" in issue.message for issue in ops.validate())


def test_engine_errors_name_the_query_not_the_probe(source):
    problem = asyncio.run(check_sql_config({"sql": BROKEN}, {"source": source}))
    assert "price" in problem and "LIMIT 0" not in problem


def test_unchanged_sql_is_not_probed_again(source, executed):
    ops = query_dashboard(FlowDash(components=COMPONENTS, context={SOURCES_KEY: {"data": source}}))
    executed.clear()

    ops.set_config("by_category", {"sql": QUERY.replace("SUM", "MAX")})
    ops.set_config("by_category", {"limit": 1})
    ops.set_config("by_category", {"table": "renamed"})

    assert [q for q in executed if q.endswith("LIMIT 0")] == [f"{QUERY.replace('SUM', 'MAX')} LIMIT 0"]


async def test_a_probe_that_times_out_is_a_warning(source, monkeypatch):
    async def slow(sql, timeout, fetch=False):
        raise QueryTimeoutError("slow")

    monkeypatch.setattr(source, "execute_with_timeout", slow)
    result = await check_sql_config({"sql": BROKEN}, {"source": source})
    try:
        from panel_flowdash import CheckWarning
    except ImportError:
        assert result is None
    else:
        assert isinstance(result, CheckWarning) and "Could not verify the query" in result


async def test_probes_run_one_at_a_time_per_source(source, monkeypatch):
    running, most = 0, 0

    async def slow(sql, timeout, fetch=False):
        nonlocal running, most
        running += 1
        most = max(most, running)
        await asyncio.sleep(0.1)
        running -= 1

    monkeypatch.setattr(source, "execute_with_timeout", slow)
    queries = [f"SELECT category FROM sales WHERE amount > {i}" for i in range(4)]
    results = await asyncio.gather(*(check_sql_config({"sql": q}, {"source": source}) for q in queries))

    assert results == [None] * 4 and most == 1


async def test_probe_timeout_is_read_when_the_check_runs(source, monkeypatch):
    seen = []

    async def record(sql, timeout, fetch=False):
        seen.append(timeout)

    monkeypatch.setattr(source, "execute_with_timeout", record)
    monkeypatch.setattr(checks_module, "PROBE_TIMEOUT", 0.5)
    await check_sql_config({"sql": QUERY}, {"source": source})
    assert seen and seen[0] <= 0.5


class UncancellableSource(DuckDBSource):
    cancels_queries = False


async def test_sources_that_cannot_stop_a_query_are_never_probed(monkeypatch):
    source = UncancellableSource(uri=":memory:", tables={"sales": "SELECT * FROM (VALUES ('A', 10)) AS t(category, amount)"})

    async def fail(*args, **kwargs):
        raise AssertionError("probed")

    monkeypatch.setattr(source, "execute_with_timeout", fail)
    assert await check_sql_config({"sql": BROKEN}, {"source": source}) is None

    source.get_schema("sales")
    problem = await check_sql_config({"sql": BROKEN}, {"source": source})
    assert "price" in problem
    assert await check_sql_config({"sql": QUERY}, {"source": source}) is None
    assert "read-only" in await check_sql_config({"sql": "DROP TABLE sales"}, {"source": source})


def stored_schema_source(dialect, columns):
    """A source of ``dialect`` whose stored schema has a table ``sales`` with ``columns``."""
    return SimpleNamespace(dialect=dialect, _get_schema_cache=lambda: {"sales": {c: {} for c in columns}})


@pytest.mark.parametrize("dialect, columns, sql", [
    ("postgres", ["Amount"], 'SELECT "Amount" FROM sales'),
    ("postgres", ["amount"], "SELECT Amount FROM sales"),
    ("postgres", ["Amount"], "SELECT amount FROM sales"),
    ("mysql", ["Amount"], "SELECT amount FROM sales"),
    ("mysql", ["amount"], "SELECT `Amount` FROM sales"),
    ("snowflake", ["AMOUNT"], "SELECT amount FROM sales"),
    ("snowflake", ["AMOUNT"], 'SELECT "AMOUNT" FROM sales'),
    ("snowflake", ["Amount"], 'SELECT "Amount", amount FROM sales'),
    ("duckdb", ["Amount"], "SELECT amount, \"AMOUNT\" FROM sales"),
    ("duckdb", ["amount"], 'SELECT s."Amount" FROM sales AS s'),
])
def test_column_check_never_refuses_on_case_alone(dialect, columns, sql):
    check_columns(sql, stored_schema_source(dialect, columns))


@pytest.mark.parametrize("dialect, sql", [
    ("postgres", 'SELECT "Price" FROM sales'),
    ("mysql", "SELECT price FROM sales"),
    ("snowflake", "SELECT PRICE FROM sales"),
    ("duckdb", 'SELECT "price" FROM sales'),
])
def test_column_check_refuses_columns_missing_in_every_case(dialect, sql):
    with pytest.raises(ValueError, match="(?i)price"):
        check_columns(sql, stored_schema_source(dialect, ["Amount"]))


async def test_a_failed_probe_rewrite_does_not_hold_the_sources_lock(source, monkeypatch):
    from lumen.transforms.sql import SQLLimit

    def broken(self, sql):
        raise ValueError("cannot rewrite")

    with monkeypatch.context() as patch:
        patch.setattr(SQLLimit, "apply", broken)
        assert "cannot rewrite" in await check_sql_config({"sql": QUERY}, {"source": source})
    assert await check_sql_config({"sql": QUERY}, {"source": source}) is None


async def test_accepted_probes_are_forgotten_when_tables_change_or_on_refresh(source, executed):
    sql = "SELECT category, amount FROM sales"
    probes = lambda: [q for q in executed if q.endswith("LIMIT 0")]  # noqa: E731

    assert await check_sql_config({"sql": sql}, {"source": source}) is None
    assert await check_sql_config({"sql": sql}, {"source": source}) is None
    assert len(probes()) == 1

    forget_probes(source)
    assert await check_sql_config({"sql": sql}, {"source": source}) is None
    assert len(probes()) == 2

    source.tables = {**source.tables, "other": "SELECT 1 AS x"}
    assert await check_sql_config({"sql": sql}, {"source": source}) is None
    assert len(probes()) == 3
