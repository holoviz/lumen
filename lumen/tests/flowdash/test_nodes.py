import asyncio
import threading
import time

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import (
    DashboardOps, OperationError, dashboard_from_yaml, dashboard_to_yaml,
)
from panel_flowdash.ui.node_preview import hook_preview, summarize_value

from lumen.flowdash import (
    COMPONENTS, DATA_ID, FILTER_ID, NODE_TIMEOUT, PREVIEW_ROWS, RESULT_LIMIT,
    SOURCE_ID, SQL_ID, TABLE_ID, VIEW_ID, SQLNode, ViewNode, bind_source,
    check_chart, check_query, compile_sql, nodes as flowdash_nodes,
    primed_result, result_sql, set_row_ceiling,
)
from lumen.flowdash.nodes import responsive_spec
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.sources.duckdb import DuckDBSource

QUERY = "SELECT category, SUM(amount) AS total FROM sales GROUP BY category"
SPEC = {"mark": "bar", "encoding": {"x": {"field": "category", "type": "nominal"}, "y": {"field": "total", "type": "quantitative"}}}


def output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


def failure(ops, name, port="data"):
    value = output(ops, name, port)
    assert not isinstance(value, Pipeline)
    return str(getattr(value, "exception", value))


def without_turns(text):
    data = yaml.safe_load(text)
    data.pop("turns", None)
    return data


def view_error(ops, name):
    return ops.editor._component_instances[ops._instance(name)]._error or ""


def place_unchecked(ops, component, config, name):
    """Place a node with a config its check would refuse, as a dashboard saved before the check existed has."""
    ops.add_node(component, {}, name=name)
    ops.editor._apply_config_patch(ops._instance(name), config)


def editor_with(source, ref="data"):
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source, ref)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": ref}, name=ref)
    return ops


def dashboard(source, sql=QUERY):
    ops = editor_with(source)
    ops.add_node(SQL_ID, {"sql": sql, "table": "by_category"}, name="by_category")
    ops.connect("data.source", "by_category.source")
    ops.add_node(VIEW_ID, {"spec": SPEC}, name="chart")
    ops.connect("by_category.pipeline", "chart.pipeline")
    return ops


def test_sql_node_runs_query_once_in_source_engine(source, executed):
    ops = dashboard(source)

    pipeline = output(ops, "by_category", "pipeline")
    data = output(ops, "by_category", "data")

    assert isinstance(pipeline, Pipeline)
    assert output(ops, "chart", "pipeline") is pipeline
    assert sorted(map(tuple, data.values.tolist())) == [("A", 15), ("B", 20)]
    assert [q for q in executed if not q.endswith("LIMIT 0")] == [" ".join(result_sql(source, QUERY).split())]


def test_primed_result_is_not_executed_again(source, executed):
    statement = result_sql(source, QUERY)
    frame = source.execute(statement)
    executed.clear()
    with primed_result(source, statement, frame):
        pipeline = compile_sql(source, QUERY, "by_category")
    assert pipeline.data is frame
    assert executed == []


def test_sql_node_limit_caps_rows(source):
    ops = dashboard(source, "SELECT * FROM sales")
    ops.set_config("by_category", {"limit": 2})
    assert len(output(ops, "by_category", "data")) == 2


def test_sql_node_limit_cannot_exceed_host_ceiling():
    big = DuckDBSource(uri=":memory:", tables={"big": "SELECT * FROM range(300000) AS t(i)"})
    ops = editor_with(big)
    ops.add_node(SQL_ID, {"sql": "SELECT * FROM big", "limit": 10**12}, name="rows")
    ops.connect("data.source", "rows.source")
    assert len(output(ops, "rows", "data")) == RESULT_LIMIT

    set_row_ceiling(ops.editor, 200_000)
    ops.set_config("rows", {"limit": None})
    assert len(output(ops, "rows", "data")) == 200_000
    ops.set_config("rows", {"limit": 50})
    assert len(output(ops, "rows", "data")) == 50


def test_node_path_stops_slow_query_after_node_timeout(source):
    ops = dashboard(source)
    endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n, sales"
    started = time.monotonic()
    ops.set_config("by_category", {"sql": endless})
    assert "did not finish" in failure(ops, "by_category")
    assert time.monotonic() - started < NODE_TIMEOUT + 5


def test_sql_node_reports_failed_source(source):
    ops = editor_with(source)
    place_unchecked(ops, SOURCE_ID, {"ref": "missing"}, "other")
    ops.add_node(SQL_ID, {"sql": QUERY}, name="q")
    ops.connect("other.source", "q.source")
    assert "The app provides no source `missing`; its sources are: `data`." in failure(ops, "q")


def test_sql_node_times_out(source):
    endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n, sales"
    with pytest.raises(Exception, match="did not finish within 0.5 seconds"):
        compile_sql(source, endless, "endless", timeout=0.5)


def test_sql_node_table_does_not_shadow_source_table(source):
    ops = editor_with(source)
    ops.add_node(SQL_ID, {"sql": "SELECT * FROM sales WHERE amount > 5", "table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")

    assert len(output(ops, "sales", "data")) == 2


@pytest.mark.parametrize("sql, error", [
    ("DROP TABLE sales", "read-only"),
    ("SELECT 1; DELETE FROM sales", "read-only"),
    ("CREATE TABLE copy AS SELECT * FROM sales", "read-only"),
    ("INSERT INTO sales VALUES ('C', 1)", "read-only"),
    ("SELECT 42 AS answer", "reads no table"),
    ("SELECT * FROM read_text('/etc/hosts')", "READ_TEXT"),
    ("SELECT * FROM read_blob('/etc/hosts')", "READ_BLOB"),
    ("SELECT * FROM glob('/etc/*')", "GLOB"),
    ("SELECT * FROM read_csv('/etc/hosts')", "READ_CSV"),
    ("SELECT * FROM '/etc/hosts'", "not a table of this source"),
    ("SELECT * FROM 'https://example.invalid/x.parquet'", "not a table of this source"),
    ("SELECT * FROM duckdb_settings()", "duckdb_settings"),
    ("SELECT * FROM information_schema.tables", "information_schema.tables"),
    ("SELECT * FROM other_table", "not a table of this source"),
    ("WITH x AS (SELECT 1 AS x) SELECT * FROM x", "reads no table"),
    ("WITH a AS (SELECT 1 AS x), b AS (SELECT * FROM a) SELECT * FROM b", "reads no table"),
    ("WITH sales AS (SELECT 42 AS total) SELECT * FROM sales", "reads no table"),
    ("WITH unused AS (SELECT * FROM sales) SELECT 42 AS answer FROM (SELECT 1) AS t", "reads no table"),
    ("SELECT current_setting('home_directory') AS h FROM sales", "current_setting"),
    ("SELECT getenv('HOME') AS h FROM sales", "getenv"),
    ("SELECT * FROM other.sales", "not a table of this source"),
])
def test_sql_node_refuses_unsafe_queries_on_every_path(source, executed, sql, error):
    ops = editor_with(source)
    place_unchecked(ops, SQL_ID, {"sql": sql, "table": "by_category"}, "by_category")
    ops.connect("data.source", "by_category.source")
    assert error in failure(ops, "by_category")
    assert ops.preview()["nodes"]["by_category"]["outputs"]["data"]["type"] == "failure"

    ops.set_config("by_category", {"sql": QUERY})
    assert len(output(ops, "by_category", "data")) == 2
    with pytest.raises(OperationError, match="was not changed") as refused:
        ops.set_config("by_category", {"sql": sql})
    assert error in str(refused.value)
    assert ops._item(ops._instance("by_category"))["config"]["sql"] == QUERY
    assert len(output(ops, "by_category", "data")) == 2
    ops.editor._apply_config_patch(ops._instance("by_category"), {"sql": sql})
    assert error in failure(ops, "by_category")
    assert any(error in issue.message for issue in ops.validate())

    text = dashboard_to_yaml(ops.editor)
    imported = FlowDash(components=COMPONENTS)
    bind_source(imported, source)
    imported.import_dashboard(dashboard_from_yaml(text))
    assert error in failure(DashboardOps(imported), "by_category")
    assert not any(word in query.upper() for query in executed for word in ("DROP", "INSERT", "CREATE", "DELETE", "/ETC/", "HTTPS:", "DUCKDB_SETTINGS", "INFORMATION_SCHEMA"))


@pytest.mark.parametrize("sql", [
    "SELECT * FROM main.sales",
    "SELECT * FROM memory.main.sales",
    "WITH a AS (SELECT * FROM sales), b AS (SELECT * FROM a) SELECT * FROM b",
    "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 3) SELECT * FROM n, sales",
])
def test_check_query_accepts(source, sql):
    check_query(sql, source)


def test_check_query_allows_ctes_and_declared_table_functions():
    fixtures = Path(__file__).parents[3] / "tests" / "evals" / "fixtures"
    orders = f"read_csv('{fixtures / 'orders.csv'}')"
    source = DuckDBSource(uri=":memory:", tables={"orders_csv": orders})

    check_query("WITH paid AS (SELECT * FROM orders_csv WHERE status = 'paid') SELECT COUNT(*) FROM paid", source)
    check_query(f"SELECT COUNT(*) FROM {orders}", source)
    with pytest.raises(ValueError, match="READ_CSV"):
        check_query(f"SELECT * FROM read_csv('{fixtures / 'refunds.json'}')", source)


def test_source_node_has_no_spec_to_build_a_source_from(source):
    ops = editor_with(source)
    with pytest.raises(OperationError, match="no config field 'spec'"):
        ops.add_node(SOURCE_ID, {"spec": {"type": "duckdb", "uri": ":memory:", "initializers": ["COPY (SELECT 1) TO 'x.csv'"]}}, name="evil")
    with pytest.raises(OperationError, match="no config field 'spec'"):
        ops.set_config("data", {"spec": {"type": "duckdb", "uri": ":memory:"}})


def test_source_node_reports_unbound_ref(source):
    ops = editor_with(source)
    with pytest.raises(OperationError, match="no source `missing`; its sources are: `data`"):
        ops.add_node(SOURCE_ID, {"ref": "missing"}, name="other")
    place_unchecked(ops, SOURCE_ID, {"ref": "missing"}, "other")

    assert "The app provides no source `missing`" in failure(ops, "other", "source")


def test_bindings_belong_to_their_editor():
    first = DuckDBSource(uri=":memory:", tables={"a_table": "SELECT 1 AS x"})
    second = DuckDBSource(uri=":memory:", tables={"b_table": "SELECT 2 AS y"})
    a, b = editor_with(first), editor_with(second)

    a.replace_yaml(a.get_dashboard())
    assert output(a, "data", "source") is first
    assert output(b, "data", "source") is second

    b.editor.dispose()
    del b
    a.replace_yaml(a.get_dashboard())
    assert output(a, "data", "source") is first

    unbound = FlowDash(components=COMPONENTS)
    unbound.import_dashboard(dashboard_from_yaml(dashboard_to_yaml(a.editor)))
    assert "provides no Lumen sources" in failure(DashboardOps(unbound), "data", "source")


def test_yaml_round_trip_reruns_pushed_down_query(source, executed):
    text = dashboard_to_yaml(dashboard(source).editor)
    executed.clear()

    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    editor.import_dashboard(dashboard_from_yaml(text))
    ops = DashboardOps(editor)

    assert without_turns(dashboard_to_yaml(editor)) == without_turns(text)
    data_node = yaml.safe_load(text)["nodes"]["data"]
    assert (data_node["component"], data_node["config"]) == (SOURCE_ID, {"ref": "data"})
    assert output(ops, "chart", "pipeline") is output(ops, "by_category", "pipeline")
    assert len(output(ops, "by_category", "data")) == 2
    assert executed == [" ".join(result_sql(source, QUERY).split())]


@pytest.mark.parametrize("spec, error", [
    ({**SPEC, "data": {"url": "https://example.invalid/x.csv"}}, "remove `data`"),
    ({"mark": "image", "encoding": {"url": {"value": "https://example.invalid/i.png"}}}, "URLs"),
    ({**SPEC, "encoding": {**SPEC["encoding"], "href": {"value": "https://example.invalid/"}}}, "URLs"),
    ({"mark": "bar", "encoding": {"x": {"field": "revenue", "type": "nominal"}}}, "revenue"),
])
def test_view_node_checks_spec_on_every_path(source, spec, error):
    ops = dashboard(source)
    with pytest.raises(OperationError, match="was not changed") as refused:
        ops.set_config("chart", {"spec": spec})
    assert error in str(refused.value)
    ops.editor._apply_config_patch(ops._instance("chart"), {"spec": spec})
    assert error in view_error(ops, "chart")

    ops.add_node(VIEW_ID, {"spec": spec}, name="other_chart")
    ops.connect("by_category.pipeline", "other_chart.pipeline")
    assert error in view_error(ops, "other_chart")


@pytest.mark.parametrize("spec", [
    {"layer": [SPEC, {"mark": "rule", "data": {"values": [{"total": 1}]}, "encoding": {"y": {"field": "total", "type": "quantitative"}}}]},
    {**SPEC, "data": {"sequence": {"start": 0, "stop": 10}}},
    {**SPEC, "datasets": {"x": [{"total": 1}]}},
    {**SPEC, "transform": [{"lookup": "category", "from": {"data": {"name": "other"}, "key": "category", "fields": ["total"]}}]},
])
def test_check_chart_refuses_data_at_any_depth(spec):
    with pytest.raises(ValueError, match="remove `data` and `datasets`"):
        check_chart(spec, ["category", "total"])


def test_source_node_picks_up_a_binding_made_after_import(source):
    text = dashboard_to_yaml(dashboard(source).editor)
    editor = FlowDash(components=COMPONENTS)
    editor.import_dashboard(dashboard_from_yaml(text))
    ops = DashboardOps(editor)
    assert "editor_options = {\"context\": {\"lumen.sources\"" in failure(ops, "data", "source")

    bind_source(editor, source)
    assert output(ops, "data", "source") is source
    assert len(output(ops, "by_category", "data")) == 2


def test_view_node_drops_named_data(source):
    ops = dashboard(source)
    ops.set_config("chart", {"spec": {**SPEC, "data": {"name": "by_category"}}})
    view = ops.editor._component_instances[ops._instance("chart")]._object
    assert "data" not in view.spec


@pytest.mark.parametrize("size, rendered, sizing_mode", [
    ({}, {"width": "container", "height": "container"}, "stretch_both"),
    ({"width": 300}, {"width": 300, "height": "container"}, "stretch_height"),
    ({"height": 200}, {"width": "container", "height": 200}, "stretch_width"),
    ({"width": 300, "height": 200}, {"width": 300, "height": 200}, None),
])
def test_view_node_fills_its_tile_unless_the_spec_sets_a_size(source, size, rendered, sizing_mode):
    ops = dashboard(source)
    ops.set_config("chart", {"spec": {**SPEC, **size}})

    pane = ops.editor._component_instances[ops._instance("chart")]._object.panel
    assert {key: pane.object[key] for key in ("width", "height")} == rendered
    assert pane.get_root().sizing_mode == sizing_mode
    assert ops.node_config("chart")["spec"] == {**SPEC, **size}


@pytest.mark.parametrize("spec", [
    {"hconcat": [SPEC, SPEC]},
    {"facet": {"field": "category"}, "spec": SPEC},
    {"repeat": ["total"], "spec": SPEC},
    *({**SPEC, "encoding": {**SPEC["encoding"], channel: {"field": "category"}}} for channel in ("row", "column", "facet")),
    {"layer": [{**SPEC, "encoding": {**SPEC["encoding"], "column": {"field": "category"}}}]},
    {**SPEC, "autosize": "pad"},
    {**SPEC, "autosize": {"type": "none"}},
])
def test_responsive_spec_leaves_compound_and_fixed_charts_alone(spec):
    assert responsive_spec(spec) is spec


@pytest.mark.parametrize("view, rendered", [
    ({"continuousWidth": 300}, {"height": "container"}),
    ({"discreteWidth": 300}, {"height": "container"}),
    ({"continuousHeight": 200}, {"width": "container"}),
    ({"discreteHeight": {"step": 20}}, {"width": "container"}),
    ({"continuousWidth": 300, "continuousHeight": 200}, {}),
])
def test_responsive_spec_keeps_view_sizes_from_config(view, rendered):
    spec = {**SPEC, "config": {"view": view}}
    assert responsive_spec(spec) == {**rendered, **spec}


def test_responsive_spec_does_not_change_the_spec():
    spec = dict(SPEC)
    assert responsive_spec(spec) == {**SPEC, "width": "container", "height": "container"}
    assert spec == SPEC


def test_a_sql_node_refuses_a_pipeline_on_its_source_input(source):
    ops = dashboard(source)
    ops.add_node(SQL_ID, {"sql": "SELECT 1 AS x FROM sales"}, name="other")

    # panel-flowdash does not type-check the members of a multiple input, so the node refuses it.
    ops.connect("by_category.pipeline", "other.source")
    assert "source takes a Lumen SQL source or a SQL node's result; by_category.pipeline gives a Lumen pipeline." in failure(ops, "other")


def _instance(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def test_source_node_preview_uses_cached_schema_without_querying(source, executed):
    ops = editor_with(source)
    node = _instance(ops, "data")
    executed.clear()

    assert node.__flowdash_preview__() == "1 table: sales"
    assert executed == []

    source.get_schema("sales")
    executed.clear()

    cached = node.__flowdash_preview__(output="source")
    assert list(cached.columns) == ["table", "rows", "columns", "column names"]
    assert cached.to_dict("records") == [
        {"table": "sales", "rows": 3, "columns": 2, "column names": "category, amount"}
    ]
    assert executed == []

    preview = hook_preview(node)
    assert preview is not None and preview.kind == "table"


def test_source_node_preview_in_memory_and_error_states(source):
    mem = InMemorySource(tables={"items": pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})})
    ops = editor_with(mem)
    preview = _instance(ops, "data").__flowdash_preview__()
    assert preview.to_dict("records") == [{"table": "items", "rows": 4, "columns": 2, "column names": "a, b"}]

    ops.add_node(SOURCE_ID, {}, name="empty")
    assert _instance(ops, "empty").__flowdash_preview__() is None

    place_unchecked(ops, SOURCE_ID, {"ref": "missing"}, "broken")
    assert _instance(ops, "broken").__flowdash_preview__() is None


def test_sql_node_preview_returns_dataframe_without_requerying(source, executed):
    ops = dashboard(source)
    node = _instance(ops, "by_category")
    executed.clear()

    for arg in (None, "data", "pipeline"):
        frame = node.__flowdash_preview__() if arg is None else node.__flowdash_preview__(output=arg)
        assert isinstance(frame, pd.DataFrame)
        assert not isinstance(frame, Pipeline)
        assert len(frame) == 2
    assert executed == []

    rendered = hook_preview(node)
    assert rendered is not None and rendered.kind == "table"
    assert "<Pipeline" not in rendered.html

    ops.add_node(SQL_ID, {}, name="empty_sql")
    assert _instance(ops, "empty_sql").__flowdash_preview__() is None

    place_unchecked(ops, SQL_ID, {"sql": "DROP TABLE sales"}, "bad_sql")
    ops.connect("data.source", "bad_sql.source")
    assert _instance(ops, "bad_sql").__flowdash_preview__() is None


def numbers_editor(rows=120):
    big = DuckDBSource(uri=":memory:", tables={
        "nums": f"SELECT i, CASE WHEN i % 2 = 0 THEN 'even' ELSE 'odd' END AS kind FROM range({rows}) AS t(i)"
    })
    ops = editor_with(big)
    ops.add_node(TABLE_ID, {"table": "nums"}, name="nums")
    ops.connect("data.source", "nums.source")
    return big, ops


def slow_queries(monkeypatch, source, delay):
    """Make each query of ``source`` take ``delay`` seconds longer, counting them."""
    calls = []
    original = source.execute_with_timeout

    async def slow(sql, timeout, *args, **kwargs):
        calls.append(sql)
        await asyncio.sleep(delay)
        return await original(sql, timeout, *args, **kwargs)

    monkeypatch.setattr(source, "execute_with_timeout", slow)
    return calls


def refreshes(node):
    revisions = []
    node.param.watch(lambda event: revisions.append(event.new), "_preview_rev")
    return revisions


def test_table_and_filter_node_previews_cap_rows_and_cache(executed):
    _, ops = numbers_editor()
    ops.add_node(FILTER_ID, {"field": "kind"}, name="filtered")
    ops.connect("nums.pipeline", "filtered.pipeline")
    ops.add_node("Widgets/Select", {"default_options": ["even", "odd"]}, name="pick")
    ops.connect("pick.selected", "filtered.value")
    _instance(ops, "pick")._widget.value = "even"

    table_node = _instance(ops, "nums")
    filter_node = _instance(ops, "filtered")
    executed.clear()

    t1 = table_node.__flowdash_preview__()
    assert isinstance(t1, pd.DataFrame) and len(t1) == PREVIEW_ROWS
    after_first = list(executed)
    assert len(after_first) == 1

    t2 = table_node.__flowdash_preview__(output="pipeline")
    assert len(t2) == PREVIEW_ROWS
    assert executed == after_first

    f1 = filter_node.__flowdash_preview__(output="filtered")
    assert isinstance(f1, pd.DataFrame) and len(f1) == PREVIEW_ROWS
    assert set(f1["kind"]) == {"even"}
    after_filter = list(executed)

    f2 = filter_node.__flowdash_preview__()
    assert len(f2) == PREVIEW_ROWS
    assert executed == after_filter

    assert hook_preview(table_node).kind == "table"
    assert hook_preview(filter_node).kind == "table"

    ops.add_node(TABLE_ID, {}, name="empty_table")
    ops.add_node(FILTER_ID, {}, name="empty_filter")
    assert _instance(ops, "empty_table").__flowdash_preview__() is None
    assert _instance(ops, "empty_filter").__flowdash_preview__() is None

    place_unchecked(ops, TABLE_ID, {"table": "missing_tbl"}, "bad_table")
    ops.connect("data.source", "bad_table.source")
    assert _instance(ops, "bad_table").__flowdash_preview__() is None


def test_table_preview_caption_counts_rows_not_the_preview_cap():
    _, ops = numbers_editor(rows=120)
    node = _instance(ops, "nums")

    capped = node.__flowdash_preview__()
    assert len(capped) == PREVIEW_ROWS
    assert summarize_value(capped) == f"{PREVIEW_ROWS}+ rows, 2 columns"
    assert f"{PREVIEW_ROWS} rows" not in hook_preview(node).html

    # Once a reader has fetched all the rows, the caption gives their number.
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("nums.pipeline", "rows.pipeline")
    full = node.__flowdash_preview__()
    assert len(full) == PREVIEW_ROWS
    assert summarize_value(full) == "120 rows, 2 columns"
    assert type(full.head(2)) is pd.DataFrame

    ops.set_config("nums", {"limit": 10})
    assert summarize_value(_instance(ops, "nums").__flowdash_preview__()) == "10 rows, 2 columns"


def test_table_preview_on_the_event_loop_does_not_wait_for_a_slow_query(monkeypatch):
    big, ops = numbers_editor()
    node = _instance(ops, "nums")
    calls = slow_queries(monkeypatch, big, 1)
    revisions = refreshes(node)

    async def card_refreshes():
        started = time.monotonic()
        first = node.__flowdash_preview__()
        again = node.__flowdash_preview__()
        waited = time.monotonic() - started
        await asyncio.wrap_future(flowdash_nodes._preview_future(node._object))
        for _ in range(3):
            await asyncio.sleep(0)
        return first, again, waited

    first, again, waited = asyncio.run(card_refreshes())

    assert waited < 0.5
    assert first == again == "Fetching rows..."
    assert len(calls) == 1
    # The card refreshes once the rows are there, and then shows them without another query.
    assert revisions == [1]
    rows = node.__flowdash_preview__()
    assert len(rows) == PREVIEW_ROWS
    assert len(calls) == 1


def test_filter_preview_fetches_only_the_latest_value(monkeypatch):
    big, ops = numbers_editor()
    ops.add_node(FILTER_ID, {"field": "kind"}, name="filtered")
    ops.connect("nums.pipeline", "filtered.pipeline")
    node = _instance(ops, "filtered")
    slow_queries(monkeypatch, big, 0.3)
    revisions = refreshes(node)

    async def drag():
        futures = []
        for value in ("even", "odd", "even", "odd"):
            node.value = value
            assert node.__flowdash_preview__() == "Fetching rows..."
            futures.append(flowdash_nodes._preview_future(node._object))
        for future in futures:
            if not future.cancelled():
                await asyncio.wrap_future(future)
        for _ in range(3):
            await asyncio.sleep(0)
        return futures

    futures = asyncio.run(drag())

    assert any(future.cancelled() for future in futures)
    assert revisions == [1]
    assert set(node.__flowdash_preview__()["kind"]) == {"odd"}


def test_failed_preview_is_kept_and_not_queried_again(monkeypatch, caplog):
    big, ops = numbers_editor()
    node = _instance(ops, "nums")
    calls = []

    async def failing(sql, timeout, *args, **kwargs):
        calls.append(sql)
        raise TimeoutError(f"The query did not finish within {timeout} seconds.\nMore detail")

    monkeypatch.setattr(big, "execute_with_timeout", failing)

    with caplog.at_level("ERROR"):
        assert node.__flowdash_preview__() == "No preview: The query did not finish within 5 seconds."
        preview = hook_preview(node)
    assert preview is not None and "No preview" in preview.html
    assert len(calls) == 1
    assert not [r for r in caplog.records if r.levelname == "ERROR"]

    async def card_refresh():
        return node.__flowdash_preview__()

    assert asyncio.run(card_refresh()).startswith("No preview")
    assert len(calls) == 1

    flowdash_nodes.forget(node._object)
    node.__flowdash_preview__()
    assert len(calls) == 2


def test_data_node_preview(source):
    ops = editor_with(source)
    ops.add_node(TABLE_ID, {"table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("sales.pipeline", "rows.pipeline")

    node = _instance(ops, "rows")
    frame = node.__flowdash_preview__(output="data")
    assert isinstance(frame, pd.DataFrame) and len(frame) == 3
    assert hook_preview(node).kind == "table"

    ops.add_node(DATA_ID, {}, name="empty_data")
    assert _instance(ops, "empty_data").__flowdash_preview__() is None


def chart_editor():
    big = DuckDBSource(uri=":memory:", tables={"pts": "SELECT i AS x, i * 2 AS y FROM range(80) AS t(i)"})
    ops = editor_with(big)
    ops.add_node(TABLE_ID, {"table": "pts"}, name="pts")
    ops.connect("data.source", "pts.source")
    spec = {"mark": "line", "encoding": {"x": {"field": "x", "type": "quantitative"}, "y": {"field": "y", "type": "quantitative"}}}
    ops.add_node(VIEW_ID, {"spec": spec}, name="chart")
    ops.connect("pts.pipeline", "chart.pipeline")
    return ops


def test_view_node_preview_is_a_line_of_text_on_a_host_without_chart_previews(monkeypatch):
    monkeypatch.setattr(flowdash_nodes, "_preview_kinds", lambda: {"table", "text"})
    ops = chart_editor()
    node = _instance(ops, "chart")

    assert node.__flowdash_preview__() == "Line chart of x and y, 80 rows"
    preview = hook_preview(node)
    assert "__flowdash" not in preview.html and "&lt;" not in preview.html

    node._object = SimpleNamespace(spec={"mark": {"type": "bar"}}, pipeline=SimpleNamespace(data=None), panel=None)
    assert node.__flowdash_preview__() == "Bar chart"


def test_view_node_preview_on_panel_flowdash_main_shows_no_internal_keys():
    node = _instance(chart_editor(), "chart")
    preview = hook_preview(node)
    assert preview is not None
    if "chart" not in flowdash_nodes._preview_kinds():
        assert "__flowdash_preview_kind__" not in preview.html
        assert "fd-preview-entries" not in preview.html


def test_view_node_preview_returns_an_evenly_sampled_chart_payload(monkeypatch):
    monkeypatch.setattr(flowdash_nodes, "_preview_kinds", lambda: {"table", "chart"})
    ops = chart_editor()
    node = _instance(ops, "chart")
    payload = node.__flowdash_preview__()
    assert payload["__flowdash_preview_kind__"] == "chart"
    assert payload["format"] == "vega-lite"
    assert payload["spec"] == node._object.spec
    data = payload["data"]
    assert isinstance(data, pd.DataFrame) and len(data) == PREVIEW_ROWS
    assert data["x"].iloc[0] == 0 and data["x"].iloc[-1] == 79
    assert data["x"].is_monotonic_increasing

    # Mutating the returned spec copy does not alter the live view.
    payload["spec"]["mark"] = "point"
    assert node._object.spec["mark"] == "line"

    node._object = SimpleNamespace(spec={"mark": "bar"}, pipeline=SimpleNamespace(data=None), panel=None)
    assert node.__flowdash_preview__(output="chart")["data"] is None

    ops.add_node(VIEW_ID, {}, name="empty_view")
    assert _instance(ops, "empty_view").__flowdash_preview__() is None

    ops.add_node(VIEW_ID, {"spec": {"mark": "bar", "encoding": {"x": {"field": "nope", "type": "nominal"}}}}, name="bad_view")
    ops.connect("pts.pipeline", "bad_view.pipeline")
    assert _instance(ops, "bad_view").__flowdash_preview__() is None


def test_sql_is_a_code_field_and_the_view_spec_uses_the_spec_editor():
    assert SQLNode.param.sql.metadata == {"flowdash": {"widget": "code", "language": "sql"}}
    assert ViewNode.param.spec.metadata["flowdash"]["widget"] == "spec"


def test_register_port_type_passes_the_options_panel_flowdash_accepts(monkeypatch):
    calls = []

    def with_color(name, *, aliases=(), subtype_of=(), columns=None, color=None):
        calls.append((name, color, columns))
        return name.lower()

    def without_color(name, *, aliases=(), subtype_of=(), columns=None):
        calls.append((name, None, columns))
        return name.lower()

    def neither(name, *, aliases=(), subtype_of=()):
        calls.append((name, None, None))
        return name.lower()

    def any_options(name, **options):
        calls.append((name, options.get("color"), options.get("columns")))
        return name.lower()

    for register, expected in [
        (with_color, ("lumen.Pipeline", "#0072B5", len)),
        (without_color, ("lumen.Pipeline", None, len)),
        (neither, ("lumen.Pipeline", None, None)),
        (any_options, ("lumen.Pipeline", "#0072B5", len)),
    ]:
        calls.clear()
        monkeypatch.setattr(flowdash_nodes, "register_port_type", register)
        assert flowdash_nodes._register_port_type("lumen.Pipeline", color="#0072B5", columns=len) == "lumen.pipeline"
        assert calls == [expected]


def test_sql_completions_offer_tables_and_columns(source):
    choices = flowdash_nodes.Choices
    if choices is None:
        pytest.skip("panel-flowdash without Choices")
    words = flowdash_nodes.sql_completions({"sql": ""}, {"source": source})["sql"]
    assert isinstance(words, choices) and not words.restrict
    assert set(words.values()) >= {"sales", "category", "amount"}
    assert words.types["sales"] == "table"
    assert flowdash_nodes.sql_completions({}, {}) is None


def test_sql_preview_follows_the_output(source):
    ops = dashboard(source)
    node = _instance(ops, "by_category")
    data = node.__flowdash_preview__(output="data")
    assert data is node._object.data
    pipeline = node.__flowdash_preview__(output="pipeline")
    assert isinstance(pipeline, pd.DataFrame) and pipeline is not data
    assert len(pipeline) == 2


def test_sql_error_names_line_and_column_and_clears_on_a_fix(source):
    node = SQLNode(source=source, sql="SELECT category,\n  amout FROM sales")
    assert node._error.startswith("Line 2, column 3: ")
    assert "\x1b" not in node._error
    node.sql = QUERY
    assert node._error is None and node._object is not None


async def test_sql_node_runs_the_query_off_the_event_loop(source, monkeypatch):
    loop_thread = threading.get_ident()
    threads = []
    compile_sql_ = flowdash_nodes.compile_sql

    def recording(*args, **kwargs):
        threads.append(threading.get_ident())
        return compile_sql_(*args, **kwargs)

    monkeypatch.setattr(flowdash_nodes, "compile_sql", recording)
    monkeypatch.setattr(flowdash_nodes, "_in_served_session", lambda: True)
    node = SQLNode(source=source, sql=QUERY)
    assert node._object is None
    for _ in range(200):
        if node._object is not None:
            break
        await asyncio.sleep(0.01)
    assert node._object is not None and len(node._object.data) == 2
    assert threads and loop_thread not in threads

    node.sql = "SELECT nope FROM sales"
    node.sql = QUERY
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(threads) == 3 and node._object is not None:
            break
    await asyncio.sleep(0.05)
    assert node._error is None and len(node._object.data) == 2


async def test_sql_node_outside_a_served_session_resolves_at_once(source):
    node = SQLNode(source=source, sql=QUERY)
    assert node._object is not None and len(node._object.data) == 2


def test_source_preview_without_stored_sizes_names_the_tables(source):
    ops = editor_with(source)
    preview = _instance(ops, "data").__flowdash_preview__()
    assert preview == "1 table: sales"


@pytest.mark.parametrize("sql, value", [
    ("SELECT SUM(amount) FROM sales", 35),
    ("SELECT AVG(amount) AS mean FROM sales", 35 / 3),
    ("SELECT SUM(amount) FROM sales WHERE category = 'Z'", None),
])
def test_sql_node_value_is_the_cell_of_a_one_by_one_result(source, sql, value):
    ops = dashboard(source, sql)
    result = output(ops, "by_category", "value")
    assert result is None if value is None else result == pytest.approx(value)
    node = ops.editor._component_instances[ops._instance("by_category")]
    assert node.__flowdash_preview__("value") == result


@pytest.mark.parametrize("sql, reason", [
    (QUERY, "one row and one column, e.g. SELECT SUM(x) FROM t; this query returns 2 rows and 2 columns"),
    ("SELECT MAX(category) FROM sales", "needs a number; the query's 'max(category)' is str 'B'"),
])
def test_sql_node_value_is_empty_for_other_results(source, sql, reason):
    ops = dashboard(source, sql)
    assert output(ops, "by_category", "value") is None
    assert isinstance(output(ops, "by_category", "pipeline"), Pipeline)
    node = ops.editor._component_instances[ops._instance("by_category")]
    assert reason in node.__flowdash_preview__("value")


def test_sql_node_value_feeds_a_kpi(source):
    ops = dashboard(source, "SELECT SUM(amount) AS total FROM sales")
    ops.add_node("Widgets/KPI", {"label": "Total"}, name="kpi")
    ops.connect("by_category.value", "kpi.value")
    assert ops.editor._component_instances[ops._instance("kpi")].value == 35


def test_sql_field_is_labelled_sql():
    assert SQLNode.param.sql.label == "SQL"
