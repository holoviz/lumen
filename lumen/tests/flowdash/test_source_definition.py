import asyncio
import threading

import pandas as pd
import pytest
import yaml

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps
from panel_flowdash.ui import node_preview

from lumen.flowdash import (
    COMPONENTS, SOURCE_ID, bind_source, nodes as flowdash_nodes,
)
from lumen.flowdash.redact import REDACTED
from lumen.sources.base import InMemorySource
from lumen.sources.duckdb import DuckDBSource

SECRETS = ["AKIAEXAMPLE", "abc/def+ghi", "env-snowflake-token", "s3cr%2Ft", "hunter22"]


def assert_no_secret(text):
    for secret in SECRETS:
        assert secret not in text


def editor_with(source, ref="data"):
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source, ref)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": ref}, name=ref)
    return ops


def instance(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def test_source_node_definition_is_its_redacted_spec_as_yaml(monkeypatch):
    monkeypatch.setenv("LUMEN_TEST_DB_TOKEN", "env-snowflake-token")
    source = DuckDBSource(uri=":memory:", tables={"sales": "SELECT 1 AS amount"})
    source.param.update(initializers=[
        "CREATE SECRET s3_data (TYPE s3, KEY_ID 'AKIAEXAMPLE', SECRET 'abc/def+ghi')",
        "SET s3_session_token = 'env-snowflake-token'",
    ])
    ops = editor_with(source)

    language, text = instance(ops, "data").__flowdash_definition__()

    assert language == "yaml"
    assert_no_secret(text)
    spec = yaml.safe_load(text)
    assert list(spec)[0] == "type"
    assert spec["type"] == "duckdb"
    assert spec["tables"] == {"sales": "SELECT 1 AS amount"}
    assert spec["initializers"] == ["CREATE ***", "SET ***"]


def test_source_node_definition_hides_a_sqlalchemy_url_password(monkeypatch):
    pytest.importorskip("sqlalchemy")
    from lumen.sources.sqlalchemy import SQLAlchemySource

    # As on GitHub's Windows runners; postgresql:// holds the value but not as a word.
    monkeypatch.setenv("PGUSER", "postgres")
    source = SQLAlchemySource(url="sqlite://")
    source.param.update(url="postgresql://alice:s3cr%2Ft@db.example.com/sales", connect_args={"sslpassword": "hunter22"})
    ops = editor_with(source)

    _, text = instance(ops, "data").__flowdash_definition__()

    assert_no_secret(text)
    spec = yaml.safe_load(text)
    assert spec["url"] == "postgresql://db.example.com/sales"
    assert spec["connect_args"] == {"sslpassword": REDACTED}


def test_source_node_without_a_source_defines_nothing():
    ops = editor_with(InMemorySource(tables={"items": pd.DataFrame({"a": [1]})}))
    ops.add_node(SOURCE_ID, {}, name="empty")
    assert instance(ops, "empty").__flowdash_definition__() is None
    assert instance(ops, "data").__flowdash_definition__()[0] == "yaml"


@pytest.mark.skipif(
    not hasattr(node_preview, "call_definition_hook"),
    reason="the installed panel-flowdash has no __flowdash_definition__ hook",
)
def test_flowdash_reads_the_source_definition():
    source = DuckDBSource(uri=":memory:", tables={"sales": "SELECT 1 AS amount"})
    ops = editor_with(source)
    language, text = node_preview.call_definition_hook(instance(ops, "data"))
    assert language == "yaml"
    assert "type: duckdb" in text


def test_source_overview_lists_tables_rows_columns_and_names():
    frame = pd.DataFrame({f"c{i}": [1, 2, 3] for i in range(8)})
    ops = editor_with(InMemorySource(tables={"wide": frame, "small": pd.DataFrame({"a": [1]})}))
    overview = instance(ops, "data").__flowdash_preview__()
    assert list(overview.columns) == ["table", "rows", "columns", "column names"]
    rows = {row["table"]: row for row in overview.to_dict("records")}
    assert rows["wide"]["rows"] == 3 and rows["wide"]["columns"] == 8
    assert rows["wide"]["column names"] == "c0, c1, c2, c3, c4, c5 +2 more"
    assert rows["small"] == {"table": "small", "rows": 1, "columns": 1, "column names": "a"}


def slow_schemas(monkeypatch, source):
    """Hold each schema read of ``source`` until released, counting the tables read."""
    release, calls = threading.Event(), []
    original = source.get_schema

    def slow(table=None, limit=None, shuffle=False):
        calls.append(table)
        release.wait(5)
        return original(table, limit, shuffle)

    monkeypatch.setattr(source, "get_schema", slow)
    return release, calls


def test_source_overview_reads_schemas_off_the_event_loop(monkeypatch, executed):
    source = DuckDBSource(uri=":memory:", tables={
        "big": "SELECT i, i * 2 AS twice FROM range(1000) AS t(i)",
        "small": "SELECT 1 AS one",
    })
    ops = editor_with(source)
    node = instance(ops, "data")
    release, calls = slow_schemas(monkeypatch, source)
    revisions = []
    node.param.watch(lambda event: revisions.append(event.new), "_preview_rev")
    executed.clear()

    async def card_refreshes():
        first = node.__flowdash_preview__()
        again = node.__flowdash_preview__()
        # Nothing ran on the loop: the worker is still waiting to read the first schema.
        assert executed == []
        release.set()
        await asyncio.wrap_future(flowdash_nodes._SCHEMA_PENDING[source][0])
        for _ in range(3):
            await asyncio.sleep(0)
        return first, again

    first, again = asyncio.run(card_refreshes())

    assert first == again == "2 tables: big, small"
    assert calls == ["big", "small"]
    assert revisions == [1]
    overview = node.__flowdash_preview__()
    assert overview.to_dict("records") == [
        {"table": "big", "rows": 1000, "columns": 2, "column names": "i, twice"},
        {"table": "small", "rows": 1, "columns": 1, "column names": "one"},
    ]
    assert calls == ["big", "small"]


def test_source_overview_off_the_loop_does_not_query(executed):
    source = DuckDBSource(uri=":memory:", tables={"sales": "SELECT 1 AS amount"})
    node = instance(editor_with(source), "data")
    executed.clear()
    assert node.__flowdash_preview__() == "1 table: sales"
    assert executed == []


def test_source_overview_does_not_read_a_failing_schema_again(monkeypatch):
    source = DuckDBSource(uri=":memory:", tables={"bad": "SELECT * FROM missing_table"})
    node = instance(editor_with(source), "data")
    calls, release = [], threading.Event()

    def failing(table=None, limit=None, shuffle=False):
        calls.append(table)
        release.wait(5)
        raise RuntimeError("no such table")

    monkeypatch.setattr(source, "get_schema", failing)

    async def twice():
        node.__flowdash_preview__()
        future = flowdash_nodes._SCHEMA_PENDING[source][0]
        release.set()
        await asyncio.wrap_future(future)
        for _ in range(3):
            await asyncio.sleep(0)
        return node.__flowdash_preview__()

    assert asyncio.run(twice()) == "1 table: bad"
    assert calls == ["bad"]
    assert source not in flowdash_nodes._SCHEMA_PENDING


def test_source_overview_reads_each_schema_once_when_the_source_does_not_store_it(monkeypatch):
    source = DuckDBSource(uri=":memory:", tables={"nums": "SELECT i FROM range(10) AS t(i)"}, cache_schema=False)
    node = instance(editor_with(source), "data")
    calls = []
    original = source.get_schema

    def counted(table=None, limit=None, shuffle=False):
        calls.append(table)
        return original(table, limit, shuffle)

    monkeypatch.setattr(source, "get_schema", counted)
    revisions = []
    node.param.watch(lambda event: revisions.append(event.new), "_preview_rev")

    async def settle():
        node.__flowdash_preview__()
        for _ in range(20):
            if (pending := flowdash_nodes._SCHEMA_PENDING.get(source)) is not None:
                await asyncio.wrap_future(pending[0])
            for _ in range(3):
                await asyncio.sleep(0)
            node.__flowdash_preview__()
        return node.__flowdash_preview__()

    overview = asyncio.run(settle())

    assert calls == ["nums"]
    assert revisions == [1]
    assert overview.to_dict("records") == [{"table": "nums", "rows": 10, "columns": 1, "column names": "i"}]


def overview_on_thread(node, name):
    result = []
    worker = threading.Thread(target=lambda: result.append(node.__flowdash_preview__()), name=name)
    worker.start()
    worker.join(10)
    return result[0]


def test_source_overview_on_flowdash_preview_worker_reads_schemas_straight_away():
    source = DuckDBSource(uri=":memory:", tables={"sales": "SELECT 1 AS amount"})
    node = instance(editor_with(source), "data")
    overview = overview_on_thread(node, "flowdash-preview_0")
    assert overview.to_dict("records") == [{"table": "sales", "rows": 1, "columns": 1, "column names": "amount"}]


def test_source_overview_on_another_thread_does_not_query(executed):
    source = DuckDBSource(uri=":memory:", tables={"sales": "SELECT 1 AS amount"})
    node = instance(editor_with(source), "data")
    executed.clear()
    assert overview_on_thread(node, "agent-tool") == "1 table: sales"
    assert executed == []


def test_source_overview_reads_a_schema_again_once_it_expires(monkeypatch):
    source = DuckDBSource(uri=":memory:", tables={"nums": "SELECT 1 AS i"}, cache_schema=False)
    node = instance(editor_with(source), "data")
    calls = []
    original = source.get_schema

    def counted(table=None, limit=None, shuffle=False):
        calls.append(table)
        return original(table, limit, shuffle)

    monkeypatch.setattr(source, "get_schema", counted)
    overview_on_thread(node, "flowdash-preview_0")
    overview_on_thread(node, "flowdash-preview_0")
    assert calls == ["nums"]
    monkeypatch.setattr(flowdash_nodes, "_SCHEMA_TTL", -1)
    overview_on_thread(node, "flowdash-preview_0")
    assert calls == ["nums", "nums"]
