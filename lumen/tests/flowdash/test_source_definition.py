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
from lumen.flowdash.redact import REDACTED, redact_spec, redact_text
from lumen.sources.base import InMemorySource
from lumen.sources.duckdb import DuckDBSource

ENV = {
    "HOME": "/home/analyst",
    "PWD": "/home/analyst/project",
    "PGPASSWORD": "pg-env-password",
    "SNOWFLAKE_TOKEN": "env-snowflake-token",
    "WAREHOUSE": "ANALYTICS_WH",
}

SECRETS = [
    "s3cr%2Ft", "hunter22", "tok-123456", "AKIAEXAMPLE", "abc/def+ghi", "zzz-secret",
    "sf-password", "sf-token", "pk-password", "pg-env-password", "env-snowflake-token",
    "client-secret", "oauth-refresh", "odbc-pass", "ANALYTICS_WH",
]


def dump(spec, environ=ENV):
    return yaml.safe_dump(redact_spec(spec, environ=environ), sort_keys=False)


def assert_no_secret(text):
    for secret in SECRETS:
        assert secret not in text


def test_sqlalchemy_spec_hides_uri_password_and_driver_secrets():
    spec = {
        "type": "sqlalchemy",
        "url": "postgresql://alice:s3cr%2Ft@db.example.com:5432/sales?sslmode=require&password=hunter22",
        "connect_args": {"sslpassword": "hunter22", "options": "-c statement_timeout=5000", "connect_timeout": 5},
        "engine_kwargs": {"pool_size": 3},
        "query_params": {"token": "tok-123456", "application_name": "lumen"},
        "username": "alice",
        "password": "hunter22",
        "tables": ["orders"],
    }
    text = dump(spec)
    assert_no_secret(text)
    redacted = yaml.safe_load(text)
    assert redacted["url"] == f"postgresql://alice:{REDACTED}@db.example.com:5432/sales?sslmode=require&password={REDACTED}"
    assert redacted["connect_args"] == {"sslpassword": REDACTED, "options": REDACTED, "connect_timeout": 5}
    assert redacted["engine_kwargs"] == {"pool_size": 3}
    assert redacted["query_params"] == {"token": REDACTED, "application_name": "lumen"}
    assert redacted["username"] == "alice"
    assert redacted["password"] == REDACTED
    assert redacted["tables"] == ["orders"]


def test_snowflake_style_spec_hides_credentials_and_keeps_the_rest():
    spec = {
        "type": "snowflake",
        "account": "xy12345.eu-central-1",
        "user": "ANALYST",
        "authenticator": "snowflake",
        "password": "sf-password",
        "token": "sf-token",
        "private_key": b"pem-private-key-bytes",
        "private_key_password": "pk-password",
        "warehouse": "ANALYTICS_WH",
        "database": "SALES",
        "conn_kwargs": {"client_session_keep_alive": True, "oauth_client_secret": "client-secret"},
        "tables": {"orders": "SELECT * FROM ORDERS"},
    }
    text = dump(spec)
    assert_no_secret(text)
    redacted = yaml.safe_load(text)
    assert redacted["private_key"] == redacted["private_key_password"] == REDACTED
    assert redacted["password"] == redacted["token"] == REDACTED
    # The warehouse came from an environment variable, so it is hidden too.
    assert redacted["warehouse"] == REDACTED
    assert redacted["conn_kwargs"] == {"client_session_keep_alive": True, "oauth_client_secret": REDACTED}
    for key in ("account", "user", "authenticator", "database", "tables"):
        assert redacted[key] == spec[key]


def test_duckdb_spec_hides_secrets_created_or_set_on_connect():
    spec = {
        "type": "duckdb",
        "uri": ":memory:",
        "initializers": [
            "INSTALL httpfs",
            "CREATE SECRET s3_data (TYPE s3, KEY_ID 'AKIAEXAMPLE', SECRET 'abc/def+ghi', REGION 'eu-west-1')",
            "SET s3_secret_access_key = 'zzz-secret'",
            "SET s3_access_key_id='AKIAEXAMPLE'",
        ],
        "tables": {
            "trips": "SELECT * FROM read_parquet('s3://bucket/trips/*.parquet')",
            "api_keys": "SELECT * FROM read_csv('https://user:hunter22@example.com/keys.csv')",
        },
    }
    text = dump(spec)
    assert_no_secret(text)
    redacted = yaml.safe_load(text)
    assert redacted["initializers"][0] == "INSTALL httpfs"
    assert "REGION 'eu-west-1'" in redacted["initializers"][1]
    assert f"KEY_ID '{REDACTED}'" in redacted["initializers"][1]
    assert redacted["tables"] == {
        "trips": spec["tables"]["trips"],
        "api_keys": f"SELECT * FROM read_csv('https://user:{REDACTED}@example.com/keys.csv')",
    }


def test_nested_credentials_and_odbc_strings_are_hidden():
    spec = {
        "type": "bigquery",
        "project_id": "my-project",
        "credentials": {"type": "service_account", "private_key": "pem-text"},
        "extra": [{"auth": {"refresh": "oauth-refresh"}}],
        "dsn": "Driver={ODBC Driver 18};Server=db;UID=sa;PWD=odbc-pass;Encrypt=yes",
    }
    text = dump(spec)
    assert_no_secret(text)
    assert "pem-text" not in text
    redacted = yaml.safe_load(text)
    assert redacted["credentials"] == REDACTED
    assert redacted["project_id"] == "my-project"
    assert "Server=db" in redacted["dsn"]


def test_environment_values_are_hidden_but_not_the_working_directory():
    spec = {
        "type": "duckdb",
        "uri": "/home/analyst/project/data.duckdb",
        "tables": {"t": "SELECT * FROM read_csv('/home/analyst/project/t.csv')"},
        "url": "postgresql://reader@db/x?passfile=none",
        "comment": "uses pg-env-password and env-snowflake-token",
        "password": "",
    }
    redacted = redact_spec(spec, environ=ENV)
    assert redacted["uri"] == spec["uri"]
    assert redacted["tables"] == spec["tables"]
    assert redacted["comment"] == f"uses {REDACTED} and {REDACTED}"
    assert redacted["password"] == ""
    assert redact_text("pg-env-password", environ=ENV) == REDACTED


def test_uri_password_with_unescaped_characters_is_hidden():
    for url in (
        "mysql+pymysql://bob:p/a?ss@host/db",
        "redis://:only-password@cache:6379/0",
        "snowflake://user:pw%40x@account/db/schema?warehouse=wh",
    ):
        text = redact_text(url, environ={})
        assert REDACTED in text
        for part in ("p/a?ss", "only-password", "pw%40x"):
            assert part not in text
    assert redact_text("sqlite:////data/shop.db", environ={}) == "sqlite:////data/shop.db"
    assert redact_text("https://example.com/a@b", environ={}) == "https://example.com/a@b"


def test_objects_yaml_cannot_show_are_named_not_printed():
    redacted = redact_spec({"type": "inmemory", "tables": {"items": pd.DataFrame({"a": [1]})}, "blob": b"raw"}, environ={})
    assert redacted == {"type": "inmemory", "tables": {"items": "<DataFrame>"}, "blob": REDACTED}
    yaml.safe_dump(redacted)


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


def test_source_node_definition_hides_a_sqlalchemy_url_password():
    pytest.importorskip("sqlalchemy")
    from lumen.sources.sqlalchemy import SQLAlchemySource

    source = SQLAlchemySource(url="sqlite://")
    source.param.update(url="postgresql://alice:s3cr%2Ft@db.example.com/sales", connect_args={"sslpassword": "hunter22"})
    ops = editor_with(source)

    _, text = instance(ops, "data").__flowdash_definition__()

    assert_no_secret(text)
    assert f"postgresql://alice:{REDACTED}@db.example.com/sales" in text


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
