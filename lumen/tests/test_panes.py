"""Tests for the Panel components Lumen renders its views with."""
import base64
import io

import pandas as pd
import pyarrow as pa
import pytest

from lumen.panes.mosaic import Mosaic

SPEC = {
    "plot": [
        {"mark": "dot", "data": {"from": "t"}, "x": "a", "y": "b"}
    ]
}


@pytest.fixture
def pane():
    """A Mosaic pane over a small registered frame, with replies captured."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    pane = Mosaic(SPEC, data={"t": df})
    pane._sent = []
    pane._send_msg = pane._sent.append
    return pane


def test_registers_data_and_holds_spec(pane):
    assert pane.spec == SPEC
    assert pane.connection.query("SELECT count(*) FROM t").fetchone()[0] == 3


def test_accepts_an_existing_connection():
    """A caller's DuckDB connection is queried directly rather than replaced."""
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS a")
    pane = Mosaic(SPEC, con=con)

    assert pane.connection is con
    assert pane.connection.query("SELECT a FROM t").fetchone()[0] == 1


def test_arrow_query_round_trips_as_base64_ipc(pane):
    """Arrow results are base64-encoded because Panel's ESM message channel is
    JSON-only; the browser decodes them back into an Arrow table."""
    pane._handle_msg({"type": "arrow", "uuid": "q1", "sql": "SELECT * FROM t ORDER BY a"})

    msg = pane._sent[-1]
    assert msg["type"] == "arrow"
    assert msg["uuid"] == "q1"
    table = pa.ipc.open_stream(io.BytesIO(base64.b64decode(msg["data"]))).read_all()
    assert table.to_pydict() == {"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]}


def test_json_query_returns_records(pane):
    pane._handle_msg({"type": "json", "uuid": "q2", "sql": "SELECT count(*) AS n FROM t"})

    assert pane._sent[-1] == {"type": "json", "uuid": "q2", "result": [{"n": 3}]}


def test_exec_query_runs_statement(pane):
    pane._handle_msg({"type": "exec", "uuid": "q3", "sql": "CREATE TABLE t2 AS SELECT 1 AS x"})

    assert pane._sent[-1] == {"type": "exec", "uuid": "q3"}
    assert pane.connection.query("SELECT x FROM t2").fetchone()[0] == 1


@pytest.mark.parametrize("msg", [
    {"type": "arrow", "uuid": "bad-sql", "sql": "SELECT * FROM does_not_exist"},
    {"type": "nonsense", "uuid": "bad-type", "sql": "SELECT 1"},
])
def test_failed_query_replies_with_the_error_and_its_uuid(pane, msg):
    """A failure must still answer, and answer the right query: the browser
    keys pending promises by uuid, so a dropped reply hangs the chart forever
    instead of surfacing anything."""
    pane._handle_msg(msg)

    reply = pane._sent[-1]
    assert reply["uuid"] == msg["uuid"]
    assert reply["error"]


def test_esm_module_is_packaged():
    """The pane is useless without its ESM module resolving next to it."""
    esm = Mosaic._esm_path(compiled=False)
    assert esm is not None and esm.is_file()
    assert "@uwdata/mosaic-spec" in esm.read_text()


def test_importmap_uses_a_single_mosaic_entry_point():
    """mosaic-spec must be the only Mosaic import: pulling mosaic-core in
    separately would give the pane a different Coordinator than the one the
    spec instantiates its plots against, and its connector would never be used."""
    imports = Mosaic._process_importmap()["imports"]

    mosaic_imports = {k for k in imports if k.startswith("@uwdata/")}
    assert mosaic_imports == {"@uwdata/mosaic-spec", "@uwdata/flechette"}
