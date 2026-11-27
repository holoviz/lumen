"""What the app's data is and what uses it, as the Data section reads it, without a view."""
import dataclasses

import pytest

pytest.importorskip("duckdb")

from panel_flowdash import MemoryDashboardStore
from panel_flowdash.auth import Identity, Permission
from panel_flowdash.dashboard_store import DashboardEdge, DashboardItem

from lumen.flowdash import SOURCE_ALIASES_KEY, SOURCES_KEY
from lumen.flowdash.catalog import catalog_tables
from lumen.flowdash.contracts import (
    METRIC_ID, SOURCE_ID, SQL_ID, TABLE_ID,
)
from lumen.flowdash.data import (
    DashboardRef, metric_tables, metric_usage, sql_tables, table_usage,
    visible_dashboards,
)
from lumen.flowdash.uploads import UPLOADS_KEY, UploadError, Uploads
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "orders": "SELECT * FROM (VALUES (1, 30), (2, 20)) AS t(order_id, amount)",
    "customers": "SELECT * FROM (VALUES (1, 'Ada')) AS t(customer_id, name)",
}


@pytest.fixture
def context():
    shop = DuckDBSource(uri=":memory:", tables=TABLES, name="shop")
    fleet = DuckDBSource(uri=":memory:", tables={"turbines": "SELECT 1 AS t_cap"}, name="fleet")
    return {SOURCES_KEY: {"shop": shop, "fleet": fleet}, SOURCE_ALIASES_KEY: {"data": "shop"}}


def _dashboard(store, title, items, edges=(), owner="ana", permission=None):
    model = store.create_dashboard(owner, title)
    model.items = [DashboardItem(iid, component, config=config) for iid, component, config in items]
    model.edges = [DashboardEdge(source, "source", target, "source") for source, target in edges]
    if permission is not None:
        model.permission = permission
    store.save_dashboard(model)
    return model


def _titles(refs):
    return [ref.title for ref in refs]


def test_the_catalog_lists_every_table_with_whether_the_assistant_uses_it(context):
    tables, errors = catalog_tables(context, {"shop": {"tables": {"customers": {"assistant": False}}}})
    assert [(t.ref, t.table, t.assistant) for t in tables] == [
        ("shop", "orders", True), ("shop", "customers", False), ("fleet", "turbines", True),
    ]
    assert errors == {} and all(t.upload is None for t in tables)


def test_the_catalog_names_uploads_and_sources_that_cannot_list_tables(context, tmp_path):
    class Broken(DuckDBSource):
        def get_tables(self):
            raise RuntimeError("connection refused")

    class Admin:
        _identity = Identity("ana")

        def _still_admin(self):
            return True

    context[SOURCES_KEY]["remote"] = Broken(uri=":memory:", tables={})
    uploads = context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", context[SOURCES_KEY])
    uploads.add(b"region,revenue\nNorth,1\n", "sales.csv", app=Admin())
    tables, errors = catalog_tables(context, {})
    assert [(t.ref, t.table) for t in tables][-1] == ("sales", "sales")
    assert tables[-1].upload.uploaded_by == "ana" and tables[0].upload is None
    assert "connection refused" in errors["remote"]


def test_sql_tables_leave_out_ctes():
    sql = "WITH paid AS (SELECT * FROM orders) SELECT * FROM paid JOIN customers USING (customer_id)"
    assert sorted(sql_tables(sql)) == ["customers", "orders"]
    assert sql_tables("not sql at all (") == []


def test_table_usage_follows_source_nodes_aliases_and_earlier_sql_nodes(context):
    store = MemoryDashboardStore()
    _dashboard(store, "By ref", [("t", TABLE_ID, {"ref": "data", "table": "orders"})])
    _dashboard(store, "Wired", [
        ("s", SOURCE_ID, {"ref": "shop"}),
        ("t", TABLE_ID, {"ref": "", "table": "customers"}),
    ], edges=[("s", "t")])
    _dashboard(store, "SQL", [
        ("s", SOURCE_ID, {"ref": "fleet"}),
        ("q1", SQL_ID, {"sql": "SELECT * FROM turbines", "table": "capacity"}),
        ("q2", SQL_ID, {"sql": "WITH c AS (SELECT * FROM capacity) SELECT * FROM c"}),
    ], edges=[("s", "q1"), ("q1", "q2")])
    _dashboard(store, "Unserved", [("t", TABLE_ID, {"ref": "gone", "table": "orders"})])
    usage = table_usage(store._all_dashboards(), context)
    assert _titles(usage[("shop", "orders")]) == ["By ref"]
    assert _titles(usage[("shop", "customers")]) == ["Wired"]
    assert _titles(usage[("fleet", "turbines")]) == ["SQL"]
    # An earlier node's result is read by name too; only real tables are looked up.
    assert ("fleet", "capacity") in usage
    assert not any(ref == "gone" for ref, _ in usage)


def test_metric_usage_and_the_tables_a_metric_reads(context):
    store = MemoryDashboardStore()
    model = _dashboard(store, "Fleet", [("m", METRIC_ID, {"metric": "capacity"})])
    assert metric_usage(store._all_dashboards()) == {"capacity": [DashboardRef(model.dashboard_id, "Fleet")]}
    assert metric_tables({"source": "data", "table": "orders"}, context) == [("shop", "orders")]
    sql = {"source": "shop", "sql": "SELECT name, SUM(amount) FROM orders JOIN customers ON 1 = 1 LIMIT :n"}
    assert sorted(metric_tables(sql, context)) == [("shop", "customers"), ("shop", "orders")]
    assert metric_tables({"source": "gone", "table": "orders"}, context) == []


class Session:
    def __init__(self, store, user="bo", groups=(), admin=False):
        self.store = store
        self._identity = Identity(user, groups=frozenset(groups))
        self.is_admin = admin

    def _default_allow(self):
        return False


def test_a_session_only_sees_the_dashboards_it_may_open():
    store = MemoryDashboardStore()
    _dashboard(store, "Ana's", [], owner="ana")
    _dashboard(store, "Bo's", [], owner="bo")
    _dashboard(store, "Shared", [], owner="ana", permission=Permission(allow_groups=frozenset({"ops"})))
    assert sorted(_titles(visible_dashboards(Session(store, admin=True)))) == ["Ana's", "Bo's", "Shared"]
    assert _titles(visible_dashboards(Session(store))) == ["Bo's"]
    assert sorted(_titles(visible_dashboards(Session(store, groups=["ops"])))) == ["Bo's", "Shared"]
    assert visible_dashboards(object()) == []


# ---------------------------------------------------------------------------
# Sources, metric checks and what the assistant leaves out


def test_source_summaries_say_what_each_source_is_and_how_it_got_there(context, tmp_path):
    from lumen.flowdash.app import resolve_sources, source_refs
    from lumen.flowdash.data import source_summaries

    (tmp_path / "turbines.csv").write_text("p_name,p_cap\nAlta,6.0\n")
    served = source_refs(resolve_sources([str(tmp_path / "turbines.csv")]))
    context[SOURCES_KEY].update(served)
    uploads = context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", context[SOURCES_KEY])

    class Admin:
        _identity = Identity("jon")

        def _still_admin(self):
            return True

    uploads.add(b"region,revenue\nNorth,1\n", "regional sales.csv", app=Admin())
    tables, _ = catalog_tables(context, {})
    summaries = {s.ref: s for s in source_summaries(context, tables)}
    assert summaries["turbines"].origin == "From the command line" and summaries["turbines"].files == ("turbines.csv",)
    assert summaries["shop"].origin == "Served by the app" and summaries["shop"].tables == 2
    upload = summaries["regional_sales"]
    assert upload.kind == "Upload" and upload.files == ("regional sales.csv",)
    assert upload.origin.startswith("Uploaded by jon, ")


METRICS = """\
metrics:
  paid:
    source: shop
    table: orders
    measures: {total: SUM(amount)}
  gone:
    source: shop
    table: refunds
    measures: {total: SUM(amount)}
  names:
    source: data
    sql: SELECT name FROM customers
"""


def test_metric_statuses_read_what_the_app_uses(context, tmp_path):
    from lumen.flowdash.data import metric_statuses
    from lumen.flowdash.metrics import METRICS_KEY, ProjectMetrics
    from lumen.flowdash.project import PROJECT_FILE, load_project

    path = tmp_path / PROJECT_FILE
    path.write_text(METRICS)
    project = load_project(path, context)
    statuses, general = metric_statuses(project)
    assert [(s.name, s.check) for s in statuses] == [("paid", "valid"), ("gone", "table"), ("names", "valid")]
    assert "has no table 'refunds'" in statuses[1].message and general == []
    project.context[METRICS_KEY] = metrics = ProjectMetrics(project)
    assert list(metrics) == ["paid", "names"] and metrics.failures["gone"].kind == "table"
    assert [s.check for s in metric_statuses(project)[0]] == ["valid", "table", "valid"]


def test_a_metric_over_a_table_turned_off_is_left_out_of_the_assistant(context):
    from lumen.flowdash import CATALOG_KEY, Catalog
    from lumen.flowdash.data import hidden_reads_of

    context[CATALOG_KEY] = Catalog({"shop": frozenset({"customers"})})
    assert hidden_reads_of({"source": "shop", "sql": "SELECT name FROM customers"}, context) == ["customers"]
    assert hidden_reads_of({"source": "data", "table": "customers"}, context) == ["customers"]
    assert hidden_reads_of({"source": "shop", "table": "orders"}, context) == []


def test_a_dashboards_metric_node_says_why_its_metric_does_not_run(context, tmp_path):
    from panel_flowdash import FlowDash
    from panel_flowdash.agent import (
        DashboardOps, OperationError, OutputUnavailable,
    )

    from lumen.flowdash import COMPONENTS, METRICS_KEY, bind_source
    from lumen.flowdash.metrics import MetricSet
    from lumen.flowdash.project import PROJECT_FILE, load_project

    path = tmp_path / PROJECT_FILE
    path.write_text(METRICS)
    metrics = load_project(path, context).get("metrics")
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, context[SOURCES_KEY]["shop"], "shop")
    editor.context[METRICS_KEY] = metrics
    ops = DashboardOps(editor)
    try:
        with pytest.raises(OperationError, match="The source 'shop' has no table 'refunds'"):
            ops.add_node(METRIC_ID, {"metric": "gone"}, name="refunds")
        ops.add_node(METRIC_ID, {"metric": "paid", "measures": ["total"]}, name="paid")
        # The table goes away later, e.g. the source was rebuilt: the node says what the check found.
        editor.context[METRICS_KEY] = MetricSet(
            {"names": metrics["names"]}, failures={"paid": dataclasses.replace(metrics.failures["gone"], metric=metrics["paid"])},
        )
        node = editor._component_instances[ops._instance("paid")]
        node._refresh()
        with pytest.raises(OutputUnavailable, match="The source 'shop' has no table 'refunds'"):
            ops.node_output("paid.pipeline")
        provenance = node.__flowdash_provenance__()
        assert provenance["label"] == "Metric: paid" and "has no table 'refunds'" in provenance["detail"]
    finally:
        editor.dispose()


# ---------------------------------------------------------------------------
# Previews, the assistant switch on upload, and connected sources


class Admin:
    _identity = Identity("ana")

    def __init__(self, admin=True):
        self.admin = admin

    def _still_admin(self):
        return self.admin


def _project(tmp_path, context):
    from lumen.flowdash.project import PROJECT_FILE, load_project

    path = tmp_path / PROJECT_FILE
    path.write_text("uploads:\n  enabled: true\n")
    project = load_project(path, context)
    project.context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", context[SOURCES_KEY], project=project)
    return project


def test_a_preview_reads_the_file_and_keeps_nothing(context, tmp_path):
    project = _project(tmp_path, context)
    uploads = project.context[UPLOADS_KEY]
    preview = uploads.preview(b"region,revenue,day\nNorth,1.5,2026-01-02\nSouth,2,2026-01-03\n", "sales.csv", app=Admin())
    assert preview.ref == "sales" and preview.tables == ["sales"]
    assert preview.previews[0].columns == [
        ("region", "string", "North, South"), ("revenue", "number", "1.5, 2"), ("day", "date", "2026-01-02, 2026-01-03"),
    ]
    assert uploads.uploads() == [] and "sales" not in context[SOURCES_KEY]
    assert not (uploads.directory.exists() and any(uploads.directory.iterdir()))


def test_an_upload_can_be_kept_from_the_assistant(context, tmp_path):
    from lumen.flowdash.catalog import raw_section

    project = _project(tmp_path, context)
    uploads = project.context[UPLOADS_KEY]
    uploads.add(b"region,revenue\nNorth,1\n", "sales.csv", app=Admin(admin=False), assistant=False)
    assert raw_section(project) == {"sales": {"tables": {"sales": {"assistant": False}}}}


def test_an_admin_connects_a_source_that_comes_back_after_a_restart(context, tmp_path):
    import duckdb

    project = _project(tmp_path, context)
    uploads = project.context[UPLOADS_KEY]
    db = tmp_path / "plant.db"
    connection = duckdb.connect(str(db))
    connection.execute("CREATE TABLE readings AS SELECT 1 AS kw")
    connection.close()
    with pytest.raises(UploadError, match="Only admins"):
        uploads.read_location(str(db), app=Admin(admin=False))
    preview, source = uploads.read_location(str(db), app=Admin())
    assert (preview.ref, preview.tables) == ("plant", ["readings"])
    added = uploads.connect(str(db), app=Admin(), source=source)
    assert added.ref == "plant" and context[SOURCES_KEY]["plant"] is source and added.added_by == "ana"

    restored = Uploads(tmp_path / "uploads", {}, project=project)
    restored.restore()
    assert restored.connection("plant").location == str(db.resolve())
    assert restored.sources["plant"].get_tables() == ["readings"]
    assert "Only admins" in restored.disconnect("plant", app=Admin(admin=False))[0].message
    assert restored.disconnect("plant", app=Admin()) == [] and "plant" not in restored.sources
    with pytest.raises(UploadError, match="Could not read it"):
        uploads.read_location(str(tmp_path / "nowhere.csv"), app=Admin())


# ---------------------------------------------------------------------------
# A connected source's credentials

pytest.importorskip("panel_flowdash.secret_store")

SECRET_URL = "postgresql://ana:hunter2@db.example.com/fleet"


@pytest.fixture
def secret_key(monkeypatch):
    from panel_flowdash.secret_store import (
        KEY_ENV, encryption_available, fernet_key,
    )

    if not encryption_available():
        pytest.skip("needs cryptography")
    key = fernet_key()
    monkeypatch.setenv(KEY_ENV, key)
    return key


@pytest.fixture
def opened(monkeypatch):
    """Connecting reads any location into a small DuckDB source, keeping the locations it was given."""
    seen = []

    def fake_open(self, location, ref):
        seen.append(location)
        return DuckDBSource(uri=":memory:", tables={"readings": "SELECT 1 AS kw"}, name=ref)

    monkeypatch.setattr(Uploads, "_open", fake_open)
    return seen


def test_a_password_is_stored_encrypted_and_read_back(secret_key):
    from lumen.flowdash.uploads import (
        Connection, resolved_location, stored_location,
    )

    kept, secret = stored_location(SECRET_URL)
    assert kept == "postgresql://ana:***@db.example.com/fleet" and set(secret) == {"encrypted", "key_id"}
    assert "hunter2" not in str(secret)
    assert resolved_location(Connection("fleet", kept, "ana", "", secret=secret)) == SECRET_URL
    assert stored_location("/data/plant.db") == ("/data/plant.db", None)


def test_a_password_given_as_a_reference_is_kept_as_written(monkeypatch):
    from lumen.flowdash.uploads import (
        Connection, resolved_location, stored_location,
    )

    reference = "postgresql://ana:${PG_PASSWORD}@db.example.com/fleet"
    assert stored_location(reference) == (reference, None)
    monkeypatch.setenv("PG_PASSWORD", "hunter2")
    assert resolved_location(Connection("fleet", reference, "ana", "")) == SECRET_URL
    monkeypatch.delenv("PG_PASSWORD")
    with pytest.raises(UploadError, match="credentials can't be read.*PG_PASSWORD is not set"):
        resolved_location(Connection("fleet", reference, "ana", ""))


def test_without_encryption_a_written_password_is_refused(monkeypatch):
    from panel_flowdash import secret_store

    from lumen.flowdash.uploads import stored_location

    monkeypatch.setattr(secret_store, "encryption_available", lambda: False)
    with pytest.raises(UploadError, match=r"panel-flowdash\[secrets\].*\$\{PG_PASSWORD\}"):
        stored_location(SECRET_URL)


def test_the_manifest_never_holds_the_password_and_a_restart_connects_again(context, tmp_path, secret_key, opened):
    import json

    project = _project(tmp_path, context)
    uploads = project.context[UPLOADS_KEY]
    connection = uploads.connect(SECRET_URL, "warehouse", app=Admin())
    assert connection.location == "postgresql://ana:***@db.example.com/fleet"
    manifest = (uploads.directory / "manifest.json").read_text()
    assert "hunter2" not in manifest
    assert json.loads(manifest)["sources"][0]["secret"]["key_id"]

    restored = Uploads(tmp_path / "uploads", {}, project=project)
    restored.restore()
    assert opened[-1] == SECRET_URL and "warehouse" in restored.sources
    assert restored.connection_problem("warehouse") is None


def test_a_source_whose_key_is_gone_stays_listed_until_its_credentials_are_entered_again(
    context, tmp_path, secret_key, opened, monkeypatch,
):
    from panel_flowdash.secret_store import KEY_ENV, fernet_key

    from lumen.flowdash.data import source_summaries

    project = _project(tmp_path, context)
    project.context[UPLOADS_KEY].connect(SECRET_URL, "warehouse", app=Admin())
    monkeypatch.setenv(KEY_ENV, fernet_key())
    sources = {ref: source for ref, source in context[SOURCES_KEY].items() if ref != "warehouse"}
    restored = Uploads(tmp_path / "uploads", sources, project=project)
    restored.restore()
    assert "warehouse" not in restored.sources and restored.connection("warehouse") is not None
    problem = restored.connection_problem("warehouse")
    assert "credentials can't be read" in problem and "enter it again" in problem and "hunter2" not in problem
    summary = source_summaries({**context, SOURCES_KEY: sources, UPLOADS_KEY: restored})[-1]
    assert (summary.ref, summary.kind, summary.problem) == ("warehouse", "PostgreSQL", problem)
    assert summary.files == ("postgresql://ana:***@db.example.com/fleet",)

    again = restored.connect(SECRET_URL, "warehouse", app=Admin())
    assert again.ref == "warehouse" and restored.connection_problem("warehouse") is None and "warehouse" in restored.sources


def test_files_are_shown_relative_to_the_project_or_by_name(tmp_path):
    from lumen.flowdash.data import display_path

    inside = tmp_path / "data" / "owners.parquet"
    assert display_path(str(inside), [tmp_path]) == "data/owners.parquet"
    assert display_path("/srv/elsewhere/owners.parquet", [tmp_path]) == "owners.parquet"
    assert display_path("turbines/turbines.csv", [tmp_path]) == "turbines/turbines.csv"
    assert display_path(SECRET_URL) == "postgresql://ana:***@db.example.com/fleet"


def test_a_posix_style_path_is_shown_by_name_even_under_windows_path_semantics(monkeypatch):
    # On Windows, Path("/srv/x").is_absolute() is False (no drive letter), unlike on
    # POSIX; display_path must still recognize the leading "/" as absolute so it
    # falls back to the file name instead of showing the server's folders raw.
    import pathlib

    from lumen.flowdash.data import display_path

    monkeypatch.setattr(pathlib, "Path", pathlib.PureWindowsPath)
    assert display_path("/srv/elsewhere/owners.parquet") == "owners.parquet"


def test_the_day_a_source_was_added_has_no_platform_specific_format_code():
    import datetime as dt

    from lumen.flowdash.data import _day

    # %-d (glibc/macOS) and %#d (Windows) are mutually incompatible strftime flags;
    # _day must avoid both so it does not raise on either platform.
    assert _day(dt.datetime(2026, 11, 7, tzinfo=dt.UTC)) == "7 Nov"
    assert _day(None) == ""


def test_a_windows_style_location_names_a_source_by_its_file_stem(tmp_path, opened):
    # re.split(r"[/:?@]", ...) alone leaves Windows backslashes untouched, so a local
    # path's drive letter and directories were swept into the ref instead of the stem.
    uploads = Uploads(tmp_path / "uploads", {})
    preview, _ = uploads.read_location(r"C:\Users\ana\Documents\plant.db", app=Admin())
    assert preview.ref == "plant"
