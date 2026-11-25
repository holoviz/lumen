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
