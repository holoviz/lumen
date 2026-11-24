"""The project file's ``catalog:`` section and the admin view's Catalog section, without a browser."""
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel_flowdash.admin.form import using_app

from lumen.flowdash import (
    CATALOG_KEY, SOURCE_ALIASES_KEY, SOURCES_KEY, Catalog, ProjectCatalog,
    assistant_tables, hidden_tables,
)
from lumen.flowdash.app import build_routes
from lumen.flowdash.catalog import hidden_reads, raw_section
from lumen.flowdash.catalog_admin import (
    CatalogSettings, set_assistant, table_detail, table_info,
)
from lumen.flowdash.project import PROJECT_FILE, ProjectError, load_project
from lumen.flowdash.results import derive
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "orders": "SELECT * FROM (VALUES (1, 'paid', 30), (2, 'paid', 20), (3, 'open', 40)) AS t(order_id, status, amount)",
    "customers": "SELECT * FROM (VALUES (1, 'Ada'), (2, 'Bo')) AS t(customer_id, name)",
    "raw_events": "SELECT * FROM (VALUES ('x', 1)) AS t(kind, n)",
}

YAML = """\
general:
  admins: [ana]
catalog:
  shop:
    tables:
      raw_events:
        assistant: false
"""


class Admin:
    def __init__(self, admin=True):
        self.admin = admin

    def _still_admin(self):
        return self.admin


@pytest.fixture
def source():
    return DuckDBSource(uri=":memory:", tables=TABLES, name="shop")


@pytest.fixture
def context(source):
    return {SOURCES_KEY: {"shop": source}, SOURCE_ALIASES_KEY: {"data": "shop"}}


@pytest.fixture
def project(tmp_path, context):
    path = tmp_path / PROJECT_FILE
    path.write_text(YAML)
    return load_project(path, context)


def _app_context(context, project):
    return {**context, CATALOG_KEY: ProjectCatalog(project)}


def test_the_section_reads_the_tables_turned_off(project):
    assert project.get("catalog") == Catalog({"shop": frozenset({"raw_events"})})


@pytest.mark.parametrize(("text", "line", "message"), [
    ("catalog: [shop]\n", 1, "maps each source"),
    ("catalog:\n  shop: [orders]\n", 2, "takes `tables:`"),
    ("catalog:\n  shop:\n    hidden: [orders]\n", 3, "Unknown key 'hidden'"),
    ("catalog:\n  shop:\n    tables: [orders]\n", 3, "map each table"),
    ("catalog:\n  shop:\n    tables:\n      orders: false\n", 4, "takes `assistant: false`"),
    ("catalog:\n  shop:\n    tables:\n      orders:\n        assistant: no thanks\n", 5, "true or false"),
    ("catalog:\n  shop:\n    tables:\n      orders:\n        describe: x\n", 5, "Unknown key 'describe'"),
])
def test_a_wrong_shape_is_reported_where_it_is_written(tmp_path, context, text, line, message):
    path = tmp_path / PROJECT_FILE
    path.write_text(text)
    with pytest.raises(ProjectError) as info:
        load_project(path, context)
    problem = info.value.problems[0]
    assert message in problem.message and problem.line == line


def test_an_unknown_source_or_table_is_kept_with_a_warning(tmp_path, context):
    path = tmp_path / PROJECT_FILE
    path.write_text("catalog:\n  warehouse:\n    tables:\n      x: {assistant: false}\n"
                    "  shop:\n    tables:\n      ordres: {assistant: false}\n")
    project = load_project(path, context)
    assert project.get("catalog").hidden == {"warehouse": {"x"}, "shop": {"ordres"}}
    messages = [w.message for w in project.warnings]
    assert any("does not serve" in m and "'warehouse'" in m and "shop" in m for m in messages)
    assert any("no table 'ordres'" in m for m in messages)


def test_hidden_tables_follow_aliases_and_query_results(project, context, source):
    context = _app_context(context, project)
    assert hidden_tables(source, context) == {"raw_events"}
    assert assistant_tables(source, context) == ["orders", "customers"]
    result = derive(source, "paid", "SELECT * FROM orders WHERE status = 'paid'")
    assert hidden_tables(result, context) == {"raw_events"}
    other = DuckDBSource(uri=":memory:", tables=TABLES)
    assert hidden_tables(other, context) == frozenset()
    assert hidden_tables(source, {**context, CATALOG_KEY: Catalog({"data": frozenset({"orders"})})}) == {"orders"}
    assert hidden_tables(source, {SOURCES_KEY: {"shop": source}}) == frozenset()


def test_an_earlier_name_of_a_hidden_table_is_hidden_too(tmp_path):
    from lumen.flowdash.nodes import add_legacy_tables

    source = DuckDBSource(uri=":memory:", tables=dict(TABLES), name="shop")
    add_legacy_tables(source, {"raw_events_csv": "raw_events"})
    context = {SOURCES_KEY: {"shop": source}, CATALOG_KEY: Catalog({"shop": frozenset({"raw_events"})})}
    assert hidden_tables(source, context) == {"raw_events", "raw_events_csv"}


@pytest.mark.parametrize(("sql", "read"), [
    ("SELECT * FROM orders", []),
    ("SELECT * FROM raw_events", ["raw_events"]),
    ('SELECT * FROM "RAW_EVENTS"', ["raw_events"]),
    ("SELECT * FROM main.raw_events", ["raw_events"]),
    ("SELECT * FROM other.raw_events", []),
    ("WITH raw_events AS (SELECT 1 AS n) SELECT * FROM raw_events", []),
    ("SELECT * FROM orders JOIN (SELECT kind FROM raw_events) r ON true", ["raw_events"]),
])
def test_hidden_reads_finds_the_hidden_tables_a_query_names(source, sql, read):
    assert hidden_reads(sql, source, frozenset({"raw_events"})) == read


def test_a_result_table_named_like_a_hidden_table_is_not_one(source):
    result = derive(source, "raw_events", "SELECT * FROM orders")
    assert hidden_reads("SELECT * FROM raw_events", result, frozenset({"raw_events"})) == []


def test_an_admin_turns_a_table_off_and_on(project, context, source):
    catalog = ProjectCatalog(project)
    assert set_assistant(project, "shop", "orders", False, app=Admin()) == []
    assert raw_section(project) == {"shop": {"tables": {"raw_events": {"assistant": False}, "orders": {"assistant": False}}}}
    assert catalog.hidden == {"shop": {"raw_events", "orders"}}

    assert set_assistant(project, "shop", "raw_events", True, app=Admin()) == []
    assert set_assistant(project, "shop", "orders", True, app=Admin()) == []
    assert raw_section(project) == {}
    assert catalog.hidden == {}
    assert load_project(project.path).raw("general") == {"admins": ["ana"]}


def test_turning_a_table_on_clears_it_under_an_alias(tmp_path, context):
    path = tmp_path / PROJECT_FILE
    path.write_text("catalog:\n  data:\n    tables:\n      orders: {assistant: false}\n      customers: {assistant: false}\n")
    project = load_project(path, context)
    assert set_assistant(project, "shop", "orders", True, app=Admin()) == []
    assert raw_section(project) == {"data": {"tables": {"customers": {"assistant": False}}}}


@pytest.mark.parametrize("app", [None, Admin(admin=False)], ids=["no app", "no longer admin"])
def test_only_an_admin_turns_tables_off(project, app):
    before = open(project.path).read()
    assert "Only admins" in set_assistant(project, "shop", "orders", False, app=app)[0].message
    assert open(project.path).read() == before


def test_build_routes_gives_the_app_the_catalog_the_admin_view_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,t_state,p_cap\nAlta,CA,6.0\n")
    (tmp_path / "readings.csv").write_text("t,value\n1,2\n")
    (tmp_path / PROJECT_FILE).write_text(YAML.replace("shop", "local").replace("raw_events", "readings"))
    app = build_routes(["turbines.csv", "readings.csv"], db_path=tmp_path / "dash.db",
                       project=tmp_path / PROJECT_FILE)["/"]()
    context = app._editor.context
    source = context[SOURCES_KEY]["local"]
    assert assistant_tables(source, context) == ["turbines"]
    assert set_assistant(app._settings_project(), "local", "readings", True, app=Admin()) == []
    assert assistant_tables(source, context) == ["turbines", "readings"]


def test_table_info_and_detail_read_columns_rows_and_first_rows(source):
    info = table_info(source, "orders")
    assert (info.rows, info.columns, info.error) == (3, ["order_id", "status", "amount"], None)
    source.describe("orders", "One row per order", columns={"amount": "Order value in EUR"})
    detail = table_detail(source, "orders")
    assert detail.rows == 3 and detail.description == "One row per order"
    columns = {c.name: c for c in detail.columns}
    assert columns["amount"].type == "integer" and columns["amount"].values == "20 to 40"
    assert columns["amount"].description == "Order value in EUR"
    assert columns["status"].values in ("paid, open", "open, paid")
    assert list(detail.sample.columns) == ["order_id", "status", "amount"] and len(detail.sample) == 3


def test_table_info_reports_a_table_it_cannot_read():
    source = DuckDBSource(uri=":memory:", tables={"broken": "SELECT * FROM missing_table"})
    info = table_info(source, "broken")
    assert info.error and "missing_table" in info.error


def _settings(project, admin=True, editable=True):
    with using_app(Admin(admin)):
        return CatalogSettings(project, editable=editable)


def _row_text(row):
    return "".join(str(getattr(c, "object", "")) for c in row.select() if isinstance(getattr(c, "object", None), str))


def test_the_section_lists_every_table_with_its_rows_and_columns(project):
    view = _settings(project)
    rows = {(r["ref"], r["table"]): r for r in view.rows()}
    assert list(rows) == [("shop", "orders"), ("shop", "customers"), ("shop", "raw_events")]
    assert [r["assistant"] for r in rows.values()] == [True, True, False]
    assert rows[("shop", "orders")]["info"].rows == 3
    texts = [_row_text(row) for row in view._table._rows]
    assert "orders" in texts[0] and "3 columns" in texts[0] and "order_id, status, amount" in texts[0]
    assert "DuckDB" in texts[0]
    assert [c.label for c in view._chip_row] == ["All 3", "Off 1"]


def test_the_switch_turns_a_table_off_for_everyone(project, context, source):
    view = _settings(project)
    switches = [c for c in view._table._rows[0].select() if "flowdash-catalog-assistant" in getattr(c, "css_classes", [])]
    switches[0].value = False
    assert raw_section(project)["shop"]["tables"]["orders"] == {"assistant": False}
    assert hidden_tables(source, _app_context(context, project)) == {"orders", "raw_events"}
    assert [c.label for c in view._chip_row] == ["All 3", "Off 2"]


def test_the_switch_refuses_someone_no_longer_admin(project):
    view = _settings(project, admin=False)
    assert not view.set_assistant("shop", "orders", False)
    assert "Only admins" in view._alert.object and view._alert.visible
    assert "orders" not in raw_section(project)["shop"]["tables"]
    assert _settings(project, editable=False)._table._rows[0].select(lambda o: o.__class__.__name__ == "Switch")[0].disabled


def test_search_and_chips_filter_the_tables(project):
    view = _settings(project)
    view.query = "customer_id"
    assert [r["table"] for r in view.visible_rows()] == ["customers"]
    view.query = ""
    view.filter = "off"
    assert [r["table"] for r in view.visible_rows()] == ["raw_events"]
    assert len(view._table._rows) == 1


def test_several_sources_get_a_chip_each(tmp_path, source):
    other = DuckDBSource(uri=":memory:", tables={"t": "SELECT 1 AS x"}, name="other")
    path = tmp_path / PROJECT_FILE
    path.write_text("")
    project = load_project(path, {SOURCES_KEY: {"shop": source, "other": other}})
    view = _settings(project)
    assert [c.label for c in view._chip_row] == ["All 4", "shop 3", "other 1"]
    view.filter = "source:other"
    assert [r["table"] for r in view.visible_rows()] == ["t"]


def test_the_drawer_shows_a_tables_columns_and_first_rows(project):
    view = _settings(project)
    view.open_table("shop", "orders")
    assert view._drawer.open and view._drawer_title.object == "orders"
    assert "Uses it" in view._details.object and "<dd>3</dd>" in view._details.object
    assert "order_id" in view._columns.object and "20 to 40" in view._columns.object
    assert "paid" in view._sample.object and view._detail_body.visible
    view.close_table()
    assert not view._drawer.open and view.selected is None


def test_a_long_list_is_cut_short_until_searched(tmp_path, monkeypatch):
    import lumen.flowdash.catalog_admin as catalog_admin

    monkeypatch.setattr(catalog_admin, "ROWS_SHOWN", 2)
    source = DuckDBSource(uri=":memory:", tables={f"t{i}": f"SELECT {i} AS x" for i in range(5)}, name="many")
    path = tmp_path / PROJECT_FILE
    path.write_text("")
    view = _settings(load_project(path, {SOURCES_KEY: {"many": source}}))
    assert len(view._table._rows) == 2
    assert view._more.visible and "Showing 2 of 5 tables" in view._more.object
    view.query = "t4"
    assert len(view._table._rows) == 1 and not view._more.visible


def test_a_source_that_cannot_list_its_tables_says_so(tmp_path, source):
    class Broken(DuckDBSource):
        def get_tables(self):
            raise RuntimeError("connection refused")

    path = tmp_path / PROJECT_FILE
    path.write_text("")
    view = _settings(load_project(path, {SOURCES_KEY: {"shop": source, "remote": Broken(uri=":memory:", tables={})}}))
    assert [r["ref"] for r in view.rows()] == ["shop"] * 3
    assert "connection refused" in _row_text(view._table._rows[-1])


def test_the_section_escapes_what_the_source_says(tmp_path):
    hostile = "<img src=x onerror=alert(1)>"
    source = DuckDBSource(uri=":memory:", tables={"t": f"SELECT '{hostile}' AS \"{hostile}\""}, name="shop")
    source.describe("t", hostile)
    path = tmp_path / PROJECT_FILE
    path.write_text("")
    view = _settings(load_project(path, {SOURCES_KEY: {"shop": source}}))
    view.open_table("shop", "t")
    html = _row_text(view._table._rows[0]) + view._columns.object + view._sample.object + view._details.object
    assert "&lt;img src=x onerror=alert(1)&gt;" in html and "<img" not in html
