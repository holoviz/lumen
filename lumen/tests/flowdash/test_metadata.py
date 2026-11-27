"""
Explicit schema/metadata refresh, TableStatsStore column stats in the table
drawer, admin-only LLM enrichment with review/edit, and persistence across
restarts (Data view item 1).
"""

from __future__ import annotations

import json

from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel_flowdash import MemoryDashboardStore
from panel_flowdash.admin.form import using_app
from panel_flowdash.auth import AuthConfig, Identity
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from lumen.agent.data import _METASETS, metaset
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.data.sharing import sharing
from lumen.data.stats import TableStatsStore, get_stats_store, set_stats_store
from lumen.flowdash import DESCRIPTIONS_KEY, SOURCES_KEY
from lumen.flowdash.app import build_routes
from lumen.flowdash.catalog import ColumnInfo, table_detail
from lumen.flowdash.catalog_admin import CatalogSettings, _columns_html
from lumen.flowdash.data_section import DataPage
from lumen.flowdash.metadata import (
    DESCRIPTIONS_FILE, NO_MODEL, NOT_ADMIN_ENRICH, NOT_ADMIN_REFRESH,
    apply_descriptions, build_enrichment_prompt, configured_model,
    description_store, enrich_table, refresh_source, refresh_table,
)
from lumen.flowdash.nodes import _SCHEMAS_READ, _read_schemas
from lumen.flowdash.project import PROJECT_FILE, load_project
from lumen.flowdash.uploads import UPLOADS_KEY, Uploads
from lumen.sources.duckdb import DuckDBSource

TABLES = {
    "orders": (
        "SELECT * FROM (VALUES "
        "(1, 'paid', 30, '2026-01-01'), "
        "(2, 'paid', 20, '2026-01-02'), "
        "(3, 'open', NULL, '2026-01-03'), "
        "(4, 'open', 40, '2026-01-04')"
        ") AS t(order_id, status, amount, ordered_on)"
    ),
    "customers": "SELECT * FROM (VALUES (1, 'Ada'), (2, 'Bo')) AS t(customer_id, name)",
}


def _fake_model(description: str = "Customer orders with payment status and amount.", columns: dict[str, str] | None = None):
    """A deterministic offline pydantic-ai FunctionModel returning a TableEnrichment tool call."""
    payload = {
        "description": description,
        "columns": columns
        if columns is not None
        else {
            "order_id": "Unique identifier for each order",
            "status": "Order payment status (paid or open)",
            "amount": "Order total in USD",
            "ordered_on": "ISO date when the order was placed",
            "nonexistent_col": "Should be filtered out",
        },
    }

    def respond(messages, info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args=payload)])

    return FunctionModel(respond)


class AgentSession:
    """The parts of panel-flowdash's AgentSession that model resolution reads, without a model policy."""

    def __init__(self, model):
        self.runner = SimpleNamespace(model=model, agent=None, run_kwargs={})
        self.ledger = None

    def plan_request(self, target):
        return None, {}, None


class Session:
    def __init__(self, project, *, admin: bool = True, user: str = "ana", agent_model=None):
        self.project = project
        self.is_admin = self.admin = admin
        self.store = MemoryDashboardStore()
        self._identity = Identity(user)
        self._agent_session = AgentSession(agent_model) if agent_model is not None else None
        self.notified: list[tuple[str, str]] = []

    def _still_admin(self) -> bool:
        return self.admin

    def _settings_project(self):
        return self.project

    def _default_allow(self) -> bool:
        return False

    def _notify(self, severity: str, message: str, duration: int = 3000):
        self.notified.append((severity, message))


@pytest.fixture(autouse=True)
def isolated_stats_store(tmp_path):
    """Keep TableStatsStore isolated per test so disk caches never leak across tests."""
    store = TableStatsStore(cache_dir=tmp_path / "stats_cache")
    set_stats_store(store)
    yield store
    set_stats_store(None)


@pytest.fixture
def source():
    return DuckDBSource(uri=":memory:", tables=dict(TABLES), name="shop")


@pytest.fixture
def project(tmp_path, source):
    sources = {"shop": source}
    path = tmp_path / PROJECT_FILE
    path.write_text("general:\n  admins: [ana]\n")
    proj = load_project(path, {SOURCES_KEY: sources})
    uploads = proj.context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", sources, project=proj)
    proj.context[DESCRIPTIONS_KEY] = uploads.descriptions
    return proj


def test_refresh_table_invalidates_all_caches_and_reads_fresh_data(source):
    # Warm schema, metadata, overview schema and TableStatsStore caches.
    detail1 = table_detail(source, "orders")
    assert detail1.rows == 4
    _read_schemas(source, ["orders"])
    assert "orders" in _SCHEMAS_READ.get(source, {})
    assert get_stats_store().get(source, "orders") is not None

    # Mutate underlying table definition so row count and columns change.
    source.tables["orders"] = (
        "SELECT * FROM (VALUES (1, 'paid', 30, '2026-01-01', 'US'), (2, 'paid', 20, '2026-01-02', 'DE')) AS t(order_id, status, amount, ordered_on, country)"
    )

    # Passive read still sees stale cached schema/metadata.
    assert source.get_metadata("orders")["rows"] == 4

    # Explicit refresh invalidates all caches and returns the updated schema, metadata and stats.
    detail2 = refresh_table(source, "orders")
    assert detail2.rows == 2
    assert [c.name for c in detail2.columns] == ["order_id", "status", "amount", "ordered_on", "country"]
    assert detail2.stats is not None and detail2.stats.rows == 2
    assert "orders" not in _SCHEMAS_READ.get(source, {})


def test_refresh_source_invalidates_and_fetches_every_table(source):
    table_detail(source, "orders")
    table_detail(source, "customers")
    source.tables["invoices"] = "SELECT 1 AS invoice_id, 99.5 AS total"

    infos, error = refresh_source(source)
    assert error is None
    assert set(infos) == {"orders", "customers", "invoices"}
    assert infos["invoices"].rows == 1
    assert infos["invoices"].columns == ["invoice_id", "total"]
    assert get_stats_store().get(source, "invoices") is not None


def test_refresh_source_reads_and_profiles_a_capped_number_of_tables(source, monkeypatch):
    for i in range(4):
        source.tables[f"extra_{i}"] = f"SELECT {i} AS n"
    profiled = []
    store = get_stats_store()
    compute = store.compute
    monkeypatch.setattr(store, "compute", lambda src, table: profiled.append(table) or compute(src, table))

    infos, error = refresh_source(source, limit=3, profile=1)

    assert error is None
    assert len(infos) == 3
    assert profiled == [next(iter(infos))]


def _refresh_buttons(page):
    return [o for o in page.__panel__().select() if "lumen-data-source-refresh" in getattr(o, "css_classes", [])]


def test_catalog_and_data_page_explicit_refresh_actions(project, source):
    session = Session(project, admin=True)
    page = DataPage(session, project, admin=True, navigate=lambda _url: None)
    assert len(_refresh_buttons(page)) == 1

    # Mutate table and click per-source refresh.
    source.tables["customers"] = "SELECT * FROM (VALUES (1, 'Ada'), (2, 'Bo'), (3, 'Cy')) AS t(customer_id, name)"
    page.refresh_source("shop")
    assert page.catalog._infos[("shop", "customers")].rows == 3
    assert any("Refreshed metadata for shop" in msg for _, msg in session.notified)

    # Open drawer and test per-table refresh.
    page.catalog.open_table("shop", "customers")
    assert page.catalog._detail is not None and page.catalog._detail.rows == 3
    source.tables["customers"] = "SELECT * FROM (VALUES (1, 'Ada')) AS t(customer_id, name)"
    page.catalog.refresh_selected()
    assert page.catalog._detail.rows == 1
    assert page.catalog._infos[("shop", "customers")].rows == 1
    assert any("Refreshed metadata for customers" in msg for _, msg in session.notified)


def test_viewers_cannot_refresh_metadata(project, source):
    viewer = Session(project, admin=False)
    page = DataPage(viewer, project, admin=False, navigate=lambda _url: None)
    assert _refresh_buttons(page) == []
    page.catalog.open_table("shop", "orders")
    assert not page.catalog._drawer_refresh.visible

    table_detail(source, "orders")
    page.refresh_source("shop")
    page.catalog.refresh_table("shop", "orders")
    assert get_stats_store().get(source, "orders") is not None

    # An admin view whose session lost its admin rights is refused too.
    revoked = Session(project, admin=False)
    view = CatalogSettings(project, editable=True, app=revoked)
    view.refresh_source("shop")
    assert NOT_ADMIN_REFRESH in view._alert.object
    assert get_stats_store().get(source, "orders") is not None


def test_table_stats_store_invalidate_removes_memory_and_disk_entries(tmp_path, isolated_stats_store):
    db_file = tmp_path / "shop.duckdb"
    import duckdb

    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE items AS SELECT 1 AS id, 'a' AS label")
    conn.close()
    src = DuckDBSource(uri=str(db_file), name="file_shop")
    stats = isolated_stats_store.compute(src, "items")
    assert stats.rows == 1
    assert isolated_stats_store.get(src, "items") is not None
    assert list((tmp_path / "stats_cache").glob("*.json"))

    isolated_stats_store.invalidate(src, "items")
    assert isolated_stats_store.get(src, "items") is None
    assert not list((tmp_path / "stats_cache").glob("*.json"))


def test_table_stats_store_invalidate_whole_source_names_its_disk_entries(tmp_path, isolated_stats_store):
    import duckdb

    db_file = tmp_path / "shop.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE items AS SELECT 1 AS id")
    conn.execute("CREATE TABLE tags AS SELECT 'a' AS label")
    conn.close()
    src = DuckDBSource(uri=str(db_file), name="file_shop")
    isolated_stats_store.compute(src, "items")
    isolated_stats_store.compute(src, "tags")
    assert len(list((tmp_path / "stats_cache").glob("*.json"))) == 2

    # A new store has the entries on disk only, as after a restart, and drops just the tables it is given.
    fresh = TableStatsStore(cache_dir=tmp_path / "stats_cache")
    fresh.invalidate(src, tables=["items"])
    assert len(list((tmp_path / "stats_cache").glob("*.json"))) == 1
    isolated_stats_store.invalidate(src)
    assert isolated_stats_store.get(src, "tags") is None
    assert not list((tmp_path / "stats_cache").glob("*.json"))


def test_drawer_shows_stats_that_take_longer_than_it_waits(project, source, monkeypatch):
    from concurrent.futures import Future

    store = get_stats_store()
    pending: Future = Future()
    monkeypatch.setattr(store, "compute_within", lambda src, table, timeout: (None, pending))
    session = Session(project, admin=True)
    view = CatalogSettings(project, editable=True, app=session)
    computed = store.compute(source, "orders")
    store.invalidate(source, "orders")

    pending.set_result(computed)
    # Outside a served session the background work runs at once, so the drawer is drawn again with the stats.
    monkeypatch.setattr(store, "get", lambda src, table: computed if pending.done() and view._detail is not None else None)
    view.open_table("shop", "orders")
    assert view._detail.stats is computed
    assert "<th>Nulls</th>" in view._columns.object


def test_table_detail_and_drawer_show_table_stats_store_metrics(project, source):
    detail = table_detail(source, "orders")
    assert detail.stats is not None
    assert detail.stats_note == "Exact"
    cols = {c.name: c for c in detail.columns}
    assert cols["amount"].nulls == "25%"
    assert cols["order_id"].nulls == "0%"
    assert "unique" in cols["order_id"].distinct
    assert cols["status"].distinct == "2"
    assert "paid (2)" in cols["status"].stats_summary
    assert "open (2)" in cols["status"].stats_summary
    # Backward compatibility of existing ColumnInfo fields.
    assert cols["amount"].type == "integer"
    assert cols["amount"].values == "20 to 40"

    session = Session(project, admin=True)
    with using_app(session):
        view = CatalogSettings(project, editable=True, app=session)
    view.open_table("shop", "orders")
    assert "<dt>Stats</dt><dd>Exact</dd>" in view._details.object
    assert "<th>Nulls</th>" in view._columns.object
    assert "<th>Distinct</th>" in view._columns.object
    assert "25%" in view._columns.object
    assert "paid (2)" in view._columns.object


def test_columns_html_renders_pk_and_fk_and_escapes_values():
    html = _columns_html(
        [
            ColumnInfo(
                name="id",
                type="integer",
                values="1 to 10",
                nulls="0%",
                distinct="10 (unique)",
                primary_key=True,
                references="users.<script>",
            ),
        ]
    )
    assert '<span class="badge">PK</span>' in html
    assert "users.&lt;script&gt;" in html and "<script>" not in html


def _real_app(tmp_path, monkeypatch, ai: str = "", **kwargs):
    from panel_flowdash.ai import ledger_for

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("panel_flowdash.app.resolve_identity", lambda _c: Identity(user="ana"))
    monkeypatch.delenv("FLOWDASH_AGENT_MODEL", raising=False)
    monkeypatch.delenv("FLOWDASH_AGENT_MODELS", raising=False)
    (tmp_path / "orders.csv").write_text("order_id,amount\n1,30\n2,20\n")
    (tmp_path / PROJECT_FILE).write_text("general:\n  admins: [ana]\n" + ai)
    app = build_routes(
        ["orders.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE, auth_config=AuthConfig(),
        **kwargs,
    )["/"]()
    return app, ledger_for(tmp_path)


def test_configured_model_is_none_without_an_assistant(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOWDASH_AGENT_MODEL", "test")
    assert configured_model(None) is None
    app, _ = _real_app(tmp_path, monkeypatch)
    assert app._agent_session is None
    assert configured_model(app) is None


async def test_enrichment_follows_the_apps_model_policy(tmp_path, monkeypatch):
    from panel_agents import ModelChoice

    main = _fake_model()
    app, ledger = _real_app(
        tmp_path, monkeypatch, "ai:\n  models:\n    - id: fn:main\n      monthly_limit: 1\n",
        agent_model=["test"], agent_models=[ModelChoice("test", model="test"), ModelChoice("fn:main", model=main)],
    )
    source = app._editor.context[SOURCES_KEY]["orders"]
    # The model picked in the assistant's drawer is the one enrichment runs on.
    app._agent_session.runner.model = main
    monkeypatch.setattr(app, "_still_admin", lambda: True)
    model = configured_model(app)
    assert model.model_id == "fn:main" and model.ledger is ledger

    draft = await enrich_table(source, "orders", ref="orders", app=app)
    assert draft.description == "Customer orders with payment status and amount."
    # The request counts against the model's monthly spend, as the assistant's do.
    assert ledger.month("fn:main")[1] == 1

    # Over its monthly limit with no fallback, enrichment is refused as the assistant's requests are.
    ledger.add("fn:main", 5.0)
    with pytest.raises(ValueError, match="monthly spending limit"):
        await enrich_table(source, "orders", ref="orders", app=app)


async def test_enrich_table_uses_model_filters_unknown_columns_and_respects_sharing(project, source):
    admin = Session(project, admin=True, agent_model=_fake_model())
    draft = await enrich_table(source, "orders", ref="shop", app=admin)
    assert draft.description == "Customer orders with payment status and amount."
    assert set(draft.columns) == {"order_id", "status", "amount", "ordered_on"}
    assert "nonexistent_col" not in draft.columns

    # Non-admin is rejected before calling any model.
    viewer = Session(project, admin=False, agent_model=_fake_model())
    with pytest.raises(PermissionError, match="Only admins"):
        await enrich_table(source, "orders", ref="shop", app=viewer)

    # Prompt respects lumen.data.sharing=none (no sample rows or literal values).
    detail = table_detail(source, "orders")
    with sharing.param.update(share="none"):
        prompt = build_enrichment_prompt("shop", "orders", detail, source=source)
    assert "Sample rows" not in prompt
    assert "paid" not in prompt


async def test_drawer_enrich_review_edit_save_and_assistant_metaset_sync(project, source):
    # Warm the agent's cached Metaset BEFORE enrichment.
    meta_before = await metaset(source)
    slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}orders"
    assert not meta_before.catalog[slug].description
    assert source in _METASETS

    admin = Session(project, admin=True, user="ana", agent_model=_fake_model())
    view = CatalogSettings(project, editable=True, app=admin)
    view.open_table("shop", "orders")
    assert view._enrich_btn.visible and not view._enrich_btn.disabled

    # Generate draft: opens review panel, does NOT save yet.
    assert view.start_enrich()
    assert view._enrich_panel.visible
    assert view._enrich_table_input.value == "Customer orders with payment status and amount."
    assert view._enrich_col_inputs["amount"].value == "Order total in USD"
    assert not source.descriptions
    assert not description_store(project).path.exists()

    # Edit draft in review panel before saving.
    view._enrich_table_input.value = "Reviewed orders table description."
    view._enrich_col_inputs["amount"].value = "Reviewed order value in EUR."
    assert view.save_descriptions()
    assert not view._enrich_panel.visible

    # Drawer, table list and source.get_metadata() immediately reflect the saved descriptions.
    assert "Reviewed orders table description." in view._details.object
    assert "Reviewed order value in EUR." in view._columns.object
    md = source.get_metadata("orders")
    assert md["description"] == "Reviewed orders table description."
    assert md["columns"]["amount"]["description"] == "Reviewed order value in EUR."

    # Assistant's Metaset invalidates and shows the saved descriptions.
    meta_after = await metaset(source)
    assert meta_after is not meta_before
    assert meta_after.catalog[slug].description == "Reviewed orders table description."
    assert "Reviewed orders table description." in meta_after.compact_context()
    assert "Reviewed order value in EUR." in meta_after.table_detail(slug)

    # Calling get_metadata() again still retains column descriptions (no in-place pop bug).
    md_again = source.get_metadata("orders")
    assert md_again["columns"]["amount"]["description"] == "Reviewed order value in EUR."


def test_viewer_and_revoked_admin_cannot_enrich_or_save_descriptions(project, source):
    viewer = Session(project, admin=False, agent_model=_fake_model())
    viewer_view = CatalogSettings(project, editable=False, app=viewer)
    viewer_view.open_table("shop", "orders")
    assert not viewer_view._enrich_btn.visible
    assert not viewer_view._edit_desc_btn.visible
    assert not viewer_view.start_enrich()
    assert not viewer_view.open_descriptions_editor()

    # Admin whose rights were revoked mid-session is refused server-side on both enrich and save.
    revoked = Session(project, admin=False, agent_model=_fake_model())
    view = CatalogSettings(project, editable=True, app=revoked)
    view.open_table("shop", "orders")
    assert not view.start_enrich()
    assert NOT_ADMIN_ENRICH in view._enrich_alert.object

    # Even if the editor was opened while admin, revoking admin before save stops the write.
    revoked.admin = True
    assert view.open_descriptions_editor()
    view._enrich_table_input.value = "Should not save"
    revoked.admin = False
    assert not view.save_descriptions()
    assert NOT_ADMIN_ENRICH in view._enrich_alert.object
    assert not source.descriptions
    assert not description_store(project).path.exists()


def test_descriptions_editor_waits_for_the_drawer_to_load(project, source):
    store = description_store(project)
    admin = Session(project, admin=True, agent_model=_fake_model())
    assert store.save("shop", "orders", "Orders", {"amount": "Order total"}, source=source, app=admin) == []
    view = CatalogSettings(project, editable=True, app=admin)
    view.selected = ("shop", "orders")
    view._detail = None
    view._render_head()

    assert view._edit_desc_btn.disabled and view._enrich_btn.disabled
    assert not view.open_descriptions_editor()
    assert not view.start_enrich()
    assert not view.save_descriptions()
    assert store.get("shop", "orders").columns == {"amount": "Order total"}
    assert source.descriptions["orders"]["columns"]["amount"] == {"description": "Order total"}

    view.open_table("shop", "orders")
    assert not view._edit_desc_btn.disabled
    assert view.open_descriptions_editor()
    assert view._enrich_col_inputs["amount"].value == "Order total"


def test_enrich_button_disabled_without_configured_model(project, monkeypatch):
    admin = Session(project, admin=True, agent_model=None)
    view = CatalogSettings(project, editable=True, app=admin)
    view.open_table("shop", "orders")
    assert view._enrich_btn.visible and view._enrich_btn.disabled
    assert view._enrich_btn.description == NO_MODEL
    assert not view.start_enrich()
    assert NO_MODEL in view._enrich_alert.object


async def test_descriptions_persist_beside_uploads_manifest_and_survive_restart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,t_state,p_cap\nAlta,CA,6.0\nRoscoe,TX,4.5\n")
    (tmp_path / PROJECT_FILE).write_text("general:\n  admins: [ana]\n")

    # First run: save descriptions through the app's DescriptionStore.
    app1 = build_routes(["turbines.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE)["/"]()
    ctx1 = app1._editor.context
    store1 = ctx1[DESCRIPTIONS_KEY]
    src1 = ctx1[SOURCES_KEY]["turbines"]
    assert store1.path == tmp_path / "uploads" / DESCRIPTIONS_FILE

    admin = Session(app1._settings_project(), admin=True, user="ana")
    problems = store1.save(
        "turbines",
        "turbines",
        "Wind turbine installations with rated capacity.",
        {"p_name": "Wind project name", "p_cap": "Project capacity in MW"},
        source=src1,
        app=admin,
        origin="llm",
    )
    assert problems == []
    assert store1.path.is_file()
    raw = json.loads(store1.path.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["sources"]["turbines"]["turbines"]["origin"] == "llm"
    assert raw["sources"]["turbines"]["turbines"]["updated_by"] == "ana"

    # Simulate restart: fresh build_routes() on the same project folder.
    app2 = build_routes(["turbines.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE)["/"]()
    src2 = app2._editor.context[SOURCES_KEY]["turbines"]
    assert src2 is not src1
    md = src2.get_metadata("turbines")
    assert md["description"] == "Wind turbine installations with rated capacity."
    assert md["columns"]["p_cap"]["description"] == "Project capacity in MW"

    # Fresh assistant Metaset also sees the persisted descriptions immediately after restart.
    meta = await metaset(src2)
    assert "Wind turbine installations with rated capacity." in meta.compact_context()
    assert "Project capacity in MW" in meta.table_detail(f"turbines{SOURCE_TABLE_SEPARATOR}turbines")

    # Clearing descriptions removes the entry from descriptions.json and source.descriptions.
    assert store1.save("turbines", "turbines", "", {}, source=src2, app=admin) == []
    assert not src2.descriptions
    assert json.loads(store1.path.read_text(encoding="utf-8"))["sources"] == {}


def test_apply_descriptions_does_not_query_source(source, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("Startup restore must not query the source")

    monkeypatch.setattr(source, "get_tables", boom)
    monkeypatch.setattr(source, "get_metadata", boom)
    apply_descriptions(source, "orders", "Orders table", {"amount": "Order amount"})
    assert source.descriptions["orders"]["description"] == "Orders table"
    assert source.descriptions["orders"]["columns"]["amount"] == {"description": "Order amount"}


def test_upload_descriptions_follow_the_upload(tmp_path, project):
    uploads = project.context[UPLOADS_KEY]
    store = uploads.descriptions
    admin = Session(project, admin=True)
    upload = uploads.add(b"id,amount\n1,30\n", "sales.csv", app=admin)
    source = uploads.sources[upload.ref]
    assert store.save(upload.ref, "sales", "Sales", {"amount": "Total"}, source=source, app=admin) == []

    # Deleting the upload takes its descriptions with it into the trash...
    assert uploads.delete(upload.ref, app=admin) == []
    assert store.get(upload.ref, "sales") is None

    # ...so a different file uploaded under the same name starts without them, now and after a restart.
    other = uploads.add(b"id,price\n1,9\n", "sales.csv", app=admin)
    assert other.ref == upload.ref
    assert "sales" not in uploads.sources[other.ref].descriptions
    assert store.get(other.ref, "sales") is None
    assert uploads.delete(other.ref, app=admin) == []

    # Bringing the first upload back brings its descriptions back.
    recovered = uploads.recover(upload.id, app=admin)
    assert store.get(recovered.ref, "sales").description == "Sales"
    assert uploads.sources[recovered.ref].descriptions["sales"]["description"] == "Sales"


def test_descriptions_apply_to_a_source_added_while_the_app_runs(project):
    uploads = project.context[UPLOADS_KEY]
    uploads.descriptions.put("sales", {"sales": {"description": "Saved before"}})
    upload = uploads.add(b"id,amount\n1,30\n", "sales.csv", app=Session(project, admin=True))
    assert uploads.sources[upload.ref].get_metadata("sales")["description"] == "Saved before"
