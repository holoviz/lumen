"""
Browser tests for Data view item 1: explicit per-source and per-table metadata
refresh, TableStatsStore column stats in the table drawer, and admin-only LLM
enrichment with editable review and persistence.
"""

from __future__ import annotations

import json

from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pytest_playwright")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel.tests.util import serve_component
from panel_flowdash import MemoryDashboardStore
from panel_flowdash.auth import Identity
from playwright.sync_api import expect
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from lumen.data.stats import TableStatsStore, set_stats_store
from lumen.flowdash import DESCRIPTIONS_KEY, SOURCES_KEY
from lumen.flowdash.data_section import DataPage
from lumen.flowdash.metadata import DESCRIPTIONS_FILE
from lumen.flowdash.project import PROJECT_FILE, load_project
from lumen.flowdash.uploads import UPLOADS_KEY, Uploads
from lumen.sources.duckdb import DuckDBSource

pytestmark = pytest.mark.filterwarnings("ignore:reference already known")


@pytest.fixture(autouse=True)
def stop_servers_and_isolate_stats(tmp_path):
    set_stats_store(TableStatsStore(cache_dir=tmp_path / "stats_cache"))
    yield
    from panel.io.state import state

    state.kill_all_servers()
    set_stats_store(None)


def _fake_model():
    payload = {
        "description": "Suggested orders table description from LLM.",
        "columns": {
            "order_id": "Unique order identifier",
            "status": "Payment status of the order",
            "amount": "Order amount in USD",
        },
    }

    def respond(messages, info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args=payload)])

    return FunctionModel(respond)


class AgentSession:
    def __init__(self, model):
        self.runner = SimpleNamespace(model=model, agent=None, run_kwargs={})
        self.ledger = None

    def plan_request(self, target):
        return None, {}, None


class _Session:
    def __init__(self, project, *, admin: bool = True, user: str = "ana", agent_model=None):
        self.project = project
        self.is_admin = self.admin = admin
        self.store = MemoryDashboardStore()
        self._identity = Identity(user)
        self._agent_session = AgentSession(agent_model) if agent_model is not None else None

    def _still_admin(self) -> bool:
        return self.admin

    def _settings_project(self):
        return self.project

    def _default_allow(self) -> bool:
        return False

    def _notify(self, severity: str, message: str, duration: int = 3000):
        pass


def _make_project(tmp_path):
    source = DuckDBSource(
        uri=":memory:",
        tables={
            "orders": ("SELECT * FROM (VALUES (1, 'paid', 30), (2, 'paid', 20), (3, 'open', NULL), (4, 'open', 40)) AS t(order_id, status, amount)"),
        },
        name="shop",
    )
    sources = {"shop": source}
    path = tmp_path / PROJECT_FILE
    path.write_text("general:\n  admins: [ana]\n")
    project = load_project(path, {SOURCES_KEY: sources})
    uploads = project.context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", sources, project=project)
    project.context[DESCRIPTIONS_KEY] = uploads.descriptions
    return project, source


def test_admin_refreshes_metadata_views_stats_and_enriches_with_llm(page, tmp_path):
    project, source = _make_project(tmp_path)
    session = _Session(project, admin=True, user="ana", agent_model=_fake_model())
    view = DataPage(session, project, admin=True, navigate=lambda _url: None)

    page.set_viewport_size({"width": 1440, "height": 900})
    serve_component(page, view)

    expect(page.locator(".lumen-data-title")).to_have_text("Tables", timeout=20000)
    orders_row = page.locator(".flowdash-admin-row-shop--orders")
    expect(orders_row).to_contain_text("3 columns", timeout=15000)

    # 1. Explicit per-source refresh picks up newly added column/row counts.
    source.tables["orders"] = (
        "SELECT * FROM (VALUES (1, 'paid', 30), (2, 'paid', 20), (3, 'open', NULL), (4, 'open', 40), (5, 'paid', 50)) AS t(order_id, status, amount)"
    )
    page.locator(".lumen-data-source-shop .lumen-data-source-refresh button").click()
    expect(orders_row).to_contain_text("5", timeout=15000)

    # 2. Open table drawer and verify TableStatsStore column stats are shown.
    orders_row.get_by_text("orders").click()
    drawer = page.locator(".flowdash-admin-catalog-drawer")
    expect(drawer).to_contain_text("Exact", timeout=15000)
    expect(drawer).to_contain_text("Nulls")
    expect(drawer).to_contain_text("Distinct")
    expect(drawer).to_contain_text("20%")
    expect(drawer).to_contain_text("paid (3)")

    # 3. Explicit per-table refresh button inside the drawer works.
    drawer.locator(".flowdash-catalog-refresh button").click()
    expect(drawer).to_contain_text("Exact", timeout=15000)

    # 4. Admin clicks "Enrich with LLM" -> editable review panel opens, nothing persisted yet.
    drawer.locator(".flowdash-catalog-enrich button").click()
    review = drawer.locator(".flowdash-catalog-enrich-review")
    expect(review).to_be_visible(timeout=15000)
    table_input = review.locator(".flowdash-enrich-table-desc textarea").first
    expect(table_input).to_have_value("Suggested orders table description from LLM.")
    assert not (tmp_path / "uploads" / DESCRIPTIONS_FILE).exists()

    # 5. Edit table and column descriptions before saving.
    table_input.fill("Edited orders table description.")
    amount_input = review.locator(".flowdash-enrich-col-amount input").first
    expect(amount_input).to_have_value("Order amount in USD")
    amount_input.fill("Edited order amount in EUR")
    review.locator(".flowdash-enrich-save button").click()

    # Saved descriptions appear in the drawer and persist in uploads/descriptions.json.
    expect(review).to_be_hidden(timeout=15000)
    expect(drawer).to_contain_text("Edited orders table description.")
    expect(drawer).to_contain_text("Edited order amount in EUR")

    persisted = json.loads((tmp_path / "uploads" / DESCRIPTIONS_FILE).read_text(encoding="utf-8"))
    entry = persisted["sources"]["shop"]["orders"]
    assert entry["description"] == "Edited orders table description."
    assert entry["columns"]["amount"] == "Edited order amount in EUR"
    assert entry["origin"] == "llm"


def test_viewer_sees_stats_but_not_refresh_or_llm_enrichment(page, tmp_path):
    project, _ = _make_project(tmp_path)
    session = _Session(project, admin=False, user="bo", agent_model=_fake_model())
    view = DataPage(session, project, admin=False, navigate=lambda _url: None)

    page.set_viewport_size({"width": 1440, "height": 900})
    serve_component(page, view)

    orders_row = page.locator(".flowdash-admin-row-shop--orders")
    expect(orders_row).to_contain_text("3 columns", timeout=15000)
    orders_row.get_by_text("orders").click()
    drawer = page.locator(".flowdash-admin-catalog-drawer")
    expect(drawer).to_contain_text("Exact", timeout=15000)
    expect(drawer.locator(".flowdash-catalog-refresh")).to_be_hidden()
    expect(page.locator(".lumen-data-source-refresh")).to_have_count(0)
    expect(page.locator(".flowdash-catalog-row-refresh")).to_have_count(0)
    expect(drawer.locator(".flowdash-catalog-enrich")).to_be_hidden()
    expect(drawer.locator(".flowdash-catalog-edit-descriptions")).to_be_hidden()
