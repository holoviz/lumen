"""Tests for inheriting metrics from a dbt Semantic Layer into FlowDash."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash, MemoryDashboardStore
from panel_flowdash.agent import DashboardOps
from panel_flowdash.auth import Identity
from panel_flowdash.secret_store import KEY_ENV, encrypt_secret, fernet_key

from lumen.agent.data import metric_line
from lumen.flowdash import (
    COMPONENTS, METRIC_ID, METRICS_KEY, SOURCES_KEY, bind_source, dbt_metrics,
    fetch, metric_pipeline,
)
from lumen.flowdash.data import metric_statuses, metric_tables
from lumen.flowdash.data_section import DataPage
from lumen.flowdash.dbt_metrics import (
    DBT_CLIENT_KEY, DBT_ORIGIN, DbtConfig, _make_client, clear_dbt_cache,
)
from lumen.flowdash.metrics import (
    ProjectMetrics, Query, aggregate, bind_sql, read_metrics,
)
from lumen.flowdash.metrics_edit import (
    check_metric, delete_metric, raw_section, save_metric,
)
from lumen.flowdash.project import (
    PROJECT_FILE, ProjectError, load_project, parse_project,
)
from lumen.flowdash.uploads import UPLOADS_KEY, Uploads
from lumen.sources.duckdb import DuckDBSource

from .dbt_fakes import (
    MIRRORS, FakeSemanticLayer, dimension, measure, metric, semantic_sql,
)

ORDERS_SQL = (
    "SELECT * FROM (VALUES "
    "(1, 'North', 'paid', DATE '2025-01-15', 100.0, 20.0, TRUE), "
    "(2, 'North', 'paid', DATE '2025-02-10', 200.0, 50.0, TRUE), "
    "(3, 'South', 'open', DATE '2025-02-20', 50.0, 10.0, FALSE)"
    ") AS t(order_id, region, status, order_date, amount, cost, is_paid)"
)

ORDER_DIMS = {"order__region": "region", "order__status": "status"}


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_dbt_cache()
    yield
    clear_dbt_cache()


@pytest.fixture
def warehouse():
    return DuckDBSource(uri=":memory:", tables={"orders": ORDERS_SQL}, name="warehouse")


def _dims():
    return [
        dimension("order__region", description="Sales region"),
        dimension("order__status"),
        dimension("order__order_date", "TIME", grains=["MONTH", "YEAR"]),
        dimension("customer__region"),
    ]


def _revenue(name="revenue", agg="SUM", expr="amount"):
    return metric(
        name, measures=[measure("order_amount", agg, expr, agg_time_dimension="order_date")],
        dimensions=_dims(), description="Total order revenue",
    )


def _project(warehouse, client, extra="", tables="    tables:\n      revenue: orders\n"):
    text = f"metrics:\n  dbt:\n    source: warehouse\n    environment_id: 12345\n{tables}{extra}"
    return parse_project(text, PROJECT_FILE, {SOURCES_KEY: {"warehouse": warehouse}, DBT_CLIENT_KEY: client})


def test_fakes_have_only_fields_of_the_real_dbtsl_models():
    models = pytest.importorskip("dbtsl.models")
    import dataclasses

    for name, mirror in MIRRORS.items():
        real = {f.name for f in dataclasses.fields(getattr(models, name))}
        assert {f.name for f in dataclasses.fields(mirror)} <= real, name


# ---------------------------------------------------------------------------
# Config validation & secret handling


@pytest.mark.parametrize(
    "text, expected",
    [
        ("metrics:\n  dbt:\n    environment_id: 123\n", "dbt needs a `source`"),
        ("metrics:\n  dbt:\n    source: warehouse\n", "positive integer `environment_id`"),
        ("metrics:\n  dbt:\n    source: warehouse\n    environment_id: true\n", "positive integer `environment_id`"),
        ("metrics:\n  dbt:\n    source: warehouse\n    environment_id: 123\n    hst: example.com\n", "Did you mean 'host'?"),
        ("metrics:\n  dbt:\n    source: warehouse\n    environment_id: 123\n    auth_token: raw-plaintext-secret\n", "never written in the file"),
        ("metrics:\n  dbt:\n    source: warehouse\n    environment_id: 123\n    metrics: []\n", "non-empty list of metric names"),
        ("metrics:\n  dbt:\n    source: warehouse\n    environment_id: 123\n    table: orders\n", "table"),
    ],
)
def test_dbt_config_validation(text, expected):
    with pytest.raises(ProjectError) as info:
        read_metrics(parse_project(text, PROJECT_FILE))
    assert any(expected in str(p) for p in info.value.problems), [str(p) for p in info.value.problems]


def test_dbt_secret_store_env_and_encrypted(monkeypatch):
    captured = {}

    class StubClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_dbtsl = types.ModuleType("dbtsl")
    fake_asyncio = types.ModuleType("dbtsl.asyncio")
    fake_asyncio.AsyncSemanticLayerClient = StubClient
    monkeypatch.setitem(sys.modules, "dbtsl", fake_dbtsl)
    monkeypatch.setitem(sys.modules, "dbtsl.asyncio", fake_asyncio)

    monkeypatch.setenv("DBT_SVC_TOKEN", "super-secret-env-token")
    _make_client(DbtConfig(source="warehouse", environment_id=42, auth_token="${DBT_SVC_TOKEN}"))
    assert captured == {
        "environment_id": 42,
        "auth_token": "super-secret-env-token",
        "host": "semantic-layer.cloud.getdbt.com",
    }

    pytest.importorskip("cryptography")
    monkeypatch.setenv(KEY_ENV, fernet_key())
    encrypted = encrypt_secret("super-secret-fernet-token")
    captured.clear()
    _make_client(DbtConfig(source="warehouse", environment_id=99, host="custom.getdbt.com", auth_token=encrypted))
    assert captured == {
        "environment_id": 99,
        "auth_token": "super-secret-fernet-token",
        "host": "custom.getdbt.com",
    }


# ---------------------------------------------------------------------------
# Which metrics are translated locally


def test_mapped_simple_metric_is_local_with_its_own_dimensions(warehouse):
    client = FakeSemanticLayer(
        [_revenue()], {"revenue": semantic_sql("orders", "SUM(amount) AS revenue", ORDER_DIMS)},
    )
    project = _project(warehouse, client)
    metrics = read_metrics(project)

    assert project.warnings == []
    rev = metrics["revenue"]
    assert rev.origin == DBT_ORIGIN and rev.read_only
    assert rev.form == "structured" and rev.table == "orders"
    assert rev.measures["revenue"].expr == "SUM(amount)"
    # Dimensions keep dbt's qualified names; one joining another semantic model is left out
    # even though the orders table has a same-named column.
    assert set(rev.dimensions) == {"order__region", "order__status", "metric_time"}
    assert rev.dimensions["order__region"].column == "region"
    mt = rev.dimensions["metric_time"]
    assert mt.time and mt.column == "order_date" and mt.grains == ("year", "month")
    assert rev.parameters == {}

    agg = aggregate(rev, Query(measures=["revenue"], dimensions=["order__region"]), {}, "duckdb")
    assert agg.measures and agg.dimensions
    df = fetch(metric_pipeline(rev, warehouse, Query(measures=["revenue"], dimensions=["order__region"], order_by=["-revenue"])))
    assert df.to_dict("records") == [
        {"order__region": "North", "revenue": 300.0},
        {"order__region": "South", "revenue": 50.0},
    ]
    assert "revenue (source warehouse, table orders)" in metric_line(rev)


def test_filtered_simple_metric_without_a_filter_field_uses_compile_sql(warehouse):
    # The dbtsl models say nothing about filters; dbt's compiled SQL does.
    client = FakeSemanticLayer(
        [_revenue("paid_revenue")],
        {"paid_revenue": semantic_sql("orders", "SUM(amount) AS paid_revenue", ORDER_DIMS, where="is_paid")},
    )
    metrics = read_metrics(_project(warehouse, client, tables="    tables:\n      paid_revenue: orders\n"))
    paid = metrics["paid_revenue"]
    assert paid.form == "sql" and "WHERE is_paid" in paid.sql and paid.dimensions == {}
    df = fetch(metric_pipeline(paid, warehouse, Query()))
    assert df["paid_revenue"].iloc[0] == pytest.approx(300.0)


def test_unmapped_simple_metric_uses_compile_sql(warehouse):
    client = FakeSemanticLayer([_revenue()], {"revenue": semantic_sql("orders", "SUM(amount) AS revenue", ORDER_DIMS)})
    metrics = read_metrics(_project(warehouse, client, tables=""))
    assert metrics["revenue"].form == "sql"
    assert [c["metrics"] for c in client.compile_calls] == [["revenue"]]


def test_mapped_metric_compiled_over_another_table_uses_compile_sql(warehouse):
    client = FakeSemanticLayer([_revenue()], {"revenue": "SELECT SUM(amount) AS revenue FROM analytics.invoices"})
    metrics = read_metrics(_project(warehouse, client))
    # Kept as compiled SQL, which then fails its check against the source rather than reading orders.
    assert "revenue" not in metrics and metrics.failures["revenue"].metric.form == "sql"


def test_qualified_table_in_compiled_sql_matches_the_mapping(warehouse):
    client = FakeSemanticLayer([_revenue()], {"revenue": "SELECT SUM(amount) AS revenue FROM analytics.prod.orders AS src"})
    assert read_metrics(_project(warehouse, client))["revenue"].form == "structured"


@pytest.mark.parametrize("agg", ["MEDIAN", "PERCENTILE"])
def test_non_portable_aggregations_use_compile_sql(warehouse, agg):
    client = FakeSemanticLayer([_revenue(agg=agg)], {"revenue": semantic_sql("orders", "MEDIAN(amount) AS revenue")})
    assert read_metrics(_project(warehouse, client))["revenue"].form == "sql"


def test_ratio_and_derived_metrics_use_compile_sql_even_when_mapped(warehouse):
    # A ratio whose name matches a measure, and measures listed in an order that is no
    # numerator/denominator contract: neither is translated locally.
    ratio = metric(
        "order_amount", "RATIO", measures=[measure("orders", "COUNT", "1"), measure("order_amount", "SUM", "amount")],
    )
    derived = metric("gross_profit", "DERIVED", measures=[measure("order_amount", "SUM", "amount")])
    client = FakeSemanticLayer(
        [ratio, derived],
        {
            "order_amount": "SELECT SUM(amount) / NULLIF(COUNT(1), 0) AS order_amount FROM orders",
            "gross_profit": "SELECT SUM(amount) - SUM(cost) AS gross_profit FROM orders",
        },
    )
    metrics = read_metrics(_project(warehouse, client, tables="    tables:\n      order_amount: orders\n      gross_profit: orders\n"))
    assert metrics["order_amount"].form == "sql"
    assert metrics["gross_profit"].form == "sql"
    assert fetch(metric_pipeline(metrics["order_amount"], warehouse, Query()))["order_amount"].iloc[0] == pytest.approx(350.0 / 3)


def test_metric_requiring_metric_time_keeps_it_or_falls_back(warehouse):
    rolling = metric(
        "rolling_revenue", "CUMULATIVE", requires_metric_time=True,
        measures=[measure("order_amount", "SUM", "amount", agg_time_dimension="order_date")],
    )
    client = FakeSemanticLayer(
        [rolling], {"rolling_revenue": "SELECT order_date AS metric_time, SUM(amount) AS rolling_revenue FROM orders GROUP BY order_date"},
    )
    metrics = read_metrics(_project(warehouse, client, tables="    tables:\n      rolling_revenue: orders\n"))
    rolling = metrics["rolling_revenue"]
    assert rolling.form == "sql"
    assert client.compile_calls == [{"metrics": ["rolling_revenue"], "group_by": ["metric_time"]}]
    assert "rolling_revenue" in bind_sql(rolling, {}, "duckdb")


def test_sql_and_structured_dbt_metrics_run_as_metric_nodes(warehouse):
    rolling = metric("rolling_revenue", "CUMULATIVE", requires_metric_time=True)
    client = FakeSemanticLayer(
        [_revenue(), rolling],
        {
            "revenue": semantic_sql("orders", "SUM(amount) AS revenue", ORDER_DIMS),
            "rolling_revenue": "SELECT order_date AS metric_time, SUM(amount) AS rolling_revenue FROM orders GROUP BY order_date",
        },
    )
    metrics = read_metrics(_project(warehouse, client))
    editor = FlowDash(components=COMPONENTS)
    try:
        bind_source(editor, warehouse, "warehouse")
        editor.context[METRICS_KEY] = metrics
        ops = DashboardOps(editor)
        ops.add_node(METRIC_ID, {"metric": "revenue", "measures": ["revenue"], "dimensions": ["order__status"]}, name="rev")
        ops.add_node(METRIC_ID, {"metric": "rolling_revenue"}, name="roll")
        assert set(ops.node_output("rev.data")["order__status"]) == {"paid", "open"}
        assert len(ops.node_output("roll.data")) == 3
    finally:
        editor.dispose()


def test_runs_safely_inside_active_event_loop(warehouse):
    client = FakeSemanticLayer([_revenue()], {"revenue": semantic_sql("orders", "SUM(amount) AS revenue")})

    async def inside_loop():
        return read_metrics(_project(warehouse, client))

    assert list(asyncio.run(inside_loop())) == ["revenue"]


# ---------------------------------------------------------------------------
# Caching, timeout and refresh


def test_fetch_is_cached_and_refetched_once_expired(warehouse, monkeypatch):
    client = FakeSemanticLayer([_revenue()], {"revenue": semantic_sql("orders", "SUM(amount) AS revenue", ORDER_DIMS)})
    read_metrics(_project(warehouse, client))
    calls = len(client.compile_calls)
    assert client.metric_calls == 1

    read_metrics(_project(warehouse, client))
    assert client.metric_calls == 1 and len(client.compile_calls) == calls

    monkeypatch.setattr(dbt_metrics, "CACHE_TTL", 0)
    read_metrics(_project(warehouse, client))
    assert client.metric_calls == 2


def test_slow_dbt_times_out_without_holding_local_metrics(warehouse, monkeypatch):
    monkeypatch.setattr(dbt_metrics, "TIMEOUT", 0.05)
    client = FakeSemanticLayer([_revenue()], {"revenue": "SELECT 1"}, delay=1)
    local = "  local_orders:\n    source: warehouse\n    table: orders\n    measures:\n      n: COUNT(*)\n"
    project = _project(warehouse, client, extra=local)
    metrics = read_metrics(project)
    assert list(metrics) == ["local_orders"]
    assert "no answer within 0.05 seconds" in metrics.dbt_error


def test_project_metrics_refetch_after_the_cache_expires(warehouse, tmp_path, monkeypatch):
    revenue = _revenue()
    client = FakeSemanticLayer([revenue], {"revenue": semantic_sql("orders", "SUM(amount) AS revenue"), "order_count": "SELECT COUNT(*) AS order_count FROM orders"})
    path = tmp_path / PROJECT_FILE
    path.write_text("metrics:\n  dbt:\n    source: warehouse\n    environment_id: 10\n")
    project = load_project(path, {SOURCES_KEY: {"warehouse": warehouse}, DBT_CLIENT_KEY: client})
    pm = ProjectMetrics(project)
    assert list(pm) == ["revenue"]

    client._metrics = [revenue, metric("order_count")]
    assert list(pm) == ["revenue"]
    monkeypatch.setattr(dbt_metrics, "CACHE_TTL", 0)
    assert list(pm) == ["revenue", "order_count"]


def test_metric_statuses_without_project_metrics_never_asks_dbt(warehouse, tmp_path):
    client = FakeSemanticLayer([_revenue()], {"revenue": semantic_sql("orders", "SUM(amount) AS revenue")})
    path = tmp_path / PROJECT_FILE
    path.write_text("metrics:\n  dbt:\n    source: warehouse\n    environment_id: 10\n")
    project = load_project(path, {SOURCES_KEY: {"warehouse": warehouse}, DBT_CLIENT_KEY: client})
    clear_dbt_cache()
    calls = client.metric_calls

    statuses, _ = metric_statuses(project)
    assert statuses == [] and client.metric_calls == calls

    load_project(path, project.context)
    statuses, _ = metric_statuses(project)
    assert [(s.name, s.origin) for s in statuses] == [("revenue", DBT_ORIGIN)]


# ---------------------------------------------------------------------------
# Coexistence with YAML metrics, read-only enforcement, failure isolation & UI


class _Admin:
    _identity = Identity("ana")

    def __init__(self, admin=True):
        self.admin = admin
        self.is_admin = admin
        self.store = MemoryDashboardStore()

    def _still_admin(self):
        return self.admin

    def _notify(self, severity, message, duration=3000):
        pass


def test_coexistence_readonly_guards_and_yaml_preservation(warehouse, tmp_path):
    client = FakeSemanticLayer(
        [_revenue(), _revenue("manual_metric")],
        {"revenue": "SELECT SUM(amount) AS revenue FROM orders", "manual_metric": "SELECT SUM(amount) AS manual_metric FROM orders"},
    )
    path = tmp_path / PROJECT_FILE
    path.write_text(
        """\
metrics:
  dbt:
    source: warehouse
    environment_id: 10
  manual_metric:
    description: Hand-authored metric
    source: warehouse
    table: orders
    measures:
      total: SUM(amount)
"""
    )
    project = load_project(path, {SOURCES_KEY: {"warehouse": warehouse}, DBT_CLIENT_KEY: client})
    project.context[METRICS_KEY] = pm = ProjectMetrics(project)

    assert list(pm) == ["manual_metric", "revenue"]
    assert pm["manual_metric"].origin is None
    assert pm["revenue"].origin == DBT_ORIGIN
    assert any("already defined in metrics" in str(w) for w in project.warnings)

    # Reserved name 'dbt' and read-only dbt metrics cannot be saved or deleted.
    assert any("reserved" in p.message for p in check_metric(project, "dbt", {"source": "warehouse", "table": "orders", "measures": {"x": "COUNT(*)"}}))
    assert any("read-only" in p.message for p in check_metric(project, "revenue", {"source": "warehouse", "table": "orders", "measures": {"x": "COUNT(*)"}}))
    assert any("read-only" in p.message for p in delete_metric(project, "revenue", app=_Admin()))
    assert any("not a metric" in p.message for p in delete_metric(project, "dbt", app=_Admin()))

    # Saving a new manual metric preserves the dbt: block in flowdash.yaml.
    problems = save_metric(
        project,
        "extra_metric",
        {"source": "warehouse", "table": "orders", "measures": {"n": "COUNT(*)"}},
        app=_Admin(),
    )
    assert problems == []
    saved_raw = raw_section(project)
    assert "dbt" in saved_raw and saved_raw["dbt"]["environment_id"] == 10
    assert "extra_metric" in saved_raw
    assert metric_tables(saved_raw["dbt"], project.context) == []


def test_dbt_connection_failure_isolated_and_retried(warehouse, tmp_path):
    broken = True

    def factory(_config):
        if broken:
            raise RuntimeError("cannot reach https://user:secretpass@dbt.example.com")
        return FakeSemanticLayer([_revenue()], {"revenue": "SELECT SUM(amount) AS revenue FROM orders"})

    path = tmp_path / PROJECT_FILE
    path.write_text(
        """\
metrics:
  dbt:
    source: warehouse
    environment_id: 10
  local_orders:
    source: warehouse
    table: orders
    measures:
      n: COUNT(*)
"""
    )
    project = load_project(path, {SOURCES_KEY: {"warehouse": warehouse}, DBT_CLIENT_KEY: factory})
    project.context[METRICS_KEY] = pm = ProjectMetrics(project, retry=0)

    # Local YAML metric keeps working; dbt error is redacted and surfaced.
    assert list(pm) == ["local_orders"]
    assert pm.dbt_error is not None and "secretpass" not in pm.dbt_error
    statuses, general = metric_statuses(project)
    assert [s.name for s in statuses] == ["local_orders"]
    assert any("dbt Semantic Layer" in msg for msg in general)

    # Once dbt recovers, ProjectMetrics retries and picks up dbt metrics.
    broken = False
    assert list(pm) == ["local_orders", "revenue"]
    assert pm.dbt_error is None


def test_metrics_page_marks_dbt_metrics_and_hides_edit_controls(warehouse, tmp_path):
    client = FakeSemanticLayer(
        [
            metric("revenue", measures=[measure("amount", "SUM", "amount")], dimensions=[dimension("order__region")],
                   description="Revenue from dbt"),
            metric("rolling_revenue", "CUMULATIVE", description="Rolling revenue"),
        ],
        {
            "revenue": semantic_sql("orders", "SUM(amount) AS revenue", ORDER_DIMS),
            "rolling_revenue": "SELECT region, SUM(amount) AS rolling_revenue FROM orders GROUP BY region",
        },
    )
    path = tmp_path / PROJECT_FILE
    path.write_text(
        """\
metrics:
  dbt:
    source: warehouse
    environment_id: 10
    tables:
      revenue: orders
  local_orders:
    description: Local orders
    source: warehouse
    table: orders
    measures:
      n: COUNT(*)
"""
    )
    sources = {"warehouse": warehouse}
    project = load_project(path, {SOURCES_KEY: sources, DBT_CLIENT_KEY: client})
    project.context[UPLOADS_KEY] = Uploads(tmp_path / "uploads", sources, project=project)
    project.context[METRICS_KEY] = ProjectMetrics(project)

    admin = _Admin(admin=True)
    # 1. Metrics list page
    list_page = DataPage(admin, project, "metrics", admin=True, navigate=lambda _url: None)
    ms = list_page.metrics
    assert [s.name for s in ms.statuses()] == ["local_orders", "revenue", "rolling_revenue"]
    assert "2 from dbt" in ms._count.object
    assert "origin:dbt" in ms._chips
    # Only the local YAML metric gets an Edit/Duplicate/Delete menu.
    assert set(ms._menus) == {"local_orders"}

    # Filtering by origin:dbt shows only dbt metrics.
    ms.filter = "origin:dbt"
    assert [s.name for s in ms.statuses() if ms._passes(s)] == ["revenue", "rolling_revenue"]

    # 2. Individual dbt metric page (structured)
    dbt_page = DataPage(admin, project, "metrics/revenue", admin=True, navigate=lambda _url: None)
    text = "".join(str(o.object) for o in dbt_page.__panel__().select() if isinstance(getattr(o, "object", None), str))
    assert "From dbt" in text and "Read-only" in text
    assert "dbt Semantic Layer (read-only)" in text
    assert not any("lumen-metric-edit" in getattr(o, "css_classes", []) for o in dbt_page.__panel__().select())
    assert not any("lumen-metric-menu" in getattr(o, "css_classes", []) for o in dbt_page.__panel__().select())

    # Try it still runs the dbt metric cleanly.
    res = dbt_page.metric.run()
    assert res.error is None and res.rows == [["North", 300.0], ["South", 50.0]]

    # 3. Individual dbt metric page (compiled SQL fallback)
    sql_page = DataPage(admin, project, "metrics/rolling_revenue", admin=True, navigate=lambda _url: None)
    sql_text = "".join(str(o.object) for o in sql_page.__panel__().select() if isinstance(getattr(o, "object", None), str))
    assert "compiled SQL" in sql_text
    sql_res = sql_page.metric.run()
    assert sql_res.error is None and len(sql_res.rows) == 2

    # 4. The drawer shows a dbt metric read-only, and a local one editable again.
    ms.open_metric("revenue")
    assert ms._drawer.open and ms.selected is None
    assert ms._editor._yaml.disabled and not ms._save.visible and ms._cancel.label == "Done"
    assert not any(part.visible for part in ms._editor._preview_parts)
    ms.close_metric()
    ms.open_metric("local_orders")
    assert not ms._editor._yaml.disabled and ms._save.visible and ms._cancel.label == "Cancel"
    assert all(part.visible for part in ms._editor._preview_parts)
