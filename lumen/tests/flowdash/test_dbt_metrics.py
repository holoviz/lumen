"""Tests for inheriting metrics from a dbt Semantic Layer into FlowDash."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import sys
import types

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash, MemoryDashboardStore
from panel_flowdash.agent import DashboardOps
from panel_flowdash.auth import Identity
from panel_flowdash.secret_store import KEY_ENV, fernet_key

from lumen.agent.data import metric_line
from lumen.flowdash import (
    COMPONENTS, METRIC_ID, METRICS_KEY, SOURCES_KEY, bind_source, fetch,
    metric_pipeline,
)
from lumen.flowdash.data import metric_statuses, metric_tables
from lumen.flowdash.data_section import DataPage
from lumen.flowdash.dbt_metrics import (
    DBT_CLIENT_KEY, DBT_ORIGIN, DbtConfig, _make_client, encrypt_dbt_token,
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

# ---------------------------------------------------------------------------
# Lightweight stubs mirroring dbtsl.models


class MetricType(enum.Enum):
    SIMPLE = "SIMPLE"
    RATIO = "RATIO"
    DERIVED = "DERIVED"
    CUMULATIVE = "CUMULATIVE"
    CONVERSION = "CONVERSION"


class DimensionType(enum.Enum):
    CATEGORICAL = "CATEGORICAL"
    TIME = "TIME"


class TimeGranularity(enum.Enum):
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    QUARTER = "QUARTER"
    YEAR = "YEAR"
    MINUTE = "MINUTE"


@dataclass
class FakeMeasure:
    name: str
    agg: str
    expr: str | None = None
    agg_time_dimension: str | None = None
    semantic_model: str | None = None
    filter: Any = None


@dataclass
class FakeDimension:
    name: str
    type: DimensionType = DimensionType.CATEGORICAL
    expr: str | None = None
    description: str = ""
    queryable_granularities: list[TimeGranularity] = field(default_factory=list)


@dataclass
class FakeTypeParams:
    measure: Any = None
    numerator: Any = None
    denominator: Any = None
    expr: str | None = None
    metrics: list[Any] = field(default_factory=list)
    filter: Any = None


@dataclass
class FakeInputRef:
    name: str
    alias: str | None = None
    filter: Any = None
    offset_window: str | None = None
    offset_to_grain: str | None = None


@dataclass
class FakeMetric:
    name: str
    type: MetricType = MetricType.SIMPLE
    description: str = ""
    measures: list[FakeMeasure] = field(default_factory=list)
    dimensions: list[FakeDimension] = field(default_factory=list)
    queryable_granularities: list[TimeGranularity] = field(default_factory=list)
    type_params: FakeTypeParams | None = None
    filter: Any = None
    requires_metric_time: bool = False
    parameters: dict[str, Any] | None = None


class FakeAsyncClient:
    """Async dbt Semantic Layer client stub matching ``AsyncSemanticLayerClient``."""

    def __init__(self, metrics: list[FakeMetric], compiled: dict[str, str] | None = None, fail_compile: set[str] | None = None):
        self._metrics = metrics
        self._compiled = compiled or {}
        self._fail_compile = fail_compile or set()
        self.compile_calls: list[dict[str, Any]] = []
        self.sessions = 0

    @contextlib.asynccontextmanager
    async def session(self):
        self.sessions += 1
        yield self

    async def metrics(self):
        return list(self._metrics)

    async def compile_sql(self, **kwargs):
        self.compile_calls.append(kwargs)
        name = kwargs["metrics"][0]
        if name in self._fail_compile:
            raise RuntimeError(f"compile_sql failed for {name}")
        return self._compiled[name]


ORDERS_SQL = (
    "SELECT * FROM (VALUES "
    "(1, 'North', 'paid', DATE '2025-01-15', 100.0, 20.0, TRUE), "
    "(2, 'North', 'paid', DATE '2025-02-10', 200.0, 50.0, TRUE), "
    "(3, 'South', 'open', DATE '2025-02-20', 50.0, 10.0, FALSE)"
    ") AS t(order_id, region, status, order_date, amount, cost, is_paid)"
)


@pytest.fixture
def warehouse():
    return DuckDBSource(uri=":memory:", tables={"orders": ORDERS_SQL}, name="warehouse")


def _sample_dbt_metrics() -> list[FakeMetric]:
    dims = [
        FakeDimension("customer__region", DimensionType.CATEGORICAL, description="Sales region"),
        FakeDimension("status", DimensionType.CATEGORICAL),
        FakeDimension("metric_time", DimensionType.TIME, queryable_granularities=[TimeGranularity.MONTH, TimeGranularity.YEAR, TimeGranularity.MINUTE]),
        FakeDimension("other_model__ghost", DimensionType.CATEGORICAL),
    ]
    return [
        FakeMetric(
            name="revenue",
            type=MetricType.SIMPLE,
            description="Total order revenue",
            measures=[FakeMeasure("order_amount", "SUM", expr="amount", agg_time_dimension="order_date", semantic_model="orders")],
            dimensions=dims,
            parameters={"min_order": {"type": "integer", "column": "order_id", "op": ">=", "default": 1}},
        ),
        FakeMetric(
            name="order_count",
            type=MetricType.SIMPLE,
            description="Number of orders",
            measures=[FakeMeasure("orders", "COUNT", expr="1", agg_time_dimension="order_date", semantic_model="orders")],
            dimensions=dims,
        ),
        FakeMetric(
            name="total_cost",
            type=MetricType.SIMPLE,
            description="Total order cost",
            measures=[FakeMeasure("order_cost", "SUM", expr="cost", agg_time_dimension="order_date", semantic_model="orders")],
            dimensions=dims,
        ),
        FakeMetric(
            name="avg_order_value",
            type=MetricType.RATIO,
            description="Revenue per order",
            type_params=FakeTypeParams(
                numerator=FakeInputRef("revenue"),
                denominator=FakeInputRef("order_count"),
            ),
            dimensions=dims,
        ),
        FakeMetric(
            name="gross_profit",
            type=MetricType.DERIVED,
            description="Revenue minus cost",
            type_params=FakeTypeParams(
                expr="rev - cost",
                metrics=[FakeInputRef("revenue", alias="rev"), FakeInputRef("total_cost", alias="cost")],
            ),
            dimensions=dims,
        ),
        FakeMetric(
            name="rolling_revenue",
            type=MetricType.CUMULATIVE,
            description="Cumulative rolling revenue",
            requires_metric_time=True,
            measures=[FakeMeasure("order_amount", "SUM", expr="amount")],
        ),
    ]


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

    # 1. ${ENV_VAR} reference resolves lazily without leaking.
    monkeypatch.setenv("DBT_SVC_TOKEN", "super-secret-env-token")
    cfg = DbtConfig(source="warehouse", environment_id=42, auth_token="${DBT_SVC_TOKEN}")
    _make_client(cfg)
    assert captured == {
        "environment_id": 42,
        "auth_token": "super-secret-env-token",
        "host": "semantic-layer.cloud.getdbt.com",
    }

    # 2. Encrypted secret round-trips through panel_flowdash.secret_store.
    pytest.importorskip("cryptography")
    monkeypatch.setenv(KEY_ENV, fernet_key())
    encrypted = encrypt_dbt_token("super-secret-fernet-token")
    assert set(encrypted) == {"encrypted", "key_id"}
    assert "super-secret-fernet-token" not in str(encrypted)

    captured.clear()
    cfg_enc = DbtConfig(source="warehouse", environment_id=99, host="custom.getdbt.com", auth_token=encrypted)
    _make_client(cfg_enc)
    assert captured == {
        "environment_id": 99,
        "auth_token": "super-secret-fernet-token",
        "host": "custom.getdbt.com",
    }


def test_encrypt_dbt_token_refuses_without_cryptography(monkeypatch):
    from panel_flowdash import secret_store

    monkeypatch.setattr(secret_store, "encryption_available", lambda: False)
    with pytest.raises(ValueError, match="panel-flowdash\\[secrets\\]"):
        encrypt_dbt_token("secret-token")


# ---------------------------------------------------------------------------
# Conversion, compile_sql fallback, and MetricNode execution


def test_local_conversion_and_selective_compile_sql_fallback(warehouse):
    client = FakeAsyncClient(
        _sample_dbt_metrics(),
        compiled={
            "rolling_revenue": "SELECT order_date AS metric_time, SUM(amount) AS rolling_revenue FROM orders GROUP BY order_date",
        },
    )
    yaml_text = """\
metrics:
  dbt:
    source: warehouse
    environment_id: 12345
    auth_token: ${DBT_AUTH_TOKEN}
"""
    context = {SOURCES_KEY: {"warehouse": warehouse}, DBT_CLIENT_KEY: client}
    project = parse_project(yaml_text, PROJECT_FILE, context)
    metrics = read_metrics(project)

    assert project.warnings == []
    assert set(metrics) == {"revenue", "order_count", "total_cost", "avg_order_value", "gross_profit", "rolling_revenue"}
    assert all(m.origin == DBT_ORIGIN and m.read_only for m in metrics.values())

    # Local structured metrics did NOT trigger compile_sql.
    assert [c["metrics"] for c in client.compile_calls] == [["rolling_revenue"]]
    assert client.compile_calls[0]["group_by"] == ["metric_time"]

    # Check SIMPLE metric shape.
    rev = metrics["revenue"]
    assert rev.form == "structured" and rev.table == "orders"
    assert rev.measures["revenue"].expr == "SUM(amount)"
    # Entity-prefixed customer__region mapped to column 'region'; cross-table ghost skipped.
    assert "customer__region" in rev.dimensions and rev.dimensions["customer__region"].column == "region"
    assert "other_model__ghost" not in rev.dimensions
    # metric_time mapped to agg_time_dimension 'order_date' and grains intersected with GRAINS.
    mt = rev.dimensions["metric_time"]
    assert mt.time is True and mt.column == "order_date" and mt.grains == ("year", "month")
    assert "min_order" in rev.parameters and rev.parameters["min_order"].default == 1

    # Check RATIO and DERIVED metrics converted locally into structured metrics.
    assert metrics["avg_order_value"].form == "structured"
    assert metrics["gross_profit"].form == "structured"
    assert "NULLIF" in metrics["avg_order_value"].measures["avg_order_value"].expr

    # Check CUMULATIVE metric fell back to SQL form.
    rolling = metrics["rolling_revenue"]
    assert rolling.form == "sql" and "SUM(amount)" in rolling.sql

    # Verify unchanged aggregate(), bind_sql(), metric_pipeline(), MetricNode, and agent metric_line().
    agg = aggregate(rev, Query(measures=["revenue"], dimensions=["customer__region"]), {"min_order": 1}, "duckdb")
    assert agg.measures and agg.dimensions
    sql = bind_sql(rolling, {}, "duckdb")
    assert "rolling_revenue" in sql

    df_rev = fetch(metric_pipeline(rev, warehouse, Query(measures=["revenue"], dimensions=["customer__region"], order_by=["-revenue"])))
    assert df_rev.to_dict("records") == [
        {"customer__region": "North", "revenue": 300.0},
        {"customer__region": "South", "revenue": 50.0},
    ]

    df_profit = fetch(metric_pipeline(metrics["gross_profit"], warehouse, Query(measures=["gross_profit"])))
    assert df_profit["gross_profit"].iloc[0] == pytest.approx(270.0)

    df_ratio = fetch(metric_pipeline(metrics["avg_order_value"], warehouse, Query(measures=["avg_order_value"])))
    assert df_ratio["avg_order_value"].iloc[0] == pytest.approx(350.0 / 3.0)

    # MetricNode executes both structured and fallback SQL dbt metrics unchanged.
    editor = FlowDash(components=COMPONENTS)
    try:
        bind_source(editor, warehouse, "warehouse")
        editor.context[METRICS_KEY] = metrics
        ops = DashboardOps(editor)
        ops.add_node(METRIC_ID, {"metric": "revenue", "measures": ["revenue"], "dimensions": ["status"]}, name="rev")
        ops.add_node(METRIC_ID, {"metric": "rolling_revenue"}, name="roll")
        out_rev = ops.node_output("rev.data")
        assert set(out_rev["status"]) == {"paid", "open"}
        out_roll = ops.node_output("roll.data")
        assert len(out_roll) == 3
    finally:
        editor.dispose()

    # Agent listing works unchanged.
    line = metric_line(rev)
    assert "revenue (source warehouse, table orders)" in line
    assert "customer__region" in line


def test_filtered_and_offset_and_boolean_aggs(warehouse):
    client = FakeAsyncClient(
        [
            FakeMetric(
                name="paid_orders",
                type=MetricType.SIMPLE,
                measures=[FakeMeasure("is_paid", "SUM_BOOLEAN", expr="is_paid", semantic_model="orders")],
            ),
            FakeMetric(
                name="filtered_rev",
                type=MetricType.SIMPLE,
                filter="{{ Dimension('status') }} = 'paid'",
                measures=[FakeMeasure("amount", "SUM", expr="amount", semantic_model="orders")],
            ),
            FakeMetric(
                name="prev_month_rev",
                type=MetricType.DERIVED,
                type_params=FakeTypeParams(
                    expr="a",
                    metrics=[FakeInputRef("paid_orders", alias="a", offset_window="1 month")],
                ),
            ),
        ],
        compiled={
            "filtered_rev": "SELECT SUM(amount) AS filtered_rev FROM orders WHERE status = 'paid'",
            "prev_month_rev": "SELECT SUM(amount) AS prev_month_rev FROM orders",
        },
    )
    yaml_text = """\
metrics:
  dbt:
    source: warehouse
    environment_id: 7
"""
    project = parse_project(yaml_text, PROJECT_FILE, {SOURCES_KEY: {"warehouse": warehouse}, DBT_CLIENT_KEY: client})
    metrics = read_metrics(project)
    assert metrics["paid_orders"].form == "structured"
    df = fetch(metric_pipeline(metrics["paid_orders"], warehouse, Query(measures=["paid_orders"])))
    assert df["paid_orders"].iloc[0] == 2
    assert metrics["filtered_rev"].form == "sql"
    assert metrics["prev_month_rev"].form == "sql"
    assert [c["metrics"][0] for c in client.compile_calls] == ["filtered_rev", "prev_month_rev"]


def test_runs_safely_inside_active_event_loop(warehouse):
    client = FakeAsyncClient([FakeMetric(name="revenue", measures=[FakeMeasure("amount", "SUM", expr="amount", semantic_model="orders")])])
    yaml_text = "metrics:\n  dbt:\n    source: warehouse\n    environment_id: 5\n"

    async def inside_loop():
        project = parse_project(yaml_text, PROJECT_FILE, {SOURCES_KEY: {"warehouse": warehouse}, DBT_CLIENT_KEY: client})
        return read_metrics(project)

    metrics = asyncio.run(inside_loop())
    assert list(metrics) == ["revenue"]


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
    client = FakeAsyncClient(
        [
            FakeMetric(name="revenue", description="From dbt", measures=[FakeMeasure("amount", "SUM", expr="amount", semantic_model="orders")]),
            FakeMetric(name="manual_metric", description="Collides with yaml", measures=[FakeMeasure("amount", "SUM", expr="amount", semantic_model="orders")]),
        ]
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
        return FakeAsyncClient(
            [
                FakeMetric(name="revenue", measures=[FakeMeasure("amount", "SUM", expr="amount", semantic_model="orders")]),
            ]
        )

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
    client = FakeAsyncClient(
        [
            FakeMetric(
                name="revenue",
                description="Revenue from dbt",
                measures=[FakeMeasure("amount", "SUM", expr="amount", semantic_model="orders")],
                dimensions=[FakeDimension("region", DimensionType.CATEGORICAL)],
            ),
            FakeMetric(
                name="rolling_revenue",
                type=MetricType.CUMULATIVE,
                description="Rolling revenue",
            ),
        ],
        compiled={"rolling_revenue": "SELECT region, SUM(amount) AS rolling_revenue FROM orders GROUP BY region"},
    )
    path = tmp_path / PROJECT_FILE
    path.write_text(
        """\
metrics:
  dbt:
    source: warehouse
    environment_id: 10
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
