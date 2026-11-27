"""
Inherit metrics from a dbt Semantic Layer into FlowDash's ``metrics:`` section.

Declared under ``metrics:`` in ``flowdash.yaml``::

    metrics:
      dbt:
        source: warehouse
        environment_id: 12345
        host: semantic-layer.cloud.getdbt.com   # optional
        auth_token: ${DBT_AUTH_TOKEN}           # optional; ${VAR} or encrypted secret
        tables:                                 # optional dbt metric -> table map
          revenue: orders
        metrics: [revenue, order_count]         # optional allowlist of dbt metric names

Each dbt metric becomes a standard ``Metric`` (with ``origin="dbt"``), so
``MetricNode``, ``aggregate()`` and ``bind_sql()`` work unchanged.

The dbt Semantic Layer API does not say whether a metric has a filter or which
semantic model a measure comes from, so by default every metric is the SQL
that ``client.compile_sql`` returns for it, without dimensions or grains. A
metric becomes a structured ``Metric`` (sliceable by dimension and time) only
when all of this holds: it is ``SIMPLE`` with one measure of a plain
aggregation, it is mapped to a table under ``tables:``, and the SQL dbt
compiles for it (and for each dimension kept) reads only that table, without
joins, filters or window functions. Anything unknown means compiled SQL.

Results are cached per configuration for ``CACHE_TTL`` seconds, so saving a
local metric does not refetch from dbt; ``ProjectMetrics`` refetches once
the cache has expired, which is how upstream dbt changes show up. A fetch
gives up after ``TIMEOUT`` seconds.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import sqlglot

from sqlglot import exp

from ..sources.base import BaseSQLSource
from .metrics import (
    _NAME, DBT_KEY, DBT_ORIGIN, GRAINS, Dimension, Measure, Metric,
    MetricFailure, check_expression, expression_columns,
)
from .project import Section

if TYPE_CHECKING:
    from .metrics import _Reader

logger = logging.getLogger(__name__)

# A context key through which callers or tests supply a dbt client, or a factory taking the DbtConfig.
DBT_CLIENT_KEY = "lumen.dbt_client"

DEFAULT_HOST = "semantic-layer.cloud.getdbt.com"

# Seconds a fetch (listing and compiling every metric) may take, and a successful one is reused.
TIMEOUT = 30.0
CACHE_TTL = 300.0

_COMPILE_CONCURRENCY = 8

_DBT_KEYS = {"source", "environment_id", "host", "auth_token", "tables", "metrics"}

# dbt aggregation types with the same meaning as a portable SQL aggregate over one table.
_SIMPLE_AGGS = {
    "SUM": lambda target: f"SUM({target})",
    "COUNT": lambda target: "COUNT(*)" if target in ("", "1", "*") else f"COUNT({target})",
    "COUNT_DISTINCT": lambda target: f"COUNT(DISTINCT {target})",
    "AVERAGE": lambda target: f"AVG({target})",
    "MIN": lambda target: f"MIN({target})",
    "MAX": lambda target: f"MAX({target})",
    "SUM_BOOLEAN": lambda target: f"SUM(CASE WHEN {target} THEN 1 ELSE 0 END)",
}

_NOT_PLAIN = (exp.Join, exp.Where, exp.Having, exp.Qualify, exp.Window, exp.SetOperation)


@dataclass(frozen=True)
class DbtConfig:
    """Connection and mapping settings for inheriting metrics from a dbt Semantic Layer."""

    source: str
    environment_id: int
    host: str = DEFAULT_HOST
    auth_token: Any = None
    tables: tuple[tuple[str, str], ...] = ()
    metrics: tuple[str, ...] | None = None

    @property
    def table_map(self) -> dict[str, str]:
        return dict(self.tables)

    def cache_key(self) -> str:
        return json.dumps(
            [self.source, self.environment_id, self.host, self.auth_token, self.tables, self.metrics],
            sort_keys=True, default=str,
        )


def read_dbt_config(reader: _Reader, spec: Any) -> DbtConfig | None:
    """Validate the ``metrics.dbt`` mapping with ``reader`` and return a ``DbtConfig``."""
    from panel_flowdash.secret_store import check_secret

    path = (DBT_KEY,)
    before = len(reader.problems)
    spec = reader.mapping(spec, path, "dbt", _DBT_KEYS)
    if spec is None:
        return None

    source = reader.text(spec.get("source"), (*path, "source"), "source")
    if not source:
        reader.fail(path, "dbt needs a `source`: the ref of a source the app serves.")

    env_id = spec.get("environment_id")
    if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id <= 0:
        reader.fail((*path, "environment_id"), "dbt needs a positive integer `environment_id`.")

    host = reader.text(spec.get("host"), (*path, "host"), "host")
    if "host" in spec and not host:
        reader.fail((*path, "host"), "dbt `host` must be a non-empty hostname.")
    host = host or DEFAULT_HOST

    auth_token = spec.get("auth_token")
    if (secret_problem := check_secret(auth_token, "auth_token")) is not None:
        reader.fail((*path, "auth_token"), secret_problem)

    tables: dict[str, str] = {}
    if "tables" in spec and spec["tables"] is not None:
        raw_tables = reader.mapping(spec["tables"], (*path, "tables"), "dbt tables")
        if raw_tables is not None:
            for key, value in raw_tables.items():
                if not isinstance(key, str) or not key:
                    reader.fail((*path, "tables", key), "dbt table mapping keys must be dbt metric names.")
                elif not isinstance(value, str) or not value:
                    reader.fail((*path, "tables", key), f"dbt table for {key!r} must be a non-empty table name.")
                else:
                    tables[key] = value

    allowlist: tuple[str, ...] | None = None
    if "metrics" in spec and spec["metrics"] is not None:
        raw_metrics = spec["metrics"]
        if not isinstance(raw_metrics, list) or not raw_metrics or not all(isinstance(m, str) and m for m in raw_metrics):
            reader.fail((*path, "metrics"), "dbt `metrics` must be a non-empty list of metric names.")
        else:
            allowlist = tuple(dict.fromkeys(raw_metrics))

    if len(reader.problems) > before:
        return None
    return DbtConfig(
        source=source,
        environment_id=env_id,
        host=host,
        auth_token=auth_token,
        tables=tuple(tables.items()),
        metrics=allowlist,
    )


def _make_client(config: DbtConfig) -> Any:
    """Instantiate a dbt Semantic Layer client via ``DbtslMixin``, resolving any stored secret."""
    from panel_flowdash.secret_store import SecretError, resolve_secret

    from ..ai.services import DbtslMixin

    token: str | None = None
    if config.auth_token is not None:
        try:
            token = resolve_secret(config.auth_token)
        except SecretError as exc:
            raise ValueError(f"dbt auth_token could not be resolved: {exc}") from None
    service = DbtslMixin(environment_id=config.environment_id, auth_token=token, host=config.host)
    return service._instantiate_client()


def _run_async(coro: Any) -> Any:
    """Run ``coro`` synchronously, even when called from inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _call_maybe_async(func: Any, *args: Any, **kwargs: Any) -> Any:
    result = func(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value).upper()
    if hasattr(value, "name"):
        return str(value.name).upper()
    return str(value).upper()


def _text(obj: Any, *names: str) -> str:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _declares_filter(obj: Any) -> bool:
    """Whether ``obj`` carries filter or type parameter details, which a local translation would drop."""
    return any(getattr(obj, name, None) for name in ("filter", "type_params", "offset_window", "offset_to_grain"))


def _clean_query(sql: Any, what: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError(f"dbt compile_sql returned no SQL for {what}.")
    cleaned = sql.strip().rstrip(";").strip()
    try:
        parsed = sqlglot.parse_one(cleaned)
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError(f"dbt compile_sql returned unparseable SQL for {what}: {exc}") from None
    if not isinstance(parsed, exp.Query):
        raise ValueError(f"dbt compile_sql did not return a SELECT or WITH query for {what}.")
    return cleaned


def _scans_only(sql: str, table: str, dialect: str | None) -> bool:
    """Whether the compiled ``sql`` reads ``table`` alone, without joins, filters or windows."""
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.SqlglotError:
        return False
    if any(parsed.find(kind) for kind in _NOT_PLAIN):
        return False
    ctes = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    read = {t.name.lower() for t in parsed.find_all(exp.Table) if t.name.lower() not in ctes}
    return read == {table.rsplit(".", 1)[-1].strip('"`[]').lower()}


def _measure_sql(measure: Any, columns: Mapping[str, Any], dialect: str | None) -> str | None:
    """One dbt ``Measure`` as an aggregate over ``columns``, or None if that is not certain."""
    if _declares_filter(measure):
        return None
    builder = _SIMPLE_AGGS.get(_enum_name(getattr(measure, "agg", None)))
    if builder is None:
        return None
    target = _text(measure, "expr", "name")
    try:
        check_expression(target, aggregate=False, dialect=dialect)
        sql = builder(target)
        check_expression(sql, aggregate=True, dialect=dialect)
    except ValueError:
        return None
    lowered = {c.lower() for c in columns}
    if any(col.lower() not in lowered for col in expression_columns(sql, dialect)):
        return None
    return sql


def _convert_grains(raw_grains: Any) -> tuple[str, ...]:
    allowed = {_enum_name(g).lower() for g in (raw_grains or ())}
    grains = tuple(g for g in GRAINS if g in allowed)
    return grains or GRAINS


def _dimension(
    dbt_dim: Any, metric_grains: Any, columns: Mapping[str, Any], dialect: str | None,
) -> tuple[str, Dimension] | None:
    """``(group_by, Dimension)`` for one dbt dimension over ``columns``, or None."""
    group_by = _text(dbt_dim, "qualified_name", "name")
    if not _NAME.fullmatch(group_by):
        return None
    lowered = {c.lower(): c for c in columns}
    raw_expr = _text(dbt_dim, "expr", "name")
    if not raw_expr:
        return None
    column: str | None = None
    if _NAME.fullmatch(raw_expr):
        if raw_expr.lower() not in lowered:
            return None
        column = lowered[raw_expr.lower()]
        expr = exp.column(column, quoted=True).sql()
    else:
        try:
            check_expression(raw_expr, aggregate=False, dialect=dialect)
        except ValueError:
            return None
        if any(col.lower() not in lowered for col in expression_columns(raw_expr, dialect)):
            return None
        expr = raw_expr
    is_time = group_by == "metric_time" or _enum_name(getattr(dbt_dim, "type", None)) == "TIME"
    grains = _convert_grains(getattr(dbt_dim, "queryable_granularities", None) or metric_grains) if is_time else ()
    return group_by, Dimension(
        name=group_by, expr=expr, column=column, time=is_time, grains=grains, format=None,
        description=_text(dbt_dim, "description", "label"),
    )


class _Converter:
    """Converts the dbt metrics of one fetch, compiling SQL through ``client`` a few calls at a time."""

    def __init__(self, client: Any, config: DbtConfig, source_tables: set[str] | None, schema_loader: Any, dialect: str | None):
        self.client = client
        self.config = config
        self.source_tables = source_tables
        self.schema_loader = schema_loader
        self.dialect = dialect
        self._limit = asyncio.Semaphore(_COMPILE_CONCURRENCY)

    async def compile(self, name: str, group_by: list[str]) -> str:
        compile_fn = getattr(self.client, "compile_sql", None)
        if compile_fn is None:
            raise ValueError("the dbt client has no compile_sql.")
        kwargs: dict[str, Any] = {"metrics": [name]}
        if group_by:
            kwargs["group_by"] = group_by
        async with self._limit:
            sql = await _call_maybe_async(compile_fn, **kwargs)
        return _clean_query(sql, f"metric {name!r}")

    async def convert(self, dbt_metric: Any) -> Metric:
        name = _text(dbt_metric, "name")
        base = ["metric_time"] if getattr(dbt_metric, "requires_metric_time", False) else []
        sql = await self.compile(name, base)
        local = await self.local(dbt_metric, name, sql, base)
        if local is not None:
            return local
        return Metric(
            name=name, source=self.config.source, description=_text(dbt_metric, "description", "label"),
            sql=sql, origin=DBT_ORIGIN,
        )

    async def local(self, dbt_metric: Any, name: str, sql: str, base: list[str]) -> Metric | None:
        """``dbt_metric`` as a structured ``Metric`` when nothing about it is in doubt, else None."""
        table = self.config.table_map.get(name)
        measures = list(getattr(dbt_metric, "measures", None) or ())
        if (table is None or _enum_name(getattr(dbt_metric, "type", None)) != "SIMPLE"
                or _declares_filter(dbt_metric) or len(measures) != 1
                or (self.source_tables is not None and table not in self.source_tables)
                or not _scans_only(sql, table, self.dialect)):
            return None
        columns = self.schema_loader(table) if self.schema_loader is not None else None
        if not columns:
            return None
        measure_sql = _measure_sql(measures[0], columns, self.dialect)
        if measure_sql is None:
            return None

        agg_time = _text(measures[0], "agg_time_dimension") or None
        metric_grains = getattr(dbt_metric, "queryable_granularities", None)
        candidates: dict[str, Dimension] = {}
        dbt_dims = [
            d for d in getattr(dbt_metric, "dimensions", None) or () if _text(d, "qualified_name", "name") != "metric_time"
        ]
        if agg_time:
            # metric_time is the measure's agg_time_dimension, at the grain a query asks for.
            agg_dim = next((d for d in dbt_dims if _text(d, "name") == agg_time), None)
            dbt_dims.append(SimpleNamespace(
                name="metric_time", type="TIME", expr=_text(agg_dim, "expr") or agg_time,
                queryable_granularities=getattr(agg_dim, "queryable_granularities", None),
            ))
        for dbt_dim in dbt_dims:
            converted = _dimension(dbt_dim, metric_grains, columns, self.dialect)
            if converted is not None and converted[0] != name and converted[0] not in candidates:
                candidates[converted[0]] = converted[1]

        async def keeps(group_by: str) -> bool:
            # A dimension on another semantic model compiles to a join; only one on this table is local.
            try:
                compiled = await self.compile(name, list(dict.fromkeys([*base, group_by])))
            except Exception:
                return False
            return _scans_only(compiled, table, self.dialect)

        kept = await asyncio.gather(*(keeps(g) for g in candidates))
        dimensions = {g: d for (g, d), ok in zip(candidates.items(), kept, strict=True) if ok}
        if base and not all(g in dimensions for g in base):
            return None
        description = _text(dbt_metric, "description", "label")
        return Metric(
            name=name,
            source=self.config.source,
            description=description,
            table=table,
            measures={name: Measure(name=name, expr=measure_sql, format=None, description=description)},
            dimensions=dimensions,
            origin=DBT_ORIGIN,
        )

    async def run(self) -> tuple[dict[str, Metric], dict[str, MetricFailure]]:
        session = self.client.session()
        if hasattr(session, "__aenter__"):
            async with session:
                return await self._convert_all()
        with session:
            return await self._convert_all()

    async def _convert_all(self) -> tuple[dict[str, Metric], dict[str, MetricFailure]]:
        from panel_flowdash.value_text import redact_secrets

        raw_metrics = list(await _call_maybe_async(self.client.metrics))
        allow = set(self.config.metrics) if self.config.metrics is not None else None
        selected = [
            m for m in raw_metrics
            if _NAME.fullmatch(_text(m, "name")) and (allow is None or _text(m, "name") in allow)
        ]
        results = await asyncio.gather(*(self.convert(m) for m in selected), return_exceptions=True)
        converted: dict[str, Metric] = {}
        failures: dict[str, MetricFailure] = {}
        for dbt_metric, result in zip(selected, results, strict=True):
            name = _text(dbt_metric, "name")
            if isinstance(result, Metric):
                converted[name] = result
                continue
            if not isinstance(result, Exception):
                raise result
            placeholder = Metric(
                name=name, source=self.config.source, description=_text(dbt_metric, "description", "label"),
                sql="", origin=DBT_ORIGIN,
            )
            failures[name] = MetricFailure(placeholder, redact_secrets(str(result)), (DBT_KEY,))
        return converted, failures


_cache: dict[str, tuple[float, dict[str, Metric]]] = {}
_cache_lock = threading.Lock()


def clear_dbt_cache() -> None:
    """Forget every cached dbt fetch, so the next load asks dbt again."""
    with _cache_lock:
        _cache.clear()


def cached_dbt_metrics(config: DbtConfig) -> dict[str, Metric] | None:
    """The metrics of the last successful fetch for ``config`` if it has not expired, without asking dbt."""
    with _cache_lock:
        entry = _cache.get(config.cache_key())
    if entry is None or time.monotonic() - entry[0] >= CACHE_TTL:
        return None
    return entry[1]


def load_dbt_metrics(
    config: DbtConfig,
    section: Section,
) -> tuple[dict[str, Metric], dict[str, MetricFailure], str | None]:
    """
    Fetch metrics from the dbt Semantic Layer configured by ``config`` and
    convert them into FlowDash ``Metric`` objects, reusing a fetch less than
    ``CACHE_TTL`` seconds old.

    Returns ``(metrics, failures, error)``. Connection-level errors are caught,
    redacted, reported via ``section.warn`` and returned in ``error`` so local
    YAML metrics keep working.
    """
    from panel_flowdash.value_text import redact_secrets

    from .nodes import SOURCES_KEY, lookup_source

    if (cached := cached_dbt_metrics(config)) is not None:
        return dict(cached), {}, None

    context = section.context or {}
    source = lookup_source(config.source, context) if SOURCES_KEY in context else None
    source_tables: set[str] | None = None
    dialect: str | None = None
    schema_cache: dict[str, Mapping[str, Any] | None] = {}

    if isinstance(source, BaseSQLSource):
        dialect = None if source.dialect == "any" else source.dialect
        try:
            source_tables = set(source.get_tables())
        except Exception:
            source_tables = None

    def schema_loader(table: str) -> Mapping[str, Any] | None:
        if not isinstance(source, BaseSQLSource):
            return None
        if table not in schema_cache:
            try:
                raw_schema = source.get_schema(table)
                schema_cache[table] = {k: v for k, v in raw_schema.items() if k != "__len__"}
            except Exception:
                schema_cache[table] = None
        return schema_cache[table]

    report = section.warn
    client_entry = context.get(DBT_CLIENT_KEY)
    try:
        if client_entry is not None:
            client = client_entry(config) if callable(client_entry) else client_entry
        else:
            client = _make_client(config)
        converter = _Converter(client, config, source_tables, schema_loader, dialect)
        metrics, failures = _run_async(asyncio.wait_for(converter.run(), TIMEOUT))
    except ImportError as exc:
        msg = redact_secrets(f"Could not connect to the dbt Semantic Layer: install `dbtsl` (`pip install dbt-sl-sdk`). ({exc})")
        report((DBT_KEY,), msg)
        return {}, {}, msg
    except TimeoutError:
        msg = f"Could not load metrics from the dbt Semantic Layer: no answer within {TIMEOUT:g} seconds."
        report((DBT_KEY,), msg)
        return {}, {}, msg
    except Exception as exc:
        msg = redact_secrets(f"Could not load metrics from the dbt Semantic Layer: {exc}")
        report((DBT_KEY,), msg)
        return {}, {}, msg

    if not failures:
        with _cache_lock:
            _cache[config.cache_key()] = (time.monotonic(), dict(metrics))
    for name, failure in failures.items():
        report((DBT_KEY,), f"dbt metric {name!r} is left out: {failure.message}")
    return metrics, failures, None
