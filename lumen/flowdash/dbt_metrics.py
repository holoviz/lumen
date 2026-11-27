"""
Inherit metrics from a dbt Semantic Layer into FlowDash's ``metrics:`` section.

Declared under ``metrics:`` in ``flowdash.yaml``::

    metrics:
      dbt:
        source: warehouse
        environment_id: 12345
        host: semantic-layer.cloud.getdbt.com   # optional
        auth_token: ${DBT_AUTH_TOKEN}           # optional; ${VAR} or encrypted secret
        table: orders                           # optional default table for local SQL
        tables:                                 # optional metric/semantic-model -> table map
          revenue: orders
        metrics: [revenue, order_count]         # optional allowlist of dbt metric names

Each dbt metric is converted into a standard ``Metric`` (with ``origin="dbt"``)
so ``MetricNode``, ``aggregate()`` and ``bind_sql()`` work unchanged. Simple,
single-table ratio and single-table derived metrics are expressed locally as
structured ``Metric`` objects; metrics that cannot be expressed locally
(cumulative, conversion, filtered, time-offset or multi-table metrics) fall back
to ``client.compile_sql(...)`` as SQL-form ``Metric`` objects.
"""

from __future__ import annotations

import asyncio
import inspect
import logging

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sqlglot

from sqlglot import exp

from ..sources.base import BaseSQLSource
from .metrics import (
    _NAME, _TYPE, GRAINS, OPS, RANGE_TYPES, SCALAR_TYPES, Dimension, Measure,
    Metric, MetricFailure, Parameter, check_expression, expression_columns,
)
from .project import Section

if TYPE_CHECKING:
    from .metrics import _Reader

logger = logging.getLogger(__name__)

# The key under ``metrics:`` in flowdash.yaml holding dbt Semantic Layer settings.
DBT_KEY = "dbt"
# Origin tag placed on metrics inherited from dbt Semantic Layer.
DBT_ORIGIN = "dbt"
# Optional context key allowing callers or tests to supply a dbt client or factory.
DBT_CLIENT_KEY = "lumen.dbt_client"

DEFAULT_HOST = "semantic-layer.cloud.getdbt.com"

_DBT_KEYS = {"source", "environment_id", "host", "auth_token", "table", "tables", "metrics"}

# dbt aggregation types expressible as standard SQL aggregates over one table.
_SIMPLE_AGGS = {
    "SUM": lambda target: f"SUM({target})",
    "COUNT": lambda target: "COUNT(*)" if target in ("", "1", "*") else f"COUNT({target})",
    "COUNT_DISTINCT": lambda target: f"COUNT(DISTINCT {target})",
    "AVERAGE": lambda target: f"AVG({target})",
    "AVG": lambda target: f"AVG({target})",
    "MIN": lambda target: f"MIN({target})",
    "MAX": lambda target: f"MAX({target})",
    "SUM_BOOLEAN": lambda target: f"SUM(CASE WHEN {target} THEN 1 ELSE 0 END)",
    "MEDIAN": lambda target: f"MEDIAN({target})",
}


@dataclass(frozen=True)
class DbtConfig:
    """Connection and mapping settings for inheriting metrics from a dbt Semantic Layer."""

    source: str
    environment_id: int
    host: str = DEFAULT_HOST
    auth_token: Any = None
    table: str | None = None
    tables: Mapping[str, str] = field(default_factory=dict)
    metrics: tuple[str, ...] | None = None


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

    table = reader.text(spec.get("table"), (*path, "table"), "table")
    if "table" in spec and not table:
        reader.fail((*path, "table"), "dbt `table` must be a table name.")

    tables: dict[str, str] = {}
    if "tables" in spec and spec["tables"] is not None:
        raw_tables = reader.mapping(spec["tables"], (*path, "tables"), "dbt tables")
        if raw_tables is not None:
            for key, value in raw_tables.items():
                if not isinstance(key, str) or not key:
                    reader.fail((*path, "tables", key), "dbt table mapping keys must be non-empty text.")
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
        table=table,
        tables=tables,
        metrics=allowlist,
    )


def encrypt_dbt_token(token: str) -> dict[str, str]:
    """
    Encrypt a dbt Semantic Layer token for storage in ``flowdash.yaml`` via
    ``panel_flowdash.secret_store``. Refuses plaintext fallback when
    ``cryptography`` is unavailable.
    """
    from panel_flowdash.secret_store import (
        INSTALL_HINT, SecretError, encrypt_secret, encryption_available,
    )

    if not encryption_available():
        raise ValueError(f"{INSTALL_HINT} Or write auth_token as an environment variable reference, e.g. '${{DBT_AUTH_TOKEN}}'.")
    try:
        return encrypt_secret(token)
    except SecretError as exc:
        raise ValueError(f"Could not encrypt dbt auth_token: {exc}") from None


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


async def _open_session(client: Any):
    """Return a context manager pair supporting both async and sync dbt client sessions."""
    session = client.session()
    if hasattr(session, "__aenter__"):
        return session, True
    return session, False


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value).upper()
    if hasattr(value, "name"):
        return str(value.name).upper()
    return str(value).upper()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _has_filter(obj: Any) -> bool:
    flt = _get(obj, "filter")
    if flt is None:
        return False
    if isinstance(flt, str):
        return bool(flt.strip())
    if isinstance(flt, (list, tuple, set, Mapping)):
        return bool(flt)
    where_filters = _get(flt, "where_filters")
    if where_filters is not None:
        return bool(where_filters)
    where_sql = _get(flt, "where_sql_template")
    if where_sql is not None:
        return bool(str(where_sql).strip())
    return True


def _has_offset(obj: Any) -> bool:
    return bool(_get(obj, "offset_window") or _get(obj, "offset_to_grain"))


def _measure_sql(measure: Any) -> str | None:
    """Convert one dbt ``Measure`` into an aggregate SQL expression, or None if unsupported."""
    if _has_filter(measure):
        return None
    agg = _enum_name(_get(measure, "agg"))
    builder = _SIMPLE_AGGS.get(agg)
    if builder is None:
        return None
    raw_expr = _get(measure, "expr")
    target = str(raw_expr).strip() if raw_expr is not None and str(raw_expr).strip() else str(_get(measure, "name") or "").strip()
    if not target and agg != "COUNT":
        return None
    sql = builder(target)
    try:
        check_expression(sql, aggregate=True)
    except ValueError:
        return None
    return sql


def _measure_table(measure: Any, config: DbtConfig, source_tables: set[str] | None) -> str | None:
    """Find the physical table in ``config.source`` backing ``measure``, if known."""
    for attr in ("table", "semantic_model", "model"):
        ref = _get(measure, attr)
        if isinstance(ref, str) and ref:
            if ref in config.tables:
                return config.tables[ref]
            if source_tables is None or ref in source_tables:
                return ref
    return None


def _resolve_metric_table(
    dbt_metric: Any,
    measures: list[Any],
    config: DbtConfig,
    source_tables: set[str] | None,
) -> str | None:
    """Resolve a single target table for ``dbt_metric``, or None if ambiguous or unknown."""
    name = str(_get(dbt_metric, "name") or "")
    if name in config.tables:
        return config.tables[name]
    for attr in ("table", "semantic_model", "model"):
        ref = _get(dbt_metric, attr)
        if isinstance(ref, str) and ref:
            if ref in config.tables:
                return config.tables[ref]
            if source_tables is None or ref in source_tables:
                return ref
    found = {_measure_table(m, config, source_tables) for m in measures}
    found.discard(None)
    if len(found) == 1:
        return next(iter(found))
    if len(found) > 1:
        return None
    return config.table


def _simple_measure_info(
    dbt_metric: Any,
    config: DbtConfig,
    source_tables: set[str] | None,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(table, aggregate_sql, agg_time_dimension)`` for a simple dbt metric."""
    if _has_filter(dbt_metric):
        return None, None, None
    type_params = _get(dbt_metric, "type_params")
    if type_params is not None and _has_filter(type_params):
        return None, None, None
    measures = list(_get(dbt_metric, "measures") or ())
    chosen = None
    tp_measure = _get(type_params, "measure") if type_params is not None else None
    if tp_measure is not None:
        if _has_filter(tp_measure):
            return None, None, None
        tp_name = tp_measure if isinstance(tp_measure, str) else _get(tp_measure, "name")
        chosen = next((m for m in measures if _get(m, "name") == tp_name), None)
        if chosen is None and not isinstance(tp_measure, str) and _get(tp_measure, "agg") is not None:
            chosen = tp_measure
    elif len(measures) == 1:
        chosen = measures[0]
    if chosen is None:
        return None, None, None
    expr = _measure_sql(chosen)
    if expr is None:
        return None, None, None
    table = _resolve_metric_table(dbt_metric, [chosen], config, source_tables)
    if not table:
        return None, None, None
    agg_time = _get(chosen, "agg_time_dimension")
    return table, expr, str(agg_time) if agg_time else None


def _resolve_input_ref(
    ref: Any,
    simple_by_name: Mapping[str, tuple[str, str, str | None]],
    measures_by_name: Mapping[str, Any],
    config: DbtConfig,
    source_tables: set[str] | None,
) -> tuple[str, str, str, str | None] | None:
    """Resolve a ratio/derived input reference to ``(alias, table, aggregate_sql, agg_time_dim)``."""
    if ref is None or _has_filter(ref) or _has_offset(ref):
        return None
    if isinstance(ref, str):
        name, alias = ref, ref
    else:
        name = _get(ref, "name")
        alias = _get(ref, "alias") or name
    if not isinstance(name, str) or not name:
        return None
    if name in simple_by_name:
        table, expr, agg_time = simple_by_name[name]
        return str(alias), table, expr, agg_time
    measure = measures_by_name.get(name) or (ref if not isinstance(ref, str) and _get(ref, "agg") is not None else None)
    if measure is None:
        return None
    expr = _measure_sql(measure)
    table = _measure_table(measure, config, source_tables) or config.table
    if expr is None or not table:
        return None
    agg_time = _get(measure, "agg_time_dimension")
    return str(alias), table, expr, str(agg_time) if agg_time else None


def _substitute_derived(expr_text: str, replacements: Mapping[str, str]) -> str | None:
    """Replace input metric identifiers in a derived expression with their aggregate SQL."""
    try:
        parsed = sqlglot.parse_one(expr_text)
    except sqlglot.errors.SqlglotError:
        return None
    parsed_replacements: dict[str, exp.Expression] = {}
    for key, sql in replacements.items():
        try:
            parsed_replacements[key] = exp.Paren(this=sqlglot.parse_one(sql))
        except sqlglot.errors.SqlglotError:
            return None

    unresolved = False

    def replace(node: exp.Expression) -> exp.Expression:
        nonlocal unresolved
        if isinstance(node, exp.Column):
            if node.table:
                unresolved = True
                return node
            if node.name in parsed_replacements:
                return parsed_replacements[node.name].copy()
            unresolved = True
        return node

    transformed = parsed.transform(replace)
    if unresolved:
        return None
    sql = transformed.sql()
    try:
        check_expression(sql, aggregate=True)
    except ValueError:
        return None
    return sql


def _convert_grains(raw_grains: Any) -> tuple[str, ...]:
    allowed = {_enum_name(g).lower() for g in (raw_grains or ())}
    grains = tuple(g for g in GRAINS if g in allowed)
    return grains or GRAINS


def _convert_dimensions(
    dbt_metric: Any,
    measure_names: set[str],
    agg_time_dimension: str | None,
    source_columns: Mapping[str, Any] | None,
    dialect: str | None,
) -> dict[str, Dimension]:
    """Convert a dbt metric's dimensions into FlowDash ``Dimension`` objects."""
    dimensions: dict[str, Dimension] = {}
    metric_grains = _get(dbt_metric, "queryable_granularities") or ()
    lowered_cols = {c.lower(): c for c in source_columns} if source_columns is not None else None

    for dbt_dim in _get(dbt_metric, "dimensions") or ():
        raw_name = str(_get(dbt_dim, "name") or "").strip()
        if not raw_name or not _NAME.fullmatch(raw_name) or raw_name in measure_names or raw_name in dimensions:
            continue
        is_time = _enum_name(_get(dbt_dim, "type")) == "TIME"
        raw_expr = _get(dbt_dim, "expr")
        expr_text = str(raw_expr).strip() if raw_expr is not None and str(raw_expr).strip() else ""

        column: str | None = None
        expr: str
        if not expr_text:
            candidate = agg_time_dimension if raw_name == "metric_time" and agg_time_dimension else raw_name
            if lowered_cols is not None and candidate.lower() not in lowered_cols and "__" in candidate:
                # Entity-prefixed dbt dimension like customer__country -> try unqualified column on this table.
                suffix = candidate.rsplit("__", 1)[1]
                if suffix.lower() in lowered_cols:
                    candidate = lowered_cols[suffix.lower()]
            elif lowered_cols is not None and candidate.lower() in lowered_cols:
                candidate = lowered_cols[candidate.lower()]
            column = candidate
            expr = exp.column(column, quoted=True).sql()
        elif _NAME.fullmatch(expr_text):
            candidate = expr_text
            if lowered_cols is not None and candidate.lower() in lowered_cols:
                candidate = lowered_cols[candidate.lower()]
            column = candidate
            expr = exp.column(column, quoted=True).sql()
        else:
            try:
                check_expression(expr_text, aggregate=False, dialect=dialect)
            except ValueError:
                continue
            expr = expr_text

        if lowered_cols is not None:
            if column is not None:
                if column.lower() not in lowered_cols:
                    continue
            elif any(col.lower() not in lowered_cols for col in expression_columns(expr, dialect)):
                continue

        dim_grains = _convert_grains(_get(dbt_dim, "queryable_granularities") or metric_grains) if is_time else ()
        description = str(_get(dbt_dim, "description") or _get(dbt_dim, "label") or "").strip()
        dimensions[raw_name] = Dimension(
            name=raw_name,
            expr=expr,
            column=column,
            time=is_time,
            grains=dim_grains,
            format=None,
            description=description,
        )
    return dimensions


def _convert_parameters(dbt_metric: Any, dimensions: Mapping[str, Dimension]) -> dict[str, Parameter]:
    """Convert any explicit dbt parameter declarations into FlowDash ``Parameter`` objects."""
    raw = _get(dbt_metric, "parameters")
    if not isinstance(raw, Mapping):
        return {}
    parameters: dict[str, Parameter] = {}
    for name, item in raw.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in dimensions:
            continue
        if isinstance(item, Parameter):
            parameters[name] = item
            continue
        if not isinstance(item, Mapping):
            continue
        given = str(item.get("type") or "string")
        match = _TYPE.fullmatch(given)
        if not match:
            continue
        kind, inner = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), None)
        if (inner is None and kind not in SCALAR_TYPES) or (inner is not None and inner not in SCALAR_TYPES):
            continue
        if kind == "range" and inner not in RANGE_TYPES:
            continue
        column = item.get("column")
        if not isinstance(column, str) or not column:
            continue
        op = str(item.get("op") or {"list": "in", "range": "between"}.get(kind, "="))
        if op not in OPS:
            continue
        parameter = Parameter(
            name=name,
            kind=kind,
            item=inner,
            default=None,
            required=bool(item.get("required", False)),
            description=str(item.get("description") or ""),
            column=column,
            op=op,
        )
        if item.get("default") is not None:
            try:
                parameter = Parameter(
                    name=parameter.name,
                    kind=parameter.kind,
                    item=parameter.item,
                    default=parameter.coerce(item["default"]),
                    required=parameter.required,
                    description=parameter.description,
                    column=parameter.column,
                    op=parameter.op,
                )
            except ValueError:
                continue
        parameters[name] = parameter
    return parameters


def _try_convert_local(
    dbt_metric: Any,
    simple_by_name: Mapping[str, tuple[str, str, str | None]],
    config: DbtConfig,
    source_tables: set[str] | None,
    schema_loader: Any,
    dialect: str | None,
) -> tuple[Metric | None, str]:
    """
    Attempt to convert ``dbt_metric`` into a local structured ``Metric``.

    Returns ``(Metric, "")`` on success or ``(None, reason)`` when it must
    fall back to ``client.compile_sql``.
    """
    name = str(_get(dbt_metric, "name") or "").strip()
    if not _NAME.fullmatch(name):
        return None, f"metric name {name!r} is not a valid FlowDash metric identifier"
    if _has_filter(dbt_metric):
        return None, "metric has a dbt filter"

    metric_type = _enum_name(_get(dbt_metric, "type")) or "SIMPLE"
    type_params = _get(dbt_metric, "type_params")
    if type_params is not None and _has_filter(type_params):
        return None, "metric type_params has a dbt filter"

    measures_list = list(_get(dbt_metric, "measures") or ())
    measures_by_name = {str(_get(m, "name")): m for m in measures_list if _get(m, "name")}

    table: str | None = None
    measure_sql: str | None = None
    agg_time: str | None = None

    if metric_type == "SIMPLE":
        table, measure_sql, agg_time = _simple_measure_info(dbt_metric, config, source_tables)
        if not table or not measure_sql:
            return None, "simple metric measure or target table cannot be resolved locally"

    elif metric_type == "RATIO":
        num_ref = _get(type_params, "numerator") if type_params is not None else (measures_list[0] if len(measures_list) == 2 else None)
        den_ref = _get(type_params, "denominator") if type_params is not None else (measures_list[1] if len(measures_list) == 2 else None)
        num = _resolve_input_ref(num_ref, simple_by_name, measures_by_name, config, source_tables)
        den = _resolve_input_ref(den_ref, simple_by_name, measures_by_name, config, source_tables)
        if num is None or den is None:
            return None, "ratio metric inputs cannot be resolved locally"
        if num[1] != den[1]:
            return None, "ratio metric spans multiple tables"
        table = num[1]
        measure_sql = f"1.0 * ({num[2]}) / NULLIF(({den[2]}), 0)"
        agg_time = num[3] or den[3]

    elif metric_type == "DERIVED":
        expr_text = _get(type_params, "expr") if type_params is not None else None
        raw_inputs = (_get(type_params, "metrics") or _get(type_params, "input_metrics")) if type_params is not None else None
        if not isinstance(expr_text, str) or not expr_text.strip() or not raw_inputs:
            return None, "derived metric lacks a local expression or input metrics"
        replacements: dict[str, str] = {}
        tables_used: set[str] = set()
        for ref in raw_inputs:
            resolved = _resolve_input_ref(ref, simple_by_name, measures_by_name, config, source_tables)
            if resolved is None:
                return None, "derived metric input cannot be resolved locally"
            alias, ref_table, ref_sql, ref_time = resolved
            replacements[alias] = ref_sql
            tables_used.add(ref_table)
            agg_time = agg_time or ref_time
        if len(tables_used) != 1:
            return None, "derived metric spans multiple tables"
        table = next(iter(tables_used))
        measure_sql = _substitute_derived(expr_text, replacements)
        if measure_sql is None:
            return None, "derived metric expression cannot be translated locally"

    else:
        return None, f"metric type {metric_type!r} cannot be expressed as a single-table structured metric"

    try:
        check_expression(measure_sql, aggregate=True, dialect=dialect)
    except ValueError as exc:
        return None, f"measure expression invalid locally: {exc}"

    source_columns = schema_loader(table) if schema_loader is not None else None
    description = str(_get(dbt_metric, "description") or _get(dbt_metric, "label") or "").strip()
    measures = {name: Measure(name=name, expr=measure_sql, format=None, description=description)}
    dimensions = _convert_dimensions(dbt_metric, set(measures), agg_time, source_columns, dialect)
    parameters = _convert_parameters(dbt_metric, dimensions)
    return (
        Metric(
            name=name,
            source=config.source,
            description=description,
            table=table,
            sql=None,
            measures=measures,
            dimensions=dimensions,
            parameters=parameters,
            origin=DBT_ORIGIN,
        ),
        "",
    )


async def _compile_fallback(client: Any, dbt_metric: Any, config: DbtConfig, reason: str) -> Metric:
    """Fall back to ``client.compile_sql`` and wrap the result as a SQL-form ``Metric``."""
    name = str(_get(dbt_metric, "name") or "").strip()
    description = str(_get(dbt_metric, "description") or _get(dbt_metric, "label") or "").strip()
    compile_fn = getattr(client, "compile_sql", None)
    if compile_fn is None:
        raise ValueError(f"Metric {name!r} cannot be expressed locally ({reason}) and the dbt client has no compile_sql.")

    kwargs: dict[str, Any] = {"metrics": [name]}
    if _get(dbt_metric, "requires_metric_time"):
        kwargs["group_by"] = ["metric_time"]

    sql = await _call_maybe_async(compile_fn, **kwargs)
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError(f"dbt compile_sql returned no SQL for metric {name!r}.")
    cleaned = sql.strip().rstrip(";").strip()
    try:
        parsed = sqlglot.parse_one(cleaned)
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError(f"dbt compile_sql returned unparseable SQL for {name!r}: {exc}") from None
    if not isinstance(parsed, exp.Query):
        raise ValueError(f"dbt compile_sql did not return a SELECT or WITH query for {name!r}.")

    logger.debug("dbt metric %r fell back to client.compile_sql (%s).", name, reason)
    return Metric(
        name=name,
        source=config.source,
        description=description,
        table=None,
        sql=cleaned,
        measures={},
        dimensions={},
        parameters={},
        origin=DBT_ORIGIN,
    )


async def _fetch_and_convert(
    client: Any,
    config: DbtConfig,
    source_tables: set[str] | None,
    schema_loader: Any,
    dialect: str | None,
) -> tuple[dict[str, Metric], dict[str, MetricFailure]]:
    session, is_async = await _open_session(client)
    if is_async:
        async with session:
            return await _convert_in_session(client, config, source_tables, schema_loader, dialect)
    with session:
        return await _convert_in_session(client, config, source_tables, schema_loader, dialect)


async def _convert_in_session(
    client: Any,
    config: DbtConfig,
    source_tables: set[str] | None,
    schema_loader: Any,
    dialect: str | None,
) -> tuple[dict[str, Metric], dict[str, MetricFailure]]:
    raw_metrics = list(await _call_maybe_async(client.metrics))

    # Pre-index simple metrics so ratio and derived metrics can resolve their inputs locally.
    simple_by_name: dict[str, tuple[str, str, str | None]] = {}
    for m in raw_metrics:
        m_name = str(_get(m, "name") or "").strip()
        m_type = _enum_name(_get(m, "type")) or "SIMPLE"
        if m_name and m_type == "SIMPLE":
            table, expr, agg_time = _simple_measure_info(m, config, source_tables)
            if table and expr:
                simple_by_name[m_name] = (table, expr, agg_time)

    allow = set(config.metrics) if config.metrics is not None else None
    converted: dict[str, Metric] = {}
    failures: dict[str, MetricFailure] = {}

    for dbt_metric in raw_metrics:
        name = str(_get(dbt_metric, "name") or "").strip()
        if not name or (allow is not None and name not in allow):
            continue
        if not _NAME.fullmatch(name):
            continue
        local, reason = _try_convert_local(dbt_metric, simple_by_name, config, source_tables, schema_loader, dialect)
        if local is not None:
            converted[name] = local
            continue
        try:
            converted[name] = await _compile_fallback(client, dbt_metric, config, reason)
        except Exception as exc:
            from panel_flowdash.value_text import redact_secrets

            placeholder = Metric(
                name=name,
                source=config.source,
                description=str(_get(dbt_metric, "description") or _get(dbt_metric, "label") or "").strip(),
                sql="",
                origin=DBT_ORIGIN,
            )
            failures[name] = MetricFailure(placeholder, redact_secrets(str(exc)), (DBT_KEY,))
    return converted, failures


def load_dbt_metrics(
    config: DbtConfig,
    section: Section,
) -> tuple[dict[str, Metric], dict[str, MetricFailure], str | None]:
    """
    Fetch metrics from the dbt Semantic Layer configured by ``config`` and
    convert them into FlowDash ``Metric`` objects.

    Returns ``(metrics, failures, error)``. Connection-level errors are caught,
    redacted, reported via ``section.warn`` and returned in ``error`` so local
    YAML metrics keep working.
    """
    from panel_flowdash.value_text import redact_secrets

    from .nodes import SOURCES_KEY, lookup_source

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

    # Section.warn logs the warning and keeps it in the project's warnings.
    report = section.warn
    client_entry = context.get(DBT_CLIENT_KEY)
    try:
        if client_entry is not None:
            client = client_entry(config) if callable(client_entry) else client_entry
        else:
            client = _make_client(config)
        metrics, failures = _run_async(_fetch_and_convert(client, config, source_tables, schema_loader, dialect))
    except ImportError as exc:
        msg = redact_secrets(f"Could not connect to the dbt Semantic Layer: install `dbtsl` (`pip install dbt-sl-sdk`). ({exc})")
        report((DBT_KEY,), msg)
        return {}, {}, msg
    except Exception as exc:
        msg = redact_secrets(f"Could not load metrics from the dbt Semantic Layer: {exc}")
        report((DBT_KEY,), msg)
        return {}, {}, msg

    for name, failure in failures.items():
        report((DBT_KEY,), f"dbt metric {name!r} is left out: {failure.message}")
    return metrics, failures, None


def metric_spec(metric: Metric) -> dict[str, Any]:
    """Serialize ``metric`` into a plain YAML-like mapping for display in Data/Metrics views."""
    spec: dict[str, Any] = {"source": metric.source}
    if metric.description:
        spec["description"] = metric.description
    if metric.form == "sql":
        spec["sql"] = metric.sql or ""
    else:
        spec["table"] = metric.table
        spec["measures"] = {
            name: {
                "expr": m.expr,
                **({"format": m.format} if m.format else {}),
                **({"description": m.description} if m.description else {}),
            }
            for name, m in metric.measures.items()
        }
        if metric.dimensions:
            spec["dimensions"] = {
                name: {
                    **({"column": d.column} if d.column is not None else {"expr": d.expr}),
                    **({"type": "time", "grains": list(d.grains)} if d.time else {}),
                    **({"format": d.format} if d.format else {}),
                    **({"description": d.description} if d.description else {}),
                }
                for name, d in metric.dimensions.items()
            }
    if metric.parameters:
        spec["parameters"] = {
            name: {
                "type": p.type,
                **({"default": p.default} if p.default is not None else {}),
                **({"required": True} if p.required else {}),
                **({"column": p.column} if p.column else {}),
                **({"op": p.op} if p.op else {}),
                **({"description": p.description} if p.description else {}),
            }
            for name, p in metric.parameters.items()
        }
    return spec
