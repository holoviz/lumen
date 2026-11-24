"""
Checks every query and chart spec of a node must pass, whoever wrote them:
the agent, a person editing the node, an imported file or a replayed trace.
"""
from __future__ import annotations

import copy

from typing import Any

import sqlglot

from sqlglot import exp

from ..ai.agents.sql import validate_read_only_sql
from ..sources.base import BaseSQLSource

# Keys whose value makes a Vega-Lite chart fetch or link to something.
URL_KEYS = ("url", "href", "baseURL")

# Scalar functions that read the engine's settings, session or environment, or load code.
DENIED_FUNCTIONS = frozenset({
    "current_setting", "getenv", "load_extension", "current_user", "user", "session_user",
    "current_database", "current_catalog", "current_schema", "current_schemas", "version",
    "pragma_version", "pragma_platform", "duckdb_settings", "set_config", "getvariable",
})


def _read(dialect: str) -> str | None:
    return None if dialect == "any" else dialect


def _relations(source: BaseSQLSource) -> tuple[set[str], set[str]]:
    """
    The table names a query on ``source`` may use, and the table functions
    and file references it may use because a declared table is defined by
    one, e.g. ``read_csv('orders.csv')``.
    """
    names = {table.lower() for table in source.get_tables()}
    functions = set()
    tables = source.tables if isinstance(source.tables, dict) else {}
    for expr in tables.values():
        if not isinstance(expr, str):
            continue
        try:
            parsed = sqlglot.parse_one(expr, read=_read(source.dialect))
        except sqlglot.errors.SqlglotError:
            continue
        candidates = [parsed] if isinstance(parsed, exp.Func) else []
        candidates += [table.this for table in parsed.find_all(exp.Table) if not isinstance(table.this, exp.Identifier)]
        candidates += [table for table in parsed.find_all(exp.Table) if isinstance(table.this, exp.Identifier)]
        for item in candidates:
            if isinstance(item, exp.Table):
                names.add(item.name.lower())
            else:
                functions.add(item.sql(dialect=_read(source.dialect)))
    return names, functions


def _function_name(func: exp.Func) -> str:
    return (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()


def _without_ctes(node: exp.Expression) -> exp.Expression:
    node = node.copy()
    for with_ in list(node.find_all(exp.With)):
        with_.pop()
    return node


def _reads_a_table(parsed: exp.Expression, ctes: dict[str, exp.Expression]) -> bool:
    """Whether the result of ``parsed`` depends on a table, following CTE references into their bodies."""
    def reads(node, visiting):
        for table in _without_ctes(node).find_all(exp.Table):
            name = table.name.lower()
            if isinstance(table.this, exp.Identifier) and not table.db and name in ctes:
                if name not in visiting and reads(ctes[name], visiting | {name}):
                    return True
            else:
                return True
        return False

    return reads(parsed, frozenset())


def _declared(table: exp.Table, names: set[str]) -> bool:
    """Whether ``table`` names a declared table, bare or qualified with the default schema and catalog."""
    if table.name.lower() not in names:
        return ".".join(part.name for part in table.parts).lower() in names
    return table.db.lower() in ("", "main") and table.catalog.lower() in ("", "memory")


def check_query(sql: str, source: BaseSQLSource) -> None:
    """
    Raise ``ValueError`` unless ``sql`` is one read-only query whose result
    depends on at least one table, that reads only the source's declared
    tables (bare, or qualified with DuckDB's default ``main`` schema and
    ``memory`` catalog), CTEs of its own, or the table functions and files the
    declared tables are defined by, and that calls none of the scalar
    functions in ``DENIED_FUNCTIONS``. Other scalar functions are not
    restricted.
    """
    validate_read_only_sql(sql, source.dialect)
    parsed = sqlglot.parse_one(sql, read=_read(source.dialect))
    ctes = {cte.alias_or_name.lower(): cte.this for cte in parsed.find_all(exp.CTE)}
    if not _reads_a_table(parsed, ctes):
        raise ValueError("The query reads no table. Select from the source's tables instead of returning literal values.")
    for func in parsed.find_all(exp.Func):
        if _function_name(func) in DENIED_FUNCTIONS:
            raise ValueError(f"The query calls {_function_name(func)}(), which reads the engine's settings or environment.")
    names, functions = _relations(source)
    for table in parsed.find_all(exp.Table):
        if isinstance(table.this, exp.Identifier):
            if (not table.db and table.name.lower() in ctes) or _declared(table, names):
                continue
            qualified = ".".join(part.name for part in table.parts)
            raise ValueError(
                f"The query reads {qualified.lower()!r}, which is not a table of this source. "
                f"Its tables are: {', '.join(sorted(source.get_tables()))}."
            )
        if table.this.sql(dialect=_read(source.dialect)) not in functions:
            raise ValueError(
                f"The query calls {table.this.sql(dialect=_read(source.dialect))}; queries may only read the "
                f"source's tables: {', '.join(sorted(source.get_tables()))}."
            )


def _transform_fields(spec: Any) -> set[str]:
    from ..ai.agents.vega_lite import VegaLiteAgent

    fields: set[str] = set()
    if isinstance(spec, dict):
        if isinstance(spec.get("transform"), list):
            fields.update(VegaLiteAgent._extract_as_keys(spec["transform"]))
        for value in spec.values():
            fields |= _transform_fields(value)
    elif isinstance(spec, list):
        for item in spec:
            fields |= _transform_fields(item)
    return fields


def _has_key(spec: Any, keys: tuple[str, ...]) -> bool:
    if isinstance(spec, dict):
        return any(key in keys or _has_key(value, keys) for key, value in spec.items())
    if isinstance(spec, list):
        return any(_has_key(item, keys) for item in spec)
    return False


def check_chart(spec: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """
    The spec to render for a chart of a table with ``columns``, or
    ``ValueError`` saying what to fix. The data comes from the node's input,
    which Lumen's ``VegaLiteView`` injects as the top-level ``data`` when it
    renders, so a top-level ``data`` that only names a dataset is dropped and
    any other ``data`` or ``datasets``, at any depth, is refused, as is any
    ``url``, ``href`` or ``baseURL``; ``usermeta`` (embed options) is dropped.
    """
    from ..ai.utils import _spec_field_references, normalize_vegalite_spec

    if not isinstance(spec, dict):
        raise ValueError("The spec must be a Vega-Lite JSON object.")
    spec = copy.deepcopy(spec)
    if isinstance(spec.get("data"), dict) and set(spec["data"]) <= {"name"}:
        spec.pop("data")
    spec.pop("usermeta", None)
    if _has_key(spec, ("data", "datasets")):
        raise ValueError("The chart's data comes from the data node; remove `data` and `datasets` from the spec.")
    if _has_key(spec, URL_KEYS):
        raise ValueError("Charts may not load or link to URLs; remove `url` and `href` from the spec.")
    unknown = _spec_field_references(spec, "vega-lite") - set(columns) - _transform_fields(spec)
    if unknown:
        raise ValueError(
            f"The spec references fields the data does not have: {', '.join(sorted(unknown))}. "
            f"Its columns are: {', '.join(columns)}."
        )
    try:
        normalize_vegalite_spec(copy.deepcopy(spec))
    except Exception as exc:
        raise ValueError(f"Invalid Vega-Lite spec: {exc}") from exc
    return spec


def pipeline_columns(pipeline: Any) -> list[str]:
    """The columns of a Lumen pipeline, from its schema, so nothing is fetched."""
    return [str(column) for column in pipeline.schema if column != "__len__"]


# SEAM(panel-flowdash#163): the functions below have the signature of the
# ``check_config=`` hook in ``@register`` metadata, check(config, inputs),
# where ``config`` is the node's merged config and ``inputs`` its live input
# values without None and failures. Once the hook is released each is passed
# to the ``@register`` call of its node, so a config that fails is refused
# before it is applied instead of leaving the node in an error state. They
# only inspect: none runs a query or fetches rows.

def check_table_config(config: dict[str, Any], inputs: dict[str, Any]) -> str | None:
    """Check a ``Lumen/table`` config: the table must be one of the source's."""
    source, table = inputs.get("source"), config.get("table")
    if source is None or not table:
        return None
    tables = source.get_tables()
    if table not in tables:
        return f"The source has no table {table!r}. Its tables are: {', '.join(sorted(tables))}."
    return None


def check_filter_config(config: dict[str, Any], inputs: dict[str, Any]) -> str | None:
    """Check a ``Lumen/filter`` config: the field must be a column of the input pipeline."""
    pipeline, field = inputs.get("pipeline"), config.get("field")
    if pipeline is None or not field:
        return None
    columns = pipeline_columns(pipeline)
    if field not in columns:
        return f"The data has no column {field!r}. Its columns are: {', '.join(columns)}."
    return None


def check_sql_config(config: dict[str, Any], inputs: dict[str, Any]) -> str | None:
    """Check a ``Lumen/sql`` config with ``check_query``, without running the query."""
    source, sql = inputs.get("source"), (config.get("sql") or "").strip()
    if source is None or not sql:
        return None
    if not isinstance(source, BaseSQLSource):
        return f"{type(source).__name__} does not run SQL."
    try:
        check_query(sql, source)
    except ValueError as exc:
        return str(exc)
    return None


def check_view_config(config: dict[str, Any], inputs: dict[str, Any]) -> str | None:
    """Check a ``Lumen/view`` config with ``check_chart`` against the input's columns."""
    pipeline, spec = inputs.get("pipeline"), config.get("spec")
    if pipeline is None or not spec:
        return None
    try:
        check_chart(spec, pipeline_columns(pipeline))
    except ValueError as exc:
        return str(exc)
    return None
