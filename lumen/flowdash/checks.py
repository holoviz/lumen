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
URL_KEYS = ("url", "href")


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


def check_query(sql: str, source: BaseSQLSource) -> None:
    """
    Raise ``ValueError`` unless ``sql`` is one read-only query that reads at
    least one table, and only the source's declared tables, CTEs of its own,
    or the table functions and files the declared tables are defined by.
    """
    validate_read_only_sql(sql, source.dialect)
    parsed = sqlglot.parse_one(sql, read=_read(source.dialect))
    tables = list(parsed.find_all(exp.Table))
    if not tables:
        raise ValueError("The query reads no table. Select from the source's tables instead of returning literal values.")
    names, functions = _relations(source)
    ctes = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    for table in tables:
        if isinstance(table.this, exp.Identifier):
            name = table.name.lower()
            qualified = ".".join(part.name for part in table.parts).lower()
            if (not table.db and name in names | ctes) or qualified in names:
                continue
            raise ValueError(
                f"The query reads {qualified!r}, which is not a table of this source. "
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


def _url_keys(spec: Any) -> bool:
    if isinstance(spec, dict):
        return any(key in URL_KEYS or _url_keys(value) for key, value in spec.items())
    if isinstance(spec, list):
        return any(_url_keys(item) for item in spec)
    return False


def check_chart(spec: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """
    The spec to render for a chart of a table with ``columns``, or
    ``ValueError`` saying what to fix. The data comes from the node's input,
    so ``data`` and ``datasets`` are dropped, and nothing may name a URL.
    """
    from ..ai.utils import _spec_field_references, normalize_vegalite_spec

    if not isinstance(spec, dict):
        raise ValueError("The spec must be a Vega-Lite JSON object.")
    spec = copy.deepcopy(spec)
    data = spec.pop("data", None)
    spec.pop("datasets", None)
    if isinstance(data, dict) and ("url" in data or "values" in data):
        raise ValueError("The chart's data comes from the data node; remove `data` from the spec.")
    if _url_keys(spec):
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
