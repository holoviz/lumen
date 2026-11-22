"""
Schema contracts of FlowDash dashboards: the tables, columns and column
types the nodes downstream of a ``Lumen/source`` node read, so the dashboard
can be rerun on other data and refuse data it cannot use before any query
runs.

The columns of each ``Lumen/sql`` node come from ``lumen.data.contracts``;
the fields of filter and view nodes on a ``Lumen/table`` node, or on a SQL
node that selects ``*``, are columns of that table. A node is covered fully
when every column it reads is known, partly when it reads a table with
``*``, and not at all when its SQL does not parse or its columns cannot be
resolved.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..data.contracts import (
    FULL, NONE, PARTIAL, query_columns, schema_problems, source_dialect,
    source_schema,
)
from ..sources.base import Source
from .defaults import field_column

# The ids of Lumen's FlowDash components, which saved dashboards refer to.
SOURCE_ID = "Lumen/source"
TABLE_ID = "Lumen/table"
FILTER_ID = "Lumen/filter"
SQL_ID = "Lumen/sql"
DATA_ID = "Lumen/data"
VIEW_ID = "Lumen/view"


@dataclass
class NodeUsage:
    """What one node downstream of a source reads from the source's tables."""

    node: str
    component: str
    coverage: str
    tables: dict[str, set[str]] = field(default_factory=dict)
    reason: str | None = None
    unresolved: set[str] = field(default_factory=set)

    def add(self, table: str, column: str | None = None):
        columns = self.tables.setdefault(table, set())
        if column is not None:
            columns.add(column)


@dataclass(frozen=True)
class Problem:
    node: str | None
    message: str
    warning: bool = False


def _edges(doc: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    edges = []
    for edge in doc.get("edges") or []:
        source, _, source_port = str(edge.get("from", "")).partition(".")
        target, _, target_port = str(edge.get("to", "")).partition(".")
        edges.append((source, source_port, target, target_port))
    return edges


def _resolver(schema: Mapping[str, Mapping[str, Any]]):
    tables = {name.lower(): name for name in schema}

    def table(name: str) -> str:
        return tables.get(name.lower(), name)

    def column(table_name: str, name: str) -> str:
        lowered = {c.lower(): c for c in schema.get(table(table_name)) or {}}
        return lowered.get(name.lower(), name)

    return table, column


def sql_usage(node: str, sql: str, schema: Mapping[str, Mapping[str, Any]],
              dialect: str | None = None) -> NodeUsage:
    """What the query of a ``Lumen/sql`` node reads, resolved against ``schema``."""
    found = query_columns(sql, schema, dialect)
    return NodeUsage(node, SQL_ID, found.coverage, found.tables, found.reason, found.unresolved)


def _spec_fields(spec: Any) -> set[str]:
    fields: set[str] = set()
    if isinstance(spec, dict):
        for key, value in spec.items():
            if key == "field" and isinstance(value, str):
                fields.add(field_column(value))
            else:
                fields |= _spec_fields(value)
    elif isinstance(spec, list):
        for item in spec:
            fields |= _spec_fields(item)
    return fields


def spec_fields(spec: Any) -> set[str]:
    """The columns a view spec reads: every string under a ``field`` key, at any depth."""
    return _spec_fields(spec)


def _view_fields(component: Any) -> Callable[[Any], set[str]] | None:
    """How the view node of ``component`` reads columns from its spec, or None if it is no view node."""
    from .nodes import COMPONENTS, LumenViewNode

    cls = COMPONENTS.get(component) if isinstance(component, str) else None
    if isinstance(cls, type) and issubclass(cls, LumenViewNode):
        return cls.spec_fields
    return None


def _pipeline_readers(doc: Mapping[str, Any], node: str) -> list[str]:
    """The filter, data and view nodes fed by ``node``'s pipeline, through filter nodes."""
    nodes = doc.get("nodes") or {}
    edges = _edges(doc)
    found, queue = [], [node]
    while queue:
        current = queue.pop(0)
        for source, _, target, _ in edges:
            if source != current or target in found:
                continue
            component = (nodes.get(target) or {}).get("component")
            if component in (FILTER_ID, DATA_ID) or _view_fields(component) is not None:
                found.append(target)
                if component == FILTER_ID:
                    queue.append(target)
    return found


def _reader_fields(doc: Mapping[str, Any], node: str) -> dict[str, set[str]]:
    nodes = doc.get("nodes") or {}
    fields = {}
    for reader in _pipeline_readers(doc, node):
        config = (nodes.get(reader) or {}).get("config") or {}
        component = nodes[reader].get("component")
        if component == FILTER_ID and config.get("field"):
            fields[reader] = {config["field"]}
        elif (read := _view_fields(component)) is not None:
            fields[reader] = read(config.get("spec") or {})
    return fields


def usages(doc: Mapping[str, Any], source_node: str, schema: Mapping[str, Mapping[str, Any]],
           dialect: str | None = None) -> list[NodeUsage]:
    """
    What each node reading ``source_node`` directly reads of its tables, and the
    nodes that read those nodes' output where it passes table columns through:
    filters and views on a table node, or on a SQL node that selects ``*``.
    """
    nodes = doc.get("nodes") or {}
    table_name, column_name = _resolver(schema)
    found = []
    for source, _, target, _ in _edges(doc):
        if source != source_node:
            continue
        node = nodes.get(target) or {}
        component, config = node.get("component"), node.get("config") or {}
        if component == SQL_ID:
            usage = sql_usage(target, config.get("sql") or "", schema, dialect)
            found.append(usage)
            if usage.coverage != PARTIAL:
                continue
            starred = list(usage.tables)
            for reader, fields in _reader_fields(doc, target).items():
                reader_usage = NodeUsage(reader, nodes[reader]["component"], PARTIAL,
                                         reason=f"It reads {target}, which selects *.")
                for name in fields:
                    for table in starred:
                        if name.lower() in {c.lower() for c in schema.get(table) or {}}:
                            reader_usage.add(table, column_name(table, name))
                if reader_usage.tables:
                    found.append(reader_usage)
        elif component == TABLE_ID:
            table = table_name(config.get("table") or "")
            if not table:
                continue
            usage = NodeUsage(target, TABLE_ID, FULL)
            usage.add(table)
            found.append(usage)
            for reader, fields in _reader_fields(doc, target).items():
                reader_usage = NodeUsage(reader, nodes[reader]["component"], FULL)
                for name in fields:
                    reader_usage.add(table, column_name(table, name))
                found.append(reader_usage)
    return found


def expects(doc: Mapping[str, Any], source_node: str, schema: Mapping[str, Mapping[str, str]],
            dialect: str | None = None) -> dict[str, Any]:
    """
    The contract of ``source_node``: the tables its downstream nodes read and,
    for each, the columns they read with the family ``schema`` gives them.
    """
    tables: dict[str, dict[str, str]] = {}
    for usage in usages(doc, source_node, schema, dialect):
        for table, columns in usage.tables.items():
            entry = tables.setdefault(table, {})
            for column in columns:
                entry[column] = (schema.get(table) or {}).get(column, "other")
    return {"tables": {t: dict(sorted(c.items())) for t, c in sorted(tables.items())}}


def check(doc: Mapping[str, Any], source_node: str, contract: Mapping[str, Any],
          schema: Mapping[str, Mapping[str, str]], dialect: str | None = None) -> list[Problem]:
    """
    The problems of a source whose tables have ``schema`` against
    ``contract``, each naming the node that reads the missing or changed
    table or column. Nodes the contract does not cover fully are warnings.
    """
    expected = contract.get("tables") or {}
    # The nodes' columns are resolved against the schema they were recorded with.
    node_usages = usages(doc, source_node, expected, dialect)
    problems: list[Problem] = []
    for problem in schema_problems(expected, schema):
        readers = [
            u.node for u in node_usages if problem.table in u.tables
            and (problem.column is None or problem.column in u.tables[problem.table])
        ]
        problems += [Problem(node, problem.message, problem.warning) for node in readers] or [
            Problem(None, problem.message, problem.warning)
        ]
    for usage in node_usages:
        if usage.coverage != FULL:
            problems.append(Problem(usage.node, f"Not fully covered by the contract: {usage.reason}",
                                    warning=True))
    return problems


def check_unrecorded(doc: Mapping[str, Any], source_node: str,
                     schema: Mapping[str, Mapping[str, str]], dialect: str | None = None) -> list[Problem]:
    """
    The problems of a source with tables ``schema`` for the nodes reading
    ``source_node`` when no contract was recorded: the tables and columns the
    nodes read, resolved against ``schema`` itself, must exist. Types are not
    checked, since nothing recorded what they were.
    """
    lowered = {t.lower(): t for t in schema}
    problems: list[Problem] = []
    for usage in usages(doc, source_node, schema, dialect):
        if usage.coverage == NONE and not usage.unresolved:
            problems.append(Problem(usage.node, f"Not covered by the contract: {usage.reason}", warning=True))
        read = {c.lower() for t in usage.tables for c in schema.get(lowered.get(t.lower(), t), {})}
        for column in sorted(usage.unresolved):
            if column.lower() not in read:
                problems.append(Problem(usage.node, f"No table it reads has a column {column!r}."))
        for table, columns in usage.tables.items():
            actual = table if table in schema else lowered.get(table.lower())
            if actual is None:
                problems.append(Problem(usage.node, f"The data has no table {table!r}."))
                continue
            names = {c.lower() for c in schema[actual]}
            for column in sorted(columns):
                if column.lower() not in names:
                    problems.append(Problem(usage.node, f"Table {table!r} has no column {column!r}."))
        if usage.coverage == PARTIAL:
            problems.append(Problem(usage.node, f"Not fully covered by the contract: {usage.reason}", warning=True))
    return problems


def read_tables(doc: Mapping[str, Any], source_node: str, dialect: str | None = None) -> list[str]:
    """The tables the nodes reading ``source_node`` name, found without any schema."""
    tables: list[str] = []
    for usage in usages(doc, source_node, {}, dialect):
        tables += [t for t in usage.tables if t not in tables]
    return tables


def source_contract(doc: Mapping[str, Any], source_node: str, source: Source) -> dict[str, Any]:
    """The contract of ``source_node`` with ``source`` bound to it, asking the source only about the tables read."""
    dialect = source_dialect(source)
    schema = source_schema(source, read_tables(doc, source_node, dialect))
    return expects(doc, source_node, schema, dialect)


class SourceContract:
    """
    The ``contract`` of ``Lumen/source`` nodes, see ``panel_flowdash.parameters``:
    a parameter on a source node's ``ref`` records the tables, columns and
    column types its downstream nodes read, and a source bound to it later
    must have them.
    """

    def _source(self, context: Any) -> Source | None:
        from .nodes import host_sources

        ref = context.value
        return host_sources(context.context).get(ref) if isinstance(ref, str) else None

    def expects(self, context: Any) -> dict[str, Any] | None:
        source = self._source(context)
        if source is None:
            return None
        return source_contract(context.document, context.parameter.node, source)

    def check(self, context: Any, expects: dict[str, Any]) -> list[Any]:
        from panel_flowdash.parameters import ContractProblem

        from .nodes import host_sources, missing_source

        source = self._source(context)
        if source is None:
            return [ContractProblem(None, missing_source(str(context.value), host_sources(context.context)))]
        tables = list(expects.get("tables") or {})
        problems = check(
            context.document, context.parameter.node, expects,
            source_schema(source, tables), source_dialect(source),
        )
        return [ContractProblem(p.node, p.message, p.warning) for p in problems]
