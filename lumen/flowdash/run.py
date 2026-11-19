"""
Run a FlowDash dashboard of Lumen nodes without a page or a model: bind its
parameters and sources, check the data against the dashboard's schema
contracts, compute every node and report each one. What ``lumen run`` does.
"""
from __future__ import annotations

import copy
import os
import re

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps
from panel_flowdash.agent.yaml_format import (
    DashboardYAMLError, dashboard_from_dict, parse_yaml,
)
from panel_flowdash.parameters import (
    ContractProblem, ParameterError, bind_document, check_contracts,
)

from ..sources.base import Source
from .contracts import (
    SOURCE_ID, SourceContract, check_unrecorded, read_tables, source_dialect,
    source_schema,
)
from .nodes import COMPONENTS, SOURCES_KEY

OK, FAILED, CONTRACT, USAGE = 0, 1, 2, 3

_TABLE_ENTRY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$")

_CONTRACTS = {SOURCE_ID: SourceContract()}


class RunError(Exception):
    """The dashboard cannot be run as asked, e.g. a source is unbound or a value malformed."""


@dataclass
class RunReport:
    """What a run found: per node, its status and outputs, or the contract problems that stopped it."""

    status: str
    exit_code: int
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    problems: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"status": self.status}
        for key in ("message", "problems", "warnings", "questions", "nodes"):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data


def _entries(value: str) -> list[tuple[str | None, str]]:
    """``path``, or ``table=path`` entries separated by commas, as (table, data) pairs."""
    parts = value.split(",")
    named = [_TABLE_ENTRY.match(part.strip()) for part in parts]
    if all(named):
        return [(m[1], m[2]) for m in named]
    return [(None, value)]


def resolve_source(value: str | os.PathLike | Source, tables: list[str] | None = None) -> Source:
    """
    The source ``value`` names: a file, directory, URL or connection string as
    ``lumen serve`` reads them, or ``table=path`` entries separated by commas
    to name each table. A single file whose tables the dashboard reads
    exactly one of, ``tables``, gets that table's name.
    """
    from ..sources.data import resolve_data

    if isinstance(value, Source):
        return value
    entries = _entries(str(value))
    if entries[0][0] is None:
        data: Any = str(value)
        if tables and len(tables) == 1 and os.path.isfile(data):
            data = {tables[0]: data}
    else:
        data = {table: path for table, path in entries}
    sources = resolve_data(data)
    if len(sources) != 1:
        raise RunError(f"{value!r} resolves to {len(sources)} sources; give one source per ref.")
    return sources[0]


def _problem(name: str | None, problem: ContractProblem | Any) -> dict[str, Any]:
    entry = {"node": problem.node, "message": problem.message}
    if name:
        entry["parameter"] = name
    return entry


def _source_nodes(doc: Mapping[str, Any]) -> dict[str, str]:
    """The source nodes of a document and the ref each names."""
    return {
        name: str((node.get("config") or {}).get("ref") or "")
        for name, node in (doc.get("nodes") or {}).items()
        if isinstance(node, Mapping) and node.get("component") == SOURCE_ID
    }


def _questions(doc: Mapping[str, Any], nodes: list[str]) -> list[str]:
    """The questions of the turns that created ``nodes``, to ask again of the new data."""
    origins = {(doc.get("nodes") or {}).get(n, {}).get("origin") for n in nodes}
    return [
        turn["question"] for turn in doc.get("turns") or []
        if turn.get("id") in origins and turn.get("question")
    ]


def run_dashboard(text: str | bytes, params: Mapping[str, Any] | None = None,
                  sources: Mapping[str, str | os.PathLike | Source] | None = None,
                  rows: int = 5) -> RunReport:
    """
    Run the dashboard ``text`` with ``params`` bound and ``sources`` given to
    its source nodes by ref, and report each node.

    A parameter of type ``source`` takes anything ``resolve_source`` reads,
    or a Lumen source; the source is given to the dashboard under the
    parameter's name. Every other value is read as its parameter's type.
    Before anything computes, each source is checked against the contract
    its parameter recorded or, for a source given by ref, against the
    tables and columns the nodes reading it name. A failure stops the run
    and names the nodes.
    """
    params, given = dict(params or {}), dict(sources or {})
    try:
        doc = parse_yaml(text)
        dashboard_from_dict(doc)
    except DashboardYAMLError as exc:
        return RunReport("usage", USAGE, message=f"Not a dashboard: {exc}")
    declared = doc.get("parameters") or {}
    bound: dict[str, Source] = {}
    values: dict[str, Any] = {}
    try:
        for name, value in params.items():
            parameter = declared.get(name)
            if parameter is not None and parameter.get("type") == "source":
                expects = parameter.get("expects") or {}
                tables = list(expects.get("tables") or {}) or read_tables(doc, parameter["node"])
                bound[name] = resolve_source(value, tables)
                values[name] = name
            else:
                values[name] = value
        for ref, value in given.items():
            nodes = [n for n, r in _source_nodes(doc).items() if r == ref]
            tables = sorted({t for n in nodes for t in read_tables(doc, n)})
            bound[ref] = resolve_source(value, tables)
        doc = bind_document(doc, values)
    except (ParameterError, RunError) as exc:
        return RunReport("usage", USAGE, message=str(exc))
    except Exception as exc:
        return RunReport("usage", USAGE, message=f"Could not read the data: {exc}")
    context = {SOURCES_KEY: bound}
    unbound = sorted({ref for ref in _source_nodes(doc).values() if ref and ref not in bound})
    if unbound:
        return RunReport("usage", USAGE, message=(
            "Bind the data of the dashboard's sources: "
            + ", ".join(f"--source {ref}=<data>" for ref in unbound) + "."
        ))

    problems, warnings = [], []
    found = check_contracts(doc, list(values), _CONTRACTS.get, context)
    for name, items in found.items():
        for problem in items:
            (warnings if problem.warning else problems).append(_problem(name, problem))
    # Sources nothing recorded a contract for are checked against what their nodes name.
    unrecorded = set(given) | {
        name for name in values
        if name in bound and not (declared[name].get("expects") or {}).get("tables")
    }
    for node, ref in _source_nodes(doc).items():
        if ref not in unrecorded:
            continue
        source = bound[ref]
        dialect = source_dialect(source)
        schema = source_schema(source, read_tables(doc, node, dialect))
        for problem in check_unrecorded(doc, node, schema, dialect):
            (warnings if problem.warning else problems).append(_problem(None, problem))
    if problems:
        affected = [p["node"] for p in problems if p["node"]]
        return RunReport("contract", CONTRACT, problems=problems, warnings=warnings,
                         questions=_questions(doc, affected),
                         message="The data does not fit the dashboard.")

    editor = FlowDash(components=COMPONENTS, notifications=False, inspect_outputs=True,
                      context=copy.copy(context), history=False)
    try:
        editor.import_dashboard(doc)
        report = DashboardOps(editor).preview(rows=rows)
    finally:
        editor.dispose()
    failed = report["status"] != "ok"
    return RunReport("failed" if failed else "ok", FAILED if failed else OK,
                     nodes=report["nodes"], warnings=warnings)


def format_report(report: RunReport) -> str:
    """The report as text, one line per node or problem."""
    lines = []
    if report.message:
        lines.append(report.message)
    for problem in report.problems:
        where = problem["node"] or problem.get("parameter") or "data"
        lines.append(f"  {where}: {problem['message']}")
    if report.questions:
        lines.append("Questions that built the affected nodes, to ask again of the new data:")
        lines += [f"  - {q}" for q in report.questions]
    for warning in report.warnings:
        lines.append(f"  warning: {warning['node'] or 'data'}: {warning['message']}")
    width = max((len(n) for n in report.nodes), default=0)
    for name, node in report.nodes.items():
        line = f"{name.ljust(width)}  {node['status']}"
        if node.get("message"):
            line += f": {node['message']}"
        for port, sample in (node.get("outputs") or {}).items():
            if isinstance(sample, dict) and sample.get("type") == "DataFrame":
                line += f"  {port}: {sample['rows']} rows x {len(sample['columns'])} columns"
        lines.append(line)
    return "\n".join(lines)
