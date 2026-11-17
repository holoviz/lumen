"""
Recorded sessions as eval cases.

A session file holds the name of the fixture a conversation ran on and, per
turn, its question, the nodes it added or changed and the dashboard YAML after
it. The case it becomes asks the same questions on the same fixture, and its
expectations are computed by rerunning each turn's dashboard there, with no
model: the rows and columns of the turn's SQL node, the table its SQL reads
first and the mark of the last turn's chart.
"""
from __future__ import annotations

import os

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
import yaml

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, dashboard_from_yaml
from pydantic_evals import Case

from ..ai.evals.harness import MAX_ROWS, Expected, Inputs
from ..flowdash import (
    COMPONENTS, SOURCE_ID, bind_source, unbind_source,
)
from . import _graph
from .evals import answer_nodes, chart_summary, result_rows
from .headless import TurnResult

FORMAT_VERSION = 1


@dataclass
class SessionTurn:
    question: str
    nodes: list[str]
    dashboard: str


@dataclass
class Session:
    fixture: str
    turns: list[SessionTurn]

    @property
    def questions(self) -> list[str]:
        return [turn.question for turn in self.turns]


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(str, lambda dumper, value: dumper.represent_scalar(
    "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None))


def dump_session(session: Session) -> str:
    data = {
        "lumen_session": FORMAT_VERSION,
        "fixture": session.fixture,
        "turns": [{"question": t.question, "nodes": t.nodes, "dashboard": t.dashboard} for t in session.turns],
    }
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)


def save_session(path: str | os.PathLike, turns: list[TurnResult], fixture: str) -> None:
    """Write the turns of a conversation on ``fixture`` as a session file."""
    session = Session(fixture, [SessionTurn(t.prompt, [*t.nodes, *t.changed], t.graph) for t in turns])
    Path(path).write_text(dump_session(session), encoding="utf-8")


def load_session(path: str | os.PathLike) -> Session:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("lumen_session") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a version {FORMAT_VERSION} Lumen session file.")
    turns = [SessionTurn(t["question"], list(t["nodes"]), t["dashboard"]) for t in data["turns"]]
    return Session(data["fixture"], turns)


def rerun(dashboard: str, source: Any) -> FlowDash:
    """Load ``dashboard`` in a new editor with its source nodes bound to ``source``."""
    editor = FlowDash(components=COMPONENTS)
    doc = dashboard_from_yaml(dashboard)
    refs = [
        item["config"]["ref"] for item in doc["items"]
        if item["component_id"] == SOURCE_ID and (item.get("config") or {}).get("ref")
    ]
    for ref in refs:
        bind_source(source, ref)
    try:
        editor.import_dashboard(doc)
    finally:
        for ref in refs:
            unbind_source(ref)
    return editor


def _first_table(sql: str) -> str | None:
    tables = sorted(table.name for table in sqlglot.parse_one(sql).find_all(sqlglot.exp.Table))
    return tables[0].lower() if tables else None


def expected(session: Session, source: Any) -> Expected:
    """The expectations of a session, from each turn's dashboard rerun on ``source``."""
    turn_rows, tables = [], []
    rows = columns = mark = None
    for turn in session.turns:
        editor = rerun(turn.dashboard, source)
        ops = DashboardOps(editor)
        try:
            data, views = answer_nodes(ops, turn.nodes)
            rows = columns = None
            if data is not None:
                columns, rows, count = result_rows(_graph.output(ops, data, "data"))
                rows = rows if count <= MAX_ROWS else None
            turn_rows.append(rows)
            tables.append(_first_table(_graph.config_of(ops, data)["sql"]) if data is not None else None)
            marks, _ = chart_summary([_graph.config_of(ops, name).get("spec") or {} for name in views])
            mark = marks[-1] if marks else None
        finally:
            editor.dispose()
    return Expected(
        rows=rows, columns=columns, chart_mark=mark,
        turn_rows=turn_rows if any(r is not None for r in turn_rows) else None,
        turn_tables=tables if all(tables) else None,
    )


def session_case(path: str | os.PathLike, source_factory: Callable[[Inputs], Any]) -> Case:
    """A harness case asking the questions of the session at ``path`` on its fixture."""
    session = load_session(path)
    inputs = Inputs(session.questions, fixture=session.fixture)
    return Case(name=f"session_{Path(path).stem}", inputs=inputs, metadata=expected(session, source_factory(inputs)))
