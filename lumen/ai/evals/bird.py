"""Opt-in execution-accuracy evaluation on BIRD Mini-Dev SQLite databases."""

import json
import sqlite3

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from lumen.sources.sqlalchemy import SQLAlchemySource

from .harness import Expected, Inputs, Output


def download_questions(path: Path, url: str) -> None:
    """Fetch the revision-pinned, publicly licensed Mini-Dev question file."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing question file: {path}")
    questions = httpx.get(url, follow_redirects=True, timeout=15).raise_for_status().content
    records = json.loads(questions)
    if not isinstance(records, list) or not records or not all(key in records[0] for key in ("question_id", "db_id", "SQL", "question")):
        raise ValueError("Unexpected BIRD Mini-Dev question file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(questions)


def database_path(root: Path, db_id: str) -> Path:
    if not db_id.replace("_", "").isalnum():
        raise ValueError(f"Invalid BIRD database identifier: {db_id!r}")
    candidates = [root / db_id / f"{db_id}.sqlite", root / "dev_databases" / db_id / f"{db_id}.sqlite",
                  root / "minidev" / "MINIDEV" / "dev_databases" / db_id / f"{db_id}.sqlite"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"BIRD database {db_id!r} not found under {root}; extract Mini-Dev dev_databases first")


def execute_read_only(path: Path, sql: str) -> set[tuple]:
    """Match Mini-Dev EX set semantics without modifying the database."""
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5) as connection:
        connection.execute("PRAGMA query_only=ON")
        steps = 0

        def progress():
            nonlocal steps
            steps += 1
            return steps > 50_000

        connection.set_progress_handler(progress, 1000)
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise ValueError("BIRD prediction must return rows")
        rows = set(cursor.fetchall())
        connection.set_progress_handler(None, 0)
        return rows


def bird_source(path: Path) -> SQLAlchemySource:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro&uri=true"
    with sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True) as connection:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    return SQLAlchemySource(url=f"sqlite:///{uri}", tables=tables, name=path.stem)


@dataclass
class BirdExecution(Evaluator[Inputs, Output, Expected]):
    root: Path

    def evaluate(self, ctx: EvaluatorContext[Inputs, Output, Expected]) -> dict[str, bool]:
        gold = ctx.metadata.gold_sql
        if not gold or not ctx.output.turns:
            return {"execution_accuracy": False}
        sql = ctx.output.turns[-1].sql
        if not sql:
            return {"execution_accuracy": False}
        path = database_path(self.root, ctx.inputs.fixture.removeprefix("bird:"))
        reference = execute_read_only(path, gold)
        try:
            predicted = execute_read_only(path, sql)
        except (sqlite3.Error, ValueError, InterruptedError):
            return {"execution_accuracy": False}
        return {"execution_accuracy": predicted == reference}
