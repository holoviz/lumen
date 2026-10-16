"""Opt-in execution-accuracy evaluation on BIRD Mini-Dev SQLite databases."""

import json
import multiprocessing
import sqlite3
import time

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


def _execute_sql(path: Path, sql: str, timeout: float) -> set[tuple]:
    """Match Mini-Dev EX set semantics without modifying the database."""
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5) as connection:
        connection.execute("PRAGMA query_only=ON")
        def progress():
            return time.monotonic() >= deadline

        deadline = time.monotonic() + timeout
        connection.set_progress_handler(progress, 1000)
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise ValueError("BIRD prediction must return rows")
        rows = set(cursor.fetchall())
        connection.set_progress_handler(None, 0)
        return rows


def _score_query(path: Path, sql: str, timeout: float, output):
    try:
        output.send((True, _execute_sql(path, sql, timeout)))
    except Exception as exc:
        output.send((False, (type(exc).__name__, str(exc))))
    finally:
        output.close()


# The spawned scorer re-imports this module, which takes seconds on a loaded
# machine. The SQL deadline is enforced inside the child, so the parent only
# needs to outlast it plus that startup.
SCORER_STARTUP_GRACE = 120


def execute_read_only(path: Path, sql: str, timeout: float = 30) -> set[tuple]:
    """Run untrusted SQL with a wall-clock deadline in a terminable process."""
    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.get_context("spawn").Process(target=_score_query, args=(path, sql, timeout, sender))
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout + SCORER_STARTUP_GRACE):
            raise TimeoutError(f"BIRD SQL exceeded {timeout:g}s")
        valid, result = receiver.recv()
        if not valid:
            name, message = result
            if name == "OperationalError" and message == "interrupted":
                raise TimeoutError(f"BIRD SQL exceeded {timeout:g}s")
            raise sqlite3.OperationalError(f"{name}: {message}")
        return result
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
        process.join()


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
        try:
            reference = execute_read_only(path, gold)
        except (sqlite3.Error, ValueError, TimeoutError) as exc:
            # Raising records an evaluator failure and leaves the case without an
            # execution_accuracy result, so it counts as unscorable, not as a model miss.
            raise RuntimeError(f"Gold SQL failed: {type(exc).__name__}: {exc}") from exc
        try:
            predicted = execute_read_only(path, sql)
        except (sqlite3.Error, ValueError, TimeoutError):
            return {"execution_accuracy": False}
        return {"execution_accuracy": predicted == reference}
