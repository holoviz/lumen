"""
Data checks: a rule the data should keep, as SQL or in plain language,
and the rows that break it. A failed check fails an ``assert``, so checks
run in pytest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlglot

from pydantic import Field, create_model

from ..sources import Source
from ..sources.base import BaseSQLSource
from ..transforms.sql import SQLLimit
from .agents.sql import validate_read_only_sql
from .answer import (
    _as_input_type, _default_llm, _read_kinds, _run,
)
from .llm import Llm
from .sharing import sharing
from .utils import format_error

#: Offending rows kept on a result; count has them all
MAX_ROWS = 1_000


@dataclass(frozen=True)
class CheckResult:
    """A rule checked against the data, with the rows that break it."""

    rule: str
    #: The query that returns the rows breaking the rule
    sql: str
    #: How many rows break the rule
    count: int
    #: Up to MAX_ROWS of them, in the type the data was read in
    rows: Any
    _preview: str = field(default="", repr=False)

    @property
    def passed(self) -> bool:
        return self.count == 0

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        if self.passed:
            return f"CheckResult(passed: {self.rule})"
        return f"CheckResult(failed, {self.count:,} rows break: {self.rule})\n{self.sql}\n{self._preview}"

    def raise_(self) -> CheckResult:
        """Raise an AssertionError naming the rule and the rows that break it, if any do."""
        if not self.passed:
            raise AssertionError(repr(self))
        return self


def _is_sql(rule: str, dialect: str) -> bool:
    """Whether the rule is written as SQL rather than in plain language."""
    try:
        statements = sqlglot.parse(rule, read=dialect)
    except sqlglot.errors.ParseError:
        return False
    # Plain words can parse too, as a column with an alias or a comparison
    # such as "mass > 0"; only a statement is SQL, and a write is refused later
    return any(isinstance(statement, (
        sqlglot.exp.Query, sqlglot.exp.Pivot, sqlglot.exp.DML, sqlglot.exp.DDL, sqlglot.exp.Drop,
        sqlglot.exp.Alter, sqlglot.exp.Command,
    )) for statement in statements)


async def _rule_sql(source: Source, rule: str, table: str | None, llm: Llm) -> str:
    """One LLM call writing the query that returns the rows breaking a rule."""
    tables = [table] if table else source.get_tables()
    # share="none" allows names only
    typed = sharing.share != "none"
    schemas = "\n".join(
        f"{name}: " + ", ".join(
            f"{column} ({prop.get('type')})" if typed else column
            for column, prop in source.get_schema(name).items() if column != "__len__"
        )
        for name in tables
    )
    model = create_model("Check", query=(str, Field(description=(
        f"One {source.dialect} SELECT returning every row that breaks the rule, and no rows when the rule holds."
    ))))
    out = await llm.invoke(
        [{"role": "user", "content": f"Rule: {rule}\n\nTables and columns:\n{schemas}"}],
        system="Write the SQL query that finds the rows breaking the rule.", response_model=model,
    )
    return out.query


async def acheck(source: Source, rule: str, table: str | None = None, *, llm: Llm | None = None) -> CheckResult:
    """
    Check a rule against the data and return the rows that break it.

    rule is a SELECT returning the offending rows, run as written, or a rule
    in plain language, e.g. "no penguin weighs under 2 kg", which one LLM
    call turns into that SELECT; result.sql holds it, so a test can pin it.
    table limits a plain-language rule to one table.
    """
    if not isinstance(source, BaseSQLSource):
        raise ValueError("check runs SQL, so it needs a SQL source, such as one from lumen.read.")
    dialect = source.dialect
    sql = rule if _is_sql(rule, dialect) else await _rule_sql(source, rule, table, llm or _default_llm())
    # A trailing semicolon would end the query inside the wrappers below
    sql = sql.strip().rstrip(";").strip()
    validate_read_only_sql(sql, dialect)
    limited = SQLLimit(limit=MAX_ROWS, write=dialect).apply(f"SELECT * FROM ({sql}) AS broken")
    try:
        count = int(source.execute(f"SELECT COUNT(*) FROM ({sql}) AS broken").iloc[0, 0])
        rows = source.execute(limited)
    except Exception as e:
        raise ValueError(f"The check's query failed: {format_error(e)}\n{sql}") from e
    return CheckResult(
        rule=rule, sql=sql, count=count, rows=_as_input_type(rows, _read_kinds(source, sql)),
        _preview=rows.head(10).to_string(index=False),
    )


def check(source: Source, rule: str, table: str | None = None, *, llm: Llm | None = None) -> CheckResult:
    """Check a rule against the data. See acheck."""
    return _run(acheck(source, rule, table, llm=llm), "Checking the data")
