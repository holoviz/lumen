"""
Checks every query and chart spec of a node must pass, whoever wrote them:
the agent, a person editing the node, an imported file or a replayed trace.
"""
from __future__ import annotations

import asyncio
import copy
import re
import threading
import time
import weakref

from collections import OrderedDict
from typing import Any

import sqlglot

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.qualify import qualify
from sqlglot.tokens import TokenType

from ..data.sql import validate_read_only_sql
from ..data.utils import ANSI_ESCAPE, format_tool_error
from ..sources.base import BaseSQLSource, QueryTimeoutError
from ..transforms.sql import SQLLimit
from .view_settings import VEGA_LITE_SETTINGS

# Seconds the engine may take to plan a query in ``check_sql_config``, waiting
# for the source's previous probe included.
PROBE_TIMEOUT = 3

# Queries the engine accepted, remembered per source, so a change of a SQL
# node's limit or table does not ask the engine again about the same query.
PROBE_CACHE_SIZE = 32

_GUARD = threading.Lock()
_PROBE_LOCKS: weakref.WeakKeyDictionary[BaseSQLSource, threading.Lock] = weakref.WeakKeyDictionary()
_PROBED: weakref.WeakKeyDictionary[BaseSQLSource, OrderedDict[tuple[str, str], None]] = weakref.WeakKeyDictionary()

# Keys whose value makes a Vega-Lite chart fetch or link to something.
URL_KEYS = ("url", "href", "baseURL")

# Scalar functions that read the engine's settings, session or environment, or load code.
DENIED_FUNCTIONS = frozenset({
    "current_setting", "getenv", "load_extension", "current_user", "user", "session_user",
    "current_database", "current_catalog", "current_schema", "current_schemas", "version",
    "pragma_version", "pragma_platform", "duckdb_settings", "set_config", "getvariable",
})

_CANDIDATE_BINDINGS_RE = re.compile(r'Candidate bindings:\s*(?::\s*)?"([^"]+)"')
_LINE_BLOCK_RE = re.compile(
    r"(?m)^(?P<prefix>LINE\s+(?P<line>\d+):\s?)(?P<snippet>[^\n]*)\n(?P<carets>[ \t]*\^+)"
)
_LINE_COL_RE = re.compile(r"Line\s+(\d+),\s*Col:\s*(\d+)")


class CheckDiagnostic(str):
    """A config check error message with optional 1-based position and suggestion.

    Subclasses ``str`` so callers expecting a plain string from a check
    continue to work unchanged.
    """

    line: int | None
    column: int | None
    end_column: int | None
    field: str | None
    suggestion: str | None

    def __new__(
        cls,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
        end_column: int | None = None,
        field: str | None = None,
        suggestion: str | None = None,
    ) -> CheckDiagnostic:
        obj = super().__new__(cls, message)
        obj.line = line
        obj.column = column
        obj.end_column = end_column
        obj.field = field
        obj.suggestion = suggestion
        return obj

    def __repr__(self) -> str:
        attrs = []
        for name in ("line", "column", "end_column", "field", "suggestion"):
            value = getattr(self, name)
            if value is not None:
                attrs.append(f"{name}={value!r}")
        suffix = f", {', '.join(attrs)}" if attrs else ""
        return f"CheckDiagnostic({super().__repr__()}{suffix})"


class _DiagnosticError(ValueError):
    """``ValueError`` carrying a :class:`CheckDiagnostic` for config checks."""

    def __init__(self, diagnostic: CheckDiagnostic) -> None:
        super().__init__(str(diagnostic))
        self.diagnostic = diagnostic


def _offset_to_line_col(text: str, start: int, end: int | None = None) -> tuple[int, int, int | None]:
    before = text[:start]
    line = before.count("\n") + 1
    line_start = before.rfind("\n") + 1
    column = start - line_start + 1
    if end is None:
        return line, column, None
    span = text[start:end]
    first_line_len = span.index("\n") if "\n" in span else len(span)
    return line, column, column + first_line_len


def _token_char_spans(
    text: str,
    target: str,
    *,
    dialect: str | None = None,
    is_call: bool = False,
    is_table: bool = False,
    allow_string: bool = False,
) -> list[tuple[int, int]]:
    if not target:
        return []
    try:
        tokens = sqlglot.tokenize(text, read=dialect)
    except sqlglot.errors.SqlglotError:
        return []
    if allow_string:
        string_spans = [
            (tok.start, tok.end + 1)
            for tok in tokens
            if tok.token_type == TokenType.STRING and tok.text == target
        ]
        if string_spans:
            return string_spans
    try:
        pattern = sqlglot.tokenize(target, read=dialect)
    except sqlglot.errors.SqlglotError:
        pattern = []
    if not pattern:
        return []

    def match(pat: list[Any], *, require_call: bool = False) -> list[tuple[int, int]]:
        n = len(pat)
        preferred: list[tuple[int, int]] = []
        fallback: list[tuple[int, int]] = []
        for i in range(len(tokens) - n + 1):
            window = tokens[i : i + n]
            if not all(
                a.text.lower() == b.text.lower()
                and ((a.token_type == TokenType.STRING) == (b.token_type == TokenType.STRING))
                for a, b in zip(window, pat, strict=True)
            ):
                continue
            if require_call and not (i + n < len(tokens) and tokens[i + n].token_type == TokenType.L_PAREN):
                continue
            span = (window[0].start, window[-1].end + 1)
            if is_call and i + n < len(tokens) and tokens[i + n].token_type == TokenType.L_PAREN:
                preferred.append(span)
            elif is_table and i > 0 and tokens[i - 1].token_type in (TokenType.FROM, TokenType.JOIN):
                preferred.append(span)
            else:
                fallback.append(span)
        return preferred or fallback

    spans = match(pattern)
    if not spans and len(pattern) > 1 and pattern[1].token_type == TokenType.L_PAREN:
        spans = match([pattern[0]], require_call=True)
    return spans


def _locate_token(
    sql: str,
    target: str,
    *,
    dialect: str | None = None,
    occurrence: int = 0,
    is_call: bool = False,
    is_table: bool = False,
    allow_string: bool = False,
) -> tuple[int | None, int | None, int | None]:
    spans = _token_char_spans(
        sql,
        target,
        dialect=dialect,
        is_call=is_call,
        is_table=is_table,
        allow_string=allow_string,
    )
    if not spans:
        return None, None, None
    start, end = spans[occurrence] if occurrence < len(spans) else spans[0]
    return _offset_to_line_col(sql, start, end)


def _sqlglot_diagnostic(
    exc: sqlglot.errors.SqlglotError,
    sql: str,
    *,
    prefix: str = "",
) -> CheckDiagnostic:
    clean = f"{prefix}{ANSI_ESCAPE.sub('', str(exc))}"
    errors = getattr(exc, "errors", None)
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        err = errors[0]
        line = err.get("line")
        col = err.get("col")
        highlight = err.get("highlight") or ""
        if isinstance(line, int) and isinstance(col, int):
            column = max(1, col - len(highlight) + 1) if highlight else col
            end_column = col + 1 if highlight else None
            return CheckDiagnostic(clean, line=line, column=column, end_column=end_column, field="sql")
    match = _LINE_COL_RE.search(clean)
    if match:
        return CheckDiagnostic(
            clean,
            line=int(match.group(1)),
            column=int(match.group(2)),
            field="sql",
        )
    return CheckDiagnostic(clean, field="sql")


def _line_char_offset(text: str, line_no: int) -> int | None:
    if line_no <= 1:
        return 0
    lines = text.splitlines(keepends=True)
    if line_no > len(lines):
        return None
    return sum(len(line) for line in lines[: line_no - 1])


def extract_sql_error_location(
    message: str,
    sql: str | None = None,
    *,
    probe: str | None = None,
    dialect: str | None = None,
) -> CheckDiagnostic:
    """Parse 1-based position and suggestion from a SQL engine error message.

    Strips any ``LINE n:`` caret block from the returned message while keeping
    the recovered ``line``, ``column``, ``end_column`` and ``suggestion`` on
    the returned :class:`CheckDiagnostic`. When ``probe`` differs from ``sql``
    (e.g. after ``SQLLimit`` rewrites the query), the offending token is
    mapped back to its occurrence in ``sql``.
    """
    text = ANSI_ESCAPE.sub("", message)
    suggestion_match = _CANDIDATE_BINDINGS_RE.search(text)
    suggestion = suggestion_match.group(1) if suggestion_match else None
    clean = re.split(r"\n+LINE\s+\d+:", text, maxsplit=1)[0].strip()

    line: int | None = None
    column: int | None = None
    end_column: int | None = None

    block = _LINE_BLOCK_RE.search(text)
    if block:
        line_no = int(block.group("line"))
        prefix_len = len(block.group("prefix"))
        snippet = block.group("snippet")
        carets = block.group("carets")
        caret_start = len(carets) - len(carets.lstrip(" \t"))
        caret_end = len(carets)
        rel_start = caret_start - prefix_len
        rel_end = caret_end - prefix_len
        if rel_end - rel_start == 1 and 0 <= rel_start < len(snippet):
            # DuckDB before 2.0 puts a single caret under the first character of the token.
            word = re.match(r'"[^"]*"|\w+', snippet[rel_start:])
            if word:
                rel_end = rel_start + len(word.group())
        token = snippet[rel_start:rel_end] if 0 <= rel_start <= len(snippet) else ""

        if sql is None or probe is None or probe == sql:
            if not snippet.startswith("...") and rel_start >= 0:
                line = line_no
                column = rel_start + 1
                end_column = rel_end + 1 if rel_end > rel_start else None
            elif sql is not None and token:
                line_text = sql.splitlines()[line_no - 1] if 0 < line_no <= len(sql.splitlines()) else ""
                spans = _token_char_spans(line_text, token, dialect=dialect) if line_text else []
                if spans:
                    line = line_no
                    column = spans[0][0] + 1
                    end_column = spans[0][1] + 1
                else:
                    line, column, end_column = _locate_token(sql, token, dialect=dialect)
        elif token:
            probe_offset: int | None = None
            if not snippet.startswith("...") and rel_start >= 0:
                base = _line_char_offset(probe, line_no)
                if base is not None:
                    probe_offset = base + rel_start
            elif snippet.startswith("...") and rel_start >= 3:
                needle = snippet[3:rel_end]
                idx = probe.find(needle)
                if idx != -1:
                    probe_offset = idx + (rel_start - 3)
            occurrence = 0
            if probe_offset is not None:
                probe_spans = _token_char_spans(probe, token, dialect=dialect)
                occurrence = sum(1 for start, _end in probe_spans if start < probe_offset)
            line, column, end_column = _locate_token(
                sql, token, dialect=dialect, occurrence=occurrence
            )
    else:
        match = _LINE_COL_RE.search(clean)
        if match:
            line = int(match.group(1))
            column = int(match.group(2))

    return CheckDiagnostic(
        clean,
        line=line,
        column=column,
        end_column=end_column,
        field="sql",
        suggestion=suggestion,
    )


def _adjust_leading_whitespace(
    diag: CheckDiagnostic,
    raw_sql: str,
    *,
    field: str | None = None,
) -> CheckDiagnostic:
    target_field = field if field is not None else diag.field
    leading = raw_sql[: len(raw_sql) - len(raw_sql.lstrip())]
    if not leading or diag.line is None:
        if target_field == diag.field:
            return diag
        return CheckDiagnostic(
            str(diag),
            line=diag.line,
            column=diag.column,
            end_column=diag.end_column,
            field=target_field,
            suggestion=diag.suggestion,
        )
    line_offset = leading.count("\n")
    col_offset = len(leading.rsplit("\n", 1)[-1]) if diag.line == 1 else 0
    return CheckDiagnostic(
        str(diag),
        line=diag.line + line_offset,
        column=None if diag.column is None else diag.column + col_offset,
        end_column=None if diag.end_column is None else diag.end_column + col_offset,
        field=target_field,
        suggestion=diag.suggestion,
    )


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
    dialect = _read(source.dialect)
    parsed = sqlglot.parse_one(sql, read=dialect)
    ctes = {cte.alias_or_name.lower(): cte.this for cte in parsed.find_all(exp.CTE)}
    if not _reads_a_table(parsed, ctes):
        raise ValueError("The query reads no table. Select from the source's tables instead of returning literal values.")
    for func in parsed.find_all(exp.Func):
        func_name = _function_name(func)
        if func_name in DENIED_FUNCTIONS:
            line, column, end_column = _locate_token(sql, func_name, dialect=dialect, is_call=True)
            raise _DiagnosticError(CheckDiagnostic(
                f"The query calls {func_name}(), which reads the engine's settings or environment.",
                line=line,
                column=column,
                end_column=end_column,
                field="sql",
            ))
    names, functions = _relations(source)
    for table in parsed.find_all(exp.Table):
        if isinstance(table.this, exp.Identifier):
            if (not table.db and table.name.lower() in ctes) or _declared(table, names):
                continue
            qualified = ".".join(part.name for part in table.parts)
            line, column, end_column = _locate_token(
                sql,
                qualified,
                dialect=dialect,
                is_table=True,
                allow_string=table.this.quoted,
            )
            raise _DiagnosticError(CheckDiagnostic(
                f"The query reads {qualified.lower()!r}, which is not a table of this source. "
                f"Its tables are: {', '.join(sorted(source.get_tables()))}.",
                line=line,
                column=column,
                end_column=end_column,
                field="sql",
            ))
        if table.this.sql(dialect=dialect) not in functions:
            target = _function_name(table.this) if isinstance(table.this, exp.Func) else table.this.sql(dialect=dialect)
            line, column, end_column = _locate_token(sql, target, dialect=dialect, is_call=isinstance(table.this, exp.Func))
            raise _DiagnosticError(CheckDiagnostic(
                f"The query calls {table.this.sql(dialect=dialect)}; queries may only read the "
                f"source's tables: {', '.join(sorted(source.get_tables()))}.",
                line=line,
                column=column,
                end_column=end_column,
                field="sql",
            ))


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


# The ``check_config=`` hooks of the nodes, see ``panel_flowdash.register``:
# check(config, inputs), where ``config`` is the node's whole config with the
# change applied and ``inputs`` its live input values that are set and not
# failed. A config that fails is refused in the inspector and by the agent
# tools before it is applied. None of them fetches rows.

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


def _folded(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Identifier):
        return exp.Identifier(this=node.this.lower(), quoted=node.quoted)
    return node


def check_columns(sql: str, source: BaseSQLSource) -> None:
    """
    Raise ``ValueError`` if ``sql`` names a column that none of the tables it
    reads has, using the schemas ``source`` already stored. Nothing is asked
    of the engine, so it checks only queries whose tables all have a stored
    schema. Names are compared case-insensitively whatever the dialect's
    rules, so a column is only refused when no column of that name exists
    in any case; the engine reports a column of the wrong case when the
    query runs.
    """
    stored = source._get_schema_cache() or {}
    parsed = sqlglot.parse_one(sql, read=_read(source.dialect))
    ctes = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    schema = {}
    for table in parsed.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier) or table.db:
            return
        name = table.name.lower()
        if name in ctes:
            continue
        columns = next((value for key, value in stored.items() if key.lower() == name), None)
        if not isinstance(columns, dict):
            return
        schema[name] = {str(column).lower(): "UNKNOWN" for column in columns if column != "__len__"}
    try:
        qualify(parsed.transform(_folded), schema=schema, validate_qualify_columns=True)
    except OptimizeError as exc:
        if "could not be resolved" in str(exc):
            message = f"{str(exc).split('. Line:')[0]}."
            match = re.search(r"Column '([^']+)' could not be resolved", message)
            line, column, end_column = (None, None, None)
            if match:
                line, column, end_column = _locate_token(
                    sql, match.group(1), dialect=_read(source.dialect)
                )
            raise _DiagnosticError(CheckDiagnostic(
                message,
                line=line,
                column=column,
                end_column=end_column,
                field="sql",
            )) from exc


def _engine_error(
    exc: Exception, probe: str, sql: str, dialect: str | None = None
) -> CheckDiagnostic:
    """The engine's error for ``probe``, as if ``sql`` itself had been run."""
    raw_full = ANSI_ESCAPE.sub("", str(exc))
    diag = extract_sql_error_location(raw_full, sql=sql, probe=probe, dialect=dialect)
    text = format_tool_error(exc).replace(probe, sql)
    clean = re.split(r"\n+LINE\s+\d+:", text, maxsplit=1)[0].strip()
    return CheckDiagnostic(
        f"The engine refused the query: {clean}",
        line=diag.line,
        column=diag.column,
        end_column=diag.end_column,
        field="sql",
        suggestion=diag.suggestion,
    )


async def probe_query(sql: str, source: BaseSQLSource, timeout: float | None = None) -> str | None:
    """
    Ask ``source``'s engine to bind and plan ``sql`` without returning rows,
    by running it with ``LIMIT 0``, and return the engine's error or None.

    Probes of one source run one at a time. ``check_sql_config`` only
    probes sources that stop a statement when it times out
    (``cancels_queries``), so a probe never keeps running in a worker
    thread. A probe that does not
    finish within ``timeout`` seconds (``PROBE_TIMEOUT`` by default), waiting
    for the source's previous probe included, raises ``QueryTimeoutError``.
    Queries the engine accepted are remembered until the source's table
    definitions change or ``forget_probes`` is called.
    """
    timeout = PROBE_TIMEOUT if timeout is None else timeout
    probe = SQLLimit(limit=0, write=source.dialect, identify=False).apply(sql)
    # A table the host redefines makes the queries accepted before unknown again.
    key = (sql, repr(source.tables))
    with _GUARD:
        probed = _PROBED.setdefault(source, OrderedDict())
        lock = _PROBE_LOCKS.setdefault(source, threading.Lock())
        if key in probed:
            probed.move_to_end(key)
            return None
    deadline = time.monotonic() + timeout
    while not lock.acquire(blocking=False):
        if time.monotonic() >= deadline:
            raise QueryTimeoutError(f"The engine did not get to the query within {timeout} seconds.")
        await asyncio.sleep(0.02)
    try:
        await source.execute_with_timeout(probe, max(deadline - time.monotonic(), 0.01))
    except QueryTimeoutError:
        raise
    except Exception as exc:
        problem = _engine_error(exc, probe, sql, dialect=_read(source.dialect))
    else:
        problem = None
    finally:
        lock.release()
    if problem is None:
        with _GUARD:
            probed[key] = None
            while len(probed) > PROBE_CACHE_SIZE:
                probed.popitem(last=False)
    return problem


def forget_probes(source: BaseSQLSource) -> None:
    """Forget the queries ``source``'s engine accepted, e.g. when the dashboard refreshes."""
    with _GUARD:
        _PROBED.pop(source, None)


def check_unbound_sql(sql: str) -> None:
    """
    Raise ``ValueError`` unless ``sql`` is one read-only query in the
    dialect of at least one source the host provides, for a SQL node no
    source is connected to yet. Without host sources the dialect is unknown,
    so SQL that does not parse generically is left to the check that runs
    once a source is connected.
    """
    from .nodes import host_sources

    dialects = sorted({
        source.dialect or "any" for source in host_sources().values() if isinstance(source, BaseSQLSource)
    })
    errors: list[Exception] = []
    for dialect in dialects or ["any"]:
        try:
            validate_read_only_sql(sql, dialect)
        except ValueError:
            raise
        except sqlglot.errors.SqlglotError as exc:
            errors.append(exc)
        else:
            return
    if dialects:
        raise _DiagnosticError(
            _sqlglot_diagnostic(errors[0], sql, prefix="The query does not parse: ")
        ) from errors[0]


def _unverified(message: str) -> Any:
    """A warning that does not refuse the config, on panel-flowdash versions that have one."""
    try:
        from panel_flowdash import CheckWarning
    except ImportError:
        return None
    return CheckWarning(message)


async def check_sql_config(config: dict[str, Any], inputs: dict[str, Any]) -> Any:
    """
    Check a ``Lumen/sql`` config without running the full query.

    Without a source only the SQL itself is checked: it must parse and be one
    read-only query. With one, ``check_query`` checks it against the
    source's tables. A source that can stop a statement (``cancels_queries``)
    then binds it with ``probe_query``; if the engine does not answer in
    time, the query could not be verified, which is a warning rather than
    a problem. Other sources are never probed, since a slow probe would keep
    running; there ``check_columns`` compares the columns with the stored
    schemas instead.
    """
    raw_sql = config.get("sql") or ""
    sql = raw_sql.strip()
    if not sql:
        return None
    source = inputs.get("source")
    try:
        if source is None:
            check_unbound_sql(sql)
            return None
        if not isinstance(source, BaseSQLSource):
            return CheckDiagnostic(f"{type(source).__name__} does not run SQL.", field="sql")
        await asyncio.to_thread(check_query, sql, source)
        if not source.cancels_queries:
            await asyncio.to_thread(check_columns, sql, source)
            return None
    except _DiagnosticError as exc:
        return _adjust_leading_whitespace(exc.diagnostic, raw_sql, field="sql")
    except sqlglot.errors.SqlglotError as exc:
        return _adjust_leading_whitespace(_sqlglot_diagnostic(exc, sql), raw_sql, field="sql")
    except ValueError as exc:
        return CheckDiagnostic(ANSI_ESCAPE.sub("", str(exc)), field="sql")
    try:
        problem = await probe_query(sql, source)
    except _DiagnosticError as exc:
        return _adjust_leading_whitespace(exc.diagnostic, raw_sql, field="sql")
    except sqlglot.errors.SqlglotError as exc:
        return _adjust_leading_whitespace(_sqlglot_diagnostic(exc, sql), raw_sql, field="sql")
    except ValueError as exc:
        return CheckDiagnostic(ANSI_ESCAPE.sub("", str(exc)), field="sql")
    except QueryTimeoutError:
        return _unverified(
            f"Could not verify the query: the engine did not answer within {PROBE_TIMEOUT} seconds. "
            "The node reports any error when it runs the query."
        )
    if problem is None:
        return None
    if isinstance(problem, CheckDiagnostic):
        return _adjust_leading_whitespace(problem, raw_sql, field="sql")
    return CheckDiagnostic(str(problem), field="sql")


def check_source_config(config: dict[str, Any], inputs: dict[str, Any]) -> str | None:
    """Check a ``Lumen/source`` config: the ref must name a source the host provides."""
    from .nodes import host_sources, missing_source

    ref = config.get("ref")
    if not ref:
        return None
    sources = host_sources()
    return None if ref in sources else missing_source(ref, sources)


def check_view_config(config: dict[str, Any], inputs: dict[str, Any]) -> str | None:
    """Check a ``Lumen/view`` config with ``check_chart`` against the input's columns."""
    pipeline, spec = inputs.get("pipeline"), config.get("spec")
    if not spec:
        return None
    try:
        from panel_flowdash.spec_settings import problems
    except ImportError:  # panel-flowdash without spec settings
        problems = None
    if problems is not None and (found := problems(spec, VEGA_LITE_SETTINGS)):
        return " ".join(found)
    if pipeline is None:
        return None
    try:
        check_chart(spec, pipeline_columns(pipeline))
    except _DiagnosticError as exc:
        return exc.diagnostic
    except ValueError as exc:
        return CheckDiagnostic(str(exc), field="spec")
    return None
