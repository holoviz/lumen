"""
Redaction of a component's spec before it is shown.

A source spec can carry credentials in many forms: a password in a
connection URI, API keys and tokens as settings, secrets passed to the
driver, SQL run on connect such as DuckDB's ``CREATE SECRET``, and values
read from environment variables. Rather than look for these, which cannot
cover every form, :func:`redact_spec` keeps the spec's structure and shows
only values known to be safe: names, flags, numbers of known settings,
locations rebuilt without their user info and query, and SQL with every
string literal and every statement that sets a value hidden. Everything
else reads ``***``.
"""
from __future__ import annotations

import logging
import os
import re
import threading

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import sqlglot

from sqlglot import exp

REDACTED = "***"

# Settings whose value is shown when it is a plain name, number or flag.
_SAFE_SETTINGS = {
    "type", "driver", "drivername", "dialect", "port", "schema", "warehouse",
    "filter_in_sql", "cache_per_query", "cache_schema", "cache_data",
    "cache_with_dask", "read_only", "ephemeral", "load_schema", "paramstyle",
    "schema_timeout_seconds", "geometry_crs", "materialize", "dataframe_backend",
}
# Settings whose value is a location, shown rebuilt without user info, query or fragment.
_LOCATION_SETTINGS = {"uri", "url", "host", "database", "cache_dir", "root"}
# Settings holding SQL, shown with every string literal hidden.
_SQL_SETTINGS = {"initializers", "sql_expr"}
# Settings keyed by table name; the names are shown and each table's definition is checked.
_TABLE_SETTINGS = {"tables", "excluded_tables", "table_params", "mirrors"}

# A key shown as it is: a lower case setting name, as Lumen's parameters are.
_SAFE_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# A plain name, e.g. a table, a dialect or a driver.
_NAME = re.compile(r"^[A-Za-z0-9_.+\-]{1,128}$")
_PATH = re.compile(r"^[A-Za-z0-9_.~/\\:+\-*%]+$")
_FILE_EXTENSION = re.compile(
    r"\.(csv|tsv|txt|parquet|json|jsonl|ndjson|xlsx|xls|arrow|feather|db|duckdb|sqlite|sqlite3|gz|zip|nc|zarr)$",
    re.IGNORECASE,
)
# Statements that set or store a value, e.g. ``SET s3_secret_access_key TO '...'``, shown by their keyword only.
_SETTING_STATEMENTS = (exp.Command, exp.Set, exp.Pragma)
# Statements shown with their string literals hidden; any other is hidden entirely, e.g. a bare expression.
_SHOWN_STATEMENTS = (exp.Query, exp.Install, exp.Attach, exp.Detach, exp.Create, exp.Insert, exp.Use)

# Environment variables whose value a path or a name commonly contains or equals.
_PLAIN_ENV = {
    "HOME", "PWD", "OLDPWD", "PATH", "USER", "USERNAME", "LOGNAME", "SHELL", "LANG",
    "TERM", "TMPDIR", "TEMP", "TMP", "HOSTNAME", "PYTHONPATH",
}
# Shorter values of environment variables not named like a credential are hidden only where they are
# a whole value, as they are too likely to be part of unrelated text.
_MIN_ENV_INSIDE = 6
# Environment variables named like a credential, whose value is hidden even inside a longer string.
_SECRET_ENV = re.compile(
    r"PASS|PW|SECRET|TOKEN|KEY|CRED|AUTH|SIG|SAS|CERT|PRIVATE|CODE|PIN|SESSION|COOKIE|DSN",
    re.IGNORECASE,
)


def _location(text: str) -> str | None:
    """``text`` as a URL rebuilt from its scheme, host, port and path, or as a plain path, or None."""
    if "://" in text:
        try:
            parts = urlsplit(text)
            port = parts.port
        except ValueError:
            return None
        host = parts.hostname or ""
        if not parts.scheme or not _PATH.match(parts.path or "/") or (host and not _NAME.match(host)):
            return None
        return f"{parts.scheme}://{host}{f':{port}' if port else ''}{parts.path}"
    path = text.split("?", 1)[0].split("#", 1)[0]
    if not path or not _PATH.match(path):
        return None
    return path


def _file_location(text: str) -> str | None:
    """``text`` rebuilt by :func:`_location` if it reads as a URL or a file path, e.g. a literal in SQL."""
    if "://" in text or "/" in text or "\\" in text or _FILE_EXTENSION.search(text.split("?", 1)[0]):
        return _location(text)
    return None


def _string_node(node: exp.Expression) -> bool:
    if isinstance(node, exp.Literal):
        return node.is_string
    return type(node).__name__.endswith("String")


def _hide_literal(node: exp.Expression) -> exp.Expression:
    if _string_node(node):
        value = node.name if isinstance(node, exp.Literal) else None
        location = _file_location(value) if value else None
        return exp.Literal.string(location or REDACTED)
    if isinstance(node, exp.Identifier) and node.quoted and not _NAME.match(node.name):
        location = _file_location(node.name)
        return exp.to_identifier(location or REDACTED, quoted=True)
    return node


class _QuietParse(logging.Filter):
    """Drop sqlglot's log records, which quote the SQL, while this module parses SQL that may hold secrets."""

    def __init__(self):
        super().__init__()
        self.local = threading.local()

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(self.local, "active", False)


_QUIET = _QuietParse()
logging.getLogger("sqlglot").addFilter(_QUIET)


def _parse(text: str, dialect: str | None) -> list[exp.Expression | None]:
    _QUIET.local.active = True
    try:
        return sqlglot.parse(text, read=dialect, error_level=sqlglot.ErrorLevel.RAISE)
    finally:
        _QUIET.local.active = False


def _redact_sql(text: str, dialect: str | None = "duckdb") -> str:
    """
    ``text`` parsed as SQL and written again with every string literal
    hidden, unless it is a location, comments dropped, and each statement
    that sets or stores a value shown by its keyword only. SQL that does not
    parse is hidden entirely.
    """
    try:
        statements = _parse(text, dialect)
    except Exception:
        return REDACTED
    out = []
    for statement in statements:
        if statement is None:
            continue
        if not isinstance(statement, _SHOWN_STATEMENTS) or statement.find(exp.Command) is not None:
            keyword = statement.this if isinstance(statement, exp.Command) else statement.key
            shown = isinstance(statement, _SETTING_STATEMENTS) and isinstance(keyword, str) and _NAME.match(keyword)
            out.append(f"{keyword.upper()} {REDACTED}" if shown else REDACTED)
            continue
        try:
            out.append(statement.transform(_hide_literal).sql(dialect=dialect, comments=False))
        except Exception:
            out.append(REDACTED)
    return "; ".join(out) if out else REDACTED


class _Redactor:

    def __init__(self, environ: Mapping[str, str]):
        plain = {
            name: value for name, value in environ.items()
            if isinstance(value, str) and name.upper() not in _PLAIN_ENV
        }
        self._whole = {value for value in plain.values() if len(value) >= 4}
        self._inside = [
            value for name, value in plain.items()
            if len(value) >= (3 if _SECRET_ENV.search(name) else _MIN_ENV_INSIDE)
        ]

    def shown(self, text: str | None) -> str:
        """``text`` unless it is None or holds an environment variable's value, ``***`` otherwise."""
        if text is None or text in self._whole or any(value in text for value in self._inside):
            return REDACTED
        return text

    def _key(self, key: Any, index: int) -> str:
        if isinstance(key, str) and _SAFE_KEY.match(key) and self.shown(key) == key:
            return key
        return f"{REDACTED}{index}"

    def _scalar(self, key: str, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value if key in _SAFE_SETTINGS else REDACTED
        if not isinstance(value, str):
            return REDACTED if isinstance(value, (bytes, bytearray)) else f"<{type(value).__name__}>"
        if key in _LOCATION_SETTINGS:
            return self.shown(_location(value))
        if key in _SQL_SETTINGS:
            return self.shown(_redact_sql(value))
        if key in _SAFE_SETTINGS and _NAME.match(value):
            return self.shown(value)
        return REDACTED

    def _table(self, value: Any) -> Any:
        """A table's definition: its name, a location, SQL with its literals hidden, or a nested spec."""
        if not isinstance(value, str):
            return self.value("", value)
        if _NAME.match(value):
            return self.shown(value)
        if ("://" in value or "/" in value or "\\" in value) and (location := _location(value)) is not None:
            return self.shown(location)
        return self.shown(_redact_sql(value))

    def _tables(self, key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            out = {}
            for index, (name, item) in enumerate(value.items()):
                shown = self.shown(name) if isinstance(name, str) and len(name) <= 128 else REDACTED
                out[shown if shown != REDACTED else f"{REDACTED}{index}"] = (
                    self._table(item) if key == "tables" else self.value("", item)
                )
            return out
        if isinstance(value, (list, tuple)):
            return [self._table(item) for item in value]
        return self.value(key, value)

    def value(self, key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                self._key(name, index): (
                    self._tables(name, item) if name in _TABLE_SETTINGS else self.value(str(name), item)
                )
                for index, (name, item) in enumerate(value.items())
            }
        if isinstance(value, (list, tuple, set)):
            return [self.value(key, item) for item in value]
        return self._scalar(key, value)


def redact_spec(spec: Any, environ: Mapping[str, str] | None = None) -> Any:
    """
    A copy of ``spec`` safe to show, in plain types YAML can dump.

    Keys are kept, except ones that are not lower case setting names, and
    values are shown only where known to be safe: flags, the names and
    numbers of known settings such as ``type``, ``port`` or ``schema``,
    locations (``uri``, ``url``, ``host``, ``database``) rebuilt from their
    scheme, host, port and path, table names, and SQL (``initializers``,
    table definitions) with every string literal hidden unless it is a
    location, and every statement that sets a value reduced to its keyword.
    Anything that equals an environment variable's value, or contains the
    value of one named like a credential, is hidden too. Every other value
    reads ``***``, and objects YAML cannot show are named by their type.
    """
    return _Redactor(os.environ if environ is None else environ).value("", spec)
