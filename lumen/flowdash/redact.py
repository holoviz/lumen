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
else, and any shown word that looks like a token, reads ``***``.
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
from sqlglot.dialects import Dialect
from sqlglot.tokens import Token, TokenType

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
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
_FILE_EXTENSION = re.compile(
    r"\.(csv|tsv|txt|parquet|json|jsonl|ndjson|xlsx|xls|arrow|feather|db|duckdb|sqlite|sqlite3|gz|zip|nc|zarr)$",
    re.IGNORECASE,
)
# Names of settings, arguments or columns whose value is a credential, e.g. ``s3_secret_access_key``.
_SECRET_NAME = re.compile(r"pass|pwd|secret|token|key|credential|auth|sig|cookie|session|private|cert", re.IGNORECASE)
# Prefixes of well known token formats: GitHub, GitLab, AWS, OpenAI and similar, Slack, JWTs, Google, Hugging Face.
_TOKEN_PREFIX = re.compile(r"^(gh[pousr]_|github_pat_|glpat-|AKIA|ASIA|sk-|xox[abpors]-|eyJ|AIza|hf_)")
_WORD = re.compile(r"[A-Za-z0-9]+")

# Statements that set or store a value, e.g. ``SET s3_secret_access_key TO '...'``, shown by their keyword only.
_SETTING_STATEMENTS = (exp.Command, exp.Set, exp.Pragma)
# Statements shown with their string literals hidden; any other is hidden entirely, e.g. a bare expression.
_SHOWN_STATEMENTS = (exp.Query, exp.Install, exp.Attach, exp.Detach, exp.Create, exp.Insert, exp.Use)
# Tokens around an argument's value, e.g. ``s3_secret_access_key = '...'`` or ``{'key': '...'}``.
_ASSIGNS = {TokenType.EQ, TokenType.COLON, TokenType.COLON_EQ, TokenType.FARROW}
_NAMES = {TokenType.VAR, TokenType.IDENTIFIER}

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


def _opaque(text: str) -> bool:
    """
    Whether ``text`` holds a word that looks like a token rather than a
    name: a known token prefix, letters and digits mixed over 8 or more
    characters, mixed case over 10 or more, or any run of 24 or more.
    """
    if _TOKEN_PREFIX.match(text):
        return True
    for word in _WORD.findall(text):
        if _TOKEN_PREFIX.match(word) or len(word) >= 24:
            return True
        if len(word) >= 8 and any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
            return True
        if len(word) >= 10 and sum(c.isupper() for c in word[1:]) >= 2 and sum(c.islower() for c in word) >= 2:
            return True
    return False


def _plain_name(text: str) -> bool:
    return bool(_NAME.match(text)) and "://" not in text and not _opaque(text)


def _path(path: str) -> str | None:
    """``path`` with each segment that looks like a token hidden, or None if it is not a plain path."""
    if not _PATH.match(path or "/"):
        return None
    return "/".join(REDACTED if _opaque(segment) else segment for segment in path.split("/"))


def _location(text: str) -> str | None:
    """``text`` as a URL rebuilt from its scheme, host, port and path, or as a plain path, or None."""
    if "://" in text:
        try:
            parts = urlsplit(text)
            port = parts.port
        except ValueError:
            return None
        host = parts.hostname or ""
        path = _path(parts.path)
        if not parts.scheme or path is None or (host and not _plain_name(host)):
            return None
        return f"{parts.scheme}://{host}{f':{port}' if port else ''}{path if parts.path else ''}"
    path = text.split("?", 1)[0].split("#", 1)[0]
    return _path(path) if path else None


def _literal_location(text: str) -> str | None:
    """
    ``text`` rebuilt by :func:`_location` if it reads as a location: it has
    a URL scheme, starts like an absolute or relative path, or is a file
    name with a data file's extension.
    """
    if _SCHEME.match(text) or text.startswith(("/", "./", "../", "~")):
        return _location(text)
    if "/" not in text and "\\" not in text and _FILE_EXTENSION.search(text):
        return _location(text)
    return None


def _string_token(token: Token) -> bool:
    name = token.token_type.name
    return name == "STRING" or name.endswith("_STRING")


class _QuietParse(logging.Filter):
    """Drop sqlglot's log records, which quote the SQL, while this module parses SQL that may hold secrets."""

    def __init__(self):
        super().__init__()
        self.local = threading.local()

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(self.local, "active", False)


_QUIET = _QuietParse()
logging.getLogger("sqlglot").addFilter(_QUIET)


def _quietly(func, *args, **kwargs):
    _QUIET.local.active = True
    try:
        return func(*args, **kwargs)
    finally:
        _QUIET.local.active = False


def _statements(tokens: list[Token]) -> list[list[Token]]:
    chunks: list[list[Token]] = [[]]
    for token in tokens:
        if token.token_type == TokenType.SEMICOLON:
            chunks.append([])
        else:
            chunks[-1].append(token)
    return [chunk for chunk in chunks if chunk]


def _masked(tokens: list[Token], index: int) -> str | None:
    """The text a literal or name token is shown as, or None to show it as written."""
    token = tokens[index]
    before = tokens[index - 2:index] if index >= 2 else []
    named_secret = (
        len(before) == 2 and before[1].token_type in _ASSIGNS
        and before[0].token_type in (*_NAMES, TokenType.STRING)
        and _SECRET_NAME.search(before[0].text) is not None
    )
    if _string_token(token):
        location = None if named_secret or token.token_type != TokenType.STRING else _literal_location(token.text)
        return "'{}'".format((location or REDACTED).replace("'", "''"))
    if token.token_type in _NAMES and (_SECRET_NAME.search(token.text) or _opaque(token.text)):
        return REDACTED
    return None


def _write(text: str, tokens: list[Token]) -> str:
    """
    The original text of ``tokens`` with literals and names hidden as
    :func:`_masked` says. Comments and hints between tokens are dropped.
    """
    out = []
    for index, token in enumerate(tokens):
        if index:
            gap = text[tokens[index - 1].end + 1:token.start]
            out.append(gap if not gap or gap.isspace() else "\n" if "\n" in gap else " ")
        masked = _masked(tokens, index)
        out.append(masked if masked is not None else text[token.start:token.end + 1])
    return "".join(out)


def _redact_sql(text: str, dialect: str = "duckdb") -> str:
    """
    ``text`` as written, with every string literal hidden unless it is a
    location, names that look like credentials or tokens hidden, comments
    dropped, and each statement that sets or stores a value shown by its
    keyword only. SQL that does not tokenize or parse is hidden entirely.
    """
    try:
        tokens = _quietly(Dialect.get_or_raise(dialect).tokenize, text)
    except Exception:
        return REDACTED
    out = []
    for chunk in _statements(tokens):
        source = text[chunk[0].start:chunk[-1].end + 1]
        try:
            parsed = _quietly(sqlglot.parse, source, read=dialect, error_level=sqlglot.ErrorLevel.RAISE)
        except Exception:
            out.append(REDACTED)
            continue
        statement = parsed[0] if len(parsed) == 1 else None
        if statement is None:
            out.append(REDACTED)
        elif not isinstance(statement, _SHOWN_STATEMENTS) or statement.find(exp.Command) is not None:
            keyword = statement.this if isinstance(statement, exp.Command) else statement.key
            shown = isinstance(statement, _SETTING_STATEMENTS) and isinstance(keyword, str) and _NAME.match(keyword)
            out.append(f"{keyword.upper()} {REDACTED}" if shown else REDACTED)
        else:
            out.append(_write(text, chunk))
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

    def _name(self, name: Any, index: int, pattern: re.Pattern = _NAME) -> str:
        """A key or table name as it is if it is a plain name, ``***<index>`` otherwise, so keys stay unique."""
        if isinstance(name, str) and pattern.match(name) and _plain_name(name) and self.shown(name) == name:
            return name
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
        if key in _SAFE_SETTINGS and _plain_name(value):
            return self.shown(value)
        return REDACTED

    def _table(self, value: Any) -> Any:
        """A table's definition: its name, a location, SQL with its literals hidden, or a nested spec."""
        if not isinstance(value, str):
            return self.value("", value)
        if _plain_name(value):
            return self.shown(value)
        if (location := _literal_location(value)) is not None:
            return self.shown(location)
        return self.shown(_redact_sql(value))

    def _tables(self, key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                self._name(name, index): self._table(item) if key == "tables" else self.value("", item)
                for index, (name, item) in enumerate(value.items())
            }
        if isinstance(value, (list, tuple)):
            return [self._table(item) for item in value]
        return self.value(key, value)

    def value(self, key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                self._name(name, index, _SAFE_KEY): (
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
    table definitions) as written with every string literal hidden unless
    it is a location, and every statement that sets a value reduced to its
    keyword. Any shown word that looks like a token, anything that equals
    an environment variable's value, and anything containing the value of
    one named like a credential are hidden too. Every other value reads
    ``***``, and objects YAML cannot show are named by their type.
    """
    return _Redactor(os.environ if environ is None else environ).value("", spec)
