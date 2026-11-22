"""
Redaction of credentials in a component's spec before it is shown.

A source spec can carry a password in a connection URI, API keys and
tokens as settings, secrets in ``connect_args`` or in SQL run on connect,
e.g. DuckDB's ``CREATE SECRET``, and values read from environment
variables. :func:`redact_spec` returns a copy of a spec that is safe to
display, erring on the side of hiding too much.
"""
from __future__ import annotations

import os
import re

from collections.abc import Mapping
from typing import Any

REDACTED = "***"

# Parts of a setting's name that mark its value as a credential, e.g. ``api_key`` or ``sslpassword``.
_SECRET_NAME = re.compile(
    r"pass(word|wd|phrase)?|pwd|secret|token|key|credential|auth(orization)?$|cookie|signature",
    re.IGNORECASE,
)
# Settings whose string values all go to a driver, so any of them may be a credential.
_DRIVER_ARGS = re.compile(r"^(connect|connection|conn|driver)_?(args|kwargs|options|params)$", re.IGNORECASE)
# Settings keyed by table name, which says nothing about whether the value is a credential.
_TABLE_SETTINGS = {"tables", "table_params", "mirrors", "excluded_tables"}
# Settings that name a credential but say how to sign in, not what with.
_NOT_SECRET = {"authenticator", "key_column", "primary_key", "index_key"}

_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_QUOTED = r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\""
_WORD = r"[A-Za-z0-9_.-]*"
_SECRET_WORD = r"(?:pass(?:word|wd|phrase)?|pwd|secret|token|key|credential|auth|signature)"
# ``name = value`` or ``name: value`` pairs inside a string, e.g. URI query
# parameters, ODBC connection strings and DuckDB ``SET s3_secret_access_key = '...'``.
_ASSIGNED = re.compile(
    rf"(?i)(\b{_WORD}{_SECRET_WORD}{_WORD}\s*[=:]\s*)({_QUOTED}|[^\s;&,)'\"]+)"
)
# ``NAME 'value'`` pairs, e.g. DuckDB's ``CREATE SECRET (TYPE s3, KEY_ID '...', SECRET '...')``.
_SPACED = re.compile(rf"(?i)(\b{_WORD}{_SECRET_WORD}{_WORD}\s+)({_QUOTED})")
# Inside a longer string, the values of environment variables named like a secret are hidden,
# e.g. ``PGPASSWORD`` or ``DB_PWD``, but not ``PWD``, the working directory.
_ENV_SECRET = re.compile(r"PASSWORD|PASSWD|PASSPHRASE|SECRET|TOKEN|APIKEY|CREDENTIAL", re.IGNORECASE)
_ENV_SECRET_PARTS = {"PASS", "PWD", "KEY", "AUTH", "CERT", "PRIVATE", "PAT", "DSN", "URL", "URI"}
# Shorter environment values are too likely to match unrelated settings, e.g. a port.
_MIN_ENV_VALUE = 8


def _secret_name(name: Any) -> bool:
    if not isinstance(name, str) or name.lower() in _NOT_SECRET:
        return False
    return _SECRET_NAME.search(name) is not None


def _redact_uri_passwords(text: str) -> str:
    """Hide the password of each URI in ``text``, e.g. ``postgresql://user:***@host/db``."""
    out, pos = [], 0
    for match in _SCHEME.finditer(text):
        if match.start() < pos:
            continue
        start = match.end()
        end = len(text)
        for i in range(start, len(text)):
            if text[i].isspace() or text[i] in "'\"":
                end = i
                break
        segment = text[start:end]
        if "@" not in segment:
            continue
        authority = re.split(r"[/?#]", segment, maxsplit=1)[0]
        # A password with an unescaped ``/`` or ``?`` ends the authority early, so fall back to the last ``@``.
        userinfo = authority.rpartition("@")[0] if "@" in authority else segment.rpartition("@")[0]
        user, colon, _ = userinfo.partition(":")
        if not colon:
            continue
        out.append(text[pos:start])
        out.append(f"{user}:{REDACTED}")
        pos = start + len(userinfo)
    out.append(text[pos:])
    return "".join(out)


def _env_secret(name: str) -> bool:
    parts = name.upper().split("_")
    if len(parts) > 1 and _ENV_SECRET_PARTS.intersection(parts):
        return True
    return _ENV_SECRET.search(name) is not None or parts[0] in _ENV_SECRET_PARTS - {"PWD"}


def _env_values(environ: Mapping[str, str]) -> tuple[set[str], list[str]]:
    """The values to hide wherever they are a whole setting, and those to hide inside longer strings."""
    whole, inside = set(), []
    for name, value in environ.items():
        if not isinstance(value, str) or len(value) < _MIN_ENV_VALUE:
            continue
        whole.add(value)
        if _env_secret(name):
            inside.append(value)
    return whole, sorted(inside, key=len, reverse=True)


def redact_text(text: str, environ: Mapping[str, str] | None = None) -> str:
    """Hide the credentials in a string: URI passwords, secret assignments and environment values."""
    whole, inside = _env_values(os.environ if environ is None else environ)
    return _redact_string(text, whole, inside)


def _redact_string(text: str, whole: set[str], inside: list[str]) -> str:
    if text in whole:
        return REDACTED
    for value in inside:
        if value in text:
            text = text.replace(value, REDACTED)
    text = _redact_uri_passwords(text)
    text = _ASSIGNED.sub(lambda m: m.group(1) + REDACTED, text)
    return _SPACED.sub(lambda m: m.group(1) + f"'{REDACTED}'", text)


def _plain(value: Any) -> Any:
    """``value`` as a type YAML can show, naming the type of anything else instead of printing it."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return REDACTED
    return f"<{type(value).__name__}>"


def _redact(value: Any, whole: set[str], inside: list[str], hide_strings: bool, names: bool = True) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (
                REDACTED if names and _secret_name(key) and item not in (None, "", {}, [])
                else _redact(
                    item, whole, inside, hide_strings or bool(_DRIVER_ARGS.match(str(key))),
                    names=key not in _TABLE_SETTINGS,
                )
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact(item, whole, inside, hide_strings) for item in value]
    if isinstance(value, str):
        return REDACTED if hide_strings and value else _redact_string(value, whole, inside)
    return _plain(value)


def redact_spec(spec: Any, environ: Mapping[str, str] | None = None) -> Any:
    """
    A copy of ``spec`` with its credentials replaced by ``***``, in plain
    types YAML can dump.

    Hidden are the values of settings named like a credential (password,
    secret, token, key, credentials and the like) at any depth, every string
    passed to a driver as ``connect_args`` or similar, the password of each
    URI, credentials assigned inside strings such as SQL run on connect,
    and any value that is the value of an environment variable, or contains
    the value of one named like a secret.
    """
    whole, inside = _env_values(os.environ if environ is None else environ)
    return _redact(spec, whole, inside, False)
