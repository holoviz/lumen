"""
Session authorization and per-resource access for Lumen FlowDash.

Builds on ``panel_flowdash.auth`` (``Identity``, ``Permission``) and live admin
checks (``app.still_admin()``) so Lumen FlowDash uses one authorization model
across nodes, the Data section, and the agent.

Per-resource ``access:`` rules in ``flowdash.yaml`` (under ``catalog:`` for
sources and tables, and ``metrics:`` for metrics) accept the four access keys
of ``panel_flowdash.auth.Permission``: ``allow_users``, ``allow_groups``,
``deny_users`` and ``deny_groups``.

Evaluation precedence:
1. Explicit ``deny_users`` / ``deny_groups`` denies access, even for admins.
2. An empty permission allows everyone (preserving open-by-default apps and
   headless ``lumen run``).
3. A non-empty permission without a session ``Identity`` fails closed.
4. An admin bypasses allowlists unless explicitly denied in step 1.
5. Non-empty ``allow_users`` / ``allow_groups`` allows only matching sessions.
6. Deny-only rules that did not match allow access.

Table access inherits source access; metric access inherits source and table
access so a metric cannot silently bypass restrictions on its underlying data.
"""

from __future__ import annotations

import logging
import typing as t

from collections.abc import Callable, Mapping, Sequence

from panel_flowdash.auth import Identity, Permission

from ..sources.base import Source
from .project import Problem, Section

logger = logging.getLogger(__name__)

# Editor context keys attached per session by ``attach_access`` / ``bind_identity``.
IDENTITY_KEY = "lumen.identity"
ADMIN_KEY = "lumen.is_admin"

ACCESS_KEYS = frozenset({"allow_users", "allow_groups", "deny_users", "deny_groups"})

_UNSET: t.Any = object()

# What a malformed raw ``access:`` mapping (one read outside the section loader,
# which reports it) evaluates to: an allowlist naming no real user, so it is
# admins only. Admins still reach the resource, e.g. to see and fix a broken metric.
_ADMINS_ONLY = Permission(allow_users=frozenset({"__lumen_invalid_access__"}))

# Table functions that read files, URLs or other databases. On a source with
# ``access:`` rules, SQL may not call them: a file a restricted table is defined
# over could be read through one without naming the table.
FILE_READERS = frozenset({
    "delta_scan", "glob", "iceberg_metadata", "iceberg_scan", "iceberg_snapshots", "mysql_query", "mysql_scan",
    "parquet_file_metadata", "parquet_kv_metadata", "parquet_metadata", "parquet_scan", "parquet_schema",
    "postgres_query", "postgres_scan", "query", "query_table", "read_blob", "read_csv", "read_csv_auto",
    "read_json", "read_json_auto", "read_json_objects", "read_json_objects_auto", "read_ndjson",
    "read_ndjson_auto", "read_ndjson_objects", "read_parquet", "read_text", "read_xlsx", "sniff_csv",
    "sqlite_query", "sqlite_scan", "st_read",
})
# Characters of a table name that make DuckDB read it as a file path (``FROM 'data.csv'``).
_PATH_CHARS = frozenset("./\\*:?")


# ---------------------------------------------------------------------------
# Session identity and admin checks


def is_admin(app: t.Any) -> bool:
    """
    Whether ``app``'s session is still an admin, with the project file read
    again. Fails closed when ``app`` is missing, lacks a live admin check, or
    the check raises.
    """
    if app is None:
        return False
    check = getattr(app, "still_admin", None) or getattr(app, "_still_admin", None)
    if not callable(check):
        return False
    try:
        return bool(check())
    except Exception as exc:
        from panel_flowdash.value_text import error_text

        logger.warning("Could not check whether this session is still an admin: %s", error_text(exc))
        return False


def session_identity(context: Mapping[str, t.Any] | None = None, *, app: t.Any = None) -> Identity | None:
    """The current session's ``Identity``, from ``app`` or ``context`` (or ``dashboard_context()``)."""
    if app is not None:
        identity = getattr(app, "_identity", None)
        if isinstance(identity, Identity):
            return identity
    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    identity = context.get(IDENTITY_KEY)
    return identity if isinstance(identity, Identity) else None


def session_is_admin(context: Mapping[str, t.Any] | None = None, *, app: t.Any = None) -> bool:
    """
    Whether the current session has admin privileges, from ``app`` or
    ``context`` (or ``dashboard_context()``), as found when the session
    started: reading access is checked on every edit, so it does not read the
    project file again. Saves and admin actions use ``is_admin``, which does.
    Fails closed on errors.
    """
    if app is not None:
        return _session_admin(app)
    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    value = context.get(ADMIN_KEY)
    if callable(value):
        try:
            return bool(value())
        except Exception as exc:
            from panel_flowdash.value_text import error_text

            logger.warning("Could not check session admin status: %s", error_text(exc))
            return False
    return bool(value)


def _session_admin(app: t.Any) -> bool:
    """``app``'s admin status as found when its session started, False when it has none or the check fails."""
    check = getattr(app, "_authorize_admin", None)
    if not callable(check):
        return False
    try:
        return bool(check())
    except Exception as exc:
        from panel_flowdash.value_text import error_text

        logger.warning("Could not check session admin status: %s", error_text(exc))
        return False


class _LazyAdmin:
    """A session's admin status, asked only once a rule needs it and then remembered."""

    __slots__ = ("_check", "_value")

    def __init__(self, check: Callable[[], bool]):
        self._check = check
        self._value: bool | None = None

    def __call__(self) -> bool:
        if self._value is None:
            self._value = bool(self._check())
        return self._value


def _admin_value(admin: bool | Callable[[], bool]) -> bool:
    return admin() if callable(admin) else bool(admin)


def bind_identity(
    editor: t.Any,
    identity: Identity | None,
    *,
    admin: bool | Callable[[], bool] = False,
) -> None:
    """Attach ``identity`` and ``admin`` status to ``editor.context`` for per-resource access checks."""
    if editor is None or not isinstance(getattr(editor, "context", None), dict):
        return
    if identity is None:
        editor.context.pop(IDENTITY_KEY, None)
    else:
        editor.context[IDENTITY_KEY] = identity
    editor.context[ADMIN_KEY] = admin


def attach_access(app: t.Any) -> None:
    """Attach ``app``'s session identity and its session admin status to its editor context."""
    editor = getattr(app, "_editor", None)
    if editor is None or not isinstance(getattr(editor, "context", None), dict):
        return
    identity = getattr(app, "_identity", None)
    bind_identity(editor, identity if isinstance(identity, Identity) else None, admin=lambda: _session_admin(app))


# ---------------------------------------------------------------------------
# Parsing ``access:`` YAML rules


def _parse_names(value: t.Any) -> list[str] | None:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else None
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                return None
            names.append(item.strip())
        return names
    return None


def _access_names(value: t.Any) -> tuple[dict[str, list[str]], list[tuple[tuple[str, ...], str, bool]]]:
    """The names of each key of an ``access:`` mapping, and its errors as (path suffix, message, is a key)."""
    known = ", ".join(f"`{k}`" for k in sorted(ACCESS_KEYS))
    if not isinstance(value, Mapping):
        return {}, [((), f"`access` must be a mapping with {known}.", False)]
    parsed: dict[str, list[str]] = {}
    errors: list[tuple[tuple[str, ...], str, bool]] = []
    for key, item in value.items():
        if key not in ACCESS_KEYS:
            errors.append(((key,), f"Unknown key {key!r} in `access`; known: {known}.", True))
        elif (names := _parse_names(item)) is None:
            errors.append(((key,), f"`{key}` must be a list of non-empty names.", False))
        else:
            parsed[key] = names
    return parsed, errors


def parse_access(
    value: t.Any,
    section: Section,
    path: Sequence[str | int],
) -> tuple[Permission, list[Problem]]:
    """
    Parse and validate an ``access:`` mapping in ``flowdash.yaml`` into a
    ``panel_flowdash.auth.Permission`` and any positioned ``Problem``s.
    """
    if value is None:
        return Permission(), []
    parsed, errors = _access_names(value)
    if errors:
        return Permission(), [
            section.problem((*path, *suffix), message, key=True) if key else section.problem((*path, *suffix), message)
            for suffix, message, key in errors
        ]
    return Permission.from_spec(**parsed), []


def _raw_access_permission(value: t.Any) -> Permission:
    """Parse a raw ``access:`` dict outside a section loader, failing closed if malformed."""
    if value is None:
        return Permission()
    parsed, errors = _access_names(value)
    return _ADMINS_ONLY if errors else Permission.from_spec(**parsed)


# ---------------------------------------------------------------------------
# Evaluating access


def can_access(
    permission: Permission | None,
    identity: Identity | None,
    *,
    admin: bool | Callable[[], bool] = False,
) -> bool:
    """
    Whether ``identity`` (with ``admin``) may access a resource governed by
    ``permission``. Explicit deny always wins; non-empty rules fail closed
    without an identity; admins bypass allowlists unless denied. A callable
    ``admin`` is only asked when an allowlist would otherwise decide.
    """
    if permission is None or permission.is_empty:
        return True
    if identity is None:
        return False
    if permission.deny_users and identity.is_user(permission.deny_users):
        return False
    if permission.deny_groups and identity.in_groups(permission.deny_groups):
        return False
    if not (permission.allow_users or permission.allow_groups):
        return True
    if identity.is_user(permission.allow_users) or identity.in_groups(permission.allow_groups):
        return True
    return _admin_value(admin)


def _resolve_session(
    context: Mapping[str, t.Any] | None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> tuple[Mapping[str, t.Any], Identity | None, bool | Callable[[], bool]]:
    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    resolved_identity = session_identity(context, app=app) if identity is _UNSET else identity
    if admin is None:
        session = context
        resolved_admin: bool | Callable[[], bool] = _LazyAdmin(lambda: session_is_admin(session, app=app))
    else:
        resolved_admin = admin
    return context, resolved_identity, resolved_admin


def _catalog_of(context: Mapping[str, t.Any]) -> t.Any:
    from .catalog import CATALOG_KEY

    catalog = context.get(CATALOG_KEY)
    if catalog is None:
        return None
    getter = getattr(catalog, "catalog", None)
    return getter() if callable(getter) else catalog


def _root(source: Source) -> Source:
    from .results import is_derived, root_of

    return root_of(source) if is_derived(source) else source


def _legacy_target(source: Source, table: str) -> str | None:
    """The underlying table name when ``table`` is an unlisted legacy view on ``source``."""
    from .catalog import _LEGACY_TARGET
    from .nodes import LEGACY_TABLE

    declared = getattr(source, "tables", None)
    if not isinstance(declared, Mapping):
        return None
    sql = declared.get(table)
    if isinstance(sql, str) and LEGACY_TABLE in sql and (match := _LEGACY_TARGET.search(sql)):
        return match.group(1).replace('""', '"')
    return None


def source_permissions(
    source_or_ref: Source | str,
    context: Mapping[str, t.Any] | None = None,
) -> list[Permission]:
    """Every non-empty source-level ``Permission`` configured for ``source_or_ref`` in the catalog."""
    from .nodes import lookup_source

    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    catalog = _catalog_of(context)
    source_access = getattr(catalog, "source_access", None)
    if not source_access:
        return []
    if isinstance(source_or_ref, str):
        resolved = lookup_source(source_or_ref, context)
        if resolved is None:
            perm = source_access.get(source_or_ref)
            return [perm] if perm is not None and not perm.is_empty else []
        root = _root(resolved)
    else:
        root = _root(source_or_ref)
    perms: list[Permission] = []
    for ref, perm in source_access.items():
        if perm.is_empty:
            continue
        if (isinstance(source_or_ref, str) and ref == source_or_ref) or lookup_source(ref, context) is root:
            perms.append(perm)
    return perms


def table_permissions(
    source: Source,
    table: str,
    context: Mapping[str, t.Any] | None = None,
) -> list[Permission]:
    """Every non-empty table-level ``Permission`` configured for ``table`` (or its legacy target) on ``source``."""
    from .nodes import lookup_source

    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    catalog = _catalog_of(context)
    table_access = getattr(catalog, "table_access", None)
    if not table_access:
        return []
    root = _root(source)
    names = {table.strip().strip('"').lower()}
    if (target := _legacy_target(root, table)) is not None:
        names.add(target.lower())
    perms: list[Permission] = []
    for ref, tables_map in table_access.items():
        if lookup_source(ref, context) is not root:
            continue
        for configured, perm in tables_map.items():
            if not perm.is_empty and configured.lower() in names:
                perms.append(perm)
    return perms


def can_access_source(
    source_or_ref: Source | str,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> bool:
    """Whether the session may access ``source_or_ref``."""
    context, resolved_identity, resolved_admin = _resolve_session(
        context,
        app=app,
        identity=identity,
        admin=admin,
    )
    return all(can_access(perm, resolved_identity, admin=resolved_admin) for perm in source_permissions(source_or_ref, context))


def can_access_table(
    source: Source,
    table: str,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> bool:
    """Whether the session may access ``table`` of ``source`` (requires access to both source and table)."""
    context, resolved_identity, resolved_admin = _resolve_session(
        context,
        app=app,
        identity=identity,
        admin=admin,
    )
    if not can_access_source(source, context, identity=resolved_identity, admin=resolved_admin):
        return False
    return all(can_access(perm, resolved_identity, admin=resolved_admin) for perm in table_permissions(source, table, context))


def restricted_tables(
    source: Source,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> frozenset[str]:
    """
    The tables of ``source`` (including legacy aliases) that the session is
    not permitted to read.
    """
    from .catalog import _with_legacy_names
    from .nodes import lookup_source

    context, resolved_identity, resolved_admin = _resolve_session(
        context,
        app=app,
        identity=identity,
        admin=admin,
    )
    root = _root(source)
    if not can_access_source(root, context, identity=resolved_identity, admin=resolved_admin):
        try:
            return frozenset(root.get_tables())
        except Exception:
            return frozenset()
    catalog = _catalog_of(context)
    table_access = getattr(catalog, "table_access", None)
    if not table_access:
        return frozenset()
    denied: set[str] = set()
    for ref, tables_map in table_access.items():
        if lookup_source(ref, context) is not root:
            continue
        for table, perm in tables_map.items():
            if not can_access(perm, resolved_identity, admin=resolved_admin):
                denied.add(table)
    return _with_legacy_names(root, denied) if denied else frozenset()


def unauthorized_reads(
    sql: str,
    source: Source,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> list[str]:
    """The tables referenced by ``sql`` on ``source`` that the session is not permitted to read."""
    from .catalog import hidden_reads

    restricted = restricted_tables(source, context, app=app, identity=identity, admin=admin)
    if not restricted:
        return []
    return hidden_reads(sql, source, restricted)


def has_access_rules(source: Source, context: Mapping[str, t.Any] | None = None) -> bool:
    """Whether any ``access:`` rule in the catalog applies to ``source`` (its root) or one of its tables."""
    from .nodes import lookup_source

    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    if source_permissions(source, context):
        return True
    table_access = getattr(_catalog_of(context), "table_access", None)
    if not table_access:
        return False
    root = _root(source)
    return any(
        any(not perm.is_empty for perm in tables_map.values()) and lookup_source(ref, context) is root
        for ref, tables_map in table_access.items()
    )


def _function_name(node: t.Any) -> str:
    from sqlglot import exp

    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return type(node).sql_name().lower() if isinstance(node, exp.Func) else type(node).__name__.lower()


def file_reads(sql: str, source: Source) -> list[str]:
    """
    The file-reading table functions (``read_parquet``, ``glob``, ...), other
    table functions and file paths read as tables (``FROM 'data.csv'``) that
    ``sql``, a query on ``source``, uses.
    """
    import sqlglot

    from sqlglot import exp

    parsed = sqlglot.parse_one(sql, read=getattr(source, "dialect", None))
    found: dict[str, None] = {}
    for table in parsed.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            found[_function_name(table.this)] = None
        elif _PATH_CHARS & set(table.name):
            found[table.name] = None
    for func in parsed.find_all(exp.Func):
        if (name := _function_name(func)) in FILE_READERS:
            found[name] = None
    return list(found)


def access_denied_file_reads(reads: Sequence[str]) -> str:
    names = ", ".join(repr(name) for name in reads)
    return (f"This source has access rules, so its queries may only read its tables by name; {names} "
            "reads files or other data directly and is not allowed.")


def sql_access_denied(
    sql: str,
    source: Source,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> str | None:
    """
    Why the session may not run ``sql`` on ``source``, None if it may: it
    reads a table the session may not read, or the source has access rules
    and the query reads files or other data through a table function.
    """
    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    if not has_access_rules(source, context):
        return None
    if reads := file_reads(sql, source):
        return access_denied_file_reads(reads)
    if denied := unauthorized_reads(sql, source, context, app=app, identity=identity, admin=admin):
        return access_denied_tables(denied)
    return None


def can_access_metric(
    metric: t.Any,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> bool:
    """
    Whether the session may access ``metric`` (a ``metrics.Metric``, a
    ``data.MetricStatus``, or a raw ``metrics:`` spec dict). Requires access
    to the metric itself and to its backing source and tables.
    """
    from .data import metric_tables
    from .nodes import lookup_source

    context, resolved_identity, resolved_admin = _resolve_session(
        context,
        app=app,
        identity=identity,
        admin=admin,
    )
    # Extract the metric's own permission, source ref, and table spec.
    target = metric
    perm: Permission | None = None
    source_ref: str | None = None
    if hasattr(metric, "metric") and hasattr(metric, "spec"):
        if metric.metric is not None:
            target = metric.metric
            perm = getattr(target, "access", None)
            source_ref = getattr(target, "source", None)
        else:
            target = metric.spec
            perm = _raw_access_permission(target.get("access")) if isinstance(target, Mapping) else None
            raw_src = target.get("source") if isinstance(target, Mapping) else None
            source_ref = raw_src if isinstance(raw_src, str) else None
    elif hasattr(metric, "access"):
        perm = metric.access
        source_ref = getattr(metric, "source", None)
    elif isinstance(metric, Mapping):
        perm = _raw_access_permission(metric.get("access"))
        raw_src = metric.get("source")
        source_ref = raw_src if isinstance(raw_src, str) else None

    if not can_access(perm, resolved_identity, admin=resolved_admin):
        return False
    if (sql := _metric_sql(target)) and source_ref:
        source = lookup_source(source_ref, context)
        if source is not None and has_access_rules(source, context):
            try:
                if file_reads(sql, source):
                    return False
            except Exception:
                return False
    if source_ref and not can_access_source(
        source_ref,
        context,
        identity=resolved_identity,
        admin=resolved_admin,
    ):
        return False
    for ref, table in metric_tables(target, context):
        src = lookup_source(ref, context)
        if src is not None and not can_access_table(
            src,
            table,
            context,
            identity=resolved_identity,
            admin=resolved_admin,
        ):
            return False
    return True


def _metric_sql(metric: t.Any) -> str | None:
    if isinstance(metric, Mapping):
        sql = metric.get("sql")
    else:
        sql = getattr(metric, "sql", None)
    return sql if isinstance(sql, str) and sql.strip() else None


def authorized_sources(
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> dict[str, Source]:
    """The host sources in ``context`` that the session is permitted to access, by ref."""
    from .nodes import host_sources

    context, resolved_identity, resolved_admin = _resolve_session(
        context,
        app=app,
        identity=identity,
        admin=admin,
    )
    return {
        ref: source
        for ref, source in host_sources(context).items()
        if can_access_source(source, context, identity=resolved_identity, admin=resolved_admin)
    }


def authorized_tables(
    source: Source,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> list[str]:
    """The listed tables of ``source`` that the session is permitted to access."""
    from .nodes import listed_tables

    context, resolved_identity, resolved_admin = _resolve_session(
        context,
        app=app,
        identity=identity,
        admin=admin,
    )
    if not can_access_source(source, context, identity=resolved_identity, admin=resolved_admin):
        return []
    restricted = restricted_tables(source, context, identity=resolved_identity, admin=resolved_admin)
    if not restricted:
        return listed_tables(source)
    return [table for table in listed_tables(source) if table not in restricted]


def authorized_metrics(
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> dict[str, t.Any]:
    """The host metrics in ``context`` that the session is permitted to access, by name."""
    from .nodes import host_metrics

    context, resolved_identity, resolved_admin = _resolve_session(
        context,
        app=app,
        identity=identity,
        admin=admin,
    )
    return {name: metric for name, metric in host_metrics(context).items() if can_access_metric(metric, context, identity=resolved_identity, admin=resolved_admin)}


# ---------------------------------------------------------------------------
# User-facing denial messages


def access_denied_source(ref: str) -> str:
    return f"You do not have permission to access source {ref!r}."


def access_denied_tables(tables: Sequence[str], ref: str | None = None) -> str:
    names = ", ".join(repr(table) for table in sorted(tables))
    of = f" of source {ref!r}" if ref else ""
    plural = len(tables) != 1
    return f"You do not have permission to access table{'s' * plural} {names}{of}."


def access_denied_metric(name: str) -> str:
    return f"You do not have permission to access metric {name!r}."


__all__ = [
    "ACCESS_KEYS",
    "ADMIN_KEY",
    "IDENTITY_KEY",
    "access_denied_file_reads",
    "access_denied_metric",
    "access_denied_source",
    "access_denied_tables",
    "attach_access",
    "authorized_metrics",
    "authorized_sources",
    "authorized_tables",
    "bind_identity",
    "can_access",
    "can_access_metric",
    "can_access_source",
    "can_access_table",
    "file_reads",
    "has_access_rules",
    "is_admin",
    "parse_access",
    "restricted_tables",
    "session_identity",
    "session_is_admin",
    "source_permissions",
    "sql_access_denied",
    "table_permissions",
    "unauthorized_reads",
]
