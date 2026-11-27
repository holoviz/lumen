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
import re
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
_LEGACY_TARGET = re.compile(r'FROM "((?:[^"]|"")+)"')

# Sentinel permission that fails closed when a raw ``access:`` mapping is malformed.
_DENY_ALL = Permission(allow_users=frozenset({"__lumen_invalid_access__"}))


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
    ``context`` (or ``dashboard_context()``). Prefers live ``still_admin()``
    checks when available and fails closed on errors.
    """
    if app is not None:
        check = getattr(app, "still_admin", None) or getattr(app, "_still_admin", None)
        if callable(check):
            return is_admin(app)
        return bool(getattr(app, "is_admin", getattr(app, "_is_admin", False)))
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
    """Attach ``app``'s session identity and live admin checker to its editor context."""
    editor = getattr(app, "_editor", None)
    if editor is None or not isinstance(getattr(editor, "context", None), dict):
        return
    identity = getattr(app, "_identity", None)
    bind_identity(editor, identity if isinstance(identity, Identity) else None, admin=lambda: is_admin(app))


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
    if not isinstance(value, Mapping):
        known = ", ".join(f"`{k}`" for k in sorted(ACCESS_KEYS))
        return Permission(), [section.problem(path, f"`access` must be a mapping with {known}.")]
    problems: list[Problem] = []
    parsed: dict[str, list[str]] = {}
    for key, item in value.items():
        if key not in ACCESS_KEYS:
            known = ", ".join(f"`{k}`" for k in sorted(ACCESS_KEYS))
            problems.append(section.problem((*path, key), f"Unknown key {key!r} in `access`; known: {known}.", key=True))
            continue
        names = _parse_names(item)
        if names is None:
            problems.append(section.problem((*path, key), f"`{key}` must be a list of non-empty names."))
        else:
            parsed[key] = names
    if problems:
        return Permission(), problems
    return Permission.from_spec(**parsed), []


def _raw_access_permission(value: t.Any) -> Permission:
    """Parse a raw ``access:`` dict outside a section loader, failing closed if malformed."""
    if value is None:
        return Permission()
    if not isinstance(value, Mapping):
        return _DENY_ALL
    parsed: dict[str, list[str]] = {}
    for key, item in value.items():
        if key not in ACCESS_KEYS:
            return _DENY_ALL
        names = _parse_names(item)
        if names is None:
            return _DENY_ALL
        parsed[key] = names
    return Permission.from_spec(**parsed)


# ---------------------------------------------------------------------------
# Evaluating access


def can_access(
    permission: Permission | None,
    identity: Identity | None,
    *,
    admin: bool = False,
) -> bool:
    """
    Whether ``identity`` (with ``admin``) may access a resource governed by
    ``permission``. Explicit deny always wins; non-empty rules fail closed
    without an identity; admins bypass allowlists unless denied.
    """
    if permission is None or permission.is_empty:
        return True
    if identity is None:
        return False
    if permission.deny_users and identity.is_user(permission.deny_users):
        return False
    if permission.deny_groups and identity.in_groups(permission.deny_groups):
        return False
    if admin:
        return True
    if permission.allow_users or permission.allow_groups:
        return identity.is_user(permission.allow_users) or identity.in_groups(permission.allow_groups)
    return True


def _resolve_session(
    context: Mapping[str, t.Any] | None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | None = None,
) -> tuple[Mapping[str, t.Any], Identity | None, bool]:
    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    resolved_identity = session_identity(context, app=app) if identity is _UNSET else identity
    resolved_admin = session_is_admin(context, app=app) if admin is None else bool(admin)
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
    admin: bool | None = None,
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
    admin: bool | None = None,
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
    admin: bool | None = None,
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
    admin: bool | None = None,
) -> list[str]:
    """The tables referenced by ``sql`` on ``source`` that the session is not permitted to read."""
    from .catalog import hidden_reads

    restricted = restricted_tables(source, context, app=app, identity=identity, admin=admin)
    if not restricted:
        return []
    return hidden_reads(sql, source, restricted)


def can_access_metric(
    metric: t.Any,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | None = None,
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


def authorized_sources(
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | None = None,
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
        and can_access_source(ref, context, identity=resolved_identity, admin=resolved_admin)
    }


def authorized_tables(
    source: Source,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | None = None,
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
    admin: bool | None = None,
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
    "is_admin",
    "parse_access",
    "restricted_tables",
    "session_identity",
    "session_is_admin",
    "source_permissions",
    "table_permissions",
    "unauthorized_reads",
]
