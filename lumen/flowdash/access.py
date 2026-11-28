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
from dataclasses import dataclass, field

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


_ACCESS_ORDER = ("allow_users", "allow_groups", "deny_users", "deny_groups")


def access_to_dict(permission: Permission | None) -> dict[str, list[str]]:
    """Serialize the four resource access fields of ``permission`` as sorted lists, omitting empty ones."""
    if permission is None:
        return {}
    result: dict[str, list[str]] = {}
    for key in _ACCESS_ORDER:
        values = getattr(permission, key, ())
        if values:
            result[key] = sorted(values)
    return result


def normalize_access(
    value: Permission | Mapping[str, t.Any] | None,
    *,
    section: str | None = None,
    keys: Sequence[str] = (),
) -> tuple[Permission, list[Problem]]:
    """Validate and normalize a ``Permission`` or raw ``access:`` mapping into a resource ``Permission``."""
    if value is None:
        return Permission(), []
    if isinstance(value, Permission):
        return value, []
    parsed, errors = _access_names(value)
    if errors:
        return Permission(), [
            Problem(message, section=section, keys=(*keys, *suffix))
            for suffix, message, _ in errors
        ]
    return Permission.from_spec(**parsed), []


def lockout_reason(permission: Permission | None, identity: Identity | None) -> str | None:
    """
    Why saving ``permission`` would lock out the admin ``identity``, or None if
    they would still have access. Clearing a rule never locks anyone out.
    """
    if permission is None or permission.is_empty:
        return None
    if can_access(permission, identity, admin=True):
        return None
    if identity is None:
        return "This rule would lock you out because your session has no identity, so it was not saved."
    matched_users = sorted(identity.user_names & permission.deny_users)
    matched_groups = sorted(identity.groups & permission.deny_groups)
    reasons = []
    if matched_users:
        who = ", ".join(repr(u) for u in matched_users)
        reasons.append(f"denies your user ({who})")
    if matched_groups:
        groups = ", ".join(repr(g) for g in matched_groups)
        reasons.append(f"denies your group{'s' * (len(matched_groups) != 1)} ({groups})")
    detail = " and ".join(reasons) or "denies your account"
    return f"This rule {detail}, which would lock you out, so it was not saved."


@dataclass(frozen=True)
class InheritedRule:
    """An access rule inherited from an upstream source, alias or table."""

    label: str
    permission: Permission


@dataclass(frozen=True)
class EffectiveAccess:
    """A resource's own ``access:`` rule together with every rule it inherits."""

    own: Permission = field(default_factory=Permission)
    inherited: tuple[InheritedRule, ...] = ()
    blocked_reason: str | None = None

    @property
    def has_own(self) -> bool:
        return not self.own.is_empty

    @property
    def has_inherited(self) -> bool:
        return any(not rule.permission.is_empty for rule in self.inherited)

    @property
    def is_empty(self) -> bool:
        return not self.has_own and not self.has_inherited and self.blocked_reason is None


def _join_clause(users: frozenset[str], groups: frozenset[str]) -> str:
    parts = []
    if users:
        names = ", ".join(sorted(users))
        parts.append(f"user{'s' * (len(users) != 1)} {names}")
    if groups:
        names = ", ".join(sorted(groups))
        parts.append(f"group{'s' * (len(groups) != 1)} {names}")
    return " or ".join(parts)


def describe_permission(permission: Permission | None) -> str:
    """A concise human-readable summary of one ``Permission``."""
    if permission is None or permission.is_empty:
        return "Open to everyone"
    clauses = []
    if allow := _join_clause(permission.allow_users, permission.allow_groups):
        clauses.append(f"Allow {allow}")
    if deny := _join_clause(permission.deny_users, permission.deny_groups):
        clauses.append(f"{'deny' if clauses else 'Deny'} {deny}")
    return "; ".join(clauses)


def access_badge(effective: EffectiveAccess) -> tuple[str, str]:
    """Short status label and chip tone (``outlined``, ``primary``, ``warning`` or ``error``) for ``effective``."""
    if effective.blocked_reason:
        return "Blocked", "error"
    if effective.is_empty:
        return "Open", "outlined"
    if effective.has_own and effective.has_inherited:
        return "Restricted (own + inherited)", "warning"
    if effective.has_own:
        return "Restricted", "warning"
    return "Inherited", "primary"


def _section_access_maps(
    context: Mapping[str, t.Any],
    section: Mapping[str, t.Any] | None,
) -> tuple[Mapping[str, Permission], Mapping[str, Mapping[str, Permission]]]:
    if section is not None:
        source_access: dict[str, Permission] = {}
        table_access: dict[str, dict[str, Permission]] = {}
        for ref, spec in section.items():
            if not isinstance(spec, Mapping):
                continue
            if "access" in spec:
                perm = _raw_access_permission(spec.get("access"))
                if not perm.is_empty:
                    source_access[str(ref)] = perm
            tables = spec.get("tables")
            if isinstance(tables, Mapping):
                for tbl, settings in tables.items():
                    if isinstance(settings, Mapping) and "access" in settings:
                        perm = _raw_access_permission(settings.get("access"))
                        if not perm.is_empty:
                            table_access.setdefault(str(ref), {})[str(tbl)] = perm
        return source_access, table_access
    catalog = _catalog_of(context)
    return (
        getattr(catalog, "source_access", None) or {},
        getattr(catalog, "table_access", None) or {},
    )


def effective_source_access(
    ref: str,
    context: Mapping[str, t.Any] | None = None,
    *,
    section: Mapping[str, t.Any] | None = None,
) -> EffectiveAccess:
    """The own and alias-inherited access rules applying to source ``ref``."""
    own = Permission()
    inherited: list[InheritedRule] = []
    for other, perm in _labelled_source_permissions(ref, context, section=section):
        if other == ref:
            own = perm
        else:
            inherited.append(InheritedRule(f"Source alias {other!r}", perm))
    return EffectiveAccess(own=own, inherited=tuple(inherited))


def effective_table_access(
    ref: str,
    source: Source | None,
    table: str,
    context: Mapping[str, t.Any] | None = None,
    *,
    section: Mapping[str, t.Any] | None = None,
) -> EffectiveAccess:
    """The own and inherited access rules applying to ``table`` of source ``ref``."""
    from .nodes import lookup_source

    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    resolved = source if source is not None else lookup_source(ref, context)
    target_norm = table.strip().strip('"').lower()
    table_rules = _labelled_table_permissions(resolved, table, context, ref=ref, section=section)

    own_index: int | None = None
    for idx, (other, configured, _perm) in enumerate(table_rules):
        if other != ref:
            continue
        if configured == table:
            own_index = idx
            break
        if own_index is None and configured.lower() == target_norm:
            own_index = idx
    own = table_rules[own_index][2] if own_index is not None else Permission()

    inherited: list[InheritedRule] = []
    src_target: Source | str = resolved if resolved is not None else ref
    for other, perm in _labelled_source_permissions(src_target, context, ref=ref, section=section):
        label = f"Source {other!r}" if other == ref else f"Source alias {other!r}"
        inherited.append(InheritedRule(label, perm))
    for idx, (other, configured, perm) in enumerate(table_rules):
        if idx != own_index:
            inherited.append(InheritedRule(f"Table {other}.{configured}", perm))
    return EffectiveAccess(own=own, inherited=tuple(inherited))


def effective_metric_access(
    metric: t.Any,
    context: Mapping[str, t.Any] | None = None,
    *,
    section: Mapping[str, t.Any] | None = None,
) -> EffectiveAccess:
    """The own and inherited access rules applying to ``metric`` (a ``Metric``, ``MetricStatus`` or spec dict)."""
    from .data import metric_tables
    from .nodes import lookup_source

    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    target, own, source_ref = _metric_parts(metric)
    blocked_reason = _metric_file_read_reason(target, source_ref, context, section=section)
    inherited: list[InheritedRule] = []
    seen: set[tuple[str, Permission]] = set()
    seen_sources: set[str] = set()

    def add(rule: InheritedRule) -> None:
        key = (rule.label, rule.permission)
        if not rule.permission.is_empty and key not in seen:
            seen.add(key)
            inherited.append(rule)

    if source_ref:
        for other, perm in _labelled_source_permissions(source_ref, context, section=section):
            seen_sources.add(other)
            label = f"Source {other!r}" if other == source_ref else f"Source alias {other!r}"
            add(InheritedRule(label, perm))

    for t_ref, t_name in metric_tables(target, context):
        src = lookup_source(t_ref, context)
        if src is None:
            continue
        for other, perm in _labelled_source_permissions(src, context, ref=t_ref, section=section):
            if other not in seen_sources:
                seen_sources.add(other)
                label = f"Source {other!r}" if other == t_ref else f"Source alias {other!r}"
                add(InheritedRule(label, perm))
        for other, configured, perm in _labelled_table_permissions(src, t_name, context, ref=t_ref, section=section):
            add(InheritedRule(f"Table {other}.{configured}", perm))

    return EffectiveAccess(own=own, inherited=tuple(inherited), blocked_reason=blocked_reason)


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


def _labelled_source_permissions(
    source_or_ref: Source | str,
    context: Mapping[str, t.Any] | None = None,
    *,
    ref: str | None = None,
    section: Mapping[str, t.Any] | None = None,
) -> list[tuple[str, Permission]]:
    """Every ``(ref, Permission)`` pair configured for ``source_or_ref`` in the catalog or ``section``."""
    from .nodes import lookup_source

    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    source_access, _ = _section_access_maps(context, section)
    if not source_access:
        return []
    if isinstance(source_or_ref, str):
        target_ref = source_or_ref
        resolved = lookup_source(source_or_ref, context)
    else:
        target_ref = ref
        resolved = source_or_ref
    if resolved is None:
        if target_ref is None:
            return []
        perm = source_access.get(target_ref)
        return [(target_ref, perm)] if perm is not None and not perm.is_empty else []
    root = _root(resolved)
    return [
        (other, perm)
        for other, perm in source_access.items()
        if not perm.is_empty
        and ((target_ref is not None and other == target_ref) or lookup_source(other, context) is root)
    ]


def _labelled_table_permissions(
    source: Source | None,
    table: str,
    context: Mapping[str, t.Any] | None = None,
    *,
    ref: str | None = None,
    section: Mapping[str, t.Any] | None = None,
) -> list[tuple[str, str, Permission]]:
    """Every ``(ref, configured_table, Permission)`` tuple for ``table`` (or its legacy target) on ``source``."""
    from .nodes import lookup_source

    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    _, table_access = _section_access_maps(context, section)
    if not table_access:
        return []
    resolved = source if source is not None else (lookup_source(ref, context) if ref is not None else None)
    target_norm = table.strip().strip('"').lower()
    if resolved is None:
        if ref is None:
            return []
        return [
            (ref, configured, perm)
            for configured, perm in (table_access.get(ref) or {}).items()
            if not perm.is_empty and configured.lower() == target_norm
        ]
    root = _root(resolved)
    names = {target_norm}
    if (target := _legacy_target(root, table)) is not None:
        names.add(target.lower())
    perms: list[tuple[str, str, Permission]] = []
    for other, tables_map in table_access.items():
        if not ((ref is not None and other == ref) or lookup_source(other, context) is root):
            continue
        for configured, perm in tables_map.items():
            if not perm.is_empty and configured.lower() in names:
                perms.append((other, configured, perm))
    return perms


def source_permissions(
    source_or_ref: Source | str,
    context: Mapping[str, t.Any] | None = None,
) -> list[Permission]:
    """Every non-empty source-level ``Permission`` configured for ``source_or_ref`` in the catalog."""
    return [perm for _, perm in _labelled_source_permissions(source_or_ref, context)]


def table_permissions(
    source: Source,
    table: str,
    context: Mapping[str, t.Any] | None = None,
) -> list[Permission]:
    """Every non-empty table-level ``Permission`` configured for ``table`` (or its legacy target) on ``source``."""
    return [perm for _, _, perm in _labelled_table_permissions(source, table, context)]


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


def _has_access_rules(
    source: Source,
    context: Mapping[str, t.Any],
    *,
    section: Mapping[str, t.Any] | None = None,
) -> bool:
    from .nodes import lookup_source

    if _labelled_source_permissions(source, context, section=section):
        return True
    _, table_access = _section_access_maps(context, section)
    if not table_access:
        return False
    root = _root(source)
    return any(
        any(not perm.is_empty for perm in tables_map.values()) and lookup_source(ref, context) is root
        for ref, tables_map in table_access.items()
    )


def has_access_rules(source: Source, context: Mapping[str, t.Any] | None = None) -> bool:
    """Whether any ``access:`` rule in the catalog applies to ``source`` (its root) or one of its tables."""
    if context is None:
        from panel_flowdash import dashboard_context

        context = dashboard_context()
    return _has_access_rules(source, context)


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


def _metric_parts(metric: t.Any) -> tuple[t.Any, Permission, str | None]:
    """Extract ``(target, own_permission, source_ref)`` from a ``Metric``, ``MetricStatus`` or spec dict."""
    if hasattr(metric, "metric") and hasattr(metric, "spec"):
        if metric.metric is not None:
            target = metric.metric
            raw_src = getattr(target, "source", None)
            return target, getattr(target, "access", None) or Permission(), raw_src if isinstance(raw_src, str) else None
        target = metric.spec
        if isinstance(target, Mapping):
            raw_src = target.get("source")
            return target, _raw_access_permission(target.get("access")), raw_src if isinstance(raw_src, str) else None
        return target, Permission(), None
    if hasattr(metric, "access"):
        raw_src = getattr(metric, "source", None)
        return metric, metric.access or Permission(), raw_src if isinstance(raw_src, str) else None
    if isinstance(metric, Mapping):
        raw_src = metric.get("source")
        return metric, _raw_access_permission(metric.get("access")), raw_src if isinstance(raw_src, str) else None
    return metric, Permission(), None


def _metric_file_read_reason(
    target: t.Any,
    source_ref: str | None,
    context: Mapping[str, t.Any],
    *,
    section: Mapping[str, t.Any] | None = None,
) -> str | None:
    """Why a SQL metric is denied on a restricted source due to direct file reads or unparseable SQL."""
    from .nodes import lookup_source

    if not ((sql := _metric_sql(target)) and source_ref):
        return None
    source = lookup_source(source_ref, context)
    if source is None or not _has_access_rules(source, context, section=section):
        return None
    try:
        reads = file_reads(sql, source)
    except Exception:
        return "SQL could not be checked for direct file reads on a restricted source."
    if reads:
        return access_denied_file_reads(reads)
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
    target, perm, source_ref = _metric_parts(metric)
    if not can_access(perm, resolved_identity, admin=resolved_admin):
        return False
    if _metric_file_read_reason(target, source_ref, context) is not None:
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
    "EffectiveAccess",
    "InheritedRule",
    "access_badge",
    "access_denied_file_reads",
    "access_denied_metric",
    "access_denied_source",
    "access_denied_tables",
    "access_to_dict",
    "attach_access",
    "authorized_metrics",
    "authorized_sources",
    "authorized_tables",
    "bind_identity",
    "can_access",
    "can_access_metric",
    "can_access_source",
    "can_access_table",
    "describe_permission",
    "effective_metric_access",
    "effective_source_access",
    "effective_table_access",
    "file_reads",
    "has_access_rules",
    "is_admin",
    "lockout_reason",
    "normalize_access",
    "parse_access",
    "restricted_tables",
    "session_identity",
    "session_is_admin",
    "source_permissions",
    "sql_access_denied",
    "table_permissions",
    "unauthorized_reads",
]
