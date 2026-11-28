"""
Admin-only access rule summary rendering and adapters for panel-flowdash's
reusable ``PermissionEditor`` and ``PermissionDialog`` in Lumen's Data section.

Shows each resource's effective access rule including inheritance (source to
table, source and tables to metric), previews changes live, warns and refuses
rules that would lock out the saving admin, and hides interactive editors when
running against a ``panel-flowdash`` release without ``permission_editor``.
"""
from __future__ import annotations

import typing as t

from collections.abc import Callable, Sequence

from panel_flowdash.admin.kit import chip, esc
from panel_flowdash.auth import Permission
from panel_flowdash.project import Problem

from .access import (
    EffectiveAccess, access_badge, describe_permission, lockout_reason,
    session_identity,
)

ACCESS_CSS = """
.k .access-head { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.k .access-rules { margin: 6px 0 4px 0; padding-left: 18px; font-size: 0.8125rem;
  color: var(--mui-palette-text-primary); }
.k .access-rules li { margin: 2px 0; }
.k .access-rules b { font-weight: 500; }
.k .access-note { margin-top: 4px; }
"""


def _permission_editor_classes() -> tuple[t.Any, t.Any] | None:
    try:
        from panel_flowdash.admin.permission_editor import (
            PermissionDialog, PermissionEditor,
        )
    except ImportError:
        return None
    return PermissionEditor, PermissionDialog


def permission_editor_supported() -> bool:
    """Whether ``panel_flowdash`` provides the reusable ``PermissionEditor`` and ``PermissionDialog``."""
    return _permission_editor_classes() is not None


def effective_access_html(effective: EffectiveAccess, *, compact: bool = False) -> str:
    """Render an ``EffectiveAccess`` summary as escaped HTML using admin kit styling."""
    label, tone = access_badge(effective)
    badge = chip(label, tone)
    if effective.is_empty:
        text = "Open to everyone" if compact else "Open to everyone with app access."
        return f'<div class="k"><div class="access-head">{badge}<span class="muted">{esc(text)}</span></div></div>'
    items: list[str] = []
    if effective.blocked_reason:
        items.append(f"<li><b>Blocked:</b> {esc(effective.blocked_reason)}</li>")
    for rule in effective.inherited:
        if not rule.permission.is_empty:
            items.append(f"<li><b>Inherited from {esc(rule.label)}:</b> {esc(describe_permission(rule.permission))}</li>")
    if effective.has_own:
        items.append(f"<li><b>This resource:</b> {esc(describe_permission(effective.own))}</li>")
    elif effective.has_inherited:
        items.append('<li><b>This resource:</b> <span class="muted">No own rule (inherits above)</span></li>')
    if compact:
        if effective.has_own and not effective.has_inherited and not effective.blocked_reason:
            return (
                f'<div class="k"><div class="access-head">{badge}'
                f'<span>{esc(describe_permission(effective.own))}</span></div></div>'
            )
        return f'<div class="k"><div class="access-head">{badge}</div><ul class="access-rules">{"".join(items)}</ul></div>'
    n_rules = sum(1 for r in effective.inherited if not r.permission.is_empty) + int(effective.has_own)
    note = (
        "Effective access requires passing every rule above. Admins pass allowlists unless explicitly denied."
        if n_rules > 1
        else "Admins pass allowlists unless explicitly denied."
    )
    return (
        f'<div class="k"><div class="access-head">{badge}</div>'
        f'<ul class="access-rules">{"".join(items)}</ul>'
        f'<div class="muted small access-note">{esc(note)}</div></div>'
    )


def make_effective_display(effective: EffectiveAccess) -> Callable[[Permission], str]:
    """Build an ``effective_display`` callback combining ``effective``'s inheritance with the live draft."""
    inherited = effective.inherited
    blocked_reason = effective.blocked_reason

    def _display(perm: Permission) -> str:
        return effective_access_html(
            EffectiveAccess(own=perm, inherited=inherited, blocked_reason=blocked_reason),
        )

    return _display


def make_lockout_check(app: t.Any, project: t.Any) -> Callable[[Permission], str | None]:
    """Build a ``lockout_check`` callback evaluating ``lockout_reason`` against the session identity."""

    def _check(perm: Permission) -> str | None:
        if app is None:
            return None
        context = getattr(project, "context", None) if project is not None else None
        identity = session_identity(context, app=app)
        return lockout_reason(perm, identity)

    return _check


def build_access_editor(
    *,
    title: str = "Edit access rules",
    save_label: str = "Save access",
    framed: bool = True,
    app: t.Any = None,
    project: t.Any = None,
    on_save: Callable[[Permission], Sequence[Problem]] | None = None,
    on_cancel: Callable[[], t.Any] | None = None,
) -> t.Any:
    """Create a ``PermissionEditor`` configured for Lumen's Data pages, or None if unavailable."""
    classes = _permission_editor_classes()
    if classes is None:
        return None
    PermissionEditor, _ = classes
    return PermissionEditor(
        title=title,
        caption_text="Comma-separated user or group names. Leave all fields empty to clear this resource's own rule.",
        save_label=save_label,
        framed=framed,
        app=app,
        project=project,
        lockout_check=make_lockout_check(app, project),
        effective_stylesheets=(ACCESS_CSS,),
        on_save=on_save,
        on_cancel=on_cancel,
        css_classes=("lumen-access-editor",),
    )


def build_access_dialog(*, app: t.Any = None, project: t.Any = None) -> t.Any:
    """Create a ``PermissionDialog`` configured for Lumen's source cards, or None if unavailable."""
    classes = _permission_editor_classes()
    if classes is None:
        return None
    _, PermissionDialog = classes
    return PermissionDialog(
        app=app,
        project=project,
        lockout_check=make_lockout_check(app, project),
        effective_stylesheets=(ACCESS_CSS,),
        css_classes=("lumen-access-dialog", "lumen-access-editor"),
    )


def load_access_editor(
    editor: t.Any,
    effective: EffectiveAccess,
    *,
    title: str | None = None,
    app: t.Any = None,
    project: t.Any = None,
) -> None:
    """Populate a ``PermissionEditor`` with ``effective`` and Lumen's preview/lockout hooks."""
    editor.load(
        effective.own,
        title=title,
        app=app,
        project=project,
        effective_display=make_effective_display(effective),
        lockout_check=make_lockout_check(app, project),
    )


def open_access_dialog(
    dialog: t.Any,
    title: str,
    effective: EffectiveAccess,
    on_save: Callable[[Permission], Sequence[Problem]],
    *,
    app: t.Any = None,
    project: t.Any = None,
) -> None:
    """Open a ``PermissionDialog`` with ``effective`` and Lumen's preview/lockout hooks."""
    dialog.open(
        title,
        effective.own,
        on_save,
        app=app,
        project=project,
        effective_display=make_effective_display(effective),
        lockout_check=make_lockout_check(app, project),
    )
