"""
Admin-only UI editors for per-resource ``access:`` rules on sources, tables
and metrics in Lumen's Data section.

Shows the effective access rule including inheritance (source to table,
source and tables to metric), previews changes live, warns and refuses rules
that would lock out the saving admin, and saves through the project file
writers in ``catalog.py`` and ``metrics_edit.py``.
"""
from __future__ import annotations

import contextlib
import re
import typing as t

from collections.abc import Callable, Iterable, Sequence

import panel as pn
import panel_material_ui as pmui

from panel.util import edit_readonly
from panel.viewable import Viewer
from panel_flowdash.admin.kit import (
    caption, cell, chip, esc,
)
from panel_flowdash.auth import Permission
from panel_flowdash.project import Problem

from .access import (
    EffectiveAccess, InheritedRule, access_badge, describe_permission,
    lockout_reason, session_identity,
)

_SPLIT = re.compile(r"[,\n]+")

_ACCESS_CSS = """
.k .access-head { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.k .access-rules { margin: 6px 0 4px 0; padding-left: 18px; font-size: 0.8125rem;
  color: var(--mui-palette-text-primary); }
.k .access-rules li { margin: 2px 0; }
.k .access-rules b { font-weight: 500; }
.k .access-note { margin-top: 4px; }
"""


def parse_name_list(text: str) -> list[str]:
    """Split comma- or newline-separated user or group names, stripping blanks and duplicates."""
    seen: dict[str, None] = {}
    for part in _SPLIT.split(text or ""):
        name = part.strip()
        if name:
            seen[name] = None
    return list(seen)


def format_name_list(names: Iterable[str]) -> str:
    """Format a set or sequence of user or group names as comma-separated text."""
    return ", ".join(sorted(names))


def known_principals(app: t.Any = None, project: t.Any = None) -> tuple[list[str], list[str]]:
    """Discover known user and group names from ``app`` and ``project`` for helper hints."""
    users: set[str] = set()
    groups: set[str] = set()
    if app is not None:
        with contextlib.suppress(Exception):
            if callable(getattr(app, "_known_users", None)):
                users |= {str(u) for u in app._known_users() if u}
        with contextlib.suppress(Exception):
            if callable(getattr(app, "_known_groups", None)):
                groups |= {str(g) for g in app._known_groups() if g}
    if project is not None:
        perms = (getattr(project, "data", None) or {}).get("permissions")
        if isinstance(perms, dict):
            admin_groups = perms.get("admin_groups")
            if isinstance(admin_groups, list):
                groups |= {str(g) for g in admin_groups if isinstance(g, str) and g.strip()}
            people = perms.get("people")
            if isinstance(people, dict):
                for user, info in people.items():
                    if isinstance(user, str) and user.strip():
                        users.add(user.strip())
                    if isinstance(info, dict) and isinstance(info.get("groups"), list):
                        groups |= {str(g) for g in info["groups"] if isinstance(g, str) and g.strip()}
    return sorted(users), sorted(groups)


def effective_access_html(effective: EffectiveAccess, *, compact: bool = False) -> str:
    """Render an ``EffectiveAccess`` summary as escaped HTML using admin kit styling."""
    label, tone = access_badge(effective)
    badge = chip(label, tone)
    if effective.is_empty:
        text = "Open to everyone" if compact else "Open to everyone with app access."
        return f'<div class="k"><div class="access-head">{badge}<span class="muted">{esc(text)}</span></div></div>'
    items: list[str] = []
    for rule in effective.inherited:
        if not rule.permission.is_empty:
            items.append(f"<li><b>Inherited from {esc(rule.label)}:</b> {esc(describe_permission(rule.permission))}</li>")
    if effective.has_own:
        items.append(f"<li><b>This resource:</b> {esc(describe_permission(effective.own))}</li>")
    elif effective.has_inherited:
        items.append('<li><b>This resource:</b> <span class="muted">No own rule (inherits above)</span></li>')
    if compact:
        if effective.has_own and not effective.has_inherited:
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


def _set_text(widget: pmui.TextInput, text: str) -> None:
    with edit_readonly(widget):
        widget.param.update(value=text, value_input=text)


class AccessRuleEditor(Viewer):
    """
    Admin-only editor for one resource's ``access:`` rule, with live effective
    rule preview (including inherited rules) and self-lockout warning.
    """

    def __init__(
        self,
        *,
        title: str = "Edit access rules",
        save_label: str = "Save access",
        framed: bool = True,
        app: t.Any = None,
        project: t.Any = None,
        on_save: Callable[[Permission], Sequence[Problem]] | None = None,
        on_cancel: Callable[[], t.Any] | None = None,
        **params,
    ):
        super().__init__(**params)
        self._app = app
        self._project = project
        self._on_save = on_save
        self._on_cancel = on_cancel
        self._inherited: tuple[InheritedRule, ...] = ()
        self._loading = False
        self._texts: dict[str, str] = {
            "allow_users": "",
            "allow_groups": "",
            "deny_users": "",
            "deny_groups": "",
        }

        self._title = pmui.Typography(title, variant="subtitle2", margin=0)
        self._caption = caption(
            "Comma-separated user or group names. Leave all fields empty to clear this resource's own rule.",
            margin=(2, 0, 4, 0),
        )
        self._effective_view = cell("", stylesheets=[_ACCESS_CSS], css_classes=["lumen-access-effective"])
        self._lockout_warn = pmui.Alert(
            object="", severity="warning", visible=False, sizing_mode="stretch_width", margin=0,
            css_classes=["lumen-access-lockout-warning"],
        )
        self._alert = pmui.Alert(
            object="", severity="error", visible=False, sizing_mode="stretch_width", margin=0,
            css_classes=["lumen-access-error"],
        )

        self._allow_users = self._make_input("allow_users", "Allow users", "e.g. alice, bob")
        self._allow_groups = self._make_input("allow_groups", "Allow groups", "e.g. hr, finance")
        self._deny_users = self._make_input("deny_users", "Deny users", "e.g. contractor")
        self._deny_groups = self._make_input("deny_groups", "Deny groups", "e.g. guests")

        self._clear_btn = pmui.Button(
            label="Clear rule", variant="text", color="default", size="small", margin=0,
            css_classes=["lumen-access-clear"],
        )
        self._clear_btn.on_click(lambda _event: self.clear())
        self._cancel_btn = pmui.Button(
            label="Cancel", variant="outlined", color="default", size="small", margin=0,
            css_classes=["lumen-access-cancel"],
        )
        self._cancel_btn.on_click(lambda _event: self.cancel())
        self._save_btn = pmui.Button(
            label=save_label, icon="save", variant="contained", color="primary", size="small", margin=0,
            css_classes=["lumen-access-save"],
        )
        self._save_btn.on_click(lambda _event: self.save())

        inner = pn.Column(
            *([self._title] if framed else []),
            self._caption,
            self._effective_view,
            self._lockout_warn,
            self._alert,
            pn.Row(
                self._allow_users, self._allow_groups, margin=(4, 0, 0, 0), sizing_mode="stretch_width",
                styles={"gap": "12px", "flex-wrap": "wrap"},
            ),
            pn.Row(
                self._deny_users, self._deny_groups, margin=(4, 0, 0, 0), sizing_mode="stretch_width",
                styles={"gap": "12px", "flex-wrap": "wrap"},
            ),
            pn.Row(
                self._clear_btn, pn.layout.HSpacer(), self._cancel_btn, self._save_btn,
                margin=(10, 0, 0, 0), sizing_mode="stretch_width", styles={"gap": "8px", "align-items": "center"},
            ),
            margin=0, sizing_mode="stretch_width", styles={"gap": "6px"},
        )
        if framed:
            self._layout: t.Any = pmui.Paper(
                inner, variant="outlined", margin=0, sizing_mode="stretch_width",
                sx={"p": "14px 16px", "borderRadius": "8px"},
                css_classes=["lumen-access-editor"],
            )
        else:
            inner.css_classes = ["lumen-access-editor"]
            self._layout = inner
        self._refresh_preview()

    def _make_input(self, key: str, label: str, placeholder: str) -> pmui.TextInput:
        css = key.replace("_", "-")
        widget = pmui.TextInput(
            label=label, value="", placeholder=placeholder, size="small",
            sizing_mode="stretch_width", margin=0, styles={"flex": "1 1 200px", "min-width": "180px"},
            css_classes=[f"lumen-access-{css}"],
        )

        def on_change(event):
            if self._loading:
                return
            self._texts[key] = event.new or ""
            self._refresh_preview()

        widget.param.watch(on_change, "value")
        widget.param.watch(on_change, "value_input")
        return widget

    def load(
        self,
        effective: EffectiveAccess,
        *,
        title: str | None = None,
        app: t.Any = None,
        project: t.Any = None,
    ) -> None:
        """Populate the editor from ``effective`` and refresh helper hints and preview."""
        if app is not None:
            self._app = app
        if project is not None:
            self._project = project
        if title is not None:
            self._title.object = title
        self._inherited = effective.inherited
        own = effective.own
        values = {
            "allow_users": format_name_list(own.allow_users),
            "allow_groups": format_name_list(own.allow_groups),
            "deny_users": format_name_list(own.deny_users),
            "deny_groups": format_name_list(own.deny_groups),
        }
        users, groups = known_principals(self._app, self._project)
        user_hint = f"Known: {', '.join(users[:6])}" + (f" +{len(users) - 6}" if len(users) > 6 else "") if users else ""
        group_hint = f"Known: {', '.join(groups[:6])}" + (f" +{len(groups) - 6}" if len(groups) > 6 else "") if groups else ""
        self._loading = True
        try:
            with pn.io.hold():
                self._texts.update(values)
                _set_text(self._allow_users, values["allow_users"])
                _set_text(self._allow_groups, values["allow_groups"])
                _set_text(self._deny_users, values["deny_users"])
                _set_text(self._deny_groups, values["deny_groups"])
                self._allow_users.helper_text = user_hint
                self._allow_groups.helper_text = group_hint
                self._deny_users.helper_text = "Denied even for admins"
                self._deny_groups.helper_text = "Denied even for admins"
                self._alert.visible = False
        finally:
            self._loading = False
        self._refresh_preview()

    def permission(self) -> Permission:
        """The ``Permission`` represented by the current input fields."""
        return Permission.from_spec(
            allow_users=parse_name_list(self._texts["allow_users"]),
            allow_groups=parse_name_list(self._texts["allow_groups"]),
            deny_users=parse_name_list(self._texts["deny_users"]),
            deny_groups=parse_name_list(self._texts["deny_groups"]),
        )

    def current_effective(self) -> EffectiveAccess:
        """The effective access combining inherited rules with the current input values."""
        return EffectiveAccess(own=self.permission(), inherited=self._inherited)

    def _refresh_preview(self) -> None:
        effective = self.current_effective()
        self._effective_view.object = effective_access_html(effective)
        context = getattr(self._project, "context", None) if self._project is not None else None
        identity = session_identity(context, app=self._app) if (context is not None or self._app is not None) else None
        reason = lockout_reason(effective.own, identity) if self._app is not None else None
        self._lockout_warn.param.update(object=reason or "", visible=bool(reason))

    def clear(self) -> None:
        """Clear all four fields (removing this resource's own rule on Save)."""
        self._loading = True
        try:
            with pn.io.hold():
                for key, widget in (
                    ("allow_users", self._allow_users),
                    ("allow_groups", self._allow_groups),
                    ("deny_users", self._deny_users),
                    ("deny_groups", self._deny_groups),
                ):
                    self._texts[key] = ""
                    _set_text(widget, "")
                self._alert.visible = False
        finally:
            self._loading = False
        self._refresh_preview()

    def show_problems(self, problems: Sequence[Problem]) -> None:
        self._alert.param.update(object="\n\n".join(p.message for p in problems), visible=bool(problems))

    def save(self) -> bool:
        """Invoke ``on_save`` with the current ``Permission``; show problems if refused."""
        if self._on_save is None:
            return False
        problems = list(self._on_save(self.permission()) or ())
        self.show_problems(problems)
        return not problems

    def cancel(self) -> None:
        self._alert.visible = False
        if self._on_cancel is not None:
            self._on_cancel()

    def __panel__(self):
        return self._layout


class AccessRuleDialog(Viewer):
    """Modal dialog hosting an ``AccessRuleEditor``, used on source cards."""

    def __init__(self, *, app: t.Any = None, project: t.Any = None, **params):
        super().__init__(**params)
        self._handler: Callable[[Permission], Sequence[Problem]] | None = None
        self.editor = AccessRuleEditor(
            framed=False, app=app, project=project,
            on_save=self._handle_save, on_cancel=self.close,
        )
        # Put the dialog class on the content rendered inside MUI's portal, not the empty host wrapper.
        self.editor._layout.css_classes = ["lumen-access-dialog", "lumen-access-editor"]
        self._dialog = pmui.Dialog(
            self.editor, title="Access rules", open=False, show_close_button=True, width_option="sm",
        )

    @property
    def is_open(self) -> bool:
        return bool(self._dialog.open)

    def open(
        self,
        title: str,
        effective: EffectiveAccess,
        on_save: Callable[[Permission], Sequence[Problem]],
        *,
        app: t.Any = None,
        project: t.Any = None,
    ) -> None:
        self._handler = on_save
        self.editor.load(effective, app=app, project=project)
        self._dialog.param.update(title=title, open=True)

    def close(self) -> None:
        self._dialog.open = False

    def _handle_save(self, perm: Permission) -> Sequence[Problem]:
        if self._handler is None:
            return []
        problems = list(self._handler(perm) or ())
        if not problems:
            self.close()
        return problems

    def __panel__(self):
        return self._dialog
