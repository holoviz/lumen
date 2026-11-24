"""
The Uploads section of panel-flowdash's admin view: whether people may
upload data files, who, which file types and how large. It edits the
project file's ``uploads:`` section (see ``lumen.flowdash.uploads``) with
the admin page's Save and Discard.
"""
from __future__ import annotations

import contextlib
import typing as t

from collections.abc import Sequence

import panel_material_ui as pmui

from panel_flowdash.admin.form import SettingsForm, current_app
from panel_flowdash.project import Project

from .uploads import (
    DEFAULT_MAX_MB, SECTION, UPLOADS_KEY, _settings_groups, extension_of,
    upload_formats,
)

_DEFAULTS = {"enabled": False, "groups": [], "formats": None, "max_mb": DEFAULT_MAX_MB}


def _clean(value: dict[str, t.Any]) -> dict[str, t.Any]:
    """Drop the settings left at their default, so the file says only what was set."""
    return {key: v for key, v in value.items() if not (key in _DEFAULTS and v == _DEFAULTS[key])}


class UploadsForm(SettingsForm):
    """
    The Uploads section. Who may upload is a choice of the groups the app
    knows, empty for everyone, as Permissions picks who gets the assistant;
    the file types are the registered ones, all of them when the file lists
    none, so a type registered later is offered without a change here.
    """

    def __init__(self, project: Project, editable: bool = True, **params):
        uploads = (project.context or {}).get(UPLOADS_KEY)
        self._formats = dict(uploads.formats if uploads is not None else upload_formats())
        super().__init__(project, SECTION, _settings_groups(), editable=editable, from_form=_clean, **params)

    def _known_groups(self, value: Sequence[str]) -> list[str]:
        known = set(value)
        with contextlib.suppress(Exception):
            known |= set((current_app() or self._app)._known_groups())
        return sorted(known)

    def _names(self, setting, value):
        if setting.key == "formats":
            return pmui.MultiChoice(
                options={f"{f.label} (.{ext})": ext for ext, f in self._formats.items()},
                value=list(self._formats) if value is None else [e for e in self._formats if e in value],
                placeholder="None", width=360, margin=0,
            )
        return pmui.MultiChoice(
            options=self._known_groups(value or []), value=list(value or []), placeholder="Everyone",
            width=360, margin=0,
        )

    def _chosen(self, value: t.Any) -> list[str] | None:
        """The file types ``value`` lists in the registry's order, or None for all of them."""
        chosen = {extension_of(v) for v in value}
        return None if chosen >= set(self._formats) else [e for e in self._formats if e in chosen]

    def _effective(self, key):
        value = super()._effective(key)
        if key == "groups":
            return sorted(value or [])
        if key == "formats" and value is not None:
            return self._chosen(value)
        return value

    def _widget_value(self, key):
        if key not in ("groups", "formats") or key not in self._widgets:
            return super()._widget_value(key)
        value = list(self._widgets[key].value)
        return sorted(value) if key == "groups" else self._chosen(value)

    def _set_widget(self, key, value):
        if key == "formats":
            self._widgets[key].value = list(self._formats) if value is None else list(value)
        elif key == "groups":
            widget = self._widgets[key]
            widget.options = self._known_groups(value or [])
            widget.value = list(value or [])
        else:
            super()._set_widget(key, value)


def uploads_settings(project: Project, editable: bool) -> UploadsForm:
    """Build the admin view's Uploads section."""
    return UploadsForm(project, editable)
