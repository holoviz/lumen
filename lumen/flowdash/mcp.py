"""
MCP server integrations in Lumen's FlowDash app: feature-detected wiring to
``panel_flowdash.mcp`` and ``panel_flowdash.admin.mcp`` with Lumen's ``access:``
semantics.

When installed on a panel-flowdash release without MCP support, ``mcp_supported``
is False and Lumen keeps its ``branch_2.0`` behavior.
"""
from __future__ import annotations

import fnmatch
import inspect
import logging
import typing as t

from collections.abc import Callable, Mapping

from .access import (
    can_access, is_admin, session_identity, session_is_admin,
)

logger = logging.getLogger(__name__)

_UNSET: t.Any = object()
_HOOK_ATTR = "_lumen_mcp_tool_authorize"


def mcp_supported() -> bool:
    """Whether the installed panel-flowdash supports MCP servers and ``tool_authorize``."""
    try:
        import panel_flowdash.admin.mcp
        import panel_flowdash.mcp  # noqa: F401

        from panel_flowdash.app import FlowDashApp
    except ImportError:
        return False
    return "tool_authorize" in FlowDashApp.param


def _app_admin(app: t.Any) -> bool:
    """Session admin check for ``app``, supporting both FlowDashApp and test session stubs."""
    if app is None:
        return False
    if callable(getattr(app, "_authorize_admin", None)):
        return session_is_admin(app=app)
    if callable(getattr(app, "still_admin", None)) or callable(getattr(app, "_still_admin", None)):
        return is_admin(app)
    return False


def _resolve_mcp_session(
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> tuple[t.Any, bool | Callable[[], bool]]:
    """Resolve ``(identity, admin)`` without leaking one session's admin flag to another identity."""
    resolved_identity = session_identity(context, app=app) if identity is _UNSET else identity
    if admin is not None:
        return resolved_identity, admin
    if resolved_identity is None:
        return None, False
    if app is not None:
        app_identity = getattr(app, "_identity", None)
        if identity is _UNSET or resolved_identity == app_identity:
            return resolved_identity, lambda: _app_admin(app)
        return resolved_identity, False
    return resolved_identity, lambda: session_is_admin(context)


def _settings_project(app: t.Any) -> t.Any:
    getter = getattr(app, "_settings_project", None)
    if callable(getter):
        return getter()
    return getattr(app, "project", None)


def can_access_mcp_server(
    server: t.Any,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> bool:
    """Whether the session may access MCP ``server`` under Lumen's ``access:`` semantics."""
    resolved_identity, resolved_admin = _resolve_mcp_session(
        context, app=app, identity=identity, admin=admin,
    )
    return can_access(getattr(server, "access", None), resolved_identity, admin=resolved_admin)


def can_access_mcp_tool(
    server: t.Any,
    raw_tool: str,
    context: Mapping[str, t.Any] | None = None,
    *,
    app: t.Any = None,
    identity: t.Any = _UNSET,
    admin: bool | Callable[[], bool] | None = None,
) -> bool:
    """
    Whether the session may access ``raw_tool`` on MCP ``server``.

    Requires access to the server itself and every matching ``tool_access``
    pattern, evaluated with ``lumen.flowdash.access.can_access``.
    """
    resolved_identity, resolved_admin = _resolve_mcp_session(
        context, app=app, identity=identity, admin=admin,
    )
    if not can_access(getattr(server, "access", None), resolved_identity, admin=resolved_admin):
        return False
    tool_access = getattr(server, "tool_access", None) or {}
    for pattern, perm in tool_access.items():
        if fnmatch.fnmatchcase(raw_tool, pattern):
            if not can_access(perm, resolved_identity, admin=resolved_admin):
                return False
    return True


def visible_mcp_servers(
    project: t.Any,
    app: t.Any = None,
    *,
    admin: bool = False,
) -> list[t.Any]:
    """
    MCP servers visible in the Data section for the current session: admins see
    disabled servers too so they can re-enable or edit them; everyone else sees
    only enabled servers their ``access:`` rules allow.
    """
    if not mcp_supported():
        return []
    from panel_flowdash.mcp import configured_servers

    context = getattr(project, "context", None) or {}
    return [
        server
        for server in configured_servers(project, app).values()
        if (admin or server.enabled)
        and can_access_mcp_server(server, context, app=app, admin=admin)
    ]


def make_tool_authorize(
    app: t.Any,
    *,
    inner: Callable[[t.Any, str], bool] | None = None,
) -> Callable[[t.Any, str], bool]:
    """
    Build a ``FlowDashApp.tool_authorize(identity, tool_name)`` hook that
    enforces MCP server and tool ``access:`` rules with Lumen's ``can_access``
    semantics, composed with any caller-supplied ``inner`` hook.
    """
    from panel_flowdash.mcp import configured_servers, parse_mcp_tool_name

    def authorize(identity: t.Any, tool_name: str) -> bool:
        if inner is not None:
            try:
                if not bool(inner(identity, tool_name)):
                    return False
            except Exception:
                logger.exception("tool_authorize hook failed for %r; denying tool", tool_name)
                return False
        parsed = parse_mcp_tool_name(tool_name)
        if parsed is None:
            return True
        server_name, raw_tool = parsed
        project = _settings_project(app)
        servers = configured_servers(project, app)
        server = servers.get(server_name)
        if server is None or not server.enabled:
            return False
        context = getattr(project, "context", None) or {}
        admin = (
            _app_admin(app)
            if identity is not None and identity == getattr(app, "_identity", None)
            else False
        )
        return can_access_mcp_tool(server, raw_tool, context, identity=identity, admin=admin)

    setattr(authorize, _HOOK_ATTR, True)
    return authorize


def attach_mcp_access(
    app: t.Any,
    *,
    inner: Callable[[t.Any, str], bool] | None = None,
) -> bool:
    """Attach Lumen's MCP ``tool_authorize`` hook to ``app`` when panel-flowdash supports it."""
    if app is None or not mcp_supported() or not hasattr(app, "tool_authorize"):
        return False
    existing = inner if inner is not None else getattr(app, "tool_authorize", None)
    if existing is not None and getattr(existing, _HOOK_ATTR, False):
        return True
    app.tool_authorize = make_tool_authorize(app, inner=existing)
    return True


def integrations_view(
    project: t.Any,
    *,
    app: t.Any = None,
    editable: bool = False,
    on_change: Callable[[], t.Any] | None = None,
) -> t.Any:
    """Build panel-flowdash's ``MCPSettings`` view configured for Lumen's Data section."""
    from panel_flowdash.admin.mcp import MCPSettings
    from panel_flowdash.mcp import configured_servers, mcp_tool_name

    context = getattr(project, "context", None) or {}

    def server_filter(server: t.Any) -> bool:
        if not editable and not server.enabled:
            return False
        return can_access_mcp_server(server, context, app=app, admin=editable)

    def tool_filter(server_name: str, raw_tool: str) -> bool:
        server = configured_servers(project, app).get(server_name)
        if server is None:
            return False
        if not can_access_mcp_tool(server, raw_tool, context, app=app, admin=editable):
            return False
        if not editable and app is not None and hasattr(app, "_tool_authorized"):
            return bool(app._tool_authorized(mcp_tool_name(server_name, raw_tool)))
        return True

    kwargs: dict[str, t.Any] = {
        "editable": editable,
        "app": app,
        "server_filter": server_filter,
        "tool_filter": tool_filter,
        "show_access": True,
        "on_change": on_change,
    }
    if "show_add" in inspect.signature(MCPSettings.__init__).parameters:
        kwargs["show_add"] = False
    return MCPSettings(project, **kwargs)


__all__ = [
    "attach_mcp_access",
    "can_access_mcp_server",
    "can_access_mcp_tool",
    "integrations_view",
    "make_tool_authorize",
    "mcp_supported",
    "visible_mcp_servers",
]
