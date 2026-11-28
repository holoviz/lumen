"""
MCP server integrations in Lumen's FlowDash app: feature-detected wiring to
``panel_flowdash.mcp`` and ``panel_flowdash.admin.mcp``.

When installed on a panel-flowdash release without MCP support, ``mcp_supported``
is False and Lumen keeps its ``branch_2.0`` behavior.
"""
from __future__ import annotations

import typing as t

from collections.abc import Callable

from .access import session_identity


def mcp_supported() -> bool:
    """Whether the installed panel-flowdash supports MCP servers and admin redirects."""
    try:
        import panel_flowdash.admin.mcp
        import panel_flowdash.mcp  # noqa: F401

        from panel_flowdash.app import FlowDashApp
    except ImportError:
        return False
    return "admin_section_redirects" in FlowDashApp.param


def visible_mcp_servers(
    project: t.Any,
    app: t.Any = None,
    *,
    admin: bool = False,
) -> list[t.Any]:
    """
    MCP servers visible in the Data section for the current session: admins see
    every configured server (including disabled servers and servers whose
    ``access:`` rule denies them, so they can manage and fix rules); everyone
    else sees only enabled servers allowed by ``mcp_server_authorized``.
    """
    if not mcp_supported():
        return []
    from panel_flowdash.mcp import configured_servers, mcp_server_authorized

    servers = list(configured_servers(project, app).values())
    if admin:
        return servers
    context = getattr(project, "context", None) or {}
    identity = session_identity(context, app=app)
    return [
        server
        for server in servers
        if server.enabled and mcp_server_authorized(server, identity, is_admin=False)
    ]


def integrations_view(
    project: t.Any,
    *,
    app: t.Any = None,
    editable: bool = False,
    on_change: Callable[[], t.Any] | None = None,
) -> t.Any:
    """Build panel-flowdash's ``MCPSettings`` view configured for Lumen's Data section."""
    from panel_flowdash.admin.mcp import MCPSettings

    return MCPSettings(
        project,
        editable=editable,
        app=app,
        show_access=True,
        show_add=False,
        on_change=on_change,
    )


__all__ = [
    "integrations_view",
    "mcp_supported",
    "visible_mcp_servers",
]
