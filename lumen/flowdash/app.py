"""
A FlowDash app over data the host provides, built from Lumen's nodes; what
``lumen serve --flowdash`` serves.
"""
from __future__ import annotations

import os

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import param

from panel_flowdash import FlowDash
from panel_flowdash.app import FlowDashApp
from panel_flowdash.dashboard_store import DashboardStore

from ..sources.base import Source
from .nodes import SOURCE_ID, bind_source

LIBRARY = "lumen.flowdash"


class LumenFlowDash(FlowDash):
    """
    The FlowDash editor with the host's sources bound, so its source nodes
    can name them by ref.

    SEAM(panel-flowdash#166): once ``FlowDash.context`` is released this
    class goes, and the app passes
    ``editor_options={"context": {"lumen.sources": sources}}`` instead.
    """

    lumen_sources = param.Dict(default={}, doc="The sources source nodes may use, by ref.")

    def __init__(self, **params):
        super().__init__(**params)
        for ref, source in self.lumen_sources.items():
            bind_source(self, source, ref)


def source_refs(sources: Sequence[Source]) -> dict[str, Source]:
    """The refs source nodes name ``sources`` by: ``data``, then ``data_2``, ``data_3`` and so on."""
    return {("data" if i == 0 else f"data_{i + 1}"): source for i, source in enumerate(sources)}


def resolve_sources(data: Sequence[str | os.PathLike | Source]) -> list[Source]:
    """
    Sources for files, URLs, database connection strings or sources, read
    as ``lumen serve`` reads them for Lumen AI.
    """
    from ..ai.ui import ExplorerUI

    return ExplorerUI._resolve_data([d if isinstance(d, Source) else str(d) for d in data])


def build_routes(
    data: Sequence[str | os.PathLike | Source] = (),
    *,
    db_path: str | os.PathLike | None = None,
    agent_model: str | None = None,
    title: str = "Lumen",
    **params: Any,
) -> dict[str, Any]:
    """
    The routes of a FlowDash app whose dashboards are built from Lumen's
    nodes over ``data``, for ``panel.serve`` or ``build_applications``.

    Parameters
    ----------
    data
        Files, URLs, database connection strings or sources. Source nodes
        name them by the refs ``source_refs`` gives.
    db_path
        The SQLite database dashboards are saved in; ``dashboards.db`` in
        the working directory by default.
    agent_model
        A pydantic-ai model, e.g. ``"openai:gpt-6-luna"``, for Lumen's
        FlowDash agent. Without it the app has no agent.
    title
        The app's title.
    **params
        Further ``FlowDashApp`` parameters.
    """
    sources = source_refs(resolve_sources(data))
    if agent_model:
        from ..agent import create_agent

        agent: Any = create_agent(agent_model)
    else:
        agent = False
    db_path = Path(db_path) if db_path else Path.cwd() / "dashboards.db"
    editor_options = dict(params.pop("editor_options", {}), lumen_sources=sources)
    return FlowDashApp.build_routes(
        project_dir=str(Path.cwd()),
        registry={},
        store=DashboardStore(str(db_path)),
        component_libraries=[LIBRARY, *params.pop("component_libraries", [])],
        editor_class=LumenFlowDash,
        editor_options=editor_options,
        agent=agent,
        agent_sensitive_components=[SOURCE_ID, *params.pop("agent_sensitive_components", [])],
        title=title,
        **params,
    )
