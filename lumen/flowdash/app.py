"""
A FlowDash app over data the host provides, built from Lumen's nodes; what
``lumen serve --flowdash`` serves.
"""
from __future__ import annotations

import os
import re

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from panel_flowdash.agent.models import parse_models
from panel_flowdash.app import FlowDashApp
from panel_flowdash.dashboard_store import DashboardStore

from ..sources.base import Source
from ..sources.data import PROVIDED_SOURCE_NAME
from .nodes import SOURCE_ALIASES_KEY, SOURCE_ID, SOURCES_KEY

LIBRARY = "lumen.flowdash"


# param names a source it was not given a name for by its type and a counter.
_AUTO_NAME = re.compile(r"[A-Z]\w*\d{5}")
# Lumen names the table of a data file by its file name, e.g. turbines_csv.
_FILE_SUFFIX = re.compile(r"_(csv|tsv|parquet|parq|json|jsonl|ndjson)$")
# The ref of the source holding several data files.
FILES_REF = "files"


def _slug(text: str) -> str:
    return re.sub(r"\W+", "_", text).strip("_").lower()


def _ref_names(source: Source) -> tuple[int, str, list[str]]:
    """
    How ``source`` is named (by its name, files, location, table or type),
    the ref and the further names that resolve to it.
    """
    tables = list(source.get_tables())
    if source.name == PROVIDED_SOURCE_NAME:
        stems = sorted({_slug(_FILE_SUFFIX.sub("", table)) for table in tables} - {""})
        return 1, (stems[0] if len(stems) == 1 else FILES_REF), [FILES_REF, *stems]
    if not _AUTO_NAME.fullmatch(source.name) and (name := _slug(source.name)):
        return 0, name, []
    location = getattr(source, "uri", None) or getattr(source, "url", None)
    if isinstance(location, str) and location != ":memory:":
        path = re.sub(r"^\w+:/*", "/", location).split("?")[0]
        if stem := _slug(Path(path).stem):
            return 2, stem, []
    if len(tables) == 1 and (table := _slug(tables[0])):
        return 3, table, []
    return 4, _slug(type(source).__name__.removesuffix("Source")) or "source", []


def source_refs(sources: Sequence[Source]) -> dict[str, Source]:
    """
    The refs source nodes name ``sources`` by, in the order of ``sources``.

    A ref does not depend on the order or number of the sources, so a saved
    dashboard keeps its sources when files are added, removed or reordered:
    a source named by its host is named so, a database file or connection by
    its file or database name, and the data files ``lumen serve`` reads into
    one source by the file's stem, e.g. ``turbines`` for ``turbines.csv``,
    or ``files`` when there are several. Sources that would share a ref get
    ``_2``, ``_3`` and so on: a source named by its host first, then by
    location, type and tables, and only sources alike in all of these by
    their order.
    ``source_aliases`` gives the further names a source resolves by.
    """
    keys = {}
    for source in sources:
        kind, name, _ = _ref_names(source)
        location = getattr(source, "uri", None) or getattr(source, "url", None)
        keys[id(source)] = (name, kind, str(location or ""), type(source).__name__, *sorted(source.get_tables()))
    refs: dict[int, str] = {}
    taken: set[str] = set()
    for source in sorted(sources, key=lambda s: keys[id(s)]):
        base = keys[id(source)][0]
        ref, n = base, 1
        while ref in taken:
            n += 1
            ref = f"{base}_{n}"
        refs[id(source)] = ref
        taken.add(ref)
    return {refs[id(source)]: source for source in sources}


def source_aliases(refs: Mapping[str, Source]) -> dict[str, str]:
    """
    Further names that resolve to the refs of ``refs`` (as ``source_refs``
    orders them), for dashboards saved before refs were stable: ``data``,
    ``data_2``, ``data_3`` and so on name the sources by position, as
    ``lumen serve`` named them then, and ``files`` and the stem of each
    data file name the source holding the data files. A ref always wins
    over an alias of the same name.
    """
    aliases: dict[str, str] = {}
    for i, (ref, source) in enumerate(refs.items()):
        for alias in ("data" if i == 0 else f"data_{i + 1}", *_ref_names(source)[2]):
            if alias not in refs:
                aliases.setdefault(alias, ref)
    return aliases


def resolve_sources(data: Sequence[str | os.PathLike | Source]) -> list[Source]:
    """
    Sources for files, URLs, database connection strings or sources, read
    as ``lumen serve`` reads them for Lumen AI.
    """
    from ..sources.data import resolve_data

    return resolve_data([d if isinstance(d, Source) else str(d) for d in data])


def build_routes(
    data: Sequence[str | os.PathLike | Source] = (),
    *,
    db_path: str | os.PathLike | None = None,
    agent_model: str | Sequence[str] | None = None,
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
        name them by the refs ``source_refs`` gives, or by an alias from
        ``source_aliases``.
    db_path
        The SQLite database dashboards are saved in; ``dashboards.db`` in
        the working directory by default.
    agent_model
        A pydantic-ai model, e.g. ``"openai:gpt-6-luna"``, for Lumen's
        FlowDash agent, or several the person using the agent may pick
        from, each written as ``id`` or ``id=label``. The first is the
        default. Without it the app has no agent.
    title
        The app's title.
    **params
        Further ``FlowDashApp`` parameters.
    """
    sources = source_refs(resolve_sources(data))
    models = parse_models(agent_model, source="agent_model")
    if models:
        from ..agent import create_agent

        default = models[0]["id"]
        # With a choice every request names its model, so an unusable default must not fail here.
        agent: Any = create_agent(default, **({"defer_model_check": True} if len(models) > 1 else {}))
        params.setdefault("agent_model", default)
        params.setdefault("agent_models", models if len(models) > 1 else [])
    else:
        agent = False
    db_path = Path(db_path) if db_path else Path.cwd() / "dashboards.db"
    editor_options = dict(params.pop("editor_options", {}))
    editor_options["context"] = {
        **editor_options.get("context", {}), SOURCES_KEY: sources, SOURCE_ALIASES_KEY: source_aliases(sources),
    }
    return FlowDashApp.build_routes(
        project_dir=str(Path.cwd()),
        registry={},
        store=DashboardStore(str(db_path)),
        component_libraries=[LIBRARY, *params.pop("component_libraries", [])],
        editor_options=editor_options,
        agent=agent,
        agent_sensitive_components=[SOURCE_ID, *params.pop("agent_sensitive_components", [])],
        title=title,
        **params,
    )
