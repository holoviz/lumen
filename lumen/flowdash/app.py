"""
A FlowDash app over data the host provides, built from Lumen's nodes; what
``lumen serve --flowdash`` serves.
"""
from __future__ import annotations

import os
import re

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from panel_flowdash.agent.models import parse_models
from panel_flowdash.app import FlowDashApp
from panel_flowdash.dashboard_store import DashboardStore

from ..sources.base import Source
from ..sources.data import PROVIDED_SOURCE_NAME
from .nodes import SOURCE_ALIASES_KEY, SOURCE_ID, SOURCES_KEY

LIBRARY = "lumen.flowdash"


# Lumen reads files with these suffixes into one DuckDB source, see ``resolve_data``.
_DATA_FILES = (".parq", ".parquet", ".csv", ".json", ".tsv", ".jsonl", ".ndjson")
# Lumen names the table of a data file by its path, e.g. turbines_csv.
_FILE_SUFFIX = re.compile(r"_(csv|tsv|parquet|parq|json|jsonl|ndjson)$")
# The refs lumen serve gave by position before refs were stable; never issued now.
_POSITIONAL = re.compile(r"data(_\d+)?")
# The ref of the source holding several data files.
FILES_REF = "files"


def _slug(text: str) -> str:
    return re.sub(r"\W+", "_", text).strip("_").lower()


def _type_slug(source: Source) -> str:
    return _slug(type(source).__name__.removesuffix("Source")) or "source"


def _auto_named(source: Source) -> bool:
    """Whether param named ``source`` by its type and a counter, as it does without a name."""
    return re.fullmatch(rf"{re.escape(type(source).__name__)}\d{{5}}", source.name) is not None


def file_stems(data: Sequence[str | os.PathLike | Source]) -> list[str]:
    """
    The stems of the data files in ``data`` that Lumen reads into one
    source, from their file names, whatever directory or URL they are in.
    """
    return [stem for name in _file_names(data).values() if (stem := _slug(name.rsplit(".", 1)[0]))]


def _file_names(data: Sequence[str | os.PathLike | Source]) -> dict[str, str]:
    """
    The table names of the data files in ``data`` by their path: the file
    name, so a table is named the same from any directory. A file listed
    twice, under any spelling, is read once. When files of one name come
    from different directories none keeps the bare name: each is prefixed
    with as many of its parent directories as tell them apart (``x_a.csv``
    and ``y_a.csv`` for ``x/a.csv`` and ``y/a.csv``), never taking another
    file's name, so a query saved against one of them fails instead of
    reading the other.
    """
    found: dict[str, tuple[str, tuple[str, ...], str]] = {}
    for item in data:
        if isinstance(item, Source):
            continue
        text = str(item)
        remote = "://" in text
        path = PurePosixPath(urlsplit(text).path) if remote else Path(text).resolve()
        if path.name.lower().endswith(_DATA_FILES):
            key = text if remote else str(path)
            found[text] = (path.name, tuple(part for part in path.parent.parts if _slug(part)), key)
    files: dict[str, tuple[str, tuple[str, ...]]] = {key: (name, parents) for name, parents, key in found.values()}
    groups: dict[str, list[str]] = {}
    for key, (name, _) in sorted(files.items()):
        groups.setdefault(name.lower(), []).append(key)
    taken = set(groups)
    names: dict[str, str] = {}
    for keys in groups.values():
        if len(keys) == 1:
            names[keys[0]] = files[keys[0]][0]
            continue
        for depth in range(1, max(len(files[key][1]) for key in keys) + 2):
            qualified = {
                key: "_".join([*files[key][1][-depth:], files[key][0]]) if depth <= len(files[key][1])
                else "_".join([*files[key][1], str(i), files[key][0]])
                for i, key in enumerate(keys)
            }
            lowered = [name.lower() for name in qualified.values()]
            if len(set(lowered)) == len(lowered) and not taken & set(lowered):
                break
        names.update(qualified)
        taken.update(name.lower() for name in qualified.values())
    return {text: names[key] for text, (_, _, key) in found.items()}


def _names(source: Source, stems: Sequence[str]) -> tuple[str, str, list[str]]:
    """
    The ref ``source`` is named by unless another source claims it, the tag
    that tells it apart then, and further names that resolve to it.
    """
    tables = list(source.get_tables())
    if source.name == PROVIDED_SOURCE_NAME:
        stems = sorted(set(stems) or {_slug(_FILE_SUFFIX.sub("", table)) for table in tables} - {""})
        tagged = [f"{stem}_{FILES_REF}" for stem in stems]
        return (stems[0] if len(stems) == 1 else FILES_REF), FILES_REF, [FILES_REF, *stems, *tagged]
    if not _auto_named(source) and (name := _slug(source.name)):
        return name, _type_slug(source), [f"{name}_{_type_slug(source)}"]
    location = getattr(source, "uri", None) or getattr(source, "url", None)
    if isinstance(location, str) and location != ":memory:":
        parts = urlsplit(location)
        path = PurePosixPath(parts.path) if "://" in location else Path(location)
        if stem := _slug(path.stem):
            tag = _slug(path.suffix) or _slug(parts.scheme) or _type_slug(source)
            return stem, tag, [f"{stem}_{tag}"]
    if len(tables) == 1 and (table := _slug(tables[0])):
        return table, _type_slug(source), [f"{table}_{_type_slug(source)}"]
    return _type_slug(source), "", []


def _claims(sources: Sequence[Source], stems: Sequence[str]) -> tuple[dict[int, tuple[str, str, list[str]]], set[str]]:
    """Each source's names, and the names more than one source claims, or that look positional."""
    names = {id(source): _names(source, stems) for source in sources}
    counts: dict[str, int] = {}
    for base, _, extra in names.values():
        for name in {base, *extra}:
            counts[name] = counts.get(name, 0) + 1
    clashing = {name for name, n in counts.items() if n > 1 or _POSITIONAL.fullmatch(name)}
    return names, clashing


def source_refs(sources: Sequence[Source], stems: Sequence[str] = ()) -> dict[str, Source]:
    """
    The refs source nodes name ``sources`` by, in the order of ``sources``.

    A ref depends on the data's name, not on its position: a source named by
    its host is named so, a database file or connection by its file or
    database name, and the data files ``lumen serve`` reads into one source
    by the file's stem (``stems``, see ``file_stems``), e.g. ``turbines``
    for ``turbines.csv``, or ``files`` when there are several.

    A name that two sources claim (counting each data file's stem) is no
    source's ref: each of them is named with a tag of its kind instead,
    e.g. ``a_db`` and ``a_files`` for ``a.db`` and ``a.csv``, so a saved
    dashboard naming ``a`` reports the missing source rather than reading
    the other one. ``data`` and ``data_2``, ``data_3`` and so on, the refs
    ``lumen serve`` gave by position before, are tagged the same way.
    Sources still alike get ``_2``, ``_3`` in the order of their location,
    type and tables. ``source_aliases`` gives further names that resolve.
    """
    names, clashing = _claims(sources, stems)
    keys = {}
    for source in sources:
        base, tag, _ = names[id(source)]
        if base in clashing and tag:
            base = f"{base}_{tag}"
        location = getattr(source, "uri", None) or getattr(source, "url", None)
        keys[id(source)] = (base, str(location or ""), type(source).__name__, *sorted(source.get_tables()))
    refs: dict[int, str] = {}
    taken: set[str] = set()
    for source in sorted(sources, key=lambda s: keys[id(s)]):
        base = keys[id(source)][0]
        ref, n = base, 1
        while ref in taken or _POSITIONAL.fullmatch(ref):
            n += 1
            ref = f"{base}_{n}"
        refs[id(source)] = ref
        taken.add(ref)
    return {refs[id(source)]: source for source in sources}


def source_aliases(refs: Mapping[str, Source], stems: Sequence[str] = ()) -> dict[str, str]:
    """
    Further names that resolve to the refs of ``refs`` (as ``source_refs``
    orders them), for dashboards saved before: ``data``, ``data_2``,
    ``data_3`` and so on name the sources by position, as ``lumen serve``
    named them then, and ``files`` and the stem of each data file name the
    source holding the data files, so adding or removing a file keeps a
    saved ref; the tagged ref a source gets when its name clashes names it
    too. A name that is a ref, or that two sources claim, is no alias.
    """
    names, clashing = _claims(list(refs.values()), stems)
    aliases: dict[str, str] = {}
    for i, ref in enumerate(refs):
        aliases["data" if i == 0 else f"data_{i + 1}"] = ref
    for ref, source in refs.items():
        base, _, extra = names[id(source)]
        for alias in {base, *extra}:
            if alias not in refs and alias not in clashing and alias not in aliases:
                aliases[alias] = ref
    return aliases


def resolve_sources(data: Sequence[str | os.PathLike | Source]) -> list[Source]:
    """
    Sources for files, URLs, database connection strings or sources, read
    as ``lumen serve`` reads them for Lumen AI, except that a data file's
    table is named by its file name rather than its path (``turbines_csv``
    for ``turbines.csv`` in any directory), so a query keeps its tables
    when the same files are served from elsewhere.
    """
    from ..sources.data import resolve_data

    names = _file_names(data)
    return resolve_data([
        d if isinstance(d, Source) else (names[str(d)], str(d)) if str(d) in names else str(d) for d in data
    ])


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
    stems = file_stems(data)
    sources = source_refs(resolve_sources(data), stems)
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
        **editor_options.get("context", {}), SOURCES_KEY: sources, SOURCE_ALIASES_KEY: source_aliases(sources, stems),
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
