"""
A FlowDash app over data the host provides, built from Lumen's nodes; what
``lumen serve --flowdash`` serves.
"""
from __future__ import annotations

import logging
import os
import re
import weakref

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import param

from panel_flowdash.agent.models import parse_models
from panel_flowdash.app import FlowDashApp
from panel_flowdash.dashboard_store import DashboardStore

from ..sources.base import Source
from ..sources.data import PROVIDED_SOURCE_NAME
from .nodes import SOURCE_ALIASES_KEY, SOURCES_KEY

LIBRARY = "lumen.flowdash"


# Lumen reads files with these suffixes into one DuckDB source, see ``resolve_data``.
_DATA_FILES = (".parq", ".parquet", ".csv", ".json", ".tsv", ".jsonl", ".ndjson")
# The refs lumen serve gave by position before refs were named after the data.
_POSITIONAL = re.compile(r"data(_\d+)?")
# Names the source of several served files had before; still resolve as aliases.
FILES_REF = "files"

# The source the served data files are read into: the files in command-line order
# and the names a served folder or the files' folders gave it.
_FILES: weakref.WeakKeyDictionary[Source, tuple[list[_DataFile], list[str]]] = weakref.WeakKeyDictionary()
# The sources ``resolve_sources`` made from paths, URLs and connection strings, which
# ``name_sources`` may name; a source the host passed keeps its name.
_BUILT: weakref.WeakSet[Source] = weakref.WeakSet()
# The names of a served source from before ``name_sources`` named it by its ref.
_NAMES: weakref.WeakKeyDictionary[Source, tuple[str, list[str]]] = weakref.WeakKeyDictionary()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DataFile:
    """A data file ``lumen serve`` reads into the one DuckDB source of all data files."""

    text: str
    """The path or URL as given."""
    key: str
    """The resolved path, or the URL, so a file given twice is read once."""
    stem: str
    extension: str
    parent: str
    remote: bool

    @classmethod
    def of(cls, text: str) -> _DataFile | None:
        remote = "://" in text
        path = PurePosixPath(urlsplit(text).path) if remote else Path(text).resolve()
        name = path.name
        if not name.lower().endswith(_DATA_FILES):
            return None
        stem, _, extension = name.rpartition(".")
        # A folder counts only when given, never the working directory's name.
        given_folder = remote or Path(text).parent != Path(".")
        return cls(text, text if remote else str(path), _slug(stem) or "data", _slug(extension),
                   _slug(path.parent.name) if given_folder else "", remote)


def _slug(text: str) -> str:
    return re.sub(r"\W+", "_", text).strip("_").lower()


def _type_slug(source: Source) -> str:
    return _slug(type(source).__name__.removesuffix("Source")) or "source"


def _auto_named(source: Source) -> bool:
    """Whether param named ``source`` by its type and a counter, as it does without a name."""
    return re.fullmatch(rf"{re.escape(type(source).__name__)}\d{{5}}", source.name) is not None


def _numbered(names: Sequence[str]) -> list[str]:
    """
    ``names`` made unique in their order: the first of a name keeps it, later
    ones get ``_2``, ``_3`` and so on, never taking a name given as is.
    """
    reserved, taken, unique = set(names), set(), []
    for name in names:
        candidate, n = name, 1
        while candidate in taken or (candidate != name and candidate in reserved):
            n += 1
            candidate = f"{name}_{n}"
        taken.add(candidate)
        unique.append(candidate)
    return unique


def _served_files(
    data: Sequence[str | os.PathLike | Source],
) -> tuple[list[_DataFile], str | None, list[str]]:
    """
    The data files of ``data`` in command-line order, a folder standing for
    the data files in it by name, each file once; and the folder's name when
    a single folder is all the data files given; and the names of the
    folders given.
    """
    files: dict[str, _DataFile] = {}
    given = []
    for item in data:
        if not isinstance(item, (str, os.PathLike)):
            continue
        text = str(item)
        if "://" not in text and Path(text).is_dir():
            given.append(_slug(Path(text).resolve().name))
            texts = [str(p) for p in sorted(Path(text).iterdir()) if p.is_file()]
        elif _DataFile.of(text) is not None:
            given.append(None)
            texts = [text]
        else:
            continue
        for entry in texts:
            if (file := _DataFile.of(entry)) is not None:
                files.setdefault(file.key, file)
    folder = given[0] if len(given) == 1 else None
    return list(files.values()), folder, sorted({name for name in given if name})


def _source_name(files: Sequence[_DataFile], folder: str | None) -> str:
    """The name of the source of the served data files: the file, the folder, or else ``local`` or ``data``."""
    if folder:
        return folder
    if len(files) == 1:
        return files[0].stem
    return "data" if any(file.remote for file in files) else "local"


def _legacy_tables(files: Sequence[_DataFile], tables: Sequence[str]) -> dict[str, str]:
    """
    Names earlier releases gave the files' tables, by the table each now
    names: the path as given (``x_a_csv`` for ``x/a.csv``) and the file name
    with its extension (``a_csv``), alone or after its folder. A name two
    files had, or that is a table now, is left out.
    """
    from ..util import normalize_table_name

    claims: dict[str, set[str]] = {}
    for file, table in zip(files, tables, strict=True):
        names = {normalize_table_name(file.text), f"{file.stem}_{file.extension}"}
        if file.parent:
            names.add(f"{file.parent}_{file.stem}_{file.extension}")
        for name in names:
            claims.setdefault(name, set()).add(table)
    return {
        name: next(iter(targets)) for name, targets in claims.items()
        if len(targets) == 1 and name not in tables
    }


def resolve_sources(data: Sequence[str | os.PathLike | Source]) -> list[Source]:
    """
    Sources for files, folders, URLs, database connection strings or
    sources, read as ``lumen serve`` reads them for Lumen AI, in
    command-line order (the data files' source where the first of them is
    given), except for how the data files are named.

    The data files go into one source named after the data: the file's
    stem for one file, the folder's name for a folder, else ``local`` (or
    ``data`` when some are URLs). Each file's table is its stem, e.g.
    ``turbines`` for ``turbines.csv``; files of one stem get ``_2``, ``_3``
    in command-line order, so adding a file never renames a table a saved
    dashboard reads. The names earlier releases gave the tables still
    work in queries where only one file had them, but are not listed.
    """
    from ..sources.data import resolve_data
    from .nodes import add_legacy_tables, set_served_files

    files, folder, folders = _served_files(data)
    tables = _numbered([file.stem for file in files])
    stems = [file.stem for file in files]
    if clashing := [(table, file) for file, table in zip(files, tables, strict=True) if stems.count(file.stem) > 1]:
        logger.warning(
            "Data files share a name, so their tables are numbered in the order given: %s",
            ", ".join(f"{table}: {file.text}" for table, file in clashing),
        )
    items: list[Any] = []
    positions: list[int] = []
    first_file = None
    for position, item in enumerate(data):
        if not isinstance(item, (str, os.PathLike)):
            items.append(item)
            positions.append(position)
            continue
        text = str(item)
        if ("://" not in text and Path(text).is_dir()) or _DataFile.of(text) is not None:
            if first_file is None:
                first_file = position
                items.extend((table, file.text) for file, table in zip(files, tables, strict=True))
            continue
        items.append(text)
        positions.append(position)
    sources = resolve_data(items)
    provided = next((s for s in sources if s.name == PROVIDED_SOURCE_NAME), None) if files else None
    others = [s for s in sources if s is not provided]
    passed = {id(item) for item in data if isinstance(item, Source)}
    _BUILT.update(s for s in sources if id(s) not in passed)
    if provided is not None:
        with param.edit_constant(provided):
            provided.name = _source_name(files, folder)
        add_legacy_tables(provided, _legacy_tables(files, tables))
        _FILES[provided] = (files, folders)
        set_served_files(provided, {table: file.text for file, table in zip(files, tables, strict=True)})
    if len(others) != len(positions):
        # Data that resolve_data reads into sources of its own making, e.g. data
        # frames; its order is then the one to keep.
        return sources
    order = dict(zip(map(id, others), positions, strict=True))
    if provided is not None:
        order[id(provided)] = first_file
    return sorted(sources, key=lambda s: order[id(s)])


def _names(source: Source) -> tuple[str, list[str]]:
    """
    The name ``source`` is named by, and the names earlier releases of
    ``lumen serve --flowdash`` may have given it.
    """
    if source in _NAMES:
        return _NAMES[source]
    served = _FILES.get(source)
    if served is not None:
        files, folders = served
        stems = sorted({file.stem for file in files})
        legacy = [FILES_REF, "local", *stems, *(f"{stem}_{FILES_REF}" for stem in stems), *folders]
        return _slug(source.name) or "data", legacy
    tables = list(source.get_tables())
    if not _auto_named(source) and (name := _slug(source.name)):
        return name, [f"{name}_{_type_slug(source)}"]
    location = getattr(source, "uri", None) or getattr(source, "url", None)
    if isinstance(location, str) and location != ":memory:":
        parts = urlsplit(location)
        path = PurePosixPath(parts.path) if "://" in location else Path(location)
        if stem := _slug(path.stem):
            tag = _slug(path.suffix) or _slug(parts.scheme) or _type_slug(source)
            return stem, [f"{stem}_{tag}"]
    if len(tables) == 1 and (table := _slug(tables[0])):
        return table, [f"{table}_{_type_slug(source)}"]
    return _type_slug(source), []


def source_refs(sources: Sequence[Source]) -> dict[str, Source]:
    """
    The refs source nodes name ``sources`` by, given in command-line order
    as ``resolve_sources`` returns them.

    A ref is the source's name, not its position: the name ``resolve_sources``
    gives the data files' source, a name the host gave a source, or a
    database file's or connection's file or database name (``shop`` for
    ``shop.db``). Sources of one name get ``_2``, ``_3`` in command-line
    order, so the first given keeps the name and a source added later
    never takes over a saved dashboard's ref.
    """
    refs = _numbered([_names(source)[0] for source in sources])
    return dict(zip(refs, sources, strict=True))


def name_sources(refs: Mapping[str, Source]) -> None:
    """
    Name each source ``resolve_sources`` made by its ref, so its source node
    shows the name a dashboard uses, keeping the names it had for
    ``source_aliases``. A source the host passed keeps its own name, which
    other apps and its caches may rely on.
    """
    for ref, source in refs.items():
        if source not in _BUILT:
            continue
        _NAMES.setdefault(source, _names(source))
        if source.name != ref:
            with param.edit_constant(source):
                source.name = ref


def source_aliases(refs: Mapping[str, Source]) -> dict[str, str]:
    """
    Further names that resolve to the refs of ``refs``, for dashboards
    saved by earlier releases: ``data``, ``data_2`` and so on by position in
    the order those releases read the data (database files and connections
    first, then the data files), and the names they gave a source (``files``,
    a data file's stem, ``shop_db``). A ref always wins over an alias, and a
    name two sources may have had resolves to neither.
    """
    sources = list(refs.values())
    ref_of = {id(source): ref for ref, source in refs.items()}
    legacy_order = [s for s in sources if s not in _FILES] + [s for s in sources if s in _FILES]
    aliases = {
        ("data" if i == 0 else f"data_{i + 1}"): ref_of[id(source)] for i, source in enumerate(legacy_order)
    }
    claims: dict[str, set[str]] = {}
    for ref, source in refs.items():
        base, legacy = _names(source)
        for name in {base, *legacy}:
            claims.setdefault(name, set()).add(ref)
    for name, targets in claims.items():
        if name in refs or name in aliases or _POSITIONAL.fullmatch(name):
            continue
        if len(targets) == 1:
            aliases[name] = next(iter(targets))
    return {alias: ref for alias, ref in aliases.items() if alias not in refs}


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
    aliases = source_aliases(sources)
    name_sources(sources)
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
        **editor_options.get("context", {}), SOURCES_KEY: sources, SOURCE_ALIASES_KEY: aliases,
    }
    return FlowDashApp.build_routes(
        project_dir=str(Path.cwd()),
        registry={},
        store=DashboardStore(str(db_path)),
        component_libraries=[LIBRARY, *params.pop("component_libraries", [])],
        editor_options=editor_options,
        agent=agent,
        agent_sensitive_components=list(params.pop("agent_sensitive_components", [])),
        title=title,
        **params,
    )
