"""
The project file: one YAML file of named sections that configure an app,
e.g. ``metrics:``, with every value's line and column kept so a problem is
reported where it is written.

``lumen serve --flowdash`` reads ``flowdash.yaml`` (``PROJECT_FILE``),
given with ``--project`` or found next to the data or in the working
directory (``find_project``). Each section is read by whoever registered it
(``register_section``); a top-level key no one registered is ignored with a
warning, so a file written for a newer app still loads.

The module depends on PyYAML only, so other apps can share the file and
this loader.
"""
from __future__ import annotations

import logging
import os

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_FILE = "flowdash.yaml"

logger = logging.getLogger(__name__)

Path_ = tuple[str | int, ...]

# The last element of a path that names where a key is written rather than its value.
KEY = object()

_SECTIONS: dict[str, str] = {}


def register_section(name: str, description: str) -> None:
    """Declare a top-level section of the project file, so loading it does not warn."""
    _SECTIONS[name] = description


def known_sections() -> dict[str, str]:
    return dict(_SECTIONS)


@dataclass(frozen=True)
class Problem:
    message: str
    path: str | None = None
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        where = ":".join(str(part) for part in (self.path, self.line, self.column) if part is not None)
        return f"{where}: {self.message}" if where else self.message


class ProjectError(ValueError):
    """Problems of a project file, each with the file, line and column it concerns."""

    def __init__(self, problems: Sequence[Problem]):
        self.problems = list(problems)
        super().__init__("\n".join(map(str, self.problems)))


@dataclass
class ProjectFile:
    """
    A loaded project file: its data and where each key and value is
    written, by path, e.g. ``("metrics", "capacity", "table")``.
    """

    path: str | None
    data: dict[str, Any]
    positions: dict[Path_, tuple[int, int]] = field(default_factory=dict)

    def section(self, name: str) -> Any:
        return self.data.get(name)

    def position(self, path: Path_) -> tuple[int | None, int | None]:
        """The 1-based line and column of the value at ``path``, or of its nearest enclosing value."""
        path = tuple(path)
        while path:
            if path in self.positions:
                return self.positions[path]
            path = path[:-1]
        return self.positions.get((), (None, None))

    def problem(self, path: Path_, message: str, key: bool = False) -> Problem:
        """A problem at the value at ``path``, or with ``key`` at the key naming it."""
        line, column = self.position((*path, KEY) if key and path else path)
        return Problem(message, self.path, line, column)


def _mark(node: yaml.Node) -> tuple[int, int]:
    return node.start_mark.line + 1, node.start_mark.column + 1


class _Loader(yaml.SafeLoader):
    pass


def _build(loader: _Loader, node: yaml.Node, path: Path_, positions: dict, problems: list, source: str | None):
    positions[path] = _mark(node)
    if isinstance(node, yaml.MappingNode):
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if not isinstance(key, (str, int, float, bool)) or key is None:
                problems.append(Problem("Keys must be text.", source, *_mark(key_node)))
                continue
            if key in result:
                problems.append(Problem(f"{key!r} is given twice.", source, *_mark(key_node)))
                continue
            result[key] = _build(loader, value_node, (*path, key), positions, problems, source)
            # A mapping or list is reported where its key is, a scalar where its value is.
            if isinstance(value_node, (yaml.MappingNode, yaml.SequenceNode)) or value_node.tag == "tag:yaml.org,2002:null":
                positions[(*path, key)] = _mark(key_node)
            positions[(*path, key, KEY)] = _mark(key_node)
        return result
    if isinstance(node, yaml.SequenceNode):
        return [
            _build(loader, item, (*path, i), positions, problems, source) for i, item in enumerate(node.value)
        ]
    return loader.construct_object(node, deep=True)


def parse_project(text: str, path: str | os.PathLike | None = None) -> ProjectFile:
    """
    The project file in ``text``, with the position of every value.
    Raises ``ProjectError`` if it is not YAML, not a mapping or gives a key
    twice. Sections are not checked; unknown ones are warned about.
    """
    name = str(path) if path is not None else None
    loader = _Loader(text)
    try:
        try:
            node = loader.get_single_node()
        except yaml.MarkedYAMLError as exc:
            mark = exc.problem_mark or exc.context_mark
            line, column = (mark.line + 1, mark.column + 1) if mark else (None, None)
            raise ProjectError([Problem(f"Not valid YAML: {exc.problem or exc}", name, line, column)]) from None
        if node is None:
            return ProjectFile(name, {})
        positions: dict[Path_, tuple[int, int]] = {}
        problems: list[Problem] = []
        try:
            data = _build(loader, node, (), positions, problems, name)
        except yaml.MarkedYAMLError as exc:
            mark = exc.problem_mark or exc.context_mark
            line, column = (mark.line + 1, mark.column + 1) if mark else (None, None)
            raise ProjectError([Problem(f"Not valid YAML: {exc.problem or exc}", name, line, column)]) from None
    finally:
        loader.dispose()
    if not isinstance(data, dict):
        raise ProjectError([Problem("A project file is a mapping of sections, e.g. `metrics:`.", name, 1, 1)])
    if problems:
        raise ProjectError(problems)
    project = ProjectFile(name, data, positions)
    for key in data:
        if key not in _SECTIONS:
            known = ", ".join(f"`{s}`" for s in sorted(_SECTIONS)) or "none"
            logger.warning("%s", project.problem((key,), key=True, message=f"Ignoring the unknown section {key!r}; the known sections are: {known}."))
    return project


def load_project(path: str | os.PathLike) -> ProjectFile:
    """Read and parse the project file at ``path``, see ``parse_project``."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProjectError([Problem(f"Cannot read the project file: {exc.strerror or exc}", str(path))]) from None
    return parse_project(text, path)


def find_project(data: Iterable[Any] = (), cwd: str | os.PathLike | None = None) -> Path | None:
    """
    The project file an app serving ``data`` reads when none is given:
    ``PROJECT_FILE`` in the directory of the first local data file that has
    one, else in ``cwd`` (the working directory by default).
    """
    candidates: list[Path] = []
    for item in data:
        if not isinstance(item, (str, os.PathLike)) or "://" in str(item):
            continue
        path = Path(item)
        if path.exists():
            candidates.append((path if path.is_dir() else path.parent).resolve() / PROJECT_FILE)
    candidates.append(Path(cwd or Path.cwd()).resolve() / PROJECT_FILE)
    return next((candidate for candidate in dict.fromkeys(candidates) if candidate.is_file()), None)
