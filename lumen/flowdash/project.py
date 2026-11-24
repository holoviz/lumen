"""
The project file, ``flowdash.yaml``: one YAML file of named sections that
configure an app, e.g. ``metrics:``, with every value's line and column kept
so a problem is reported where it is written.

panel-flowdash's ``panel_flowdash.project`` reads it, and Lumen registers
its ``metrics:`` section there (see ``lumen.flowdash.metrics``). This module
re-exports that loader, and on a panel-flowdash release without one gives a
fallback with the same API that only reads, so it can be deleted once Lumen
requires a panel-flowdash with the loader.
"""
from __future__ import annotations

import logging
import os

from collections.abc import (
    Callable, Iterable, Mapping, Sequence,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from panel_flowdash.project import (
        KEY, PROJECT_FILE, Problem, Project, ProjectError, ProjectSection,
        Section, SectionError, load_project, parse_project, register_section,
        sections,
    )
except ImportError:
    import yaml

    PROJECT_FILE = "flowdash.yaml"

    logger = logging.getLogger(__name__)

    # The last element of a path that names where a key is written rather than its value.
    KEY = object()

    @dataclass(frozen=True)
    class Problem:  # type: ignore[no-redef]
        message: str
        path: str | None = None
        line: int | None = None
        column: int | None = None

        def __str__(self) -> str:
            where = ":".join(str(part) for part in (self.path, self.line, self.column) if part is not None)
            return f"{where}: {self.message}" if where else self.message

    class ProjectError(ValueError):  # type: ignore[no-redef]
        """Problems of a project file, each with the file, line and column it concerns."""

        def __init__(self, problems: Sequence[Problem]):
            self.problems = list(problems)
            super().__init__("\n".join(map(str, self.problems)))

    class SectionError(ValueError):  # type: ignore[no-redef]
        """A section loader's problem at ``path`` inside its section."""

        def __init__(self, message: str, path: Sequence[str | int] = (), key: bool = False):
            super().__init__(message)
            self.message, self.path, self.key = message, tuple(path), key

    @dataclass(frozen=True)
    class ProjectSection:  # type: ignore[no-redef]
        name: str
        loader: Callable[[Any], Any]
        title: str = ""
        icon: str | None = None
        description: str = ""

    _SECTIONS: dict[str, ProjectSection] = {}

    def register_section(section: ProjectSection) -> ProjectSection:  # type: ignore[no-redef]
        _SECTIONS[section.name] = section
        return section

    def sections() -> dict[str, ProjectSection]:  # type: ignore[no-redef]
        return dict(_SECTIONS)

    @dataclass(frozen=True)
    class Section:  # type: ignore[no-redef]
        """One top-level section of a project file, as a loader receives it."""

        name: str
        value: Any
        project: Project

        @property
        def context(self) -> Mapping[str, Any]:
            return self.project.context

        def problem(self, path: Sequence[str | int], message: str, key: bool = False) -> Problem:
            return self.project.problem((self.name, *path), message, key=key)

        def warn(self, path: Sequence[str | int], message: str, key: bool = False) -> Problem:
            problem = self.problem(path, message, key=key)
            self.project.warnings.append(problem)
            logger.warning("%s", problem)
            return problem

    @dataclass
    class Project:  # type: ignore[no-redef]
        """A loaded project file: its data, where each value is written and each section's value."""

        path: str | None = None
        data: dict[str, Any] = field(default_factory=dict)
        positions: dict[tuple, tuple[int, int]] = field(default_factory=dict)
        context: Mapping[str, Any] = field(default_factory=dict)
        warnings: list[Problem] = field(default_factory=list)
        _values: dict[str, Any] = field(default_factory=dict, repr=False)

        def get(self, name: str, default: Any = None) -> Any:
            if name not in self.data:
                return default
            return self._values.get(name, self.data[name])

        def position(self, path: Sequence) -> tuple[int | None, int | None]:
            path = tuple(path)
            while path:
                if path in self.positions:
                    return self.positions[path]
                path = path[:-1]
            return self.positions.get((), (None, None))

        def problem(self, path: Sequence, message: str, key: bool = False) -> Problem:
            path = tuple(path)
            line, column = self.position((*path, KEY) if key and path else path)
            return Problem(message, self.path, line, column)

        def _load_all(self) -> None:
            problems: list[Problem] = []
            for name in self.data:
                section = _SECTIONS.get(name)
                if section is None:
                    known = ", ".join(f"`{s}`" for s in sorted(_SECTIONS)) or "none"
                    warning = self.problem(
                        (name,), f"Ignoring the unknown section {name!r}; the known sections are: {known}.", key=True,
                    )
                    self.warnings.append(warning)
                    logger.warning("%s", warning)
                    continue
                try:
                    self._values[name] = section.loader(Section(name, self.data[name], self))
                except ProjectError as exc:
                    problems.extend(exc.problems)
                except SectionError as exc:
                    problems.append(self.problem((name, *exc.path), exc.message, key=exc.key))
                except (TypeError, ValueError) as exc:
                    problems.append(self.problem((name,), f"Section {name!r}: {exc}", key=True))
            if problems:
                raise ProjectError(problems)

    def _mark(node: yaml.Node) -> tuple[int, int]:
        return node.start_mark.line + 1, node.start_mark.column + 1

    def _build(loader: yaml.SafeLoader, node: yaml.Node, path: tuple, positions: dict, problems: list,
               source: str | None) -> Any:
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
                if isinstance(value_node, (yaml.MappingNode, yaml.SequenceNode)) or value_node.tag.endswith(":null"):
                    positions[(*path, key)] = _mark(key_node)
                positions[(*path, key, KEY)] = _mark(key_node)
            return result
        if isinstance(node, yaml.SequenceNode):
            return [_build(loader, item, (*path, i), positions, problems, source) for i, item in enumerate(node.value)]
        return loader.construct_object(node, deep=True)

    def parse_project(text: str, path: str | os.PathLike | None = None,  # type: ignore[no-redef]
                      context: Mapping[str, Any] | None = None) -> Project:
        """
        Parse the project file in ``text`` and load each registered section
        with ``context``. Raises ``ProjectError`` with every problem found.
        """
        name = str(path) if path is not None else None
        loader = yaml.SafeLoader(text)
        problems: list[Problem] = []
        positions: dict[tuple, tuple[int, int]] = {}
        try:
            node = loader.get_single_node()
            data = _build(loader, node, (), positions, problems, name) if node is not None else {}
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
        project = Project(name, data, positions, dict(context or {}))
        project._load_all()
        return project

    def load_project(path: str | os.PathLike,  # type: ignore[no-redef]
                     context: Mapping[str, Any] | None = None) -> Project:
        """Read and parse the project file at ``path``, see ``parse_project``."""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProjectError([Problem(f"Cannot read the project file: {exc.strerror or exc}", str(path))]) from None
        return parse_project(text, path, context)


def project_beside(data: Iterable[Any]) -> Path | None:
    """
    The project file an app serving ``data`` reads when none is given:
    ``PROJECT_FILE`` in the directory of the first local data file or
    folder that has one. The working directory is not searched.
    """
    for item in data:
        if not isinstance(item, (str, os.PathLike)) or "://" in str(item):
            continue
        path = Path(item)
        if path.exists():
            candidate = (path if path.is_dir() else path.parent).resolve() / PROJECT_FILE
            if candidate.is_file():
                return candidate
    return None


__all__ = [
    "KEY", "PROJECT_FILE", "Problem", "Project", "ProjectError", "ProjectSection", "Section",
    "SectionError", "load_project", "parse_project", "project_beside", "register_section", "sections",
]
