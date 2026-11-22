"""Reading nodes, their config, live outputs and wiring through ``DashboardOps``."""
from __future__ import annotations

import re

from collections.abc import Collection
from typing import Any

import yaml

from panel_flowdash.agent import (
    DashboardOps, OperationError, OutputUnavailable,
)

from ..flowdash import bound_sources


def nodes(ops: DashboardOps, component: str | None = None) -> list[str]:
    """Names of the nodes in the order they were placed, optionally of one component."""
    return [node["name"] for node in ops.nodes(component)]


def component_of(ops: DashboardOps, name: str) -> str:
    return next(node["component"] for node in ops.nodes() if node["name"] == name)


def config_of(ops: DashboardOps, name: str) -> dict[str, Any]:
    return ops.node_config(name)


def output(ops: DashboardOps, name: str, port: str) -> Any:
    """
    The current value of a node's output port. A node that is waiting or
    failed raises ``OperationError`` with the reason, which the authoring
    tools pass to the model to fix.
    """
    try:
        return ops.node_output(f"{name}.{port}")
    except OutputUnavailable as exc:
        raise OperationError(str(exc)) from exc


def _edges(ops: DashboardOps) -> list[dict[str, str]]:
    return yaml.safe_load(ops.get_dashboard()).get("edges") or []


def upstream(ops: DashboardOps, name: str) -> str | None:
    """The node feeding ``name``'s first connected input."""
    return next((edge["from"].split(".")[0] for edge in _edges(ops) if edge["to"].split(".")[0] == name), None)


def feeding(ops: DashboardOps, target: str) -> str | None:
    """The output port, as ``node.port``, connected to the input port ``target`` (``node.port``)."""
    return next((edge["from"] for edge in _edges(ops) if edge["to"] == target), None)


def reading(ops: DashboardOps, output: str) -> list[str]:
    """The nodes whose inputs read the output port ``output`` (``node.port``)."""
    return list(dict.fromkeys(edge["to"].split(".")[0] for edge in _edges(ops) if edge["from"] == output))


def input_port(ops: DashboardOps, name: str) -> str | None:
    """
    The output a new SQL node reading node ``name`` connects to when that is
    an earlier SQL node: its ``result``. None for a source node, a name that
    is no node or a name the app provides a source under, which names that
    source; ``OperationError`` for any other node.
    """
    if name not in nodes(ops) or name in bound_sources(ops.editor):
        return None
    component = component_of(ops, name)
    if component == "Lumen/sql":
        return f"{name}.result"
    if component == "Lumen/source":
        return None
    raise OperationError(f"{name!r} is a {component} node; pass a source or an earlier SQL node.")


# The output port that carries a Lumen pipeline, by component.
PIPELINE_PORTS = {"Lumen/table": "pipeline", "Lumen/filter": "filtered", "Lumen/sql": "pipeline"}


def pipeline_port(ops: DashboardOps, name: str) -> str | None:
    """The port of node ``name`` that outputs a Lumen pipeline, if it has one."""
    return PIPELINE_PORTS.get(component_of(ops, name))


def edited(ops: DashboardOps, name: str, before: Collection[str]) -> str:
    """
    The node an edit of node ``name`` changed: ``name``, or the copy of it
    the edit placed, one of the nodes not in ``before``, as panel-flowdash's
    ``ExplorationOps`` copies an earlier answer's node to change it. The
    copy is named after the original with a number, e.g. ``totals_2``.
    """
    base = re.escape(re.sub(r"_\d+$", "", name))
    copy = re.compile(base + r"_\d+")
    component = component_of(ops, name)
    return next(
        (n for n in nodes(ops) if n not in before and copy.fullmatch(n) and component_of(ops, n) == component), name,
    )
