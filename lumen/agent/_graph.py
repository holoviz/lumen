"""Reading nodes, their config, live outputs and wiring through ``DashboardOps``."""
from __future__ import annotations

from typing import Any

import yaml

from panel_flowdash.agent import DashboardOps


def nodes(ops: DashboardOps, component: str | None = None) -> list[str]:
    """Names of the nodes in the order they were placed, optionally of one component."""
    return [node["name"] for node in ops.nodes(component)]


def component_of(ops: DashboardOps, name: str) -> str:
    return next(node["component"] for node in ops.nodes() if node["name"] == name)


def config_of(ops: DashboardOps, name: str) -> dict[str, Any]:
    return ops.node_config(name)


def output(ops: DashboardOps, name: str, port: str) -> Any:
    """The current value of a node's output port; raises ``OperationError`` for a failed output."""
    return ops.node_output(name, port)


def upstream(ops: DashboardOps, name: str) -> str | None:
    """The node feeding ``name``'s first connected input."""
    edges = yaml.safe_load(ops.get_dashboard()).get("edges") or []
    return next((edge["from"].split(".")[0] for edge in edges if edge["to"].split(".")[0] == name), None)
