"""Reading nodes, their config, live outputs and wiring through ``DashboardOps``."""
from __future__ import annotations

from typing import Any

import yaml

from panel_flowdash.agent import (
    DashboardOps, OperationError, OutputUnavailable,
)


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


def upstream(ops: DashboardOps, name: str) -> str | None:
    """The node feeding ``name``'s first connected input."""
    edges = yaml.safe_load(ops.get_dashboard()).get("edges") or []
    return next((edge["from"].split(".")[0] for edge in edges if edge["to"].split(".")[0] == name), None)


# The output port that carries a Lumen pipeline, by component.
PIPELINE_PORTS = {"Lumen/table": "pipeline", "Lumen/filter": "filtered", "Lumen/sql": "pipeline"}


def pipeline_port(ops: DashboardOps, name: str) -> str | None:
    """The port of node ``name`` that outputs a Lumen pipeline, if it has one."""
    return PIPELINE_PORTS.get(component_of(ops, name))


# FlowDash's built-in text components and the config field holding each one's text.
HEADING_ID, MARKDOWN_ID, DIVIDER_ID = "Text/Heading", "Text/Markdown", "Text/Divider"
TEXT_FIELDS = {HEADING_ID: "text", MARKDOWN_ID: "text", DIVIDER_ID: "label"}
