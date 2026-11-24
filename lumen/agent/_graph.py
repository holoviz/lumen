"""Reading nodes and their live outputs from a FlowDash editor by node name."""
from __future__ import annotations

from typing import Any

from panel_flowdash.agent import DashboardOps, OperationError
from panel_flowdash.agent.yaml_format import node_names
from panel_flowdash.dataflow_engine import failure_text
from panel_flowdash.util import is_failure


def nodes(ops: DashboardOps, component: str | None = None) -> list[str]:
    """Names of the nodes on the canvas in the order they were placed, optionally of one component."""
    items = ops.editor.export_dashboard()["items"]
    names = node_names(items)
    return [
        names[item["instance_id"]] for item in items
        if component is None or item["component_id"] == component
    ]


def component_of(ops: DashboardOps, name: str) -> str:
    return ops._item(ops._instance(name))["component_id"]


def config_of(ops: DashboardOps, name: str) -> dict[str, Any]:
    return dict(ops._item(ops._instance(name)).get("config") or {})


def output(ops: DashboardOps, name: str, port: str) -> Any:
    """The current value of a node's output port; raises ``OperationError`` for a failed output."""
    state = ops.editor.graph.get_state(ops._instance(name))
    if state is None or port not in state.param:
        raise OperationError(f"'{name}' has no output '{port}'.")
    value = getattr(state, port)
    if is_failure(value):
        raise OperationError(f"'{name}' failed: {failure_text(value)}")
    return value
