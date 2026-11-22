"""What keeps the Explore view's agent from changing the dashboard it answers questions about."""
from __future__ import annotations

import weakref

from collections import OrderedDict
from typing import Any

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import ModelRetry
from pydantic_ai.capabilities import AbstractCapability

try:
    from panel_flowdash.agent import ExplorationOps
except ImportError:
    ExplorationOps = None

# Runs whose placed nodes are remembered at once; older runs have ended.
RUNS_REMEMBERED = 64

# The tools that place a node named by their ``name`` argument.
PLACING_TOOLS = ("add_sql", "add_chart")


class _Placed:
    """The nodes one run placed: their live components, and the ids of those without one."""

    def __init__(self):
        self.components = weakref.WeakSet()
        self.ids: set[str] = set()

    def add(self, node: tuple[str, Any]):
        instance, component = node
        if component is not None:
            self.components.add(component)
        else:
            self.ids.add(instance)

    def __contains__(self, node: tuple[str, Any]) -> bool:
        instance, component = node
        return component in self.components if component is not None else instance in self.ids


class OwnNodesOnly(AbstractCapability[FlowDashDeps]):
    """
    Allows ``set_config`` only on a node the current run placed with
    ``add_sql`` or ``add_chart``, so a node of the dashboard, an earlier
    answer or a node someone else adds while the run goes on stays as it is,
    as Explore's answers do. Nodes are told apart by their live component,
    so a node that replaces one of the run's under its name is not the
    run's; a node without one, e.g. of a function component, by its id,
    and only one the run placed may change.

    panel-flowdash's ``ExplorationOps`` freezes earlier answers itself and
    copies one on an edit. Until a release has it, the Explore runner works
    on the dashboard's own ops, and this keeps the freeze; on an
    ``ExplorationOps`` it steps aside, so the copy happens.
    """

    def __init__(self):
        self._placed: OrderedDict[str, _Placed] = OrderedDict()

    def _run(self, ctx) -> _Placed:
        run = ctx.run_id or ""
        if run not in self._placed:
            self._placed[run] = _Placed()
            while len(self._placed) > RUNS_REMEMBERED:
                self._placed.popitem(last=False)
        return self._placed[run]

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        if ExplorationOps is not None and isinstance(ctx.deps.ops, ExplorationOps):
            return await handler(args)
        placed = self._run(ctx)
        ops = ctx.deps.ops
        name = args.get("name")
        if tool_def.name == "set_config" and (node := _node(ops, name)) is not None and node not in placed:
            raise ModelRetry(
                f"{name!r} was not placed for this question: the dashboard's nodes and earlier answers stay as "
                f"they are in Explore. Build on it instead, e.g. add_sql(source={name!r}) to query its result."
            )
        result = await handler(args)
        if tool_def.name in PLACING_TOOLS and (node := _node(ops, name)) is not None:
            placed.add(node)
        return result


def _node(ops, name) -> tuple[str, Any] | None:
    """The instance id and live component (None for a function component) of node ``name``, or None if there is none."""
    try:
        instance = ops._instance(name) if isinstance(name, str) else None
    except OperationError:
        return None
    return None if instance is None else (instance, ops.editor._component_instances.get(instance))
