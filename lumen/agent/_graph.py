"""Reading nodes, their config, live outputs and wiring through ``DashboardOps``."""
from __future__ import annotations

import logging
import re
import weakref

from collections.abc import Collection
from typing import Any

import yaml

from panel_flowdash.agent import (
    DashboardOps, OperationError, OutputUnavailable,
)

from ..flowdash import (
    COMPONENTS, LumenViewNode, bound_sources, host_metrics,
)
from ..flowdash.data import hidden_reads_of

logger = logging.getLogger(__name__)


def metrics(ops: DashboardOps) -> dict[str, Any]:
    """
    The app's metrics by name, copied once into a plain dict. The host's
    mapping may re-read its project file on every access, so a caller reads
    this snapshot instead of the live view: a metric removed while it is
    read is left out, and when the mapping cannot be read at all there are
    none. A failure is logged by its type only, as its message may quote
    the project file. A metric computed from a table the catalog turns off
    for the assistant is left out, as the table is.
    """
    snapshot: dict[str, Any] = {}
    try:
        live = host_metrics(ops.editor.context)
        names = list(live)
    except Exception as exc:
        logger.warning("Could not read the app's metrics (%s).", type(exc).__name__)
        return snapshot
    for name in names:
        try:
            snapshot[name] = live[name]
        except KeyError:
            continue
        except Exception as exc:
            logger.warning("Could not read the app's metric %r (%s).", name, type(exc).__name__)
    # A metric computed from a table the catalog turns off for the assistant is left out with it.
    try:
        hidden = {name for name, metric in snapshot.items() if hidden_reads_of(metric, ops.editor.context)}
    except Exception as exc:
        logger.warning("Could not check the metrics' tables against the catalog (%s).", type(exc).__name__)
        hidden = set()
    return {name: metric for name, metric in snapshot.items() if name not in hidden}


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


def provenance(ops: DashboardOps, name: str) -> Any:
    """
    The provenance a node's numbers have, its own or the nearest upstream
    node's, e.g. the "Ad hoc" label of a chart of a SQL node's rows, or None
    when no node reports one or the editor has no provenance.
    """
    if not hasattr(ops.editor, "node_provenance"):
        return None
    found = ops.editor.node_provenance(ops._instance(name))
    return found[0] if found else None


def inputs_of(ops: DashboardOps, name: str) -> dict[str, Any]:
    """The literal values set on a node's unconnected inputs."""
    node = (yaml.safe_load(ops.get_dashboard()).get("nodes") or {}).get(name) or {}
    return dict(node.get("inputs") or {})


def _edges(ops: DashboardOps) -> list[dict[str, str]]:
    return yaml.safe_load(ops.get_dashboard()).get("edges") or []


def upstream(ops: DashboardOps, name: str) -> str | None:
    """The node feeding ``name``'s first connected input."""
    return next((edge["from"].split(".")[0] for edge in _edges(ops) if edge["to"].split(".")[0] == name), None)


def feeding(ops: DashboardOps, target: str) -> str | None:
    """The output port, as ``node.port``, connected to the input port ``target`` (``node.port``)."""
    return next((edge["from"] for edge in _edges(ops) if edge["to"] == target), None)


def feeding_all(ops: DashboardOps, target: str) -> list[str]:
    """Every output port, as ``node.port``, connected to the input port ``target``, in connection order."""
    return [edge["from"] for edge in _edges(ops) if edge["to"] == target]


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
PIPELINE_PORTS = {
    "Lumen/table": "pipeline", "Lumen/filter": "filtered", "Lumen/sql": "pipeline", "Lumen/metric": "pipeline",
    "Lumen/aggregate": "result", "Lumen/sort": "result", "Lumen/select": "result", "Lumen/derive": "result",
}


def pipeline_port(ops: DashboardOps, name: str) -> str | None:
    """The port of node ``name`` that outputs a Lumen pipeline, if it has one."""
    return PIPELINE_PORTS.get(component_of(ops, name))


# FlowDash's built-in text components and the config field holding each one's text.
HEADING_ID, MARKDOWN_ID, DIVIDER_ID = "Text/Heading", "Text/Markdown", "Text/Divider"
TEXT_FIELDS = {HEADING_ID: "text", MARKDOWN_ID: "text", DIVIDER_ID: "label"}


def view_class(component: str) -> type | None:
    """The ``LumenViewNode`` class registered as ``component``, or None for any other component."""
    cls = COMPONENTS.get(component)
    return cls if isinstance(cls, type) and issubclass(cls, LumenViewNode) else None


# The run of the operations' latest edit and the copies panel-flowdash's ``ExplorationOps``
# made of earlier answers' nodes in it, by the original's name, which later edits in the run reach.
_COPIES: weakref.WeakKeyDictionary[Any, tuple[str | None, dict[str, str]]] = weakref.WeakKeyDictionary()


def edited(ops: DashboardOps, name: str, before: Collection[str]) -> str:
    """
    The node an edit of node ``name`` changed: ``name``, or the copy of it
    the edit reached, as panel-flowdash's ``ExplorationOps`` copies an
    earlier answer's node to change it, once per run. A copy the edit
    placed is one of the nodes not in ``before``, named after the original
    with a number, e.g. ``totals_2``; a later edit in the same run by the
    original's name reaches that copy again.
    """
    from panel_agents import current_run_id

    run = current_run_id()
    if (entry := _COPIES.get(ops)) is None or entry[0] != run:
        entry = _COPIES[ops] = (run, {})
    copies = entry[1]
    base = re.escape(re.sub(r"_\d+$", "", name))
    copy = re.compile(base + r"_\d+")
    component = component_of(ops, name)
    placed = next(
        (n for n in nodes(ops) if n not in before and copy.fullmatch(n) and component_of(ops, n) == component), None,
    )
    if placed is not None:
        copies[name] = placed
        return placed
    earlier = copies.get(name)
    return earlier if earlier is not None and earlier in nodes(ops) else name
