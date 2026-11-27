"""What only Lumen knows about a dashboard's report, added to FlowDash's report tools through its hooks."""

from __future__ import annotations

import logging

from typing import Any

from panel_flowdash.agent import DashboardOps

from ..flowdash import (
    DECKGL_ID, METRIC_ID, TABLE_VIEW_ID, VIEW_ID,
)
from . import _graph
from .edits import _cut, kind, spec_summary

try:
    import panel_flowdash.report as _report
except ImportError:
    _report = None

logger = logging.getLogger(__name__)

# panel-flowdash releases with report placeholder hints also take Lumen's catalogue additions.
HAS_REPORT_HOOKS = _report is not None and hasattr(_report, "PlaceholderHint")

VIEW_COMPONENTS = (VIEW_ID, TABLE_VIEW_ID, DECKGL_ID)
# The most characters Lumen adds after FlowDash's own catalogue, and per entry.
MAX_CONTEXT = 1500
_ENTRY = 200

# Added to the instructions of an agent with FlowDash's report writing tools, which carry the story rules.
REPORT_INSTRUCTIONS = """
- In a report, cite numbers from the data or value output of a SQL or metric node; a view's
  pipeline is not data. Place a total, share or ranking the text needs with add_sql or
  add_metric first.
"""


def _metric_line(name: str, config: dict[str, Any], metrics: dict[str, Any]) -> str | None:
    metric = metrics.get(config.get("metric"))
    if metric is None:
        return None
    picked = config.get("measures") or list(metric.measures)
    measures = [f"{m} = {metric.measures[m].expr}" for m in picked if m in metric.measures]
    what = f"sql {metric.sql}" if metric.sql is not None else "; ".join(measures)
    by = config.get("dimensions") or []
    parts = [f"metric {metric.name!r}", metric.description, what, f"by {', '.join(map(str, by))}" if by else ""]
    return f"- {name}: " + _cut(". ".join(p for p in parts if p), _ENTRY)


def report_context(ops: DashboardOps) -> str | None:
    """
    Lumen's additions to the data catalogue FlowDash's get_report gives the
    agent, for ``FlowDashApp(agent_report_context=...)``: what each chart,
    table or map shows, and the definition behind each metric node, so prose
    can say what a figure means. Nodes the user may not read are left out.
    """
    lines: list[str] = []
    for component in VIEW_COMPONENTS:
        for name in _graph.nodes(ops, component):
            try:
                spec = ops.node_config(name).get("spec")
            except Exception:
                continue
            lines.append(f"- {name} ({kind(component)}): " + _cut(spec_summary(component, spec), _ENTRY))
    metric_nodes = _graph.nodes(ops, METRIC_ID)
    metrics = _graph.metrics(ops) if metric_nodes else {}
    for name in metric_nodes:
        try:
            line = _metric_line(name, ops.node_config(name), metrics)
        except Exception as exc:
            logger.warning("Could not describe metric node %r for the report (%s).", name, type(exc).__name__)
            continue
        if line:
            lines.append(line)
    if not lines:
        return None
    text = "What Lumen's views show and the definitions behind metric nodes:"
    for i, line in enumerate(lines):
        if len(text) + len(line) + 1 > MAX_CONTEXT:
            text += f"\n- ... and {len(lines) - i} more."
            break
        text += "\n" + line
    return text
