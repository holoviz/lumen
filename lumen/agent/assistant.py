"""The FlowDash agent with Lumen's data and authoring tools."""
from __future__ import annotations

from typing import Any

from panel_flowdash.agent import (
    FlowDashDeps, OperationError, create_agent as create_flowdash_agent,
)
from pydantic_ai import Agent, RunContext

from ..flowdash import SOURCE_ID, SQL_ID, VIEW_ID
from ..sources.base import BaseSQLSource
from . import _graph
from .authoring import authoring_toolset
from .data import data_toolset

INSTRUCTIONS = """\
## Answering questions about data
The user asks questions about the data of the dashboard's Lumen source nodes.
- Look before you write SQL: list_tables, describe_table for the columns you use, and
  distinct_values for the stored form of each text value you filter on. run_sql is for
  checks of your own; the user does not see its result.
- Place each result that answers the question with add_sql, and each chart with add_chart
  on a SQL node. Every number you state must come from a node you placed or read; never
  compute or estimate one yourself.
- Name nodes descriptively in snake_case.
- Write one SELECT or WITH query per node. Do not add LIMIT unless asked, keep the column
  names of the data unless asked to name them, and leave out NULLs unless the question is
  about missing data.
- Finish with a short answer in text.
"""

DIALECT_NOTES = {
    "duckdb": """\
DuckDB notes:
- Quote column names with double quotes and strings with single quotes.
- Range predicates: "date" >= '2023-06-01' AND "date" < '2023-09-01'.
- Window filtering: QUALIFY ROW_NUMBER() OVER (...) <= 5.
- Use TRY_CAST for conversions that may fail.""",
    "snowflake": "Snowflake notes:\n- No quotes around database, schema or table names.",
}


def lumen_context(ops) -> str:
    """The source nodes with their SQL dialects, the dialect notes, and the Lumen nodes placed so far."""
    lines, dialects = [], []
    for name in _graph.nodes(ops, SOURCE_ID):
        try:
            source = _graph.output(ops, name, "source")
        except OperationError:
            continue
        if isinstance(source, BaseSQLSource):
            lines.append(f"- {name} ({source.dialect} SQL)")
            dialects.append(source.dialect)
    text = "Source nodes:\n" + ("\n".join(lines) or "- none")
    notes = [DIALECT_NOTES[d] for d in dict.fromkeys(dialects) if d in DIALECT_NOTES]
    if notes:
        text += "\n\n" + "\n\n".join(notes)
    placed = []
    for name in _graph.nodes(ops):
        component = _graph.component_of(ops, name)
        if component == SQL_ID:
            try:
                frame = _graph.output(ops, name, "data")
                columns = ", ".join(map(str, frame.columns)) if frame is not None else "no result"
            except OperationError as exc:
                columns = str(exc)
            placed.append(f"- {name} (SQL; columns: {columns})")
        elif component == VIEW_ID:
            placed.append(f"- {name} (chart)")
    if placed:
        text += "\n\nLumen nodes on the dashboard:\n" + "\n".join(placed)
    return text


def create_agent(model: Any = None, *, instructions: str | None = None, **kwargs: Any) -> Agent[FlowDashDeps, str]:
    """
    The FlowDash agent with Lumen's data and authoring toolsets.

    Parameters
    ----------
    model
        A pydantic-ai model or model name, e.g. ``"openai:gpt-6-luna"``.
    instructions
        Extra instructions appended to Lumen's.
    **kwargs
        Passed on to ``panel_flowdash.agent.create_agent``.
    """
    toolsets = [data_toolset(), authoring_toolset(), *kwargs.pop("toolsets", [])]
    text = INSTRUCTIONS + (f"\n{instructions}" if instructions else "")
    agent = create_flowdash_agent(model, instructions=text, toolsets=toolsets, **kwargs)

    @agent.instructions
    def data_context(ctx: RunContext[FlowDashDeps]) -> str:
        return lumen_context(ctx.deps.ops)

    return agent
