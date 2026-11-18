"""The FlowDash agent with Lumen's data and authoring tools."""
from __future__ import annotations

from typing import Any

from panel_flowdash.agent import (
    FlowDashDeps, create_agent as create_flowdash_agent,
)
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition

from .authoring import authoring_toolset
from .data import data_toolset

# The instructions stay the same for a whole conversation, whatever the
# dashboard holds, so the provider's prompt cache keeps them; what changes
# reaches the model through tool results.
INSTRUCTIONS = """\
## Answering questions about data
The user asks questions about the data of the dashboard's Lumen source nodes.
- Look before you write SQL: list_tables (it also names the source nodes, their SQL
  dialect and the SQL nodes already placed), describe_table for the columns you use, and
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

# gpt-6-luna and similar models otherwise reason before most tool calls,
# which costs output tokens and time without changing the answers.
MODEL_SETTINGS = {"openai_reasoning_effort": "none"}

# The dashboard tools a data agent has no use for: it neither stores,
# opens nor shares dashboards.
HIDDEN_TOOLS = frozenset({"save_dashboard", "delete_dashboard", "share_dashboard", "open_dashboard", "list_dashboards"})


async def _hide_tools(ctx: RunContext[FlowDashDeps], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return [tool for tool in tool_defs if tool.name not in HIDDEN_TOOLS]


def create_agent(model: Any = None, *, instructions: str | None = None, model_settings: dict[str, Any] | None = None,
                 **kwargs: Any) -> Agent[FlowDashDeps, str]:
    """
    The FlowDash agent with Lumen's data and authoring toolsets, without the
    dashboard tools in ``HIDDEN_TOOLS``.

    Parameters
    ----------
    model
        A pydantic-ai model or model name, e.g. ``"openai:gpt-6-luna"``.
    instructions
        Extra instructions appended to Lumen's.
    model_settings
        pydantic-ai model settings for every request; defaults to
        ``MODEL_SETTINGS``. Settings a model does not know are ignored.
    **kwargs
        Passed on to ``panel_flowdash.agent.create_agent``.
    """
    model_settings = MODEL_SETTINGS if model_settings is None else model_settings
    toolsets = [data_toolset(), authoring_toolset(), *kwargs.pop("toolsets", [])]
    capabilities = [PrepareTools(_hide_tools), *kwargs.pop("capabilities", [])]
    text = INSTRUCTIONS + (f"\n{instructions}" if instructions else "")
    return create_flowdash_agent(model, instructions=text, toolsets=toolsets, capabilities=capabilities,
                                 model_settings=dict(model_settings), **kwargs)
