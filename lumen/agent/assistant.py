"""The FlowDash agent with Lumen's data and authoring tools."""
from __future__ import annotations

from collections.abc import Collection
from typing import Any

from panel_flowdash.agent import (
    FlowDashDeps, create_agent as create_flowdash_agent,
)
from pydantic_ai import Agent

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
# which costs output tokens and time without changing the answers. The
# output cap stops a response that degenerates into whitespace.
MODEL_SETTINGS = {"openai_reasoning_effort": "none", "max_tokens": 4000}

# The dashboard tools a data agent needs besides its own: changing or
# removing a node it placed. Pass ``dashboard_tools=None`` for all of them.
DATA_DASHBOARD_TOOLS = ("set_config", "remove_node")


def create_agent(model: Any = None, *, instructions: str | None = None, model_settings: dict[str, Any] | None = None,
                 dashboard_tools: Collection[str] | None = DATA_DASHBOARD_TOOLS, **kwargs: Any) -> Agent[FlowDashDeps, str]:
    """
    The FlowDash agent with Lumen's data and authoring toolsets.

    Parameters
    ----------
    model
        A pydantic-ai model or model name, e.g. ``"openai:gpt-6-luna"``.
    instructions
        Extra instructions appended to Lumen's.
    model_settings
        pydantic-ai model settings for every request; defaults to
        ``MODEL_SETTINGS``. Settings a model does not know are ignored.
    dashboard_tools
        The FlowDash dashboard tools to offer, by default
        ``DATA_DASHBOARD_TOOLS``; None offers all of them.
    **kwargs
        Passed on to ``panel_flowdash.agent.create_agent``.
    """
    model_settings = MODEL_SETTINGS if model_settings is None else model_settings
    toolsets = [data_toolset(), authoring_toolset(), *kwargs.pop("toolsets", [])]
    text = INSTRUCTIONS + (f"\n{instructions}" if instructions else "")
    return create_flowdash_agent(model, instructions=text, toolsets=toolsets, tools=dashboard_tools,
                                 model_settings=dict(model_settings), **kwargs)
