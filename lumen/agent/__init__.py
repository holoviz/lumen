"""
One pydantic-ai agent that answers questions about data by building a FlowDash
dashboard of Lumen nodes. Experimental: everything here may change.

Needs panel-flowdash with its agent extra (panel-agents and pydantic-ai),
which Lumen does not install; nothing else in Lumen imports this package.
"""
try:
    import panel_agents  # noqa: F401
    import panel_flowdash  # noqa: F401
    import pydantic_ai  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lumen.agent needs panel-flowdash with its agent extra, which is not installed: "
        "pip install 'panel-flowdash[agent]'"
    ) from exc

from .assistant import create_agent
from .authoring import authoring_toolset
from .data import data_toolset
from .headless import AgentSession, TurnResult

__all__ = ["AgentSession", "TurnResult", "authoring_toolset", "create_agent", "data_toolset"]
