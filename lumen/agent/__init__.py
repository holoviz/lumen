"""
One pydantic-ai agent that answers questions about data by building a FlowDash
dashboard of Lumen nodes. Experimental: everything here may change.

Needs panel-flowdash with its agent extra (panel-agents >= 0.1.0b3 and
pydantic-ai), which Lumen does not install; nothing else in Lumen imports this
package.
"""
try:
    import panel_agents
    import panel_flowdash  # noqa: F401
    import pydantic_ai  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lumen.agent needs panel-flowdash with its agent extra, which is not installed: "
        "pip install 'panel-flowdash[agent]' 'panel-agents>=0.1.0b3'"
    ) from exc

from packaging.version import InvalidVersion, Version

# Replaying recorded sessions whose history the agent shortened needs 0.1.0b3.
PANEL_AGENTS_MINIMUM = "0.1.0b3"

try:
    _version = Version(panel_agents.__version__)
except InvalidVersion:
    _version = None
# A source checkout without version metadata reports 0.0.0+unknown and is let through.
if _version is not None and _version.local is None and _version < Version(PANEL_AGENTS_MINIMUM):
    raise ImportError(
        f"lumen.agent needs panel-agents >= {PANEL_AGENTS_MINIMUM}, found {panel_agents.__version__}: "
        f"pip install 'panel-agents>={PANEL_AGENTS_MINIMUM}'"
    )

from .assistant import create_agent
from .authoring import authoring_toolset
from .data import data_toolset
from .headless import AgentSession, TurnResult

__all__ = ["AgentSession", "TurnResult", "authoring_toolset", "create_agent", "data_toolset"]
