import panel as pn

from ..util import AI_EXTRA_HINT, missing_ai_modules

if _missing := missing_ai_modules():
    raise ModuleNotFoundError(f"{AI_EXTRA_HINT} (missing: {', '.join(_missing)})", name=_missing[0])
del _missing

from ..sources.data import read
from . import (
    actor, agents, embeddings, llm,
)
from .analysis import Analysis
from .answer import (
    Answer, Chat, aask, ask, chat, clear_cache, configure,
)
from .coordinator import Coordinator, DependencyResolver, Planner
from .decisions import (
    Choice, DecisionModel, DecisionResult, Jev, Noul, OpenRouterDecisionModel,
    Score,
)
from .ui import ChatUI, ExplorerUI
from .vector_store import DuckDBVectorStore, NumpyVectorStore

# Declared so type checkers treat these as public now that lumen ships py.typed
__all__ = [
    "Analysis", "Answer", "Chat", "ChatUI", "Choice", "Coordinator",
    "DecisionModel", "DecisionResult", "DependencyResolver", "DuckDBVectorStore",
    "ExplorerUI", "Jev", "Noul", "NumpyVectorStore", "OpenRouterDecisionModel",
    "Planner", "Score", "aask", "actor", "agents", "ask", "chat", "clear_cache", "configure",
    "embeddings", "llm", "read",
]

pn.chat.message.DEFAULT_AVATARS.update({
    "lumen": "https://holoviz.org/assets/lumen.png",
    "dataset": "🗂️",
    "sql": "🗄️",
    "router": "🚦",
})
