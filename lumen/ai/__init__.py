import panel as pn

from . import (
    actor, agents, embeddings, llm,
)
from .analysis import Analysis
from .answer import (
    Answer, Chat, aask, ask, chat, configure,
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
    "Planner", "Score", "aask", "actor", "agents", "ask", "chat", "configure",
    "embeddings", "llm",
]

pn.chat.message.DEFAULT_AVATARS.update({
    "lumen": "https://holoviz.org/assets/lumen.png",
    "dataset": "🗂️",
    "sql": "🗄️",
    "router": "🚦",
})
