import panel as pn

from ..sources.data import read
from . import (
    actor, agents, embeddings, llm,
)
from .analysis import Analysis
from .answer import (
    Answer, Answers, Chat, aask, aask_many, ask, ask_many, chat, clear_cache,
    configure,
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
    "Analysis", "Answer", "Answers", "Chat", "ChatUI", "Choice", "Coordinator",
    "DecisionModel", "DecisionResult", "DependencyResolver", "DuckDBVectorStore",
    "ExplorerUI", "Jev", "Noul", "NumpyVectorStore", "OpenRouterDecisionModel",
    "Planner", "Score", "aask", "aask_many", "actor", "agents", "ask", "ask_many", "chat", "clear_cache", "configure",
    "embeddings", "llm", "read",
]

pn.chat.message.DEFAULT_AVATARS.update({
    "lumen": "https://holoviz.org/assets/lumen.png",
    "dataset": "🗂️",
    "sql": "🗄️",
    "router": "🚦",
})
