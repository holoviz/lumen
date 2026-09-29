import panel as pn

from . import (  # noqa
    actor, agents, embeddings, llm,
)
from .analysis import Analysis  # noqa
from .coordinator import Coordinator, DependencyResolver, Planner  # noqa
from .decisions import (  # noqa
    Choice, DecisionModel, DecisionResult, Jev, Noul, Score,
)
from .ui import ChatUI, ExplorerUI  # noqa
from .vector_store import DuckDBVectorStore, NumpyVectorStore  # noqa

pn.chat.message.DEFAULT_AVATARS.update({
    "lumen": "https://holoviz.org/assets/lumen.png",
    "dataset": "🗂️",
    "sql": "🗄️",
    "router": "🚦",
})
