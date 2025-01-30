import panel as pn

from . import agents, embeddings, llm  # noqa
from .analysis import Analysis  # noqa
from .coordinator import Coordinator, DependencyResolver, Planner  # noqa
from .memory import memory  # noqa
from .ui import ChatUI, ExplorerUI  # noqa
from .vector_store import DuckDBVectorStore, NumpyVectorStore  # noqa

pn.chat.message.DEFAULT_AVATARS.update({
    "lumen": "https://holoviz.org/assets/lumen.png",
    "dataset": "🗂️",
    "sql": "🗄️",
    "router": "🚦",
})
