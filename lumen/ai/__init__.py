import panel as pn

from . import agents, embeddings, llm  # noqa
from .agents import Analysis  # noqa
from .coordinator import Coordinator, Planner, Resolver  # noqa
from .memory import memory  # noqa
from .ui import ChatUI, ExplorerUI  # noqa

pn.chat.message.DEFAULT_AVATARS.update({
    "lumen": "https://holoviz.org/assets/lumen.png",
    "dataset": "🗂️",
    "sql": "🗄️",
    "router": "🚦",
})
