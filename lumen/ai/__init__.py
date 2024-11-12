import panel as pn

from . import agents, embeddings, llm  # noqa
from .agents import Analysis  # noqa
from .memory import memory  # noqa
from .orchestrator import Orchestrator, Planner  # noqa
from .ui import UI, Explorer  # noqa

pn.chat.message.DEFAULT_AVATARS.update({
    "lumen": "https://holoviz.org/assets/lumen.png",
    "dataset": "🗂️",
    "sql": "🗄️",
    "router": "🚦",
})
