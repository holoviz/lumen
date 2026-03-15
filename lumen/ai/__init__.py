import panel as pn

from . import (  # noqa
    actor, agents, embeddings, llm,
)
from .analysis import Analysis  # noqa
from .coordinator import Coordinator, DependencyResolver, Planner  # noqa
from .ui import ChatUI, ExplorerUI  # noqa
from .vector_store import DuckDBVectorStore, NumpyVectorStore  # noqa

pn.chat.message.DEFAULT_AVATARS.update({
    "lumen": "https://holoviz.org/assets/lumen.png",
    "dataset": "🗂️",
    "sql": "🗄️",
    "router": "🚦",
})


def _patch_chat_area_input_file_reset():
    try:
        from panel_material_ui.base import BASE_PATH
        bundle_path = BASE_PATH / "dist" / "panel-material-ui.bundle.js"
        if not bundle_path.exists():
            return
        content = bundle_path.read_text(encoding="utf-8")
        if 'R.current&&(R.current.value="")' in content:
            return
        content = content.replace(
            "ee.splice(W,1),v(ee),_.current=ee}",
            'ee.splice(W,1),v(ee),_.current=ee,R.current&&(R.current.value="")}',
        )
        content = content.replace(
            "v(ue),_.current=ue}}",
            'v(ue),_.current=ue,W.target.value=""}}',
        )
        bundle_path.write_text(content, encoding="utf-8")
    except Exception:
        pass


_patch_chat_area_input_file_reset()
