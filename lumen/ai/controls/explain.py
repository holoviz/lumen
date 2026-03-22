from __future__ import annotations

from pathlib import Path

import param

from jinja2 import Environment, FileSystemLoader
from panel.viewable import Viewer
from panel_material_ui import (
    Card, IconButton, Markdown, Row,
)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

SPEC_TYPE_MAP = {
    "sql": "query",
    "yaml": "specification",
    "json": "specification",
    "vega-lite": "visualization spec",
}


class ExplainControls(Viewer):
    """Controls for explaining editor specs using AI.

    Works with any LumenEditor that has a ``spec`` and ``language``
    attribute. The system prompt adapts based on the editor's language.
    """

    interface = param.Parameter()

    layout_kwargs = param.Dict(default={})

    task = param.Parameter()

    view = param.Parameter(doc="The View containing the spec to explain")

    _clicking = param.Boolean(default=False)

    def __init__(self, **params):
        super().__init__(**params)
        language = getattr(self.view, "language", "yaml").upper()
        self._icon = IconButton(
            icon="psychology",
            active_icon="check",
            description=f"Explain {language}",
            margin=(5, 0),
            toggle_duration=1000,
            size="small",
            color="primary",
            icon_size="0.9em",
        )
        self._icon.param.watch(self._on_click, "clicks")
        self._row = Row(self._icon, **self.layout_kwargs)

    def _render_system_prompt(self) -> str:
        language = getattr(self.view, "language", "yaml")
        spec_type = SPEC_TYPE_MAP.get(language, "code")
        env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR / "ExplainControls")),
            keep_trailing_newline=False,
        )
        template = env.get_template("main.jinja2")
        return template.render(language=language, spec_type=spec_type)

    def _on_click(self, event):
        if not self._clicking:
            self._clicking = True
            from panel.io import state
            state.execute(self._explain)

    async def _explain(self):
        spec = self.view.spec
        if not spec or not spec.strip():
            self._clicking = False
            return

        language = getattr(self.view, "language", "yaml")
        self._icon.disabled = True
        message = None
        try:
            llm = self.task.actor.llm
            system = self._render_system_prompt()
            messages = [
                {"role": "user", "content": f"Explain this {language}:\n\n```{language}\n{spec}\n```"}
            ]

            async for chunk in llm.stream(
                messages=messages,
                system=system,
            ):
                if chunk:
                    message = self.interface.stream(
                        chunk,
                        replace=True,
                        message=message,
                        user="Assistant",
                    )
        except Exception as e:
            md = Markdown(
                f"```\n{e}\n```",
                margin=0,
                sizing_mode="stretch_width",
            )
            self.interface.stream(
                Card(md, title="\u274c Failed to generate explanation"),
                user="Assistant",
            )
        finally:
            self._icon.disabled = False
            self._clicking = False

    def __panel__(self):
        return self._row
