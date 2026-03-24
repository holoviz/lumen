from __future__ import annotations

from datetime import datetime

import param

from ..config import PROMPTS_DIR
from ..utils import render_template
from .revision import RevisionControls

SPEC_TYPE_MAP = {
    "sql": "query",
    "yaml": "specification",
    "json": "specification",
    "vega-lite": "visualization spec",
}


class ExplainControls(RevisionControls):
    """Controls for explaining editor specs using AI.

    Works with any LumenEditor that has a ``spec`` and ``language``
    attribute. Clicking the icon opens a text input where the user can
    optionally describe which part to explain. Leaving it blank
    explains the full spec.
    """

    input_kwargs = {
        "placeholder": "What part to explain? Leave blank for full explanation.",
    }

    toggle_kwargs = {
        "icon": "psychology",
        "description": "Explain this code",
    }

    _explain_count = param.Integer(default=0)

    def _enter_reason(self, _):
        self._focus = self._text_input.value_input.strip()
        self._text_input.value = ""
        self.param.update(active=False, _explain_count=self._explain_count + 1)

    @param.depends("_explain_count", watch=True)
    async def _explain(self):
        spec = self.view.spec
        if not spec or not spec.strip():
            return

        focus = getattr(self, "_focus", "")
        language = self.view.language
        spec_type = SPEC_TYPE_MAP.get(language, "code")

        system = render_template(
            PROMPTS_DIR / "ExplainControls" / "main.jinja2",
            language=language,
            spec_type=spec_type,
            spec=spec,
            focus=focus,
            current_datetime=datetime.now(),
            memory={},
            actor_name="ExplainControls",
        )

        user_content = f"Focus on: {focus}" if focus else "Explain the spec above."
        messages = [{"role": "user", "content": user_content}]

        llm = self.task.actor.llm
        message = None
        try:
            async for chunk in llm.stream(messages=messages, system=system):
                if chunk:
                    message = self.interface.stream(
                        chunk,
                        replace=True,
                        message=message,
                        user="Assistant",
                    )
        except Exception as e:
            self._report_status(
                f"```\n{e}\n```",
                title="\u274c Failed to generate explanation",
            )
