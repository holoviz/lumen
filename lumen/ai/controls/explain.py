from __future__ import annotations

import param

from .revision import RevisionControls


class ExplainControls(RevisionControls):
    """Controls for explaining editor specs using AI.

    Works with any LumenEditor that has a ``spec`` and ``language``
    attribute. Clicking the icon opens a text input where the user can
    optionally describe which part to explain. Leaving it blank
    explains the overall goal.
    """

    input_kwargs = {
        "placeholder": "Which part? Leave blank for overview.",
    }

    toggle_kwargs = {
        "icon": "psychology",
        "description": "Explain this code",
    }

    _explain_count = param.Integer(default=0)

    def __init__(self, **params):
        super().__init__(**params)
        self._user_question = ""

    def _enter_reason(self, _):
        self._user_question = self._text_input.value_input.strip()
        self._text_input.value = ""
        self.param.update(active=False, _explain_count=self._explain_count + 1)

    @param.depends("_explain_count", watch=True)
    async def _explain(self):
        messages = list(self.task.history)
        message = None
        try:
            with self.task.actor.param.update(interface=self.interface):
                async for chunk in self.task.actor.explain(
                    self._user_question, messages, self.task.out_context, self.view
                ):
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
