from abc import abstractmethod
from typing import Any

import param

from .llm import Message
from .utils import render_template


class Actor(param.Parameterized):

    prompt_overrides = param.Dict(default={}, doc="""
        Overrides the prompt's 'instructions' or 'context' jinja2 blocks.
        Is a nested dictionary with the prompt name (e.g. main) as the key
        and the block names as the inner keys with the new content as the
        values.""")

    prompt_templates = param.Dict(default={}, doc="""
        The paths to the prompt's jinja2 templates.""")

    def _render_prompt(self, prompt_name: str, **context) -> str:
        context["memory"] = self._memory
        prompt = render_template(
            self.prompt_templates[prompt_name],
            prompt_overrides=self.prompt_overrides.get(prompt_name, {}),
            **context
        )
        return prompt

    async def _render_main_prompt(self, messages: list[Message], **context) -> str:
        """
        Renders the main prompt using the provided prompts.
        """
        main_prompt = self._render_prompt("main", **context)
        return main_prompt

    @abstractmethod
    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> Any:
        """
        Responds to the provided messages.
        """
