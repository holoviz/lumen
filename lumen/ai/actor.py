from abc import abstractmethod
from pathlib import Path
from types import FunctionType
from typing import Any

import param

from pydantic import BaseModel

from .llm import Message
from .utils import log_debug, render_template, warn_on_unused_variables


class Actor(param.Parameterized):

    template_overrides = param.Dict(default={}, doc="""
        Overrides the template's 'instructions', 'context', 'embeddings', or 'examples' jinja2 blocks.
        Is a nested dictionary with the prompt name (e.g. main) as the key
        and the block names as the inner keys with the new content as the
        values.""")

    prompts = param.Dict(default={}, doc="""
        A dict of the prompt name, like 'main' as key nesting another dict
        with keys like 'template' and 'model'.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._validate_template_overrides()
        self._validate_prompts()

    def _validate_template_overrides(self):
        valid_prompt_names = self.param["prompts"].default.keys()
        for prompt_name, template_override in self.template_overrides.items():
            if not isinstance(template_override, dict):
                raise ValueError(
                    "`template_overrides` must be a nested dictionary with prompt names as keys, "
                    "e.g. {'main': {'instructions': 'custom instructions'}}, but got "
                    f"{self.template_overrides} instead."
                )
            if prompt_name not in valid_prompt_names:
                raise ValueError(
                    f"Prompt {prompt_name!r} is not a valid prompt name. "
                    f"Valid prompt names are {valid_prompt_names}."
                )
            for block_name in template_override:
                if block_name not in ["instructions", "context", "embeddings", "examples"]:
                    raise ValueError(
                        f"Block {block_name!r} is not a valid block name. "
                        f"Valid block names are ['instructions', 'context', 'embeddings', 'examples']."
                    )

    def _validate_prompts(self):
        for prompt_name in self.prompts:
            if prompt_name not in self.param.prompts.default:
                raise ValueError(
                    f"Prompt {prompt_name!r} is not a valid prompt name. "
                    f"Valid prompt names are {self.param.prompts.default.keys()}."
                )
            if "template" not in self.prompts[prompt_name]:
                raise ValueError(
                    f"Prompt {prompt_name!r} does not provide a prompt template."
                )
            extra_keys = set(self.prompts[prompt_name].keys()) - {"template", "model"}
            if extra_keys:
                raise ValueError(
                    f"Prompt {prompt_name!r} has unexpected keys {extra_keys}. "
                    "Valid keys are 'template' and 'model'."
                )

    def _get_model(self, prompt_name: str, **context) -> type[BaseModel]:
        if prompt_name in self.prompts and "model" in self.prompts[prompt_name]:
            prompt_spec = self.prompts[prompt_name]
        else:
            prompt_spec = self.param.prompts.default[prompt_name]
        if "model" not in prompt_spec:
            raise KeyError(f"Prompt {prompt_name!r} does not provide a model.")
        model_spec = prompt_spec["model"]
        if isinstance(model_spec, FunctionType):
            model = model_spec(**context)
        else:
            model = model_spec
        return model

    def _render_prompt(self, prompt_name: str, **context) -> str:
        if prompt_name in self.prompts and "template" in self.prompts[prompt_name]:
            prompt_spec = self.prompts[prompt_name]
        else:
            prompt_spec = self.param.prompts.default[prompt_name]
        if "template" not in prompt_spec:
            raise KeyError(f"Prompt {prompt_name!r} does not provide a prompt template.")
        prompt_template = prompt_spec["template"]

        overrides = self.template_overrides.get(prompt_name, {})
        prompt_label = f"\033[92m{self.name[:-5]}.prompts['{prompt_name}']['template']\033[0m"
        context["memory"] = self._memory
        if isinstance(prompt_template, str) and not Path(prompt_template).exists():
            # check if all the format_kwargs keys are contained prompt_template
            # e.g. the key, "memory", is not used in "string template".format(memory=memory)
            format_kwargs = dict(**overrides, **context)
            warn_on_unused_variables(prompt_template, format_kwargs, prompt_label)
            try:
                prompt = prompt_template.format(**format_kwargs)
            except KeyError as e:
                # check if all the format variables in prompt_template
                # are available from format_kwargs, e.g. the key, "var",
                # is not available in context "string template {var}".format(memory=memory)
                raise KeyError(
                    f"Unexpected template variable: {e}. To resolve this, "
                    f"please ensure overrides contains the {e} key"
                ) from e
        else:
            prompt = render_template(
                prompt_template,
                overrides=overrides,
                **context
            )
        log_debug(f"Below is the rendered prompt from {prompt_label}:\n{prompt}")
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
