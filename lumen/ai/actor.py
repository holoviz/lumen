from abc import abstractmethod
from pathlib import Path
from types import FunctionType
from typing import Any

import param

from pydantic import BaseModel

from .llm import Llm, Message
from .memory import _Memory, memory
from .utils import log_debug, render_template, warn_on_unused_variables


class Actor(param.Parameterized):

    llm = param.ClassSelector(class_=Llm, doc="""
        The LLM implementation to query.""")

    memory = param.ClassSelector(class_=_Memory, default=None, doc="""
        Local memory which will be used to provide the agent context.
        If None the global memory will be used.""")

    prompts = param.Dict(default={}, doc="""
        A dictionary of prompts used by the actor, indexed by prompt name.
        Each prompt should be defined as a dictionary containing a template
        'template' and optionally a 'model' and 'tools'.""")

    template_overrides = param.Dict(default={}, doc="""
        Overrides the template's 'instructions', 'context', 'tools', or 'examples' jinja2 blocks.
        Is a nested dictionary with the prompt name (e.g. main) as the key
        and the block names as the inner keys with the new content as the
        values.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._validate_template_overrides()
        self._validate_prompts()
        self._tools = {}
        for prompt_name in self.prompts:
            self._tools[prompt_name] = [
                tool if isinstance(tool, Actor) else tool()
                for tool in self._lookup_prompt_key(prompt_name, "tools")
            ]

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

    def _validate_prompts(self):
        for prompt_name in self.prompts:
            if prompt_name not in self.param.prompts.default:
                raise ValueError(
                    f"Prompt {prompt_name!r} is not a valid prompt name. "
                    f"Valid prompt names are {self.param.prompts.default.keys()}."
                )
            extra_keys = set(self.prompts[prompt_name].keys()) - {"template", "model", "tools"}
            if extra_keys:
                raise ValueError(
                    f"Prompt {prompt_name!r} has unexpected keys {extra_keys}. "
                    "Valid keys are 'template', 'model' and 'tools'."
                )

    @property
    def _memory(self) -> _Memory:
        return memory if self.memory is None else self.memory

    def _lookup_prompt_key(self, prompt_name: str, key: str):
        if prompt_name in self.prompts and key in self.prompts[prompt_name]:
            prompt_spec = self.prompts[prompt_name]
        else:
            prompt_spec = self.param.prompts.default[prompt_name]
        if key not in prompt_spec:
            if key == "tools":
                return []
            raise KeyError(f"Prompt {prompt_name!r} does not provide a {key!r}.")
        return prompt_spec[key]

    def _get_model(self, prompt_name: str, **context) -> type[BaseModel]:
        model_spec = self._lookup_prompt_key(prompt_name, "model")
        if isinstance(model_spec, FunctionType):
            model = model_spec(**context)
        else:
            model = model_spec
        return model

    async def _use_tools(self, prompt_name: str, messages: list[Message]) -> str:
        tools_context = ""
        for tool in self._tools[prompt_name]:
            if all(requirement in self._memory for requirement in tool.requires):
                with tool.param.update(llm=self.llm, memory=self.memory):
                    tools_context += await tool.respond(messages)
        return tools_context

    async def _render_prompt(self, prompt_name: str, messages: list[Message], **context) -> str:
        prompt_template = self._lookup_prompt_key(prompt_name, "template")
        overrides = self.template_overrides.get(prompt_name, {})
        context["memory"] = self._memory
        if "tools" not in context:
            context["tools"] = await self._use_tools(prompt_name, messages)

        prompt_label = f"\033[92m{self.name[:-5]}.prompts['{prompt_name}']['template']\033[0m"
        if isinstance(prompt_template, str) and not Path(prompt_template).exists():
            # check if all the format_kwargs keys are contained in prompt_template
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

    @abstractmethod
    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> Any:
        """
        Responds to the provided messages.
        """
