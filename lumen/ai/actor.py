from abc import abstractmethod
from contextlib import nullcontext
from pathlib import Path
from types import FunctionType
from typing import Any

import param

from panel.chat import ChatFeed
from pydantic import BaseModel

from .llm import Llm, Message
from .memory import _Memory, memory
from .utils import log_debug, render_template, warn_on_unused_variables


class NullStep:
    def __init__(self):
        self.status = None

    def stream(self, text):
        log_debug(f"[{text}")


class Actor(param.Parameterized):

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The interface for the Coordinator to interact with.""")

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
        if self.interface is None:
            self._null_step = NullStep()

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
            extra_keys = set(self.prompts[prompt_name].keys()) - {"template", "response_model", "tools", "llm_spec"}
            if extra_keys:
                raise ValueError(
                    f"Prompt {prompt_name!r} has unexpected keys {extra_keys}. "
                    "Valid keys are 'template', 'response_model', 'tools', and 'llm_spec'."
                )

    @property
    def _memory(self) -> _Memory:
        return memory if self.memory is None else self.memory

    def _lookup_prompt_key(self, prompt_name: str, key: str):
        if prompt_name in self.prompts and key in self.prompts[prompt_name]:
            prompt_spec = self.prompts[prompt_name]
        elif prompt_name in self.param.prompts.default and key in self.param.prompts.default[prompt_name]:
            prompt_spec = self.param.prompts.default[prompt_name]
        else:
            for cls in type(self).__mro__:
                if issubclass(cls, Actor):
                    if key in cls.param.prompts.default.get(prompt_name, {}):
                        prompt_spec = cls.param.prompts.default[prompt_name]
                        break
            else:
                prompt_spec = {}
        if key not in prompt_spec:
            if key == "tools":
                return []
            raise KeyError(f"Prompt {prompt_name!r} does not provide a {key!r}.")
        return prompt_spec[key]

    def _get_model(self, prompt_name: str, **context) -> type[BaseModel]:
        model_spec = self._lookup_prompt_key(prompt_name, "response_model")
        if isinstance(model_spec, FunctionType):
            model = model_spec(**context)
        else:
            model = model_spec
        return model

    def _add_step(self, title: str = "", **kwargs):
        """Private contextmanager for adding steps to the interface.

        If self.interface is None, returns a nullcontext that captures calls.
        Otherwise, returns the interface's add_step contextmanager.
        """
        return nullcontext(self._null_step) if self.interface is None else self.interface.add_step(title=title, **kwargs)

    async def _gather_prompt_context(self, prompt_name: str, messages: list[Message], **context):
        context["memory"] = self._memory
        context["actor_name"] = self.name
        return context

    async def _render_prompt(self, prompt_name: str, messages: list[Message], **context) -> str:
        prompt_template = self._lookup_prompt_key(prompt_name, "template")
        overrides = self.template_overrides.get(prompt_name, {})
        context = await self._gather_prompt_context(prompt_name, messages, **context)

        prompt_label = f"\033[92m{self.name}.prompts['{prompt_name}']['template']\033[0m"
        try:
            path_exists = Path(prompt_template).exists()
        except OSError:
            path_exists = False
        if isinstance(prompt_template, str) and not path_exists:
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
        prompt = prompt.strip()
        log_debug(f"{prompt_label}:\n\033[90m{prompt}\033[0m", show_length=True)
        return prompt

    async def _invoke_prompt(
        self,
        prompt_name: str,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        model_spec: str | None = None,
        **context
    ) -> Any:
        """
        Convenience method that combines rendering a prompt and invoking the LLM.

        Parameters
        ----------
        prompt_name : str
            Name of the prompt template to use
        messages : list[Message]
            The conversation messages
        response_model : type[BaseModel], optional
            Pydantic model to structure the response, if None will use the model
            defined in the prompt configuration
        model_spec : str, optional
            Specification for which LLM to use, if None will use the model_spec
            defined in the prompt configuration or fall back to self.llm_spec_key
        **context : dict
            Additional context variables to pass to the prompt template

        Returns
        -------
        The structured response from the LLM
        """
        system = await self._render_prompt(prompt_name, messages, **context)

        if response_model is None:
            try:
                response_model = self._get_model(prompt_name, **context)
            except (KeyError, AttributeError):
                pass

        if model_spec is None:
            try:
                model_spec = self._lookup_prompt_key(prompt_name, "llm_spec")
            except KeyError:
                model_spec = self.llm_spec_key

        result = await self.llm.invoke(
            messages=messages,
            system=system,
            response_model=response_model,
            model_spec=model_spec,
        )

        return result

    async def _stream_prompt(
        self,
        prompt_name: str,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        model_spec: str | None = None,
        field: str | None = None,
        **context
    ):
        """
        Convenience method that combines rendering a prompt and streaming responses from the LLM.

        Parameters
        ----------
        prompt_name : str
            Name of the prompt template to use
        messages : list[Message]
            The conversation messages
        response_model : type[BaseModel], optional
            Pydantic model to structure the response, if None will use the model
            defined in the prompt configuration
        model_spec : str, optional
            Specification for which LLM to use, if None will use the model_spec
            defined in the prompt configuration or fall back to self.llm_spec_key
        field : str, optional
            Specific field to extract from the response model
        **context : dict
            Additional context variables to pass to the prompt template

        Yields
        ------
        Chunks of the response from the LLM as they are generated
        """
        # Render the prompt
        system = await self._render_prompt(prompt_name, messages, **context)

        # Determine the response model
        if response_model is None:
            try:
                response_model = self._get_model(prompt_name, **context)
            except (KeyError, AttributeError):
                # If no model is specified, return raw response
                pass

        # Determine the model specification
        if model_spec is None:
            try:
                model_spec = self._lookup_prompt_key(prompt_name, "llm_spec")
            except KeyError:
                model_spec = self.llm_spec_key

        # Stream from the LLM
        async for chunk in self.llm.stream(
            messages=messages,
            system=system,
            response_model=response_model,
            model_spec=model_spec,
            field=field
        ):
            yield chunk

    @abstractmethod
    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> Any:
        """
        Responds to the provided messages.
        """

    @classmethod
    def get_prompt_template(cls, key: str = "main") -> str:
        """
        Returns the template for the given prompt key.

        Parameters
        ----------
        key : str, optional
            The key of the prompt to return the template for. Default is "main".

        Returns
        -------
        The template associated with the given prompt key.
        """
        return cls._lookup_prompt_key(cls, key, "template").read_text()

    @property
    def llm_spec_key(self):
        # Remove "Agent" suffix from class name
        name = self.__class__.__name__.replace("Agent", "")

        result = ""
        i = 0
        while i < len(name):
            char = name[i]

            # Check if this is part of an acronym (current char is uppercase and next char is uppercase too)
            is_part_of_acronym = (
                char.isupper() and
                i + 1 < len(name) and
                name[i + 1].isupper()
            )

            # Add underscore before uppercase letters, unless it's part of an acronym
            if char.isupper() and i > 0 and not is_part_of_acronym and not name[i - 1].isupper():
                result += "_"

            # Add the lowercase character
            result += char.lower()
            i += 1

        return result


class ContextProvider(param.Parameterized):
    """
    ContextProvider describes the interface an Actor has to define
    to be invokable by other Actors, including its purpose so other
    actors can decide when to use it and the context values it
    requires and provides.
    """

    provides = param.List(default=[], readonly=True, doc="""
        List of context values it provides to current working memory.""")

    purpose = param.String(default="", doc="""
        Describes the purpose of this actor for consumption of
        other actors that might invoke it.""")

    requires = param.List(default=[], readonly=True, doc="""
        List of context values it requires to be in memory.""")

    async def requirements(self, messages: list[Message]) -> list[str]:
        return self.requires
