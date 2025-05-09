import datetime

from abc import abstractmethod
from contextlib import nullcontext
from pathlib import Path
from types import FunctionType
from typing import Any

import param

from panel.chat import ChatFeed
from pydantic import BaseModel

from .config import PROMPTS_DIR
from .llm import Llm, Message
from .memory import _Memory, memory
from .utils import log_debug, render_template, warn_on_unused_variables


class NullStep:
    def __init__(self):
        self.status = None

    def stream(self, text):
        log_debug(f"[{text}")


class LLMUser(param.Parameterized):
    """
    Mixin for classes that use prompts with LLMs.
    Provides parameters and methods for prompt templating and LLM interactions.
    """

    llm = param.ClassSelector(class_=Llm, doc="""
        The LLM implementation to query.""")

    prompts = param.Dict(default={
        "main": {"template": PROMPTS_DIR / "Actor" / "main.jinja2"},
    }, doc="""
        A dictionary of prompts, indexed by prompt name.
        Each prompt should be defined as a dictionary containing a template
        'template' and optionally a 'model' and 'tools'.""")

    template_overrides = param.Dict(default={}, doc="""
        Overrides the template's blocks (instructions, context, tools, examples).
        Is a nested dictionary with the prompt name (e.g. main) as the key
        and the block names as the inner keys with the new content as the
        values.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._validate_template_overrides()
        self._validate_prompts()

    def _validate_template_overrides(self):
        """Validate that template overrides are correctly formatted."""
        valid_prompt_names = self.param["prompts"].default.keys()
        for prompt_name, template_override in self.template_overrides.items():
            if not isinstance(template_override, dict):
                raise ValueError(
                    "`template_overrides` must be a nested dictionary with prompt names as keys, "
                    "e.g. {'main': {'instructions': 'custom instructions'}}, but got "
                    f"{self.template_overrides} instead."
                )
            if prompt_name != "main" and prompt_name not in valid_prompt_names:
                raise ValueError(
                    f"Prompt {prompt_name!r} is not a valid prompt name. "
                    f"Valid prompt names are {valid_prompt_names}."
                )

    def _validate_prompts(self):
        """Validate that prompts have correct structure."""
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

    def _lookup_prompt_key(self, prompt_name: str, key: str):
        """Look up a specific key in a prompt template, with inheritance."""
        if prompt_name in self.prompts and key in self.prompts[prompt_name]:
            prompt_spec = self.prompts[prompt_name]
        elif prompt_name in self.param.prompts.default and key in self.param.prompts.default[prompt_name]:
            prompt_spec = self.param.prompts.default[prompt_name]
        else:
            for cls in type(self).__mro__:
                if hasattr(cls, 'param') and hasattr(cls.param, 'prompts'):
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
        """Get the response model for a prompt."""
        model_spec = self._lookup_prompt_key(prompt_name, "response_model")
        if isinstance(model_spec, FunctionType):
            model = model_spec(**context)
        else:
            model = model_spec
        return model

    async def _gather_prompt_context(self, prompt_name: str, messages: list[Message], **context):
        """Gather context for the prompt template."""
        context["current_datetime"] = datetime.datetime.now()
        return context

    async def _render_prompt(self, prompt_name: str, messages: list[Message], **context) -> str:
        """Render a prompt template with context."""
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
            format_kwargs = dict(**overrides, **context)
            warn_on_unused_variables(prompt_template, format_kwargs, prompt_label)
            try:
                prompt = prompt_template.format(**format_kwargs)
            except KeyError as e:
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
        Render a prompt and invoke the LLM.

        Parameters
        ----------
        prompt_name : str
            Name of the prompt template to use
        messages : list[Message]
            The conversation messages
        response_model : type[BaseModel], optional
            Pydantic model to structure the response
        model_spec : str, optional
            Specification for which LLM to use
        **context : dict
            Additional context variables for the prompt template

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
        Render a prompt and stream responses from the LLM.

        Parameters
        ----------
        prompt_name : str
            Name of the prompt template to use
        messages : list[Message]
            The conversation messages
        response_model : type[BaseModel], optional
            Pydantic model to structure the response
        model_spec : str, optional
            Specification for which LLM to use
        field : str, optional
            Specific field to extract from the response model
        **context : dict
            Additional context variables for the prompt template

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

    @property
    def llm_spec_key(self):
        """
        Converts class name to a snake_case model identifier.
        Used for looking up model configurations in model_kwargs.
        """
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


class Actor(LLMUser):

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The interface for the Coordinator to interact with.""")

    memory = param.ClassSelector(class_=_Memory, default=None, doc="""
        Local memory which will be used to provide the agent context.
        If None the global memory will be used.""")

    def __init__(self, **params):
        super().__init__(**params)
        if self.interface is None:
            self._null_step = NullStep()

    def _add_step(self, title: str = "", **kwargs):
        """Private contextmanager for adding steps to the interface.

        If self.interface is None, returns a nullcontext that captures calls.
        Otherwise, returns the interface's add_step contextmanager.
        """
        return nullcontext(self._null_step) if self.interface is None else self.interface.add_step(title=title, **kwargs)

    @property
    def _memory(self) -> _Memory:
        return memory if self.memory is None else self.memory

    async def _gather_prompt_context(self, prompt_name: str, messages: list[Message], **context):
        """Gather context for the prompt template."""
        context = await super()._gather_prompt_context(prompt_name, messages, **context)
        context["memory"] = self._memory
        context["actor_name"] = self.name
        return context

    @abstractmethod
    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> Any:
        """
        Responds to the provided messages.
        """


class ContextProvider(param.Parameterized):
    """
    ContextProvider describes the interface an Actor has to define
    to be invokable by other Actors, including its purpose so other
    actors can decide when to use it and the context values it
    requires and provides.
    """

    conditions = param.List(default=[], doc="""
        Specific criteria that determine when this actor should be invoked.
        These conditions establish explicit relationships between actors in the system,
        defining the circumstances under which this actor becomes relevant.
        While 'purpose' describes what the actor does, conditions specify
        the precise situations that warrant its use.""")

    exclusions = param.List(default=[], doc="""
        List of context values that this actor should not be invoked with.
        This is useful for actors that are not relevant in certain contexts
        or for actors that should not be invoked in certain situations.""")

    not_with = param.List(default=[], doc="""
        List of actors that this actor should not be invoked with.""")

    provides = param.List(default=[], readonly=True, doc="""
        List of context values it provides to current working memory.""")

    purpose = param.String(default="", doc="""
        A descriptive statement of this actor's functionality and capabilities.
        Serves as a high-level explanation for other actors to understand
        what this actor does and when it might be useful to invoke it.""")

    requires = param.List(default=[], readonly=True, doc="""
        List of context values it requires to be in memory.""")

    async def requirements(self, messages: list[Message]) -> list[str]:
        return self.requires

    def __str__(self):
        string = (
            f"- {self.name[:-5]}: {' '.join(self.purpose.strip().split())}\n"
            f"  Requires: `{'`, `'.join(self.requires)}`\n"
            f"  Provides: `{'`, `'.join(self.provides)}`\n"
        )
        if self.conditions:
            string += "  Conditions:\n" + "\n".join(f"  - {condition}" for condition in self.conditions) + "\n"
        if self.not_with:
            string += "  Not with:\n" + "\n".join(f"  - {not_with}" for not_with in self.not_with) + "\n"
        return string
