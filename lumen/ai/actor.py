import datetime
import inspect

from abc import abstractmethod
from contextlib import nullcontext
from pathlib import Path
from types import FunctionType
from typing import Any

import param

from panel.chat import ChatFeed
from panel.layout.base import ListLike, NamedListLike
from pydantic import BaseModel

from .config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from .context import ContextModel, TContext
from .llm import Llm, Message
from .utils import (
    class_name_to_llm_spec_key, log_debug, render_template,
    warn_on_unused_variables, wrap_logfire_on_method,
)


def _expand_llm_tool_entries(entries: list[Any] | None, context: TContext) -> list[Any]:
    """
    Expand :attr:`LLMUser.llm_tools` entries: keep plain tools; resolve callables
    ``f(context)`` or no-arg ``f()`` to a tool or list of tools.
    """
    from .tools.base import FunctionTool

    if not entries:
        return []
    out: list[Any] = []
    for item in entries:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            out.extend(_expand_llm_tool_entries(list(item), context))
            continue
        if isinstance(item, FunctionTool):
            out.append(item)
            continue
        if inspect.isclass(item):
            out.append(item)
            continue
        if callable(item):
            sig = inspect.signature(item)
            produced = item(context) if len(sig.parameters) else item()
            if inspect.isawaitable(produced):
                raise TypeError(
                    "llm_tools callables must be synchronous; async factories are not supported."
                )
            if produced is None:
                continue
            if isinstance(produced, (list, tuple)):
                out.extend(produced)
            else:
                out.append(produced)
            continue
        out.append(item)
    return out


def _merge_prompt_tools(
    registration: list[Any] | None,
    explicit: list[Any] | None,
    context: TContext,
) -> list[Any] | None:
    merged = _expand_llm_tool_entries(registration, context)
    if explicit:
        merged = merged + list(explicit)
    return merged if merged else None


class NullStep:

    def __init__(self):
        self.status = None

    def stream(self, text, **kwargs):
        log_debug(f"[NullStep] {text}")


class LLMUser(param.Parameterized):
    """
    Mixin for classes that use prompts with LLMs.
    Provides parameters and methods for prompt templating and LLM interactions.
    """

    llm = param.ClassSelector(class_=Llm, doc="""
        The LLM implementation to query.""")

    llm_tools = param.List(default=[], doc="""
        Extra LLM tools for every ``_invoke_prompt`` / ``_stream_prompt`` on this actor.
        Entries may be tool instances or callables: ``f(context)`` or no-arg ``f()`` returning
        a single tool or a list of tools.""")

    prompts = param.Dict(default={
        "main": {"template": PROMPTS_DIR / "Actor" / "main.jinja2"},
    }, doc="""
        A dictionary of prompts, indexed by prompt name.
        Each prompt should be defined as a dictionary containing a template
        'template' and optionally a 'model' and 'tools'.""")

    steps_layout = param.ClassSelector(default=None, class_=(ListLike, NamedListLike), allow_None=True, doc="""
        The layout progress updates will be streamed to.""")

    template_overrides = param.Dict(default={}, doc="""
        Overrides specific blocks inside a prompt template without replacing
        the entire template. Useful for injecting domain knowledge, adding
        rules, or changing agent behaviour for specific tasks.

        Structure: ``{prompt_name: {block_name: new_content}}``.

        - **prompt_name**: The prompt to override, e.g. ``"main"``.
        - **block_name**: The Jinja2 block inside that template to replace.
          Common blocks (defined in ``Actor/main.jinja2``):
          ``global``, ``datetime``, ``instructions``, ``examples``,
          ``tools``, ``context``, ``errors``, ``footer``.
        - **new_content**: The replacement string. Use ``{{ super() }}`` to
          keep the original block content and append after it.

        Can be set at the class level (subclassing) or on an instance.

        **Example — subclassing**::

            INSTRUCTION_OVERRIDE = \"\"\"
            {{ super() }}

            <appended prompt>
            \"\"\"

            class UXSQLAgent(SQLAgent):
                template_overrides = {
                    "main": {"instructions": INSTRUCTION_OVERRIDE}
                }

        **Example — instance level**::

            agent = SQLAgent(template_overrides={
                "main": {"instructions": "{{ super() }}\\nBe concise."}
            })
        """)


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

    def _add_step(self, title: str = "", **kwargs):
        """Private contextmanager for adding steps to the interface.

        If self.interface is None, returns a nullcontext that captures calls.
        Otherwise, returns the interface's add_step contextmanager.
        """
        if self.steps_layout is not None and 'steps_layout' not in kwargs:
            kwargs['steps_layout'] = self.steps_layout
        return nullcontext(self._null_step) if self.interface is None else self.interface.add_step(title=title, **kwargs)

    async def _gather_prompt_context(self, prompt_name: str, messages: list[Message], context: TContext, **kwargs):
        """Gather context for the prompt template."""
        prompt_context = dict(kwargs)
        prompt_context["memory"] = context
        prompt_context["current_datetime"] = datetime.datetime.now()
        if self.interface:
            msgs = self.interface.serialize(
                filter_by=lambda msgs: [msg for msg in msgs if isinstance(msg.object, str)]
            )
            # Walk backward collecting assistant messages.
            # Messages after the last user message are "current";
            # messages before it (but after an earlier user message)
            # are labeled "from previous" so downstream consumers
            # can distinguish stale context from fresh output.
            history = []
            found_user = False
            for msg in reversed(msgs):
                if msg["role"].lower() == "user":
                    if found_user:
                        break
                    found_user = True
                    continue
                prefix = "(from previous) " if found_user else ""
                history.append(f'{prefix}{msg["role"]}: """\n{msg["content"]}\n"""\n')
            prompt_context["chat_history"] = "\n".join(reversed(history))
        return prompt_context

    async def _render_prompt(self, prompt_name: str, messages: list[Message], context: TContext, **kwargs) -> str:
        """Render a prompt template with context."""
        prompt_template = self._lookup_prompt_key(prompt_name, "template")
        overrides = self.template_overrides.get(prompt_name, {})
        prompt_context = await self._gather_prompt_context(prompt_name, messages, context, **kwargs)

        prompt_label = f"\033[92m{self.name}.prompts['{prompt_name}']['template']\033[0m"
        try:
            path_exists = Path(prompt_template).exists()
        except OSError:
            path_exists = False
        if isinstance(prompt_template, str) and not path_exists:
            # check if all the format_kwargs keys are contained in prompt_template
            format_kwargs = dict(**overrides, **prompt_context)
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
                **prompt_context
            )
        prompt = prompt.strip()
        log_debug(prompt_label)
        return prompt

    async def _invoke_prompt(
        self,
        prompt_name: str,
        messages: list[Message],
        context: TContext,
        response_model: type[BaseModel] | None = None,
        model_spec: str | None = None,
        model_index: int | None = None,
        model_kwargs: dict | None = None,
        tools: list | None = None,
        **prompt_kwargs
    ) -> Any:
        """
        Render a prompt and invoke the LLM.

        Parameters
        ----------
        prompt_name : str
            Name of the prompt template to use
        messages : list[Message]
            The conversation messages
        context : TContext
            The context dictionary
        response_model : type[BaseModel], optional
            Pydantic model to structure the response
        model_spec : str, optional
            Specification for which LLM to use
        model_kwargs : dict, optional
            Additional context variables for determining the model_spec or response_model
        model_index : int, optional
            The index of the model to subset if the model spec returns a list of models
        tools : list, optional
            Per-call tools forwarded to ``llm.invoke``, after :attr:`llm_tools` are expanded.
        **kwargs : dict
            Additional context variables for the prompt template

        Returns
        -------
        The structured response from the LLM
        """
        system = await self._render_prompt(prompt_name, messages, context, **prompt_kwargs)
        if response_model is None:
            try:
                response_model = self._get_model(prompt_name, **(model_kwargs or {}))
            except (KeyError, AttributeError):
                pass

        if model_spec is None:
            try:
                model_spec = self._lookup_prompt_key(prompt_name, "llm_spec")
            except KeyError:
                model_spec = self.llm_spec_key

        if model_index is not None:
            model_spec = model_spec[model_index]

        invoke_kw: dict = {}
        merged_tools = _merge_prompt_tools(self.llm_tools, tools, context)
        if merged_tools is not None:
            invoke_kw["tools"] = merged_tools

        result = await self.llm.invoke(
            messages=messages,
            system=system,
            response_model=response_model,
            model_spec=model_spec,
            **invoke_kw,
        )
        return result

    async def _stream_prompt(
        self,
        prompt_name: str,
        messages: list[Message],
        context: TContext,
        response_model: type[BaseModel] | None = None,
        model_spec: str | None = None,
        model_kwargs: dict | None = None,
        model_index: int | None = None,
        field: str | None = None,
        tools: list | None = None,
        **kwargs
    ):
        """
        Render a prompt and stream responses from the LLM.

        Parameters
        ----------
        prompt_name : str
            Name of the prompt template to use
        messages : list[Message]
            The conversation messages
        context : TContext
            The context dictionary
        response_model : type[BaseModel], optional
            Pydantic model to structure the response
        model_spec : str, optional
            Specification for which LLM to use
        model_kwargs : dict, optional
            Additional context variables for determining the model_spec or response_model
        model_index : int, optional
            The index of the model to subset if the model spec returns a list of models
        field : str, optional
            Specific field to extract from the response model
        tools : list, optional
            Per-call tools forwarded to ``llm.stream``, after :attr:`llm_tools` are expanded.
        **kwargs : dict
            Additional context variables for the prompt template

        Yields
        ------
        Chunks of the response from the LLM as they are generated
        """
        # Render the prompt
        system = await self._render_prompt(prompt_name, messages, context, **kwargs)

        # Determine the response model
        if response_model is None:
            try:
                response_model = self._get_model(prompt_name, **(model_kwargs or {}))
            except (KeyError, AttributeError):
                pass

        # Determine the model specification
        if model_spec is None:
            try:
                model_spec = self._lookup_prompt_key(prompt_name, "llm_spec")
            except KeyError:
                model_spec = self.llm_spec_key

        if model_index is not None:
            model_spec = model_spec[model_index]

        stream_merged = _merge_prompt_tools(self.llm_tools, tools, context)

        # Stream from the LLM
        async for chunk in self.llm.stream(
            messages=messages,
            system=system,
            response_model=response_model,
            model_spec=model_spec,
            field=field,
            tools=stream_merged,
        ):
            yield chunk

    @property
    def llm_spec_key(self):
        """
        Converts class name to a snake_case model identifier.
        Used for looking up model configurations in model_kwargs.
        """
        return class_name_to_llm_spec_key(type(self).__name__)



class Actor(LLMUser):

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The interface for the Coordinator to interact with.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._null_step = NullStep()

    def __init_subclass__(cls, **kwargs):
        """
        Apply wrap_logfire to all the subclasses' respond automatically
        """
        super().__init_subclass__(**kwargs)
        wrap_logfire_on_method(cls, "respond")

    def _add_step(self, title: str = "", **kwargs):
        """Private contextmanager for adding steps to the interface.

        If self.interface is None, returns a nullcontext that captures calls.
        Otherwise, returns the interface's add_step contextmanager.
        """
        return nullcontext(self._null_step) if self.interface is None else self.interface.add_step(title=title, **kwargs)

    async def _gather_prompt_context(self, prompt_name: str, messages: list[Message], context: TContext, **kwargs):
        """Gather context for the prompt template."""
        context = await super()._gather_prompt_context(prompt_name, messages, context, **kwargs)
        context["actor_name"] = self.name
        context["source_table_sep"] = SOURCE_TABLE_SEPARATOR
        return context

    @abstractmethod
    async def respond(
        self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]
    ) -> tuple[list[Any], ContextModel]:
        """
        Responds to the provided messages and context, returning
        a list of visual outputs and a ContextModel containing
        the context for subsequent Actors.
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

    purpose = param.String(default="", doc="""
        A descriptive statement of this actor's functionality and capabilities.
        Serves as a high-level explanation for other actors to understand
        what this actor does and when it might be useful to invoke it.""")

    input_schema: type[ContextModel] = ContextModel
    output_schema: type[ContextModel] = ContextModel

    async def prepare(self, context: TContext):
        pass

    async def requirements(self, messages: list[Message]) -> list[str]:
        return list(self.input_schema.__annotations__)

    def __str__(self):
        string = (
            f"- {self.name[:-5]}: {' '.join(self.purpose.strip().split())}\n"
            f"  Requires: `{'`, `'.join(self.input_schema.__required_keys__)}`\n"
            f"  Provides: `{'`, `'.join(self.output_schema.__required_keys__)}`\n"
        )
        if self.conditions:
            string += "  Conditions:\n" + "\n".join(f"  - {condition}" for condition in self.conditions) + "\n"
        if self.not_with:
            string += "  Not with:\n" + "\n".join(f"  - {not_with}" for not_with in self.not_with) + "\n"
        return string
