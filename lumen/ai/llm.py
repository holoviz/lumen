from __future__ import annotations

import asyncio
import base64
import json
import os
import traceback

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Literal, NotRequired, TypedDict,
)

import instructor
import panel as pn
import param
import requests

from instructor import Mode, patch
from instructor.dsl.partial import Partial
from instructor.processing.multimodal import Image
from openai import OpenAI as OpenAIClient
from pydantic import BaseModel

from .interceptor import Interceptor
from .services import (
    PROVIDER_ENV_VARS, AnthropicMixin, AzureMistralAIMixin, AzureOpenAIMixin,
    BedrockMixin, GenAIMixin, LlamaCppMixin, MistralAIMixin, OpenAIMixin,
)
from .utils import (
    format_exception, format_msg_content, log_debug, truncate_string,
)

if TYPE_CHECKING:
    from .tools import FunctionTool
    from .tools.mcp import MCPTool

class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | Image | list[dict[str, Any]]
    name: NotRequired[str]
    tool_call_id: NotRequired[str]
    tool_calls: NotRequired[list[dict[str, Any]]]


class ImageResponse(BaseModel):
    # To easily analyze images, we need instructor patch activated,
    # so we use a pass-thru dummy string basemodel
    output: str


BASE_MODES = list(Mode)

# LLM Provider Configuration
# Mapping from provider names to LLM class names
LLM_PROVIDERS = {
    'openai': 'OpenAI',
    'google': 'Google',
    'anthropic': 'Anthropic',
    'anthropic-bedrock': 'AnthropicBedrock',
    'bedrock': 'Bedrock',
    'mistral': 'MistralAI',
    'azure-openai': 'AzureOpenAI',
    'azure-mistral': 'AzureMistralAI',
    'groq': 'Groq',
    "ai-navigator": "AINavigator",
    "ai-catalyst": "AICatalyst",
    'ollama': 'Ollama',
    'llama-cpp': 'LlamaCpp',
    'litellm': 'LiteLLM',
}


def get_available_llm() -> type[Llm] | None:
    """
    Detect and instantiate an available LLM provider by checking environment variables
    and attempting to instantiate each provider in order.

    Returns
    -------
    type[Llm] | None
        The LLM class if successful, or None if no provider is available.
    """
    for provider, class_name in LLM_PROVIDERS.items():
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var and not os.environ.get(env_var):
            continue

        try:
            provider_cls = globals()[class_name]
            return provider_cls
        except KeyError:
            continue
    return None


class Llm(param.Parameterized):
    """
    Base class for LLM implementations with standardized client caching.

    Subclasses MUST implement _create_base_client().
    Subclasses MAY set _instructor_wrapper (default: "openai").
    Subclasses MAY override _create_instructor_client() or _get_completion_method().
    """

    create_kwargs = param.Dict(default={"max_retries": 1}, doc="""
        Additional keyword arguments to pass to the LLM provider
        when calling chat.completions.create. Defaults to no instructor retries
        since agents handle retries themselves.""")

    mode = param.Selector(default=Mode.JSON_SCHEMA, objects=BASE_MODES, doc="""
        The calling mode used by instructor to guide the LLM towards generating
        outputs matching the schema.""")

    interceptor = param.ClassSelector(default=None, class_=Interceptor, doc="""
        Intercepter instance to intercept LLM calls, e.g. for logging.""")

    model_kwargs = param.Dict(default={}, doc="""
        LLM model definitions indexed by type. Supported types include
        'default', 'reasoning' and 'sql'. Agents may pick which model to
        invoke for different reasons.""")

    tools = param.List(default=[], doc="""
        Default tools that are always available to this LLM instance.
        These are combined with any tools passed directly to invoke()
        or stream() calls. Accepts the same types as the tools argument
        on those methods (dicts, FunctionTool, or MCPTool instances).""")

    logfire_tags = param.List(default=None, doc="""
        Whether to log LLM calls and responses to logfire.
        If a list of tags is provided, those tags will be used for logging.
        Suppresses streaming responses if enabled since
        logfire does not track token usage on stream.""")

    display_name = param.String(default="", constant=True, doc="Display name for UI")

    select_models = param.List(default=[], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.7, bounds=(0, None), constant=True, doc="""
        The temperature to use for sampling.""")

    timeout = param.Number(default=120, bounds=(1, None), constant=True, doc="""
        The timeout in seconds for API calls.""")

    _ready = param.Boolean(default=False, doc="""
        Whether the LLM has been initialized and is ready to use.""")

    # Whether the LLM supports logging to logfire
    _supports_logfire = False

    # Whether the LLM supports streaming of any kind
    _supports_stream = True

    # Whether the LLM supports streaming of Pydantic model output
    _supports_model_stream = True

    # Whether the LLM supports vision (multimodal image content)
    _supports_vision = True

    # Used for `instructor.from_*` wrapping; subclasses may override
    _instructor_wrapper: str = "openai"

    __abstract = True

    def models(self) -> set[str]:
        """
        Return the set of available model identifiers from this provider.

        By default returns an empty set. Subclasses should override to
        query their provider's API for available models.

        Returns
        -------
        set[str]
            Set of available model identifiers.
        """
        return set()

    def __init__(self, **params):
        if "mode" in params:
            if isinstance(params["mode"], str):
                params["mode"] = Mode[params["mode"].upper()]
        super().__init__(**params)
        # Instance-level client caches
        self._base_client = None
        self._instructor_clients: dict[Mode, Any] = {}

        if self.logfire_tags is not None and not self._supports_logfire:
            raise ValueError(
                f"LLM {self.__class__.__name__} does not support logfire."
            )
        self._update_logfire_tags()
        if not self.model_kwargs.get("default"):
            raise ValueError(
                f"Please specify a 'default' model in the model_kwargs "
                f"parameter for {self.__class__.__name__}."
            )

    @param.depends("logfire_tags", watch=True)
    def _update_logfire_tags(self):
        if self.logfire_tags is not None and self._supports_logfire:
            import logfire
            logfire.configure(send_to_logfire=True)
            self._logfire = logfire.Logfire(tags=self.logfire_tags)
        else:
            self._logfire = None

    def _get_model_kwargs(self, model_spec: str | dict) -> dict[str, Any]:
        """
        Can specify model kwargs as a dict or as a string that is a key in the model_kwargs
        or as a string that is a model type; else the actual name of the model.
        """
        if isinstance(model_spec, dict):
            return model_spec

        model_kwargs = self.model_kwargs.get(model_spec) or self.model_kwargs["default"]
        return dict(model_kwargs)

    def _get_create_kwargs(self, response_model: type[BaseModel] | None) -> dict[str, Any]:
        kwargs = dict(self.create_kwargs)
        if not response_model:
            # Only available if instructor patched
            kwargs.pop("max_retries", None)
        return kwargs

    @property
    def _client_kwargs(self) -> dict[str, Any]:
        return {}

    def _create_base_client(self, **kwargs) -> Any:
        """Create the underlying SDK client (e.g., AsyncOpenAI, AsyncAnthropic)."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _create_base_client()")

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        """Wrap the base client with instructor. Override for non-standard wrapping."""
        wrapper_fn = getattr(instructor, f"from_{self._instructor_wrapper}")
        return wrapper_fn(base_client, mode=mode)

    def _get_completion_method(self) -> Callable:
        """Get completion method from base client. Override for non-OpenAI APIs."""
        return self._base_client.chat.completions.create

    def _get_cached_client(
        self,
        response_model: type[BaseModel] | None = None,
        model: str | None = None,
        **kwargs
    ) -> Callable:
        """Get cached client callable - base or instructor-wrapped depending on response_model."""
        mode = kwargs.pop("mode", self.mode)

        if self._base_client is None:
            self._base_client = self._create_base_client(**kwargs)

        if response_model:
            if mode not in self._instructor_clients:
                self._instructor_clients[mode] = self._create_instructor_client(self._base_client, mode)
            client = self._instructor_clients[mode]
            return partial(client.chat.completions.create, model=model, **self._get_create_kwargs(response_model))
        else:
            return partial(self._get_completion_method(), model=model, **self._get_create_kwargs(response_model))

    def _add_system_message(
        self,
        messages: list[Message],
        system: str,
        input_kwargs: dict[str, Any]
    ) -> tuple[list[Message], dict[str, Any]]:
        if system:
            messages = [Message(role="system", content=system)] + messages
        return messages, input_kwargs

    def _serialize_image_pane(self, image: bytes | pn.pane.image.ImageBase | Image) -> Image | None:
        if isinstance(image, Image):
            return image

        image = image.object if isinstance(image, pn.pane.image.ImageBase) else image
        if isinstance(image, (Path, str)) and Path(image).is_file():
            image_object = Image.from_path(image)
        elif isinstance(image, str):
            image_object = Image.from_url(image)
        elif isinstance(image, bytes):
            base64_str = base64.b64encode(image).decode('utf-8')
            image_object = Image.from_raw_base64(base64_str)
        else:
            image_object = None
        return image_object

    def _check_for_image(self, messages: list[Message]) -> tuple[list[Message], bool]:
        contains_image = False
        for i, message in enumerate(messages):
            content = message.get("content")
            if isinstance(content, (bytes, Image, pn.pane.image.ImageBase)):
                messages[i]["content"] = self._serialize_image_pane(content)
                contains_image = True

            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, (bytes, Image, pn.pane.image.ImageBase)):
                        new_content.append(self._serialize_image_pane(item))
                        contains_image = True
                    else:
                        new_content.append(item)
                messages[i]["content"] = new_content
        return messages, contains_image

    @staticmethod
    def _normalize_multimodal_messages(messages: list[Message]) -> list[Message]:
        """Convert instructor Image objects and bare strings in list content
        to OpenAI-native content-part dicts.

        When response_model is absent (e.g. during the tool-loop phase),
        the raw OpenAI client is used and it cannot handle instructor
        Image objects.  This method normalises them.
        """
        for message in messages:
            content = message.get("content")
            if isinstance(content, Image):
                message["content"] = [{
                    "type": "image_url",
                    "image_url": {"url": f"data:{content.media_type};base64,{content.source}"},
                }]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, Image):
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{item.media_type};base64,{item.source}"},
                        })
                    elif isinstance(item, str):
                        new_content.append({"type": "text", "text": item})
                    else:
                        new_content.append(item)
                message["content"] = new_content
        return messages

    @classmethod
    def warmup(cls, model_kwargs: dict | None):
        """
        Allows LLM provider to perform actions that ensure that
        the model(s) are ready to run, e.g. downloading the model
        files.
        """

    async def invoke(
        self,
        messages: list[Message],
        system: str = "",
        response_model: type[BaseModel] | None = None,
        allow_partial: bool = False,
        model_spec: str | dict = "default",
        tools: list[dict[str, Any] | FunctionTool | MCPTool] | None = None,
        **input_kwargs,
    ) -> BaseModel | str:
        """
        Invokes the LLM and returns its response.

        Arguments
        ---------
        messages: list[Message]
            A list of messages to feed to the LLM.
        system: str
            A system message to provide to the LLM.
        response_model: BaseModel | None
            A Pydantic model that the LLM should materialize.
        allow_partial: bool
            Whether to allow the LLM to only partially fill
            the provided response_model.
        tools: list[dict[str, Any] | FunctionTool | MCPTool] | None
            Tool definitions, FunctionTool, or MCPTool instances to pass through.
            When ``response_model`` is also given, the client runs tool calls in a loop
            until none are requested, then requests the structured response (tools may be unused).
        model: Literal['default' | 'reasoning' | 'sql']
            The model as listed in the model_kwargs parameter
            to invoke to answer the query.

        Returns
        -------
        The completed response_model.
        """
        system = system.strip().replace("\n\n", "\n")
        messages, input_kwargs = self._add_system_message(messages, system, input_kwargs)
        max_tool_rounds = int(input_kwargs.pop("max_tool_rounds", 16))

        kwargs = dict(self._client_kwargs)
        kwargs.update(input_kwargs)
        combined_tools = self._combine_tools(tools)
        tool_specs, tool_instances, tool_contexts = self._normalize_tools(combined_tools)
        if tool_specs is not None:
            kwargs["tools"] = tool_specs

        messages, contains_image = self._check_for_image(messages)
        if contains_image:
            # Currently instructor does not support streaming with multimodal
            # https://github.com/567-labs/instructor/issues/1872
            kwargs["stream"] = False

        structured_model: type[BaseModel] | None = None
        if response_model is not None:
            if allow_partial and issubclass(response_model, BaseModel):
                structured_model = Partial[response_model]  # type: ignore[assignment]
            else:
                structured_model = response_model  # type: ignore[assignment]
        elif contains_image:
            structured_model = ImageResponse

        output = await self._run_tool_loop(
            messages,
            structured_model,
            tool_instances,
            tool_contexts,
            model_spec=model_spec,
            max_tool_rounds=max_tool_rounds,
            **kwargs
        )
        if output is None or output == "":
            raise ValueError("LLM failed to return valid output.")
        return output

    async def _run_tool_loop(
        self,
        messages: list[Message],
        structured_model: type[BaseModel] | None,
        tool_instances: dict,
        tool_contexts: dict,
        model_spec: str | dict = "default",
        max_tool_rounds: int = 16,
        **kwargs
    ) -> BaseModel | str:
        if structured_model is not None and not tool_instances:
            kwargs["response_model"] = structured_model
        else:
            kwargs.pop("response_model", None)
            # Without response_model the raw client is used, which
            # cannot handle instructor Image objects in list content.
            messages = self._normalize_multimodal_messages(messages)

        output = await self.run_client(model_spec, messages, **kwargs)
        if not tool_instances:
            return output

        messages_curr = list(messages)
        for _ in range(max_tool_rounds):
            tool_calls = self._extract_tool_calls(output)
            if not tool_calls:
                break
            tool_calls_message = self._tool_calls_message(tool_calls)
            tool_messages = await self._run_tool_calls(
                tool_instances, tool_calls, tool_contexts, messages_curr
            )
            if not tool_messages:
                break
            messages_curr = messages_curr + [tool_calls_message] + tool_messages
            output = await self.run_client(model_spec, messages_curr, **kwargs)

        if structured_model:
            kwargs["response_model"] = structured_model
            output = await self.run_client(model_spec, messages_curr, **kwargs)
        return output

    @classmethod
    def _get_delta(cls, chunk) -> str:
        if chunk.choices:
            return chunk.choices[0].delta.content or ""
        return ""

    @classmethod
    def _get_content(cls, response) -> str | BaseModel:
        """Extract content from a non-streaming response. Override for non-OpenAI APIs."""
        if hasattr(response, "choices"):
            return response.choices[0].message.content
        elif isinstance(response, (BaseModel, str)):
            return response
        return str(response)

    def _combine_tools(
        self,
        tools: list[dict[str, Any] | FunctionTool | MCPTool] | None,
    ) -> list[dict[str, Any] | FunctionTool | MCPTool] | None:
        """Combine instance-level ``self.tools`` with per-call *tools*."""
        if self.tools and tools:
            return list(self.tools) + list(tools)
        elif self.tools:
            return list(self.tools)
        return tools

    @classmethod
    def _normalize_tools(
        cls,
        tools: list[dict[str, Any] | FunctionTool | MCPTool] | None,
    ) -> tuple[list[dict[str, Any]] | None, dict[str, FunctionTool | MCPTool], dict[str, Any]]:
        tool_instances: dict[str, FunctionTool | MCPTool] = {}
        tool_contexts: dict[str, Any] = {}
        if tools is None:
            return None, tool_instances, tool_contexts
        tool_specs: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict) and "tool" in tool:
                tool_context = tool.get("context")
                tool = tool.get("tool")
            else:
                tool_context = None
            if callable(tool) and hasattr(tool, "__lumen_tool_annotations__"):
                from .tools import FunctionTool
                tool = FunctionTool(tool)
            if hasattr(tool, "_model"):
                tool_instances[tool.name] = tool  # type: ignore[assignment]
                if tool_context is not None:
                    tool_contexts[tool.name] = tool_context
                tool_specs.append(cls._tool_spec_from_tool(tool))  # type: ignore[arg-type]
            else:
                tool_specs.append(tool)  # type: ignore[arg-type]
        return tool_specs, tool_instances, tool_contexts

    @classmethod
    def _tool_spec_from_tool(cls, tool: FunctionTool | MCPTool) -> dict[str, Any]:
        schema = tool._model.model_json_schema()
        description = tool.purpose or schema.get("description") or ""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": description,
                "parameters": schema,
            },
        }

    @classmethod
    def _extract_tool_calls(cls, response: Any) -> list[Any]:
        if hasattr(response, "choices") and response.choices:
            message = getattr(response.choices[0], "message", None)
            tool_calls = getattr(message, "tool_calls", None) if message else None
            if tool_calls:
                return list(tool_calls)
        if isinstance(response, dict) and response.get("tool_calls"):
            return list(response["tool_calls"])
        return []

    @classmethod
    def _extract_stream_tool_calls(cls, chunk: Any) -> list[dict[str, Any]]:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = getattr(chunk.choices[0], "delta", None)
            tool_calls = getattr(delta, "tool_calls", None) if delta else None
            if tool_calls:
                return list(tool_calls)
        if isinstance(chunk, dict):
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                tool_calls = delta.get("tool_calls") if isinstance(delta, dict) else None
                if tool_calls:
                    return list(tool_calls)
        return []

    @classmethod
    def _accumulate_tool_calls(
        cls,
        accum: dict[int, dict[str, Any]],
        order: list[int],
        tool_calls: list[dict[str, Any]],
    ):
        for call in tool_calls:
            if isinstance(call, dict):
                index = call.get("index")
                call_id = call.get("id")
                function = call.get("function") or {}
                name = function.get("name") or call.get("name")
                arguments = function.get("arguments") or call.get("arguments") or ""
            else:
                index = getattr(call, "index", None)
                call_id = getattr(call, "id", None)
                function = getattr(call, "function", None)
                name = getattr(function, "name", None) if function else None
                arguments = getattr(function, "arguments", None) if function else ""
            if index is None:
                index = max(order, default=-1) + 1
            if index not in accum:
                accum[index] = {"id": call_id, "name": name, "arguments": ""}
                order.append(index)
            if call_id:
                accum[index]["id"] = call_id
            if name:
                accum[index]["name"] = name
            if arguments:
                accum[index]["arguments"] += arguments

    @classmethod
    def _tool_calls_from_accum(cls, accum: dict[int, dict[str, Any]], order: list[int]) -> list[dict[str, Any]]:
        tool_calls = []
        for index in order:
            entry = accum[index]
            tool_calls.append({
                "id": entry.get("id"),
                "type": "function",
                "function": {
                    "name": entry.get("name"),
                    "arguments": entry.get("arguments", ""),
                },
            })
        return tool_calls

    @classmethod
    def _normalize_tool_call_for_message(cls, call: Any) -> dict[str, Any]:
        if isinstance(call, dict):
            call_dict = dict(call)
            function = call_dict.get("function")
            if function is None:
                function = {
                    "name": call_dict.get("name"),
                    "arguments": call_dict.get("arguments", ""),
                }
            elif not isinstance(function, dict):
                function = {
                    "name": getattr(function, "name", None),
                    "arguments": getattr(function, "arguments", ""),
                }
            else:
                function = dict(function)
            call_dict["function"] = function
            call_dict["type"] = call_dict.get("type") or "function"
            return call_dict

        function = getattr(call, "function", None)
        return {
            "id": getattr(call, "id", None),
            "type": getattr(call, "type", None) or "function",
            "function": {
                "name": getattr(function, "name", None) if function else getattr(call, "name", None),
                "arguments": (
                    getattr(function, "arguments", "")
                    if function
                    else getattr(call, "arguments", "")
                ),
            },
        }

    @classmethod
    def _tool_calls_message(cls, tool_calls: list[dict[str, Any]]) -> Message:
        return Message(
            role="assistant",
            content="",
            tool_calls=[cls._normalize_tool_call_for_message(call) for call in tool_calls],
        )

    @classmethod
    def _format_tool_result(cls, result: Any) -> str:
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result)
        except TypeError:
            return str(result)

    @classmethod
    def _parse_tool_call(cls, call: Any) -> tuple[str | None, dict[str, Any], str | None]:
        if isinstance(call, dict):
            function = call.get("function") or {}
            name = function.get("name") or call.get("name")
            args = function.get("arguments") or call.get("arguments") or {}
            call_id = call.get("id")
        else:
            function = getattr(call, "function", None)
            name = getattr(function, "name", None) if function else None
            args = getattr(function, "arguments", None) if function else {}
            call_id = getattr(call, "id", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return name, args or {}, call_id

    async def _run_tool_calls(
        self,
        tool_instances: dict[str, FunctionTool | MCPTool],
        tool_calls: list[Any],
        tool_contexts: dict[str, Any],
        messages: list[Message],
    ) -> list[Message]:
        from .tools import FunctionTool, MCPTool

        async def run_single_tool_call(call: Any) -> Message | None:
            name, arguments, call_id = self._parse_tool_call(call)
            if not name:
                log_debug(
                    f"LLM tool call skipped: missing tool name (call_id={call_id!r})",
                    prefix="[LLM tools]",
                )
                return None
            if name not in tool_instances:
                log_debug(
                    "LLM tool call skipped: unknown tool "
                    f"{name!r} (call_id={call_id!r}); registered: {sorted(tool_instances)}",
                    prefix="[LLM tools]",
                )
                return None
            tool = tool_instances[name]
            context = tool_contexts.get(name, {})
            for requirement in tool.requires:
                if requirement not in arguments and requirement in context:
                    arguments[requirement] = context[requirement]
            try:
                args_repr = truncate_string(
                    json.dumps(arguments, default=str, ensure_ascii=False),
                    max_length=4000,
                )
                log_debug(
                    f"LLM tool call start tool={name!r} call_id={call_id!r} arguments={args_repr}",
                    prefix="[LLM tools]",
                )
                if isinstance(tool, MCPTool):
                    result = await tool.execute(**arguments)
                elif isinstance(tool, FunctionTool):
                    if asyncio.iscoroutinefunction(tool.function):
                        result = await tool.function(**arguments)
                    else:
                        # Synchronous function, run in thread
                        result = await asyncio.to_thread(tool.function, **arguments)
                else:
                    raise TypeError(f"Unsupported tool type for {name!r}: {type(tool)!r}")
                formatted = self._format_tool_result(result)
                log_debug(
                    f"LLM tool call result tool={name!r} call_id={call_id!r}\n"
                    f"{truncate_string(formatted, max_length=16000)}",
                    prefix="[LLM tools]",
                    show_length=True,
                )
            except Exception:
                log_debug(
                    [
                        f"LLM tool call failed tool={name!r} call_id={call_id!r}",
                        traceback.format_exc(),
                    ],
                    prefix="[LLM tools]",
                    show_sep="above",
                )
                raise
            return Message(
                role="tool",
                content=formatted,
                name=name,
                tool_call_id=call_id,
            )
        results = await asyncio.gather(
            *(run_single_tool_call(call) for call in tool_calls)
        )
        return [msg for msg in results if msg is not None]

    async def initialize(self, log_level: str):
        try:
            self._ready = False
            await self.invoke(
                messages=[{'role': 'user', 'content': 'Ready? "Y" or "N"'}],
                model_spec="ui",
            )
            self._ready = True
        except Exception as e:
            self._ready = False
            raise e

    async def stream(
        self,
        messages: list[Message],
        system: str = "",
        response_model: type[BaseModel] | None = None,
        field: str | None = None,
        model_spec: str | dict = "default",
        tools: list[dict[str, Any] | FunctionTool | MCPTool] | None = None,
        **kwargs,
    ):
        """
        Invokes the LLM and streams its response.

        Arguments
        ---------
        messages: list[Message]
            A list of messages to feed to the LLM.
        system: str
            A system message to provide to the LLM.
        response_model: type[BaseModel] | None
            A Pydantic model that the LLM should materialize.
        field: str
            The field in the response_model to stream.
        tools: list[dict[str, Any] | FunctionTool | MCPTool] | None
            Tool definitions, FunctionTool, or MCPTool instances to pass through.
        model: Literal['default' | 'reasoning' | 'sql']
            The model as listed in the model_kwargs parameter
            to invoke to answer the query.

        Yields
        ------
        The string or response_model field.
        """
        combined_tools = self._combine_tools(tools)
        tool_specs, tool_instances, tool_contexts = self._normalize_tools(combined_tools)
        messages, contains_image = self._check_for_image(messages)
        if self.logfire_tags is not None or contains_image:
            output = await self.invoke(
                messages,
                system=system,
                response_model=response_model,
                model_spec=model_spec,
                tools=tool_specs,
                **kwargs,
            )
            if response_model is not None:
                # Return Pydantic model as-is, or extract field if specified
                if field is not None and hasattr(output, field):
                    output = getattr(output, field)
            elif isinstance(output, str):
                pass  # Already a string
            else:
                # No response_model and not a string - extract content from raw response
                output = self._get_content(output)
            yield output
            return

        if ((response_model and not self._supports_model_stream) or
            not self._supports_stream):
            output = await self.invoke(
                messages,
                system=system,
                response_model=response_model,
                model_spec=model_spec,
                tools=tool_specs,
                **kwargs,
            )
            yield self._get_content(output)
            return

        string = ""
        chunks = await self.invoke(
            messages,
            system=system,
            response_model=response_model,
            stream=True,
            allow_partial=True,
            model_spec=model_spec,
            tools=tool_specs,
            **kwargs,
        )
        if isinstance(chunks, BaseModel):
            yield getattr(chunks, field) if field is not None else chunks
            return

        tool_call_accum: dict[int, dict[str, Any]] = {}
        tool_call_order: list[int] = []
        response_id: str | None = None
        try:
            async for chunk in chunks:
                chunk_response = getattr(chunk, "response", None)
                chunk_response_id = getattr(chunk_response, "id", None)
                if chunk_response_id:
                    response_id = chunk_response_id
                if response_model is None:
                    string += self._get_delta(chunk)
                    yield string
                    tool_calls = self._extract_stream_tool_calls(chunk)
                    if tool_calls:
                        self._accumulate_tool_calls(tool_call_accum, tool_call_order, tool_calls)
                else:
                    yield getattr(chunk, field) if field is not None else chunk
        except TypeError:
            # Handle synchronous iterators
            for chunk in chunks:
                chunk_response = getattr(chunk, "response", None)
                chunk_response_id = getattr(chunk_response, "id", None)
                if chunk_response_id:
                    response_id = chunk_response_id
                if response_model is None:
                    string += self._get_delta(chunk)
                    yield string
                    tool_calls = self._extract_stream_tool_calls(chunk)
                    if tool_calls:
                        self._accumulate_tool_calls(tool_call_accum, tool_call_order, tool_calls)
                else:
                    yield getattr(chunk, field) if field is not None else chunk

        if response_model is None and tool_instances and tool_call_accum:
            tool_calls = self._tool_calls_from_accum(tool_call_accum, tool_call_order)
            tool_messages = await self._run_tool_calls(tool_instances, tool_calls, tool_contexts, messages)
            if tool_messages:
                if (
                    getattr(self, "api", None) == "responses"
                    and hasattr(self, "_tool_messages_to_response_inputs")
                ):
                    next_messages = self._tool_messages_to_response_inputs(tool_messages)
                    next_kwargs = dict(kwargs)
                    if response_id:
                        next_kwargs["previous_response_id"] = response_id
                    async for chunk in self.stream(
                        next_messages,  # type: ignore[arg-type]
                        system=system,
                        response_model=response_model,
                        field=field,
                        model_spec=model_spec,
                        tools=tools,
                        **next_kwargs,
                    ):
                        yield chunk
                else:
                    tool_calls_message = self._tool_calls_message(tool_calls)
                    async for chunk in self.stream(
                        messages + [tool_calls_message] + tool_messages,
                        system=system,
                        response_model=response_model,
                        field=field,
                        model_spec=model_spec,
                        tools=tools,
                        **kwargs,
                    ):
                        yield chunk

    async def run_client(self, model_spec: str | dict, messages: list[Message], **kwargs):
        log_debug(f"Input messages: \033[95m{len(messages)} messages\033[0m including system")
        previous_role = None
        for i, message in enumerate(messages):
            role = message["role"]
            if role == "system":
                continue
            role_char = "u" if role == "user" else "a"
            log_debug(f"Message \033[95m{i} ({role_char})\033[0m: {format_msg_content(message['content'])}")
            if previous_role == role:
                log_debug(
                    "\033[91mWARNING: Two consecutive messages from the same role; "
                    "some providers disallow this.\033[0m"
                )
            previous_role = role

        client = await self.get_client(model_spec, **kwargs)
        result = await client(messages=messages, **kwargs)
        if response_model := kwargs.get("response_model"):
            log_debug(f"Response model: \033[93m{response_model.__name__!r}\033[0m")
            if isinstance(result, ImageResponse):
                result = result.output
        log_debug(f"LLM Response: \033[95m{truncate_string(str(result), max_length=1000)}\033[0m\n---")
        return result


class LlamaCpp(Llm, LlamaCppMixin):
    """
    A LLM implementation using Llama.cpp Python wrapper together with huggingface_hub to fetch the models.
    """

    chat_format = param.String(constant=True)

    display_name = param.String(default="Llama.cpp", constant=True)

    mode = param.Selector(default=Mode.JSON_SCHEMA, objects=BASE_MODES)

    model_kwargs = param.Dict(default={
        "default": {
            "repo_id": "unsloth/Qwen3-32B-GGUF",
            "filename": "Qwen3-32B-Q5_K_M.gguf",
            "chat_format": "qwen",
        },
    })

    select_models = param.List(default=[
        "unsloth/Qwen3-32B-GGUF",
        "unsloth/Qwen3-Coder-32B-A3B-Instruct-GGUF",
        "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF",
        "meta-llama/Llama-3.3-70B-Instruct-GGUF",
        "nvidia/Nemotron-3-Nano-30B-GGUF"
    ], constant=True)

    temperature = param.Number(default=0.4, bounds=(0, None), constant=True)

    # LlamaCpp doesn't use from_* wrapper - uses patch(create=...)
    _instructor_wrapper = None

    def _get_model_kwargs(self, model_spec: str | dict) -> dict[str, Any]:
        if isinstance(model_spec, dict):
            return model_spec

        if model_spec in self.model_kwargs or "/" not in model_spec:
            model_kwargs = super()._get_model_kwargs(model_spec)
        else:
            base_kwargs = self.model_kwargs["default"]
            model_kwargs = self.resolve_model_spec(model_spec, base_kwargs)

        if "n_ctx" not in model_kwargs:
            model_kwargs["n_ctx"] = 0

        try:
            return self._instantiate_client_kwargs(model_kwargs=model_kwargs)
        except Exception:
            return dict(model_kwargs)

    @property
    def _client_kwargs(self) -> dict[str, Any]:
        return {"temperature": self.temperature}

    def _create_base_client(self, **kwargs) -> Any:
        return self._instantiate_client(**kwargs)

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        raw_client = base_client.create_chat_completion_openai_v1
        return patch(create=raw_client, mode=mode)

    def _get_cache_key(self, model_spec: str | dict, mode: Mode) -> tuple:
        return ("llamacpp", str(model_spec), mode)

    def _load_and_cache_model(self, model_spec: str | dict, mode: Mode, **kwargs) -> Any:
        base_client = self._create_base_client(**kwargs)
        client_callable = self._create_instructor_client(base_client, mode)
        pn.state.cache[self._get_cache_key(model_spec, mode)] = client_callable
        return client_callable

    @classmethod
    def warmup(cls, model_kwargs: dict | None):
        model_kwargs = model_kwargs or cls.model_kwargs
        if 'default' not in model_kwargs:
            model_kwargs['default'] = cls.model_kwargs['default']
        cls._warmup_models(model_kwargs)

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        mode = model_kwargs.pop("mode", self.mode)

        cache_key = self._get_cache_key(model_spec, mode)
        if cached := pn.state.cache.get(cache_key):
            return cached

        llm_kwargs = self._instantiate_client_kwargs(
            model_kwargs=model_kwargs,
            n_gpu_layers=-1,
            seed=128,
            logits_all=False,
        )

        return await asyncio.to_thread(self._load_and_cache_model, model_spec, mode, **llm_kwargs)

    async def run_client(self, model_spec: str | dict, messages: list[Message], **kwargs):
        client = await self.get_client(model_spec, **kwargs)
        return client(messages=messages, **kwargs)


class OpenAI(Llm, OpenAIMixin):
    """
    An LLM implementation using the OpenAI cloud.
    """

    api = param.Selector(default="chat_completions", objects=["chat_completions", "responses"], doc="""
        OpenAI API primitive to use.
        - ``chat_completions``: Uses ``/v1/chat/completions`` (default)
        - ``responses``: Uses ``/v1/responses``""")

    display_name = param.String(default="OpenAI", constant=True)

    mode = param.Selector(default=Mode.TOOLS)

    model_kwargs = param.Dict(default={
        "default": {"model": "gpt-5.4-mini"},  # Use standard models, not reasoning models (gpt-5, o4-mini)
        "ui": {"model": "gpt-5.4-nano"},
    })

    select_models = param.List(default=[
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano"
    ], constant=True, doc="""Warning: Reasoning models (gpt-5, o4-mini) are much slower and not suitable for dialog interfaces.""")

    temperature = param.Number(default=0.25, bounds=(0, None), constant=True)

    _supports_logfire = True

    @classmethod
    def _resolve_openai_mode(cls, mode: Mode) -> Mode:
        if mode in (Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS):
            return mode
        return Mode.RESPONSES_TOOLS

    @classmethod
    def _transform_responses_tools(cls, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return tools
        transformed: list[dict[str, Any]] = []
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get("type") == "function"
                and isinstance(tool.get("function"), dict)
            ):
                # Chat Completions format -> Responses format
                function = tool["function"]
                transformed.append({
                    "type": "function",
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                })
            else:
                transformed.append(tool)
        return transformed

    @classmethod
    def _tool_messages_to_response_inputs(cls, tool_messages: list[Message]) -> list[dict[str, Any]]:
        inputs: list[dict[str, Any]] = []
        for message in tool_messages:
            call_id = message.get("tool_call_id")
            if not call_id:
                continue
            inputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": message["content"],
            })
        return inputs

    def models(self) -> set[str]:
        """Return the set of available model identifiers from OpenAI."""
        client = OpenAIClient(api_key=self.api_key, timeout=5)
        return {m.id for m in client.models.list().data}

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    def _get_model_kwargs(self, model_spec: str | dict) -> dict[str, Any]:
        model_kwargs = super()._get_model_kwargs(model_spec)
        instance_kwargs = self._instantiate_client_kwargs()
        return {**instance_kwargs, **model_kwargs}

    def _create_base_client(self, **kwargs) -> Any:
        client = self._instantiate_client(async_client=True, **kwargs)
        if self.logfire_tags:
            self._logfire.instrument_openai(client)
        return client

    def _get_completion_method(self) -> Callable:
        if self.api == "responses":
            return self._base_client.responses.create
        return super()._get_completion_method()

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        if self.api == "responses":
            mode = self._resolve_openai_mode(mode)
        if self.interceptor:
            self.interceptor.patch_client(base_client, mode="store_inputs")
        wrapped = instructor.from_openai(base_client, mode=mode)
        if self.interceptor:
            self.interceptor.patch_client_response(wrapped)
        return wrapped

    @classmethod
    def _get_content(cls, response) -> str | BaseModel:
        if hasattr(response, "output_text"):
            return response.output_text or ""
        return super()._get_content(response)

    @classmethod
    def _get_delta(cls, chunk) -> str:
        event_type = getattr(chunk, "type", None)
        if isinstance(event_type, str) and event_type.startswith("response."):
            if event_type == "response.output_text.delta":
                return getattr(chunk, "delta", "") or ""
            return ""
        if isinstance(chunk, dict):
            chunk_type = chunk.get("type")
            if isinstance(chunk_type, str) and chunk_type.startswith("response."):
                if chunk_type == "response.output_text.delta":
                    return chunk.get("delta") or ""
                return ""
            if chunk_type == "response.output_text.delta":
                return chunk.get("delta") or ""
        return super()._get_delta(chunk)

    @classmethod
    def _extract_tool_calls(cls, response: Any) -> list[Any]:
        output_items = getattr(response, "output", None)
        if output_items is None and isinstance(response, dict):
            output_items = response.get("output")
        if output_items:
            tool_calls: list[dict[str, Any]] = []
            for item in output_items:
                item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                if item_type != "function_call":
                    continue
                call_id = item.get("call_id") if isinstance(item, dict) else getattr(item, "call_id", None)
                name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
                arguments = item.get("arguments") if isinstance(item, dict) else getattr(item, "arguments", "")
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments or ""},
                })
            if tool_calls:
                return tool_calls
        return super()._extract_tool_calls(response)

    @classmethod
    def _extract_stream_tool_calls(cls, chunk: Any) -> list[dict[str, Any]]:
        event_type = getattr(chunk, "type", None)
        if event_type is None and isinstance(chunk, dict):
            event_type = chunk.get("type")

        if event_type == "response.output_item.added":
            item = getattr(chunk, "item", None)
            output_index = getattr(chunk, "output_index", 0)
            if item is not None and getattr(item, "type", None) == "function_call":
                return [{
                    "index": output_index,
                    "id": getattr(item, "call_id", None),
                    "type": "function",
                    "function": {"name": getattr(item, "name", None), "arguments": ""},
                }]
        elif event_type == "response.function_call_arguments.delta":
            return [{
                "index": getattr(chunk, "output_index", 0),
                "type": "function",
                "function": {"name": None, "arguments": getattr(chunk, "delta", "") or ""},
            }]
        elif event_type == "response.function_call_arguments.done":
            return [{
                "index": getattr(chunk, "output_index", 0),
                "type": "function",
                "function": {
                    "name": getattr(chunk, "name", None),
                    "arguments": getattr(chunk, "arguments", "") or "",
                },
            }]

        if isinstance(chunk, dict):
            if chunk.get("type") == "response.output_item.added":
                item = chunk.get("item") or {}
                if item.get("type") == "function_call":
                    return [{
                        "index": chunk.get("output_index", 0),
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {"name": item.get("name"), "arguments": ""},
                    }]
            elif chunk.get("type") == "response.function_call_arguments.delta":
                return [{
                    "index": chunk.get("output_index", 0),
                    "type": "function",
                    "function": {"name": None, "arguments": chunk.get("delta") or ""},
                }]
            elif chunk.get("type") == "response.function_call_arguments.done":
                return [{
                    "index": chunk.get("output_index", 0),
                    "type": "function",
                    "function": {
                        "name": chunk.get("name"),
                        "arguments": chunk.get("arguments") or "",
                    },
                }]

        return super()._extract_stream_tool_calls(chunk)

    async def _run_tool_loop(
        self,
        messages: list[Message],
        structured_model: type[BaseModel] | None,
        tool_instances: dict,
        tool_contexts: dict,
        model_spec: str | dict = "default",
        max_tool_rounds: int = 16,
        **kwargs
    ) -> BaseModel | str:
        if self.api != "responses":
            return await super()._run_tool_loop(
                messages, structured_model, tool_instances, tool_contexts, model_spec, max_tool_rounds, **kwargs
            )

        if structured_model is not None and not tool_instances:
            kwargs["response_model"] = structured_model
        else:
            kwargs.pop("response_model", None)

        output = await self.run_client(model_spec, messages, **kwargs)
        if not tool_instances:
            return output

        for _ in range(max_tool_rounds):
            tool_calls = self._extract_tool_calls(output)
            if not tool_calls:
                break
            tool_messages = await self._run_tool_calls(
                tool_instances, tool_calls, tool_contexts, messages
            )
            if not tool_messages:
                break
            tool_outputs = self._tool_messages_to_response_inputs(tool_messages)
            if not tool_outputs:
                break
            next_kwargs = dict(kwargs)
            response_id = getattr(output, "id", None)
            if response_id:
                next_kwargs["previous_response_id"] = response_id
            output = await self.run_client(model_spec, tool_outputs, **next_kwargs)

        if structured_model:
            final_kwargs = dict(kwargs)
            final_kwargs["response_model"] = structured_model
            response_id = getattr(output, "id", None)
            if response_id:
                final_kwargs["previous_response_id"] = response_id
            output = await self.run_client(model_spec, [], **final_kwargs)
        return output

    async def get_client(self, model_spec: str | dict, response_model: type[BaseModel] | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        log_debug(f"LLM Model: \033[96m{model!r}\033[0m")
        mode = model_kwargs.pop("mode", self.mode)

        if self.api == "responses":
            if self._base_client is None:
                self._base_client = self._create_base_client(**model_kwargs)
            if response_model:
                mode = self._resolve_openai_mode(mode)
                if mode not in self._instructor_clients:
                    self._instructor_clients[mode] = self._create_instructor_client(self._base_client, mode)
                client = self._instructor_clients[mode]
                client_callable = partial(client.responses.create, model=model, **self._get_create_kwargs(response_model))
            else:
                client_callable = partial(self._base_client.responses.create, model=model, **self._get_create_kwargs(response_model))
        else:
            model_kwargs["mode"] = mode
            client_callable = self._get_cached_client(response_model, model=model, **model_kwargs)

        # Add timeout to the partial
        return partial(client_callable.func, *client_callable.args, timeout=self.timeout, **client_callable.keywords)

    async def run_client(self, model_spec: str | dict, messages: list[Message] | list[dict[str, Any]], **kwargs):
        if self.api == "responses":
            log_debug(f"Input messages: \033[95m{len(messages)} messages\033[0m including system")
            for i, message in enumerate(messages):
                role = message.get("role") if isinstance(message, dict) else None
                content = message.get("content") if isinstance(message, dict) else None
                if role == "system":
                    continue
                if role in ("user", "assistant", "tool"):
                    role_char = "u" if role == "user" else "a"
                    log_debug(f"Message \033[95m{i} ({role_char})\033[0m: {format_msg_content(content)}")
                else:
                    item_type = message.get("type") if isinstance(message, dict) else type(message).__name__
                    log_debug(f"Message \033[95m{i}\033[0m: [{item_type}] {truncate_string(str(message), max_length=2000)}")

            if kwargs.get("tools"):
                kwargs = dict(kwargs)
                kwargs["tools"] = self._transform_responses_tools(kwargs.get("tools"))
            client = await self.get_client(model_spec, **kwargs)
            result = await client(input=messages, **kwargs)
            log_debug(f"LLM Response: \033[95m{truncate_string(str(result), max_length=1000)}\033[0m\n---")
            return result

        return await super().run_client(model_spec, messages, **kwargs)


class AzureOpenAI(Llm, AzureOpenAIMixin):
    """
    A LLM implementation that uses the Azure OpenAI integration.
    Inherits from AzureOpenAIMixin which extends OpenAIMixin, so it has access to all OpenAI functionality
    plus Azure-specific configuration.
    """

    display_name = param.String(default="Azure OpenAI", constant=True)

    mode = param.Selector(default=Mode.TOOLS)

    model_kwargs = param.Dict(default={
        "default": {"model": "gpt-4o-mini"},
        "edit": {"model": "gpt-4o"},
    })

    select_models = param.List(default=[
        "gpt-35-turbo",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini"
    ], constant=True)

    temperature = param.Number(default=1, bounds=(0, None), constant=True)

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    def _get_model_kwargs(self, model_spec: str | dict) -> dict[str, Any]:
        model_kwargs = super()._get_model_kwargs(model_spec)
        instance_kwargs = self._instantiate_client_kwargs()
        return {**instance_kwargs, **model_kwargs}

    def _create_base_client(self, **kwargs) -> Any:
        return self._instantiate_client(async_client=True, **kwargs)

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        if self.interceptor:
            self.interceptor.patch_client(base_client, mode="store_inputs")
        wrapped = super()._create_instructor_client(base_client, mode)
        if self.interceptor:
            self.interceptor.patch_client_response(wrapped)
        return wrapped

    async def get_client(self, model_spec: str | dict, response_model: type[BaseModel] | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        model_kwargs["mode"] = model_kwargs.pop("mode", self.mode)

        client_callable = self._get_cached_client(response_model, model=model, **model_kwargs)
        return partial(client_callable.func, *client_callable.args, timeout=self.timeout, **client_callable.keywords)


class MistralAI(Llm, MistralAIMixin):
    """
    A LLM implementation that calls Mistral AI.
    """

    display_name = param.String(default="Mistral AI", constant=True)

    mode = param.Selector(default=Mode.MISTRAL_TOOLS, objects=[Mode.JSON_SCHEMA, Mode.MISTRAL_TOOLS])

    model_kwargs = param.Dict(default={
        "default": {"model": "mistral-small-latest"},
        "edit": {"model": "mistral-medium-latest"},
    })

    select_models = param.List(default=[
        "mistral-medium-latest",
        "magistral-medium-latest",
        "mistral-large-latest",
        "magistral-small-latest",
        "mistral-small-latest",
        "codestral-latest",
        "ministral-8b-latest",
        "ministral-3b-latest",
        "devstral-small-latest"
    ], constant=True)

    temperature = param.Number(default=0.7, bounds=(0, 1), constant=True)

    _supports_model_stream = False  # instructor doesn't work with Mistral's streaming
    _instructor_wrapper = "mistral"

    def models(self) -> set[str]:
        """Return the set of available model identifiers from Mistral."""
        from mistralai import Mistral
        return {m.id for m in Mistral(api_key=self.api_key).models.list().data}

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    def _create_base_client(self, **kwargs) -> Any:
        return self._instantiate_client(**kwargs)

    def _get_completion_method(self, stream: bool = False) -> Callable:
        return self._base_client.chat.stream_async if stream else self._base_client.chat.complete_async

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        return instructor.from_mistral(base_client, mode=mode, use_async=True)

    async def get_client(self, model_spec: str | dict, response_model: type[BaseModel] | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)
        stream = kwargs.get("stream", False)

        if response_model:
            client_callable = self._get_cached_client(response_model, model=model, mode=mode, **model_kwargs)
            return partial(client_callable.func, *client_callable.args, timeout_ms=self.timeout * 1000, **client_callable.keywords)
        else:
            if self._base_client is None:
                self._base_client = self._create_base_client(**model_kwargs)
            return partial(
                self._get_completion_method(stream),
                model=model,
                timeout_ms=self.timeout * 1000
            )

    @classmethod
    def _get_delta(cls, chunk):
        if chunk.data.choices:
            return chunk.data.choices[0].delta.content or ""
        return ""


class AzureMistralAI(MistralAI, AzureMistralAIMixin):
    """
    A LLM implementation that calls Mistral AI models on Azure.
    """

    display_name = param.String(default="Azure Mistral AI", constant=True)

    model_kwargs = param.Dict(default={
        "default": {"model": "azureai"},
    })

    select_models = param.List(default=[
        "azureai",
        "mistral-large",
        "mistral-small"
    ], constant=True, doc="Available models for selection dropdowns")

    def _create_base_client(self, **kwargs) -> Any:
        return self._instantiate_client(**kwargs)


class Anthropic(Llm, AnthropicMixin):
    """
    A LLM implementation that calls Anthropic models such as Claude.
    """

    display_name = param.String(default="Anthropic", constant=True)

    mode = param.Selector(default=Mode.ANTHROPIC_TOOLS, objects=[Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS])

    model_kwargs = param.Dict(default={
        "default": {"model": "claude-haiku-4-5"},
        "edit": {"model": "claude-sonnet-4-5"},
    })

    select_models = param.List(default=[
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-5"
    ], constant=True)

    temperature = param.Number(default=0.7, bounds=(0, 1), constant=True)

    _supports_logfire = True
    _supports_model_stream = True
    _instructor_wrapper = "anthropic"

    def models(self) -> set[str]:
        """Return the set of available model identifiers from Anthropic."""
        from anthropic import Anthropic as AnthropicClient
        response = AnthropicClient(api_key=self.api_key, timeout=5).models.list()
        # also handle model aliases (claude-sonnet-4-5-20250929) -> (claude-sonnet-4-5)
        return {m.id for m in response.data} | {m.id.rsplit("-", maxsplit=1)[0] for m in response.data}

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature, "max_tokens": 1024}

    def _create_base_client(self, **kwargs) -> Any:
        client = self._instantiate_client(**kwargs)
        if self.logfire_tags:
            self._logfire.instrument_anthropic(client)
        return client

    def _get_completion_method(self) -> Callable:
        return self._base_client.messages.create

    def _get_cached_client(
        self,
        response_model: type[BaseModel] | None = None,
        model: str | None = None,
        **kwargs
    ) -> Callable:
        mode = kwargs.pop("mode", self.mode)

        if self._base_client is None:
            self._base_client = self._create_base_client(**kwargs)

        if response_model:
            if mode not in self._instructor_clients:
                self._instructor_clients[mode] = self._create_instructor_client(self._base_client, mode)
            client = self._instructor_clients[mode]
            return partial(client.messages.create, model=model, **self._get_create_kwargs(response_model))
        else:
            return partial(self._get_completion_method(), model=model, **self._get_create_kwargs(response_model))

    def _messages_to_contents(self, messages: list[Message]) -> tuple[list[Message], str | None]:
        """Extract system messages and convert tool messages to Anthropic format.

        Anthropic requires system to be passed separately, not in messages array.
        Tool-call and tool-result messages are converted into the content-block
        format that the Anthropic API expects.
        """
        filtered: list[Message] = []
        system_text = None
        pending_tool_results: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_text = msg["content"]
                continue

            # Assistant message with tool_calls → assistant with tool_use content blocks
            if role == "assistant" and msg.get("tool_calls"):
                # Flush pending tool results first
                if pending_tool_results:
                    filtered.append({"role": "user", "content": pending_tool_results})
                    pending_tool_results = []
                content_blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    name, args, call_id = self._parse_tool_call(tc)
                    content_blocks.append({
                        "type": "tool_use",
                        "id": call_id or f"call_{name}",
                        "name": name,
                        "input": args,
                    })
                filtered.append({"role": "assistant", "content": content_blocks})
                continue

            # Tool result → collect into a single user message with tool_result blocks
            if role == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                })
                continue

            # Flush pending tool results before any other message
            if pending_tool_results:
                filtered.append(Message(role="user", content=pending_tool_results))
                pending_tool_results = []

            filtered.append(msg)

        if pending_tool_results:
            filtered.append(Message(role="user", content=pending_tool_results))

        return filtered, system_text

    def _add_system_message(self, messages: list[Message], system: str, input_kwargs: dict[str, Any]):
        if system:
            input_kwargs["system"] = system
        return messages, input_kwargs

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        if self.interceptor:
            raise NotImplementedError("Interceptors are not supported for Anthropic.")

        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        model_kwargs["mode"] = model_kwargs.pop("mode", self.mode)

        client_callable = self._get_cached_client(response_model, model=model, **model_kwargs)
        return partial(client_callable.func, *client_callable.args, timeout=self.timeout, **client_callable.keywords)

    @classmethod
    def _extract_tool_calls(cls, response: Any) -> list[Any]:
        """Extract tool calls from an Anthropic Message response.

        Normalises ToolUseBlocks into OpenAI-style dicts so the base-class
        pipeline works unchanged.
        """
        if hasattr(response, "content") and not isinstance(response.content, str):
            calls = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    calls.append({
                        "id": block.id,
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    })
            if calls:
                return calls
        return []

    @classmethod
    def _extract_stream_tool_calls(cls, chunk: Any) -> list[dict[str, Any]]:
        """Extract tool calls from Anthropic streaming events."""
        # content_block_start with a tool_use block
        if hasattr(chunk, "content_block"):
            block = chunk.content_block
            if getattr(block, "type", None) == "tool_use":
                return [{
                    "index": getattr(chunk, "index", 0),
                    "id": block.id,
                    "function": {
                        "name": block.name,
                        "arguments": "",
                    },
                }]
        # content_block_delta with partial JSON for tool input
        if hasattr(chunk, "delta"):
            partial = getattr(chunk.delta, "partial_json", None)
            if partial is not None:
                return [{
                    "index": getattr(chunk, "index", 0),
                    "function": {
                        "arguments": partial,
                    },
                }]
        return []

    async def run_client(self, model_spec: str | dict, messages: list[Message], **kwargs):
        """Override to handle Anthropic-specific message format."""
        log_debug(f"Input messages: \033[95m{len(messages)} messages\033[0m including system")

        # Convert OpenAI-format tool specs to Anthropic format
        tool_specs = kwargs.pop("tools", None)
        if tool_specs:
            anthropic_tools = []
            for spec in tool_specs:
                if isinstance(spec, dict) and spec.get("type") == "function":
                    func = spec["function"]
                    anthropic_tools.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    })
                else:
                    anthropic_tools.append(spec)
            kwargs["tools"] = anthropic_tools

        # Extract system from messages
        filtered_messages, extracted_system = self._messages_to_contents(messages)

        # Combine system from kwargs (via _add_system_message) with extracted system
        system = kwargs.get("system")
        if not system and extracted_system:
            kwargs["system"] = extracted_system

        previous_role = None
        for i, message in enumerate(filtered_messages):
            role = message["role"]
            role_char = "u" if role == "user" else "a"
            log_debug(f"Message \033[95m{i} ({role_char})\033[0m: {format_msg_content(message['content'])}")
            if previous_role == role:
                log_debug(
                    "\033[91mWARNING: Two consecutive messages from the same role; "
                    "some providers disallow this.\033[0m"
                )
            previous_role = role

        client = await self.get_client(model_spec, **kwargs)
        result = await client(messages=filtered_messages, **kwargs)
        if response_model := kwargs.get("response_model"):
            log_debug(f"Response model: \033[93m{response_model.__name__!r}\033[0m")
            if isinstance(result, ImageResponse):
                result = result.output
        log_debug(f"LLM Response: \033[95m{truncate_string(str(result), max_length=1000)}\033[0m\n---")
        return result

    @classmethod
    def _get_content(cls, response: Any) -> Any:
        """Extract text content from an Anthropic Message response."""
        content = getattr(response, "content", None)
        if isinstance(content, (str, BaseModel)):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                # SDK block objects (e.g. TextBlock)
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text = getattr(block, "text", None)
                    if text:
                        texts.append(text)
                    continue
                # Dict blocks (defensive for mocked/test payloads)
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if text:
                        texts.append(text)
            if texts:
                return "".join(texts)
        return response

    @classmethod
    def _get_delta(cls, chunk: Any) -> str:
        if hasattr(chunk, 'delta') and hasattr(chunk.delta, "text"):
            return chunk.delta.text
        return ""


class AnthropicBedrock(BedrockMixin, Anthropic):  # Keep it before Anthropic so API key is correct

    display_name = param.String(default="Anthropic on AWS Bedrock", constant=True)

    model_kwargs = param.Dict(default={
        "default": {"model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
        "ui": {"model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
        "edit": {"model": "us.anthropic.claude-opus-4-5-20251101-v1:0"},
    })

    def _create_base_client(self, **kwargs) -> Any:
        from anthropic.lib.bedrock import AsyncAnthropicBedrock
        return AsyncAnthropicBedrock(
            aws_access_key=self.aws_access_key_id,
            aws_secret_key=self.api_key,
            aws_session_token=self.aws_session_token,
            aws_region=self.region_name,
            **kwargs
        )


class Bedrock(Llm, BedrockMixin):
    """
    A LLM implementation that calls AWS Bedrock models using the Converse API.

    Uses boto3 bedrock-runtime client with the Converse API for a unified
    interface across different foundation models. Supports standard AWS
    credential resolution including environment variables, ~/.aws/credentials,
    and AWS SSO.
    """

    display_name = param.String(default="AWS Bedrock", constant=True)

    mode = param.Selector(default=Mode.BEDROCK_TOOLS, objects=[Mode.BEDROCK_JSON, Mode.BEDROCK_TOOLS])

    model_kwargs = param.Dict(default={
        "default": {"model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
        "ui": {"model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
        "edit": {"model": "us.anthropic.claude-opus-4-5-20251101-v1:0"},
    })

    select_models = param.List(default=[
        # Inference profiles (cross-region) - recommended for Claude 4+
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "us.anthropic.claude-opus-4-1-20250805-v1:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        # Claude 3.5 with inference profile
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        # Direct model IDs (if inference profiles don't work)
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
    ], constant=True, doc="Available Claude models on Bedrock")

    temperature = param.Number(default=0.7, bounds=(0, 1), constant=True)

    _instructor_wrapper = "bedrock"
    _supports_stream = True
    _supports_model_stream = True

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature, "maxTokens": 4096}

    def _create_base_client(self, **kwargs) -> Any:
        """Create boto3 bedrock-runtime client for inference."""
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Please install boto3 to use AWS Bedrock. "
                "You can install it with `pip install boto3`."
            ) from exc

        credentials = {}
        if self.aws_access_key_id:
            credentials["aws_access_key_id"] = self.aws_access_key_id
        if self.api_key:
            credentials["aws_secret_access_key"] = self.api_key
        if self.aws_session_token:
            credentials["aws_session_token"] = self.aws_session_token

        return boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
            **credentials,
            **kwargs
        )

    def _messages_to_contents(self, messages: list[Message]) -> tuple[list[dict], str | None]:
        """Convert messages to Bedrock Converse API format.

        Extracts system messages and returns them separately since Bedrock
        requires them to be passed via the 'system' parameter.
        """
        bedrock_messages = []
        system_text = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_text = content
                continue

            if isinstance(content, str):
                bedrock_messages.append({"role": role, "content": [{"text": content}]})
            elif isinstance(content, list):
                bedrock_content = []
                for item in content:
                    if isinstance(item, str):
                        bedrock_content.append({"text": item})
                    elif isinstance(item, dict):
                        bedrock_content.append(item)
                bedrock_messages.append({"role": role, "content": bedrock_content})

        return bedrock_messages, system_text

    def _add_system_message(self, messages: list[Message], system: str, input_kwargs: dict[str, Any]):
        if system:
            input_kwargs["system"] = [{"text": system}]
        return messages, input_kwargs

    async def _bedrock_invoke(self, messages, model, stream=False, response_model=None, **kwargs):
        """Async wrapper for Bedrock converse API."""
        bedrock_messages, extracted_system = self._messages_to_contents(messages)

        # Combine system from kwargs (via _add_system_message) with extracted system
        system = kwargs.get("system")
        if not system and extracted_system:
            system = [{"text": extracted_system}]

        if response_model:
            mode = kwargs.pop("mode", self.mode)
            if mode not in self._instructor_clients:
                self._instructor_clients[mode] = self._create_instructor_client(self._base_client, mode)
            client = self._instructor_clients[mode]
            create_kwargs = {
                "model": model,
                "messages": messages,
                "response_model": response_model,
                **self._get_create_kwargs(response_model),
            }
            if system:
                create_kwargs["system"] = system
            return await asyncio.to_thread(client.messages.create, **create_kwargs)

        call_kwargs = {
            "modelId": model,
            "messages": bedrock_messages,
            "inferenceConfig": self._client_kwargs,
        }
        if system:
            call_kwargs["system"] = system

        if stream:
            resp = await asyncio.to_thread(self._base_client.converse_stream, **call_kwargs)
            return self._wrap_stream(resp)
        else:
            resp = await asyncio.to_thread(self._base_client.converse, **call_kwargs)
            return resp["output"]["message"]["content"][0]["text"]

    async def _wrap_stream(self, response):
        """Wrap synchronous Bedrock stream as async generator."""
        for chunk in response["stream"]:
            yield chunk

    @classmethod
    def _get_delta(cls, chunk) -> str:
        if "contentBlockDelta" in chunk:
            return chunk["contentBlockDelta"]["delta"].get("text", "")
        return ""

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        log_debug(f"LLM Model: \033[96m{model!r}\033[0m")

        if self._base_client is None:
            self._base_client = self._create_base_client()

        return partial(self._bedrock_invoke, model=model)


class Google(Llm, GenAIMixin):
    """
    A LLM implementation that calls Google's Gemini models.
    """

    display_name = param.String(default="Google AI", constant=True)

    mode = param.Selector(default=Mode.GENAI_TOOLS, objects=[Mode.GENAI_TOOLS, Mode.GENAI_STRUCTURED_OUTPUTS])

    model_kwargs = param.Dict(default={
        "default": {"model": "gemini-3-flash-preview"},
    })

    select_models = param.List(default=[
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ], constant=True)

    temperature = param.Number(default=1, bounds=(0, 1), constant=True)

    _supports_logfire = True
    _supports_model_stream = True
    _instructor_wrapper = "genai"

    def models(self) -> set[str]:
        """Return the set of available model identifiers from Google AI."""
        from google import genai
        available = set()
        for m in genai.Client(api_key=self.api_key).models.list():
            available.add(m.name)
            if m.name.startswith("models/"):
                available.add(m.name[7:])  # Strip "models/" prefix
        return available

    @property
    def _client_kwargs(self):
        return {}

    def _create_base_client(self, **kwargs) -> Any:
        if self.logfire_tags:
            self._logfire.instrument_google_genai()
        return self._instantiate_client()

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        return instructor.from_genai(base_client, mode=mode, use_async=True)

    @classmethod
    def _get_content(cls, response: Any) -> str | BaseModel:
        """Extract content from a non-streaming Google GenAI response."""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    return candidate.content.parts[0].text or ""
        elif isinstance(response, (str, BaseModel)):
            return response
        return str(response)

    @classmethod
    def _messages_to_contents(cls, messages: list[Message]) -> tuple[list[dict[str, Any]], str | None]:
        """
        Transform messages into contents format expected by Google GenAI API.

        Extracts system messages and returns them separately since Google
        requires them via the system_instruction parameter.

        Parameters
        ----------
        messages : list[Message]
            List of messages with 'role', 'content', and optional 'name' fields.

        Returns
        -------
        tuple[list[dict[str, Any]], str | None]
            Tuple of (contents list, system_instruction)
        """
        contents = []
        system_instruction = None

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_instruction = content
                continue

            # Assistant message containing tool calls → model with function_call parts
            if role == "assistant" and message.get("tool_calls"):
                parts = []
                for tc in message["tool_calls"]:
                    name, args, _ = cls._parse_tool_call(tc)
                    part: dict[str, Any] = {"function_call": {"name": name, "args": args}}
                    # Preserve thought_signature required by Gemini 3 models
                    sig = tc.get("thought_signature") if isinstance(tc, dict) else None
                    if sig is not None:
                        part["thought_signature"] = sig
                    parts.append(part)
                contents.append({"role": "model", "parts": parts})
                continue

            # Tool result message → user with function_response part
            if role == "tool":
                name = message.get("name", "")
                try:
                    response_payload = json.loads(content) if isinstance(content, str) else content
                except (json.JSONDecodeError, TypeError):
                    response_payload = {"result": content}
                if not isinstance(response_payload, dict):
                    response_payload = {"result": response_payload}
                contents.append({
                    "role": "user",
                    "parts": [{"function_response": {"name": name, "response": response_payload}}],
                })
                continue

            if role != "user":
                role = "model"

            if isinstance(content, Image):
                contents.append({
                    "role": role,
                    "parts": [content.to_genai()]
                })
            else:
                contents.append({
                    "role": role,
                    "parts": [{"text": content}]
                })

        return contents, system_instruction

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        log_debug(f"LLM Model: \033[96m{model!r}\033[0m")
        mode = model_kwargs.pop("mode", self.mode)

        if response_model:
            return self._get_cached_client(response_model, model=model, mode=mode, **model_kwargs)

        # Ensure base client exists
        if self._base_client is None:
            self._base_client = self._create_base_client(**model_kwargs)

        if kwargs.get("stream"):
            return partial(self._base_client.aio.models.generate_content_stream, model=model)
        else:
            return partial(self._base_client.aio.models.generate_content, model=model)

    @classmethod
    def _extract_tool_calls(cls, response: Any) -> list[Any]:
        """Extract tool calls from a Google GenAI response.

        Normalises them into OpenAI-style dicts so the base-class
        ``_parse_tool_call``, ``_tool_calls_message``, and
        ``_run_tool_calls`` work without further overrides.

        The ``thought_signature`` returned by Gemini 3 models is preserved
        so it can be sent back in the follow-up request (required by the API).
        """
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                parts = candidate.content.parts or []
                calls = []
                for i, part in enumerate(parts):
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        args = dict(fc.args) if fc.args else {}
                        call: dict[str, Any] = {
                            "id": f"call_{fc.name}_{i}",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(args),
                            },
                        }
                        # Gemini 3 requires thought_signature to be echoed back
                        sig = getattr(part, "thought_signature", None)
                        if sig is not None:
                            call["thought_signature"] = sig
                        calls.append(call)
                if calls:
                    return calls
        return []

    @classmethod
    def _extract_stream_tool_calls(cls, chunk: Any) -> list[dict[str, Any]]:
        """Extract tool calls from a Google GenAI streaming chunk."""
        # Google streaming chunks share the same response shape
        return cls._extract_tool_calls(chunk)

    @classmethod
    def _translate_tool_specs(cls, tool_specs: list) -> Any:
        # Convert OpenAI-format tool specs to Google function declarations
        if not tool_specs:
            return

        from google.genai.types import FunctionDeclaration, Tool
        declarations = []
        for spec in tool_specs:
            if not isinstance(spec, dict) or spec.get("type") != "function":
                continue
            func = spec["function"]
            parameters = dict(func.get("parameters", {}))
            # Remove keys that are not valid in Google's schema
            parameters.pop("title", None)
            declarations.append(FunctionDeclaration(
                name=func["name"],
                description=func.get("description", ""),
                parameters=parameters,
            ))
        return Tool(function_declarations=declarations) if declarations else None

    @classmethod
    def _get_delta(cls, chunk: Any) -> str:
        """Extract delta content from streaming response, skipping function_call parts."""
        if hasattr(chunk, "candidates") and chunk.candidates:
            candidate = chunk.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                parts = candidate.content.parts or []
                texts = []
                for p in parts:
                    if getattr(p, "function_call", None) is not None:
                        continue
                    text = getattr(p, "text", None)
                    if text:
                        texts.append(text)
                return "".join(texts)
        if hasattr(chunk, "text"):
            return chunk.text or ""
        return ""

    async def run_client(self, model_spec: str | dict, messages: list[Message], **kwargs):
        """Override to handle Gemini-specific message format conversion."""
        try:
            from google.genai.types import (
                GenerateContentConfig, HttpOptions, ThinkingConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "Please install the `google-generativeai` package to use Google AI models. "
                "You can install it with `pip install -U google-genai`."
            ) from exc

        response_model = kwargs.get("response_model")
        http_options = HttpOptions(timeout=int(self.timeout * 1000))  # timeout is in milliseconds
        thinking_config = ThinkingConfig(thinking_budget=0, include_thoughts=False)

        tools = self._translate_tool_specs(kwargs.pop("tools", []))
        client = await self.get_client(model_spec, **kwargs)
        contents, system_instruction = self._messages_to_contents(messages)
        config = GenerateContentConfig(
            http_options=http_options,
            temperature=self.temperature,
            thinking_config=thinking_config,
            system_instruction=system_instruction,
            tools=tools,
        )

        if response_model:
            result = await client(messages=messages, config=config, **kwargs)
            return result

        kwargs.pop("stream", None)
        result = await client(contents=contents, config=config, **kwargs)
        return result


class AINavigator(OpenAI):
    """
    A LLM implementation that calls the [Anaconda AI Navigator](https://www.anaconda.com/products/ai-navigator) API.
    """

    display_name = param.String(default="AI Navigator", constant=True)

    endpoint = param.String(default=None, doc="""
            The API endpoint; should include the full address, including the port.""")

    def __init__(self, **params):
        if "endpoint" not in params:
            params["endpoint"] = os.environ.get("AINAVIGATOR_BASE_URL", "http://localhost:8080/v1")
        super().__init__(**params)

    mode = param.Selector(default=Mode.JSON_SCHEMA)

    model_kwargs = param.Dict(default={
        "default": {"model": "server-model"},
    })

    select_models = param.List(default=["server-model"], constant=True, doc="Available models for selection dropdowns")


class AICatalyst(OpenAI):
    """
    A LLM implementation that calls the [Anaconda AI Catalyst](https://www.anaconda.com/platform/ai-catalyst) API.
    """

    api_key = param.String(default=None, doc="The AI Catalyst API key.")

    display_name = param.String(default="AI Catalyst", constant=True)

    endpoint = param.String(default=None, doc="""
            The API endpoint; should include the full address, including the port.""")

    mode = param.Selector(default=Mode.JSON_SCHEMA)

    def __init__(self, **params):
        if "api_key" not in params:
            params["api_key"] = os.environ.get("AI_CATALYST_API_KEY")
        if "endpoint" not in params:
            params["endpoint"] = os.environ.get("AI_CATALYST_BASE_URL")
        super().__init__(**params)

    model_kwargs = param.Dict(default={
        "default": {"model": "ai_catalyst"},
    })

    select_models = param.List(default=["ai_catalyst"], constant=True, doc="Available models for selection dropdowns")


class Ollama(OpenAI):
    """
    An LLM implementation using the Ollama cloud.
    """

    api_key = param.String(default="ollama", doc="The Ollama API key; required but unused.")

    display_name = param.String(default="Ollama", constant=True)

    endpoint = param.String(default="http://localhost:11434/v1", doc="The Ollama API endpoint.")

    mode = param.Selector(default=Mode.JSON)

    model_kwargs = param.Dict(default={
        "default": {"model": "qwen3:32b"},
    })

    select_models = param.List(default=[
        "qwen3:32b",
        "qwen3-coder:32b",
        "nemotron-3-nano:30b",
        "mistral-small3.2:24b",
    ], constant=True)

    temperature = param.Number(default=0.25, bounds=(0, None), constant=True)

    def models(self, endpoint: str | None = None) -> set[str]:
        """Return the set of available model identifiers from Ollama."""
        base_url = (endpoint or self.endpoint).rstrip('/').removesuffix('/v1')
        tags_response = requests.get(f"{base_url}/api/tags", timeout=5)
        if tags_response.status_code != 200:
            return set()
        return {m.get("name", "") for m in tags_response.json().get("models", [])}


class Groq(OpenAI):
    """
    An LLM implementation using the Groq cloud API.

    Groq provides an OpenAI-compatible endpoint, so this is a thin
    subclass of :class:`OpenAI` with Groq-specific defaults.

    Set the ``GROQ_API_KEY`` environment variable or pass ``api_key``
    directly.  The provider is auto-detected when ``GROQ_API_KEY`` is
    present, or can be selected explicitly with ``--provider groq``.
    """

    api_key_env_var: str = PROVIDER_ENV_VARS['groq']

    display_name = param.String(
        default="Groq", constant=True, doc="Display name for UI"
    )

    endpoint = param.String(
        default="https://api.groq.com/openai/v1",
        doc="The Groq API endpoint.",
    )

    model_kwargs = param.Dict(default={
        "default": {"model": "llama-3.3-70b-versatile"},
    })

    select_models = param.List(default=[
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "gemma2-9b-it",
        "mistral-saba-24b",
        "qwen-qwq-32b",
    ], constant=True, doc="Available Groq models for selection dropdowns.")


class MessageModel(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None


class Choice(BaseModel):

    delta: MessageModel | None

    message: MessageModel | None

    finish_reason: str | None


class Response(BaseModel):

    choices: list[Choice]


class WebLLM(Llm):
    """WebLLM implementation using panel_web_llm. Uses patch(create=...) like LlamaCpp."""

    display_name = param.String(default="WebLLM", constant=True)

    mode = param.Parameter(default=Mode.JSON_SCHEMA)

    model_kwargs = param.Dict({'default': {'model_slug': 'Qwen2.5-7B-Instruct-q4f16_1-MLC'}})

    select_models = param.List(default=[
        "Llama-3.2-3B-Instruct-q4f16_1-MLC",
        "Phi-3.5-mini-instruct-q4f16_1-MLC",
        "Qwen2.5-7B-Instruct-q4f16_1-MLC"
    ], constant=True)

    temperature = param.Number(default=0.4, bounds=(0, None))

    # WebLLM doesn't use from_* wrapper - uses patch(create=...)
    _instructor_wrapper = None

    def __init__(self, **params):
        from panel_web_llm import WebLLM as pnWebLLM
        self._llm = pnWebLLM()
        super().__init__(**params)

    def _create_base_client(self, **kwargs) -> Any:
        return self._llm

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        return patch(create=self._create_completion, mode=mode)

    async def _create_completion(self, messages, **kwargs):
        if kwargs.get('stream', False):
            async def generator():
                async for chunk in self._llm.create_completion(messages, **kwargs):
                    yield Response(choices=[
                        Choice(
                            delta=MessageModel(
                                content=chunk['delta']['content'],
                                role=chunk['delta']['role'],
                                name=None
                            ) if chunk['delta'] else None,
                            message=None,
                            finish_reason=chunk['finish_reason']
                        )
                    ])
        else:
            content, reason, role = "", "", ""
            async for chunk in self._llm.create_completion(messages, **kwargs):
                content = chunk['message']['content']
                role = chunk['message']['role']
                reason = chunk['finish_reason']
            msg = MessageModel(content=content, name=None, role=role)
            return Response(choices=[Choice(message=msg, delta=None, finish_reason=reason)])
        return generator()

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        mode = model_kwargs.pop("mode", self.mode)

        if not self._llm.loaded:
            self._llm.param.update(**{k: v for k, v in model_kwargs.items() if k in self._llm.param})
            self._llm.param.trigger('load_model')

        # Cache instructor client by mode
        if mode not in self._instructor_clients:
            self._instructor_clients[mode] = self._create_instructor_client(self._llm, mode)
        return self._instructor_clients[mode]

    def status(self):
        return pn.Row(self._status, self._llm)

    async def initialize(self, log_level: str):
        await asyncio.sleep(2)
        progress = self._llm.param.load_status.rx().get("progress", 0)*100
        self._status.name = pn.rx('Loading LLM {:.1f}%').format(progress)
        try:
            await self.invoke(
                messages=[{'role': 'user', 'content': 'Ready? "Y" or "N"'}],
                model_spec="ui",
            )
        except Exception as e:
            self._status.param.update(
                status="failed",
                name="LLM Not Connected",
                description='❌ '+(format_exception(e, limit=3) if log_level == 'DEBUG' else "Failed to connect to LLM"),
            )
            raise e
        else:
            self._status.param.update(status="success", name='LLM Ready')


class LiteLLM(Llm):
    """
    A LLM implementation using LiteLLM that supports multiple providers
    through a unified interface.

    LiteLLM allows you to call 100+ LLMs using the same OpenAI-compatible
    input/output format, including providers like OpenAI, Anthropic, Cohere,
    Hugging Face, Azure, Vertex AI, and more.
    """

    display_name = param.String(default="LiteLLM", constant=True)

    enable_caching = param.Boolean(default=False, doc="""
        Enable LiteLLM's built-in caching for repeated queries.""")

    fallback_models = param.List(default=[], doc="""
        List of fallback models to try if the primary model fails.
        Example: ["gpt-4o-mini", "claude-3-haiku", "gemini/gemini-1.5-flash"]""")

    litellm_params = param.Dict(default={}, doc="""
        Additional parameters to pass to litellm.acompletion().
        Examples: custom_llm_provider, api_base, api_version, etc.""")

    mode = param.Selector(default=Mode.TOOLS, objects=BASE_MODES)

    model_kwargs = param.Dict(default={
        "default": {"model": "gpt-5.4-mini"},
        "edit": {"model": "anthropic/claude-sonnet-4-5"},
        "sql": {"model": "gpt-5.4-mini"},
    }, doc="""
        Model configurations by type. LiteLLM supports model strings like:
        - OpenAI: "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5-mini"
        - Anthropic: "anthropic/claude-sonnet-4-5", "anthropic/claude-haiku-4-5"
        - Google: "gemini/gemini-2.0-flash", "gemini/gemini-2.5-flash"
        - Mistral: "mistral/mistral-medium-latest", "mistral/mistral-small-latest"
        - And many more with format: "provider/model" or just "model" for defaults
    """)

    router_settings = param.Dict(default={}, doc="""
        Settings for LiteLLM Router for load balancing across multiple models.
        Example: {"routing_strategy": "least-busy", "num_retries": 3}""")

    select_models = param.List(default=[
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5-mini",
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-opus-4-1",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.5-flash",
        "mistral/mistral-medium-latest",
        "mistral/mistral-small-latest",
        "mistral/codestral-latest"
    ], constant=True)

    temperature = param.Number(default=0.7, bounds=(0, 2), constant=True)

    _supports_logfire = True
    _supports_stream = True
    _supports_model_stream = True
    _instructor_wrapper = "litellm"

    def __init__(self, **params):
        super().__init__(**params)
        self._router = None  # Lazy init
        if self.enable_caching:
            import litellm

            from litellm import Cache
            litellm.cache = Cache()
        if self.logfire_tags:
            self._logfire.instrument_litellm()

    def _get_router(self):
        """Get or create cached LiteLLM Router."""
        if self._router is None:
            from litellm import Router
            model_list = [
                {'model_name': key, 'litellm_params': config}
                for key, config in self.model_kwargs.items()
                if isinstance(config, dict) and 'model' in config
            ]
            router_kwargs = dict(self.router_settings)
            if self.fallback_models:
                router_kwargs['fallbacks'] = self.fallback_models
            self._router = Router(model_list=model_list, timeout=self.timeout, **router_kwargs)
        return self._router

    @property
    def _client_kwargs(self):
        """Base kwargs for all LiteLLM calls."""
        kwargs = {"temperature": self.temperature}
        kwargs.update(self.litellm_params)
        return kwargs

    def _create_base_client(self, **kwargs) -> Any:
        return self._get_router()

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        return instructor.from_litellm(base_client.acompletion, mode=mode)

    def _get_completion_method(self) -> Callable:
        return self._get_router().acompletion

    async def get_client(self, model_spec: str | dict, response_model: type[BaseModel] | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)

        if response_model:
            return self._get_cached_client(response_model, model=model, mode=mode, **model_kwargs)
        else:
            router = self._get_router()
            return partial(router.acompletion, model=model, **self._client_kwargs, **model_kwargs)

    @classmethod
    def _get_delta(cls, chunk) -> str:
        """Extract delta content from streaming chunks."""
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                return choice.delta.content or ""
        return ""
