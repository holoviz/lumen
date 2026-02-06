from __future__ import annotations

import asyncio
import base64
import json
import os

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Literal, TypedDict,
)

import instructor
import panel as pn
import param

from instructor import Mode, patch
from instructor.dsl.partial import Partial
from instructor.processing.multimodal import Image
from pydantic import BaseModel

from .interceptor import Interceptor
from .services import (
    AzureOpenAIMixin, BedrockMixin, LlamaCppMixin, OpenAIMixin,
)
from .utils import format_exception, log_debug, truncate_string

if TYPE_CHECKING:
    from .tools import FunctionTool
    from .tools.mcp import MCPTool

class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None
    tool_call_id: str | None
    tool_calls: list[dict[str, Any]] | None


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
    'anthropic_bedrock': 'AnthropicBedrock',
    'bedrock': 'Bedrock',
    'mistral': 'MistralAI',
    'azure-openai': 'AzureOpenAI',
    'azure-mistral': 'AzureMistralAI',
    "ai-navigator": "AINavigator",
    'ollama': 'Ollama',
    'llama-cpp': 'LlamaCpp',
    'litellm': 'LiteLLM',
}

# Environment variable mapping for providers that require API keys
# Providers not in this list (like ollama, llama-cpp) may work without env vars
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",  # AWS credentials
    "anthropic_bedrock": "AWS_ACCESS_KEY_ID",  # AWS credentials
    "mistral": "MISTRAL_API_KEY",
    "azure-mistral": "AZUREAI_ENDPOINT_KEY",
    "azure-openai": "AZUREAI_ENDPOINT_KEY",
    "google": "GEMINI_API_KEY",
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

    logfire_tags = param.List(default=None, doc="""
        Whether to log LLM calls and responses to logfire.
        If a list of tags is provided, those tags will be used for logging.
        Suppresses streaming responses if enabled since
        logfire does not track token usage on stream.""")

    _ready = param.Boolean(default=False, doc="""
        Whether the LLM has been initialized and is ready to use.""")

    # Whether the LLM supports logging to logfire
    _supports_logfire = False

    # Whether the LLM supports streaming of any kind
    _supports_stream = True

    # Whether the LLM supports streaming of Pydantic model output
    _supports_model_stream = True

    # Used for `instructor.from_*` wrapping; subclasses may override
    _instructor_wrapper: str = "openai"

    __abstract = True

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

    @abstractmethod
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
        response_model: BaseModel | None = None,
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
            messages = [{"role": "system", "content": system}] + messages
        return messages, input_kwargs

    def _serialize_image_pane(self, image: pn.pane.image.ImageBase | Image) -> Image:
        if isinstance(image, Image):
            return image

        image_object = image.object
        if isinstance(image_object, bytes):
            # convert bytes to base64 string
            base64_str = base64.b64encode(image_object).decode('utf-8')
            image = Image.from_raw_base64(base64_str)
        elif isinstance(image_object, (Path, str)) and Path(image_object).is_file():
            image = Image.from_path(image_object)
        elif isinstance(image_object, str):
            image = Image.from_url(image_object)
        return image

    def _check_for_image(self, messages: list[Message]) -> tuple[list[Message], bool]:
        contains_image = False
        for i, message in enumerate(messages):
            content = message.get("content")
            if isinstance(content, (Image, pn.pane.image.ImageBase)):
                messages[i]["content"] = self._serialize_image_pane(content)
                contains_image = True

            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, (Image, pn.pane.image.ImageBase)):
                        messages[i]["content"] = self._serialize_image_pane(item)
                        contains_image = True
        return messages, contains_image

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
        response_model: BaseModel | None = None,
        allow_partial: bool = False,
        model_spec: str | dict = "default",
        tools: list[dict[str, Any] | FunctionTool | MCPTool] | None = None,
        **input_kwargs,
    ) -> BaseModel:
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
        model: Literal['default' | 'reasoning' | 'sql']
            The model as listed in the model_kwargs parameter
            to invoke to answer the query.

        Returns
        -------
        The completed response_model.
        """
        system = system.strip().replace("\n\n", "\n")
        messages, input_kwargs = self._add_system_message(messages, system, input_kwargs)

        kwargs = dict(self._client_kwargs)
        kwargs.update(input_kwargs)
        tool_specs, tool_instances, tool_contexts = self._normalize_tools(tools)
        if tool_specs is not None:
            kwargs["tools"] = tool_specs

        messages, contains_image = self._check_for_image(messages)
        if contains_image:
            # Currently instructor does not support streaming with multimodal
            # https://github.com/567-labs/instructor/issues/1872
            kwargs["stream"] = False

        if response_model is not None:
            if allow_partial and issubclass(response_model, BaseModel):
                response_model = Partial[response_model]
            kwargs["response_model"] = response_model
        # check if any of the messages contain images
        elif response_model is None and contains_image:
            kwargs["response_model"] = ImageResponse

        output = await self.run_client(model_spec, messages, **kwargs)
        if tool_instances and response_model is None:
            tool_calls = self._extract_tool_calls(output)
            if tool_calls:
                tool_calls_message = self._tool_calls_message(tool_calls)
                tool_messages = await self._run_tool_calls(tool_instances, tool_calls, tool_contexts, messages)
                if tool_messages:
                    output = await self.run_client(
                        model_spec,
                        messages + [tool_calls_message] + tool_messages,
                        **kwargs,
                    )
        if output is None or output == "":
            raise ValueError("LLM failed to return valid output.")
        return output

    @classmethod
    def _get_delta(cls, chunk) -> str:
        if chunk.choices:
            return chunk.choices[0].delta.content or ""
        return ""

    @classmethod
    def _get_content(cls, response) -> str:
        """Extract content from a non-streaming response. Override for non-OpenAI APIs."""
        if hasattr(response, "choices"):
            return response.choices[0].message.content
        return str(response)

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
                "function": {
                    "name": entry.get("name"),
                    "arguments": entry.get("arguments", ""),
                },
            })
        return tool_calls

    @classmethod
    def _tool_calls_message(cls, tool_calls: list[dict[str, Any]]) -> Message:
        return {
            "role": "assistant",
            "content": "",
            "name": None,
            "tool_call_id": None,
            "tool_calls": tool_calls,
        }

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
        results: list[Message] = []
        for call in tool_calls:
            name, arguments, call_id = self._parse_tool_call(call)
            if not name or name not in tool_instances:
                continue
            tool = tool_instances[name]
            context = tool_contexts.get(name)
            if context:
                outputs, _ = await tool.respond(messages, context, tool_args=arguments)
                if len(outputs) == 1:
                    result = outputs[0]
                else:
                    result = outputs
            elif hasattr(tool, "execute"):
                result = await tool.execute(**arguments)
            else:
                if getattr(tool, "requires", None):
                    for requirement in tool.requires:
                        if requirement not in arguments and requirement in tool_contexts.get(name, {}):
                            arguments[requirement] = tool_contexts[name][requirement]
                if asyncio.iscoroutinefunction(tool.function):
                    result = await tool.function(**arguments)
                else:
                    result = tool.function(**arguments)
            results.append({
                "role": "tool",
                "content": self._format_tool_result(result),
                "name": name,
                "tool_call_id": call_id,
            })
        return results

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
        response_model: BaseModel | None = None,
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
        response_model: BaseModel | None
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
        tool_specs, tool_instances, tool_contexts = self._normalize_tools(tools)
        if self.logfire_tags is not None:
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
            yield await self.invoke(
                messages,
                system=system,
                response_model=response_model,
                model_spec=model_spec,
                tools=tool_specs,
                **kwargs,
            )
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
        try:
            async for chunk in chunks:
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
            tool_calls_message = self._tool_calls_message(tool_calls)
            tool_messages = await self._run_tool_calls(tool_instances, tool_calls, tool_contexts, messages)
            if tool_messages:
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
            if role == "user":
                log_debug(f"Message \033[95m{i} (u)\033[0m: {message['content']}")
            else:
                log_debug(f"Message \033[95m{i} (a)\033[0m: {message['content']}")
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
        log_debug(f"LLM Response: \033[95m{truncate_string(str(result), max_length=1000)}\033[0m\n---")
        return result


class LlamaCpp(Llm, LlamaCppMixin):
    """
    A LLM implementation using Llama.cpp Python wrapper together with huggingface_hub to fetch the models.
    """

    chat_format = param.String(constant=True)

    display_name = param.String(default="Llama.cpp", constant=True, doc="Display name for UI")

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
    ], constant=True, doc="Available models for selection dropdowns")

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
        from llama_cpp import Llama as LlamaCppModel
        return LlamaCppModel(**kwargs)

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

    display_name = param.String(default="OpenAI", constant=True, doc="Display name for UI")

    mode = param.Selector(default=Mode.TOOLS)

    model_kwargs = param.Dict(default={
        "default": {"model": "gpt-4.1-mini"},  # Use standard models, not reasoning models (gpt-5, o4-mini)
        "ui": {"model": "gpt-4.1-nano"},
    })

    select_models = param.List(default=[
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano"
    ], constant=True, doc="""Available models for selection dropdowns.
        Warning: Reasoning models (gpt-5, o4-mini) are much slower and not suitable for dialog interfaces.""")

    temperature = param.Number(default=0.25, bounds=(0, None), constant=True)

    timeout = param.Number(default=120, bounds=(1, None), constant=True, doc="""
        The timeout in seconds for OpenAI API calls.""")

    _supports_logfire = True

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

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        if self.interceptor:
            self.interceptor.patch_client(base_client, mode="store_inputs")
        wrapped = instructor.from_openai(base_client, mode=mode)
        if self.interceptor:
            self.interceptor.patch_client_response(wrapped)
        return wrapped

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        log_debug(f"LLM Model: \033[96m{model!r}\033[0m")
        model_kwargs["mode"] = model_kwargs.pop("mode", self.mode)

        client_callable = self._get_cached_client(response_model, model=model, **model_kwargs)
        # Add timeout to the partial
        return partial(client_callable.func, *client_callable.args, timeout=self.timeout, **client_callable.keywords)


class AzureOpenAI(Llm, AzureOpenAIMixin):
    """
    A LLM implementation that uses the Azure OpenAI integration.
    Inherits from AzureOpenAIMixin which extends OpenAIMixin, so it has access to all OpenAI functionality
    plus Azure-specific configuration.
    """

    display_name = param.String(default="Azure OpenAI", constant=True, doc="Display name for UI")

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
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=1, bounds=(0, None), constant=True)

    timeout = param.Number(default=120, bounds=(1, None), constant=True, doc="""
        The timeout in seconds for Azure OpenAI API calls.""")

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

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        model_kwargs["mode"] = model_kwargs.pop("mode", self.mode)

        client_callable = self._get_cached_client(response_model, model=model, **model_kwargs)
        return partial(client_callable.func, *client_callable.args, timeout=self.timeout, **client_callable.keywords)


class MistralAI(Llm):
    """
    A LLM implementation that calls Mistral AI.
    """

    api_key = param.String(default=os.getenv("MISTRAL_API_KEY"), doc="The Mistral AI API key.")

    display_name = param.String(default="Mistral AI", constant=True, doc="Display name for UI")

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
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.7, bounds=(0, 1), constant=True)

    timeout = param.Number(default=120, bounds=(1, None), constant=True, doc="""
        The timeout in seconds for Mistral AI API calls.""")

    _supports_model_stream = False  # instructor doesn't work with Mistral's streaming
    _instructor_wrapper = "mistral"

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    def _create_base_client(self, **kwargs) -> Any:
        from mistralai import Mistral
        return Mistral(api_key=self.api_key, **kwargs)

    def _get_completion_method(self, stream: bool = False) -> Callable:
        return self._base_client.chat.stream_async if stream else self._base_client.chat.complete_async

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        return instructor.from_mistral(base_client, mode=mode, use_async=True)

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
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


class AzureMistralAI(MistralAI):
    """
    A LLM implementation that calls Mistral AI models on Azure.
    """

    api_key = param.String(default=os.getenv("AZUREAI_ENDPOINT_KEY"), doc="The Azure API key")

    display_name = param.String(default="Azure Mistral AI", constant=True, doc="Display name for UI")

    endpoint = param.String(default=os.getenv('AZUREAI_ENDPOINT_URL'), doc="The Azure API endpoint to invoke.")

    model_kwargs = param.Dict(default={
        "default": {"model": "azureai"},
    })

    select_models = param.List(default=[
        "azureai",
        "mistral-large",
        "mistral-small"
    ], constant=True, doc="Available models for selection dropdowns")

    timeout = param.Number(default=120, bounds=(1, None), constant=True, doc="""
        The timeout in seconds for Mistral AI API calls.""")

    def _create_base_client(self, **kwargs) -> Any:
        from mistralai_azure import MistralAzure
        return MistralAzure(api_key=self.api_key, azure_endpoint=self.endpoint, **kwargs)


class Anthropic(Llm):
    """
    A LLM implementation that calls Anthropic models such as Claude.
    """

    api_key = param.String(default=os.getenv("ANTHROPIC_API_KEY"), doc="The Anthropic API key.")

    display_name = param.String(default="Anthropic", constant=True, doc="Display name for UI")

    mode = param.Selector(default=Mode.ANTHROPIC_TOOLS, objects=[Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS])

    model_kwargs = param.Dict(default={
        "default": {"model": "claude-haiku-4-5"},
        "edit": {"model": "claude-sonnet-4-5"},
    })

    select_models = param.List(default=[
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-5"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.7, bounds=(0, 1), constant=True)

    timeout = param.Number(default=120, bounds=(1, None), constant=True, doc="""
        The timeout in seconds for Anthropic API calls.""")

    _supports_logfire = True
    _supports_model_stream = True
    _instructor_wrapper = "anthropic"

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature, "max_tokens": 1024}

    def _create_base_client(self, **kwargs) -> Any:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=self.api_key, **kwargs)
        if self.logfire_tags:
            self._logfire.instrument_anthropic(client)
        return client

    def _get_completion_method(self) -> Callable:
        return self._base_client.messages.create

    def _get_cached_client(
        self,
        response_model: BaseModel | None = None,
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
        """Extract system messages from the messages list.

        Anthropic requires system to be passed separately, not in messages array.
        """
        filtered = []
        system_text = None
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                filtered.append(msg)
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

    async def run_client(self, model_spec: str | dict, messages: list[Message], **kwargs):
        """Override to handle Anthropic-specific message format."""
        log_debug(f"Input messages: \033[95m{len(messages)} messages\033[0m including system")

        # Extract system from messages
        filtered_messages, extracted_system = self._messages_to_contents(messages)

        # Combine system from kwargs (via _add_system_message) with extracted system
        system = kwargs.get("system")
        if not system and extracted_system:
            kwargs["system"] = extracted_system

        previous_role = None
        for i, message in enumerate(filtered_messages):
            role = message["role"]
            if role == "user":
                log_debug(f"Message \033[95m{i} (u)\033[0m: {message['content']}")
            else:
                log_debug(f"Message \033[95m{i} (a)\033[0m: {message['content']}")
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
        log_debug(f"LLM Response: \033[95m{truncate_string(str(result), max_length=1000)}\033[0m\n---")
        return result

    @classmethod
    def _get_delta(cls, chunk: Any) -> str:
        if hasattr(chunk, 'delta') and hasattr(chunk.delta, "text"):
            return chunk.delta.text
        return ""


class AnthropicBedrock(BedrockMixin, Anthropic):  # Keep it before Anthropic so API key is correct

    display_name = param.String(default="Anthropic on AWS Bedrock", constant=True, doc="Display name for UI")

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

    display_name = param.String(default="AWS Bedrock", constant=True, doc="Display name for UI")

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

    timeout = param.Number(default=120, bounds=(1, None), constant=True, doc="""
        The timeout in seconds for Bedrock API calls.""")

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


class Google(Llm):
    """
    A LLM implementation that calls Google's Gemini models.
    """

    api_key = param.String(default=os.getenv("GEMINI_API_KEY"), doc="The Google API key.")

    display_name = param.String(default="Google AI", constant=True, doc="Display name for UI")

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
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=1, bounds=(0, 1), constant=True)

    timeout = param.Number(default=120, bounds=(1, None), constant=True, doc="""
        The timeout in seconds for Google AI API calls.""")

    _supports_logfire = True
    _supports_model_stream = True
    _instructor_wrapper = "genai"

    @property
    def _client_kwargs(self):
        return {}

    def _create_base_client(self, **kwargs) -> Any:
        from google import genai
        if self.logfire_tags:
            self._logfire.instrument_google_genai()
        return genai.Client(api_key=self.api_key)

    def _create_instructor_client(self, base_client: Any, mode: Mode) -> Any:
        return instructor.from_genai(base_client, mode=mode, use_async=True)

    @classmethod
    def _get_delta(cls, chunk: Any) -> str:
        """Extract delta content from streaming response or full response."""
        if hasattr(chunk, 'text'):
            return chunk.text or ""
        if hasattr(chunk, 'content') and chunk.content:
            return chunk.content
        # Handle full response from generate_content (non-streaming fallback)
        if hasattr(chunk, 'candidates') and chunk.candidates:
            candidate = chunk.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    return candidate.content.parts[0].text or ""
        return ""

    @classmethod
    def _get_content(cls, response: Any) -> str:
        """Extract content from a non-streaming Google GenAI response."""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    return candidate.content.parts[0].text or ""
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
            elif role != "user":
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
        http_options = HttpOptions(timeout=self.timeout * 1000)  # timeout is in milliseconds
        thinking_config = ThinkingConfig(thinking_budget=0, include_thoughts=False)

        client = await self.get_client(model_spec, **kwargs)
        contents, system_instruction = self._messages_to_contents(messages)
        config = GenerateContentConfig(
            http_options=http_options,
            temperature=self.temperature,
            thinking_config=thinking_config,
            system_instruction=system_instruction,
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

    display_name = param.String(default="AI Navigator", constant=True, doc="Display name for UI")

    endpoint = param.String(
        default="http://localhost:8080/v1", doc="""
            The API endpoint; should include the full address, including the port.""")

    mode = param.Selector(default=Mode.JSON_SCHEMA)

    model_kwargs = param.Dict(default={
        "default": {"model": "server-model"},
    })

    select_models = param.List(default=["server-model"], constant=True, doc="Available models for selection dropdowns")


class Ollama(OpenAI):
    """
    An LLM implementation using the Ollama cloud.
    """

    api_key = param.String(default="ollama", doc="The Ollama API key; required but unused.")

    display_name = param.String(default="Ollama", constant=True, doc="Display name for UI")

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
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.25, bounds=(0, None), constant=True)


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

    display_name = param.String(default="WebLLM", constant=True, doc="Display name for UI")

    mode = param.Parameter(default=Mode.JSON_SCHEMA)

    model_kwargs = param.Dict({'default': {'model_slug': 'Qwen2.5-7B-Instruct-q4f16_1-MLC'}})

    select_models = param.List(default=[
        "Llama-3.2-3B-Instruct-q4f16_1-MLC",
        "Phi-3.5-mini-instruct-q4f16_1-MLC",
        "Qwen2.5-7B-Instruct-q4f16_1-MLC"
    ], constant=True, doc="Available models for selection dropdowns")

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

    display_name = param.String(default="LiteLLM", constant=True, doc="Display name for UI")

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
        "default": {"model": "gpt-4.1-mini"},
        "edit": {"model": "anthropic/claude-sonnet-4-5"},
        "sql": {"model": "gpt-4.1-mini"},
    }, doc="""
        Model configurations by type. LiteLLM supports model strings like:
        - OpenAI: "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini"
        - Anthropic: "anthropic/claude-sonnet-4-5", "anthropic/claude-haiku-4-5"
        - Google: "gemini/gemini-2.0-flash", "gemini/gemini-2.5-flash"
        - Mistral: "mistral/mistral-medium-latest", "mistral/mistral-small-latest"
        - And many more with format: "provider/model" or just "model" for defaults
    """)

    router_settings = param.Dict(default={}, doc="""
        Settings for LiteLLM Router for load balancing across multiple models.
        Example: {"routing_strategy": "least-busy", "num_retries": 3}""")

    select_models = param.List(default=[
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5-mini",
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-opus-4-1",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.5-flash",
        "mistral/mistral-medium-latest",
        "mistral/mistral-small-latest",
        "mistral/codestral-latest"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.7, bounds=(0, 2), constant=True)

    timeout = param.Number(default=120, bounds=(1, None), constant=True, doc="""
        The timeout in seconds for LiteLLM API calls.""")

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

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
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
