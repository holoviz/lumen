from __future__ import annotations

import asyncio
import base64
import os

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, TypedDict

import instructor
import panel as pn
import param

from instructor import Mode, patch
from instructor.dsl.partial import Partial
from instructor.processing.multimodal import Image
from pydantic import BaseModel

from .interceptor import Interceptor
from .services import AzureOpenAIMixin, LlamaCppMixin, OpenAIMixin
from .utils import format_exception, log_debug, truncate_string


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None


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
    Baseclass for LLM implementations.

    An LLM implementation wraps a local or cloud based LLM provider
    with instructor to enable support for correctly validating Pydantic
    models.
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

    __abstract = True

    def __init__(self, **params):
        if "mode" in params:
            if isinstance(params["mode"], str):
                params["mode"] = Mode[params["mode"].upper()]
        super().__init__(**params)
        if self.logfire_tags is not None and not self._supports_logfire:
            raise ValueError(
                f"LLM {self.__class__.__name__} does not support logfire."
            )
        if not self.model_kwargs.get("default"):
            raise ValueError(
                f"Please specify a 'default' model in the model_kwargs "
                f"parameter for {self.__class__.__name__}."
            )

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
        if output is None or output == "":
            raise ValueError("LLM failed to return valid output.")
        return output

    @classmethod
    def _get_delta(cls, chunk) -> str:
        if chunk.choices:
            return chunk.choices[0].delta.content or ""
        return ""

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
        model: Literal['default' | 'reasoning' | 'sql']
            The model as listed in the model_kwargs parameter
            to invoke to answer the query.

        Yields
        ------
        The string or response_model field.
        """
        if self.logfire_tags is not None:
            output = await self.invoke(
                messages,
                system=system,
                response_model=response_model,
                model_spec=model_spec,
                **kwargs,
            )
            if field is not None and hasattr(output, field):
                output = getattr(output, field)
            elif hasattr(output, "choices"):
                output = output.choices[0].message.content
            yield output
            return

        if ((response_model and not self._supports_model_stream) or
            not self._supports_stream):
            yield await self.invoke(
                messages,
                system=system,
                response_model=response_model,
                model_spec=model_spec,
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
            **kwargs,
        )
        if isinstance(chunks, BaseModel):
            yield getattr(chunks, field) if field is not None else chunks
            return

        try:
            async for chunk in chunks:
                if response_model is None:
                    string += self._get_delta(chunk)
                    yield string
                else:
                    yield getattr(chunk, field) if field is not None else chunk
        except TypeError:
            # Handle synchronous iterators
            for chunk in chunks:
                if response_model is None:
                    string += self._get_delta(chunk)
                    yield string
                else:
                    yield getattr(chunk, field) if field is not None else chunk

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
    A LLM implementation using Llama.cpp Python wrapper together
    with huggingface_hub to fetch the models.
    """

    chat_format = param.String(constant=True)

    display_name = param.String(default="Llama.cpp", constant=True, doc="Display name for UI")

    mode = param.Selector(default=Mode.JSON_SCHEMA, objects=BASE_MODES)

    model_kwargs = param.Dict(default={
        "default": {
            "repo_id": "unsloth/Qwen3-8B-GGUF",
            "filename": "Qwen3-8B-Q5_K_M.gguf",
            "chat_format": "qwen",
        },
    })

    select_models = param.List(default=[
        "unsloth/Qwen3-8B-GGUF",
        "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "unsloth/DeepSeek-V3.1-GGUF",
        "unsloth/gpt-oss-20b-GGUF",
        "unsloth/GLM-4.6-GGUF",
        "microsoft/Phi-4-GGUF",
        "meta-llama/Llama-3.3-70B-Instruct-GGUF"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.4, bounds=(0, None), constant=True)

    def _get_model_kwargs(self, model_spec: str | dict) -> dict[str, Any]:
        if isinstance(model_spec, dict):
            return model_spec

        # First get base model kwargs using parent implementation
        if model_spec in self.model_kwargs or "/" not in model_spec:
            model_kwargs = super()._get_model_kwargs(model_spec)
        else:
            # Use mixin to resolve repo_id/filename from model spec string
            base_kwargs = self.model_kwargs["default"]
            model_kwargs = self.resolve_model_spec(model_spec, base_kwargs)

        # Ensure n_ctx is set (0 = from model)
        if "n_ctx" not in model_kwargs:
            model_kwargs["n_ctx"] = 0

        # Merge with instance-level configuration from the mixin
        try:
            full_kwargs = self._instantiate_client_kwargs(model_kwargs=model_kwargs)
            return full_kwargs
        except Exception:
            # Fallback to just model_kwargs if there's an issue with instance config
            return dict(model_kwargs)

    @property
    def _client_kwargs(self) -> dict[str, Any]:
        return {"temperature": self.temperature}

    def _cache_model(self, model_spec: str | dict, mode: str, **kwargs):
        from llama_cpp import Llama as LlamaCpp
        llm = LlamaCpp(**kwargs)

        raw_client = llm.create_chat_completion_openai_v1
        # patch works with/without response_model
        client_callable = patch(create=raw_client, mode=mode)
        pn.state.cache[(model_spec, mode)] = client_callable
        return client_callable

    @classmethod
    def warmup(cls, model_kwargs: dict | None):
        model_kwargs = model_kwargs or cls.model_kwargs
        if 'default' not in model_kwargs:
            model_kwargs['default'] = cls.model_kwargs['default']
        # Use the mixin's warmup functionality
        cls._warmup_models(model_kwargs)

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        mode = model_kwargs.pop("mode", self.mode)
        if client_callable := pn.state.cache.get((model_spec, mode)):
            return client_callable

        # Use the mixin to handle model path resolution and kwargs
        llm_kwargs = self._instantiate_client_kwargs(
            model_kwargs=model_kwargs,
            n_gpu_layers=-1,
            seed=128,
            logits_all=False,
        )

        client_callable = await asyncio.to_thread(self._cache_model, model_spec, mode=mode, **llm_kwargs)
        return client_callable

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
        "default": {"model": "gpt-4.1-mini"},
        "ui": {"model": "gpt-4.1-nano"},
    })

    select_models = param.List(default=[
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.25, bounds=(0, None), constant=True)

    timeout = param.Number(default=120, bounds=(1, None), constant=True, doc="""
        The timeout in seconds for OpenAI API calls.""")

    _supports_logfire = True

    @param.depends("logfire_tags", watch=True, on_init=True)
    def _update_logfire_tags(self):
        if self.logfire_tags is not None:
            import logfire
            logfire.configure(send_to_logfire=True)
            self._logfire = logfire.Logfire(tags=self.logfire_tags)
        else:
            self._logfire = None

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    def _get_model_kwargs(self, model_spec: str | dict) -> dict[str, Any]:
        model_kwargs = super()._get_model_kwargs(model_spec)

        # Merge with instance-level client configuration from the mixin
        instance_kwargs = self._instantiate_client_kwargs()

        # Model-specific kwargs should override instance defaults
        merged_kwargs = {**instance_kwargs, **model_kwargs}

        return merged_kwargs

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        log_debug(f"LLM Model: \033[96m{model!r}\033[0m")
        mode = model_kwargs.pop("mode", self.mode)

        # Use the mixin to create the OpenAI client
        llm = self._instantiate_client(async_client=True, **model_kwargs)

        if self.logfire_tags:
            self._logfire.instrument_openai(llm)

        if self.interceptor:
            self.interceptor.patch_client(llm, mode="store_inputs")

        if response_model:
            llm = patch(llm, mode=mode)

        if self.interceptor:
            # must be called after instructor
            self.interceptor.patch_client_response(llm)

        client_callable = partial(llm.chat.completions.create, model=model, timeout=self.timeout, **self._get_create_kwargs(response_model))
        return client_callable


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

        # Merge with instance-level client configuration from the mixin
        instance_kwargs = self._instantiate_client_kwargs()

        # Model-specific kwargs should override instance defaults
        merged_kwargs = {**instance_kwargs, **model_kwargs}

        return merged_kwargs

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)

        # Use the mixin to create the Azure OpenAI client
        llm = self._instantiate_client(async_client=True, **model_kwargs)

        if self.interceptor:
            self.interceptor.patch_client(llm, mode="store_inputs")

        if response_model:
            llm = patch(llm, mode=mode)

        if self.interceptor:
            # must be called after instructor
            self.interceptor.patch_client_response(llm)

        client_callable = partial(llm.chat.completions.create, model=model, timeout=self.timeout, **self._get_create_kwargs(response_model))
        return client_callable


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

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        from mistralai import Mistral

        model_kwargs = self._get_model_kwargs(model_spec)
        model_kwargs["api_key"] = self.api_key
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)
        llm = Mistral(**model_kwargs)

        stream = kwargs.get("stream", False)
        llm.chat.completions = SimpleNamespace(create=None)  # make it like OpenAI for simplicity
        llm.chat.completions.create = llm.chat.stream_async if stream else llm.chat.complete_async

        if self.interceptor:
            self.interceptor.patch_client(llm, mode="store_inputs")

        if response_model:
            llm = patch(llm, mode=mode)

        if self.interceptor:
            self.interceptor.patch_client_response(llm)

        client_callable = partial(llm.chat.completions.create, model=model, timeout_ms=self.timeout * 1000, **self._get_create_kwargs(response_model))
        return client_callable

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

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        from mistralai_azure import MistralAzure

        async def llm_chat_non_stream_async(*args, **kwargs):
            response = await llm.chat.complete_async(*args, **kwargs)
            return response.choices[0].message.content

        model_kwargs = self._get_model_kwargs(model_spec)
        model_kwargs["api_key"] = self.api_key
        model_kwargs["azure_endpoint"] = self.endpoint
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)
        llm = MistralAzure(**model_kwargs)

        stream = kwargs.get("stream", False)
        llm.chat.completions = SimpleNamespace(create=None)  # make it like OpenAI for simplicity
        llm.chat.completions.create = llm.chat.stream_async if stream else llm.chat.complete_async

        if self.interceptor:
            self.interceptor.patch_client(llm, mode="store_inputs")

        if response_model:
            llm = patch(llm, mode=mode)

        if self.interceptor:
            self.interceptor.patch_client_response(llm)

        client_callable = partial(llm.chat.completions.create, model=model, timeout_ms=self.timeout * 1000, **self._get_create_kwargs(response_model))
        return client_callable


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

    _supports_model_stream = True

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature, "max_tokens": 1024}

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        if self.interceptor:
            raise NotImplementedError("Interceptors are not supported for Anthropic.")

        from anthropic import AsyncAnthropic

        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)

        llm = AsyncAnthropic(api_key=self.api_key, **model_kwargs)

        if response_model:
            client = instructor.from_anthropic(llm, mode=mode)
            return partial(client.messages.create, model=model, timeout=self.timeout, **self._get_create_kwargs(response_model))
        else:
            return partial(llm.messages.create, model=model, timeout=self.timeout, **self._get_create_kwargs(response_model))

    @classmethod
    def _get_delta(cls, chunk: Any) -> str:
        if hasattr(chunk, 'delta'):
            if hasattr(chunk.delta, "text"):
                return chunk.delta.text
        return ""

    def _add_system_message(self, messages: list[Message], system: str, input_kwargs: dict[str, Any]):
        input_kwargs["system"] = system
        return messages, input_kwargs


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

    _supports_model_stream = True

    # Cached genai client to avoid creating new aiohttp sessions on each call
    _genai_client = None

    @property
    def _client_kwargs(self):
        return {}

    def _get_genai_client(self):
        """Get or create a cached genai.Client instance."""
        if self._genai_client is None:
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
        return self._genai_client

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
    def _messages_to_contents(cls, messages: list[Message]) -> tuple[list[dict[str, Any]], str | None]:
        """
        Transform messages into contents format expected by Google GenAI API.

        Parameters
        ----------
        messages : list[Message]
            List of messages with 'role', 'content', and optional 'name' fields.

        Returns
        -------
        tuple[list[dict[str, Any]], str | None]
            Tuple of (contents list, system_instruction string or None)
        """
        contents = []
        system_instruction = None

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                system_instruction = content
                continue
            elif role == "user":
                role = "user"
            else:
                role = "model"

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

        # Reuse cached client to avoid aiohttp session issues
        llm = self._get_genai_client()

        if response_model:
            client = instructor.from_genai(llm, mode=mode, use_async=True)
            return partial(client.chat.completions.create, model=model, **self._get_create_kwargs(response_model))
        elif kwargs.get("stream"):
            return partial(llm.aio.models.generate_content_stream, model=model)
        else:
            return partial(llm.aio.models.generate_content, model=model)

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

        client_result = await self.get_client(model_spec, **kwargs)
        contents, system_instruction = self._messages_to_contents(messages)
        config = GenerateContentConfig(
            http_options=http_options,
            temperature=self.temperature,
            thinking_config=thinking_config,
            system_instruction=system_instruction,
        )

        if response_model:
            # client_result is a partial callable from instructor
            result = await client_result(messages=messages, config=config, **kwargs)
            return result

        kwargs.pop("stream", None)  # already handled
        result = await client_result(contents=contents, config=config, **kwargs)
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
        "default": {"model": "qwen3:8b"},
    })

    select_models = param.List(default=[
        "qwen3:8b",
        "qwen3-coder:30b",
        "deepseek-r1:7b",
        "llama3.3:70b",
        "llama4:latest",
        "gemma3:12b",
        "mistral-small3.2:24b",
        "qwen2.5-coder:7b",
        "phi4:14b"
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

    display_name = param.String(default="WebLLM", constant=True, doc="Display name for UI")

    mode = param.Parameter(default=Mode.JSON_SCHEMA)

    model_kwargs = param.Dict({'default': {'model_slug': 'Qwen2.5-7B-Instruct-q4f16_1-MLC'}})

    select_models = param.List(default=[
        "Llama-3.2-3B-Instruct-q4f16_1-MLC",
        "Phi-3.5-mini-instruct-q4f16_1-MLC",
        "Qwen2.5-7B-Instruct-q4f16_1-MLC"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.4, bounds=(0, None))

    def __init__(self, **params):
        from panel_web_llm import WebLLM as pnWebLLM
        self._llm = pnWebLLM()
        super().__init__(**params)

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
        client_callable = patch(create=self._create_completion, mode=mode)
        return client_callable

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

    _supports_stream = True
    _supports_model_stream = True

    def __init__(self, **params):
        super().__init__(**params)
        self._setup_router()
        # Configure caching if enabled
        if self.enable_caching:
            self._setup_caching()

    def _setup_caching(self):
        """Enable LiteLLM caching."""
        import litellm

        from litellm import Cache
        litellm.cache = Cache()

    def _setup_router(self):
        """Initialize LiteLLM Router for load balancing."""
        from litellm import Router

        # Build model list from model_kwargs
        model_list = []
        for key, config in self.model_kwargs.items():
            if isinstance(config, dict) and 'model' in config:
                model_list.append({
                    'model_name': key,
                    'litellm_params': config
                })

        # Add fallback models to router settings if provided
        router_kwargs = dict(self.router_settings)
        if self.fallback_models:
            router_kwargs['fallbacks'] = self.fallback_models

        self.router = Router(model_list=model_list, timeout=self.timeout, **router_kwargs)

    @property
    def _client_kwargs(self):
        """Base kwargs for all LiteLLM calls."""
        kwargs = {
            "temperature": self.temperature,
        }
        kwargs.update(self.litellm_params)
        return kwargs

    def _get_model_string(self, model_spec: str | dict) -> str:
        """Extract the model string from model spec."""
        model_kwargs = self._get_model_kwargs(model_spec)
        return model_kwargs.get("model")

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        """
        Get a client callable that's compatible with instructor.

        For LiteLLM, we create a wrapper around litellm.acompletion that
        can be patched by instructor.
        """
        model = self._get_model_string(model_spec)
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)

        # Use router if available, otherwise use litellm.acompletion directly
        llm = self.router.acompletion
        if response_model:
            llm = instructor.from_litellm(llm, mode=mode)
            return partial(llm.chat.completions.create, model=model, **self._get_create_kwargs(response_model))
        else:
            return partial(llm, model=model, **self._client_kwargs, **model_kwargs)

    @classmethod
    def _get_delta(cls, chunk) -> str:
        """Extract delta content from streaming chunks."""
        # LiteLLM returns OpenAI-compatible responses
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                return choice.delta.content or ""
        return ""
