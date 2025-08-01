from __future__ import annotations

import asyncio
import json
import os

from functools import partial
from types import SimpleNamespace
from typing import Any, Literal, TypedDict

import instructor
import panel as pn
import param

from instructor.dsl.partial import Partial
from instructor.patch import Mode, patch
from pydantic import BaseModel

from .interceptor import Interceptor
from .models import YesNo
from .utils import format_exception, log_debug, truncate_string


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None

BASE_MODES = list(Mode)


class Llm(param.Parameterized):
    """
    Baseclass for LLM implementations.

    An LLM implementation wraps a local or cloud based LLM provider
    with instructor to enable support for correctly validating Pydantic
    models.
    """

    create_kwargs = param.Dict(default={}, doc="""
        Additional keyword arguments to pass to the LLM provider
        when calling chat.completions.create.""")

    mode = param.Selector(default=Mode.JSON_SCHEMA, objects=BASE_MODES, doc="""
        The calling mode used by instructor to guide the LLM towards generating
        outputs matching the schema.""")

    interceptor = param.ClassSelector(default=None, class_=Interceptor, doc="""
        Intercepter instance to intercept LLM calls, e.g. for logging.""")

    model_kwargs = param.Dict(default={}, doc="""
        LLM model definitions indexed by type. Supported types include
        'default', 'reasoning' and 'sql'. Agents may pick which model to
        invoke for different reasons.""")

    _ready = param.Boolean(default=False, doc="""
        Whether the LLM has been initialized and is ready to use.""")

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
        log_debug(f"LLM Model: \033[96m{model_kwargs.get('model')!r}\033[0m")
        return dict(model_kwargs)

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

        if response_model is not None:
            if allow_partial:
                response_model = Partial[response_model]
            kwargs["response_model"] = response_model

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
                messages=[{'role': 'user', 'content': 'Are you there? YES | NO'}],
                model_spec="ui",
                response_model=YesNo
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
        try:
            async for chunk in chunks:
                if response_model is None:
                    delta = self._get_delta(chunk)
                    string += delta
                    yield string
                else:
                    yield getattr(chunk, field) if field is not None else chunk
        except TypeError:
            for chunk in chunks:
                if response_model is None:
                    delta = self._get_delta(chunk)
                    string += delta
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


class LlamaCpp(Llm):
    """
    A LLM implementation using Llama.cpp Python wrapper together
    with huggingface_hub to fetch the models.
    """

    chat_format = param.String(constant=True)

    display_name = param.String(default="Llama.cpp", constant=True, doc="Display name for UI")

    mode = param.Selector(default=Mode.JSON_SCHEMA, objects=BASE_MODES)

    model_kwargs = param.Dict(default={
        "default": {
            "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "model_file": "qwen2.5-coder-7b-instruct-q5_k_m.gguf",
            "chat_format": "qwen",
        },
    })

    select_models = param.List(default=[
        "microsoft/Phi-3-mini-4k-instruct-gguf",
        "meta-llama/Llama-2-7b-chat-hf",
        "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "TheBloke/CodeLlama-7B-Instruct-GGUF"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.4, bounds=(0, None), constant=True)

    def _get_model_kwargs(self, model_spec: str | dict) -> dict[str, Any]:
        if isinstance(model_spec, dict):
            return model_spec

        model_kwargs = self.model_kwargs["default"]
        if model_spec in self.model_kwargs or "/" not in model_spec:
            model_kwargs = super()._get_model_kwargs(model_kwargs)
        else:
            repo, model_spec = model_spec.rsplit("/", 1)
            if ":" in model_spec:
                model_file, chat_format = model_spec.split(":")
                model_kwargs["chat_format"] = chat_format
            else:
                model_file = model_spec
            model_kwargs["repo"] = repo
            model_kwargs["model_file"] = model_file

        if "n_ctx" not in model_kwargs:
            # 0 = from model
            model_kwargs["n_ctx"] = 0
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
        huggingface_models = {
            model: llm_spec for model, llm_spec in model_kwargs.items()
            if 'repo' in llm_spec or 'repo_id' in llm_spec
        }
        if not huggingface_models:
            return

        from huggingface_hub import hf_hub_download
        print(f"{cls.__name__} provider is downloading following models:\n\n{json.dumps(huggingface_models, indent=2)}")  # noqa: T201
        for kwargs in model_kwargs.values():
            repo = kwargs.get('repo', kwargs.get('repo_id'))
            model_file = kwargs.get('model_file')
            hf_hub_download(repo, model_file)

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        model_kwargs = self._get_model_kwargs(model_spec)
        mode = model_kwargs.pop("mode", self.mode)
        if client_callable := pn.state.cache.get((model_spec, mode)):
            return client_callable
        if 'repo' in model_kwargs:
            from huggingface_hub import hf_hub_download
            repo = model_kwargs.get('repo', model_kwargs.get('repo_id'))
            model_file = model_kwargs.get('model_file')
            model_path = await asyncio.to_thread(hf_hub_download, repo, model_file)
        elif 'model_path' in model_kwargs:
            model_path = model_kwargs['model_path']
        else:
            raise ValueError(
                "LlamaCpp.model_kwargs must contain either a 'repo' and 'model_file' "
                "(to fetch a model using `huggingface_hub` or a 'model_path' pointing "
                "to a model on disk."
            )
        llm_kwargs = dict(
            model_path=model_path,
            n_gpu_layers=-1,
            seed=128,
            logits_all=False,
            use_mlock=True,
            verbose=False,
        )
        llm_kwargs.update(model_kwargs)
        client_callable = await asyncio.to_thread(self._cache_model, model_spec, mode=mode, **llm_kwargs)
        return client_callable

    async def run_client(self, model_spec: str | dict, messages: list[Message], **kwargs):
        client = await self.get_client(model_spec, **kwargs)
        return client(messages=messages, **kwargs)


class OpenAI(Llm):
    """
    An LLM implementation using the OpenAI cloud.
    """

    api_key = param.String(doc="The OpenAI API key.")

    display_name = param.String(default="OpenAI", constant=True, doc="Display name for UI")

    endpoint = param.String(doc="The OpenAI API endpoint.")

    mode = param.Selector(default=Mode.TOOLS)

    model_kwargs = param.Dict(default={
        "default": {"model": "gpt-4o-mini"},
        "sql": {"model": "gpt-4.1-mini"},
        "vega_lite": {"model": "gpt-4.1-mini"},
        "edit": {"model": "gpt-4.1-mini"},
    })

    organization = param.String(doc="The OpenAI organization to charge.")

    select_models = param.List(default=[
        "gpt-3.5-turbo",
        "gpt-4.1-mini",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.25, bounds=(0, None), constant=True)

    use_logfire = param.Boolean(default=False, doc="""
        Whether to log LLM calls and responses to logfire.""")

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        import openai

        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)
        if self.endpoint:
            model_kwargs["base_url"] = self.endpoint
        if self.api_key:
            model_kwargs["api_key"] = self.api_key
        if self.organization:
            model_kwargs["organization"] = self.organization
        llm = openai.AsyncOpenAI(**model_kwargs)

        if self.interceptor:
            self.interceptor.patch_client(llm, mode="store_inputs")

        if response_model:
            llm = patch(llm, mode=mode)

        if self.interceptor:
            # must be called after instructor
            self.interceptor.patch_client_response(llm)

        client_callable = partial(llm.chat.completions.create, model=model, **self.create_kwargs)

        if self.use_logfire:
            import logfire
            logfire.configure()
            logfire.instrument_openai(llm)
        return client_callable


class AzureOpenAI(Llm):
    """
    A LLM implementation that uses the Azure OpenAI integration.
    """

    api_key = param.String(default=os.getenv("AZUREAI_ENDPOINT_KEY"), doc="The Azure API key.")

    api_version = param.String(default="2024-10-21", doc="The Azure AI Studio API version.")

    display_name = param.String(default="Azure OpenAI", constant=True, doc="Display name for UI")

    endpoint = param.String(default=os.getenv('AZUREAI_ENDPOINT_URL'), doc="The Azure AI Studio endpoint.")

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

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        import openai

        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)
        if self.api_version:
            model_kwargs["api_version"] = self.api_version
        if self.api_key:
            model_kwargs["api_key"] = self.api_key
        if self.endpoint:
            model_kwargs["azure_endpoint"] = self.endpoint
        llm = openai.AsyncAzureOpenAI(**model_kwargs)

        if self.interceptor:
            self.interceptor.patch_client(llm, mode="store_inputs")

        if response_model:
            llm = patch(llm, mode=mode)

        if self.interceptor:
            # must be called after instructor
            self.interceptor.patch_client_response(llm)

        client_callable = partial(llm.chat.completions.create, model=model, **self.create_kwargs)
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
        "edit": {"model": "mistral-large-latest"},
    })

    select_models = param.List(default=[
        "codestral-latest",
        "mistral-7b-instruct",
        "mistral-large-latest",
        "mistral-small-latest",
        "mixtral-8x7b-instruct"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.7, bounds=(0, 1), constant=True)

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

        client_callable = partial(llm.chat.completions.create, model=model, **self.create_kwargs)
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

        client_callable = partial(llm.chat.completions.create, model=model, **self.create_kwargs)
        return client_callable


class AnthropicAI(Llm):
    """
    A LLM implementation that calls Anthropic models such as Claude.
    """

    api_key = param.String(default=os.getenv("ANTHROPIC_API_KEY"), doc="The Anthropic API key.")

    display_name = param.String(default="Anthropic", constant=True, doc="Display name for UI")

    mode = param.Selector(default=Mode.ANTHROPIC_TOOLS, objects=[Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS])

    model_kwargs = param.Dict(default={
        "default": {"model": "claude-3-5-haiku-latest"},
        "edit": {"model": "claude-3-5-sonnet-latest"},
    })

    select_models = param.List(default=[
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-4-sonnet-latest"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.7, bounds=(0, 1), constant=True)

    _supports_model_stream = True

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature, "max_tokens": 1024}

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        if self.interceptor:
            raise NotImplementedError("Interceptors are not supported for AnthropicAI.")

        from anthropic import AsyncAnthropic

        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)

        llm = AsyncAnthropic(api_key=self.api_key, **model_kwargs)

        if response_model:
            client = instructor.from_anthropic(llm, mode=mode)
            return partial(client.messages.create, model=model, **self.create_kwargs)
        else:
            return partial(llm.messages.create, model=model, **self.create_kwargs)

    @classmethod
    def _get_delta(cls, chunk: Any) -> str:
        if hasattr(chunk, 'delta'):
            if hasattr(chunk.delta, "text"):
                return chunk.delta.text
        return ""

    def _add_system_message(self, messages: list[Message], system: str, input_kwargs: dict[str, Any]):
        input_kwargs["system"] = system
        return messages, input_kwargs


class GoogleAI(Llm):
    """
    A LLM implementation that calls Google's Gemini models.
    """

    api_key = param.String(default=os.getenv("GEMINI_API_KEY"), doc="The Google API key.")

    display_name = param.String(default="Google AI", constant=True, doc="Display name for UI")

    mode = param.Selector(default=Mode.GENAI_TOOLS, objects=[Mode.GENAI_TOOLS, Mode.GENAI_STRUCTURED_OUTPUTS])

    model_kwargs = param.Dict(default={
        "default": {"model": "gemini-2.0-flash"},  # Cost-optimized, low latency
        "edit": {"model": "gemini-2.5-flash-preview-05-20"},  # Thinking model, balanced price/performance
    })

    select_models = param.List(default=[
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=1, bounds=(0, 1), constant=True)

    _supports_model_stream = True

    @property
    def _client_kwargs(self):
        return {}

    @classmethod
    def _get_delta(cls, chunk: Any) -> str:
        """Extract delta content from streaming response."""
        if hasattr(chunk, 'text'):
            return chunk.text
        elif hasattr(chunk, 'content') and chunk.content:
            return chunk.content
        return ""

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        from google import genai
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)

        llm = genai.Client(api_key=self.api_key, **model_kwargs)

        if response_model:
            client = instructor.from_genai(llm, mode=mode, use_async=True)
            return partial(client.chat.completions.create, model=model, **self.create_kwargs)
        else:
            chat = llm.aio.models
            if kwargs.pop("stream"):
                return partial(chat.generate_content_stream, model=model, **self.create_kwargs)
            else:
                return partial(chat.generate_content, model=model, **self.create_kwargs)

    async def run_client(self, model_spec: str | dict, messages: list[Message], **kwargs):
        """Override to handle Gemini-specific message format conversion."""
        try:
            from google.genai.types import GenerateContentConfig
        except ImportError as exc:
            raise ImportError(
                "Please install the `google-generativeai` package to use Google AI models. "
                "You can install it with `pip install -U google-genai`."
            ) from exc

        client = await self.get_client(model_spec, **kwargs)

        if kwargs.get("response_model"):
            config = GenerateContentConfig(temperature=self.temperature)
            return await client(messages=messages, config=config, **kwargs)
        else:
            kwargs.pop("stream")
            system_instruction = next(
                message["content"] for message in messages if message["role"] == "system"
            )
            config = GenerateContentConfig(temperature=self.temperature, system_instruction=system_instruction)
            prompt = messages.pop(-1)["content"]
            return await client(contents=[prompt], **kwargs)


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
        "default": {"model": "qwen2.5-coder:7b"},
    })

    select_models = param.List(default=[
        "codellama:7b",
        "llama2:13b",
        "llama3.2:latest",
        "mistral:latest",
        "qwen2.5-coder:7b"
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
                messages=[{'role': 'user', 'content': 'Are you there? YES | NO'}],
                model_spec="ui",
                response_model=YesNo
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
        "default": {"model": "gpt-4o-mini"},
        "edit": {"model": "claude-3-5-sonnet-latest"},
        "sql": {"model": "gpt-4o-mini"},
    }, doc="""
        Model configurations by type. LiteLLM supports model strings like:
        - OpenAI: "gpt-4", "gpt-4o-mini"
        - Anthropic: "claude-3-5-sonnet-latest", "claude-3-haiku"
        - Google: "gemini/gemini-pro", "gemini/gemini-1.5-flash"
        - And many more with format: "provider/model" or just "model" for defaults
    """)

    router_settings = param.Dict(default={}, doc="""
        Settings for LiteLLM Router for load balancing across multiple models.
        Example: {"routing_strategy": "least-busy", "num_retries": 3}""")

    select_models = param.List(default=[
        "anthropic/claude-3-haiku",
        "azure/gpt-4",
        "claude-3-5-sonnet-latest",
        "gemini/gemini-pro",
        "gpt-4o-mini",
        "openai/gpt-4"
    ], constant=True, doc="Available models for selection dropdowns")

    temperature = param.Number(default=0.7, bounds=(0, 2), constant=True)

    _supports_stream = True
    _supports_model_stream = True

    def __init__(self, **params):
        super().__init__(**params)
        # Configure caching if enabled
        if self.enable_caching:
            self._setup_caching()

    def _setup_caching(self):
        """Enable LiteLLM caching."""
        import litellm

        from litellm import Cache
        litellm.cache = Cache()

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
        import litellm

        model = self._get_model_string(model_spec)
        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
        mode = model_kwargs.pop("mode", self.mode)

        llm = litellm.acompletion
        if response_model:
            llm = instructor.from_litellm(llm, mode=mode)
            return partial(llm.chat.completions.create, model=model, **self.create_kwargs)
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
