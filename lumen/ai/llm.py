from __future__ import annotations

import asyncio
import os

from functools import partial
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING, Any, Literal, TypedDict,
)

import instructor
import panel as pn
import param

from instructor.dsl.partial import Partial
from instructor.patch import Mode, patch
from pydantic import BaseModel

from lumen.ai.utils import log_debug

from .interceptor import Interceptor

if TYPE_CHECKING:
    MODEL_TYPE = Literal["default" | "reasoning" | "sql"]


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None


BASE_MODES = [Mode.JSON_SCHEMA, Mode.JSON, Mode.FUNCTIONS, Mode.TOOLS]


class Llm(param.Parameterized):
    """
    Baseclass for LLM implementations.

    An LLM implementation wraps a local or cloud based LLM provider
    with instructor to enable support for correctly validating Pydantic
    models.
    """

    mode = param.Selector(default=Mode.JSON_SCHEMA, objects=BASE_MODES, doc="""
        The calling mode used by instructor to guide the LLM towards generating
        outputs matching the schema.""")

    interceptor = param.ClassSelector(default=None, class_=Interceptor, doc="""
        Intercepter instance to intercept LLM calls, e.g. for logging.""")

    model_kwargs = param.Dict(default={}, doc="""
        LLM model definitions indexed by type. Supported types include
        'default', 'reasoning' and 'sql'. Agents may pick which model to
        invoke for different reasons.""")

    # Whether the LLM supports streaming of any kind
    _supports_stream = True

    # Whether the LLM supports streaming of Pydantic model output
    _supports_model_stream = True

    __abstract = True

    def _get_model_kwargs(self, model_spec: MODEL_TYPE | dict) -> dict[str, Any]:
        if model_spec in self.model_kwargs:
            model_kwargs = self.model_kwargs.get(model_spec) or self.model_kwargs["default"]
        else:
            model_kwargs = self.model_kwargs["default"]
            model_kwargs["model"] = model_spec  # override the default model with user provided model
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

    async def invoke(
        self,
        messages: list[Message],
        system: str = "",
        response_model: BaseModel | None = None,
        allow_partial: bool = False,
        model_spec: MODEL_TYPE | dict = "default",
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

    async def run_client(self, model_spec: MODEL_TYPE | dict, messages: list[Message], **kwargs):
        if response_model := kwargs.get("response_model"):
            log_debug(f"\033[93m{response_model.__name__}\033[0m model used", show_sep=True)

        previous_role = None
        for i, message in enumerate(messages):
            role = message["role"]
            if role == "system":
                continue
            if role == "user":
                log_debug(f"\033[95m{i} (u)\033[0m. {message['content']}")
            else:
                log_debug(f"\033[95m{i} (a)\033[0m. {message['content']}")
            if previous_role == role:
                log_debug(
                    "\033[91mWARNING: Two consecutive messages from the same role; "
                    "some providers disallow this.\033[0m"
                )
            previous_role = role
        log_debug(f"Length is \033[94m{len(messages)} messages\033[0m including system")

        client = await self.get_client(model_spec, **kwargs)
        return await client(messages=messages, **kwargs)


class Llama(Llm):
    """
    A LLM implementation using Llama.cpp Python wrapper together
    with huggingface_hub to fetch the models.
    """

    chat_format = param.String(constant=True)

    temperature = param.Number(default=0.4, bounds=(0, None), constant=True)

    model_kwargs = param.Dict(default={
        "default": {
            "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "model_file": "qwen2.5-coder-7b-instruct-q5_k_m.gguf",
            "chat_format": "qwen",
            "n_ctx": 131072,
        },
    })

    def _get_model_kwargs(self, model_spec: MODEL_TYPE | dict) -> dict[str, Any]:
        if isinstance(model_spec, dict):
            return model_spec

        if model_spec in self.model_kwargs or "/" not in model_spec:
            model_kwargs = self.model_kwargs.get(model_spec) or self.model_kwargs["default"]
        else:
            model_kwargs = self.model_kwargs["default"]
            repo, model_spec = model_spec.rsplit("/", 1)
            if ":" in model_spec:
                model_file, chat_format = model_spec.split(":")
                model_kwargs["chat_format"] = chat_format
            else:
                model_file = model_spec
            model_kwargs["repo"] = repo
            model_kwargs["model_file"] = model_file
        return dict(model_kwargs)

    @property
    def _client_kwargs(self) -> dict[str, Any]:
        return {"temperature": self.temperature}

    def _cache_model(self, model_spec: MODEL_TYPE | dict, **kwargs):
        from llama_cpp import Llama as LlamaCpp
        llm = LlamaCpp(**kwargs)

        raw_client = llm.create_chat_completion_openai_v1
        # patch works with/without response_model
        client_callable = patch(
            create=raw_client,
            mode=Mode.JSON_SCHEMA,
        )
        pn.state.cache[model_spec] = client_callable
        return client_callable

    async def get_client(self, model_spec: MODEL_TYPE | dict, response_model: BaseModel | None = None, **kwargs):
        if client_callable := pn.state.cache.get(model_spec):
            return client_callable
        from huggingface_hub import hf_hub_download

        model_kwargs = self._get_model_kwargs(model_spec)
        repo = model_kwargs["repo"]
        model_file = model_kwargs["model_file"]
        chat_format = model_kwargs["chat_format"]
        n_ctx = model_kwargs["n_ctx"]
        model_path = await asyncio.to_thread(hf_hub_download, repo, model_file)
        llm_kwargs = dict(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            seed=128,
            chat_format=chat_format,
            logits_all=False,
            use_mlock=True,
            verbose=False
        )
        client_callable = await asyncio.to_thread(self._cache_model, model_spec, **llm_kwargs)
        return client_callable

    async def run_client(self, model_spec: MODEL_TYPE | dict, messages: list[Message], **kwargs):
        client = await self.get_client(model_spec, **kwargs)
        return client(messages=messages, **kwargs)


class OpenAI(Llm):
    """
    An LLM implementation using the OpenAI cloud.
    """

    api_key = param.String(doc="The OpenAI API key.")

    endpoint = param.String(doc="The OpenAI API endpoint.")

    mode = param.Selector(default=Mode.TOOLS)

    temperature = param.Number(default=0.2, bounds=(0, None), constant=True)

    organization = param.String(doc="The OpenAI organization to charge.")

    model_kwargs = param.Dict(default={
        "default": {"model": "gpt-4o-mini"},
        "reasoning": {"model": "gpt-4o"},
    })

    use_logfire = param.Boolean(default=False, doc="""
        Whether to log LLM calls and responses to logfire.""")

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    async def get_client(self, model_spec: MODEL_TYPE | dict, response_model: BaseModel | None = None, **kwargs):
        import openai

        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
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
            llm = patch(llm)

        if self.interceptor:
            # must be called after instructor
            self.interceptor.patch_client_response(llm)

        client_callable = partial(llm.chat.completions.create, model=model)

        if self.use_logfire:
            import logfire
            logfire.configure()
            logfire.instrument_openai(llm)
        return client_callable


class AzureOpenAI(Llm):
    """
    A LLM implementation that uses the Azure OpenAI integration.
    """

    api_key = param.String(doc="The Azure API key.")

    api_version = param.String(doc="The Azure AI Studio API version.")

    endpoint = param.String(doc="The Azure AI Studio endpoint.")

    mode = param.Selector(default=Mode.TOOLS)

    temperature = param.Number(default=0.2, bounds=(0, None), constant=True)

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    async def get_client(self, model_spec: MODEL_TYPE | dict, response_model: BaseModel | None = None, **kwargs):
        import openai

        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")
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
            llm = patch(llm)

        if self.interceptor:
            # must be called after instructor
            self.interceptor.patch_client_response(llm)

        client_callable = partial(llm.chat.completions.create, model=model)
        return client_callable


class MistralAI(Llm):
    """
    A LLM implementation that calls Mistral AI.
    """

    api_key = param.String(default=os.getenv("MISTRAL_API_KEY"), doc="The Mistral AI API key.")

    mode = param.Selector(default=Mode.MISTRAL_TOOLS, objects=[Mode.JSON_SCHEMA, Mode.MISTRAL_TOOLS])

    temperature = param.Number(default=0.7, bounds=(0, 1), constant=True)

    model_kwargs = param.Dict(default={
        "default": {"model": "mistral-small-latest"},
        "reasoning": {"model": "mistral-large-latest"},
    })

    _supports_model_stream = False  # instructor doesn't work with Mistral's streaming

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        from mistralai import Mistral

        model_kwargs = self._get_model_kwargs(model_spec)
        model_kwargs["api_key"] = self.api_key
        model = model_kwargs.pop("model")
        llm = Mistral(**model_kwargs)

        stream = kwargs.get("stream", False)
        llm.chat.completions = SimpleNamespace(create=None)  # make it like OpenAI for simplicity
        llm.chat.completions.create = llm.chat.stream_async if stream else llm.chat.complete_async

        if self.interceptor:
            self.interceptor.patch_client(llm, mode="store_inputs")

        if response_model:
            llm = patch(llm)

        if self.interceptor:
            self.interceptor.patch_client_response(llm)

        client_callable = partial(llm.chat.completions.create, model=model)
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

    api_key = param.String(default=os.getenv("AZURE_API_KEY"), doc="The Azure API key")

    endpoint = param.String(default=os.getenv("AZURE_ENDPOINT"), doc="The Azure API endpoint to invoke.")

    model_kwargs = param.Dict(default={
        "default": {"model": "azureai"},
    })

    async def get_client(self, model_spec: str | dict, response_model: BaseModel | None = None, **kwargs):
        from mistralai_azure import MistralAzure

        async def llm_chat_non_stream_async(*args, **kwargs):
            response = await llm.chat.complete_async(*args, **kwargs)
            return response.choices[0].message.content

        model_kwargs = self._get_model_kwargs(model_spec)
        model_kwargs["api_key"] = self.api_key
        model_kwargs["azure_endpoint"] = self.endpoint
        model = model_kwargs.pop("model")
        llm = MistralAzure(**model_kwargs)

        stream = kwargs.get("stream", False)
        llm.chat.completions = SimpleNamespace(create=None)  # make it like OpenAI for simplicity
        llm.chat.completions.create = llm.chat.stream_async if stream else llm.chat.complete_async

        if self.interceptor:
            self.interceptor.patch_client(llm, mode="store_inputs")

        if response_model:
            llm = patch(llm)

        if self.interceptor:
            self.interceptor.patch_client_response(llm)

        client_callable = partial(llm.chat.completions.create, model=model)
        return client_callable


class AnthropicAI(Llm):
    """
    A LLM implementation that calls Anthropic models such as Claude.
    """

    api_key = param.String(default=os.getenv("ANTHROPIC_API_KEY"), doc="The Anthropic API key.")

    mode = param.Selector(default=Mode.ANTHROPIC_TOOLS, objects=[Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS])

    temperature = param.Number(default=0.7, bounds=(0, 1), constant=True)

    model_kwargs = param.Dict(default={
        "default": {"model": "claude-3-haiku-20240307"},
        "reasoning": {"model": "claude-3-5-sonnet-20240620"},
    })

    _supports_model_stream = True

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature, "max_tokens": 1024}

    async def get_client(self, model_spec: MODEL_TYPE | dict, response_model: BaseModel | None = None, **kwargs):
        if self.interceptor:
            raise NotImplementedError("Interceptors are not supported for AnthropicAI.")

        from anthropic import AsyncAnthropic

        model_kwargs = self._get_model_kwargs(model_spec)
        model = model_kwargs.pop("model")

        llm = AsyncAnthropic(api_key=self.api_key, **model_kwargs)

        if response_model:
            client = instructor.from_anthropic(llm)
            return partial(client.messages.create, model=model)
        else:
            return partial(llm.messages.create, model=model)

    @classmethod
    def _get_delta(cls, chunk: Any) -> str:
        if hasattr(chunk, 'delta'):
            if hasattr(chunk.delta, "text"):
                return chunk.delta.text
        return ""

    def _add_system_message(self, messages: list[Message], system: str, input_kwargs: dict[str, Any]):
        input_kwargs["system"] = system
        return messages, input_kwargs
