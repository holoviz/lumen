from __future__ import annotations

import os

from functools import partial

import panel as pn
import param

from instructor import from_openai
from instructor.dsl.partial import Partial
from instructor.patch import Mode, patch
from pydantic import BaseModel


class Llm(param.Parameterized):

    mode = param.Selector(
        default=Mode.JSON_SCHEMA, objects=[Mode.JSON_SCHEMA, Mode.JSON, Mode.FUNCTIONS]
    )

    use_logfire = param.Boolean(default=False)

    # Allows defining a dictionary of default models.
    model_kwargs = param.Dict(default={})

    _supports_model_stream = True

    __abstract = True

    def _get_model_kwargs(self, model_key):
        if model_key in self.model_kwargs:
            model_kwargs = self.model_kwargs.get(model_key)
        else:
            model_kwargs = self.model_kwargs["default"]
        return dict(model_kwargs)

    @property
    def _client_kwargs(self):
        return {}

    async def invoke(
        self,
        messages: list | str,
        system: str = "",
        response_model: BaseModel | None = None,
        allow_partial: bool = False,
        model_key: str = "default",
        **input_kwargs,
    ) -> BaseModel:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if system:
            messages = [{"role": "system", "content": system}] + messages

        kwargs = dict(self._client_kwargs)
        kwargs.update(input_kwargs)

        if response_model is not None:
            if allow_partial:
                response_model = Partial[response_model]
            kwargs["response_model"] = response_model

        output = await self.run_client(model_key, messages, **kwargs)
        if output is None or output == "":
            raise ValueError("LLM failed to return valid output.")
        return output

    @classmethod
    def _get_delta(cls, chunk):
        if chunk.choices:
            return chunk.choices[0].delta.content or ""
        return ""

    async def stream(
        self,
        messages: list | str,
        system: str = "",
        response_model: BaseModel | None = None,
        field: str | None = None,
        model_key: str = "default",
        **kwargs,
    ):
        if response_model and not self._supports_model_stream:
            yield await self.invoke(
                messages,
                system=system,
                response_model=response_model,
                model_key=model_key,
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
            model_key=model_key,
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

    async def run_client(self, model_key, messages, **kwargs):
        client = self.get_client(model_key, **kwargs)
        return await client(messages=messages, **kwargs)


class Llama(Llm):

    chat_format = param.String(constant=True)

    temperature = param.Number(default=0.4, bounds=(0, None), constant=True)

    model_kwargs = param.Dict(default={
        "default": {
            "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "model_file": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
            "chat_format": "mistral-instruct",
        },
        "sql": {
            "repo": "TheBloke/sqlcoder2-GGUF",
            "model_file": "sqlcoder2.Q5_K_M.gguf",
            "chat_format": "chatml",
        },
    })

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    def get_client(self, model_key: str, response_model: BaseModel | None = None, **kwargs):
        if client_callable := pn.state.cache.get(model_key):
            return client_callable
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        model_kwargs = self._get_model_kwargs(model_key)
        repo = model_kwargs["repo"]
        model_file = model_kwargs["model_file"]
        chat_format = model_kwargs["chat_format"]
        llm = Llama(
            model_path=hf_hub_download(repo, model_file),
            n_gpu_layers=-1,
            n_ctx=8192,
            seed=128,
            chat_format=chat_format,
            logits_all=False,
            use_mlock=True,
            verbose=False
        )

        raw_client = llm.create_chat_completion_openai_v1
        # patch works with/without response_model
        client_callable = patch(
            create=raw_client,
            mode=Mode.JSON_SCHEMA,  # (2)!
        )
        pn.state.cache[model_key] = client_callable
        return client_callable

    async def run_client(self, model_key, messages, **kwargs):
        client = self.get_client(model_key, **kwargs)
        return await client(messages=messages, **kwargs)


class OpenAI(Llm):

    api_key = param.String()

    base_url = param.String()

    mode = param.Selector(default=Mode.FUNCTIONS)

    temperature = param.Number(default=0.2, bounds=(0, None), constant=True)

    organization = param.String()

    model_kwargs = param.Dict(default={
        "default": {"model": "gpt-4o-mini"},
        "reasoning": {"model": "gpt-4-turbo-preview"},
    })

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    def get_client(self, model_key: str, response_model: BaseModel | None = None, **kwargs):
        import openai

        model_kwargs = self._get_model_kwargs(model_key)
        model = model_kwargs.pop("model")
        if self.base_url:
            model_kwargs["base_url"] = self.base_url
        if self.api_key:
            model_kwargs["api_key"] = self.api_key
        if self.organization:
            model_kwargs["organization"] = self.organization
        llm = openai.AsyncOpenAI(**model_kwargs)
        if response_model:
            client = from_openai(llm)
            client_callable = partial(client.chat.completions.create, model=model)
        else:
            client_callable = partial(llm.chat.completions.create, model=model)

        if self.use_logfire:
            import logfire
            logfire.configure()
            logfire.instrument_openai(llm)
        return client_callable

class AzureOpenAI(Llm):

    api_key = param.String()

    api_version = param.String()

    azure_endpoint = param.String()

    mode = param.Selector(default=Mode.FUNCTIONS)

    temperature = param.Number(default=0.2, bounds=(0, None), constant=True)

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    def get_client(self, model_key: str, response_model: BaseModel | None = None, **kwargs):
        import openai

        model_kwargs = self._get_model_kwargs(model_key)
        model = model_kwargs.pop("model")
        if self.api_version:
            model_kwargs["api_version"] = self.api_version
        if self.api_key:
            model_kwargs["api_key"] = self.api_key
        if self.azure_endpoint:
            model_kwargs["azure_endpoint"] = self.azure_endpoint
        llm = openai.AsyncAzureOpenAI(**model_kwargs)
        if response_model:
            client = from_openai(llm)
            client_callable = partial(client.chat.completions.create, model=model)
        else:
            client_callable = partial(llm.chat.completions.create, model=model)
        return client_callable


class AILauncher(OpenAI):

    base_url = param.String(default="http://localhost:8080/v1")

    mode = param.Selector(default=Mode.JSON_SCHEMA)

    model_kwargs = param.Dict(default={
        "default": {"model": "gpt-3.5-turbo"},
        "reasoning": {"model": "gpt-4-turbo-preview"},
    })


class MistralAI(Llm):

    api_key = param.String(default=os.getenv("MISTRAL_API_KEY"))

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

    def get_client(self, model_key: str, response_model: BaseModel | None = None, **kwargs):
        from mistralai import Mistral

        async def llm_chat_non_stream_async(*args, **kwargs):
            response = await llm.chat.complete_async(*args, **kwargs)
            return response.choices[0].message.content

        model_kwargs = self._get_model_kwargs(model_key)
        model = model_kwargs.pop("model")

        llm = Mistral(api_key=self.api_key)
        if response_model:
            # can't use from_mistral due to new mistral API
            # https://github.com/jxnl/instructor/issues/969
            return patch(
                create=partial(llm.chat.complete_async, model=model),
                mode=self.mode,
            )

        stream = kwargs.get("stream", False)
        if stream:
            return partial(llm.chat.stream_async, model=model)
        else:
            return partial(llm_chat_non_stream_async, model=model)

    @classmethod
    def _get_delta(cls, chunk):
        if chunk.data.choices:
            return chunk.data.choices[0].delta.content or ""
        return ""

    async def invoke(
        self,
        messages: list | str,
        system: str = "",
        response_model: BaseModel | None = None,
        allow_partial: bool = False,
        model_key: str = "default",
        **input_kwargs,
    ) -> BaseModel:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if messages[0]["role"] == "assistant":
            # Mistral cannot start with assistant
            messages = messages[1:]

        return await super().invoke(
            messages,
            system,
            response_model,
            allow_partial,
            model_key,
            **input_kwargs,
        )



class AzureMistralAI(MistralAI):

    api_key = param.String(default=os.getenv("AZURE_API_KEY"))

    azure_endpoint = param.String(default=os.getenv("AZURE_ENDPOINT"))

    model_kwargs = param.Dict(default={
        "default": {"model": "azureai"},
    })

    def get_client(self, model_key: str, response_model: BaseModel | None = None, **kwargs):
        from mistralai_azure import MistralAzure

        async def llm_chat_non_stream_async(*args, **kwargs):
            response = await llm.chat.complete_async(*args, **kwargs)
            return response.choices[0].message.content

        model_kwargs = self._get_model_kwargs(model_key)
        model = model_kwargs.pop("model")

        llm = MistralAzure(azure_api_key=self.api_key, azure_endpoint=self.azure_endpoint)
        if response_model:
            return patch(
                create=partial(llm.chat.complete_async, model=model),
                mode=self.mode,
            )

        stream = kwargs.get("stream", False)
        if stream:
            return partial(llm.chat.stream_async, model=model)
        else:
            return partial(llm_chat_non_stream_async, model=model)
