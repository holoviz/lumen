from __future__ import annotations

import panel as pn
import param

from instructor.dsl.partial import Partial
from instructor.patch import Mode, patch
from pydantic import BaseModel


class Llm(param.Parameterized):

    mode = param.Selector(
        default=Mode.JSON_SCHEMA, objects=[Mode.JSON_SCHEMA, Mode.JSON, Mode.FUNCTIONS]
    )

    retry = param.Integer(default=2)

    # Allows defining a dictionary of default models.
    _models = {}

    __abstract = True

    def __init__(self, model: str | None = None, **params):
        if model is not None:
            if model in self._models:
                params = dict(self._models[model], **params)
            else:
                raise ValueError(
                    f"No model named {model!r} available. Known models include "
                    f"{', '.join(self._models)}."
                )
        super().__init__(**params)
        self._client = None

    def _create_client(self, create):
        return patch(create=create, mode=self.mode)

    @property
    def _client_kwargs(self):
        return {}

    async def invoke(
        self,
        messages: list | str,
        system: str = "",
        response_model: BaseModel | None = None,
        allow_partial: bool = True,
        **input_kwargs,
    ) -> BaseModel:
        if self._client is None:
            self._init_model()

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if system:
            messages = [{"role": "system", "content": system}] + messages

        kwargs = dict(self._client_kwargs)
        kwargs.update(input_kwargs)
        client = self._client

        if response_model is not None:
            if allow_partial:
                response_model = Partial[response_model]
            kwargs["response_model"] = response_model

        output = None
        for r in range(self.retry):
            try:
                output = await client(messages=messages, **kwargs)
                break
            except Exception as e:
                print(f"Error encountered: {e}")
                if 'response_model' in kwargs:
                    kwargs['response_model'] = response_model
                messages = messages + [{"role": "system", "content": f"You just encountered the following error, make sure you don't repeat it: {e}" }]
        print(f"Invoked LLM output: {output!r}")
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
        field: str = "output",
        **kwargs,
    ):
        if self._client is None:
            self._init_model()

        string = ""
        async for chunk in await self.invoke(
            messages,
            system=system,
            response_model=response_model,
            stream=True,
            **dict(self._client_kwargs, **kwargs),
        ):
            if response_model is None:
                delta = self._get_delta(chunk)
                string += delta
                yield string
            else:
                yield getattr(chunk, field)


class Llama(Llm):

    chat_format = param.String(constant=True)

    repo = param.String(constant=True)

    model_file = param.String(constant=True)

    temperature = param.Number(default=0.4, bounds=(0, None), constant=True)

    _models = {
        "default": {
            "repo": "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "chat_format": "chatml",
        },
        "mistral": {
            "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "model_file": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
            "chat_format": "mistral-instruct",
        },
        "sql": {
            "repo": "TheBloke/sqlcoder2-GGUF",
            "model_file": "sqlcoder2.Q5_K_M.gguf",
            "chat_format": "chatml",
        },
    }

    @property
    def _client_kwargs(self):
        return {"temperature": self.temperature}

    def _init_model(self):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        self._model = pn.state.as_cached(
            'Llama',
            Llama,
            model_path=hf_hub_download(self.repo, self.model_file),
            n_gpu_layers=-1,
            n_ctx=8192,
            seed=128,
            chat_format=self.chat_format,
            logits_all=False,
            use_mlock=True,
            verbose=False
        )
        self._raw_client = self._model.create_chat_completion_openai_v1
        self._client = self._create_client(self._raw_client)


class OpenAI(Llm):

    api_key = param.String()

    base_url = param.String()

    model_name = param.String()

    mode = param.Selector(default=Mode.FUNCTIONS)

    temperature = param.Number(default=0.2, bounds=(0, None), constant=True)

    organization = param.String()

    _models = {
        "gpt-3.5-turbo": {"model_name": "gpt-3.5-turbo"},
        "gpt-4": {"model_name": "gpt-4"},
        "gpt-4-turbo-preview": {"model_name": "gpt-4-turbo-preview"},
    }

    @property
    def _client_kwargs(self):
        return {"model": self.model_name, "temperature": self.temperature}

    def _init_model(self):
        import openai

        model_kwargs = {}
        if self.base_url:
            model_kwargs["base_url"] = self.base_url
        if self.api_key:
            model_kwargs["api_key"] = self.api_key
        if self.organization:
            model_kwargs["organization"] = self.organization
        self._model = openai.AsyncOpenAI(**model_kwargs)
        self._raw_client = self._model.chat.completions.create
        self._client = self._create_client(self._raw_client)


class AzureOpenAI(Llm):

    api_key = param.String()

    api_version = param.String()

    azure_endpoint = param.String()

    mode = param.Selector(default=Mode.FUNCTIONS)

    model_name = param.String()

    temperature = param.Number(default=0.2, bounds=(0, None), constant=True)

    @property
    def _client_kwargs(self):
        return {"model": self.model_name, "temperature": self.temperature}

    def _init_model(self):
        import openai

        model_kwargs = {}
        if self.api_version:
            model_kwargs["api_version"] = self.api_version
        if self.api_key:
            model_kwargs["api_key"] = self.api_key
        if self.azure_endpoint:
            model_kwargs["azure_endpoint"] = self.azure_endpoint
        self._model = openai.AsyncAzureOpenAI(**model_kwargs)
        self._raw_client = self._model.chat.completions.create
        self._client = self._create_client(self._raw_client)


class AILauncher(OpenAI):

    base_url = param.String(default="http://localhost:8080/v1")

    mode = param.Selector(default=Mode.JSON_SCHEMA)

    _models = {"default": {"model_name": "gpt-3.5-turbo"}}
