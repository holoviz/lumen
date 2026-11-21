from __future__ import annotations

import os

from typing import Annotated, Any, Literal

import httpx
import param

from pydantic import BaseModel, Field, TypeAdapter

from .services import PROVIDER_ENV_VARS


class Noul(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: str | dict | list | None = None
    criteria: dict[str, Any] | None = None


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str | dict | list | None = None
    criteria: dict[str, Any]


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: str | dict | list | None = None
    criteria: list[str | dict | list]


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[int | str, str | dict | list]
    probabilities: dict[int | str, float]
    confidence: float = Field(ge=0, le=1)


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class DecisionResult(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: dict[str, int | float | None]


_questions = TypeAdapter(dict[str, Question])


class DecisionModel(param.Parameterized):
    """Provider-independent interface for typed decisions over shared state."""

    model_kwargs = param.Dict(default={}, doc="Model configurations indexed by spec name, including 'default'.")

    timeout = param.Number(default=120, bounds=(1, None), doc="API timeout in seconds.")

    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        if not self.model_kwargs.get("default", {}).get("model"):
            raise ValueError("model_kwargs must define a default model.")
        self._base_client = None

    def _resolve_spec(self, model_spec: str | None) -> str:
        if model_spec is None:
            model_spec = "default"
        if model_spec not in self.model_kwargs:
            raise ValueError(f"Unknown decision model spec: {model_spec!r}.")
        return model_spec

    def _get_model(self, model_spec: str | None) -> str:
        return self.model_kwargs[self._resolve_spec(model_spec)]["model"]

    @staticmethod
    def _parse_questions(state: str | dict | list, questions: dict[str, Question | dict]) -> dict[str, Question]:
        if not questions:
            raise ValueError("At least one decision question is required.")
        if state is None:
            raise ValueError("Decision state cannot be None.")
        return _questions.validate_python(questions)

    async def aclose(self):
        """Release any provider resources held by this model."""

    async def invoke(
        self, state: str | dict | list, questions: dict[str, Question | dict], *, model_spec: str | None = None
    ) -> DecisionResult:
        """Answer all questions against a single shared state."""
        raise NotImplementedError(f"{type(self).__name__} must implement invoke().")


class Jev(DecisionModel):
    """TypeSafe System One provider. Requires the optional typesafe-sdk package."""

    api_key = param.String(default=None, allow_None=True, doc="TypeSafe API key; defaults to TYPESAFE_API_KEY.")

    base_url = param.String(default=None, allow_None=True, doc="Optional TypeSafe-compatible API root.")

    model_kwargs = param.Dict(default={"default": {"model": "jev-latest"}})

    def _create_base_client(self):
        try:
            from typesafe_sdk import AsyncTypeSafeClient
        except ImportError as e:
            raise ImportError("Install lumen[ai-typesafe] to use Jev.") from e
        return AsyncTypeSafeClient(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    async def aclose(self):
        """Release the cached provider client and its network resources."""
        if self._base_client is not None:
            await self._base_client.aclose()
            self._base_client = None

    async def invoke(
        self, state: str | dict | list, questions: dict[str, Question | dict], *, model_spec: str | None = None
    ) -> DecisionResult:
        parsed = self._parse_questions(state, questions)
        model = self._get_model(model_spec)
        if self._base_client is None:
            self._base_client = self._create_base_client()
        response = await self._base_client.system_one(
            state, {name: question.model_dump(exclude_none=True) for name, question in parsed.items()},
            model=model,
        )
        result = response.model_dump()
        if raw_response := getattr(response, "raw_http_response", None):
            result["usage"].update(raw_response.json().get("usage", {}))
        return DecisionResult.model_validate(result)


class OpenRouterDecisionModel(DecisionModel):
    """
    Decisions through OpenRouter's Decisions router.

    Uses the same question schema as Jev without requiring the TypeSafe SDK.
    Keys of a ``model_kwargs`` entry other than ``model``, such as
    ``provider`` routing preferences, are sent with the request.
    """

    api_key = param.String(default=None, allow_None=True, doc="OpenRouter API key; defaults to OPENROUTER_API_KEY.")

    base_url = param.String(default="https://openrouter.ai/api", doc="OpenRouter API root.")

    model_kwargs = param.Dict(default={"default": {"model": "~typesafe/jev-latest"}})

    def _create_base_client(self) -> httpx.AsyncClient:
        api_key = self.api_key or os.environ.get(PROVIDER_ENV_VARS["openrouter"])
        if not api_key:
            raise ValueError(f"Set {PROVIDER_ENV_VARS['openrouter']} or pass api_key to use OpenRouterDecisionModel.")
        return httpx.AsyncClient(
            base_url=self.base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=self.timeout
        )

    async def aclose(self):
        """Release the cached HTTP client."""
        if self._base_client is not None:
            await self._base_client.aclose()
            self._base_client = None

    async def invoke(
        self, state: str | dict | list, questions: dict[str, Question | dict], *, model_spec: str | None = None
    ) -> DecisionResult:
        parsed = self._parse_questions(state, questions)
        config = self.model_kwargs[self._resolve_spec(model_spec)]
        if self._base_client is None:
            self._base_client = self._create_base_client()
        response = await self._base_client.post("/alpha/decisions", json={
            **config, "state": state,
            "questions": {name: question.model_dump(exclude_none=True) for name, question in parsed.items()},
        })
        response.raise_for_status()
        return DecisionResult.model_validate(response.json())
