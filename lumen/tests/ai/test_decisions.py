"""Tests for typed decision models and the Jev and OpenRouter providers."""

import json
import sys

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import param
import pytest

from pydantic import ValidationError

from lumen.ai import decisions
from lumen.ai.decisions import (
    Choice, DecisionModel, DecisionResult, Jev, Noul, OpenRouterDecisionModel,
    Score,
)


@pytest.fixture
def typesafe_client(monkeypatch):
    """Provide a fake SDK without a network request or optional dependency."""
    client = SimpleNamespace(system_one=AsyncMock())

    class AsyncTypeSafeClient:
        def __init__(self, **kwargs):
            client.kwargs = kwargs
            self.system_one = client.system_one

        async def aclose(self):
            client.closed = True

    monkeypatch.setitem(sys.modules, "typesafe_sdk", SimpleNamespace(AsyncTypeSafeClient=AsyncTypeSafeClient))
    return client


async def test_jev_invoke_mixed_questions(typesafe_client):
    """Send all three question types in one call and preserve typed answers and usage."""
    typesafe_client.system_one.return_value = SimpleNamespace(model_dump=lambda: {
        "model": "jev-1.13.0",
        "answers": {
            "route": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "other": 0.2}, "confidence": 0.6},
            "urgent": {"type": "noul", "noul": 0.73},
            "tone": {"type": "score", "score": 1.2, "legend": {"0": "calm", "1": "angry"}, "probabilities": {"0": 0.1, "1": 0.9}, "confidence": 0.8},
        },
        "usage": {"input_tokens": 120, "output_tokens": 30, "cost": 0.000019},
    })
    model = Jev(api_key="test-key", timeout=5, model_kwargs={
        "default": {"model": "jev-latest"}, "fast": {"model": "jev-1.13.0"}
    })
    state = {"message": "Charged twice"}
    questions = {
        "route": Choice(instructions="Which team?", criteria={"billing": "Charges", "other": None}),
        "urgent": Noul(instructions="Urgent?", criteria={"true": "Time sensitive"}),
        "tone": Score(instructions="Tone?", criteria=["calm", "angry"]),
    }

    result = await model.invoke(state, questions, model_spec="fast")

    assert isinstance(result, DecisionResult)
    assert result.model == "jev-1.13.0"
    assert result.answers["route"].choice == "billing"
    assert result.answers["route"].probabilities == {"billing": 0.8, "other": 0.2}
    assert result.answers["urgent"].noul == 0.73
    assert result.answers["tone"].score == 1.2
    assert result.answers["tone"].legend == {"0": "calm", "1": "angry"}
    assert result.usage["input_tokens"] == 120
    assert result.usage["cost"] == 0.000019
    assert typesafe_client.kwargs == {"api_key": "test-key", "base_url": None, "timeout": 5}
    typesafe_client.system_one.assert_awaited_once_with(state, {
        "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "other": None}},
        "urgent": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "Time sensitive"}},
        "tone": {"type": "score", "instructions": "Tone?", "criteria": ["calm", "angry"]},
    }, model="jev-1.13.0")
    await model.aclose()
    assert typesafe_client.closed


async def test_jev_reuses_client(typesafe_client):
    """Reuse one SDK client for multiple questions sets and release it on close."""
    typesafe_client.system_one.return_value = SimpleNamespace(model_dump=lambda: {
        "model": "jev-1.13.0", "answers": {"urgent": {"type": "noul", "noul": 0.5}}, "usage": {}
    })
    model = Jev()

    await model.invoke("First", {"urgent": Noul(instructions="Urgent?")})
    client = model._base_client
    await model.invoke("Second", {"urgent": Noul(instructions="Urgent?")})

    assert model._base_client is client
    assert typesafe_client.system_one.await_count == 2
    await model.aclose()
    assert model._base_client is None
    assert typesafe_client.closed


async def test_jev_accepts_raw_question_dicts(typesafe_client):
    """Allow declarative question definitions without requiring SDK objects."""
    typesafe_client.system_one.return_value = SimpleNamespace(model_dump=lambda: {
        "model": "jev-1.13.0", "answers": {"urgent": {"type": "noul", "noul": 0.25}}, "usage": {}
    })

    result = await Jev().invoke("Hello", {"urgent": {"type": "noul", "instructions": "Urgent?"}})

    assert result.answers["urgent"].noul == 0.25
    typesafe_client.system_one.assert_awaited_once_with("Hello", {
        "urgent": {"type": "noul", "instructions": "Urgent?"}
    }, model="jev-latest")


async def test_jev_openrouter_configuration(typesafe_client):
    """Route Jev through OpenRouter's TypeSafe-compatible endpoint."""
    typesafe_client.system_one.return_value = SimpleNamespace(model_dump=lambda: {
        "model": "typesafe/jev-1.13-20260917", "answers": {"urgent": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }, raw_http_response=SimpleNamespace(json=lambda: {
        "usage": {"input_tokens": 100, "output_tokens": 20, "cost": 0.000019}
    }))
    model = Jev(
        api_key="openrouter-key", base_url="https://openrouter.ai/api",
        model_kwargs={"default": {"model": "~typesafe/jev-latest"}},
    )

    result = await model.invoke("Help", {"urgent": Noul(instructions="Urgent?")})

    assert result.usage["cost"] == 0.000019
    assert typesafe_client.kwargs["base_url"] == "https://openrouter.ai/api"
    typesafe_client.system_one.assert_awaited_once_with("Help", {
        "urgent": {"type": "noul", "instructions": "Urgent?"}
    }, model="~typesafe/jev-latest")


@pytest.mark.parametrize("question", [
    Noul(),
    Noul(instructions=None, criteria={"true": "Time sensitive", "false": None}),
    Choice(criteria={"billing": "Charges", "other": None}),
    Choice(instructions=None, criteria={"billing": "Charges", "other": None}),
    Score(criteria=["calm", "angry"]),
    {"type": "score", "instructions": None, "criteria": ["calm", "angry"]},
], ids=["noul", "noul-null", "choice", "choice-null", "score", "score-raw-null"])
async def test_jev_omits_missing_instructions(typesafe_client, question):
    """Send questions whose criterion carries the meaning without an instructions key."""
    typesafe_client.system_one.return_value = SimpleNamespace(model_dump=lambda: {
        "model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": {}
    })

    await Jev().invoke("Hello", {"q": question})

    payload = typesafe_client.system_one.await_args.args[1]["q"]
    assert "instructions" not in payload
    expected = question if isinstance(question, dict) else question.model_dump()
    assert payload.get("criteria") == expected["criteria"]


@pytest.mark.parametrize(("state", "questions", "error"), [
    ("text", {}, ValueError),
    (None, {"urgent": Noul(instructions="Urgent?")}, ValueError),
    ("text", {"urgent": {"type": "unknown", "instructions": "Urgent?"}}, ValidationError),
])
async def test_jev_rejects_invalid_requests_before_call(typesafe_client, state, questions, error):
    """Reject missing or malformed input before invoking the provider."""
    with pytest.raises(error):
        await Jev().invoke(state, questions)
    typesafe_client.system_one.assert_not_awaited()


def test_jev_requires_known_model_spec():
    """Reject unknown model specs rather than silently using the default."""
    model = Jev()
    with pytest.raises(ValueError, match="Unknown decision model spec"):
        model._get_model("missing")
    with pytest.raises(ValueError, match="default model"):
        Jev(model_kwargs={})


async def test_jev_missing_optional_sdk(monkeypatch):
    """Only require the TypeSafe SDK when invoking the Jev provider."""
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    with pytest.raises(ImportError, match="lumen\\[ai-typesafe\\]"):
        await Jev().invoke("Hello", {"urgent": Noul(instructions="Urgent?")})


async def test_decision_model_subclass_must_implement_invoke():
    """Fail loudly for an incomplete provider while still allowing teardown."""
    class Incomplete(DecisionModel):
        model_kwargs = param.Dict(default={"default": {"model": "custom"}})

    model = Incomplete()
    with pytest.raises(NotImplementedError, match="Incomplete must implement invoke"):
        await model.invoke("Hello", {"urgent": Noul()})
    await model.aclose()


@pytest.mark.parametrize("question", [
    Noul(), Choice(criteria={"billing": "Charges", "other": None}), Score(criteria=["calm", "angry"]),
], ids=["noul", "choice", "score"])
def test_questions_without_instructions_match_typesafe_sdk(question):
    """Keep the omitted-instructions wire form accepted by the SDK's own question models."""
    sdk = pytest.importorskip("typesafe_sdk")
    payload = question.model_dump(exclude_none=True)
    sdk_question = getattr(sdk, type(question).__name__).model_validate(payload)
    assert sdk_question.model_dump() == payload


def test_decision_result_accepts_typesafe_sdk_score():
    """Preserve integer score keys emitted by the SDK's parsed response."""
    sdk = pytest.importorskip("typesafe_sdk")
    response = sdk.SystemOneResponse.model_validate({
        "model": "jev-1.13.0",
        "answers": {
            "tone": {"type": "score", "score": 0.8, "legend": {0: "calm", 1: "angry"},
                     "probabilities": {0: 0.2, 1: 0.8}, "confidence": 0.6},
        },
        "usage": {"input_tokens": 10, "output_tokens": 2},
    })

    result = DecisionResult.model_validate(response.model_dump())

    assert result.answers["tone"].probabilities == {0: 0.2, 1: 0.8}


OPENROUTER_RESPONSE = {
    "id": "gen-dec-1", "model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe",
    "answers": {
        "team": {"type": "choice", "choice": "payments", "confidence": 0.75,
                 "probabilities": {"account": 0, "payments": 0.84, "frontend": 0.16}},
        "is_bug": {"type": "noul", "noul": 0.96},
        "urgency": {"type": "score", "score": 1.99, "confidence": 0.99, "legend": {"0": "Later", "1": "Soon", "2": "Now"},
                    "probabilities": {"0": 0, "1": 0.01, "2": 0.99}},
    },
    "usage": {"cost": 0.000019992, "input_tokens": 476, "output_tokens": 70},
}


@pytest.fixture
def openrouter(monkeypatch):
    """Serve the Decisions router from an in-process transport."""
    requests = []
    response = SimpleNamespace(status=200, body=OPENROUTER_RESPONSE)

    def handler(request):
        requests.append(request)
        return httpx.Response(response.status, json=response.body)

    client_class = httpx.AsyncClient
    monkeypatch.setattr(decisions.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(handler), **kwargs
    ))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return SimpleNamespace(requests=requests, response=response)


async def test_openrouter_decision_model_mixed_questions(openrouter):
    """Send every question in one Decisions request and keep the billed cost."""
    model = OpenRouterDecisionModel(api_key="test-key", timeout=5)
    state = {"ticket": "Checkout shows a blank screen after I click Pay."}

    result = await model.invoke(state, {
        "team": Choice(instructions="Which team?", criteria={"account": "Login", "payments": "Checkout", "frontend": None}),
        "is_bug": Noul(criteria={"true": "Broken behavior", "false": "A question"}),
        "urgency": Score(instructions="How urgent?", criteria=["Later", "Soon", "Now"]),
    })

    assert result.model == "typesafe/jev-1.13-20260917"
    assert result.answers["team"].choice == "payments"
    assert result.answers["is_bug"].noul == 0.96
    assert result.answers["urgency"].score == 1.99
    assert result.usage == {"cost": 0.000019992, "input_tokens": 476, "output_tokens": 70}
    request, = openrouter.requests
    assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "~typesafe/jev-latest",
        "state": state,
        "questions": {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"account": "Login", "payments": "Checkout", "frontend": None}},
            "is_bug": {"type": "noul", "criteria": {"true": "Broken behavior", "false": "A question"}},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Later", "Soon", "Now"]},
        },
    }
    await model.aclose()


async def test_openrouter_decision_model_spec_options_and_client_reuse(openrouter, monkeypatch):
    """Send extra spec keys such as provider preferences and reuse one client until closed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    model = OpenRouterDecisionModel(model_kwargs={
        "default": {"model": "~typesafe/jev-latest"},
        "pinned": {"model": "typesafe/jev-1.13", "provider": {"only": ["TypeSafe"]}},
    })

    await model.invoke("Help", {"is_bug": Noul()}, model_spec="pinned")
    client = model._base_client
    await model.invoke("Help", {"is_bug": Noul()})

    assert model._base_client is client
    first, second = (json.loads(request.content) for request in openrouter.requests)
    assert (first["model"], first["provider"]) == ("typesafe/jev-1.13", {"only": ["TypeSafe"]})
    assert second["model"] == "~typesafe/jev-latest" and "provider" not in second
    assert openrouter.requests[0].headers["Authorization"] == "Bearer env-key"
    await model.aclose()
    assert model._base_client is None and client.is_closed


async def test_openrouter_decision_model_errors(openrouter):
    """Surface missing credentials, unknown specs and provider errors."""
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        await OpenRouterDecisionModel().invoke("Help", {"is_bug": Noul()})
    model = OpenRouterDecisionModel(api_key="test-key")
    with pytest.raises(ValueError, match="Unknown decision model spec"):
        await model.invoke("Help", {"is_bug": Noul()}, model_spec="missing")
    openrouter.response.status, openrouter.response.body = 402, {"error": {"code": 402, "message": "Insufficient credits"}}
    with pytest.raises(httpx.HTTPStatusError, match="402"):
        await model.invoke("Help", {"is_bug": Noul()})
    await model.aclose()
