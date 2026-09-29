"""Tests for typed decision models and the Jev provider."""

import sys

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pydantic import ValidationError

from lumen.ai.decisions import (
    Choice, DecisionResult, Jev, Noul, Score,
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


def test_jev_missing_optional_sdk(monkeypatch):
    """Only require the TypeSafe SDK when invoking the Jev provider."""
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    with pytest.raises(ImportError, match="lumen\\[ai-typesafe\\]"):
        import asyncio
        asyncio.run(Jev().invoke("Hello", {"urgent": Noul(instructions="Urgent?")}))


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
