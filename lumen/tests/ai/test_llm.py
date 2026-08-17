"""Test suite for LLM implementations."""

import base64
import os

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

try:
    import lumen.ai as lmai

    from lumen.ai.agents.vega_lite import VegaLiteAgent
    from lumen.ai.llm import (
        MLX, Anthropic, AnthropicBedrock, AzureOpenAI, Bedrock, Google, Groq,
        LiteLLM, LlamaCpp, Llm, Message, MistralAI, Ollama, OpenAI, WebLLM,
    )
    from lumen.ai.tools import FunctionTool

except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from instructor.processing.multimodal import Image
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# (provider, required_init_kwargs, non-temperature keys retained in _client_kwargs, default temperature)
PROVIDER_SPECS = [
    pytest.param(OpenAI, {}, set(), 0.25, id="openai"),
    pytest.param(AzureOpenAI, {"api_version": "av", "endpoint": "ep"}, set(), 1, id="azure"),
    pytest.param(MistralAI, {}, set(), 0.7, id="mistral"),
    pytest.param(Anthropic, {}, {"max_tokens"}, 0.7, id="anthropic"),
    pytest.param(Bedrock, {}, {"maxTokens"}, 0.7, id="bedrock"),
    pytest.param(LlamaCpp, {}, set(), 0.4, id="llamacpp"),
    pytest.param(MLX, {}, {"max_tokens"}, 0.4, id="mlx"),
    pytest.param(LiteLLM, {}, set(), 0.7, id="litellm"),
    pytest.param(Ollama, {"api_key": "ollama"}, set(), 0.25, id="ollama"),
]

def _make_test_image() -> Image:
    """Create a tiny 1x1 PNG encoded as an instructor Image."""
    pixel = base64.b64encode(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
        b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
        b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    ).decode('utf-8')
    return Image.from_raw_base64(pixel)


def _make(provider, required, temperature="__default__"):
    # temperature is constant=True; the sentinel lets us build with the default too.
    kw = dict(required)
    kw["model_kwargs"] = {"default": {"model": "m"}}
    if temperature != "__default__":
        kw["temperature"] = temperature
    return provider(**kw)

# ---------------------------------------------------------------------------
# AzureOpenAI model kwargs tests
# ---------------------------------------------------------------------------

def test_api_key_env_var_defaults():
    """Each provider has the correct api_key_env_var class variable by default."""
    assert OpenAI.api_key_env_var == "OPENAI_API_KEY"
    assert Anthropic.api_key_env_var == "ANTHROPIC_API_KEY"
    assert Google.api_key_env_var == "GEMINI_API_KEY"
    assert MistralAI.api_key_env_var == "MISTRAL_API_KEY"


def test_api_key_populated_from_env_var(monkeypatch):
    """api_key is populated from api_key_env_var env var when not explicitly passed."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-4.1-mini"}})
    assert llm.api_key == "test-openai-key"


def test_explicit_api_key_takes_priority(monkeypatch):
    """Explicitly passed api_key takes priority over the env var."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    llm = OpenAI(api_key="explicit-key", model_kwargs={"default": {"model": "gpt-4.1-mini"}})
    assert llm.api_key == "explicit-key"


def test_api_key_env_var_override_on_class(monkeypatch):
    """Overriding api_key_env_var on the class causes api_key to be read from the new env var."""
    monkeypatch.setenv("MY_CUSTOM_KEY", "custom-key-value")
    original = OpenAI.api_key_env_var
    try:
        OpenAI.api_key_env_var = "MY_CUSTOM_KEY"
        llm = OpenAI(model_kwargs={"default": {"model": "gpt-4.1-mini"}})
        assert llm.api_key == "custom-key-value"
    finally:
        OpenAI.api_key_env_var = original


def test_api_key_env_var_override_on_subclass(monkeypatch):
    """Overriding api_key_env_var on a subclass doesn't affect the parent class."""
    monkeypatch.setenv("MY_SUBCLASS_KEY", "subclass-key-value")

    class MyOpenAI(OpenAI):
        api_key_env_var = "MY_SUBCLASS_KEY"

    llm = MyOpenAI(model_kwargs={"default": {"model": "gpt-4.1-mini"}})
    assert llm.api_key == "subclass-key-value"
    assert OpenAI.api_key_env_var == "OPENAI_API_KEY"


def test_api_key_none_when_env_var_unset(monkeypatch):
    """api_key is None when api_key_env_var is set but the env var is not."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-4.1-mini"}})
    assert llm.api_key is None


def test_api_key_from_modified_provider_env_vars(monkeypatch):
    """Modifying PROVIDER_ENV_VARS and api_key_env_var is reflected at instantiation."""
    monkeypatch.setitem(lmai.llm.PROVIDER_ENV_VARS, "openai", "MY_PATCHED_KEY")
    monkeypatch.setenv("MY_PATCHED_KEY", "patched-key-value")
    original = OpenAI.api_key_env_var
    try:
        OpenAI.api_key_env_var = lmai.llm.PROVIDER_ENV_VARS["openai"]
        llm = OpenAI(model_kwargs={"default": {"model": "gpt-4.1-mini"}})
        assert llm.api_key == "patched-key-value"
    finally:
        OpenAI.api_key_env_var = original


def test_get_available_llm_returns_local_provider_when_no_env_vars_set(monkeypatch):
    """get_available_llm returns a local provider when no cloud API keys are set."""
    for env_var in lmai.llm.PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    # Local providers (no env var required) should still be returned
    result = lmai.llm.get_available_llm()
    assert result is not None


def test_get_available_llm_returns_correct_provider(monkeypatch):
    """get_available_llm returns the provider whose env var is set."""
    for env_var in lmai.llm.PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert lmai.llm.get_available_llm() is Anthropic


def test_get_available_llm_respects_modified_provider_env_vars(monkeypatch):
    """get_available_llm picks up a modified PROVIDER_ENV_VARS entry."""
    for env_var in list(lmai.llm.PROVIDER_ENV_VARS.values()):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setitem(lmai.llm.PROVIDER_ENV_VARS, "openai", "MY_CUSTOM_OPENAI_KEY")
    monkeypatch.setenv("MY_CUSTOM_OPENAI_KEY", "custom-value")
    assert lmai.llm.get_available_llm() is OpenAI


async def test_azure_open_ai_get_model_kwargs():
    """Test that AzureOpenAI._get_model_kwargs merges instance config with model-specific kwargs."""
    model_kwargs = {
        "default": {"model": "d_m", "azure_ad_token_provider": "d_aatp"},
        "other": {"model": "r_m", "azure_ad_token_provider": "r_aatp"},
    }

    llm = AzureOpenAI(api_version="av", endpoint="ep", model_kwargs=model_kwargs)

    expected_default = {
        "model": "d_m",
        "azure_ad_token_provider": "d_aatp",
        "api_version": "av",
        "azure_endpoint": "ep",
    }
    assert llm._get_model_kwargs("default") == expected_default

    expected_other = {
        "model": "r_m",
        "azure_ad_token_provider": "r_aatp",
        "api_version": "av",
        "azure_endpoint": "ep",
    }
    assert llm._get_model_kwargs("other") == expected_other


async def test_azure_open_ai_get_model_kwargs_individual_models():
    """Test model-specific config overrides instance defaults.

    To support use case where models do not share api_version and endpoint.
    """
    model_kwargs = {
        "default": {
            "model": "d_m",
            "azure_ad_token_provider": "d_aatp",
            "api_version": "d_av",
            "azure_endpoint": "d_ep",
        },
        "other": {
            "model": "r_m",
            "azure_ad_token_provider": "r_aatp",
            "api_version": "r_av",
            "azure_endpoint": "r_ep",
        },
    }

    llm = AzureOpenAI(api_version="av", endpoint="ep", model_kwargs=model_kwargs)

    assert llm._get_model_kwargs("default") == model_kwargs["default"]
    assert llm._get_model_kwargs("other") == model_kwargs["other"]


def test_groq_registered_in_llm_providers():
    """Test that the Groq provider is registered in LLM_PROVIDERS."""
    assert "groq" in lmai.llm.LLM_PROVIDERS
    assert lmai.llm.LLM_PROVIDERS["groq"] == "Groq"


def test_groq_registered_in_provider_env_vars():
    """Test that the Groq provider env var is registered."""
    assert "groq" in lmai.llm.PROVIDER_ENV_VARS
    assert lmai.llm.PROVIDER_ENV_VARS["groq"] == "GROQ_API_KEY"


def test_groq_api_key_env_var():
    """Test that Groq has the correct api_key_env_var."""
    assert Groq.api_key_env_var == "GROQ_API_KEY"


def test_groq_defaults():
    """Test that Groq has the correct default endpoint and model."""
    groq = Groq(api_key="test-key")
    assert groq.endpoint == "https://api.groq.com/openai/v1"
    assert groq.model_kwargs["default"]["model"] == "llama-3.3-70b-versatile"


def test_get_available_llm_selects_groq(monkeypatch):
    """Test that get_available_llm() selects Groq when only GROQ_API_KEY is set."""
    for env_var in lmai.llm.PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert lmai.llm.get_available_llm() is Groq


def test_openai_get_delta_ignores_non_text_responses_events():
    """OpenAI._get_delta safely ignores non-text Responses stream events."""
    created_event = SimpleNamespace(type="response.created")
    assert OpenAI._get_delta(created_event) == ""


async def test_openai_responses_stream_tool_loop_uses_function_call_output(monkeypatch):
    """Responses stream recursion should pass function_call_output items, not tool_calls."""
    llm = OpenAI(api="responses", model_kwargs={"default": {"model": "gpt-4.1-mini"}})
    captured_calls: list[tuple[list[dict], dict]] = []
    call_count = 0

    async def fake_invoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_calls.append((messages, kwargs))
        if kwargs.get("stream"):
            if call_count == 1:
                return [
                    SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
                    SimpleNamespace(
                        type="response.function_call_arguments.done",
                        output_index=0,
                        item_id="item_1",
                        name="lookup",
                        arguments='{"query":"x"}',
                    ),
                ]
            return [SimpleNamespace(type="response.output_text.delta", delta="done")]
        return "done"

    async def fake_run_tool_calls(*args, **kwargs):
        return [{"role": "tool", "content": "{}", "name": "lookup", "tool_call_id": "call_1"}]

    def fake_normalize_tools(_tools):
        return (
            [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            {"lookup": object()},
            {},
        )

    monkeypatch.setattr(llm, "invoke", fake_invoke)
    monkeypatch.setattr(llm, "_run_tool_calls", fake_run_tool_calls)
    monkeypatch.setattr(llm, "_normalize_tools", fake_normalize_tools)

    outputs = []
    async for chunk in llm.stream(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    ):
        outputs.append(chunk)

    assert outputs[-1] == "done"
    second_messages, second_kwargs = captured_calls[1]
    assert second_messages[0]["type"] == "function_call_output"
    assert second_messages[0]["call_id"] == "call_1"
    assert "tool_calls" not in second_messages[0]
    assert second_kwargs["previous_response_id"] == "resp_1"


async def test_stream_keeps_streaming_when_tools_registered_but_unused(monkeypatch):
    """stream() should still yield deltas when tools are present but not called."""
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-4.1-mini"}})

    def fake_normalize_tools(_tools):
        return (
            [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            {"lookup": object()},
            {},
        )

    async def fake_run_client(_model_spec, _messages, **kwargs):
        assert kwargs.get("stream") is True
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="he", tool_calls=None))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="llo", tool_calls=None))]
            ),
        ]

    monkeypatch.setattr(llm, "_normalize_tools", fake_normalize_tools)
    monkeypatch.setattr(llm, "run_client", fake_run_client)

    outputs = []
    async for chunk in llm.stream(
        [{"role": "user", "content": "say hello"}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    ):
        outputs.append(chunk)

    assert outputs == ["he", "hello"]


def test_openai_responses_stream_tool_call_id_preserved_from_added_event():
    """Tool output call_id should come from function_call item (call_id), not item_id."""
    added_event = SimpleNamespace(
        type="response.output_item.added",
        output_index=0,
        item=SimpleNamespace(type="function_call", call_id="call_abc", name="lookup"),
    )
    done_event = SimpleNamespace(
        type="response.function_call_arguments.done",
        output_index=0,
        item_id="item_xyz",
        name="lookup",
        arguments='{"query":"x"}',
    )
    accum: dict[int, dict] = {}
    order: list[int] = []
    OpenAI._accumulate_tool_calls(accum, order, OpenAI._extract_stream_tool_calls(added_event))
    OpenAI._accumulate_tool_calls(accum, order, OpenAI._extract_stream_tool_calls(done_event))
    tool_calls = OpenAI._tool_calls_from_accum(accum, order)
    assert tool_calls[0]["id"] == "call_abc"


async def test_run_tool_loop_drops_max_retries_on_bare_client(monkeypatch):
    """Regression: ``max_retries`` is an instructor-only kwarg.

    When both ``response_model`` and ``tools`` are supplied (the planner's
    path), ``_run_tool_loop`` drops ``response_model`` and routes to the bare
    SDK client.  Before the fix, ``max_retries`` was left in ``kwargs`` and
    leaked through to ``AsyncMessages.create`` / ``AsyncCompletions.create``,
    raising ``TypeError: got an unexpected keyword argument 'max_retries'``.
    """

    class _Plan(BaseModel):
        next_step: str

    llm = OpenAI(model_kwargs={"default": {"model": "gpt-4.1-mini"}})
    calls: list[dict] = []

    async def fake_run_client(_model_spec, _messages, **kwargs):
        calls.append(dict(kwargs))
        return SimpleNamespace(content="done", tool_calls=None)

    monkeypatch.setattr(llm, "run_client", fake_run_client)
    monkeypatch.setattr(llm, "_extract_tool_calls", lambda _output: [])

    await llm._run_tool_loop(
        messages=[{"role": "user", "content": "hi"}],
        structured_model=_Plan,
        tool_instances={"lookup": object()},
        tool_contexts={},
        max_retries=3,
    )
    # The first call uses the bare client (no response_model). It must not
    # carry max_retries through, since the bare SDK rejects it.
    bare_call = calls[0]
    assert "response_model" not in bare_call
    assert "max_retries" not in bare_call, (
        "max_retries must be popped from kwargs on the bare-client path; "
        "it is consumed by the instructor wrapper, not the underlying SDK."
    )
    # Final structured-output call (instructor) should still carry the
    # user-passed max_retries so retry semantics are preserved end to end.
    final_call = calls[-1]
    assert final_call.get("response_model") is _Plan
    assert final_call.get("max_retries") == 3


# ---------------------------------------------------------------------------
# Adaptive request kwargs
# ---------------------------------------------------------------------------

def _bad_request(param: str) -> openai.BadRequestError:
    """The 400 OpenAI returns when a model rejects a request parameter."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(
        "rejected",
        response=httpx.Response(400, request=request),
        body={"message": "rejected", "type": "invalid_request_error", "param": param},
    )


def _capture_sends(monkeypatch, reject: list[str]) -> list[dict]:
    """Record every attempt, rejecting one parameter per entry in ``reject``."""
    calls: list[dict] = []
    rejections = list(reject)

    async def fake_send(self, _model_spec, _messages, **kwargs):
        calls.append(dict(kwargs))
        if rejections:
            raise _bad_request(rejections.pop(0))

    monkeypatch.setattr(OpenAI, "_send", fake_send)
    return calls


def test_openai_default_model_is_selectable():
    """The default has to appear in select_models: opening the settings dialog
    rewrites model_kwargs to select_models[0] when the current model is missing
    from the list, silently downgrading the default."""
    default_model = OpenAI.param.model_kwargs.default["default"]["model"]
    assert default_model == "gpt-5.6-luna"
    assert default_model in OpenAI.param.select_models.default


async def test_rejected_temperature_is_dropped_and_retried(monkeypatch):
    """gpt-5.6-luna only accepts the default temperature, which Lumen would
    otherwise send on every request."""
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-5.6-luna"}})
    calls = _capture_sends(monkeypatch, reject=["temperature"])

    # Start from _client_kwargs so this proves the shipped default is safe,
    # rather than a temperature invented by the test.
    await llm.run_client("default", [{"role": "user", "content": "hi"}], **llm._client_kwargs)

    assert calls[0]["temperature"] == 0.25
    assert "temperature" not in calls[1]


async def test_rejected_reasoning_effort_is_disabled_and_retried(monkeypatch):
    """Chat completions rejects function tools while reasoning is active; the
    API's own error names reasoning_effort as the parameter to set."""
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-5.6-luna"}})
    calls = _capture_sends(monkeypatch, reject=["temperature", "reasoning_effort"])

    await llm.run_client("default", [{"role": "user", "content": "hi"}], **llm._client_kwargs)

    assert "reasoning_effort" not in calls[1]
    assert calls[2]["reasoning_effort"] == "none"


async def test_rejection_wrapped_by_instructor_is_still_adapted(monkeypatch):
    """instructor re-raises provider errors inside its own retry exception, so
    the 400 has to be found on the cause chain rather than caught directly."""
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-5.6-luna"}})
    calls: list[dict] = []
    rejected = False

    async def fake_send(self, _model_spec, _messages, **kwargs):
        nonlocal rejected
        calls.append(dict(kwargs))
        if not rejected:
            rejected = True
            raise RuntimeError("instructor gave up") from _bad_request("temperature")

    monkeypatch.setattr(OpenAI, "_send", fake_send)

    await llm.run_client("default", [{"role": "user", "content": "hi"}], **llm._client_kwargs)

    assert "temperature" not in calls[1]


async def test_learned_fixes_are_reused_without_another_rejection(monkeypatch):
    """The adaptation is paid once per model, not on every request."""
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-5.6-luna"}})
    calls = _capture_sends(monkeypatch, reject=["temperature"])

    await llm.run_client("default", [{"role": "user", "content": "hi"}], **llm._client_kwargs)
    await llm.run_client("default", [{"role": "user", "content": "hi"}], **llm._client_kwargs)

    assert len(calls) == 3
    assert "temperature" not in calls[2]


@pytest.mark.parametrize(
    ("llm_factory", "model"),
    [
        (lambda model: OpenAI(model_kwargs={"default": {"model": model}}), "gpt-5.4-mini"),
        (lambda model: OpenAI(model_kwargs={"default": {"model": model}}), "gpt-5.4-nano"),
        (lambda model: Groq(api_key="k", model_kwargs={"default": {"model": model}}), "llama-3.3-70b-versatile"),
    ],
)
async def test_accepted_kwargs_are_left_alone(monkeypatch, llm_factory, model):
    """Models that accept sampling params must be left exactly as they were."""
    llm = llm_factory(model)
    calls = _capture_sends(monkeypatch, reject=[])

    await llm.run_client("default", [{"role": "user", "content": "hi"}], **llm._client_kwargs)

    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.25
    assert "reasoning_effort" not in calls[0]


async def test_unknown_rejection_is_raised(monkeypatch):
    """Only parameters Lumen knows how to satisfy are retried; anything else
    is the caller's problem and must not be swallowed."""
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-5.6-luna"}})
    calls = _capture_sends(monkeypatch, reject=["messages"])

    with pytest.raises(openai.BadRequestError):
        await llm.run_client("default", [{"role": "user", "content": "hi"}], **llm._client_kwargs)

    assert len(calls) == 1


async def test_explicit_create_kwargs_are_not_overridden(monkeypatch):
    """A deliberate create_kwargs choice surfaces its error rather than being
    silently rewritten."""
    llm = OpenAI(
        model_kwargs={"default": {"model": "gpt-5.6-luna"}},
        create_kwargs={"max_retries": 1, "reasoning_effort": "medium"},
    )
    calls = _capture_sends(monkeypatch, reject=["reasoning_effort"])

    with pytest.raises(openai.BadRequestError):
        await llm.run_client("default", [{"role": "user", "content": "hi"}], **llm._client_kwargs)

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# _normalize_multimodal_messages tests
# ---------------------------------------------------------------------------

class TestNormalizeMultimodalMessages:
    """Tests for Llm._normalize_multimodal_messages.

    When response_model is absent (e.g. during the tool-loop phase),
    the raw OpenAI client is used and cannot handle instructor Image
    objects.  _normalize_multimodal_messages converts them to
    OpenAI-native content-part dicts.
    """

    def test_standalone_image_converted(self):
        """A bare Image as content is wrapped in an image_url dict list."""
        img = _make_test_image()
        messages: list[Message] = [{"role": "user", "content": img}]
        result = Llm._normalize_multimodal_messages(messages)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "image_url"
        assert "base64," in content[0]["image_url"]["url"]

    def test_list_image_and_string_converted(self):
        """A [str, Image] list is converted to [{type: text}, {type: image_url}]."""
        img = _make_test_image()
        messages: list[Message] = [
            {"role": "user", "content": ["Describe this:", img]},
        ]
        result = Llm._normalize_multimodal_messages(messages)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Describe this:"}
        assert content[1]["type"] == "image_url"
        assert "base64," in content[1]["image_url"]["url"]

    def test_plain_text_unchanged(self):
        """Plain string content passes through untouched."""
        messages: list[Message] = [{"role": "user", "content": "Hello"}]
        result = Llm._normalize_multimodal_messages(messages)
        assert result[0]["content"] == "Hello"

    def test_already_normalized_dicts_unchanged(self):
        """Content that is already OpenAI-native dicts passes through."""
        messages: list[Message] = [{"role": "user", "content": [
            {"type": "text", "text": "Hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}]
        result = Llm._normalize_multimodal_messages(messages)
        content = result[0]["content"]
        assert content[0] == {"type": "text", "text": "Hi"}
        assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}

    def test_multiple_images_in_list(self):
        """Multiple Image objects in a list are all converted."""
        img1 = _make_test_image()
        img2 = _make_test_image()
        messages: list[Message] = [
            {"role": "user", "content": ["Compare:", img1, img2]},
        ]
        result = Llm._normalize_multimodal_messages(messages)
        content = result[0]["content"]
        assert len(content) == 3
        assert content[0] == {"type": "text", "text": "Compare:"}
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"

    def test_mixed_messages_only_image_ones_affected(self):
        """Non-image messages in the list are left alone."""
        img = _make_test_image()
        messages: list[Message] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": ["Chart:", img]},
        ]
        result = Llm._normalize_multimodal_messages(messages)
        assert result[0]["content"] == "You are helpful."
        assert result[1]["content"] == "Hello"
        assert result[2]["content"][0] == {"type": "text", "text": "Chart:"}
        assert result[2]["content"][1]["type"] == "image_url"


# ---------------------------------------------------------------------------
# _check_for_image tests
# ---------------------------------------------------------------------------

class TestCheckForImage:
    """Tests for Llm._check_for_image (detection + serialization)."""

    def test_plain_text_no_image(self, llm):
        """Plain text messages return False."""
        messages: list[Message] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        _, contains = llm._check_for_image(messages)
        assert not contains

    def test_single_image_detected(self, llm):
        """A bare Image in content is detected and serialized."""
        img = _make_test_image()
        messages: list[Message] = [{"role": "user", "content": img}]
        result, contains = llm._check_for_image(messages)
        assert contains
        assert isinstance(result[0]["content"], Image)

    def test_image_in_list_detected(self, llm):
        """An Image inside a list content is detected."""
        img = _make_test_image()
        messages: list[Message] = [
            {"role": "user", "content": ["Describe this chart:", img]},
        ]
        _, contains = llm._check_for_image(messages)
        assert contains

    def test_list_preserves_text_and_image(self, llm):
        """Mixed [str, Image] content keeps both parts after serialization."""
        img = _make_test_image()
        messages: list[Message] = [
            {"role": "user", "content": ["Describe this chart:", img]},
        ]
        result, contains = llm._check_for_image(messages)
        assert contains
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == "Describe this chart:"
        assert isinstance(content[1], Image)

    def test_list_without_images(self, llm):
        """A list of plain strings returns False."""
        messages: list[Message] = [
            {"role": "user", "content": ["part one", "part two"]},
        ]
        _, contains = llm._check_for_image(messages)
        assert not contains

    def test_multiple_images_in_list(self, llm):
        """Multiple images in one message are all serialized."""
        img1 = _make_test_image()
        img2 = _make_test_image()
        messages: list[Message] = [
            {"role": "user", "content": ["Compare:", img1, img2]},
        ]
        result, contains = llm._check_for_image(messages)
        assert contains
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 3
        assert content[0] == "Compare:"
        assert isinstance(content[1], Image)
        assert isinstance(content[2], Image)


# ---------------------------------------------------------------------------
# _prepare_vision_messages tests
# ---------------------------------------------------------------------------

class TestPrepareVisionMessages:
    """Tests for VegaLiteAgent._prepare_vision_messages fallback behavior."""

    @pytest.fixture
    def agent(self, llm):
        import warnings
        agent = VegaLiteAgent.__new__(VegaLiteAgent)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            agent.llm = llm
        return agent

    def test_no_vision_with_fallback(self, agent):
        """With text_fallback, appends text-only message when vision unsupported."""
        agent.llm._supports_vision = False
        msgs = [{"role": "user", "content": "hello"}]
        result = agent._prepare_vision_messages(msgs, None, "Annotate this")
        assert len(result) == 2
        assert result[1] == {"role": "user", "content": "Annotate this"}

    def test_no_editor_with_fallback(self, agent):
        """With text_fallback, appends text-only message when editor is None."""
        agent.llm._supports_vision = True
        msgs = [{"role": "user", "content": "hello"}]
        result = agent._prepare_vision_messages(msgs, None, "Annotate this")
        assert len(result) == 2
        assert result[1] == {"role": "user", "content": "Annotate this"}

    def test_image_export_fails_with_fallback(self, agent):
        """With text_fallback, appends text-only message when image export fails."""
        agent.llm._supports_vision = True
        mock_editor = MagicMock()
        mock_editor.__class__ = type("VegaLiteEditor", (), {})
        msgs = [{"role": "user", "content": "hello"}]
        with patch.object(agent, '_export_plot_image', return_value=None):
            with patch('lumen.ai.agents.vega_lite.VegaLiteEditor', mock_editor.__class__):
                result = agent._prepare_vision_messages(msgs, mock_editor, "Annotate this")
        assert len(result) == 2
        assert result[1] == {"role": "user", "content": "Annotate this"}

    def test_vision_success_returns_image_message(self, agent):
        """When vision succeeds, appends message with [content, Image]."""
        agent.llm._supports_vision = True
        mock_editor = MagicMock()
        mock_editor.__class__ = type("VegaLiteEditor", (), {})
        fake_png = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
            b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        msgs = [{"role": "user", "content": "hello"}]
        with patch.object(agent, '_export_plot_image', return_value=fake_png):
            with patch('lumen.ai.agents.vega_lite.VegaLiteEditor', mock_editor.__class__):
                result = agent._prepare_vision_messages(msgs, mock_editor, "Annotate this")
        assert len(result) == 2
        content = result[1]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == "Annotate this"
        assert isinstance(content[1], Image)


@pytest.mark.parametrize(
    "cls, cache, expected",
    [
        (Anthropic, "5m", {"type": "ephemeral"}),
        (Anthropic, "1h", {"type": "ephemeral", "ttl": "1h"}),
        (Anthropic, None, None),
        (AnthropicBedrock, "1h", None),
    ],
    ids=["5m", "1h", "off", "bedrock"],
)
def test_anthropic_cache_control(cls, cache, expected):
    llm = cls(model_kwargs={"default": {"model": "m"}}, cache=cache)
    assert llm._cache_control() == expected


def _all_llm_classes():
    """Recursively collect ``Llm`` and every subclass so new providers are
    covered automatically (``__subclasses__`` is not transitive)."""
    seen = {Llm}
    stack = [Llm]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return sorted(seen, key=lambda c: c.__name__)


@pytest.mark.parametrize("provider", _all_llm_classes(), ids=lambda c: c.__name__)
def test_temperature_param_allows_none(provider):
    assert provider.param["temperature"].allow_None is True


@pytest.mark.parametrize(
    ("provider", "required", "extra_keys", "default_temp"),
    PROVIDER_SPECS + [pytest.param(Google, {}, set(), 1, id="google")],
)
def test_construct_with_temperature_none(provider, required, extra_keys, default_temp):
    llm = _make(provider, required, temperature=None)
    assert llm.temperature is None


@pytest.mark.parametrize(("provider", "required", "extra_keys", "default_temp"), PROVIDER_SPECS)
def test_client_kwargs_omits_temperature_when_none(provider, required, extra_keys, default_temp):
    kwargs = _make(provider, required, temperature=None)._client_kwargs
    assert "temperature" not in kwargs
    assert set(kwargs) == extra_keys


@pytest.mark.parametrize("temperature", [0.5, 0.0], ids=["positive", "zero_boundary"])
@pytest.mark.parametrize(("provider", "required", "extra_keys", "default_temp"), PROVIDER_SPECS)
def test_client_kwargs_includes_temperature_when_set(provider, required, extra_keys, default_temp, temperature):
    kwargs = _make(provider, required, temperature=temperature)._client_kwargs
    assert kwargs["temperature"] == temperature
    assert extra_keys <= set(kwargs)


@pytest.mark.parametrize(("provider", "required", "extra_keys", "default_temp"), PROVIDER_SPECS)
def test_default_temperature_preserved(provider, required, extra_keys, default_temp):
    llm = _make(provider, required)
    assert llm.temperature == default_temp
    assert llm._client_kwargs["temperature"] == default_temp


def test_google_client_kwargs_unaffected_by_temperature():
    assert Google(model_kwargs={"default": {"model": "m"}}, temperature=None)._client_kwargs == {}
    assert Google(model_kwargs={"default": {"model": "m"}}, temperature=0.5)._client_kwargs == {}


def test_webllm_client_kwargs_never_includes_temperature():
    # WebLLM configures sampling on the panel_web_llm component, not via _client_kwargs.
    pytest.importorskip("panel_web_llm")
    assert WebLLM(model_kwargs={"default": {"model": "m"}}, temperature=None)._client_kwargs == {}
    assert WebLLM(model_kwargs={"default": {"model": "m"}}, temperature=0.5)._client_kwargs == {}


def test_mlx_make_sampler_handles_none():
    pytest.importorskip("mlx_lm.sample_utils")
    none_llm = MLX(model_kwargs={"default": {"model": "m"}}, temperature=None)
    set_llm = MLX(model_kwargs={"default": {"model": "m"}}, temperature=0.5)
    assert callable(none_llm._make_sampler())
    assert callable(set_llm._make_sampler())


def test_api_key_env_var_does_not_clobber_provider_default(monkeypatch):
    """An unset env var must leave the provider's own default alone.

    ``APIKeyServiceMixin`` assigned ``os.environ.get(api_key_env_var)``
    unconditionally, so a missing variable wrote None over the default. Ollama
    declares ``api_key="ollama"`` and does not accept None, which made
    ``lumen-ai serve --provider ollama`` fail at startup unless OPENAI_API_KEY
    happened to be set, despite Ollama needing no key of its own.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert Ollama().api_key == "ollama"
    assert Ollama(api_key="explicit").api_key == "explicit"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert OpenAI().api_key == "sk-from-env"


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

ROUTED_MODEL_KWARGS = {
    "default": {"model": "base-model"},
    "edit": {"model": "edit-model", "routing": {"model": "router-model"}, "description": "Best for editing tables and visualizations"},
    "sql": {"model": "sql-model"},
}


def _routing_llm():
    llm = _make(OpenAI, {})
    llm.model_kwargs = {k: dict(v) for k, v in ROUTED_MODEL_KWARGS.items()}
    return llm


def test_route_spec_model_constrains_to_current_model_kwargs_keys():
    """The Literal is rebuilt from model_kwargs at call time and rejects unknown keys."""
    llm = _routing_llm()
    route_spec_model = llm._route_spec_model()
    schema = route_spec_model.model_json_schema()["properties"]["model_spec"]
    assert set(schema["enum"]) == set(ROUTED_MODEL_KWARGS)
    assert route_spec_model(model_spec="sql").model_spec == "sql"
    with pytest.raises(ValidationError):
        route_spec_model(model_spec="bogus")


async def test_no_routing_key_passes_through(monkeypatch):
    """Without a 'routing' key the model_spec passes through unchanged and the
    routing model is never invoked."""
    llm = _routing_llm()

    async def fail_invoke(*args, **kwargs):
        raise AssertionError("routing model must not be invoked")  # pragma: no cover

    monkeypatch.setattr(llm, "invoke", fail_invoke)
    messages = [{"role": "user", "content": "hi"}]
    assert await llm._resolve_routing("sql", messages) == "sql"


async def test_routing_invoked_with_dict_spec_and_decision_used(monkeypatch):
    """When a 'routing' key is present, the routing model is invoked with a dict
    spec, a dedicated system prompt and no tools, and its decision resolves to
    the chosen entry's config dict."""
    llm = _routing_llm()
    calls = []

    async def fake_invoke(messages, **kwargs):
        calls.append((messages, kwargs))
        return SimpleNamespace(model_spec="sql")

    monkeypatch.setattr(llm, "invoke", fake_invoke)
    messages = [{"role": "user", "content": "hi"}]
    resolved = await llm._resolve_routing("edit", messages)
    assert resolved["model"] == "sql-model"
    assert "routing" not in resolved

    routed_messages, routed_kwargs = calls[0]
    assert routed_messages == messages
    assert routed_kwargs["model_spec"] == {"model": "router-model"}
    assert routed_kwargs["tools"] == []
    assert routed_kwargs["system"] == llm._routing_system_prompt()
    assert issubclass(routed_kwargs["response_model"], BaseModel)


async def test_routing_prompt_lists_options_with_descriptions(monkeypatch):
    """The routing call gets a dedicated system prompt listing every configured
    option with its optional description, and does not inherit the caller's
    system prompt."""
    llm = _routing_llm()
    llm.model_kwargs["edit"]["description"] = "Best for editing tasks"
    llm.model_kwargs["sql"]["description"] = "Best for SQL tasks"
    calls = []

    async def fake_invoke(messages, **kwargs):
        calls.append((messages, kwargs))
        return SimpleNamespace(model_spec="sql")

    monkeypatch.setattr(llm, "invoke", fake_invoke)
    messages = [
        {"role": "system", "content": "You are a data analyst."},
        {"role": "user", "content": "hi"},
    ]
    await llm._resolve_routing("edit", messages)

    routed_messages, routed_kwargs = calls[0]
    assert routed_kwargs["system"] == llm._routing_system_prompt()
    assert "Best for editing tasks" in routed_kwargs["system"]
    assert "Best for SQL tasks" in routed_kwargs["system"]
    assert routed_messages == [{"role": "user", "content": "hi"}]


async def test_routing_exception_falls_back(monkeypatch):
    """If the routing call raises, the fallback returns the original entry's
    config dict (not the bare string) so the dict-bypass prevents
    re-routing downstream."""
    llm = _routing_llm()

    async def fail_invoke(*args, **kwargs):
        raise RuntimeError("router down")

    monkeypatch.setattr(llm, "invoke", fail_invoke)
    messages = [{"role": "user", "content": "hi"}]
    result = await llm._resolve_routing("edit", messages)
    assert isinstance(result, dict)
    assert result["model"] == "edit-model"
    assert "routing" not in result
    assert "description" not in result


async def test_dict_model_spec_never_routed(monkeypatch):
    """A dict model_spec is never routed, even when its contents match keys used
    elsewhere in the config."""
    llm = _routing_llm()

    async def fail_invoke(*args, **kwargs):
        raise AssertionError("dict model_spec must not be routed")  # pragma: no cover

    monkeypatch.setattr(llm, "invoke", fail_invoke)
    messages = [{"role": "user", "content": "hi"}]
    spec = {"model": "router-model", "routing": {"model": "other"}}
    assert await llm._resolve_routing(spec, messages) == spec


def test_combine_tools_empty_list_opts_out_of_instance_tools():
    """tools=[] opts out of instance tools, while None uses them and a non-empty
    list merges them."""
    llm = _make(OpenAI, {})
    llm.tools = ["instance-tool"]
    assert llm._combine_tools(None) == ["instance-tool"]
    assert llm._combine_tools([]) == []
    assert llm._combine_tools(["per-call"]) == ["instance-tool", "per-call"]


def test_invalid_routing_config_rejected_at_construction():
    """A non-dict 'routing' entry fails loudly at construction time, not silently
    at call time."""
    with pytest.raises(ValueError, match="routing"):
        OpenAI(api_key="sk-test", model_kwargs={
            "default": {"model": "base"},
            "edit": {"model": "edit-model", "routing": "router-model"},
        })


async def test_get_client_strips_routing_key(monkeypatch):
    """The 'routing' key must never reach the SDK constructor (reviewer repro:
    AsyncOpenAI.__init__ must not receive an unexpected 'routing' kwarg)."""
    fake = MagicMock()
    monkeypatch.setattr(openai, "AsyncOpenAI", fake)
    llm = OpenAI(
        api_key="sk-test",
        model_kwargs={k: dict(v) for k, v in ROUTED_MODEL_KWARGS.items()},
    )
    client = await llm.get_client("edit")
    assert client is not None
    assert "routing" not in fake.call_args.kwargs


async def test_get_client_strips_description_key(monkeypatch):
    """The 'description' key must never reach the SDK constructor — it is
    routing metadata only."""
    fake = MagicMock()
    monkeypatch.setattr(openai, "AsyncOpenAI", fake)
    llm = OpenAI(
        api_key="sk-test",
        model_kwargs={k: dict(v) for k, v in ROUTED_MODEL_KWARGS.items()},
    )
    client = await llm.get_client("edit")
    assert client is not None
    assert "description" not in fake.call_args.kwargs


async def test_invoke_uses_routed_model_spec(monkeypatch):
    """invoke() resolves routing through the real path: the routing model runs
    first (with a dedicated prompt and no tools), then the resolved config
    reaches run_client with no 'routing' key."""
    llm = _routing_llm()
    run_client_specs = []

    async def fake_run_client(model_spec, messages, **kwargs):
        run_client_specs.append(dict(model_spec))
        if kwargs.get("response_model") is not None:
            return kwargs["response_model"](model_spec="sql")
        return "done"

    monkeypatch.setattr(llm, "run_client", fake_run_client)

    output = await llm.invoke([{"role": "user", "content": "hi"}], model_spec="edit")
    assert output == "done"
    assert [spec["model"] for spec in run_client_specs] == ["router-model", "sql-model"]
    assert all("routing" not in spec for spec in run_client_specs)


def _stream_chunks(text: str = "", tool_calls: list[dict] | None = None):
    async def gen():
        if tool_calls:
            for call in tool_calls:
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="", tool_calls=[call]))]
                )
        else:
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))]
            )

    return gen()


async def test_stream_resolves_routing_per_invoke_not_per_stream(monkeypatch):
    """Routing is resolved inside invoke(), not stream(). Both success and
    fallback paths return a dict, so the dict-bypass mechanism prevents
    re-routing across tool-loop rounds."""
    llm = _routing_llm()
    router_invocations = []
    stream_specs = []

    def add(a: int, b: int) -> int:
        """Adds two integers."""
        return a + b

    tool = FunctionTool(add)

    async def fake_run_client(model_spec, messages, **kwargs):
        spec = dict(model_spec) if isinstance(model_spec, dict) else model_spec
        if kwargs.get("response_model") is not None:
            router_invocations.append(spec)
            return kwargs["response_model"](model_spec="sql")
        stream_specs.append(spec)
        if kwargs.get("stream"):
            if len(stream_specs) == 1:
                return _stream_chunks(tool_calls=[
                    {"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": "add", "arguments": ""}},
                    {"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": None, "arguments": '{"a": 1, "b": 2}'}},
                ])
            return _stream_chunks(text="final answer")
        return "done"

    monkeypatch.setattr(llm, "run_client", fake_run_client)
    messages = [{"role": "user", "content": "hi"}]
    outputs = [out async for out in llm.stream(messages, model_spec="edit", tools=[tool])]

    assert len(router_invocations) == 1
    assert [spec["model"] for spec in router_invocations] == ["router-model"]
    assert [spec["model"] for spec in stream_specs] == ["sql-model", "sql-model"]
    assert "final answer" in outputs


async def test_stream_router_down_single_timeout(monkeypatch):
    """When the router is unreachable, _resolve_routing returns a dict
    (the fallback config) so the dict-bypass prevents re-routing on
    every subsequent call — router-down pays exactly one routing attempt
    per turn, not one per stream+invoke pair nor one per tool round."""
    llm = _routing_llm()
    routing_calls = []

    async def fake_run_client(model_spec, messages, **kwargs):
        if kwargs.get("response_model") is not None:
            routing_calls.append(model_spec)
            raise RuntimeError("router down")
        return "done"

    monkeypatch.setattr(llm, "run_client", fake_run_client)
    messages = [{"role": "user", "content": "hi"}]
    output = await llm.invoke(messages, model_spec="edit")
    assert output == "done"
    # Exactly one routing attempt — the fallback dict is returned and
    # used directly, so invoke() never tries routing a second time.
    assert len(routing_calls) == 1
    assert routing_calls[0] == {"model": "router-model"}


async def test_stream_router_down_no_reroute_across_rounds(monkeypatch):
    """When routing fails during stream(), the fallback dict prevents
    re-routing on tool-loop recursion — stream rounds use the resolved
    model directly without paying another routing timeout."""
    llm = _routing_llm()
    routing_attempts = []
    stream_specs = []

    def add(a: int, b: int) -> int:
        """Adds two integers."""
        return a + b

    tool = FunctionTool(add)

    async def fake_run_client(model_spec, messages, **kwargs):
        spec = dict(model_spec) if isinstance(model_spec, dict) else model_spec
        if kwargs.get("response_model") is not None:
            routing_attempts.append(spec)
            raise RuntimeError("router down")
        stream_specs.append(spec)
        if kwargs.get("stream"):
            if len(stream_specs) == 1:
                return _stream_chunks(tool_calls=[
                    {"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": "add", "arguments": ""}},
                    {"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": None, "arguments": '{"a": 1, "b": 2}'}},
                ])
            return _stream_chunks(text="fallback answer")
        return "done"

    monkeypatch.setattr(llm, "run_client", fake_run_client)
    messages = [{"role": "user", "content": "hi"}]
    outputs = [out async for out in llm.stream(messages, model_spec="edit", tools=[tool])]

    assert len(routing_attempts) == 1
    assert routing_attempts[0] == {"model": "router-model"}
    assert all(spec["model"] == "edit-model" for spec in stream_specs)
    assert all("routing" not in spec for spec in stream_specs)
    assert "fallback answer" in outputs
