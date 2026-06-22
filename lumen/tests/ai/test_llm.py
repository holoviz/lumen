"""Test suite for LLM implementations."""

import base64
import os

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

try:
    import lumen.ai as lmai

    from lumen.ai.agents.vega_lite import VegaLiteAgent
    from lumen.ai.llm import (
        Anthropic, AnthropicBedrock, AzureOpenAI, Google, Groq, Llm, Message,
        MistralAI, OpenAI,
    )

except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from instructor.processing.multimodal import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image() -> Image:
    """Create a tiny 1x1 PNG encoded as an instructor Image."""
    pixel = base64.b64encode(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
        b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
        b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    ).decode('utf-8')
    return Image.from_raw_base64(pixel)


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


# ---------------------------------------------------------------------------
# Anthropic prompt caching tests
# ---------------------------------------------------------------------------


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
