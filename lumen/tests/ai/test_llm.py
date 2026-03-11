"""Test suite for LLM implementations."""

import base64

import pytest

try:
    from lumen.ai.llm import AzureOpenAI, Llm, Message  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from instructor.processing.multimodal import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image() -> Image:
    """Create a tiny 1x1 PNG encoded as an instructor Image."""
    # Minimal valid 1x1 white PNG
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

async def test_azure_open_ai_get_model_kwargs():
    """Test that AzureOpenAI._get_model_kwargs merges instance config with model-specific kwargs."""
    model_kwargs = {
        "default": {"model": "d_m", "azure_ad_token_provider": "d_aatp"},
        "other": {"model": "r_m", "azure_ad_token_provider": "r_aatp"},
    }

    llm = AzureOpenAI(api_version="av", endpoint="ep", model_kwargs=model_kwargs)

    # Test default model inherits instance config
    expected_default = {
        "model": "d_m",
        "azure_ad_token_provider": "d_aatp",
        "api_version": "av",
        "azure_endpoint": "ep",
    }
    assert llm._get_model_kwargs("default") == expected_default

    # Test other model inherits instance config
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

    # Model-specific config should override instance defaults
    assert llm._get_model_kwargs("default") == model_kwargs["default"]
    assert llm._get_model_kwargs("other") == model_kwargs["other"]


# ---------------------------------------------------------------------------
# _check_for_image tests
# ---------------------------------------------------------------------------

class TestCheckForImage:
    """Tests for Llm._check_for_image (pure detection, no mutation)."""

    def test_plain_text_no_image(self, llm):
        """Plain text messages return False."""
        messages: list[Message] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        assert not llm._check_for_image(messages)

    def test_single_image_detected(self, llm):
        """A bare Image in content is detected."""
        img = _make_test_image()
        messages: list[Message] = [{"role": "user", "content": img}]
        assert llm._check_for_image(messages)

    def test_image_in_list_detected(self, llm):
        """An Image inside a list content is detected."""
        img = _make_test_image()
        messages: list[Message] = [
            {"role": "user", "content": ["Describe this chart:", img]},
        ]
        assert llm._check_for_image(messages)

    def test_list_without_images(self, llm):
        """A list of plain strings returns False."""
        messages: list[Message] = [
            {"role": "user", "content": ["part one", "part two"]},
        ]
        assert not llm._check_for_image(messages)

    def test_does_not_mutate_messages(self, llm):
        """_check_for_image must not modify the original messages."""
        img = _make_test_image()
        messages: list[Message] = [
            {"role": "user", "content": ["text", img]},
        ]
        llm._check_for_image(messages)
        # Original content should still have the raw Image object
        assert messages[0]["content"][0] == "text"
        assert messages[0]["content"][1] is img


# ---------------------------------------------------------------------------
# _prepare_image_messages tests
# ---------------------------------------------------------------------------

class TestPrepareImageMessages:
    """Tests for Llm._prepare_image_messages (serialization without mutation)."""

    def test_single_image_serialized(self, llm):
        """A bare Image is serialized into an instructor Image."""
        img = _make_test_image()
        messages: list[Message] = [{"role": "user", "content": img}]
        result = llm._prepare_image_messages(messages)
        assert isinstance(result[0]["content"], Image)

    def test_list_content_preserves_text_and_image(self, llm):
        """Mixed [str, Image] content keeps both parts."""
        img = _make_test_image()
        messages: list[Message] = [
            {"role": "user", "content": ["Describe this chart:", img]},
        ]
        result = llm._prepare_image_messages(messages)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == "Describe this chart:"
        assert isinstance(content[1], Image)

    def test_multiple_images_in_list(self, llm):
        """Multiple images in one message are all serialized."""
        img1 = _make_test_image()
        img2 = _make_test_image()
        messages: list[Message] = [
            {"role": "user", "content": ["Compare:", img1, img2]},
        ]
        result = llm._prepare_image_messages(messages)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 3
        assert content[0] == "Compare:"
        assert isinstance(content[1], Image)
        assert isinstance(content[2], Image)

    def test_plain_text_passed_through(self, llm):
        """Plain text messages are returned unchanged."""
        messages: list[Message] = [{"role": "user", "content": "Hello"}]
        result = llm._prepare_image_messages(messages)
        assert result[0]["content"] == "Hello"

    def test_does_not_mutate_original(self, llm):
        """_prepare_image_messages returns new messages, not modified originals."""
        img = _make_test_image()
        original: list[Message] = [
            {"role": "user", "content": ["text", img]},
        ]
        result = llm._prepare_image_messages(original)
        # Original should still have the raw Image
        assert original[0]["content"][1] is img
        # Result should have serialized Image
        assert isinstance(result[0]["content"][1], Image)


# ---------------------------------------------------------------------------
# _strip_images tests
# ---------------------------------------------------------------------------

class TestStripImages:
    """Tests for Llm._strip_images fallback for non-vision providers."""

    def test_plain_text_unchanged(self, llm):
        """Plain string content passes through."""
        messages = [{"role": "user", "content": "Hello"}]
        result = llm._strip_images(messages)
        assert result[0]["content"] == "Hello"

    def test_bare_image_replaced(self, llm):
        """A bare Image is replaced with placeholder text."""
        img = _make_test_image()
        messages = [{"role": "user", "content": img}]
        result = llm._strip_images(messages)
        assert result[0]["content"] == "(image omitted)"

    def test_list_with_image_keeps_text(self, llm):
        """Text parts are preserved when images are stripped."""
        img = _make_test_image()
        messages = [
            {"role": "user", "content": ["Describe this chart:", img]},
        ]
        result = llm._strip_images(messages)
        assert result[0]["content"] == "Describe this chart:"

    def test_list_with_only_images(self, llm):
        """A list containing only images becomes a placeholder."""
        img = _make_test_image()
        messages = [{"role": "user", "content": [img]}]
        result = llm._strip_images(messages)
        assert result[0]["content"] == "(image omitted)"

    def test_multiple_text_parts_joined(self, llm):
        """Multiple text parts are joined with newlines."""
        img = _make_test_image()
        messages = [
            {"role": "user", "content": ["First part", img, "Second part"]},
        ]
        result = llm._strip_images(messages)
        assert result[0]["content"] == "First part\nSecond part"

    def test_preserves_role_and_other_fields(self, llm):
        """Non-content fields are preserved after stripping."""
        img = _make_test_image()
        messages = [
            {"role": "assistant", "content": ["text", img], "name": "bot"},
        ]
        result = llm._strip_images(messages)
        assert result[0]["role"] == "assistant"
        assert result[0]["name"] == "bot"
        assert result[0]["content"] == "text"

    def test_mixed_messages(self, llm):
        """A conversation with both plain and multimodal messages."""
        img = _make_test_image()
        messages = [
            {"role": "user", "content": "What is this?"},
            {"role": "assistant", "content": "Let me look."},
            {"role": "user", "content": ["Current chart:", img]},
        ]
        result = llm._strip_images(messages)
        assert result[0]["content"] == "What is this?"
        assert result[1]["content"] == "Let me look."
        assert result[2]["content"] == "Current chart:"

    def test_does_not_mutate_original(self, llm):
        """_strip_images returns new messages, not modified originals."""
        img = _make_test_image()
        original = [{"role": "user", "content": ["text", img]}]
        result = llm._strip_images(original)
        # Original should still have the list with image
        assert isinstance(original[0]["content"], list)
        assert isinstance(original[0]["content"][1], Image)
        # Result should be stripped
        assert result[0]["content"] == "text"
