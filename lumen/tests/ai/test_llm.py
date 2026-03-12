"""Test suite for LLM implementations."""

import base64

import pytest

try:
    from lumen.ai.llm import AzureOpenAI, Message  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from instructor.processing.multimodal import Image


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
