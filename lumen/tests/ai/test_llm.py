"""Test suite for LLM implementations."""

import os

import pytest

try:
    import lumen.ai as lmai

    from lumen.ai.llm import (
        Anthropic, AzureOpenAI, Google, Groq, MistralAI, OpenAI,
    )
except ModuleNotFoundError:
	pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)


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
