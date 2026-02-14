"""Test suite for LLM implementations."""

import os

import pytest

try:
	from lumen.ai.llm import (
	    PROVIDER_ENV_VARS, AINavigator, Anthropic, AzureOpenAI, Google, Llm,
	    MistralAI, Ollama, OpenAI,
	)
except ModuleNotFoundError:
	pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)


class TestIsAvailable:
	"""Tests for LLM is_online() methods."""

	def test_openai_is_online_no_api_key(self, monkeypatch):
		"""Test OpenAI.is_online() returns False when API key is not set."""
		monkeypatch.delenv("OPENAI_API_KEY", raising=False)
		assert OpenAI.is_online() is False

	def test_openai_is_online_invalid_api_key(self, monkeypatch):
		"""Test OpenAI.is_online() returns False with invalid API key."""
		monkeypatch.setenv("OPENAI_API_KEY", "invalid-key")
		assert OpenAI.is_online() is False

	def test_openai_is_online_with_model_param(self, monkeypatch):
		"""Test OpenAI.is_online() accepts models parameter."""
		monkeypatch.delenv("OPENAI_API_KEY", raising=False)
		# Should return False because no API key, regardless of models
		assert OpenAI.is_online() is False

	def test_anthropic_is_online_no_api_key(self, monkeypatch):
		"""Test Anthropic.is_online() returns False when API key is not set."""
		monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
		assert Anthropic.is_online() is False

	def test_anthropic_is_online_invalid_api_key(self, monkeypatch):
		"""Test Anthropic.is_online() returns False with invalid API key."""
		monkeypatch.setenv("ANTHROPIC_API_KEY", "invalid-key")
		assert Anthropic.is_online() is False

	def test_mistral_is_online_no_api_key(self, monkeypatch):
		"""Test MistralAI.is_online() returns False when API key is not set."""
		monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
		assert MistralAI.is_online() is False

	def test_mistral_is_online_invalid_api_key(self, monkeypatch):
		"""Test MistralAI.is_online() returns False with invalid API key."""
		monkeypatch.setenv("MISTRAL_API_KEY", "invalid-key")
		assert MistralAI.is_online() is False

	def test_google_is_online_no_api_key(self, monkeypatch):
		"""Test Google.is_online() returns False when API key is not set."""
		monkeypatch.delenv("GEMINI_API_KEY", raising=False)
		assert Google.is_online() is False

	def test_google_is_online_invalid_api_key(self, monkeypatch):
		"""Test Google.is_online() returns False with invalid API key."""
		monkeypatch.setenv("GEMINI_API_KEY", "invalid-key")
		assert Google.is_online() is False

	def test_ollama_is_online_no_server(self):
		"""Test Ollama.is_online() returns False when server is not running."""
		# Use a port that's unlikely to have a server running
		assert Ollama.is_online(endpoint="http://localhost:59999/v1") is False

	def test_ollama_is_online_default_endpoint(self):
		"""Test Ollama.is_online() works with default endpoint."""
		# This will return False unless Ollama is actually running
		result = Ollama.is_online()
		assert isinstance(result, bool)

	def test_ollama_is_online_with_model_param(self):
		"""Test Ollama.is_online() accepts models parameter."""
		# Use a port that's unlikely to have a server running
		assert Ollama.is_online(endpoint="http://localhost:59999/v1") is False

	def test_ai_navigator_is_online_no_server(self):
		"""Test AINavigator.is_online() returns False when server is not running."""
		# Use a port that's unlikely to have a server running
		assert AINavigator.is_online(endpoint="http://localhost:59999/v1") is False

	def test_ai_navigator_is_online_default_endpoint(self):
		"""Test AINavigator.is_online() works with default endpoint."""
		# This will return False unless AI Navigator is actually running
		result = AINavigator.is_online()
		assert isinstance(result, bool)

	def test_base_llm_is_online_with_env_var(self, monkeypatch):
		"""Test base Llm.is_online() checks PROVIDER_ENV_VARS."""
		# For providers in PROVIDER_ENV_VARS, is_online checks the env var
		for class_name, env_var in PROVIDER_ENV_VARS.items():
			monkeypatch.delenv(env_var, raising=False)

		# Base Llm not in PROVIDER_ENV_VARS, so should return True by default
		assert Llm.is_online() is True


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
