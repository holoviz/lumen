"""Test suite for LLM implementations."""

import pytest

try:
	from lumen.ai.llm import AzureOpenAI  # noqa
except ModuleNotFoundError:
	pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)


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
