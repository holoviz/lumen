import os

from abc import abstractmethod

import openai
import param

# Environment variable mapping for providers that require API keys.
# Providers not in this list (like ollama, llama-cpp) don't require env vars.
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
    "anthropic-bedrock": "AWS_ACCESS_KEY_ID",
    "mistral": "MISTRAL_API_KEY",
    "azure-mistral": "AZUREAI_ENDPOINT_KEY",
    "azure-openai": "AZUREAI_ENDPOINT_KEY",
    "google": "GEMINI_API_KEY",
    "ai-catalyst": "AI_CATALYST_API_KEY",
}


class ServiceMixin(param.Parameterized):
    """
    Base mixin class that defines the standard interface for service providers.
    All service mixins should inherit from this class to ensure consistent APIs.
    """

    @abstractmethod
    def _instantiate_client_kwargs(self, **extra_kwargs) -> dict:
        """
        Get the keyword arguments for initializing a client instance.

        Parameters
        ----------
        **extra_kwargs
            Additional keyword arguments that can override defaults

        Returns
        -------
        dict
            Dictionary of keyword arguments for client initialization
        """

    @abstractmethod
    def _instantiate_client(self, **extra_kwargs):
        """
        Create and return a client instance.

        Parameters
        ----------
        **extra_kwargs
            Additional keyword arguments for client initialization

        Returns
        -------
        Client instance for the service provider
        """


class DbtslMixin(ServiceMixin):

    auth_token = param.String(default=None, doc="""
        The auth token for the dbt semantic layer client;
        if not provided will fetched from DBT_AUTH_TOKEN env var.""")

    environment_id = param.Integer(default=None, doc="""
        The environment ID for the dbt semantic layer client.""")

    host = param.String(default="semantic-layer.cloud.getdbt.com", doc="""
        The host for the dbt semantic layer client.""")

    def __init__(self, environment_id: int, **params):
        super().__init__(environment_id=environment_id, **params)

    def _instantiate_client_kwargs(self, **extra_kwargs) -> dict:
        """
        Get the keyword arguments for initializing a dbt Semantic Layer client.
        """
        kwargs = {
            "environment_id": self.environment_id,
            "auth_token": self.auth_token or os.getenv("DBT_AUTH_TOKEN"),
            "host": self.host,
        }
        kwargs.update(extra_kwargs)
        return kwargs

    def _instantiate_client(self, **extra_kwargs):
        """
        Create and return a dbt Semantic Layer client instance.
        """
        from dbtsl.asyncio import AsyncSemanticLayerClient

        kwargs = self._instantiate_client_kwargs(**extra_kwargs)
        return AsyncSemanticLayerClient(**kwargs)


class LlamaCppMixin(ServiceMixin):
    """
    Mixin class for llama-cpp-python functionality that can be shared
    between LLM implementations and embedding classes.
    """

    n_ctx = param.Integer(default=2048, doc="""
        Context length for the model.""")

    n_batch = param.Integer(default=512, doc="""
        Batch size for processing.""")

    n_gpu_layers = param.Integer(default=-1, doc="""
        Number of layers to offload to GPU. -1 for all layers.""")

    seed = param.Integer(default=128, doc="""
        Random seed for reproducible outputs.""")

    use_mlock = param.Boolean(default=True, doc="""
        Force system to keep model in RAM rather than swapping.""")

    verbose = param.Boolean(default=False, doc="""
        Enable verbose output from llama.cpp.""")

    def _get_model_path(self, model_kwargs: dict | None = None) -> str:
        """
        Get the model path, either from local path or by downloading from HuggingFace.
        """
        kwargs = (model_kwargs or self.model_kwargs)

        if 'model_path' in kwargs:
            return kwargs['model_path']
        elif 'repo_id' in kwargs and 'filename' in kwargs:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id=kwargs['repo_id'],
                filename=kwargs['filename'],
                local_files_only=False
            )
        else:
            raise ValueError(
                "model_kwargs must contain either 'model_path' or both 'repo_id' and 'filename'"
            )

    def _instantiate_client_kwargs(self, model_kwargs: dict | None = None, **extra_kwargs) -> dict:
        """
        Get the keyword arguments for initializing a Llama instance.
        """
        if model_kwargs is None:
            # If no specific model_kwargs provided, use the default model configuration
            kwargs = self.model_kwargs.get("default", {})
        else:
            kwargs = model_kwargs
        model_path = self._get_model_path(kwargs)

        llama_kwargs = {
            "model_path": model_path,
            "n_ctx": kwargs.get('n_ctx', self.n_ctx),
            "n_batch": kwargs.get('n_batch', self.n_batch),
            "n_gpu_layers": kwargs.get('n_gpu_layers', self.n_gpu_layers),
            "seed": kwargs.get('seed', self.seed),
            "use_mlock": kwargs.get('use_mlock', self.use_mlock),
            "verbose": kwargs.get('verbose', self.verbose),
        }

        # Add chat_format if specified
        if 'chat_format' in kwargs:
            llama_kwargs["chat_format"] = kwargs['chat_format']

        # Allow extra kwargs to override defaults
        llama_kwargs.update(extra_kwargs)

        return llama_kwargs

    def _instantiate_client(self, model_kwargs: dict | None = None, **extra_kwargs):
        """
        Create and return a Llama instance with the configured parameters.
        """
        from llama_cpp import Llama

        kwargs = self._instantiate_client_kwargs(model_kwargs=model_kwargs, **extra_kwargs)
        return Llama(**kwargs)

    def resolve_model_spec(self, model_spec: str, base_model_kwargs: dict) -> dict:
        """
        Resolve a model specification string into model kwargs with repo_id/filename.
        Handles formats like "repo/model:chat_format" or "repo/model".
        """
        if "/" not in model_spec:
            return base_model_kwargs

        repo_id, model_spec = model_spec.rsplit("/", 1)
        model_kwargs = dict(base_model_kwargs)

        if ":" in model_spec:
            filename, chat_format = model_spec.split(":")
            model_kwargs["chat_format"] = chat_format
        else:
            filename = model_spec

        model_kwargs["repo_id"] = repo_id
        model_kwargs["filename"] = filename
        return model_kwargs

    @classmethod
    def _warmup_models(cls, model_kwargs_dict: dict | None):
        """
        Download models from HuggingFace Hub for offline use.
        """
        if not model_kwargs_dict:
            return

        huggingface_models = {
            model: llm_spec for model, llm_spec in model_kwargs_dict.items()
            if 'repo_id' in llm_spec and 'filename' in llm_spec
        }
        if not huggingface_models:
            return

        import json

        from huggingface_hub import hf_hub_download
        print(f"{cls.__name__} provider is downloading following models:\n\n{json.dumps(huggingface_models, indent=2)}")  # noqa: T201
        for kwargs in model_kwargs_dict.values():
            repo_id = kwargs.get('repo_id')
            filename = kwargs.get('filename')
            if repo_id and filename:
                hf_hub_download(repo_id, filename)


class APIKeyServiceMixin(ServiceMixin):
    """
    Mixin for service providers that authenticate with a single API key.
    Handles env var resolution at instantiation time rather than class definition time.
    Subclasses set `api_key_env_var` and implement `_instantiate_client`.
    """

    api_key = param.String(default=None, doc="""
        The API key. If not provided, falls back to the environment variable
        named by `api_key_env_var`.""")

    api_key_env_var: str = ""

    def __init__(self, **params):
        if "api_key" not in params:
            params["api_key"] = os.environ.get(self.api_key_env_var)
        super().__init__(**params)

    @classmethod
    def _resolve_api_key(cls) -> str | None:
        """Resolve the API key from the environment variable."""
        return os.environ.get(cls.api_key_env_var)

    def _instantiate_client_kwargs(self, **extra_kwargs) -> dict:
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        kwargs.update(extra_kwargs)
        return kwargs


class AnthropicMixin(APIKeyServiceMixin):
    """
    Mixin class for Anthropic functionality that can be shared
    between LLM implementations and embedding classes.
    """

    api_key = param.String(default=None, doc="""
        The Anthropic API key. If not provided, falls back to the
        ANTHROPIC_API_KEY environment variable.""")

    api_key_env_var: str = PROVIDER_ENV_VARS['anthropic']

    def _instantiate_client(self, async_client=True, **extra_kwargs):
        from anthropic import Anthropic, AsyncAnthropic
        kwargs = self._instantiate_client_kwargs(**extra_kwargs)
        return AsyncAnthropic(**kwargs) if async_client else Anthropic(**kwargs)


class GenAIMixin(APIKeyServiceMixin):
    """
    Mixin class for Google GenAI functionality that can be shared
    between LLM implementations and embedding classes.
    """

    api_key = param.String(default=None, doc="""
        The Google API key. If not provided, falls back to the
        GEMINI_API_KEY environment variable.""")

    api_key_env_var: str = PROVIDER_ENV_VARS['google']

    def _instantiate_client(self, **extra_kwargs):
        from google import genai
        kwargs = self._instantiate_client_kwargs(**extra_kwargs)
        return genai.Client(**kwargs)


class MistralAIMixin(APIKeyServiceMixin):
    """
    Mixin class for Mistral AI functionality that can be shared
    between LLM implementations and embedding classes.
    """

    api_key = param.String(default=None, doc="""
        The Mistral AI API key. If not provided, falls back to the
        MISTRAL_API_KEY environment variable.""")

    api_key_env_var: str = PROVIDER_ENV_VARS['mistral']

    def _instantiate_client(self, **extra_kwargs):
        from mistralai import Mistral
        kwargs = self._instantiate_client_kwargs(**extra_kwargs)
        return Mistral(**kwargs)


class AzureMistralAIMixin(MistralAIMixin):
    """
    Mixin class for Azure Mistral AI functionality that extends MistralAIMixin
    with Azure-specific configuration.
    """

    api_key = param.String(default=None, doc="""
        The Azure API key. If not provided, falls back to the
        AZUREAI_ENDPOINT_KEY environment variable.""")

    api_key_env_var: str = PROVIDER_ENV_VARS['azure-mistral']

    endpoint = param.String(default=None, doc="""
        The Azure Mistral endpoint URL.""")

    def __init__(self, **params):
        if "endpoint" not in params:
            params["endpoint"] = os.environ.get("AZUREAI_ENDPOINT_URL")
        super().__init__(**params)

    def _instantiate_client(self, **extra_kwargs):
        """
        Create and return an Azure Mistral client instance.
        """
        from mistralai_azure import MistralAzure
        kwargs = self._instantiate_client_kwargs(**extra_kwargs)
        return MistralAzure(azure_endpoint=self.endpoint, **kwargs)


class OpenAIMixin(APIKeyServiceMixin):
    """
    Mixin class for OpenAI functionality that can be shared
    between LLM implementations and embedding classes.
    """

    api_key = param.String(default=None, doc="""
        The API key. If not provided, falls back to the
        OPENAI_API_KEY environment variable.""")

    api_key_env_var: str = PROVIDER_ENV_VARS['openai']

    endpoint = param.String(default=None, doc="""
        The OpenAI API endpoint. If not provided, uses default OpenAI endpoint.""")

    organization = param.String(default=None, doc="""
        The OpenAI organization to charge.""")

    def _instantiate_client_kwargs(self, **extra_kwargs) -> dict:
        kwargs = super()._instantiate_client_kwargs()
        if self.endpoint:
            kwargs["base_url"] = self.endpoint
        if self.organization:
            kwargs["organization"] = self.organization
        kwargs.update(extra_kwargs)
        return kwargs

    def _instantiate_client(self, async_client=True, **extra_kwargs):
        """
        Create and return an OpenAI client instance.
        """
        kwargs = self._instantiate_client_kwargs(**extra_kwargs)

        if async_client:
            return openai.AsyncOpenAI(**kwargs)
        else:
            return openai.OpenAI(**kwargs)


class AzureOpenAIMixin(OpenAIMixin):
    """
    Mixin class for Azure OpenAI functionality that extends OpenAI functionality
    with Azure-specific configuration.
    """

    api_version = param.String(default="2024-10-21", doc="""
        The Azure AI Studio API version.""")

    api_key = param.String(default=None, doc="""
        The Azure API key. If not provided, falls back to the
        AZUREAI_ENDPOINT_KEY environment variable.""")

    api_key_env_var: str = PROVIDER_ENV_VARS['azure-openai']

    endpoint = param.String(default=None, doc="""
        The Azure AI Studio endpoint.""")

    def __init__(self, **params):
        if "endpoint" not in params:
            params["endpoint"] = os.environ.get("AZUREAI_ENDPOINT_URL")
        super().__init__(**params)

    def _instantiate_client_kwargs(self, **extra_kwargs) -> dict:
        kwargs = super()._instantiate_client_kwargs()
        if self.api_version:
            kwargs["api_version"] = self.api_version
        if self.endpoint:
            kwargs.pop("base_url", None)
            kwargs["azure_endpoint"] = self.endpoint
        kwargs.update(extra_kwargs)
        return kwargs

    def _instantiate_client(self, async_client=True, **extra_kwargs):
        """
        Create and return an Azure OpenAI client instance.
        """
        kwargs = self._instantiate_client_kwargs(**extra_kwargs)

        if async_client:
            return openai.AsyncAzureOpenAI(**kwargs)
        else:
            return openai.AzureOpenAI(**kwargs)


class BedrockMixin(param.Parameterized):
    """
    Mixin class for AWS Bedrock functionality that can be shared
    between LLM implementations and embedding classes.
    """

    aws_access_key_id = param.String(default=None,
        doc="AWS access key ID. If not provided, boto3 will use default credentials (including SSO).")

    api_key = param.String(default=None,
        doc="AWS secret access key. If not provided, falls back to AWS_SECRET_ACCESS_KEY "
            "or boto3 default credentials (including SSO).")

    aws_session_token = param.String(default=None,
        doc="AWS session token for temporary credentials (optional).")

    region_name = param.String(default="us-east-1", doc="The AWS region name for Bedrock API calls.")

    def __init__(self, **params):
        if "aws_secret_access_key" in params:
            params["api_key"] = params.pop("aws_secret_access_key")
        if "api_key" not in params:
            params["api_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if "aws_access_key_id" not in params:
            params["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID")
        if "aws_session_token" not in params:
            params["aws_session_token"] = os.environ.get("AWS_SESSION_TOKEN")
        super().__init__(**params)
