import os

from abc import abstractmethod

import openai
import param


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
        kwargs = model_kwargs or self.model_kwargs

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
        kwargs = model_kwargs or self.model_kwargs
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


class OpenAIMixin(ServiceMixin):
    """
    Mixin class for OpenAI functionality that can be shared
    between LLM implementations and embedding classes.
    """

    api_key = param.String(default=None, doc="""
        The OpenAI API key. If not provided, will use OPENAI_API_KEY env var.""")

    endpoint = param.String(default=None, doc="""
        The OpenAI API endpoint. If not provided, uses default OpenAI endpoint.""")

    organization = param.String(default=None, doc="""
        The OpenAI organization to charge.""")

    def _instantiate_client_kwargs(self, **extra_kwargs) -> dict:
        """
        Get the keyword arguments for initializing an OpenAI client.
        """
        kwargs = {}

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.endpoint:
            kwargs["base_url"] = self.endpoint
        if self.organization:
            kwargs["organization"] = self.organization

        # Allow extra kwargs to override defaults
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

    # Override defaults to use Azure environment variables
    api_key = param.String(default=os.getenv("AZUREAI_ENDPOINT_KEY"), doc="""
        The Azure API key.""")

    endpoint = param.String(default=os.getenv('AZUREAI_ENDPOINT_URL'), doc="""
        The Azure AI Studio endpoint.""")

    def _instantiate_client_kwargs(self, **extra_kwargs) -> dict:
        """
        Get the keyword arguments for initializing an Azure OpenAI client.
        Builds on top of the base OpenAI client kwargs.
        """
        # Start with base OpenAI kwargs
        kwargs = super()._instantiate_client_kwargs()

        # Add Azure-specific parameters
        if self.api_version:
            kwargs["api_version"] = self.api_version
        if self.endpoint:
            # Azure uses 'azure_endpoint' instead of 'base_url'
            kwargs.pop("base_url", None)  # Remove base_url if it exists
            kwargs["azure_endpoint"] = self.endpoint

        # Allow extra kwargs to override defaults
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
