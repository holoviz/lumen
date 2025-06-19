"""
Multi-provider batch processing implementations for LLM providers.
"""

import asyncio
import json
import time
import uuid

from abc import abstractmethod
from typing import Any

import param

from .llm import (
    AINavigator, AnthropicAI, AzureMistralAI, AzureOpenAI, Llm, MistralAI,
    OpenAI,
)
from .utils import log_debug


class BatchProcessingNotSupportedError(Exception):
    """Raised when batch processing is not supported for a provider."""


class BatchProvider(param.Parameterized):
    """Abstract base class for batch processing providers"""

    timeout = param.Integer(default=1800, bounds=(60, 7200), doc="Maximum time to wait for batch job completion (seconds)")

    @abstractmethod
    async def create_batch(self, requests: list[dict[str, Any]]) -> str:
        """Create a batch job and return batch ID"""

    @abstractmethod
    async def monitor_batch(self, batch_id: str) -> dict[str, Any]:
        """Monitor batch job until completion"""

    @abstractmethod
    async def get_batch_results(self, batch_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Retrieve and parse batch results"""

    @abstractmethod
    def format_request(self, custom_id: str, prompt_data: dict[str, Any]) -> dict[str, Any]:
        """Format a single request for the provider's batch format"""

    @abstractmethod
    def create_request_body(self, system_prompt: str, user_content: str, model: str, **kwargs) -> dict[str, Any]:
        """Create provider-specific request body"""

    @abstractmethod
    def extract_response_content(self, result: dict[str, Any]) -> str:
        """Extract response content from provider result format"""

    @abstractmethod
    def format_response_model(self, response_model) -> dict[str, Any]:
        """Format response model for structured output"""

    @classmethod
    def create_for_llm(cls, llm: Llm, **kwargs):
        """Factory method to create appropriate provider for LLM type"""
        if isinstance(llm, (OpenAI, AzureOpenAI, AINavigator)):
            return OpenAIBatchProvider(llm=llm, **kwargs)
        elif isinstance(llm, AnthropicAI):
            return AnthropicBatchProvider(llm=llm, **kwargs)
        elif isinstance(llm, (MistralAI, AzureMistralAI)):
            return MistralBatchProvider(llm=llm, **kwargs)
        else:
            raise BatchProcessingNotSupportedError(
                f"Batch processing not supported for provider: {type(llm).__name__}. "
                f"Supported providers: OpenAI, AzureOpenAI, AnthropicAI, MistralAI"
            )


class OpenAIBatchProvider(BatchProvider):
    """OpenAI Batch API implementation"""

    llm = param.ClassSelector(class_=Llm, doc="The LLM instance to use for batch processing")

    def __init__(self, **params):
        super().__init__(**params)
        self._client = None

    async def _get_client(self):
        """Get the OpenAI client from the LLM"""
        if self._client is None:
            self._client = await self.llm.get_raw_client()
        return self._client

    def resolve_model_name(self, model_spec: str) -> str:
        """Resolve model specification using LLM model_kwargs"""
        model_kwargs = self.llm._get_model_kwargs(model_spec)
        return model_kwargs.get("model", model_kwargs.get("model_slug", "gpt-4o-mini"))

    async def create_batch(self, requests: list[dict[str, Any]]) -> str:
        client = await self._get_client()

        # Create JSONL file
        batch_file_path = f"/tmp/batch_{uuid.uuid4()}.jsonl"
        with open(batch_file_path, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        # Upload file
        with open(batch_file_path, "rb") as f:
            batch_file = await asyncio.to_thread(client.files.create, file=f, purpose="batch")

        # Create batch
        batch = await asyncio.to_thread(client.batches.create, input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")

        log_debug(f"Created OpenAI batch job: {batch.id}")
        return batch.id

    async def monitor_batch(self, batch_id: str) -> dict[str, Any]:
        """Monitor batch job with exponential backoff"""
        client = await self._get_client()
        start_time = time.time()
        wait_time = 30  # Start with 30 second polls

        while time.time() - start_time < self.timeout:
            batch_status = await asyncio.to_thread(client.batches.retrieve, batch_id)

            log_debug(f"OpenAI batch {batch_id} status: {batch_status.status}")

            if batch_status.status == "completed":
                return batch_status
            elif batch_status.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"OpenAI batch job {batch_id} failed: {batch_status.status}")

            # Exponential backoff with jitter
            await asyncio.sleep(wait_time)
            wait_time = min(wait_time * 1.2, 300)  # Cap at 5 minutes

        raise TimeoutError(f"OpenAI batch job {batch_id} timed out after {self.timeout} seconds")

    async def get_batch_results(self, batch_data) -> list[dict[str, Any]]:
        """Download and parse OpenAI batch results"""
        client = await self._get_client()

        # Download results file
        result_file_id = batch_data.output_file_id
        result_content = await asyncio.to_thread(client.files.content, result_file_id)

        # Parse JSONL results
        results = []
        for line in result_content.content.decode("utf-8").strip().split("\n"):
            if line.strip():
                results.append(json.loads(line))

        return results

    def format_request(self, custom_id: str, prompt_data: dict[str, Any]) -> dict[str, Any]:
        return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": prompt_data}

    def create_request_body(self, system_prompt: str, user_content: str, model: str, **kwargs) -> dict[str, Any]:
        """Create OpenAI-compatible request body"""
        return {"model": model, "temperature": 0.0, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]}

    def extract_response_content(self, result: dict[str, Any]) -> str:
        """Extract response content from OpenAI result format"""
        return result["response"]["body"]["choices"][0]["message"]["content"]

    def format_response_model(self, response_model) -> dict[str, Any]:
        """Format response model for OpenAI structured output"""
        schema = response_model.model_json_schema()
        return {"response_format": {"type": "json_schema", "json_schema": {"name": "structured_response", "schema": schema, "strict": True}}}


class AnthropicBatchProvider(BatchProvider):
    """Anthropic Message Batches API implementation"""

    llm = param.ClassSelector(class_=Llm, doc="The LLM instance to use for batch processing")

    def __init__(self, **params):
        super().__init__(**params)
        self._client = None

    async def _get_client(self):
        """Get the Anthropic client from the LLM"""
        if self._client is None:
            self._client = await self.llm.get_raw_client()
        return self._client

    def resolve_model_name(self, model_spec: str) -> str:
        """Resolve model specification using LLM model_kwargs"""
        model_kwargs = self.llm._get_model_kwargs(model_spec)
        return model_kwargs.get("model", model_kwargs.get("model_slug", "claude-3-5-sonnet-20241022"))

    async def create_batch(self, requests: list[dict[str, Any]]) -> str:
        client = await self._get_client()

        # Anthropic uses direct JSON array, not JSONL
        batch_requests = []
        for request in requests:
            batch_requests.append({"custom_id": request["custom_id"], "params": request["body"]})

        batch = await asyncio.to_thread(client.messages.batches.create, requests=batch_requests)

        log_debug(f"Created Anthropic batch job: {batch.id}")
        return batch.id

    async def monitor_batch(self, batch_id: str) -> dict[str, Any]:
        """Monitor Anthropic batch job"""
        client = await self._get_client()
        start_time = time.time()
        wait_time = 30

        while time.time() - start_time < self.timeout:
            batch_status = await asyncio.to_thread(client.messages.batches.retrieve, batch_id)

            log_debug(f"Anthropic batch {batch_id} status: {batch_status.processing_status}")

            if batch_status.processing_status == "ended":
                return batch_status
            elif batch_status.processing_status in ["failed", "canceled", "expired"]:
                raise Exception(f"Anthropic batch job {batch_id} failed: {batch_status.processing_status}")

            await asyncio.sleep(wait_time)
            wait_time = min(wait_time * 1.2, 300)

        raise TimeoutError(f"Anthropic batch job {batch_id} timed out after {self.timeout} seconds")

    async def get_batch_results(self, batch_data) -> list[dict[str, Any]]:
        """Get Anthropic batch results"""
        client = await self._get_client()

        # Get results using the sync iterator in a thread
        results = await asyncio.to_thread(lambda: list(client.messages.batches.results(batch_data.id)))

        return results

    def format_request(self, custom_id: str, prompt_data: dict[str, Any]) -> dict[str, Any]:
        # Anthropic doesn't need method/url wrapper
        return {"custom_id": custom_id, "body": prompt_data}

    def create_request_body(self, system_prompt: str, user_content: str, model: str, **kwargs) -> dict[str, Any]:
        """Create Anthropic-compatible request body"""
        return {"model": model, "temperature": 0.0, "system": system_prompt, "messages": [{"role": "user", "content": user_content}], "max_tokens": 1024}

    def extract_response_content(self, result: dict[str, Any]) -> str:
        """Extract response content from Anthropic result format"""
        response = result["result"]["message"]
        if "content" in response:
            # Check for tool use (structured response)
            for content_block in response["content"]:
                if content_block.get("type") == "tool_use":
                    return json.dumps(content_block["input"])
                elif content_block.get("type") == "text":
                    return content_block["text"]
        return response.get("content", "")

    def format_response_model(self, response_model) -> dict[str, Any]:
        """Format response model for Anthropic structured output"""
        schema = response_model.model_json_schema()
        return {
            "tools": [{"name": "structured_response", "description": "Provide structured response", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "structured_response"},
        }


class MistralBatchProvider(BatchProvider):
    """Mistral Batch API implementation (similar to OpenAI)"""

    llm = param.ClassSelector(class_=Llm, doc="The LLM instance to use for batch processing")

    def __init__(self, **params):
        super().__init__(**params)
        self._client = None

    async def _get_client(self):
        """Get the Mistral client from the LLM"""
        if self._client is None:
            self._client = await self.llm.get_raw_client()
        return self._client

    def resolve_model_name(self, model_spec: str) -> str:
        """Resolve model specification using LLM model_kwargs"""
        model_kwargs = self.llm._get_model_kwargs(model_spec)
        return model_kwargs.get("model", model_kwargs.get("model_slug", "mistral-large-latest"))

    async def create_batch(self, requests: list[dict[str, Any]]) -> str:
        client = await self._get_client()

        # Similar to OpenAI implementation
        batch_file_path = f"/tmp/batch_{uuid.uuid4()}.jsonl"
        with open(batch_file_path, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        # Use Mistral's batch endpoint
        with open(batch_file_path, "rb") as f:
            batch = await asyncio.to_thread(client.batch.create, input_file=f, endpoint="/v1/chat/completions")

        log_debug(f"Created Mistral batch job: {batch.id}")
        return batch.id

    async def monitor_batch(self, batch_id: str) -> dict[str, Any]:
        """Monitor Mistral batch job"""
        client = await self._get_client()
        start_time = time.time()
        wait_time = 30

        while time.time() - start_time < self.timeout:
            batch_status = await asyncio.to_thread(client.batch.retrieve, batch_id)

            log_debug(f"Mistral batch {batch_id} status: {batch_status.status}")

            if batch_status.status == "completed":
                return batch_status
            elif batch_status.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Mistral batch job {batch_id} failed: {batch_status.status}")

            await asyncio.sleep(wait_time)
            wait_time = min(wait_time * 1.2, 300)

        raise TimeoutError(f"Mistral batch job {batch_id} timed out after {self.timeout} seconds")

    async def get_batch_results(self, batch_data) -> list[dict[str, Any]]:
        """Get Mistral batch results"""
        client = await self._get_client()

        # Similar to OpenAI
        result_file_id = batch_data.output_file_id
        result_content = await asyncio.to_thread(client.files.content, result_file_id)

        results = []
        for line in result_content.content.decode("utf-8").strip().split("\n"):
            if line.strip():
                results.append(json.loads(line))

        return results

    def format_request(self, custom_id: str, prompt_data: dict[str, Any]) -> dict[str, Any]:
        return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": prompt_data}

    def create_request_body(self, system_prompt: str, user_content: str, model: str, **kwargs) -> dict[str, Any]:
        """Create Mistral-compatible request body"""
        return {"model": model, "temperature": 0.0, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]}

    def extract_response_content(self, result: dict[str, Any]) -> str:
        """Extract response content from Mistral result format"""
        return result["response"]["body"]["choices"][0]["message"]["content"]

    def format_response_model(self, response_model) -> dict[str, Any]:
        """Format response model for Mistral structured output"""
        return {"response_format": {"type": "json_object"}}
