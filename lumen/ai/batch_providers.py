"""
Multi-provider batch processing implementations for LLM providers.
"""

import asyncio
import json
import time
import uuid

from abc import abstractmethod
from pathlib import Path
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

    timeout = param.Integer(default=1800, bounds=(60, 172800), doc="Maximum time to wait for batch job completion (seconds)")
    batch_dir = param.String(default=".lumen_batch", doc="Directory to save batch state files")
    keep_states = param.Boolean(default=True, doc="Keep batch state files after successful completion")

    def __init__(self, **params):
        super().__init__(**params)
        # Ensure batch state directory exists
        Path(self.batch_dir).mkdir(exist_ok=True)

    def _get_batch_state_file(self, batch_id: str) -> Path:
        """Get the path to the batch state file for a given batch ID"""
        return Path(self.batch_dir) / f"{batch_id}.json"

    def save_batch_state(self, batch_id: str, state: dict[str, Any]) -> None:
        """Save batch state to file"""
        state_file = self._get_batch_state_file(batch_id)
        state_data = {
            "batch_id": batch_id,
            "provider": self.__class__.__name__,
            "created_at": time.time(),
            "state": state
        }

        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

        log_debug(f"Saved batch state to {state_file}")

    def load_batch_state(self, batch_id: str) -> dict[str, Any] | None:
        """Load batch state from file"""
        state_file = self._get_batch_state_file(batch_id)

        if not state_file.exists():
            return None

        try:
            with open(state_file) as f:
                state_data = json.load(f)

            # Verify this is the right provider
            if state_data.get("provider") != self.__class__.__name__:
                log_debug(f"Batch state provider mismatch: expected {self.__class__.__name__}, got {state_data.get('provider')}")
                return None

            log_debug(f"Loaded batch state from {state_file}")
            return state_data.get("state", {})

        except Exception as e:
            log_debug(f"Error loading batch state: {e}")
            return None

    def cleanup_batch_state(self, batch_id: str) -> None:
        """Remove batch state file after successful completion"""
        state_file = self._get_batch_state_file(batch_id)
        if state_file.exists():
            state_file.unlink()
            log_debug(f"Cleaned up batch state file: {state_file}")

    def list_pending_batches(self) -> list[dict[str, Any]]:
        """List all pending batch jobs for this provider"""
        batch_dir = Path(self.batch_dir)
        if not batch_dir.exists():
            return []

        pending_batches = []
        for state_file in batch_dir.glob("*.json"):
            try:
                with open(state_file) as f:
                    state_data = json.load(f)

                # Only include batches for this provider
                if state_data.get("provider") == self.__class__.__name__:
                    pending_batches.append({
                        "batch_id": state_data["batch_id"],
                        "created_at": state_data["created_at"],
                        "state_file": str(state_file)
                    })
            except Exception as e:
                log_debug(f"Error reading batch state file {state_file}: {e}")

        return sorted(pending_batches, key=lambda x: x["created_at"], reverse=True)

    async def resume_batch(self, batch_id: str) -> dict[str, Any] | None:
        """Resume monitoring an existing batch job"""
        saved_state = self.load_batch_state(batch_id)
        if not saved_state:
            log_debug(f"No saved state found for batch {batch_id}")
            return None

        log_debug(f"Resuming batch {batch_id}")

        try:
            # Monitor the batch until completion
            batch_data = await self.monitor_batch(batch_id)
            return batch_data

        except Exception as e:
            log_debug(f"Error resuming batch {batch_id}: {e}")
            return None

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

        log_debug(f"Created batch file: {batch_file_path} with {len(requests)} requests")

        # Upload file - client methods are already async, don't wrap in to_thread
        try:
            with open(batch_file_path, "rb") as f:
                batch_file = await client.files.create(file=f, purpose="batch")

            log_debug(f"Successfully uploaded batch file with ID: {batch_file.id}")

        except Exception as e:
            log_debug(f"Error uploading batch file: {e}")
            raise Exception(f"Failed to upload batch file: {e}") from e

        # Create batch - this is also async
        try:
            batch = await client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            # Save batch state immediately after creation
            self.save_batch_state(batch.id, {
                "input_file_id": batch_file.id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "num_requests": len(requests),
                "status": "created"
            })

            log_debug(f"Created OpenAI batch job: {batch.id}")
            return batch.id

        except Exception as e:
            log_debug(f"Error creating batch: {e}")
            raise Exception(f"Failed to create batch job: {e}") from e

    async def monitor_batch(self, batch_id: str) -> dict[str, Any]:
        """Monitor batch job with exponential backoff"""
        client = await self._get_client()
        start_time = time.time()
        wait_time = 30  # Start with 30 second polls

        while time.time() - start_time < self.timeout:
            batch_status = await client.batches.retrieve(batch_id)

            print(f"OpenAI batch {batch_id} status: {batch_status.status}")  # noqa: T201

            # Update saved state with current status
            saved_state = self.load_batch_state(batch_id) or {}
            saved_state["status"] = batch_status.status
            saved_state["last_checked"] = time.time()
            self.save_batch_state(batch_id, saved_state)

            if batch_status.status == "completed":
                # Only clean up state file if not keeping completed batches
                if not self.keep_states:
                    self.cleanup_batch_state(batch_id)
                else:
                    # Mark as completed in state file for reference
                    saved_state["status"] = "completed"
                    saved_state["completed_at"] = time.time()
                    self.save_batch_state(batch_id, saved_state)
                    log_debug(f"Keeping completed batch state for {batch_id}")
                return batch_status
            elif batch_status.status in ["failed", "expired", "cancelled"]:
                # Keep state file for debugging failed batches
                saved_state["error"] = f"Batch failed with status: {batch_status.status}"
                self.save_batch_state(batch_id, saved_state)
                raise Exception(f"OpenAI batch job {batch_id} failed: {batch_status.status}")

            # Exponential backoff with jitter
            await asyncio.sleep(wait_time)
            wait_time = min(wait_time * 1.2, 300)  # Cap at 5 minutes

        # Save timeout state
        saved_state = self.load_batch_state(batch_id) or {}
        saved_state["error"] = f"Timeout after {self.timeout} seconds"
        self.save_batch_state(batch_id, saved_state)

        raise TimeoutError(f"OpenAI batch job {batch_id} timed out after {self.timeout} seconds")

    async def get_batch_results(self, batch_data) -> list[dict[str, Any]]:
        """Download and parse OpenAI batch results"""
        client = await self._get_client()

        # Download results file
        result_file_id = batch_data.output_file_id
        result_content = await client.files.content(result_file_id)

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

        batch = await client.messages.batches.create(requests=batch_requests)

        log_debug(f"Created Anthropic batch job: {batch.id}")
        return batch.id

    async def monitor_batch(self, batch_id: str) -> dict[str, Any]:
        """Monitor Anthropic batch job"""
        client = await self._get_client()
        start_time = time.time()
        wait_time = 30

        while time.time() - start_time < self.timeout:
            batch_status = await client.messages.batches.retrieve(batch_id)

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

        # Get results - the results method might need to be handled differently
        # Let's try the async approach first, fall back to sync if needed
        try:
            # Try async iteration
            results = []
            async for result in client.messages.batches.results(batch_data.id):
                results.append(result)
            return results
        except AttributeError:
            # Fall back to sync iteration in a thread if async not available
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
        try:
            with open(batch_file_path, "rb") as f:
                batch = await client.batch.create(input_file=f, endpoint="/v1/chat/completions")

            log_debug(f"Created Mistral batch job: {batch.id}")
            return batch.id
        except Exception as e:
            log_debug(f"Error creating Mistral batch: {e}")
            raise Exception(f"Failed to create Mistral batch job: {e}") from e

    async def monitor_batch(self, batch_id: str) -> dict[str, Any]:
        """Monitor Mistral batch job"""
        client = await self._get_client()
        start_time = time.time()
        wait_time = 30

        while time.time() - start_time < self.timeout:
            batch_status = await client.batch.retrieve(batch_id)

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
        result_content = await client.files.content(result_file_id)

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
