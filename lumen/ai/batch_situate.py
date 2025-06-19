"""
Multi-provider batch situate processor for vector stores.
"""

import uuid

from typing import Any

import param

from .batch_providers import BatchProvider
from .utils import log_debug


class BatchSituateProcessor(param.Parameterized):
    """Provider-agnostic batch processing for chunk situating"""

    provider = param.ClassSelector(class_=BatchProvider, doc="The batch provider to use for processing")
    batch_size = param.Integer(default=1000, bounds=(1, 10000), doc="Maximum number of requests per batch")

    async def batch_situate_chunks(
        self,
        document: str,
        chunks: list[str],
        metadata: dict,
        vector_store,
        max_batch_size: int | None = None,
    ) -> dict[str, str]:
        """
        Main batch processing pipeline:
        1. Split chunks into provider-appropriate batches
        2. Create context generation batch requests
        3. Submit and monitor context generation jobs
        4. Return mapping of chunk -> context
        """
        self.vector_store = vector_store
        max_batch_size = max_batch_size or self.batch_size

        try:
            log_debug(f"Generating contexts for {len(chunks)} chunks")
            contexts = await self._batch_generate_contexts(chunks, document, metadata, max_batch_size)
            return contexts
        except Exception as e:
            log_debug(f"Batch processing error: {e}")
            raise

    async def _batch_generate_contexts(self, chunks: list[str], document: str, metadata: dict, max_batch_size: int) -> dict[str, str]:
        """Generate contexts for chunks in batches"""

        contexts = {}

        # Process in batches
        for i in range(0, len(chunks), max_batch_size):
            batch_chunks = chunks[i : i + max_batch_size]
            batch_id = f"context_gen_{uuid.uuid4().hex[:8]}"

            # Create batch requests with context
            batch_requests = await self._create_batch_requests_from_prompts("main", batch_chunks, batch_id, document=document, metadata=metadata)

            # Submit batch
            batch_job_id = await self.provider.create_batch(batch_requests)

            # Monitor and get results
            batch_data = await self.provider.monitor_batch(batch_job_id)
            batch_results = await self.provider.get_batch_results(batch_data)

            # Parse context generation results
            for result in sorted(batch_results, key=lambda x: int(x["custom_id"].split("_")[-1])):
                try:
                    chunk_index = int(result["custom_id"].split("_")[-1])
                    chunk = batch_chunks[chunk_index]
                    response_content = self.provider.extract_response_content(result)
                    contexts[chunk] = response_content.strip()
                except Exception as e:
                    log_debug(f"Error parsing context generation result: {e}")

        return contexts

    async def _create_batch_requests_from_prompts(self, prompt_name: str, chunks: list[str], batch_id: str, **context_vars) -> list[dict[str, Any]]:
        """Create batch requests using existing LLM prompt system for any provider"""

        batch_requests = []

        for i, chunk in enumerate(chunks):
            # Use vector store's _render_prompt method
            messages = [{"role": "user", "content": chunk}]
            system_prompt = await self.vector_store._render_prompt(prompt_name, messages, **context_vars)

            # Get model specification from prompt config
            try:
                model_spec = self.vector_store._lookup_prompt_key(prompt_name, "llm_spec")
            except KeyError:
                model_spec = self.vector_store.llm_spec_key

            # Resolve model name using provider
            model_name = self.provider.resolve_model_name(model_spec)

            # Create provider-specific request body
            body = self.provider.create_request_body(system_prompt=system_prompt, user_content=chunk, model=model_name)

            # Handle response model if specified
            try:
                response_model = self.vector_store._get_model(prompt_name, **context_vars)
                if response_model:
                    body.update(self.provider.format_response_model(response_model))
            except (KeyError, AttributeError):
                pass

            # Format request using provider-specific format
            formatted_request = self.provider.format_request(custom_id=f"{batch_id}_{i}", prompt_data=body)
            batch_requests.append(formatted_request)

        return batch_requests
