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
        force_new_batch: bool = False,
    ) -> dict[str, str]:
        """
        Main batch processing pipeline:
        1. Check for existing completed batches that match the current chunks (unless forced)
        2. If found and not forced, resume and return those results
        3. If not found or forced, create new batches
        4. Return mapping of chunk -> context
        """
        self.vector_store = vector_store
        max_batch_size = max_batch_size or self.batch_size

        try:
            # Check if we should skip resume and force a new batch
            resume_batch_id = None
            if not force_new_batch:
                # Only look for resumable batches if not forcing new
                resume_batch_id = await self._find_resumable_batch(chunks)

            if resume_batch_id:
                log_debug(f"Found existing completed batch: {resume_batch_id}")
                contexts = await self._resume_batch_processing(resume_batch_id, chunks)
                if contexts:  # If resume was successful, return the results
                    return contexts
                else:
                    log_debug(f"Resume failed for batch {resume_batch_id}, creating new batch")

            # No resumable batch found or resume failed, create new batch
            if not force_new_batch:
                # Prompt user before creating new batch
                response = input(f"\n❓ About to create a new batch job for {len(chunks)} chunks. Continue? [Y/n]: ").strip().lower()
                if response in ['n', 'no', 'quit', 'exit']:
                    raise RuntimeError("\n🛑 Batch creation cancelled by user.")
                elif response not in ['', 'y', 'yes']:
                    raise RuntimeError(f"\n❓ Unrecognized response '{response}'.")
            log_debug(f"Creating new batch for {len(chunks)} chunks")
            contexts = await self._batch_generate_contexts(chunks, document, metadata, max_batch_size)
            return contexts
        except Exception as e:
            log_debug(f"Batch processing error: {e}")
            raise

    async def _resume_batch_processing(self, batch_id: str, chunks: list[str]) -> dict[str, str]:
        """Resume processing from an existing batch job"""
        log_debug(f"Attempting to resume batch {batch_id}")

        # Check if we have saved state for this batch
        saved_state = self.provider.load_batch_state(batch_id)
        if not saved_state:
            raise ValueError(f"No saved state found for batch {batch_id}. Cannot resume.")

        log_debug(f"Found saved state for batch {batch_id}: {saved_state.get('status', 'unknown')}")

        # Resume monitoring the batch
        batch_data = await self.provider.resume_batch(batch_id)

        if not batch_data:
            raise Exception(f"Failed to resume batch {batch_id}")

        # Get results
        batch_results = await self.provider.get_batch_results(batch_data)

        # Parse results - we need to map them back to chunks
        # This assumes the chunks are provided in the same order as the original batch
        contexts = {}

        # Sort results by custom_id to maintain order
        for result in sorted(batch_results, key=lambda x: int(x["custom_id"].split("_")[-1])):
            try:
                chunk_index = int(result["custom_id"].split("_")[-1])
                if chunk_index < len(chunks):
                    chunk = chunks[chunk_index]
                    response_content = self.provider.extract_response_content(result)
                    contexts[chunk] = response_content.strip()
                else:
                    log_debug(f"Warning: chunk_index {chunk_index} exceeds chunks length {len(chunks)}")
            except Exception as e:
                log_debug(f"Error parsing resumed batch result: {e}")

        log_debug(f"Resumed batch processing generated {len(contexts)} contexts")
        return contexts

    async def _find_resumable_batch(self, chunks: list[str]) -> str | None:
        """Find a completed batch that can be reused for the current chunks"""
        try:
            # Get list of all pending/completed batches
            sorted_batches = self.provider.list_pending_batches()

            if not sorted_batches:
                return None

            for batch_info in sorted_batches:
                batch_id = batch_info['batch_id']
                # Load the batch state to check if it's completed
                saved_state = self.provider.load_batch_state(batch_id)
                if not saved_state:
                    continue
                print(f"Resuming batch: {batch_id} with state: {saved_state}")  # noqa: T201
                return batch_id
            return None
        except Exception as e:
            log_debug(f"Error finding resumable batch: {e}")
            return None

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
