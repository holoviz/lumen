import asyncio
import fnmatch
import importlib
import io
import json
import os
import typing as t

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import duckdb
import numpy as np
# https://duckdb.org/docs/stable/clients/python/known_issues#numpy-import-multithreading
import numpy.core.multiarray
import param
import semchunk

from panel import cache as pn_cache
from tqdm.auto import tqdm

from .actor import PROMPTS_DIR, LLMUser
from .embeddings import Embeddings, NumpyEmbeddings
from .models import YesNo
from .utils import log_debug


class VectorStore(LLMUser):
    """Abstract base class for a vector store."""

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "VectorStore" / "main.jinja2"},
            "should_situate": {
                "template": PROMPTS_DIR / "VectorStore" / "should_situate.jinja2",
                "response_model": YesNo,
            },
        },
        doc="""
        A dictionary of prompts used by the vector store, indexed by prompt name.
        Each prompt should be defined as a dictionary containing a template
        'template' and optionally a 'model' and 'tools'.""",
    )

    chunk_func = param.Callable(
        default=None,
        doc="""
        The function used to split documents into chunks.
        Must accept `text`. If None, defaults to semchunk.chunkerify
        with the chunk_size.""",
    )

    chunk_func_kwargs = param.Dict(
        default={},
        doc="""
        Additional keyword arguments to pass to the chunking function.""",
    )

    chunk_size = param.Integer(
        default=1024, doc="Maximum size of text chunks to split documents into."
    )

    chunk_tokenizer = param.String(
        default="gpt-4o-mini",
        doc="""
        If using the default chunk_func, the tokenizer used to split documents into chunks.
        Must be a valid tokenizer name from the transformers or tiktoken library.
        Otherwise, please pass in a custom chunk_func.""",
    )

    embeddings = param.ClassSelector(
        class_=Embeddings,
        default=NumpyEmbeddings(),
        allow_None=True,
        doc="Embeddings object for text processing.",
    )

    excluded_metadata = param.List(
        default=["llm_context"],
        doc="List of metadata keys to exclude when creating the embeddings.",
    )

    situate = param.Boolean(
        default=False,
        doc="""
        Whether to insert a `llm_context` key in the metadata containing
        contextual about the chunks.""",
    )

    def __init__(self, **params):
        self._add_items_lock = asyncio.Lock()  # Lock for thread-safe add_items
        super().__init__(**params)
        if self.chunk_func is None:
            self.chunk_func = semchunk.chunkerify(
                self.chunk_tokenizer, chunk_size=self.chunk_size
            )

    def _format_metadata_value(self, value) -> str:
        """Format a metadata value appropriately based on its type.

        Parameters
        ----------
        value: Any
            The metadata value to format.

        Returns
        -------
        A string representation of the metadata value.
        """
        if isinstance(value, (list, tuple)):
            return f"[{', '.join(str(v) for v in value)}]"
        return str(value)

    def _join_text_and_metadata(self, text: str, metadata: dict) -> str:
        """Join text and metadata into a single string for embedding.

        Parameters
        ----------
        text: str
            The main content text.
        metadata: dict[str, Any]
            Dictionary of metadata.

        Returns
        -------
        Combined text with metadata appended.
        """
        metadata_str = " ".join(
            f"({key}: {self._format_metadata_value(value)})"
            for key, value in metadata.items()
            if key not in self.excluded_metadata
        )
        return f"{text}\nMetadata: {metadata_str}\n"

    async def _generate_context(self, document: str, chunk: str, previous_context: str | None = None, metadata: dict | None = None) -> str:
        """Generate contextual description for a chunk using LLM.

        Parameters
        ----------
        document: str
            The complete original document
        chunk: str
            The specific chunk to contextualize
        previous_context: str
            Context from previous chunk (if any)
        metadata: dict
            Metadata associated with the chunk

        Returns
        -------
        str
            A contextual description of the chunk
        """
        if not self.llm:
            raise ValueError(
                "LLM not provided. Cannot generate contextual descriptions."
            )

        messages = [{"role": "user", "content": chunk}]
        response = await self._invoke_prompt(
            "main", messages, document=document, previous_context=previous_context, metadata=metadata, response_model=str
        )

        # Handle both response formats - string or response object
        if isinstance(response, str):
            return response
        else:
            print(
                f"Used: {response.usage.prompt_tokens} total tokens ({response.usage.prompt_tokens_details.cached_tokens} cached) and output {response.usage.completion_tokens}"
            )
            print("\n\n")
            return response.choices[0].message.content

    @pn_cache
    def _chunk_text(self, text: str, chunk_size: int | None = None, chunk_func: Callable | None = None, **chunk_func_kwargs) -> list[str]:
        """Split text into chunks of size up to self.chunk_size.

        Parameters
        ----------
        text: str
            The text to split.
        chunk_size: int
            The maximum size of each chunk. If None, uses self.chunk_size.
        chunk_func: Callable
            The function to use for chunking. If None, uses self.chunk_func.

        Returns
        -------
        List of text chunks.
        """
        # Use provided parameters or fall back to instance defaults
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_func is None:
            chunk_func = self.chunk_func

        # Simple case: no chunking needed
        if chunk_size is None or len(text) <= chunk_size:
            return [text]

        try:
            chunks = chunk_func(text, chunk_size=chunk_size, **chunk_func_kwargs)
        except TypeError:
            # Fall back if chunk_size parameter isn't supported
            chunks = chunk_func(text, **chunk_func_kwargs)

        return chunks

    async def should_situate_chunk(self, chunk: str) -> bool:
        """
        Determine whether a chunk should be situated based on its content.

        Parameters
        ----------
        chunk: str
            The chunk text to evaluate

        Returns
        -------
        bool
            Whether the chunk should be situated
        """
        if not self.llm:
            return self.situate

        try:
            # Use a user message to avoid conflicts with system instructions
            messages = [{"role": "user", "content": chunk}]
            result = await self._invoke_prompt("should_situate",  messages)
            return result.yes
        except Exception as e:
            log_debug(f"Error determining if chunk should be situated: {e}")
            # Default to the class default in case of error
            return self.situate

    async def add(
        self,
        items: list[dict],
        force_ids: list[int] | None = None,
        situate: bool | None = None,
    ) -> list[int]:
        """
        Add items to the vector store.

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.
        force_ids: list[int] = None
            Optional list of IDs to use instead of generating new ones.
        situate: bool | None
            Whether to insert a `llm_context` key in the metadata containing
            contextual about the chunks. If None, uses the class default.

        Returns
        -------
        List of assigned IDs for the added items.
        """
        all_texts = []
        all_metadata = []
        text_and_metadata_list = []

        # Use the provided situate parameter or fall back to the class default
        use_situate = self.situate if situate is None else situate

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}

            # Split text into chunks
            content_chunks = self._chunk_text(
                text, self.chunk_size, self.chunk_func, **self.chunk_func_kwargs
            )

            # Skip situating if use_situate is False
            if not use_situate or len(content_chunks) <= 1:
                should_situate = False
            else:
                should_situate = True

            # Generate contextual descriptions if situate is enabled and multiple chunks exist
            chunk_contexts = {}
            if should_situate and self.llm:
                previous_context = None  # Start with no previous context
                for chunk in content_chunks:
                    needs_context = await self.should_situate_chunk(chunk)
                    if not needs_context:
                        continue

                    context = await self._generate_context(text, chunk, previous_context, metadata)
                    chunk_contexts[chunk] = context
                    previous_context = context  # Save this context for the next chunk
            elif should_situate and not self.llm:
                raise ValueError("LLM not provided. Cannot apply situate.")

            # Process each chunk with its context
            for chunk in content_chunks:
                chunk_metadata = metadata.copy()

                # Add context to metadata if situate is enabled and multiple chunks exist
                if should_situate and chunk in chunk_contexts:
                    chunk_metadata["llm_context"] = chunk_contexts[chunk]

                text_and_metadata = self._join_text_and_metadata(chunk, chunk_metadata)
                all_texts.append(chunk)
                all_metadata.append(chunk_metadata)
                text_and_metadata_list.append(text_and_metadata)

        # Get embeddings for all chunks
        embeddings = np.array(
            await self.embeddings.embed(text_and_metadata_list), dtype=np.float32
        )

        # Implement add logic in derived classes
        return await self._add_items(all_texts, all_metadata, embeddings, force_ids)

    @abstractmethod
    async def _add_items(
        self,
        texts: list[str],
        metadata: list[dict],
        embeddings: np.ndarray,
        force_ids: list[int] | None = None,
    ) -> list[int]:
        """
        Internal method to add items to the vector store.

        Parameters
        ----------
        texts: list[str]
            List of text chunks.
        metadata: list[dict]
            List of metadata dictionaries for each chunk.
        embeddings: np.ndarray
            Matrix of embedding vectors.
        force_ids: list[int] | None
            Optional list of IDs to use instead of generating new ones.

        Returns
        -------
        List of assigned IDs for the added items.
        """

    async def upsert(self, items: list[dict], situate: bool | None = None) -> list[int]:
        """
        Add items to the vector store if similar items don't exist,
        update them if they do.

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.
        situate: bool | None
            Whether to insert a `llm_context` key in the metadata containing
            contextual about the chunks. If None, uses the class default.

        Returns
        -------
        List of assigned IDs for the added or updated items.
        """
        # Implement in derived classes
        raise NotImplementedError("Subclasses must implement upsert.")

    async def add_directory(
        self,
        directory: str | os.PathLike,
        pattern: str = "*",
        exclude_patterns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        situate: bool | None = None,
        upsert: bool = False,
    ) -> list[int]:
        """
        Recursively add files from a directory that match the pattern and don't match exclude patterns.

        Parameters
        ----------
        directory : Union[str, os.PathLike]
            The path to the directory to search for files.
        pattern : str
            Glob pattern to match files against (e.g., "*.txt", "*.py"). Default is "*".
        exclude_patterns : Optional[List[str]]
            List of patterns to exclude. Files matching any of these patterns will be skipped.
        metadata : Optional[dict[str, Any]]
            Base metadata to apply to all files. Will be extended with filename-specific metadata.
        situate : Optional[bool]
            Whether to insert a `llm_context` key in the metadata. If None, uses the class default.
        upsert : bool
            If True, will update existing items if similar content is found. Default is False.

        Returns
        -------
        List[int]
            Combined list of IDs for all added files.
        """
        if exclude_patterns is None:
            exclude_patterns = []

        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Directory {directory} does not exist or is not a directory")
        file_paths = list(directory_path.rglob(pattern))

        base_metadata = metadata or {}

        # Collect files that match the pattern and don't match exclude patterns
        all_ids = []
        for file_path in tqdm(file_paths, unit="file", desc="Embedding files"):
            # Skip directories
            if not file_path.is_file():
                continue

            # Get the relative path for exclusion checking and metadata
            rel_path = file_path.relative_to(directory_path)

            # Check against exclusion patterns
            if any(fnmatch(str(rel_path), exclude_pattern) for exclude_pattern in exclude_patterns):
                continue

            # Create file-specific metadata by extending base metadata
            file_metadata = base_metadata.copy()
            file_metadata["filename"] = str(file_path)

            all_ids.extend(
                await self.add_file(
                    filename=file_path,
                    metadata=file_metadata,
                    situate=situate,
                    upsert=upsert
                )
            )
        return all_ids

    async def add_file(
        self,
        filename: str | IO | t.Any | os.PathLike,
        ext: str | None = None,
        metadata: dict[str, Any] | None = None,
        situate: bool | None = None,
        upsert: bool = False,
    ) -> list[int]:
        """
        Adds a file or a URL to the collection.

        Parameters
        ----------
        filename (str): str | os.PathLike | IO
            The path to the file, a file-like object or a URL to be added.
        ext : str | None
            The file extension to associate with the added file.
            If not provided, it will be determined from the file or URL.
        metadata : dict | None
            A dictionary containing metadata related to the file
            (e.g., title, author, description). Defaults to None.
        situate : bool | None
            Whether to insert a `llm_context` key in the metadata containing
            contextual about the chunks. If None, uses the class default.
        upsert: bool
            If True, will update existing items if similar content is found.
            Defaults to False.

        Returns
        -------
        List of assigned IDs for the added items.
        """
        from markitdown import FileConversionException, MarkItDown, StreamInfo

        if metadata is None:
            metadata = {}
        mdit = MarkItDown()

        # Run the potentially blocking file operations in a thread
        if isinstance(filename, str) and filename.startswith(("http://", "https://")):
            doc = await asyncio.to_thread(mdit.convert_url, filename)
        elif hasattr(filename, "read"):
            doc = await asyncio.to_thread(
                mdit.convert_stream, filename, file_extension=ext
            )
        else:
            if "filename" not in metadata:
                metadata["filename"] = str(filename)
            try:
                doc = await asyncio.to_thread(
                    mdit.convert_local, filename, file_extension=ext
                )
            except FileConversionException:  # for ascii issues
                with open(filename, encoding="utf-8") as f:
                    text_content = f.read()
                doc = await asyncio.to_thread(
                    mdit.convert_stream,
                    io.BytesIO(text_content.encode("utf-8")),
                    stream_info=StreamInfo(charset="utf-8"),
                    file_extension=ext,
                )

        kwargs = {"items": [{"text": doc.text_content, "metadata": metadata}], "situate": situate}
        if upsert:
            return await self.upsert(**kwargs)
        else:
            return await self.add(**kwargs)

    @abstractmethod
    async def query(
        self,
        text: str,
        top_k: int = 5,
        filters: dict | None = None,
        threshold: float = 0.0,
        situate: bool | None = None,
    ) -> list[dict]:
        """
        Query the vector store for similar items.

        Parameters
        ----------
        text : str
            The query text.
        top_k: int
            Number of top results to return.
        filters: dict | None
            Optional metadata filters.
        threshold: float
            Minimum similarity score required for a result to be included.
        situate: bool | None
            Whether to insert a `llm_context` key in the metadata containing
            contextual about the chunks. If None, uses the class default.

        Returns
        -------
        List of results with 'id', 'text', 'metadata', and 'similarity' score.
        """

    @abstractmethod
    def filter_by(
        self, filters: dict, limit: int | None = None, offset: int = 0
    ) -> list[dict]:
        """
        Filter items by metadata without using embeddings similarity.

        Parameters
        ----------
        filters: dict[str, str]
            Dictionary of metadata key-value pairs to filter by.
        limit: int | None
            Maximum number of results to return. If None, returns all matches.
        offset: int
            Number of results to skip (for pagination).

        Returns
        -------
        List of results with 'id', 'text', and 'metadata'.
        """

    @abstractmethod
    def delete(self, ids: list[int]) -> None:
        """
        Delete items from the vector store by their IDs.

        Parameters
        ----------
        ids: list[int]
            List of IDs to delete.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all items from the vector store."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the vector store."""

    @abstractmethod
    def close(self) -> None:
        """Close the vector store and release any resources."""


class NumpyVectorStore(VectorStore):
    """
    Vector store implementation using NumPy for in-memory storage.

    :Example:

    .. code-block:: python

        from lumen.ai.vector_store import NumpyVectorStore

        vector_store = NumpyVectorStore()
        vector_store.add_file('https://lumen.holoviz.org')
        vector_store.query('LLM', threshold=0.1)

    Use upsert to avoid adding duplicate content:

    .. code-block:: python

        from lumen.ai.vector_store import NumpyVectorStore

        vector_store = NumpyVectorStore()
        vector_store.upsert([{'text': 'Hello!', 'metadata': {'source': 'greeting'}}])
        # Won't add duplicate if content is similar and metadata matches
        vector_store.upsert([{'text': 'Hello!', 'metadata': {'source': 'greeting'}}])
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.vectors = None
        self.texts: list[str] = []
        self.metadata: list[dict] = []
        self.ids: list[int] = []
        self._current_id: int = 0

    def _get_next_id(self) -> int:
        """Generate the next available ID.

        Returns
        -------
        The next unique integer ID.
        """
        self._current_id += 1
        return self._current_id

    def _cosine_similarity(
        self, query_vector: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity between query vector and stored vectors.

        Parameters
        ----------
        query_vector: np.ndarray
            The query embedding vector.
        vectors: np.ndarray
            Array of stored embedding vectors.

        Returns
        -------
        Array of cosine similarity scores.
        """
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(vectors))

        query_normalized = query_vector / query_norm

        if len(vectors) > 0:
            vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors_norm[vectors_norm == 0] = 1
            vectors_normalized = vectors / vectors_norm
            similarities = np.dot(vectors_normalized, query_normalized)
        else:
            similarities = np.array([])

        return similarities

    async def _add_items(
        self,
        texts: list[str],
        metadata: list[dict],
        embeddings: np.ndarray,
        force_ids: list[int] | None = None,
    ) -> list[int]:
        """
        Internal method to add items to the vector store.

        Parameters
        ----------
        texts: list[str]
            List of text chunks.
        metadata: list[dict]
            List of metadata dictionaries for each chunk.
        embeddings: np.ndarray
            Matrix of embedding vectors.
        force_ids: list[int] | None
            Optional list of IDs to use instead of generating new ones.

        Returns
        -------
        List of assigned IDs for the added items.
        """
        if force_ids is not None:
            # Use the provided IDs
            if len(force_ids) != len(texts):
                raise ValueError(
                    f"force_ids length ({len(force_ids)}) must match number of chunks ({len(texts)})"
                )
            new_ids = force_ids
            # Update _current_id if necessary
            self._current_id = (
                max(self._current_id, *force_ids) if force_ids else self._current_id
            )
        else:
            # Generate new IDs
            new_ids = [self._get_next_id() for _ in texts]

        if self.vectors is not None:
            embeddings = np.vstack([self.vectors, embeddings])
        self.vectors = embeddings
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        self.ids.extend(new_ids)

        return new_ids

    async def query(
        self,
        text: str,
        top_k: int = 5,
        filters: dict | None = None,
        threshold: float = 0.0,
    ) -> list[dict]:
        """
        Query the vector store for similar items.

        Parameters
        ----------
        text : str
            The query text.
        top_k: int
            Number of top results to return.
        filters: dict | None
            Optional metadata filters.
        threshold: float
            Minimum similarity score required for a result to be included.

        Returns
        -------
        List of results with 'id', 'text', 'metadata', and 'similarity' score.
        """
        if self.vectors is None:
            return []
        query_embedding = np.array((await self.embeddings.embed([text]))[0], dtype=np.float32)
        similarities = self._cosine_similarity(query_embedding, self.vectors)

        if filters and len(self.vectors) > 0:
            mask = np.ones(len(self.vectors), dtype=bool)
            for key, value in filters.items():
                mask &= np.array([item.get(key) == value for item in self.metadata])
            similarities = np.where(
                mask, similarities, -1.0
            )  # make filtered similarity values == -1

        results = []
        if len(similarities) > 0:
            sorted_indices = np.argsort(similarities)[::-1]
            for idx in sorted_indices:
                similarity = similarities[idx]
                if similarity < threshold:
                    continue
                results.append(
                    {
                        "id": self.ids[idx],
                        "text": self.texts[idx],
                        "metadata": self.metadata[idx],
                        "similarity": float(similarity),
                    }
                )
                if len(results) >= top_k:
                    break
        return results

    def filter_by(
        self, filters: dict, limit: int | None = None, offset: int = 0
    ) -> list[dict]:
        """
        Filter items by metadata without using embeddings similarity.

        Parameters
        ----------
        filters: dict[str, str]
            Dictionary of metadata key-value pairs to filter by.
        limit: int | None
            Maximum number of results to return. If None, returns all matches.
        offset: int
            Number of results to skip (for pagination).

        Returns
        -------
        List of results with 'id', 'text', and 'metadata'.
        """
        if not self.metadata or self.vectors is None:
            return []

        mask = np.ones(len(self.metadata), dtype=bool)
        for key, value in filters.items():
            mask &= np.array([item.get(key) == value for item in self.metadata])

        matching_indices = np.where(mask)[0]

        if offset:
            matching_indices = matching_indices[offset:]
        if limit is not None:
            matching_indices = matching_indices[:limit]

        return [
            {
                "id": self.ids[idx],
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
            }
            for idx in matching_indices
        ]

    def delete(self, ids: list[int]) -> None:
        """
        Delete items from the vector store by their IDs.

        Parameters
        ----------
        ids: list[int]
            List of IDs to delete.
        """
        if not ids or self.vectors is None:
            return

        keep_mask = np.ones(len(self.vectors), dtype=bool)
        id_set = set(ids)
        for idx, item_id in enumerate(self.ids):
            if item_id in id_set:
                keep_mask[idx] = False

        self.vectors = self.vectors[keep_mask]
        self.texts = [text for i, text in enumerate(self.texts) if keep_mask[i]]
        self.metadata = [meta for i, meta in enumerate(self.metadata) if keep_mask[i]]
        self.ids = [id_ for i, id_ in enumerate(self.ids) if keep_mask[i]]

    async def upsert(self, items: list[dict], situate: bool | None = None) -> list[int]:
        """
        Add items to the vector store if similar items don't exist,
        update them if they do.

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.
        situate: bool | None
            Whether to insert a `llm_context` key in the metadata containing
            contextual about the chunks. If None, uses the class default.

        Returns
        -------
        List of assigned IDs for the added or updated items.
        """
        if not items:
            return []

        if self.vectors is None or len(self.vectors) == 0:
            return await self.add(items)

        assigned_ids = []
        items_to_add = []

        # Create text-to-indices mapping for fast lookups
        text_to_indices = {}
        for idx, text in enumerate(self.texts):
            if text not in text_to_indices:
                text_to_indices[text] = []
            text_to_indices[text].append(idx)

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}

            # Check for exact text match
            match_indices = text_to_indices.get(text, [])

            # If no exact match found, check for chunked text match
            if not match_indices:
                for chunk in self._chunk_text(text, self.chunk_size, self.chunk_func, **self.chunk_func_kwargs):
                    if chunk in text_to_indices:
                        match_indices.extend(text_to_indices[chunk])

            if match_indices:
                match_found = False

                for idx in match_indices:
                    existing_id = self.ids[idx]
                    existing_meta = self.metadata[idx]

                    # Check for metadata value conflicts (same keys, different values)
                    has_value_conflict = False
                    common_keys = set(metadata.keys()) & set(existing_meta.keys())
                    for key in common_keys:
                        if metadata[key] != existing_meta[key]:
                            has_value_conflict = True
                            break

                    if has_value_conflict:
                        # If values conflict, this is not a match
                        continue

                    # If metadata keys are different but no value conflicts, update existing
                    if set(metadata.keys()) != set(existing_meta.keys()):
                        self.delete([existing_id])
                        new_ids = await self.add(
                            [{"text": text, "metadata": metadata}],
                            force_ids=[existing_id],
                            situate=situate,
                        )
                        assigned_ids.extend(new_ids)
                    else:
                        # Exact metadata match - reuse existing
                        assigned_ids.append(existing_id)

                    match_found = True
                    break

                if match_found:
                    continue

            # If no exact match, add as new item
            items_to_add.append(item)

        if items_to_add:
            new_ids = await self.add(items_to_add, situate=situate)
            assigned_ids.extend(new_ids)
        else:
            log_debug("All items already exist in the vector store.")

        return assigned_ids

    def clear(self) -> None:
        """
        Clear all items from the vector store.

        Resets the vectors, texts, metadata, and IDs to their initial empty states.
        """
        self.vectors = None
        self.texts = []
        self.metadata = []
        self.ids = []
        self._current_id = 0

    def __len__(self) -> int:
        return len(self.texts) if self.vectors is not None else 0

    def close(self) -> None:
        """Close the vector store and release any resources.

        For NumpyVectorStore, this clears vectors to free memory.
        """
        self.vectors = None
        self.texts = []
        self.metadata = []
        self.ids = []
        self._current_id = 0


class DuckDBVectorStore(VectorStore):
    """
    Vector store implementation using DuckDB for persistent storage.

    :Example:

    .. code-block:: python

        from lumen.ai.vector_store import DuckDBStore

        vector_store = DuckDBStore(uri=':memory:)
        vector_store.add_file('https://lumen.holoviz.org')
        vector_store.query('LLM', threshold=0.1)

    Use upsert to avoid adding duplicate content:

    .. code-block:: python

        from lumen.ai.vector_store import DuckDBStore

        vector_store = DuckDBStore(uri=':memory:)
        vector_store.upsert([{'text': 'Hello!', 'metadata': {'source': 'greeting'}}])
        # Won't add duplicate if content is similar and metadata matches
        vector_store.upsert([{'text': 'Hello!', 'metadata': {'source': 'greeting'}}])
    """

    uri = param.String(default=":memory:", doc="The URI of the DuckDB database")

    embeddings = param.ClassSelector(
        class_=Embeddings,
        default=None,
        allow_None=True,
        doc="Embeddings object for text processing. If None and a URI is provided, loads from the database; else NumpyEmbeddings.",
    )

    def __init__(self, **params):
        super().__init__(**params)

        connection = duckdb.connect(":memory:")
        # following the instructions from
        # https://duckdb.org/docs/stable/extensions/vss.html#persistence
        connection.execute("INSTALL 'vss';")
        connection.execute("LOAD 'vss';")
        connection.execute("SET hnsw_enable_experimental_persistence = true;")

        if self.uri == ":memory:":
            self.connection = connection
            self._initialized = False
            if self.embeddings is None:
                self.embeddings = NumpyEmbeddings()
            return
        uri_exists = Path(self.uri).exists()
        try:
            connection.execute(f"ATTACH DATABASE '{self.uri}' AS embedded;")
        except duckdb.CatalogException:
            # handle "Failure while replaying WAL file"
            # remove .wal uri on corruption
            wal_path = Path(str(self.uri) + ".wal")
            if wal_path.exists():
                wal_path.unlink()
            connection.execute(f"ATTACH DATABASE '{self.uri}' AS embedded;")
        connection.execute("USE embedded;")
        self.connection = connection
        has_documents = (
            connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'documents';"
            ).fetchone()[0]
            > 0
        )
        self._initialized = uri_exists and has_documents

        if self.uri != ":memory:" and self._initialized:
            config = self._get_embeddings_config()
            if config and self.embeddings is None:
                module_name, class_name = config["class"].rsplit(".", 1)
                module = importlib.import_module(module_name)
                embedding_class = getattr(module, class_name)
                self.embeddings = embedding_class(**config["params"])
                log_debug(f"Loaded embeddings {class_name} from database.")
            self._check_embeddings_consistency()

        if self.embeddings is None:
            self.embeddings = NumpyEmbeddings()

    def _setup_database(self, embedding_dim: int) -> None:
        """Set up the DuckDB database with necessary tables and indexes."""
        self.connection.execute("CREATE SEQUENCE IF NOT EXISTS documents_id_seq;")

        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGINT DEFAULT NEXTVAL('documents_id_seq') PRIMARY KEY,
                text VARCHAR,
                metadata JSON,
                embedding FLOAT[{embedding_dim}]
            );
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_store_metadata (
                key VARCHAR PRIMARY KEY,
                value JSON
            );
            """
        )

        # Store embedding configuration
        embedding_info = {
            "class": self.embeddings.__class__.__module__ + "." + self.embeddings.__class__.__name__,
            "params": {}
        }
        for param_name, param_obj in self.embeddings.param.objects().items():
            if param_name not in ['name']:
                value = getattr(self.embeddings, param_name)
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    embedding_info["params"][param_name] = value

        self.connection.execute(
            """
            INSERT OR REPLACE INTO vector_store_metadata (key, value)
            VALUES ('embeddings', ?::JSON);
            """,
            [json.dumps(embedding_info)]
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS embedding_index
            ON documents USING HNSW (embedding) WITH (metric = 'cosine');
            """
        )
        self._initialized = True

    def _check_embeddings_consistency(self):
        """
        Check if the provided embeddings are consistent with the stored configuration.
        Raises ValueError if there's a mismatch that would cause empty query results.
        """
        # Check if metadata table exists
        stored_config = self._get_embeddings_config() or {"class": "", "params": {}}
        stored_class = stored_config["class"]
        stored_params = stored_config["params"]

        # Get current embeddings class
        current_class = self.embeddings.__class__.__module__ + "." + self.embeddings.__class__.__name__

        # Check if classes match
        if current_class != stored_class:
            raise ValueError(
                f"Provided embeddings class '{current_class}' does not match the stored class "
                f"'{stored_class}' for this vector store. This would result in empty query results. "
                f"Use compatible embeddings or create a new vector store."
            )

        # Check if critical parameters match
        for param_name, stored_value in stored_params.items():
            if hasattr(self.embeddings, param_name):
                current_value = getattr(self.embeddings, param_name)
                if current_value != stored_value and param_name in ['model', 'embedding_dim', 'chunk_size']:
                    raise ValueError(
                        f"Provided embeddings parameter '{param_name}' value '{current_value}' "
                        f"does not match stored value '{stored_value}'. This would result in "
                        f"empty query results. Use compatible embeddings or create a new vector store."
                    )


    def _get_embeddings_config(self):
        """
        Get the embeddings configuration stored in the vector store.

        Returns
        -------
        dict or None
            The embeddings configuration or None if not available.
        """
        if not self._initialized:
            return None

        # Check if metadata table exists
        has_metadata = (
            self.connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'vector_store_metadata';"
            ).fetchone()[0]
            > 0
        )

        if not has_metadata:
            return None

        try:
            result = self.connection.execute(
                "SELECT value FROM vector_store_metadata WHERE key = 'embeddings';"
            ).fetchone()

            if result:
                return json.loads(result[0])
            return None
        except Exception as e:
            log_debug(f"Error retrieving embeddings configuration: {e}")
            return None

    async def _add_items(
        self,
        texts: list[str],
        metadata: list[dict],
        embeddings: np.ndarray,
        force_ids: list[int] | None = None,
    ) -> list[int]:
        """
        Internal method to add items to the vector store.

        Parameters
        ----------
        texts: list[str]
            List of text chunks.
        metadata: list[dict]
            List of metadata dictionaries for each chunk.
        embeddings: np.ndarray
            Matrix of embedding vectors.
        force_ids: list[int] | None
            Optional list of IDs to use instead of generating new ones.

        Returns
        -------
        List of assigned IDs for the added items.
        """
        # Validate force_ids if provided
        if force_ids is not None and len(force_ids) != len(texts):
            raise ValueError(
                f"force_ids length ({len(force_ids)}) must match number of chunks ({len(texts)})"
            )

        # Set up database if not initialized
        if not self._initialized and len(texts) > 0:
            vector_dim = embeddings.shape[1]
            self._setup_database(vector_dim)

        text_ids = []

        for i in range(len(texts)):
            vector = np.array(embeddings[i], dtype=np.float32)

            # Prepare parameters and query
            if force_ids is not None:
                query = """
                    INSERT INTO documents (id, text, metadata, embedding)
                    VALUES (?, ?, ?::JSON, ?) RETURNING id;
                    """
                params = [
                    force_ids[i],
                    texts[i],
                    json.dumps(metadata[i]),
                    vector.tolist(),
                ]
            else:
                query = """
                    INSERT INTO documents (text, metadata, embedding)
                    VALUES (?, ?::JSON, ?) RETURNING id;
                    """
                params = [texts[i], json.dumps(metadata[i]), vector.tolist()]

            # Run the potentially blocking DB operation in a thread
            async with self._add_items_lock:
                result = await asyncio.to_thread(self._execute_query, query, params)
                text_ids.append(result)

        return text_ids

    async def query(
        self,
        text: str,
        top_k: int = 5,
        filters: dict | None = None,
        threshold: float = 0.0,
    ) -> list[dict]:
        """
        Query the vector store for similar items.

        Parameters
        ----------
        text : str
            The query text.
        top_k: int
            Number of top results to return.
        filters: dict | None
            Optional metadata filters.
        threshold: float
            Minimum similarity score required for a result to be included.

        Returns
        -------
        List of results with 'id', 'text', 'metadata', and 'similarity' score.
        """
        if not self._initialized:
            return []
        query_embedding = np.array(
            (await self.embeddings.embed([text]))[0], dtype=np.float32
        ).tolist()
        vector_dim = len(query_embedding)

        base_query = f"""
            SELECT id, text, metadata,
                array_cosine_similarity(embedding, ?::REAL[{vector_dim}]) AS similarity
            FROM documents
            WHERE array_cosine_similarity(embedding, ?::REAL[{vector_dim}]) >= ?
        """
        params = [query_embedding, query_embedding, threshold]

        if filters:
            for key, value in filters.items():
                base_query += f" AND json_extract_string(metadata, '$.{key}') = ?"
                params.append(str(value))

        base_query += """
            ORDER BY similarity DESC
            LIMIT ?;
        """
        params.append(top_k)

        try:
            result = self.connection.execute(base_query, params).fetchall()
            return [
                {
                    "id": row[0],
                    "text": row[1],
                    "metadata": json.loads(row[2]),
                    "similarity": row[3],
                }
                for row in result
            ]
        except duckdb.Error as e:
            log_debug(f"Error during query: {e}")
            return []

    def filter_by(
        self, filters: dict, limit: int | None = None, offset: int = 0
    ) -> list[dict]:
        """
        Filter items by metadata without using embeddings similarity.

        Parameters
        ----------
        filters: dict[str, str]
            Dictionary of metadata key-value pairs to filter by.
        limit: int | None
            Maximum number of results to return. If None, returns all matches.
        offset: int
            Number of results to skip (for pagination).

        Returns
        -------
        List of results with 'id', 'text', and 'metadata'.
        """
        if not self._initialized:
            return []
        base_query = """
            SELECT id, text, metadata
            FROM documents
            WHERE 1=1
        """
        params = []

        for key, value in filters.items():
            base_query += f" AND json_extract_string(metadata, '$.{key}') = ?"
            params.append(str(value))

        if offset:
            base_query += " OFFSET ?"
            params.append(offset)

        if limit is not None:
            base_query += " LIMIT ?"
            params.append(limit)

        base_query += ";"

        result = self.connection.execute(base_query, params).fetchall()

        return [
            {
                "id": row[0],
                "text": row[1],
                "metadata": json.loads(row[2]),
            }
            for row in result
        ]

    def delete(self, ids: list[int]) -> None:
        """
        Delete items from the vector store by their IDs.

        Parameters
            ids: List of IDs to delete.
        """
        if not ids or not self._initialized:
            return

        placeholders = ", ".join(["?"] * len(ids))
        query = f"DELETE FROM documents WHERE id IN ({placeholders});"
        self.connection.execute(query, ids)

    def clear(self) -> None:
        """
        Clear all entries and reset sequence.

        Drops the documents table and sequence, then sets up the database again.
        """
        if not self._initialized:
            return
        self.connection.execute("DROP TABLE IF EXISTS documents;")
        self.connection.execute("DROP SEQUENCE IF EXISTS documents_id_seq;")
        self._initialized = False

    async def upsert(self, items: list[dict], situate: bool | None = None) -> list[int]:
        """
        Add items to the vector store if similar items don't exist,
        update them if they do.

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.
        situate: bool | None
            Whether to insert a `llm_context` key in the metadata containing
            contextual about the chunks. If None, uses the class default.

        Returns
        -------
        List of assigned IDs for the added or updated items.
        """
        if not items:
            return []

        if not self._initialized:
            return await self.add(items)

        assigned_ids = []
        items_to_add = []

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}

            # Check for exact text match
            query = """
                SELECT id, metadata
                FROM documents
                WHERE text = ?
            """

            # Execute the query in a thread
            result = await asyncio.to_thread(
                self._execute_query, query, [text], fetchall=True
            )

            # If no exact match, check for chunked text match
            if not result:
                chunked_results = []
                for chunk in self._chunk_text(text, self.chunk_size, self.chunk_func, **self.chunk_func_kwargs):
                    chunk_results = await asyncio.to_thread(
                        self._execute_query, query, [chunk], fetchall=True
                    )
                    chunked_results.extend(chunk_results)
                result = chunked_results

            match_found = False
            for row in result:
                item_id = row[0]
                existing_metadata = json.loads(row[1])

                # Check for metadata value conflicts
                has_value_conflict = False
                common_keys = set(metadata.keys()) & set(existing_metadata.keys())
                for key in common_keys:
                    if metadata[key] != existing_metadata[key]:
                        has_value_conflict = True
                        break

                if has_value_conflict:
                    # If values conflict, try next match
                    continue

                # If metadata keys are different but no value conflicts, update existing
                if set(metadata.keys()) != set(existing_metadata.keys()):
                    self.delete([item_id])
                    new_ids = await self.add(
                        [{"text": text, "metadata": metadata}], force_ids=[item_id]
                    )
                    assigned_ids.extend(new_ids)
                else:
                    # Exact metadata match - reuse existing
                    assigned_ids.append(item_id)

                match_found = True
                break

            if not match_found:
                # No exact text match, add as new
                items_to_add.append(item)

        if items_to_add:
            new_ids = await self.add(items_to_add, situate=situate)
            assigned_ids.extend(new_ids)
            log_debug(f"Added {len(items_to_add)} new items to the vector store.")

        return assigned_ids

    def _execute_query(self, query, params, fetchall=False):
        """Execute a DuckDB query and return results.

        Parameters
        ----------
        query: str
            SQL query to execute
        params: list
            Parameters for the query
        fetchall: bool
            If True, return all results; otherwise return the first column of the first row

        Returns
        -------
        Query results based on the fetchall parameter
        """
        result = self.connection.execute(query, params)
        if fetchall:
            return result.fetchall()
        return result.fetchone()[0]

    def __len__(self) -> int:
        if not self._initialized:
            return 0
        result = self.connection.execute("SELECT COUNT(*) FROM documents;").fetchone()
        return result[0]

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
