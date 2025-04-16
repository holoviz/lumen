import json

from abc import abstractmethod
from pathlib import Path

import duckdb
import numpy as np
import param

from .embeddings import Embeddings, NumpyEmbeddings


class VectorStore(param.Parameterized):
    """Abstract base class for a vector store."""

    chunk_size = param.Integer(
        default=1024, doc="Maximum size of text chunks to split documents into."
    )

    embeddings = param.ClassSelector(
        class_=Embeddings,
        default=NumpyEmbeddings(),
        doc="Embeddings object for text processing.",
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
            f"({key}: {self._format_metadata_value(value)})" for key, value in metadata.items()
        )
        return f"{text} {metadata_str}"

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks of size up to self.chunk_size.

        Parameters
        ----------
        text: str
            The text to split.

        Returns
        -------
        List of text chunks.
        """
        if self.chunk_size is None or len(text) <= self.chunk_size:
            return [text]

        words = text.split()
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                current_chunk += (" " + word) if current_chunk else word
            else:
                chunks.append(current_chunk)
                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    @abstractmethod
    def add(self, items: list[dict]) -> list[int]:
        """
        Add items to the vector store.

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.

        Returns
        -------
        List of assigned IDs for the added items.
        """

    @abstractmethod
    def upsert(self, items: list[dict]) -> list[int]:
        """
        Add items to the vector store if similar items don't exist, update them if they do.

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.

        Returns
        -------
        List of assigned IDs for the added or updated items.
        """

    def add_file(self, filename, ext=None, metadata=None) -> list[int]:
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

        Returns
        -------
        List of assigned IDs for the added items.
        """
        from markitdown import MarkItDown
        if metadata is None:
            metadata = {}
        mdit = MarkItDown()
        if isinstance(filename, str) and filename.startswith(('http://', 'https://')):
            doc = mdit.convert_url(filename)
        elif hasattr(filename, 'read'):
            doc = mdit.convert_stream(filename, file_extension=ext)
        else:
            if 'filename' not in metadata:
                metadata['filename'] = filename
            doc = mdit.convert_local(filename, file_extension=ext)
        return self.add([{'text': doc.text_content, 'metadata': metadata}])

    @abstractmethod
    def query(
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

    def add(self, items: list[dict], force_ids: list[int] = None) -> list[int]:
        """
        Add items to the vector store.

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.
        force_ids: list[int] = None
            Optional list of IDs to use instead of generating new ones.
            Must be the same length as flattened items after chunking.

        Returns
        -------
        List of assigned IDs for the added items.
        """
        all_texts = []
        all_metadata = []
        text_and_metadata_list = []

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}

            content_chunks = self._chunk_text(text)
            for chunk in content_chunks:
                text_and_metadata = self._join_text_and_metadata(chunk, metadata)
                all_texts.append(chunk)
                all_metadata.append(metadata)
                text_and_metadata_list.append(text_and_metadata)

        embeddings = np.array(self.embeddings.embed(text_and_metadata_list), dtype=np.float32)

        if force_ids is not None:
            # Use the provided IDs
            if len(force_ids) != len(all_texts):
                raise ValueError(f"force_ids length ({len(force_ids)}) must match number of chunks ({len(all_texts)})")
            new_ids = force_ids
            # Update _current_id if necessary
            self._current_id = max(self._current_id, max(force_ids)) if force_ids else self._current_id
        else:
            # Generate new IDs
            new_ids = [self._get_next_id() for _ in all_texts]

        if self.vectors is not None:
            embeddings = np.vstack([self.vectors, embeddings])
        self.vectors = embeddings
        self.texts.extend(all_texts)
        self.metadata.extend(all_metadata)
        self.ids.extend(new_ids)

        return new_ids

    def query(
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
        query_embedding = np.array(self.embeddings.embed([text])[0], dtype=np.float32)
        similarities = self._cosine_similarity(query_embedding, self.vectors)

        if filters and len(self.vectors) > 0:
            mask = np.ones(len(self.vectors), dtype=bool)
            for key, value in filters.items():
                mask &= np.array([item.get(key) == value for item in self.metadata])
            similarities = np.where(mask, similarities, -1.0)  # make filtered similarity values == -1

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

    def _find_match(self, text: str, metadata: dict) -> tuple[int, float]:
        """Find a matching document based on text and metadata.

        Parameters
        ----------
        text: str
            Text content to match.
        metadata: dict
            Metadata to match.

        Returns
        -------
        Tuple of (item_id, similarity_score).
        Returns (-1, 0.0) if no match is found.
        """
        # Check for exact metadata match
        if metadata:
            filtered_results = self.filter_by(metadata, limit=1)
            if filtered_results and filtered_results[0]["text"] == text:
                return filtered_results[0]["id"], 1.0

        # Check for exact text match with compatible metadata
        all_docs = self.filter_by({})
        for doc in all_docs:
            if doc["text"] != text:
                continue

            # Check if metadata is compatible
            existing_metadata = doc["metadata"]
            compatible = True

            for key in existing_metadata:
                if key in metadata and existing_metadata[key] != metadata[key]:
                    compatible = False
                    break

            if compatible:
                return doc["id"], 1.0

        # Fallback to similarity search
        text_with_metadata = self._join_text_and_metadata(text, metadata)
        similar_items = self.query(text_with_metadata, top_k=1, threshold=0.95)

        if not similar_items:
            return -1, 0.0

        # Check if metadata is compatible
        existing_item = similar_items[0]
        existing_metadata = existing_item["metadata"]
        similarity = existing_item["similarity"]

        if similarity < 0.98:
            return -1, 0.0

        # Check if existing metadata is compatible with new metadata
        for key in existing_metadata:
            if key in metadata and existing_metadata[key] != metadata[key]:
                return -1, 0.0

        return existing_item["id"], similarity

    def _similarity_search_with_embedding(self, embedding, top_k=5, threshold=0.0):
        """
        Perform similarity search using a pre-computed embedding.
        This avoids re-computing embeddings when we already have them.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        similarities = self._cosine_similarity(embedding, self.vectors)

        # Early exit if we have no similarities above threshold
        if max(similarities) < threshold:
            return []

        # Get top matches efficiently
        sorted_indices = np.argsort(similarities)[::-1]
        results = []

        for idx in sorted_indices:
            similarity = similarities[idx]
            if similarity < threshold:
                break  # Early exit - remaining similarities are below threshold

            results.append({
                "id": self.ids[idx],
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(similarity),
            })

            if len(results) >= top_k:
                break

        return results

    def upsert(self, items: list[dict]) -> list[int]:
        """
        Add items to the vector store if similar items don't exist, update them if they do.

        This optimized implementation reduces complexity and improves efficiency by:
        1. Computing embeddings only once
        2. Using a simplified match strategy with early exits
        3. Avoiding unnecessary delete and re-add operations

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.

        Returns
        -------
        List of assigned IDs for the added or updated items.
        """
        if not items:
            return []

        # Prepare data for batch processing
        texts = []
        metadata_list = []
        text_and_metadata_list = []

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}
            texts.append(text)
            metadata_list.append(metadata)
            text_and_metadata_list.append(self._join_text_and_metadata(text, metadata))

        # Pre-compute embeddings for all items at once
        item_embeddings = self.embeddings.embed(text_and_metadata_list)

        assigned_ids = []
        new_items = []
        new_item_indices = []

        # First pass: check for exact text matches (fast path)
        existing_items = self.filter_by({})
        text_to_id_map = {item["text"]: item["id"] for item in existing_items}
        id_to_metadata_map = {item["id"]: item["metadata"] for item in existing_items}

        # Process each item
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadata_list, item_embeddings)):
            # Fast path: check exact text match
            if text not in text_to_id_map:
                new_items.append({"text": text, "metadata": metadata})
                new_item_indices.append(i)
                continue

            # We have a text match, check metadata compatibility
            item_id = text_to_id_map[text]
            existing_metadata = id_to_metadata_map[item_id]

            # Check if metadata keys are different (added or removed keys)
            if set(metadata.keys()) != set(existing_metadata.keys()):
                # Delete existing and re-add with new metadata
                self.delete([item_id])
                new_ids = self.add([{"text": text, "metadata": metadata}], force_ids=[item_id])
                assigned_ids.extend(new_ids)
                continue

            # Check if any metadata value conflicts for existing keys
            if any(key in existing_metadata and existing_metadata[key] != value
                   for key, value in metadata.items()):
                new_items.append({"text": text, "metadata": metadata})
                new_item_indices.append(i)
                continue

            # Use existing item
            assigned_ids.append(item_id)

        # Process remaining items that need similarity search
        for i, item in zip(new_item_indices, new_items):
            query_embedding = item_embeddings[i]
            similar_items = self._similarity_search_with_embedding(
                query_embedding, top_k=1, threshold=0.95
            )

            # No similar items found - add as new
            if not similar_items or similar_items[0]["similarity"] < 0.98:
                new_ids = self.add([item])
                assigned_ids.extend(new_ids)
                continue

            # Check metadata compatibility
            existing_item = similar_items[0]
            existing_metadata = existing_item["metadata"]

            # If any metadata conflicts, add as new
            if any(key in existing_metadata and existing_metadata[key] != value
                   for key, value in item["metadata"].items()):
                new_ids = self.add([item])
                assigned_ids.extend(new_ids)
                continue

            # Reuse existing item
            assigned_ids.append(existing_item["id"])

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

    def __init__(self, **params):
        super().__init__(**params)
        connection = duckdb.connect(':memory:')
        # following the instructions from
        # https://duckdb.org/docs/stable/extensions/vss.html#persistence
        connection.execute("INSTALL 'vss';")
        connection.execute("LOAD 'vss';")
        connection.execute("SET hnsw_enable_experimental_persistence = true;")

        if self.uri == ':memory:':
            self.connection = connection
            self._initialized = False
            return
        uri_exists = Path(self.uri).exists()
        connection.execute(f"ATTACH DATABASE '{self.uri}' AS embedded;")
        connection.execute("USE embedded;")
        self.connection = connection
        has_documents = connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'documents';"
        ).fetchone()[0] > 0
        self._initialized = uri_exists and has_documents

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
            CREATE INDEX IF NOT EXISTS embedding_index
            ON documents USING HNSW (embedding) WITH (metric = 'cosine');
            """
        )
        self._initialized = True

    def add(self, items: list[dict], force_ids: list[int] = None) -> list[int]:
        """
        Add items to the vector store.

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.
        force_ids: list[int] = None
            Optional list of IDs to use instead of generating new ones.
            Must match the number of chunks created after chunking.

        Returns
        -------
        List of assigned IDs for the added items.
        """
        all_texts = []
        all_metadata = []
        text_and_metadata_list = []

        # First, chunk all items to determine total number of chunks
        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}

            content_chunks = self._chunk_text(text)
            for chunk in content_chunks:
                text_and_metadata = self._join_text_and_metadata(chunk, metadata)
                all_texts.append(chunk)
                all_metadata.append(metadata)
                text_and_metadata_list.append(text_and_metadata)

        # Validate force_ids if provided
        if force_ids is not None:
            if len(force_ids) != len(all_texts):
                raise ValueError(f"force_ids length ({len(force_ids)}) must match number of chunks ({len(all_texts)})")

        embeddings = self.embeddings.embed(text_and_metadata_list)
        text_ids = []

        for i in range(len(all_texts)):
            vector = np.array(embeddings[i], dtype=np.float32)
            if not self._initialized:
                self._setup_database(len(vector))

            # Use the force_id if provided
            if force_ids is not None:
                # Explicitly set the ID
                result = self.connection.execute(
                    """
                    INSERT INTO documents (id, text, metadata, embedding)
                    VALUES (?, ?, ?::JSON, ?) RETURNING id;
                    """,
                    [
                        force_ids[i],
                        all_texts[i],
                        json.dumps(all_metadata[i]),
                        vector.tolist(),
                    ],
                )
                fetched = result.fetchone()
                if fetched:
                    text_ids.append(fetched[0])
                else:
                    raise ValueError("Failed to insert item with specified ID into DuckDB.")
            else:
                # Let the database generate the ID
                result = self.connection.execute(
                    """
                    INSERT INTO documents (text, metadata, embedding)
                    VALUES (?, ?::JSON, ?) RETURNING id;
                    """,
                    [
                        all_texts[i],
                        json.dumps(all_metadata[i]),
                        vector.tolist(),
                    ],
                )
                fetched = result.fetchone()
                if fetched:
                    text_ids.append(fetched[0])
                else:
                    raise ValueError("Failed to insert item into DuckDB.")

        return text_ids

    def query(
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
        query_embedding = np.array(self.embeddings.embed([text])[0], dtype=np.float32).tolist()
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
            print(f"Error during query: {e}")
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

    def _find_match(self, text: str, metadata: dict) -> tuple[int, float]:
        """Find a matching document based on text and metadata.

        Parameters
        ----------
        text: str
            Text content to match.
        metadata: dict
            Metadata to match.

        Returns
        -------
        Tuple of (item_id, similarity_score).
        Returns (-1, 0.0) if no match is found.
        """
        if not self._initialized:
            return -1, 0.0

        # Check for exact metadata match
        if metadata:
            filtered_results = self.filter_by(metadata, limit=1)
            if filtered_results and filtered_results[0]["text"] == text:
                return filtered_results[0]["id"], 1.0

        # Check for exact text match with compatible metadata
        # Build a SQL query to find exact text matches
        query = """
            SELECT id, metadata
            FROM documents
            WHERE text = ?
        """
        result = self.connection.execute(query, [text]).fetchall()

        for row in result:
            item_id = row[0]
            existing_metadata = json.loads(row[1])

            # Check if metadata is compatible
            compatible = True
            for key in existing_metadata:
                if key in metadata and existing_metadata[key] != metadata[key]:
                    compatible = False
                    break

            if compatible:
                return item_id, 1.0

        # Fallback to similarity search
        text_with_metadata = self._join_text_and_metadata(text, metadata)
        similar_items = self.query(text_with_metadata, top_k=1, threshold=0.95)

        if not similar_items:
            return -1, 0.0

        # Check if metadata is compatible
        existing_item = similar_items[0]
        existing_metadata = existing_item["metadata"]
        similarity = existing_item["similarity"]

        if similarity < 0.98:
            return -1, 0.0

        # Check if existing metadata is compatible with new metadata
        for key in existing_metadata:
            if key in metadata and existing_metadata[key] != metadata[key]:
                return -1, 0.0

        return existing_item["id"], similarity

    def upsert(self, items: list[dict]) -> list[int]:
        """
        Add items to the vector store if similar items don't exist, update them if they do.
        Optimized for DuckDB with early exits and reduced query count.

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.

        Returns
        -------
        List of assigned IDs for the added or updated items.
        """
        if not items:
            return []

        if not self._initialized:
            # Early exit - just add everything as new if DB not initialized
            return self.add(items)

        # Prepare data for batch processing
        texts = []
        metadata_list = []
        text_and_metadata_list = []

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}
            texts.append(text)
            metadata_list.append(metadata)
            text_and_metadata_list.append(self._join_text_and_metadata(text, metadata))

        # Pre-compute embeddings once for all items
        item_embeddings = self.embeddings.embed(text_and_metadata_list)

        assigned_ids = []
        new_items = []
        new_item_indices = []

        # Build a single query to get exact text matches (reduces round trips)
        if texts:
            placeholders = ', '.join(['?'] * len(texts))
            query = f"""
                SELECT id, text, metadata
                FROM documents
                WHERE text IN ({placeholders})
            """

            result = self.connection.execute(query, texts).fetchall()

            # Create efficient lookup maps
            text_matches = {}
            for row in result:
                item_id = row[0]
                item_text = row[1]
                item_metadata = json.loads(row[2])

                if item_text in text_matches:
                    text_matches[item_text].append((item_id, item_metadata))
                else:
                    text_matches[item_text] = [(item_id, item_metadata)]
        else:
            text_matches = {}

        # Process each item
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadata_list, item_embeddings)):
            # Fast path: check if we have exact text matches
            if text not in text_matches:
                new_items.append({"text": text, "metadata": metadata})
                new_item_indices.append(i)
                continue

            # Check metadata compatibility with all matching texts
            found_compatible = False
            for item_id, existing_metadata in text_matches[text]:
                # Check if metadata keys are different
                if set(metadata.keys()) != set(existing_metadata.keys()):
                    # Need to update metadata
                    self.delete([item_id])
                    new_ids = self.add([{"text": text, "metadata": metadata}], force_ids=[item_id])
                    assigned_ids.extend(new_ids)
                    found_compatible = True
                    break

                # Check if metadata values are compatible
                if any(key in existing_metadata and existing_metadata[key] != value
                       for key, value in metadata.items()):
                    continue

                # Found compatible match
                assigned_ids.append(item_id)
                found_compatible = True
                break

            if not found_compatible:
                new_items.append({"text": text, "metadata": metadata})
                new_item_indices.append(i)

        # For remaining items, use similarity search
        for i, item in zip(new_item_indices, new_items):
            vector = np.array(item_embeddings[i], dtype=np.float32).tolist()
            vector_dim = len(vector)

            # Single query for similarity search
            query = f"""
                SELECT id, text, metadata,
                    array_cosine_similarity(embedding, ?::REAL[{vector_dim}]) AS similarity
                FROM documents
                WHERE array_cosine_similarity(embedding, ?::REAL[{vector_dim}]) >= 0.95
                ORDER BY similarity DESC
                LIMIT 1;
            """

            result = self.connection.execute(query, [vector, vector]).fetchall()

            # No similar items found - add as new
            if not result or result[0][3] < 0.98:
                new_ids = self.add([item])
                assigned_ids.extend(new_ids)
                continue

            # Check metadata compatibility with similar item
            row = result[0]
            item_id = row[0]
            existing_metadata = json.loads(row[2])

            # Check if metadata keys are different
            if set(item["metadata"].keys()) != set(existing_metadata.keys()):
                # Delete existing and add with new metadata
                self.delete([item_id])
                new_ids = self.add([item], force_ids=[item_id])
                assigned_ids.extend(new_ids)
                continue

            # If any metadata values conflict, add as new
            if any(key in existing_metadata and existing_metadata[key] != value
                   for key, value in item["metadata"].items()):
                new_ids = self.add([item])
                assigned_ids.extend(new_ids)
                continue

            # Use existing item
            assigned_ids.append(item_id)

        return assigned_ids

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
