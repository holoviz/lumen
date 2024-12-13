import json

from abc import abstractmethod

import duckdb
import numpy as np
import param

from .embeddings import Embeddings, NumpyEmbeddings


class VectorStore(param.Parameterized):
    """Abstract base class for a vector store."""

    embeddings = param.ClassSelector(
        class_=Embeddings,
        default=NumpyEmbeddings(),
        doc="Embeddings object for text processing.",
    )

    vocab_size = param.Integer(
        default=1536,
        doc="The size of the embeddings vector. Must match the embeddings model.",
    )

    chunk_size = param.Integer(
        default=512, doc="Maximum size of text chunks to split documents into."
    )

    @abstractmethod
    def add(self, items: list[dict]) -> list[int]:
        """
        Add items to the vector store.

        Args:
            items: List of dictionaries containing 'text' and optional 'metadata'.

        Returns:
            List of assigned IDs for the added items.
        """

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

        Args:
            text: The query text.
            top_k: Number of top results to return.
            filters: Optional metadata filters.
            threshold: Minimum similarity score required for a result to be included.

        Returns:
            List of results with 'id', 'text', 'metadata', and 'similarity' score.
        """

    @abstractmethod
    def filter_by(
        self, filters: dict, limit: int | None = None, offset: int = 0
    ) -> list[dict]:
        """
        Filter items by metadata without using embeddings similarity.

        Args:
            filters: Dictionary of metadata key-value pairs to filter by.
            limit: Maximum number of results to return. If None, returns all matches.
            offset: Number of results to skip (for pagination).

        Returns:
            List of results with 'id', 'text', and 'metadata'.
        """

    @abstractmethod
    def delete(self, ids: list[int]) -> None:
        """
        Delete items from the vector store by their IDs.

        Args:
            ids: List of IDs to delete.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all items from the vector store."""

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks of size up to self.chunk_size.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        if self.chunk_size is None or len(text) <= self.chunk_size:
            return [text]
        else:
            # Split text into chunks without breaking words
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


class NumpyVectorStore(VectorStore):
    """Vector store implementation using NumPy for in-memory storage."""

    def __init__(self, **params):
        super().__init__(**params)
        self.vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.texts: list[str] = []
        self.metadata: list[dict] = []
        self.ids: list[int] = []
        self._current_id: int = 0

    def _get_next_id(self) -> int:
        """Generate the next available ID.

        Returns:
            The next unique ID.
        """
        self._current_id += 1
        return self._current_id

    def _cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between the query vector and all stored vectors.

        Args:
            query_vector: Query embedding of shape (vocab_size,).

        Returns:
            Array of similarity scores.
        """
        # Normalize the query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(self.vectors))

        query_normalized = query_vector / query_norm

        # Normalize stored vectors if any
        if len(self.vectors) > 0:
            vectors_norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            vectors_norm[vectors_norm == 0] = 1  # Avoid division by zero
            vectors_normalized = self.vectors / vectors_norm

            # Calculate cosine similarity
            similarities = np.dot(vectors_normalized, query_normalized)
        else:
            similarities = np.array([])

        return similarities

    def add(self, items: list[dict]) -> list[int]:
        """
        Add items to the vector store.

        Args:
            items: List of dictionaries containing 'text' and optional 'metadata'.

        Returns:
            List of assigned IDs for the added items.
        """
        all_texts = []
        all_metadata = []

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}
            chunks = self._chunk_text(text)

            for chunk in chunks:
                all_texts.append(chunk)
                all_metadata.append(metadata)

        embeddings = self.embeddings.embed(all_texts)

        # Convert embeddings to NumPy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Generate IDs for new items
        new_ids = [self._get_next_id() for _ in all_texts]

        # Append to storage
        self.vectors = (
            np.vstack([self.vectors, embeddings_array])
            if len(self.vectors) > 0
            else embeddings_array
        )
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

        Args:
            text: The query text.
            top_k: Number of top results to return.
            filters: Optional metadata filters.
            threshold: Minimum similarity score required for a result to be included.

        Returns:
            List of results with 'id', 'text', 'metadata', and 'similarity' score.
        """
        query_embedding = np.array(self.embeddings.embed([text])[0], dtype=np.float32)

        similarities = self._cosine_similarity(query_embedding)

        if filters and len(self.vectors) > 0:
            mask = np.ones(len(self.vectors), dtype=bool)
            for key, value in filters.items():
                mask &= np.array([item.get(key) == value for item in self.metadata])
            similarities = similarities * mask

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

        Args:
            filters: Dictionary of metadata key-value pairs to filter by.
            limit: Maximum number of results to return. If None, returns all matches.
            offset: Number of results to skip (for pagination).

        Returns:
            List of results with 'id', 'text', and 'metadata'.
        """
        if not self.metadata:
            return []

        # Create mask for matching items
        mask = np.ones(len(self.metadata), dtype=bool)
        for key, value in filters.items():
            mask &= np.array([item.get(key) == value for item in self.metadata])

        # Get matching indices
        matching_indices = np.where(mask)[0]

        # Apply offset and limit
        if offset:
            matching_indices = matching_indices[offset:]
        if limit is not None:
            matching_indices = matching_indices[:limit]

        # Build results
        results = []
        for idx in matching_indices:
            results.append(
                {
                    "id": self.ids[idx],
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                }
            )

        return results

    def delete(self, ids: list[int]) -> None:
        """
        Delete items from the vector store by their IDs.

        Args:
            ids: List of IDs to delete.
        """
        if not ids:
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

    def clear(self) -> None:
        """Clear all items from the vector store."""
        self.vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.texts = []
        self.metadata = []
        self.ids = []
        self._current_id = 0


class DuckDBVectorStore(VectorStore):
    """Vector store implementation using DuckDB for persistent storage."""

    uri = param.String(doc="The URI of the DuckDB database")

    def __init__(self, **params):
        super().__init__(**params)
        self.connection = duckdb.connect(database=self.uri)
        self._setup_database()

    def _setup_database(self) -> None:
        """Set up the DuckDB database with necessary tables and indexes."""
        self.connection.execute("INSTALL 'vss';")
        self.connection.execute("LOAD 'vss';")

        self.connection.execute("CREATE SEQUENCE IF NOT EXISTS documents_id_seq;")

        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGINT DEFAULT NEXTVAL('documents_id_seq') PRIMARY KEY,
                text VARCHAR,
                embedding FLOAT[{self.vocab_size}],
                metadata JSON
            );
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS embedding_index
            ON documents USING HNSW (embedding) WITH (metric = 'cosine');
            """
        )

    def add(self, items: list[dict]) -> list[int]:
        """
        Add items to the DuckDB vector store.

        Args:
            items: List of dictionaries containing 'text' and optional 'metadata'.

        Returns:
            List of assigned IDs for the added items.
        """
        all_texts = []
        all_metadata = []

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}
            chunks = self._chunk_text(text)

            for chunk in chunks:
                all_texts.append(chunk)
                all_metadata.append(metadata)

        embeddings = self.embeddings.embed(all_texts)

        text_ids = []

        for i in range(len(all_texts)):
            text = all_texts[i]
            metadata = all_metadata[i]
            embedding = np.array(embeddings[i], dtype=np.float32).tolist()

            result = self.connection.execute(
                """
                INSERT INTO documents (text, embedding, metadata)
                VALUES (?, ?, ?::JSON) RETURNING id;
                """,
                [text, embedding, json.dumps(metadata)],
            )
            text_ids.append(result.fetchone()[0])

        return text_ids

    def query(
        self,
        text: str,
        top_k: int = 5,
        filters: dict | None = None,
        threshold: float = 0.0,
    ) -> list[dict]:
        """
        Query the DuckDB vector store for similar items.

        Args:
            text: The query text.
            top_k: Number of top results to return.
            filters: Optional metadata filters.
            threshold: Minimum similarity score required for a result to be included.

        Returns:
            List of results with 'id', 'text', 'metadata', and 'similarity' score.
        """
        query_embedding = self.embeddings.embed([text])[0]
        query_embedding = np.array(query_embedding, dtype=np.float32).tolist()

        base_query = f"""
            SELECT id, text, metadata,
                   1 - array_distance(embedding, ?::FLOAT[{self.vocab_size}], 'cosine') AS similarity
            FROM documents
            WHERE 1=1
        """
        params = [query_embedding]

        if filters:
            for key, value in filters.items():
                # Use json_extract_string for string comparison
                base_query += f" AND json_extract_string(metadata, '$.{key}') = ?"
                params.append(str(value))

        base_query += f" AND 1 - array_distance(embedding, ?::FLOAT[{self.vocab_size}], 'cosine') >= ?"
        params.extend([query_embedding, threshold])

        base_query += """
            ORDER BY similarity DESC
            LIMIT ?;
        """
        params.append(top_k)

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

    def filter_by(
        self, filters: dict, limit: int | None = None, offset: int = 0
    ) -> list[dict]:
        """
        Filter items by metadata without using embeddings similarity.

        Args:
            filters: Dictionary of metadata key-value pairs to filter by.
            limit: Maximum number of results to return. If None, returns all matches.
            offset: Number of results to skip (for pagination).

        Returns:
            List of results with 'id', 'text', and 'metadata'.
        """
        base_query = """
            SELECT id, text, metadata
            FROM documents
            WHERE 1=1
        """
        params = []

        # Add filters to query
        for key, value in filters.items():
            base_query += f" AND json_extract_string(metadata, '$.{key}') = ?"
            params.append(str(value))

        # Add offset and limit
        if offset:
            base_query += " OFFSET ?"
            params.append(offset)

        if limit is not None:
            base_query += " LIMIT ?"
            params.append(limit)

        base_query += ";"

        result = self.connection.execute(base_query, params).fetchall()

        return [
            {"id": row[0], "text": row[1], "metadata": json.loads(row[2])}
            for row in result
        ]

    def delete(self, ids: list[int]) -> None:
        """
        Delete items from the DuckDB vector store by their IDs.

        Args:
            ids: List of IDs to delete.
        """
        if not ids:
            return

        placeholders = ", ".join(["?"] * len(ids))
        query = f"DELETE FROM documents WHERE id IN ({placeholders});"
        self.connection.execute(query, ids)

    def clear(self) -> None:
        """Clear all items from the DuckDB vector store."""
        self.connection.execute("DELETE FROM documents;")
        self.connection.execute("ALTER SEQUENCE documents_id_seq RESTART WITH 1;")
