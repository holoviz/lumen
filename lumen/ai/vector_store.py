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

    def _format_metadata_value(self, value) -> str:
        """Format a metadata value appropriately based on its type.

        Args:
            value: The metadata value to format.

        Returns:
            A string representation of the metadata value.
        """
        if isinstance(value, (list, tuple)):
            return f"[{', '.join(str(v) for v in value)}]"
        return str(value)

    def _get_content_and_text_and_metadata(
        self, text: str, metadata: dict
    ) -> tuple[str, str]:
        """Get separate text strings for content and metadata.

        Args:
            text: The main content text.
            metadata: A dictionary of metadata associated with the text.

        Returns:
            A tuple containing the content text and the formatted metadata text.
        """
        metadata_items = [
            f"({key}: {self._format_metadata_value(value)})"
            for key, value in metadata.items()
        ]
        text_and_metadata = " ".join(metadata_items)
        return text, text_and_metadata

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks of size up to self.chunk_size.

        Args:
            text: The text to split.

        Returns:
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
        query_with_metadata: bool = True,
    ) -> list[dict]:
        """
        Query the vector store for similar items.

        Args:
            text: The query text.
            top_k: Number of top results to return.
            filters: Optional metadata filters.
            threshold: Minimum similarity score required for a result to be included.
            query_with_metadata: Whether to additionally query by metadata, i.e. include metadata in the text for querying.

        Returns:
            List of results with 'id', 'text', 'metadata', 'text_and_metadata', and 'similarity' score.
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
            List of results with 'id', 'text', 'metadata', and 'text_and_metadata'.
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


class NumpyVectorStore(VectorStore):
    """Vector store implementation using NumPy for in-memory storage."""

    def __init__(self, **params):
        super().__init__(**params)
        self.content_vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.text_and_metadata_vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.texts: list[str] = []
        self.text_and_metadatas: list[str] = []
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

    def _cosine_similarity(
        self, query_vector: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity between query vector and stored vectors.

        Args:
            query_vector: Query embedding of shape (vocab_size,).
            vectors: Stored embeddings to compare against.

        Returns:
            Array of similarity scores.
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

    def add(self, items: list[dict]) -> list[int]:
        """
        Add items to the vector store.

        Args:
            items: List of dictionaries containing 'text' and optional 'metadata'.

        Returns:
            List of assigned IDs for the added items.
        """
        all_texts = []
        all_text_and_metadatas = []
        all_metadata = []

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}
            content_text, text_and_metadata = self._get_content_and_text_and_metadata(
                text, metadata
            )

            content_chunks = self._chunk_text(content_text)
            for chunk in content_chunks:
                all_texts.append(chunk)
                all_text_and_metadatas.append(text_and_metadata)
                all_metadata.append(metadata)

        content_embeddings = np.array(
            self.embeddings.embed(all_texts), dtype=np.float32
        )
        text_and_metadata_embeddings = np.array(
            self.embeddings.embed(all_text_and_metadatas), dtype=np.float32
        )

        new_ids = [self._get_next_id() for _ in all_texts]

        self.content_vectors = (
            np.vstack([self.content_vectors, content_embeddings])
            if len(self.content_vectors) > 0
            else content_embeddings
        )
        self.text_and_metadata_vectors = (
            np.vstack([self.text_and_metadata_vectors, text_and_metadata_embeddings])
            if len(self.text_and_metadata_vectors) > 0
            else text_and_metadata_embeddings
        )
        self.texts.extend(all_texts)
        self.text_and_metadatas.extend(all_text_and_metadatas)
        self.metadata.extend(all_metadata)
        self.ids.extend(new_ids)

        return new_ids

    def query(
        self,
        text: str,
        top_k: int = 5,
        filters: dict | None = None,
        threshold: float = 0.0,
        query_with_metadata: bool = True,
    ) -> list[dict]:
        """
        Query the vector store for similar items.

        Args:
            text: The query text.
            top_k: Number of top results to return.
            filters: Optional metadata filters.
            threshold: Minimum similarity score required for a result to be included.
            query_with_metadata: Whether to additionally query by metadata, i.e. include metadata in the text for querying.

        Returns:
            List of results with 'id', 'text', 'metadata', 'text_and_metadata', and 'similarity' score.
        """
        query_embedding = np.array(self.embeddings.embed([text])[0], dtype=np.float32)
        vectors = self.text_and_metadata_vectors if query_with_metadata else self.content_vectors
        similarities = self._cosine_similarity(query_embedding, vectors)

        if filters and len(vectors) > 0:
            mask = np.ones(len(vectors), dtype=bool)
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
                        "text_and_metadata": self.text_and_metadatas[idx],
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
            List of results with 'id', 'text', 'metadata', and 'text_and_metadata'.
        """
        if not self.metadata:
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
                "text_and_metadata": self.text_and_metadatas[idx],
            }
            for idx in matching_indices
        ]

    def delete(self, ids: list[int]) -> None:
        """
        Delete items from the vector store by their IDs.

        Args:
            ids: List of IDs to delete.
        """
        if not ids:
            return

        keep_mask = np.ones(len(self.content_vectors), dtype=bool)
        id_set = set(ids)
        for idx, item_id in enumerate(self.ids):
            if item_id in id_set:
                keep_mask[idx] = False

        self.content_vectors = self.content_vectors[keep_mask]
        self.text_and_metadata_vectors = self.text_and_metadata_vectors[keep_mask]
        self.texts = [text for i, text in enumerate(self.texts) if keep_mask[i]]
        self.text_and_metadatas = [
            text for i, text in enumerate(self.text_and_metadatas) if keep_mask[i]
        ]
        self.metadata = [meta for i, meta in enumerate(self.metadata) if keep_mask[i]]
        self.ids = [id_ for i, id_ in enumerate(self.ids) if keep_mask[i]]

    def clear(self) -> None:
        """Clear all items from the vector store."""
        self.content_vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.text_and_metadata_vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.texts = []
        self.text_and_metadatas = []
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
                text_and_metadata VARCHAR,
                content_embedding FLOAT[{self.vocab_size}],
                metadata_embedding FLOAT[{self.vocab_size}],
                metadata JSON
            );
        """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS content_embedding_index
            ON documents USING HNSW (content_embedding) WITH (metric = 'cosine');
        """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS metadata_embedding_index
            ON documents USING HNSW (metadata_embedding) WITH (metric = 'cosine');
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
        all_text_and_metadatas = []
        all_metadata = []

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}
            content_text, text_and_metadata = self._get_content_and_text_and_metadata(
                text, metadata
            )

            content_chunks = self._chunk_text(content_text)
            for chunk in content_chunks:
                all_texts.append(chunk)
                all_text_and_metadatas.append(text_and_metadata)
                all_metadata.append(metadata)

        content_embeddings = self.embeddings.embed(all_texts)
        text_and_metadata_embeddings = self.embeddings.embed(all_text_and_metadatas)

        text_ids = []
        for i in range(len(all_texts)):
            result = self.connection.execute(
                """
                INSERT INTO documents (text, text_and_metadata, content_embedding, metadata_embedding, metadata)
                VALUES (?, ?, ?, ?, ?::JSON) RETURNING id;
                """,
                [
                    all_texts[i],
                    all_text_and_metadatas[i],
                    np.array(content_embeddings[i], dtype=np.float32).tolist(),
                    np.array(text_and_metadata_embeddings[i], dtype=np.float32).tolist(),
                    json.dumps(all_metadata[i]),
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
        query_with_metadata: bool = True,
    ) -> list[dict]:
        """
        Query the DuckDB vector store for similar items.

        Args:
            text: The query text.
            top_k: Number of top results to return.
            filters: Optional metadata filters.
            threshold: Minimum similarity score required for a result to be included.
            query_with_metadata: Whether to additionally query by metadata, i.e. include metadata in the text for querying.

        Returns:
            List of results with 'id', 'text', 'text_and_metadata', 'metadata', and 'similarity' score.
        """
        query_embedding = np.array(
            self.embeddings.embed([text])[0], dtype=np.float32
        ).tolist()
        embedding_column = (
            "metadata_embedding" if query_with_metadata else "content_embedding"
        )

        base_query = f"""
            SELECT id, text, text_and_metadata, metadata,
                   1 - array_distance({embedding_column}, ?::FLOAT[{self.vocab_size}], 'cosine') AS similarity
            FROM documents
            WHERE 1=1
        """
        params = [query_embedding]

        if filters:
            for key, value in filters.items():
                base_query += f" AND json_extract_string(metadata, '$.{key}') = ?"
                params.append(str(value))

        base_query += f" AND 1 - array_distance({embedding_column}, ?::FLOAT[{self.vocab_size}], 'cosine') >= ?"
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
                "text_and_metadata": row[2],
                "metadata": json.loads(row[3]),
                "similarity": row[4],
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
            List of results with 'id', 'text', 'text_and_metadata', and 'metadata'.
        """
        base_query = """
            SELECT id, text, text_and_metadata, metadata
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
                "text_and_metadata": row[2],
                "metadata": json.loads(row[3]),
            }
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
