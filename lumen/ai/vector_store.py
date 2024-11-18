import json

from abc import ABC, abstractmethod

import duckdb
import numpy as np

from .embeddings import Embeddings


class VectorStore(ABC):
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    @abstractmethod
    def add(self, items: list[dict]) -> list[int]:
        pass

    @abstractmethod
    def query(
        self, text: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        pass


class NumpyVectorStore(VectorStore):
    def __init__(self, embeddings: Embeddings):
        super().__init__(embeddings)
        # Initialize empty arrays and lists for storing data
        self.vectors = np.empty(
            (0, 1536), dtype=np.float32
        )  # Embedding dimension is 1536
        self.texts: list[str] = []
        self.metadata: list[dict] = []
        self.ids: list[int] = []
        self._current_id: int = 0

    def _get_next_id(self) -> int:
        """Generate the next available ID."""
        self._current_id += 1
        return self._current_id

    def _cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query vector and all stored vectors.

        Args:
            query_vector: Query embedding of shape (1536,)

        Returns:
            np.ndarray: Array of similarity scores
        """
        # Normalize vectors
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(self.vectors))

        query_normalized = query_vector / query_norm

        # Normalize stored vectors (only if we have vectors)
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
            items: List of dictionaries containing 'text' and optional 'metadata'

        Returns:
            list[int]: List of assigned IDs
        """
        texts = [item["text"] for item in items]
        embeddings = self.embeddings.embed(texts)

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Generate IDs for new items
        new_ids = [self._get_next_id() for _ in items]

        # Append to storage
        self.vectors = (
            np.vstack([self.vectors, embeddings_array])
            if len(self.vectors) > 0
            else embeddings_array
        )
        self.texts.extend(texts)
        self.metadata.extend([item.get("metadata", {}) or {} for item in items])
        self.ids.extend(new_ids)

        return new_ids

    def query(
        self, text: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        """
        Query the vector store for similar items.

        Args:
            text: Query text
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            list[Result]: List of results with id, text, metadata, and similarity score
        """
        query_embedding = np.array(self.embeddings.embed([text])[0], dtype=np.float32)

        similarities = self._cosine_similarity(query_embedding)

        if filters and len(self.vectors) > 0:
            mask = np.ones(len(self.vectors), dtype=bool)
            for key, value in filters.items():
                mask &= np.array([item.get(key) == value for item in self.metadata])
            similarities = similarities * mask

        if len(similarities) > 0:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            top_indices = []

        results = []
        for idx in top_indices:
            results.append(
                {
                    "id": self.ids[idx],
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "similarity": float(similarities[idx]),
                }
            )

        return results

    def delete(self, ids: list[int]) -> None:
        """
        Delete items from the vector store by their IDs.

        Args:
            ids: List of IDs to delete
        """
        if not ids:
            return

        keep_mask = np.ones(len(self.vectors), dtype=bool)
        for idx, item_id in enumerate(self.ids):
            if item_id in ids:
                keep_mask[idx] = False

        self.vectors = self.vectors[keep_mask]
        self.texts = [text for i, text in enumerate(self.texts) if keep_mask[i]]
        self.metadata = [meta for i, meta in enumerate(self.metadata) if keep_mask[i]]
        self.ids = [id_ for i, id_ in enumerate(self.ids) if keep_mask[i]]

    def clear(self) -> None:
        """Clear all items from the vector store."""
        self.vectors = np.empty((0, 1536), dtype=np.float32)
        self.texts = []
        self.metadata = []
        self.ids = []
        self._current_id = 0


class DuckDBVectorStore(VectorStore):
    def __init__(self, embeddings: Embeddings, db_path: str = ":memory:"):
        super().__init__(embeddings)
        self.connection = duckdb.connect(database=db_path)
        self._setup_database()

    def _setup_database(self) -> None:
        self.connection.execute("INSTALL 'vss';")
        self.connection.execute("LOAD 'vss';")

        self.connection.execute("CREATE SEQUENCE IF NOT EXISTS documents_id_seq;")

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id BIGINT DEFAULT NEXTVAL('documents_id_seq') PRIMARY KEY,
                text VARCHAR,
                embedding FLOAT[1536],
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
        texts = [item["text"] for item in items]
        embeddings = self.embeddings.embed(texts)

        text_ids = []

        for i, item in enumerate(items):
            text = item["text"]
            metadata = item.get("metadata", {})
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
        self, text: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        query_embedding = self.embeddings.embed([text])[0]
        query_embedding = np.array(query_embedding, dtype=np.float32).tolist()

        base_query = """
            SELECT id, text, metadata,
                   array_distance(embedding, ?::FLOAT[1536]) AS similarity
            FROM documents
            WHERE 1=1
        """
        params = [query_embedding]

        if filters:
            for key, value in filters.items():
                # Use json_extract_string for string comparison
                base_query += f" AND json_extract_string(metadata, '$.{key}') = ?"
                params.append(str(value))

        base_query += """
            ORDER BY similarity
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
