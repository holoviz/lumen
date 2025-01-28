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
        default=1024, doc="Maximum size of text chunks to split documents into."
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
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.vectors = np.empty((0, self.vocab_size), dtype=np.float32)
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
        new_ids = [self._get_next_id() for _ in all_texts]

        self.vectors = (
            np.vstack([self.vectors, embeddings])
            if len(self.vectors) > 0
            else embeddings
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
        """
        Clear all items from the vector store.

        Resets the vectors, texts, metadata, and IDs to their initial empty states.
        """
        self.vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.texts = []
        self.metadata = []
        self.ids = []
        self._current_id = 0

    def __len__(self) -> int:
        return len(self.vectors)


class DuckDBVectorStore(VectorStore):
    """
    Vector store implementation using DuckDB for persistent storage.

    :Example:

    .. code-block:: python

        from lumen.ai.vector_store import DuckDBStore

        vector_store = DuckDBStore(uri=':memory:)
        vector_store.add_file('https://lumen.holoviz.org')
        vector_store.query('LLM', threshold=0.1)
    """

    uri = param.String(doc="The URI of the DuckDB database")

    def __init__(self, **params):
        super().__init__(**params)
        self.connection = duckdb.connect(database=self.uri)
        self._setup_database()

    def _setup_database(self) -> None:
        """Set up the DuckDB database with necessary tables and indexes."""
        self.connection.execute("INSTALL 'vss';")
        self.connection.execute("LOAD 'vss';")
        self.connection.execute("SET hnsw_enable_experimental_persistence = true;")
        self.connection.execute("CREATE SEQUENCE IF NOT EXISTS documents_id_seq;")

        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGINT DEFAULT NEXTVAL('documents_id_seq') PRIMARY KEY,
                text VARCHAR,
                metadata JSON,
                embedding FLOAT[{self.vocab_size}]
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
        Add items to the vector store.

        Parameters
        ----------
        items: list[dict]
            List of dictionaries containing 'text' and optional 'metadata'.

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

        embeddings = self.embeddings.embed(text_and_metadata_list)

        text_ids = []
        for i in range(len(all_texts)):
            result = self.connection.execute(
                """
                INSERT INTO documents (text, metadata, embedding)
                VALUES (?, ?::JSON, ?) RETURNING id;
                """,
                [
                    all_texts[i],
                    json.dumps(all_metadata[i]),
                    np.array(embeddings[i], dtype=np.float32).tolist(),
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
        query_embedding = np.array(
            self.embeddings.embed([text])[0], dtype=np.float32
        ).tolist()

        base_query = f"""
            SELECT id, text, metadata,
                array_cosine_similarity(embedding, ?::REAL[{self.vocab_size}]) AS similarity
            FROM documents
            WHERE array_cosine_similarity(embedding, ?::REAL[{self.vocab_size}]) >= ?
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
        if not ids:
            return

        placeholders = ", ".join(["?"] * len(ids))
        query = f"DELETE FROM documents WHERE id IN ({placeholders});"
        self.connection.execute(query, ids)

    def clear(self) -> None:
        """
        Clear all entries and reset sequence.

        Drops the documents table and sequence, then sets up the database again.
        """
        self.connection.execute("DROP TABLE IF EXISTS documents;")
        self.connection.execute("DROP SEQUENCE IF EXISTS documents_id_seq;")
        self._setup_database()

    def __len__(self) -> int:
        result = self.connection.execute("SELECT COUNT(*) FROM documents;").fetchone()
        return result[0]
