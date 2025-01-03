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
        doc="Size of the embeddings vector. Must match the embeddings model."
    )

    chunk_size = param.Integer(
        default=512,
        doc="Maximum size of text chunks to split documents into."
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
        doc_threshold: float = 0.0,
        meta_threshold: float = 0.0,
        doc_weight: float = 0.5,
        meta_weight: float = 0.5,
        field_weights: dict | None = None,
    ) -> list[dict]:
        """
        Query the vector store for similar items, optionally considering both
        document-level and metadata-level embeddings. Allows different thresholds
        and weighting schemes.

        Args:
            text: The query text.
            top_k: Number of top results to return.
            filters: Optional metadata filters (key/value pairs to match exactly).
            doc_threshold: Minimum cosine similarity required at the document level.
            meta_threshold: Minimum cosine similarity required at the metadata level.
            doc_weight: Weight factor for document-level similarity.
            meta_weight: Weight factor for metadata-level similarity.
            field_weights: Optional per-field weighting for metadata fields.

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


class NumpyVectorStore(VectorStore):
    """
    Vector store implementation using NumPy for in-memory storage,
    extended to handle a two-table approach:
      - doc_ arrays for document chunks
      - meta_ arrays for metadata field embeddings
    """

    def __init__(self, **params):
        super().__init__(**params)
        # Document-level
        self.doc_vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.doc_texts: list[str] = []
        self.doc_metadata: list[dict] = []
        self.doc_ids: list[int] = []

        # Metadata-level
        self.meta_vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.meta_doc_ids: list[int] = []
        self.meta_field_names: list[str] = []
        self.meta_field_texts: list[str] = []

        self._current_id: int = 0

    def _get_next_id(self) -> int:
        self._current_id += 1
        return self._current_id

    def _cosine_similarity(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        if len(vectors) == 0:
            return np.array([])

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(vectors))

        query_normalized = query_vector / query_norm

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors_normalized = vectors / norms

        return np.dot(vectors_normalized, query_normalized)

    def _normalize_similarity(self, sim: float) -> float:
        """
        Example normalization that transforms cosine in [-1, 1] to [0, 1].
        """
        return (sim + 1.0) / 2.0

    def _calculate_combined_score(
        self,
        doc_sim: float,
        meta_sim: float,
        doc_weight: float = 0.5,
        meta_weight: float = 0.5
    ) -> float:
        """
        More flexible scoring that:
          - Normalizes each similarity
          - Uses weighting for doc vs. metadata
        """
        doc_norm = self._normalize_similarity(doc_sim)
        meta_norm = self._normalize_similarity(meta_sim)
        total_weight = doc_weight + meta_weight
        return (doc_weight * doc_norm + meta_weight * meta_norm) / total_weight

    def add(self, items: list[dict]) -> list[int]:
        new_ids = []

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}
            chunks = self._chunk_text(text)

            # Embed document chunks
            doc_embeddings = self.embeddings.embed(chunks)
            for chunk_text, emb in zip(chunks, doc_embeddings):
                emb_np = np.array(emb, dtype=np.float32)
                doc_id = self._get_next_id()

                # Store doc-level chunk
                if len(self.doc_vectors) == 0:
                    self.doc_vectors = emb_np[np.newaxis, :]
                else:
                    self.doc_vectors = np.vstack([self.doc_vectors, emb_np[np.newaxis, :]])

                self.doc_texts.append(chunk_text)
                self.doc_metadata.append(metadata)
                self.doc_ids.append(doc_id)
                new_ids.append(doc_id)

                # Optional: embed certain metadata fields
                for key, val in metadata.items():
                    # Example logic: only embed if field is a non-empty string
                    if isinstance(val, str) and val.strip():
                        meta_emb = self.embeddings.embed([val])[0]
                        meta_emb_np = np.array(meta_emb, dtype=np.float32)

                        if len(self.meta_vectors) == 0:
                            self.meta_vectors = meta_emb_np[np.newaxis, :]
                        else:
                            self.meta_vectors = np.vstack(
                                [self.meta_vectors, meta_emb_np[np.newaxis, :]]
                            )

                        self.meta_doc_ids.append(doc_id)
                        self.meta_field_names.append(key)
                        self.meta_field_texts.append(val)

        return new_ids

    def query(
        self,
        text: str,
        top_k: int = 5,
        filters: dict | None = None,
        doc_threshold: float = 0.0,
        meta_threshold: float = 0.0,
        doc_weight: float = 0.5,
        meta_weight: float = 0.5,
        field_weights: dict | None = None,
    ) -> list[dict]:
        if len(self.doc_vectors) == 0 and len(self.meta_vectors) == 0:
            return []

        # 1) Embed the query
        query_emb = self.embeddings.embed([text])[0]
        query_emb_np = np.array(query_emb, dtype=np.float32)

        # 2) Document-level similarity
        doc_sims = self._cosine_similarity(query_emb_np, self.doc_vectors)

        # If filters exist, exclude docs that don't match
        if filters:
            doc_mask = np.ones(len(doc_sims), dtype=bool)
            for k, v in filters.items():
                doc_mask &= np.array([m.get(k) == v for m in self.doc_metadata], dtype=bool)
            # Set non-matching docs to -inf so they won't appear after sorting
            doc_sims[~doc_mask] = float('-inf')

        doc_sorted_idx = np.argsort(doc_sims)[::-1]

        # Gather candidate indices (stop if similarity < doc_threshold)
        doc_candidates = []
        for idx in doc_sorted_idx:
            sim = doc_sims[idx]
            if sim < doc_threshold:
                break
            doc_candidates.append(idx)
            if len(doc_candidates) >= top_k * 5:
                break

        # Build doc_info_map => {doc_id: best doc_sim, text, metadata}
        doc_info_map = {}
        for idx in doc_candidates:
            sim = doc_sims[idx]
            doc_id = self.doc_ids[idx]
            if doc_id not in doc_info_map or sim > doc_info_map[doc_id]["doc_similarity"]:
                doc_info_map[doc_id] = {
                    "doc_similarity": sim,
                    "text": self.doc_texts[idx],
                    "metadata": self.doc_metadata[idx],
                }

        # 3) Metadata-level similarity
        meta_sims = self._cosine_similarity(query_emb_np, self.meta_vectors)
        if filters:
            meta_mask = np.ones(len(meta_sims), dtype=bool)
            for i in range(len(meta_sims)):
                parent_doc_id = self.meta_doc_ids[i]
                try:
                    doc_idx = self.doc_ids.index(parent_doc_id)
                except ValueError:
                    meta_mask[i] = False
                    continue

                for k, v in filters.items():
                    if self.doc_metadata[doc_idx].get(k) != v:
                        meta_mask[i] = False
                        break

            meta_sims[~meta_mask] = float('-inf')

        meta_sorted_idx = np.argsort(meta_sims)[::-1]
        meta_candidates = []
        for i in meta_sorted_idx:
            sim = meta_sims[i]
            if sim < meta_threshold:
                break
            meta_candidates.append(i)
            if len(meta_candidates) >= top_k * 5:
                break

        # Build meta_info_map => {doc_id: best meta_sim, (field_name, field_text)}
        meta_info_map = {}
        for i in meta_candidates:
            sim = meta_sims[i]
            doc_id = self.meta_doc_ids[i]
            field_name = self.meta_field_names[i]
            field_text = self.meta_field_texts[i]

            field_w = 1.0
            if field_weights and field_name in field_weights:
                field_w = field_weights[field_name]

            weighted_sim = sim * field_w

            if doc_id not in meta_info_map or weighted_sim > meta_info_map[doc_id]["meta_similarity"]:
                meta_info_map[doc_id] = {
                    "meta_similarity": weighted_sim,
                    "field_name": field_name,
                    "field_text": field_text,
                }

        # 4) Merge / re-rank doc-level & metadata-level similarities
        combined_results = {}
        for doc_id, doc_data in doc_info_map.items():
            doc_sim = doc_data["doc_similarity"]
            meta_sim = meta_info_map.get(doc_id, {}).get("meta_similarity", 0.0)
            score = self._calculate_combined_score(doc_sim, meta_sim, doc_weight, meta_weight)
            combined_results[doc_id] = {
                "doc_id": doc_id,
                "text": doc_data["text"],
                "metadata": doc_data["metadata"],
                "doc_similarity": doc_sim,
                "meta_similarity": meta_sim,
                "combined_score": score,
            }

        # Include docs that appear only in meta_info_map
        for doc_id, meta_data in meta_info_map.items():
            if doc_id not in combined_results:
                try:
                    doc_idx = self.doc_ids.index(doc_id)
                except ValueError:
                    continue
                doc_text = self.doc_texts[doc_idx]
                doc_meta = self.doc_metadata[doc_idx]
                doc_sim = 0.0
                meta_sim = meta_data["meta_similarity"]
                score = self._calculate_combined_score(doc_sim, meta_sim, doc_weight, meta_weight)
                combined_results[doc_id] = {
                    "doc_id": doc_id,
                    "text": doc_text,
                    "metadata": doc_meta,
                    "doc_similarity": doc_sim,
                    "meta_similarity": meta_sim,
                    "combined_score": score,
                }

        final_list = sorted(combined_results.values(), key=lambda x: x["combined_score"], reverse=True)
        final_list = final_list[:top_k]

        # 5) Build and return final results
        out = []
        for item in final_list:
            out.append({
                "id": item["doc_id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "similarity": float(item["combined_score"]),
            })
        return out

    def filter_by(self, filters: dict, limit: int | None = None, offset: int = 0) -> list[dict]:
        if not self.doc_metadata:
            return []

        mask = np.ones(len(self.doc_metadata), dtype=bool)
        for key, value in filters.items():
            mask &= np.array([m.get(key) == value for m in self.doc_metadata])

        matching_indices = np.where(mask)[0]

        if offset:
            matching_indices = matching_indices[offset:]
        if limit is not None:
            matching_indices = matching_indices[:limit]

        results = []
        for idx in matching_indices:
            results.append({
                "id": self.doc_ids[idx],
                "text": self.doc_texts[idx],
                "metadata": self.doc_metadata[idx],
            })
        return results

    def delete(self, ids: list[int]) -> None:
        if not ids or not self.doc_ids:
            return

        to_remove = set(ids)

        # Remove doc-level
        doc_keep_mask = np.array([doc_id not in to_remove for doc_id in self.doc_ids])
        self.doc_vectors = self.doc_vectors[doc_keep_mask]
        self.doc_texts = [t for t, keep in zip(self.doc_texts, doc_keep_mask) if keep]
        self.doc_metadata = [m for m, keep in zip(self.doc_metadata, doc_keep_mask) if keep]
        self.doc_ids = [i for i, keep in zip(self.doc_ids, doc_keep_mask) if keep]

        # Remove meta-level
        meta_keep_mask = np.array([doc_id not in to_remove for doc_id in self.meta_doc_ids])
        self.meta_vectors = self.meta_vectors[meta_keep_mask]
        self.meta_doc_ids = [d for d, keep in zip(self.meta_doc_ids, meta_keep_mask) if keep]
        self.meta_field_names = [n for n, keep in zip(self.meta_field_names, meta_keep_mask) if keep]
        self.meta_field_texts = [t for t, keep in zip(self.meta_field_texts, meta_keep_mask) if keep]

    def clear(self) -> None:
        self.doc_vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.doc_texts = []
        self.doc_metadata = []
        self.doc_ids = []

        self.meta_vectors = np.empty((0, self.vocab_size), dtype=np.float32)
        self.meta_doc_ids = []
        self.meta_field_names = []
        self.meta_field_texts = []

        self._current_id = 0


class DuckDBVectorStore(VectorStore):
    """Vector store implementation using DuckDB with a two-table approach."""

    uri = param.String(doc="URI of the DuckDB database")

    def __init__(self, **params):
        super().__init__(**params)
        self.connection = duckdb.connect(database=self.uri)
        self._setup_database()

    def _setup_database(self):
        """Set up DuckDB with two tables: documents + metadata_embeddings."""
        self.connection.execute("INSTALL 'vss';")
        self.connection.execute("LOAD 'vss';")
        self.connection.execute("SET hnsw_enable_experimental_persistence = true;")

        # Main documents table
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
            CREATE INDEX IF NOT EXISTS documents_embedding_index
            ON documents USING HNSW(embedding) WITH (metric='cosine');
            """
        )

        # Metadata embeddings table
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS metadata_embeddings (
                id BIGINT DEFAULT NEXTVAL('documents_id_seq') PRIMARY KEY,
                document_id BIGINT,
                field_name VARCHAR,
                field_text VARCHAR,
                field_embedding FLOAT[{self.vocab_size}]
            );
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS metadata_embeddings_index
            ON metadata_embeddings USING HNSW(field_embedding) WITH (metric='cosine');
            """
        )

    def _calculate_combined_score(
        self,
        doc_sim: float,
        meta_sim: float,
        doc_weight: float = 0.55,
        meta_weight: float = 0.45
    ) -> float:
        """
        Example method to combine document & metadata similarities.
        Normalizes from [-1,1] to [0,1] then weights & merges them.
        """
        def normalize(sim):
            return (sim + 1.0) / 2.0  # shift [-1,1] -> [0,1]

        doc_norm = normalize(doc_sim)
        meta_norm = normalize(meta_sim)
        total_weight = doc_weight + meta_weight
        return (doc_weight * doc_norm + meta_weight * meta_norm) / total_weight

    def add(self, items: list[dict]) -> list[int]:
        """
        Insert items into DuckDB:
          - chunk text, embed each chunk, store in documents table
          - optionally embed certain metadata fields, store in metadata_embeddings
        """
        inserted_ids = []
        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {}) or {}

            # Chunk text
            chunks = self._chunk_text(text)
            embeddings = self.embeddings.embed(chunks)

            # Insert each chunk as a row in `documents`
            for chunk_text, emb in zip(chunks, embeddings):
                emb_list = np.array(emb, dtype=np.float32).tolist()
                result = self.connection.execute(
                    """
                    INSERT INTO documents (text, embedding, metadata)
                    VALUES (?, ?, ?::JSON) RETURNING id;
                    """,
                    [chunk_text, emb_list, json.dumps(metadata)],
                )
                doc_id = result.fetchone()[0]
                inserted_ids.append(doc_id)

                # Now embed relevant metadata fields into `metadata_embeddings`
                for key, val in metadata.items():
                    # Example rule: only embed string fields
                    if isinstance(val, str) and val.strip():
                        meta_emb = self.embeddings.embed([val])[0]
                        meta_emb_list = np.array(meta_emb, dtype=np.float32).tolist()
                        self.connection.execute(
                            """
                            INSERT INTO metadata_embeddings (document_id, field_name, field_text, field_embedding)
                            VALUES (?, ?, ?, ?);
                            """,
                            [doc_id, key, val, meta_emb_list],
                        )

        return inserted_ids

    def query(
        self,
        text: str,
        top_k: int = 5,
        filters: dict | None = None,
        doc_threshold: float = 0.0,
        meta_threshold: float = 0.0,
        doc_weight: float = 0.5,
        meta_weight: float = 0.5,
        field_weights: dict | None = None,
    ) -> list[dict]:
        if text.strip() == "":
            return []

        # 1) embed the query
        query_emb = self.embeddings.embed([text])[0]
        query_emb_list = np.array(query_emb, dtype=np.float32).tolist()

        # A) Documents query
        # Build up the WHERE clauses
        doc_conditions = ["1=1"]  # Always true, so we can append "AND condition" easily
        doc_params = [query_emb_list]  # We'll bind the query embedding for the doc_similarity SELECT column

        # If we have filters, each filter => "json_extract_string(metadata, '$.' || ?) = ?"
        if filters:
            for k, v in filters.items():
                doc_conditions.append("json_extract_string(metadata, '$.' || ?) = ?")
                doc_params.extend([k, str(v)])

        # Add doc_threshold condition
        doc_conditions.append(f"1 - array_distance(embedding, ?::FLOAT[{self.vocab_size}]) >= ?")
        doc_params.extend([query_emb_list, doc_threshold])

        # Build final doc query
        doc_sql = f"""
            SELECT
                id,
                text,
                metadata,
                1 - array_distance(embedding, ?::FLOAT[{self.vocab_size}]) AS doc_similarity
            FROM documents
            WHERE {" AND ".join(doc_conditions)}
            ORDER BY doc_similarity DESC
            LIMIT ?;
        """
        doc_params.append(top_k * 5)

        # Execute
        doc_rows = self.connection.execute(doc_sql, doc_params).fetchall()

        # Convert to doc_id -> best doc row
        doc_info_map = {}
        for row in doc_rows:
            doc_id = row[0]
            doc_sim = float(row[3])
            if (
                doc_id not in doc_info_map
                or doc_sim > doc_info_map[doc_id]["doc_similarity"]
            ):
                doc_info_map[doc_id] = {
                    "id": doc_id,
                    "text": row[1],
                    "metadata": json.loads(row[2]),
                    "doc_similarity": doc_sim,
                }

        # B) Metadata query
        # If we have filters, we need to join against documents to apply them
        if filters:
            meta_conditions = ["TRUE"]  # “WHERE TRUE” placeholder
            meta_params = [query_emb_list]

            # Join against documents so we can filter on the same fields
            meta_sql = f"""
                SELECT
                    me.document_id,
                    me.field_name,
                    me.field_text,
                    1 - array_distance(me.field_embedding, ?::FLOAT[{self.vocab_size}]) AS meta_similarity
                FROM metadata_embeddings me
                JOIN documents d ON me.document_id = d.id
                WHERE
            """
            # Add each filter => e.g. `AND json_extract(d.metadata, '$.' || ?) = json(?)`
            for k, v in filters.items():
                meta_conditions.append("json_extract(d.metadata, '$.' || ?) = json(?)")
                meta_params.extend([k, json.dumps(v)])

            # meta_threshold
            meta_conditions.append(f"1 - array_distance(me.field_embedding, ?::FLOAT[{self.vocab_size}]) >= ?")
            meta_params.extend([query_emb_list, meta_threshold])

            meta_sql += " AND ".join(meta_conditions)
            meta_sql += """
                ORDER BY meta_similarity DESC
                LIMIT ?;
            """
            meta_params.append(top_k * 5)

            meta_rows = self.connection.execute(meta_sql, meta_params).fetchall()
        else:
            # Simpler query if no filters are present
            meta_sql = f"""
                SELECT
                    document_id,
                    field_name,
                    field_text,
                    1 - array_distance(field_embedding, ?::FLOAT[{self.vocab_size}]) AS meta_similarity
                FROM metadata_embeddings
                WHERE 1 - array_distance(field_embedding, ?::FLOAT[{self.vocab_size}]) >= ?
                ORDER BY meta_similarity DESC
                LIMIT ?;
            """
            meta_params = [
                query_emb_list,
                query_emb_list,
                meta_threshold,
                top_k * 5
            ]
            meta_rows = self.connection.execute(meta_sql, meta_params).fetchall()

        # Build meta_info_map => doc_id -> best metadata row
        meta_info_map = {}
        for row in meta_rows:
            doc_id = row[0]
            field_name = row[1]
            field_text = row[2]
            similarity = float(row[3])

            # Optional field-level weighting
            field_w = 1.0
            if field_weights and field_name in field_weights:
                field_w = field_weights[field_name]
            weighted_similarity = similarity * field_w

            if (
                doc_id not in meta_info_map
                or weighted_similarity > meta_info_map[doc_id]["meta_similarity"]
            ):
                meta_info_map[doc_id] = {
                    "meta_similarity": weighted_similarity,
                    "field_name": field_name,
                    "field_text": field_text,
                }

        # C) Merge and re-rank
        combined_map = {}
        for doc_id, doc_data in doc_info_map.items():
            doc_sim = doc_data["doc_similarity"]
            meta_sim = meta_info_map.get(doc_id, {}).get("meta_similarity", 0.0)
            combined_score = self._calculate_combined_score(doc_sim, meta_sim, doc_weight, meta_weight)
            combined_map[doc_id] = {
                "id": doc_id,
                "text": doc_data["text"],
                "metadata": doc_data["metadata"],
                "doc_similarity": doc_sim,
                "meta_similarity": meta_sim,
                "combined_score": combined_score,
            }

        # Include doc_ids that appear only in metadata
        for doc_id, meta_data in meta_info_map.items():
            if doc_id not in combined_map:
                row = self.connection.execute(
                    "SELECT text, metadata FROM documents WHERE id = ?;",
                    [doc_id],
                ).fetchone()
                if not row:
                    continue
                doc_text = row[0]
                doc_meta = json.loads(row[1])
                doc_sim = 0.0
                meta_sim = meta_data["meta_similarity"]
                combined_score = self._calculate_combined_score(doc_sim, meta_sim, doc_weight, meta_weight)
                combined_map[doc_id] = {
                    "id": doc_id,
                    "text": doc_text,
                    "metadata": doc_meta,
                    "doc_similarity": doc_sim,
                    "meta_similarity": meta_sim,
                    "combined_score": combined_score,
                }

        # Sort by combined_score desc and pick top_k
        final_list = sorted(
            combined_map.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]

        # Build return format
        results = []
        for item in final_list:
            results.append({
                "id": item["id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "similarity": item["combined_score"],
            })
        return results

    def filter_by(self, filters: dict, limit: int | None = None, offset: int = 0) -> list[dict]:
        """
        Filter items by metadata without using embeddings similarity.
        """
        base_query = """
            SELECT id, text, metadata
            FROM documents
            WHERE TRUE
        """
        params = []

        for k, v in filters.items():
            base_query += " AND json_extract_string(metadata, '$." + k + "') = ?"
            params.append(str(v))

        if offset:
            base_query += " OFFSET ?"
            params.append(offset)
        if limit is not None:
            base_query += " LIMIT ?"
            params.append(limit)

        rows = self.connection.execute(base_query, params).fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "text": row[1],
                "metadata": json.loads(row[2]),
            })
        return results

    def delete(self, ids: list[int]) -> None:
        """
        Delete items by IDs from both documents and metadata_embeddings.
        """
        if not ids:
            return
        placeholders = ", ".join(["?"] * len(ids))

        # Remove from documents
        self.connection.execute(f"DELETE FROM documents WHERE id IN ({placeholders});", ids)
        # Remove associated metadata
        self.connection.execute(
            f"DELETE FROM metadata_embeddings WHERE document_id IN ({placeholders});",
            ids
        )
    def clear(self) -> None:
        """
        Clear all entries from both tables and reset sequence by dropping/recreating everything.
        """
        self.connection.execute("DROP TABLE IF EXISTS metadata_embeddings;")
        self.connection.execute("DROP TABLE IF EXISTS documents;")
        self.connection.execute("DROP SEQUENCE IF EXISTS documents_id_seq;")
        self._setup_database()

    def __del__(self):
        if hasattr(self, 'connection'):
            self.connection.close()
