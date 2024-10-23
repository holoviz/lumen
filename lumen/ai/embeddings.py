import os

from pathlib import Path

import duckdb

DEFAULT_EMBEDDINGS_PATH = Path("embeddings")


class Embeddings:
    def __init__(self, database_path: str = ":memory:"):
        self.database_path = database_path
        self.connection = duckdb.connect(database_path)
        self.setup_database()

    def setup_database(self):
        self.connection.execute(
            """
            INSTALL vss;
            LOAD vss;
            CREATE TABLE document_data (
                id INTEGER,
                text VARCHAR,
                embedding FLOAT[1536],
                table_name VARCHAR
            );
            CREATE INDEX embedding_index ON document_data USING HNSW (embedding) WITH (metric = 'cosine');
            """
        )

    @classmethod
    def from_directory(
        cls,
        data_dir: Path,
        file_type: str = "json",
        database_path: str = ":memory:",
        table_name: str = "default",
    ):
        embeddings = cls(database_path)
        for i, path in enumerate(data_dir.glob(f"**/*.{file_type}")):
            text = path.read_text()
            embedding = embeddings.get_embedding(text)
            embeddings.connection.execute(
                """
                INSERT INTO document_data (id, text, embedding, table_name)
                VALUES (?, ?, ?, ?);
                """,
                [i, text, embedding, table_name],
            )
        return embeddings

    @classmethod
    def from_dict(cls, data: dict, database_path: str = ":memory:"):
        embeddings = cls(database_path)
        global_id = 0
        for table_name, texts in data.items():
            for text in texts:
                embedding = embeddings.get_embedding(text)
                embeddings.connection.execute(
                    """
                    INSERT INTO document_data (id, text, embedding, table_name)
                    VALUES (?, ?, ?, ?);
                    """,
                    [global_id, text, embedding, table_name],
                )
                global_id += 1
        return embeddings

    def get_embedding(self, text: str) -> list:
        raise NotImplementedError

    def get_text_chunks(
        self, text: str, chunk_size: int = 512, overlap: int = 50
    ) -> list:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks

    def get_combined_embedding(self, text: str) -> list:
        chunks = self.get_text_chunks(text)
        embeddings = [self.get_embedding(chunk) for chunk in chunks]
        combined_embedding = [sum(x) / len(x) for x in zip(*embeddings)]
        return combined_embedding

    def query(self, query_text: str, top_k: int = 1, table_name: str | None = None) -> list:
        print(query_text, "QUERY")
        query_embedding = self.get_combined_embedding(query_text)

        if table_name:
            result = self.connection.execute(
                """
                SELECT id, text, array_cosine_similarity(embedding, ?::FLOAT[1536]) AS similarity, table_name
                FROM document_data
                WHERE table_name = ?
                ORDER BY similarity DESC
                LIMIT ?;
                """,
                [query_embedding, table_name, top_k],
            ).fetchall()
        else:
            result = self.connection.execute(
                """
                SELECT id, text, array_cosine_similarity(embedding, ?::FLOAT[1536]) AS similarity, table_name
                FROM document_data
                ORDER BY similarity DESC
                LIMIT ?;
                """,
                [query_embedding, top_k],
            ).fetchall()

        return result

    def close(self):
        self.connection.close()


class OpenAIEmbeddings(Embeddings):
    def __init__(
        self, database_path: str = ":memory:", model: str = "text-embedding-3-small"
    ):
        super().__init__(database_path)
        self.model = model

    def get_embedding(self, text: str) -> list:
        from openai import OpenAI

        text = text.replace("\n", " ")
        return (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            .embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )
