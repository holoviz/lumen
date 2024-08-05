from pathlib import Path

from .config import DEFAULT_EMBEDDINGS_PATH


class Embeddings:

    def add_directory(self, data_dir: Path):
        raise NotImplementedError

    def query(self, query_texts: str) -> list:
        raise NotImplementedError


class ChromaDb(Embeddings):

    def __init__(self, collection: str, persist_dir: str = DEFAULT_EMBEDDINGS_PATH):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(persist_dir / collection))
        self.collection = self.client.get_or_create_collection(collection)

    def add_directory(self, data_dir: Path, file_type='json'):
        add_kwargs = {
            "ids": [],
            "documents": [],
        }
        for i, path in enumerate(data_dir.glob(f"**/*.{file_type}")):
            add_kwargs["ids"].append(f"{i}")
            add_kwargs["documents"].append(path.read_text())
        self.collection.add(**add_kwargs)

    def query(self, query_texts: str) -> list:
        return self.collection.query(query_texts=query_texts)["documents"]
