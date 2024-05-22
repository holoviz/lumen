import uuid

from pathlib import Path

DEFAULT_PATH = Path("embeddings")


class Embeddings:

    def add_directory(self, data_dir: Path):
        raise NotImplementedError

    def query(self, query_texts: str) -> list:
        raise NotImplementedError


class ChromaDb(Embeddings):

    def __init__(self, collection: str, persist_dir: str = DEFAULT_PATH):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(persist_dir / collection))
        self.collection = self.client.get_or_create_collection(collection)

    def _add_to_collection(self, upsert: bool = True, **add_kwargs):
        if upsert:
            self.collection.upsert(**add_kwargs)
        else:
            self.collection.add(**add_kwargs)

    def add_directory(self, data_dir: Path, file_type="json", upsert: bool = True, **add_kwargs):
        ids = []
        documents = []
        for path in data_dir.glob(f"**/*.{file_type}"):
            ids.append(uuid.uuid4().hex)
            documents.append(path.read_text())
        self._add_to_collection(upsert=upsert, ids=ids, documents=documents, **add_kwargs)

    def add_documents(self, documents: list, ids: list | None = None, upsert: bool = True, **add_kwargs):
        if ids is None:
            ids = [uuid.uuid4().hex for _ in documents]
        else:
            ids = list(ids)
        self._add_to_collection(upsert=upsert, ids=ids, documents=documents, **add_kwargs)

    def query(self, query_text: list | str, n_results: int = 1, **query_kwargs) -> list:
        return self.collection.query(query_texts=[query_text], n_results=n_results, **query_kwargs)["documents"][0]
