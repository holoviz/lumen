from abc import ABC, abstractmethod


class Embeddings(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""


class OpenAIEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [r.embedding for r in response.data]
