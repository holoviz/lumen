from abc import ABC, abstractmethod

import numpy as np


class Embeddings(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""


class NumpyEmbeddings(Embeddings):
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            words = text.lower().split()
            vector = np.zeros(self.vocab_size)
            for word in words:
                index = hash(word) % self.vocab_size
                vector[index] += 1
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            embeddings.append(vector.tolist())
        return embeddings


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
