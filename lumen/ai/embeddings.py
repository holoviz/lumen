import re

from abc import abstractmethod

import numpy as np
import param


class Embeddings(param.Parameterized):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""


class NumpyEmbeddings(Embeddings):

    vocab_size = param.Integer(default=1536, doc="The size of the vocabulary.")

    def get_char_ngrams(self, text, n=3):
        text = re.sub(r"\W+", "", text.lower())
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            ngrams = self.get_char_ngrams(text)
            vector = np.zeros(self.vocab_size)
            for ngram in ngrams:
                index = hash(ngram) % self.vocab_size
                vector[index] += 1
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            embeddings.append(vector.tolist())
        return embeddings


class OpenAIEmbeddings(Embeddings):

    api_key = param.String(doc="The OpenAI API key.")

    model = param.String(
        default="text-embedding-3-small", doc="The OpenAI model to use."
    )

    def __init__(self, **params):
        super().__init__(**params)
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [r.embedding for r in response.data]


class AzureOpenAIEmbeddings(Embeddings):

    api_key = param.String(doc="The Azure API key.")

    api_version = param.String(doc="The Azure AI Studio API version.")

    provider_endpoint = param.String(doc="The Azure AI Studio endpoint.")

    model = param.String(
        default="text-embedding-3-large", doc="The OpenAI model to use."
    )

    def __init__(self, **params):
        super().__init__(**params)
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.provider_endpoint,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [r.embedding for r in response.data]
