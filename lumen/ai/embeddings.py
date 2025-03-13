import re

from abc import abstractmethod
from typing import Any

import numpy as np
import param

from .utils import deserialize_from_spec, hash_spec, serialize_to_spec


class Embeddings(param.Parameterized):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""

    def to_spec(self) -> dict[str, Any]:
        """Return a serializable specification of this embeddings configuration."""
        return serialize_to_spec(self)

    def from_spec(self, spec: dict[str, Any]) -> "Embeddings":
        """Create an embeddings configuration from a specification."""
        return deserialize_from_spec(spec)

    @property
    def hash(self) -> str:
        """A deterministic hash of this embeddings configuration."""
        return hash_spec(self.to_spec())


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


class HuggingFaceEmbeddings:

    device = param.String(default="cpu", doc="Device to run the model on (e.g., 'cpu' or 'cuda').")

    model = param.String(default="sentence-transformers/all-MiniLM-L6-v2", doc="""
        The Hugging Face model to use.""")

    def __init__(self, **params):
        super().__init__(**params)
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModel.from_pretrained(self.model).to(self.device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        import torch
        texts = [text.replace("\n", " ") for text in texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()  # Use [CLS] token embeddings
        return embeddings
