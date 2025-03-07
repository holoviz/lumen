import re

from abc import abstractmethod

import numpy as np
import param


class Embeddings(param.Parameterized):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""


class NumpyEmbeddings(Embeddings):
    """
    NumpyEmbeddings is a simple embeddings class that uses a hash function
    to map n-grams to the vocabulary.

    Note that the default hash function is not stable across different Python
    sessions. If you need a stable hash function, you can set the `hash_func`,
    e.g. using murmurhash from the `mmh3` package.

    :Example:
    >>> embeddings = NumpyEmbeddings()
    >>> embeddings.embed(["Hello, world!", "Goodbye, world!"])
    """

    hash_func = param.Callable(default=hash, doc="""
        The hashing function to use to map n-grams to the vocabulary.""")

    embedding_dim = param.Integer(default=256, doc="The size of the embedding vector")

    vocab_size = param.Integer(default=5000, doc="The size of the vocabulary.")

    def __init__(self, **params):
        super().__init__(**params)
        self._projection = np.random.Generator(np.random.PCG64()).normal(
            0, 1, (self.vocab_size, self.embedding_dim)
        )

    def get_char_ngrams(self, text, n=3):
        text = re.sub(r"\W+", "", text.lower())
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            ngrams = self.get_char_ngrams(text)
            vector = np.zeros(self.vocab_size)

            for ngram in ngrams:
                index = self.hash_func(ngram) % self.vocab_size
                vector[index] += 1

            dense_vector = np.dot(vector, self._projection)
            norm = np.linalg.norm(dense_vector)
            if norm > 0:
                dense_vector /= norm
            embeddings.append(dense_vector.tolist())
        return embeddings


class OpenAIEmbeddings(Embeddings):
    """
    OpenAIEmbeddings is an embeddings class that uses the OpenAI API to generate embeddings.

    :Example:
    >>> embeddings = OpenAIEmbeddings()
    >>> embeddings.embed(["Hello, world!", "Goodbye, world!"])
    """

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
    """
    AzureOpenAIEmbeddings is an embeddings class that uses the Azure OpenAI API to generate embeddings.

    :Example:
    >>> embeddings = AzureOpenAIEmbeddings()
    >>> embeddings.embed(["Hello, world!", "Goodbye, world!"])
    """
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


class HuggingFaceEmbeddings(Embeddings):
    """
    HuggingFaceEmbeddings is an embeddings class that uses sentence-transformers plus
    a tokenizer model downloaded from Hugging Face to generate embeddings.

    :Example:
    >>> embeddings = HuggingFaceEmbeddings()
    >>> embeddings.embed(["Hello, world!", "Goodbye, world!"])
    """
    device = param.String(default="cpu", doc="Device to run the model on (e.g., 'cpu' or 'cuda').")

    model = param.String(default="sentence-transformers/all-MiniLM-L6-v2", doc="""
        The Hugging Face model to use.""")

    def __init__(self, **params):
        super().__init__(**params)
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModel.from_pretrained(self.model).to(self.device)
        self.embedding_dim = self._model.config.hidden_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        import torch
        texts = [text.replace("\n", " ") for text in texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()  # Use [CLS] token embeddings
        return embeddings
