import re

from abc import abstractmethod
from pathlib import Path

import numpy as np
import param

from .services import AzureOpenAIMixin, LlamaCppMixin, OpenAIMixin

STOP_WORDS = (Path(__file__).parent / "embeddings_stop_words.txt").read_text().splitlines()
STOP_WORDS_RE = re.compile(r"\b(?:{})\b".format("|".join(STOP_WORDS)), re.IGNORECASE)


class Embeddings(param.Parameterized):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
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
    >>> await embeddings.embed(["Hello, world!", "Goodbye, world!"])
    """

    embedding_dim = param.Integer(default=256, doc="The size of the embedding vector")

    hash_func = param.Callable(default=hash, doc="""
        The hashing function to use to map n-grams to the vocabulary.""")

    seed = param.Integer(default=42, doc="The seed for the random number generator.")

    vocab_size = param.Integer(default=5000, doc="The size of the vocabulary.")

    def __init__(self, **params):
        super().__init__(**params)
        self._projection = np.random.Generator(np.random.PCG64(seed=self.seed)).normal(
            0, 1, (self.vocab_size, self.embedding_dim)
        )

    def get_char_ngrams(self, text, n=3):
        text = re.sub(r"\W+", "", text.lower())
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            text = STOP_WORDS_RE.sub("", text)
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


class OpenAIEmbeddings(Embeddings, OpenAIMixin):
    """
    OpenAIEmbeddings is an embeddings class that uses the OpenAI API to generate embeddings.

    :Example:
    >>> embeddings = OpenAIEmbeddings()
    >>> await embeddings.embed(["Hello, world!", "Goodbye, world!"])
    """

    model = param.String(
        default="text-embedding-3-small", doc="The OpenAI model to use."
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.client = self._instantiate_client(async_client=True)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        texts = [text.replace("\n", " ").strip() for text in texts]
        response = await self.client.embeddings.create(input=texts, model=self.model)
        return [r.embedding for r in response.data]


class AzureOpenAIEmbeddings(Embeddings, AzureOpenAIMixin):
    """
    AzureOpenAIEmbeddings is an embeddings class that uses the Azure OpenAI API to generate embeddings.
    Inherits from AzureOpenAIMixin which extends OpenAIMixin, so it has access to all OpenAI functionality
    plus Azure-specific configuration.

    :Example:
    >>> embeddings = AzureOpenAIEmbeddings()
    >>> await embeddings.embed(["Hello, world!", "Goodbye, world!"])
    """

    model = param.String(
        default="text-embedding-3-large", doc="The OpenAI model to use."
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.client = self._instantiate_client(async_client=True)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        response = await self.client.embeddings.create(input=texts, model=self.model)
        return [r.embedding for r in response.data]


class HuggingFaceEmbeddings(Embeddings):
    """
    HuggingFaceEmbeddings is an embeddings class that uses sentence-transformers
    to generate embeddings from Hugging Face models.

    :Example:
    >>> embeddings = HuggingFaceEmbeddings()
    >>> await embeddings.embed(["Hello, world!", "Goodbye, world!"])
    """

    device = param.String(default="cpu", doc="Device to run the model on (e.g., 'cpu' or 'cuda').")

    model = param.String(default="ibm-granite/granite-embedding-107m-multilingual", doc="""
        The Hugging Face model to use.""")

    prompt_name = param.String(default=None, doc="""
        The prompt name to use for encoding queries. If None, no prompt is used.""")

    def __init__(self, **params):
        super().__init__(**params)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model, device=self.device)
        self.embedding_dim = self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        texts = [text.replace("\n", " ") for text in texts]

        # Use prompt_name if specified
        if self.prompt_name:
            embeddings = self._model.encode(texts, prompt_name=self.prompt_name)
        else:
            embeddings = self._model.encode(texts)

        return embeddings.tolist()


class LlamaCppEmbeddings(Embeddings, LlamaCppMixin):
    """
    LlamaCppEmbeddings is an embeddings class that uses the llama-cpp-python
    library to generate embeddings from GGUF models.

    :Example:
    >>> embeddings = LlamaCppEmbeddings(
    ...     model_kwargs={
    ...         "repo_id": "Qwen/Qwen3-Embedding-4B-GGUF",
    ...         "filename": "Qwen3-Embedding-4B-Q4_K_M.gguf",
    ...         "n_ctx": 512,
    ...         "n_batch": 64
    ...     }
    ... )
    >>> await embeddings.embed(["Hello, world!", "Goodbye, world!"])
    """

    model_kwargs = param.Dict(default={
        "default": {
            "repo_id": "Qwen/Qwen3-Embedding-4B-GGUF",
            "filename": "Qwen3-Embedding-4B-Q4_K_M.gguf",
        },
    })

    def __init__(self, **params):
        import llama_cpp
        super().__init__(**params)
        if "pooling_type" not in self.model_kwargs["default"]:
            self.model_kwargs["default"]["pooling_type"] = llama_cpp.LLAMA_POOLING_TYPE_CLS
        self.llm = self._instantiate_client(embedding=True)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = [self.llm.embed(text.replace("\n", " ")) for text in texts]
        return embeddings
