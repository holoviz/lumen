import numpy as np
import pytest

from lumen.ai.embeddings import NumpyEmbeddings


def test_numpy_embeddings_get_char_ngrams():
    emb = NumpyEmbeddings()
    # Shorter than n
    assert emb.get_char_ngrams("ab", n=3) == ["ab"]
    assert emb.get_char_ngrams("a", n=3) == ["a"]
    assert emb.get_char_ngrams("x", n=3) == ["x"]
    # Exact n
    assert emb.get_char_ngrams("abc", n=3) == ["abc"]
    # Longer than n
    assert emb.get_char_ngrams("abcd", n=3) == ["abc", "bcd"]
    # Empty or whitespace or punctuation only
    assert emb.get_char_ngrams("", n=3) == []
    assert emb.get_char_ngrams("   ", n=3) == []
    assert emb.get_char_ngrams("!@#$", n=3) == []


async def test_numpy_embeddings_short_strings():
    embeddings = NumpyEmbeddings()
    result = await embeddings.embed(["ab", "xy", "x"])
    assert len(result) == 3
    for vec in result:
        assert len(vec) == embeddings.embedding_dim
        assert not all(v == 0.0 for v in vec)
        assert np.isclose(np.linalg.norm(vec), 1.0)

    # Deterministic
    result2 = await embeddings.embed(["ab"])
    assert result[0] == result2[0]

    result_x = await embeddings.embed(["x"])
    assert result[2] == result_x[0]

    # Different short strings produce different embeddings
    assert result[0] != result[1]
    assert result[0] != result[2]


async def test_numpy_embeddings_whitespace_and_empty():
    embeddings = NumpyEmbeddings()
    result = await embeddings.embed(["", "   ", "\t\n  ", "!@#$"])
    assert len(result) == 4
    for vec in result:
        assert len(vec) == embeddings.embedding_dim
        assert all(v == 0.0 for v in vec)
        assert np.linalg.norm(vec) == 0.0


async def test_numpy_embeddings_longer_strings():
    embeddings = NumpyEmbeddings()
    result = await embeddings.embed(["quantum computing", "data transformation"])
    assert len(result) == 2
    for vec in result:
        assert len(vec) == embeddings.embedding_dim
        assert not all(v == 0.0 for v in vec)
        assert np.isclose(np.linalg.norm(vec), 1.0)

    # Different longer strings produce different embeddings
    assert result[0] != result[1]
