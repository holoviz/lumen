from lumen.ai.embeddings import STOP_WORDS_RE


def test_stop_words_re_literal_escaping():
    assert STOP_WORDS_RE.sub("", "co2") == "co2"
    assert STOP_WORDS_RE.sub("", "inco") == "inco"
