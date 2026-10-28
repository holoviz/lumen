from lumen.ai.embeddings import STOP_WORDS_RE


def test_stop_words_re_literal_escaping():
    # "co2" should not be matched by "co."
    assert not STOP_WORDS_RE.search("co2")
    assert STOP_WORDS_RE.sub("", "co2") == "co2"

    # "co." should be matched literally
    assert STOP_WORDS_RE.search("co.")
    assert STOP_WORDS_RE.search("Acme Co. announced")
