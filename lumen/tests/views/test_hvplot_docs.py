import pytest

from lumen.views import _hvplot_docs
from lumen.views._hvplot_docs import hvplot_param_docs


def test_parses_the_converter_docstring():
    """hvPlot documents its options under custom section titles.

    griffe's numpy parser only recognises a section called ``Parameters``, so
    every one of hvPlot's ten ``... Options`` sections comes back as an
    admonition and nothing is parsed until the titles are rewritten.
    """
    docs = hvplot_param_docs()
    assert len(docs) >= 90


def test_slash_grouped_options_are_expanded():
    """``logx/logy : bool`` documents two options on one line.

    griffe keeps the first name and drops the rest, which loses exactly the
    options the plot views need.
    """
    docs = hvplot_param_docs()
    for name in ('logx', 'logy', 'xlim', 'ylim', 'xlabel', 'ylabel', 'clabel'):
        assert docs[name], name


def test_grouped_siblings_share_the_description():
    docs = hvplot_param_docs()
    assert docs['logx'] == docs['logy']
    assert docs['xlim'] == docs['ylim']


def test_descriptions_are_one_short_line():
    """Every character here is spent again in the prompt on each plot request."""
    for name, description in hvplot_param_docs().items():
        assert '\n' not in description, name
        assert len(description) <= 200, name
        assert description == description.strip(), name


def test_no_option_is_mapped_to_an_empty_description():
    """A grouped entry whose first name has no description must not alias an
    empty one onto its siblings."""
    assert all(hvplot_param_docs().values())


def test_descriptions_are_prose_not_directives():
    """A docstring full of reStructuredText markup is noise to the model."""
    docs = hvplot_param_docs()
    assert '.. note::' not in docs['group_label']
    assert ':doc:' not in docs['by']


def test_an_unparseable_docstring_raises(monkeypatch):
    """A silent empty mapping would degrade the prompt with nothing failing."""
    hvplot_param_docs.cache_clear()
    monkeypatch.setattr(_hvplot_docs.HoloViewsConverter, '__doc__', 'Nothing to parse here.')
    with pytest.raises(RuntimeError, match='hvPlot'):
        hvplot_param_docs()
    hvplot_param_docs.cache_clear()


def test_result_is_cached():
    hvplot_param_docs.cache_clear()
    assert hvplot_param_docs() is hvplot_param_docs()


def test_parsing_is_quiet(capfd):
    """hvPlot's docstring trips several griffe warnings that nobody can act on."""
    hvplot_param_docs.cache_clear()
    hvplot_param_docs()
    assert capfd.readouterr().err == ''
