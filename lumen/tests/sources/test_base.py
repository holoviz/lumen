from lumen.sources import Source


def test_resolve_module_type():
    assert Source._get_type('lumen.sources.base.Source') is Source

