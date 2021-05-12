from lumen.filters import Filter


def test_resolve_module_type():
    assert Filter._get_type('lumen.filters.base.Filter') is Filter
