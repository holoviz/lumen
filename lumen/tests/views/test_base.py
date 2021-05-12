from lumen.views import View


def test_resolve_module_type():
    assert View._get_type('lumen.views.base.View') is View
