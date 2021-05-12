from lumen.transforms import Transform


def test_resolve_module_type():
    assert Transform._get_type('lumen.transforms.base.Transform') is Transform
