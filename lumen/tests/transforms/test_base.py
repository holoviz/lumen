import os
import pathlib

import param

from lumen.transforms import Transform


class CustomTransform(Transform):

    value = param.Selector()

    transform_type = 'test'


def test_resolve_module_type():
    assert Transform._get_type('lumen.transforms.base.Transform') is Transform


def test_transform_control_options(make_filesource):
    root = os.path.dirname(__file__)
    make_filesource(root)

    transform = Transform.from_spec({
        'type': 'test',
        'controls': [{
            'name': 'value',
            'options': ['A', 'B', 'C']
        }]
    })
    assert transform.param.value.objects == ['A', 'B', 'C']


def test_transform_control_options_by_reference(make_filesource):
    root = pathlib.Path(__file__).parent / '..' / 'sources'
    make_filesource(str(root))

    transform = Transform.from_spec({
        'type': 'test',
        'controls': [{
            'name': 'value',
            'options': '@original.test.C'
        }]
    })
    assert transform.param.value.objects == ['foo1', 'foo2', 'foo3', 'foo4', 'foo5']
