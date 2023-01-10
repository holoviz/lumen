import os
import pathlib

import pandas as pd
import param  # type: ignore

from lumen.transforms.base import (
    Count, Eval, Sum, Transform,
)


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
            'options': '$original.test.C'
        }]
    })
    assert transform.param.value.objects == ['foo1', 'foo2', 'foo3', 'foo4', 'foo5']


def test_count_transform():
    df = pd._testing.makeMixedDataFrame()

    count_df = Count.apply_to(df)

    pd.testing.assert_frame_equal(count_df, df.count().to_frame().T)


def test_sum_transform():
    df = pd._testing.makeMixedDataFrame()[['A']]

    sum_df = Sum.apply_to(df)

    pd.testing.assert_frame_equal(sum_df, df.sum().to_frame().T)


def test_eval_transform():
    df = pd._testing.makeMixedDataFrame()[['A']]

    eval_df = Eval.apply_to(df, expr='B = table.A * 2')

    df2 = df.copy()
    df2['B'] = df.A * 2

    pd.testing.assert_frame_equal(eval_df, df2)
