import os

import pandas as pd

from lumen.sources.intake_sql import IntakeSQLSource
from lumen.transforms.sql import SQLGroupBy


def test_intake_sql_get():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(source.get('test_sql'), df)

def test_intake_sql_get_schema():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    schema = source.get_schema('test_sql')
    assert schema == {
        'A': {'inclusiveMaximum': 4.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'B': {'inclusiveMaximum': 1.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'C': {'enum': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], 'type': 'string'},
        'D': {
            'format': 'datetime',
            'inclusiveMaximum': '2009-01-07 00:00:00',
            'inclusiveMinimum': '2009-01-01 00:00:00',
            'type': 'string'
        }
    }

def test_intake_sql_transforms():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]
    transformed = source.get('test_sql', sql_transforms=transforms)
    expected = df.groupby('B')['A'].sum().reset_index()
    pd.testing.assert_frame_equal(transformed, expected)

def test_intake_sql_transforms_cache():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]
    source.get('test_sql', sql_transforms=transforms)
    expected = df.groupby('B')['A'].sum().reset_index()
    cache_key = ('test_sql', 'sql_transforms', tuple(transforms))
    assert cache_key in source._cache
    pd.testing.assert_frame_equal(source._cache[cache_key], expected)
