import os

import pandas as pd

from lumen.sources.intake import IntakeSQLSource
from lumen.transforms.sql import SQLGroupBy


def test_intake_sql_get():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(source.get('test_sql'), df)

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
