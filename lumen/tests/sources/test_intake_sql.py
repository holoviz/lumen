import datetime as dt
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
    expected_sql = {
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
    expected_csv = dict(expected_sql, D={
        'format': 'datetime',
        'inclusiveMaximum': '2009-01-07T00:00:00',
        'inclusiveMinimum': '2009-01-01T00:00:00',
        'type': 'string'
    })
    assert source.get_schema('test_sql') == expected_sql
    assert 'test' not in source._schema_cache
    assert 'test_sql' in source._schema_cache
    assert source.get_schema('test') == expected_csv
    assert 'test' in source._schema_cache
    assert 'test_sql' in source._schema_cache

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
    source.clear_cache()

def test_intake_sql_filter_int():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', A=1)
    expected = df[df.A==1].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)
    source.clear_cache()

def test_intake_sql_filter_str():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', C='foo2')
    expected = df[df.C=='foo2'].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)

def test_intake_sql_filter_int_range():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', A=(1, 3))
    expected = df[(df.A>=1) & (df.A<=3)].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)

def test_intake_sql_filter_date():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', D=dt.date(2009, 1, 2))
    expected = df.iloc[1:2].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)

def test_intake_sql_filter_datetime():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', D=dt.datetime(2009, 1, 2))
    expected = df.iloc[1:2].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)

def test_intake_sql_filter_datetime_range():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', D=(dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)))
    expected = df.iloc[1:3].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)

def test_intake_sql_filter_date_range():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', D=(dt.date(2009, 1, 2), dt.date(2009, 1, 5)))
    expected = df.iloc[1:3].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)

def test_intake_sql_filter_int_range_list():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', A=[(0, 1), (3, 4)])
    expected = df[((df.A>=0) & (df.A<=1)) | ((df.A>=3) & (df.A<=4))].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)

def test_intake_sql_filter_list():
    root = os.path.dirname(__file__)
    source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', C=['foo1', 'foo3'])
    expected = df[df.C.isin(['foo1', 'foo3'])].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)

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
