import pytest

import datetime as dt
import os

import pandas as pd

from lumen.sources.intake_sql import IntakeSQLSource
from lumen.transforms.sql import SQLGroupBy


@pytest.fixture
def source():
    root = os.path.dirname(__file__)
    intake_sql_source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    return intake_sql_source


@pytest.fixture
def df_test():
    return pd._testing.makeMixedDataFrame()


@pytest.fixture
def df_test_sql():
    return pd._testing.makeMixedDataFrame()


@pytest.fixture
def df_test_sql_none():
    df = pd._testing.makeMixedDataFrame()
    df['C'] = ['foo1', None, 'foo3', None, 'foo5']
    return df


def test_intake_sql_get_tables(source):
    tables = source.get_tables()
    assert tables == ['test', 'test_sql', 'test_sql_with_none']


def test_intake_sql_get(source, df_test, df_test_sql, df_test_sql_none):
    pd.testing.assert_frame_equal(source.get('test'), df_test)
    pd.testing.assert_frame_equal(source.get('test_sql'), df_test_sql)
    pd.testing.assert_frame_equal(source.get('test_sql_with_none'), df_test_sql_none)


def test_intake_sql_get_schema(source):
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
    print('expected_csv', expected_sql, expected_csv)
    assert source.get_schema('test_sql') == expected_sql
    assert 'test' not in source._schema_cache
    assert 'test_sql' in source._schema_cache
    assert source.get_schema('test') == expected_csv
    assert 'test' in source._schema_cache
    assert 'test_sql' in source._schema_cache


def test_intake_sql_get_schema_with_none(source):
    expected_sql = {
        'A': {'inclusiveMaximum': 4.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'B': {'inclusiveMaximum': 1.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'C': {'enum': ['foo1', None, 'foo3', 'foo5'], 'type': 'string'},
        'D': {
            'format': 'datetime',
            'inclusiveMaximum': '2009-01-07 00:00:00',
            'inclusiveMinimum': '2009-01-01 00:00:00',
            'type': 'string'
        }
    }
    assert source.get_schema('test_sql_with_none') == expected_sql
    assert 'test' not in source._schema_cache
    assert 'test_sql_with_none' in source._schema_cache


def test_intake_sql_transforms(source):
    df = pd._testing.makeMixedDataFrame()
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]
    transformed = source.get('test_sql', sql_transforms=transforms)
    expected = df.groupby('B')['A'].sum().reset_index()
    pd.testing.assert_frame_equal(transformed, expected)
    source.clear_cache()


def test_intake_sql_filter_int(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', A=1)
    expected = df[df.A==1].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)
    source.clear_cache()


def test_intake_sql_filter_None(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql_with_none', C=None)
    expected = df[(df.A==1) | (df.A==3)].reset_index(drop=True)
    expected['C'] = None
    pd.testing.assert_frame_equal(filtered, expected)
    source.clear_cache()


def test_intake_sql_filter_str(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', C='foo2')
    expected = df[df.C=='foo2'].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)


def test_intake_sql_filter_int_range(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', A=(1, 3))
    expected = df[(df.A>=1) & (df.A<=3)].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)


def test_intake_sql_filter_date(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', D=dt.date(2009, 1, 2))
    expected = df.iloc[1:2].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)


def test_intake_sql_filter_datetime(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', D=dt.datetime(2009, 1, 2))
    expected = df.iloc[1:2].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)


def test_intake_sql_filter_datetime_range(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', D=(dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)))
    expected = df.iloc[1:3].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)


def test_intake_sql_filter_date_range(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', D=(dt.date(2009, 1, 2), dt.date(2009, 1, 5)))
    expected = df.iloc[1:3].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)


def test_intake_sql_filter_int_range_list(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', A=[(0, 1), (3, 4)])
    expected = df[((df.A>=0) & (df.A<=1)) | ((df.A>=3) & (df.A<=4))].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)


def test_intake_sql_filter_list(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql', C=['foo1', 'foo3'])
    expected = df[df.C.isin(['foo1', 'foo3'])].reset_index(drop=True)
    pd.testing.assert_frame_equal(filtered, expected)


def test_intake_sql_filter_list_with_None(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test_sql_with_none', C=[None, 'foo5'])
    expected = df[df.A.isin([1, 3, 4])].reset_index(drop=True)
    expected['C'] = [None, None, 'foo5']
    pd.testing.assert_frame_equal(filtered, expected)
    source.clear_cache()


def test_intake_sql_transforms_cache(source):
    df = pd._testing.makeMixedDataFrame()
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]
    source.get('test_sql', sql_transforms=transforms)
    expected = df.groupby('B')['A'].sum().reset_index()
    cache_key = source._get_key('test_sql', sql_transforms=transforms)
    assert cache_key in source._cache
    pd.testing.assert_frame_equal(source._cache[cache_key], expected)

    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]
    cache_key = source._get_key('test_sql', sql_transforms=transforms)
    assert cache_key in source._cache
