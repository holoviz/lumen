import pytest

import datetime as dt
import os

from pathlib import Path

import pandas as pd

from lumen.sources import Source
from lumen.state import state
from lumen.transforms.sql import SQLLimit


@pytest.fixture
def source(make_filesource):
    root = os.path.dirname(__file__)
    return make_filesource(root)


def test_source_resolve_module_type():
    assert Source._get_type('lumen.sources.base.Source') is Source


@pytest.mark.parametrize("filter_col_A", [314, (13, 314), ['A', 314, 'def']])
@pytest.mark.parametrize("filter_col_B", [(3, 15.9)])
@pytest.mark.parametrize("filter_col_C", [[1, 'A', 'def']])
@pytest.mark.parametrize("sql_transforms", [(None, None), (SQLLimit(limit=100), SQLLimit(limit=100))])
def test_source_table_cache_key(source, filter_col_A, filter_col_B, filter_col_C, sql_transforms):
    t1, t2 = sql_transforms
    kwargs1 = {}
    kwargs2 = {}
    if t1 is not None:
        kwargs1['sql_transforms'] = [t1]
    if t2 is not None:
        kwargs2['sql_transforms'] = [t2]
    key1 = source._get_key('test', A=filter_col_A, B=filter_col_B, C=filter_col_C, **kwargs1)
    key2 = source._get_key('test', A=filter_col_A, B=filter_col_B, C=filter_col_C, **kwargs2)
    assert key1 == key2


def test_file_source_get_query(source):
    df = source.get('test', A=(1, 2))
    expected = pd._testing.makeMixedDataFrame().iloc[1:3]
    pd.testing.assert_frame_equal(df, expected)


def test_file_source_variable(make_variable_filesource):
    root = os.path.dirname(__file__)
    source = make_variable_filesource(root)
    df = source.get('test')
    expected = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(df, expected)
    state.variables.tables = {'test': 'test2.csv'}
    df = source.get('test')
    expected = pd._testing.makeMixedDataFrame().iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(df, expected)


def test_file_source_get_query_cache(source):
    source.get('test', A=(1, 2))
    cache_key = source._get_key('test', A=(1, 2))
    assert cache_key in source._cache
    pd.testing.assert_frame_equal(
        source._cache[cache_key],
        pd._testing.makeMixedDataFrame().iloc[1:3]
    )
    cache_key = source._get_key('test', A=(1, 2))
    assert cache_key in source._cache


def test_file_source_get_query_cache_to_file(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root, cache_dir=cachedir)
    source.get('test', A=(1, 2))

    cache_key = source._get_key('test', A=(1, 2))
    cache_path = Path(cachedir) / f'{cache_key}_test.parq'
    df = pd.read_parquet(cache_path, engine='fastparquet')

    # Patch index names due to https://github.com/dask/fastparquet/issues/732
    df.index.names = [None]
    pd.testing.assert_frame_equal(
        df,
        pd._testing.makeMixedDataFrame().iloc[1:3]
    )


def test_file_source_get_query_dask(source):
    df = source.get('test', A=(1, 2), __dask=True)
    expected = pd._testing.makeMixedDataFrame().iloc[1:3]
    pd.testing.assert_frame_equal(df, expected)


def test_file_source_get_query_dask_cache(source):
    source.get('test', A=(1, 2), __dask=True)
    cache_key = source._get_key('test', A=(1, 2))
    assert cache_key in source._cache
    pd.testing.assert_frame_equal(
        source._cache[cache_key].compute(),
        pd._testing.makeMixedDataFrame().iloc[1:3]
    )


def test_file_source_filter_int(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', A=1)
    expected = df[df.A==1]
    pd.testing.assert_frame_equal(filtered, expected)


def test_file_source_filter_str(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', C='foo2')
    expected = df[df.C=='foo2']
    pd.testing.assert_frame_equal(filtered, expected)


def test_file_source_filter_int_range(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', A=(1, 3))
    expected = df[(df.A>=1) & (df.A<=3)]
    pd.testing.assert_frame_equal(filtered, expected)


def test_file_source_filter_date(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', D=dt.date(2009, 1, 2))
    expected = df.iloc[1:2]
    pd.testing.assert_frame_equal(filtered, expected)


def test_file_source_filter_datetime(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', D=dt.datetime(2009, 1, 2))
    expected = df.iloc[1:2]
    pd.testing.assert_frame_equal(filtered, expected)


def test_file_source_filter_datetime_range(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', D=(dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)))
    expected = df.iloc[1:3]
    pd.testing.assert_frame_equal(filtered, expected)


def test_file_source_filter_date_range(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', D=(dt.date(2009, 1, 2), dt.date(2009, 1, 5)))
    expected = df.iloc[1:3]
    pd.testing.assert_frame_equal(filtered, expected)


def test_file_source_filter_int_range_list(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', A=[(0, 1), (3, 4)])
    expected = df[((df.A>=0) & (df.A<=1)) | ((df.A>=3) & (df.A<=4))]
    pd.testing.assert_frame_equal(filtered, expected)


def test_file_source_filter_list(source):
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', C=['foo1', 'foo3'])
    expected = df[df.C.isin(['foo1', 'foo3'])]
    pd.testing.assert_frame_equal(filtered, expected)
