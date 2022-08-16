import datetime as dt
import os

from pathlib import Path

import pandas as pd

from lumen.sources import Source
from lumen.state import state
from lumen.transforms.sql import SQLLimit


def test_resolve_module_type():
    assert Source._get_type('lumen.sources.base.Source') is Source

def test_source_table_cache_key(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    assert source._get_key('test') == source._get_key('test')

def test_source_table_and_int_query_cache_key(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    assert source._get_key('test', A=314) == source._get_key('test', A=314)

def test_source_table_and_range_query_cache_key(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    assert source._get_key('test', A=(13, 314)) == source._get_key('test', A=(13, 314))

def test_source_table_and_list_query_cache_key(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    assert source._get_key('test', A=['A', 314, 'def']) == source._get_key('test', A=['A', 314, 'def'])

def test_source_table_and_sql_transform_cache_key(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    t1 = SQLLimit(limit=100)
    t2 = SQLLimit(limit=100)
    assert source._get_key('test', sql_transforms=[t1]) == source._get_key('test', sql_transforms=[t2])

def test_source_table_and_multi_query_sql_transform_cache_key(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    t1 = SQLLimit(limit=100)
    t2 = SQLLimit(limit=100)
    assert (
        source._get_key('test', A=1, B=(3, 15.9), C=[1, 'A', 'def'], sql_transforms=[t1]) ==
        source._get_key('test', A=1, B=(3, 15.9), C=[1, 'A', 'def'], sql_transforms=[t2])
    )

def test_file_source_get_query(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
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

def test_file_source_get_query_cache(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
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

def test_file_source_get_query_dask(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = source.get('test', A=(1, 2), __dask=True)
    expected = pd._testing.makeMixedDataFrame().iloc[1:3]
    pd.testing.assert_frame_equal(df, expected)

def test_file_source_get_query_dask_cache(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    source.get('test', A=(1, 2), __dask=True)
    cache_key = source._get_key('test', A=(1, 2))
    assert cache_key in source._cache
    pd.testing.assert_frame_equal(
        source._cache[cache_key].compute(),
        pd._testing.makeMixedDataFrame().iloc[1:3]
    )

def test_file_source_filter_int(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', A=1)
    expected = df[df.A==1]
    pd.testing.assert_frame_equal(filtered, expected)

def test_file_source_filter_str(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', C='foo2')
    expected = df[df.C=='foo2']
    pd.testing.assert_frame_equal(filtered, expected)

def test_file_source_filter_int_range(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', A=(1, 3))
    expected = df[(df.A>=1) & (df.A<=3)]
    pd.testing.assert_frame_equal(filtered, expected)

def test_file_source_filter_date(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', D=dt.date(2009, 1, 2))
    expected = df.iloc[1:2]
    pd.testing.assert_frame_equal(filtered, expected)

def test_file_source_filter_datetime(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', D=dt.datetime(2009, 1, 2))
    expected = df.iloc[1:2]
    pd.testing.assert_frame_equal(filtered, expected)

def test_file_source_filter_datetime_range(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', D=(dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)))
    expected = df.iloc[1:3]
    pd.testing.assert_frame_equal(filtered, expected)

def test_file_source_filter_date_range(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', D=(dt.date(2009, 1, 2), dt.date(2009, 1, 5)))
    expected = df.iloc[1:3]
    pd.testing.assert_frame_equal(filtered, expected)

def test_file_source_filter_int_range_list(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', A=[(0, 1), (3, 4)])
    expected = df[((df.A>=0) & (df.A<=1)) | ((df.A>=3) & (df.A<=4))]
    pd.testing.assert_frame_equal(filtered, expected)

def test_file_source_filter_list(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = pd._testing.makeMixedDataFrame()
    filtered = source.get('test', C=['foo1', 'foo3'])
    expected = df[df.C.isin(['foo1', 'foo3'])]
    pd.testing.assert_frame_equal(filtered, expected)
