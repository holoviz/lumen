import datetime as dt
import os
import hashlib

from pathlib import Path

import pandas as pd

from lumen.sources import Source
from lumen.state import state


def test_resolve_module_type():
    assert Source._get_type('lumen.sources.base.Source') is Source

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
    assert ('test', 'A', (1, 2)) in source._cache
    pd.testing.assert_frame_equal(
        source._cache[('test', 'A', (1, 2))],
        pd._testing.makeMixedDataFrame().iloc[1:3]
    )

def test_file_source_get_query_cache_to_file(make_filesource, cachedir):
    root = os.path.dirname(__file__)
    source = make_filesource(root, cache_dir=cachedir)
    source.get('test', A=(1, 2))
    cache_key = ('test', 'A', (1, 2))
    sha = hashlib.sha256(str(cache_key).encode('utf-8')).hexdigest()
    cache_path = Path(cachedir) / f'{sha}_test.parq'
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
    assert ('test', 'A', (1, 2)) in source._cache
    pd.testing.assert_frame_equal(
        source._cache[('test', 'A', (1, 2))].compute(),
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
