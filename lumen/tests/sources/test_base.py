import os
import hashlib

from pathlib import Path

import pandas as pd

from lumen.sources import Source


def test_resolve_module_type():
    assert Source._get_type('lumen.sources.base.Source') is Source

def test_file_source_get_query(make_filesource):
    root = os.path.dirname(__file__)
    source = make_filesource(root)
    df = source.get('test', A=(1, 2))
    expected = pd._testing.makeMixedDataFrame().iloc[1:3]
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
