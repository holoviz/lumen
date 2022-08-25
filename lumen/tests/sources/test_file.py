import datetime as dt
import os

from pathlib import Path

import pandas as pd
import pytest

from lumen.sources import FileSource
from lumen.state import state
from lumen.transforms.sql import SQLLimit

from .utils import (
    source_clear_cache_get_query, source_clear_cache_get_schema, source_filter,
    source_get_cache_query, source_get_schema_cache,
    source_get_schema_update_cache, source_get_tables, source_table_cache_key,
)


@pytest.fixture
def source(make_filesource):
    root = os.path.dirname(__file__)
    return make_filesource(root)


@pytest.fixture
def source_tables():
    return {'test': pd._testing.makeMixedDataFrame()}


def test_source_resolve_module_type():
    assert FileSource._get_type('lumen.sources.FileSource') is FileSource
    assert FileSource.source_type == 'file'


def test_file_source_get_tables(source, source_tables):
    assert source_get_tables(source, source_tables)


@pytest.mark.parametrize("table", ['test'])
@pytest.mark.parametrize("filter_col_A", [314, (13, 314), ['A', 314, 'def']])
@pytest.mark.parametrize("filter_col_B", [(3, 15.9)])
@pytest.mark.parametrize("filter_col_C", [[1, 'A', 'def']])
@pytest.mark.parametrize("sql_transforms", [(None, None), (SQLLimit(limit=100), SQLLimit(limit=100))])
def test_file_source_table_cache_key(source, table, filter_col_A, filter_col_B, filter_col_C, sql_transforms):
    t1, t2 = sql_transforms
    kwargs1 = {'A': filter_col_A, 'B': filter_col_B, 'C': filter_col_C}
    kwargs2 = {'A': filter_col_A, 'B': filter_col_B, 'C': filter_col_C}
    if t1 is not None:
        kwargs1['sql_transforms'] = [t1]
    if t2 is not None:
        kwargs2['sql_transforms'] = [t2]
    assert source_table_cache_key(source, table, kwargs1, kwargs2)


@pytest.mark.parametrize(
    "table_column_value_type", [
        ('test', 'A', 1, 'single_value'),
        ('test', 'A', (1, 3), 'range'),
        ('test', 'A', (1, 2), 'range'),
        ('test', 'A', [(0, 1), (3, 4)], 'range_list'),
        ('test', 'C', 'foo2', 'single_value'),
        ('test', 'C', ['foo1', 'foo3'], 'list'),
        ('test', 'D', dt.datetime(2009, 1, 2), 'single_value'),
        ('test', 'D', (dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)), 'range'),
        ('test', 'D', [dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)], 'list'),
        ('test', 'D', dt.datetime(2009, 1, 2), 'date'),
        ('test', 'D', (dt.date(2009, 1, 2), dt.date(2009, 1, 5)), 'date_range'),
    ]
)
@pytest.mark.parametrize("dask", [True, False])
def test_file_source_filter(source, table_column_value_type, dask, expected_filtered_df):
    assert source_filter(source, table_column_value_type, dask, expected_filtered_df)


@pytest.mark.parametrize(
    "table_column_value_type", [
        ('test', 'A', 1, 'single_value'),
        ('test', 'A', (1, 3), 'range'),
        ('test', 'A', (1, 2), 'range'),
        ('test', 'A', [(0, 1), (3, 4)], 'range_list'),
        ('test', 'C', 'foo2', 'single_value'),
        ('test', 'C', ['foo1', 'foo3'], 'list'),
        ('test', 'D', dt.datetime(2009, 1, 2), 'single_value'),
        ('test', 'D', (dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)), 'range'),
        ('test', 'D', [dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)], 'list'),
        ('test', 'D', dt.datetime(2009, 1, 2), 'date'),
        ('test', 'D', (dt.date(2009, 1, 2), dt.date(2009, 1, 5)), 'date_range'),
    ]
)
@pytest.mark.parametrize("dask", [True, False])
def test_file_source_get_cache_query(source, table_column_value_type, dask, expected_filtered_df):
    assert source_get_cache_query(source, table_column_value_type, dask, expected_filtered_df)


def test_file_source_get_schema_cache(source):
    assert source_get_schema_cache(source, table='test')


def test_file_source_get_schema_update_cache(source):
    assert source_get_schema_update_cache(source, table='test')


@pytest.mark.parametrize(
    "table_column_value_type", [
        ('test', 'A', 1, 'single_value'),
        ('test', 'A', [(0, 1), (3, 4)], 'range_list'),
        ('test', 'C', ['foo1', 'foo3'], 'list'),
        ('test', 'D', (dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)), 'range'),
        ('test', 'D', dt.datetime(2009, 1, 2), 'date'),
    ]
)
@pytest.mark.parametrize("dask", [True])
def test_file_source_clear_cache_get_query(source, table_column_value_type, dask):
    table, column, value, _ = table_column_value_type
    kwargs = {column: value}
    if dask is not None:
        kwargs['__dask'] = dask
    assert source_clear_cache_get_query(source, table, kwargs)


def test_file_source_clear_cache_get_schema(source):
    assert source_clear_cache_get_schema(source, table='test')


def test_file_source_get_cache_query_to_file(make_filesource, cachedir):
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


def test_file_source_variable(make_variable_filesource):
    root = os.path.dirname(__file__)
    source = make_variable_filesource(root)
    state.variables.tables = {'test': 'test2.csv'}
    df = source.get('test')
    expected = pd._testing.makeMixedDataFrame().iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(df, expected)


def test_file_source_extension_of_comlicated_url(source):
    url = "https://api.tfl.gov.uk/Occupancy/BikePoints/@{stations.stations.id}?app_key=random_numbers"
    source.tables["test"] = url
    assert source._named_files["test"][1] is None
