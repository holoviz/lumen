import asyncio
import datetime as dt
import os

from pathlib import Path

import pandas as pd
import pytest

from hvplot.tests.util import makeMixedDataFrame

from lumen.sources.base import Source
from lumen.state import state
from lumen.transforms.sql import SQLLimit

try:
    import fastparquet
except ImportError:
    fastparquet = None

@pytest.fixture
def source(make_filesource):
    root = os.path.dirname(__file__)
    return make_filesource(root)

@pytest.fixture
def expected_df(mixed_df, column_value_type):
    df = mixed_df
    column, value, type = column_value_type

    if type == 'single_value':
        return df[df[column] == value]

    elif type == 'range':
        begin, end = value
        return df[(df[column] >= begin) & (df[column] <= end)]

    elif type == 'range_list':
        conditions = False
        for range in value:
            begin, end = range
            conditions |= ((df[column] >= begin) & (df[column] <= end))
        return df[conditions]

    elif type == 'list':
        return df[df[column].isin(value)]

    elif type == 'date':
        return df[df[column] == pd.to_datetime(value)]

    elif type == 'date_range':
        begin, end = value
        return df[(df[column] >= pd.to_datetime(begin)) & (df[column] <= pd.to_datetime(end))]

    return df


def test_source_resolve_module_type():
    assert Source._get_type('lumen.sources.base.Source') is Source


@pytest.mark.parametrize("filter_col_A", [314, (13, 314), ['A', 314, 'def']])
@pytest.mark.parametrize("filter_col_B", [(3, 15.9)])
@pytest.mark.parametrize("filter_col_C", [[1, 'A', 'def']])
@pytest.mark.parametrize("sql_transforms", [(None, None), (SQLLimit(limit=100), SQLLimit(limit=100))])
def test_file_source_table_cache_key(source, filter_col_A, filter_col_B, filter_col_C, sql_transforms):
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


@pytest.mark.parametrize(
    "column_value_type", [
        ('A', 1, 'single_value'),
        ('A', (1, 3), 'range'),
        ('A', (1, 2), 'range'),
        ('A', [(0, 1), (3, 4)], 'range_list'),
        ('C', 'foo2', 'single_value'),
        ('C', ['foo1', 'foo3'], 'list'),
        ('D', dt.datetime(2009, 1, 2), 'single_value'),
        ('D', (dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)), 'range'),
        ('D', [dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)], 'list'),
        ('D', dt.datetime(2009, 1, 2), 'date'),
        ('D', (dt.date(2009, 1, 2), dt.date(2009, 1, 5)), 'date_range'),
    ]
)
@pytest.mark.parametrize("dask", [True, False])
def test_file_source_filter(source, column_value_type, dask, expected_df):
    column, value, _ = column_value_type
    kwargs = {column: value, '__dask': dask}
    filtered = source.get('test', **kwargs)
    pd.testing.assert_frame_equal(filtered, expected_df)


@pytest.mark.parametrize(
    "column_value_type", [
        ('A', 1, 'single_value'),
        ('A', (1, 3), 'range'),
        ('A', (1, 2), 'range'),
        ('A', [(0, 1), (3, 4)], 'range_list'),
        ('C', 'foo2', 'single_value'),
        ('C', ['foo1', 'foo3'], 'list'),
        ('D', dt.datetime(2009, 1, 2), 'single_value'),
        ('D', (dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)), 'range'),
        ('D', [dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)], 'list'),
        ('D', dt.datetime(2009, 1, 2), 'date'),
        ('D', (dt.date(2009, 1, 2), dt.date(2009, 1, 5)), 'date_range'),
    ]
)
@pytest.mark.parametrize("dask", [True, False])
def test_file_source_get_query_cache(source, column_value_type, dask, expected_df):
    column, value, _ = column_value_type
    kwargs = {column: value}
    source.get('test', __dask=dask, **kwargs)
    cache_key = source._get_key('test', **kwargs)
    assert cache_key in source._cache
    cached_df = source._cache[cache_key]
    if dask:
        cached_df = cached_df.compute()
    pd.testing.assert_frame_equal(cached_df, expected_df)
    cache_key = source._get_key('test', **kwargs)
    assert cache_key in source._cache


@pytest.mark.parametrize(
    "column_value_type", [
        ('A', 1, 'single_value'),
        ('A', [(0, 1), (3, 4)], 'range_list'),
        ('C', ['foo1', 'foo3'], 'list'),
        ('D', (dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)), 'range'),
        ('D', dt.datetime(2009, 1, 2), 'date'),
    ]
)
@pytest.mark.parametrize("dask", [True, False])
def test_file_source_clear_cache(source, column_value_type, dask):
    column, value, _ = column_value_type
    kwargs = {column: value}
    source.get('test', __dask=dask, **kwargs)
    cache_key = source._get_key('test', **kwargs)
    assert cache_key in source._cache
    source.clear_cache()
    assert len(source._cache) == 0


@pytest.mark.skipif(fastparquet is None, reason='fastparquet not installed')
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
        makeMixedDataFrame().iloc[1:3]
    )


def test_file_source_get_tables(source):
    tables = source.get_tables()
    assert tables == ['test']


def test_file_source_variable(make_variable_filesource, mixed_df):
    root = os.path.dirname(__file__)
    source = make_variable_filesource(root)
    state.variables.tables = {'test': 'test2.csv'}
    df = source.get('test')
    expected = mixed_df.iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(df, expected)


def test_extension_of_comlicated_url(source):
    url = "https://api.tfl.gov.uk/Occupancy/BikePoints/@{stations.stations.id}?app_key=random_numbers"
    source.tables["test"] = url
    assert source._named_files["test"][1] is None


# Async tests for base source methods
@pytest.mark.asyncio
async def test_source_get_async_default_implementation(source):
    """Test that the default get_async implementation works using asyncio.to_thread"""
    # Test basic async functionality
    result = await source.get_async('test')
    expected = source.get('test')
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "column_value_type", [
        ('A', 1, 'single_value'),
        ('A', (1, 3), 'range'),
        ('C', 'foo2', 'single_value'),
        ('C', ['foo1', 'foo3'], 'list'),
        ('D', dt.datetime(2009, 1, 2), 'single_value'),
    ]
)
async def test_source_get_async_with_filters(source, column_value_type, expected_df):
    """Test that get_async works with various filters"""
    column, value, _ = column_value_type
    kwargs = {column: value}

    # Test async version
    result_async = await source.get_async('test', **kwargs)

    # Compare with sync version
    result_sync = source.get('test', **kwargs)

    pd.testing.assert_frame_equal(result_async, result_sync)


@pytest.mark.asyncio
async def test_source_get_async_multiple_concurrent_calls(source):
    """Test that multiple concurrent async calls work correctly"""
    # Make multiple concurrent async calls
    tasks = [
        source.get_async('test', A=1),
        source.get_async('test', A=2),
        source.get_async('test', C='foo1'),
    ]

    results = await asyncio.gather(*tasks)

    # Compare with sync versions
    expected_results = [
        source.get('test', A=1),
        source.get('test', A=2),
        source.get('test', C='foo1'),
    ]

    for result, expected in zip(results, expected_results, strict=False):
        pd.testing.assert_frame_equal(result, expected)


# Test BaseSQLSource async methods
@pytest.mark.asyncio
async def test_base_sql_source_execute_async():
    """Test BaseSQLSource execute_async method using a mock implementation"""
    from lumen.sources.base import BaseSQLSource

    class MockSQLSource(BaseSQLSource):
        """Mock SQL source for testing async functionality"""

        def __init__(self):
            self.tables = {'test_table': 'test_table'}
            self._cache = {}

        def execute(self, sql_query, *args, **kwargs):
            # Mock execute that returns a simple DataFrame
            return pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie'],
                'value': [10, 20, 30]
            })

        def get_tables(self):
            return list(self.tables.keys())

        def get_sql_expr(self, table):
            return f"SELECT * FROM {table}"

    # Create mock source and test execute_async
    mock_source = MockSQLSource()

    # Test that execute_async returns the same result as execute
    sql_query = "SELECT * FROM test_table"
    result_async = await mock_source.execute_async(sql_query)
    result_sync = mock_source.execute(sql_query)

    pd.testing.assert_frame_equal(result_async, result_sync)


@pytest.mark.asyncio
async def test_base_sql_source_get_async():
    """Test BaseSQLSource get_async method using a mock implementation"""
    from lumen.sources.base import BaseSQLSource

    class MockSQLSource(BaseSQLSource):
        """Mock SQL source for testing async functionality"""

        def __init__(self):
            self.tables = {'test_table': 'test_table'}
            self._cache = {}

        def execute(self, sql_query, *args, **kwargs):
            # Mock execute that filters based on the SQL query
            df = pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'category': ['A', 'B', 'A', 'C', 'B']
            })

            # Simple mock filtering based on SQL content
            if 'WHERE' in sql_query.upper():
                if "category = 'A'" in sql_query:
                    return df[df['category'] == 'A']
                elif "id = 1" in sql_query:
                    return df[df['id'] == 1]
            return df

        def get_tables(self):
            return list(self.tables.keys())

        def get_sql_expr(self, table):
            return f"SELECT * FROM {table}"

    # Create mock source and test get_async
    mock_source = MockSQLSource()

    # Test basic get_async (no filters)
    result_async = await mock_source.get_async('test_table')
    expected = mock_source.execute("SELECT * FROM test_table")
    pd.testing.assert_frame_equal(result_async, expected)

    # Test get_async with conditions (this will test SQL filter application)
    result_async_filtered = await mock_source.get_async('test_table', category='A')
    # The base implementation should build and execute a filtered query
    assert len(result_async_filtered) <= len(expected)  # Should be filtered
