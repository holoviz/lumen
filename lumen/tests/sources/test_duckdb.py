import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest

from lumen.transforms.sql import SQLGroupBy

try:
    from lumen.sources.duckdb import DuckDBSource
except ImportError:
    DuckDBSource = None

pytestmark = pytest.mark.skipif(DuckDBSource is None, reason="Duckdb is not installed")


@pytest.fixture
def duckdb_source():
    root = os.path.dirname(__file__)
    duckdb_source = DuckDBSource(
        initializers=[
            "INSTALL sqlite;",
            "LOAD sqlite;",
            f"SET home_directory='{root}';"
        ],
        root=root,
        sql_expr='SELECT A, B, C, D::TIMESTAMP AS D FROM {table}',
        tables={
            'test_sql': f"sqlite_scan('{root + '/test.db'}', 'mixed')",
            'test_sql_with_none': f"sqlite_scan('{root + '/test.db'}', 'mixed_none')",
        }
    )
    return duckdb_source


def test_duckdb_resolve_module_type():
    assert DuckDBSource._get_type('lumen.sources.duckdb.DuckDBSource') is DuckDBSource
    assert DuckDBSource.source_type == 'duckdb'


def test_duckdb_get_tables(duckdb_source, source_tables):
    tables = duckdb_source.get_tables()
    assert not len(set(tables) - set(source_tables.keys()))
    for table in tables:
        pd.testing.assert_frame_equal(duckdb_source.get(table), source_tables[table])


def test_duckdb_get_schema(duckdb_source):
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
    assert duckdb_source.get_schema('test_sql') == expected_sql
    assert list(duckdb_source._schema_cache.keys()) == ['test_sql']


def test_duckdb_get_schema_with_none(duckdb_source):
    expected_sql = {
        'A': {'inclusiveMaximum': 4.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'B': {'inclusiveMaximum': 1.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'C': {'enum': ['foo1', np.nan, 'foo3', 'foo5'], 'type': 'string'},
        'D': {
            'format': 'datetime',
            'inclusiveMaximum': '2009-01-07 00:00:00',
            'inclusiveMinimum': '2009-01-01 00:00:00',
            'type': 'string'
        }
    }
    assert duckdb_source.get_schema('test_sql_with_none') == expected_sql
    assert list(duckdb_source._schema_cache.keys()) == ['test_sql_with_none']


def test_duckdb_get_schema_cache(duckdb_source):
    duckdb_source.get_schema('test_sql')
    assert 'test_sql' in duckdb_source._schema_cache


@pytest.mark.parametrize(
    "table_column_value_type", [
        ('test_sql', 'A', 1, 'single_value'),
        ('test_sql', 'A', (1, 3), 'range'),
        ('test_sql', 'A', [(0, 1), (3, 4)], 'range_list'),
        ('test_sql', 'C', 'foo2', 'single_value'),
        ('test_sql', 'C', ['foo1', 'foo3'], 'list'),
        ('test_sql', 'D', dt.datetime(2009, 1, 2), 'single_value'),
        ('test_sql', 'D', (dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)), 'range'),
        ('test_sql', 'D', dt.date(2009, 1, 2), 'date'),
        ('test_sql', 'D', (dt.date(2009, 1, 2), dt.date(2009, 1, 5)), 'date_range'),
        ('test_sql_with_none', 'C', None, 'single_value'),
        ('test_sql_with_none', 'C', [None, 'foo5'], 'list'),
    ]
)
@pytest.mark.parametrize("dask", [True, False])
def test_duckdb_filter(duckdb_source, table_column_value_type, dask, expected_filtered_df):
    table, column, value, _ = table_column_value_type
    kwargs = {column: value}
    filtered = duckdb_source.get(table, __dask=dask, **kwargs)
    pd.testing.assert_frame_equal(filtered, expected_filtered_df.reset_index(drop=True))


def test_duckdb_transforms(duckdb_source, source_tables):
    df_test_sql = source_tables['test_sql']
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]
    transformed = duckdb_source.get('test_sql', sql_transforms=transforms)
    expected = df_test_sql.groupby('B')['A'].sum().reset_index()
    pd.testing.assert_frame_equal(transformed, expected)


def test_duckdb_transforms_cache(duckdb_source, source_tables):
    df_test_sql = source_tables['test_sql']
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]

    duckdb_source.get('test_sql', sql_transforms=transforms)
    cache_key = duckdb_source._get_key('test_sql', sql_transforms=transforms)
    assert cache_key in duckdb_source._cache

    expected = df_test_sql.groupby('B')['A'].sum().reset_index()
    pd.testing.assert_frame_equal(duckdb_source._cache[cache_key], expected)

    cache_key = duckdb_source._get_key('test_sql', sql_transforms=transforms)
    assert cache_key in duckdb_source._cache


def test_duckdb_clear_cache(duckdb_source):
    duckdb_source.get('test_sql')
    duckdb_source.get_schema('test_sql')
    assert len(duckdb_source._cache) == 1
    assert len(duckdb_source._schema_cache) == 1
    duckdb_source.clear_cache()
    assert len(duckdb_source._cache) == 0
    assert len(duckdb_source._schema_cache) == 0
