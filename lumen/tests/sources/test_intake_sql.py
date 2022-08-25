import datetime as dt
import os

import pandas as pd
import pytest

from lumen.sources.intake_sql import IntakeSQLSource
from lumen.transforms.sql import SQLGroupBy

from .utils import (
    source_filter, source_get_schema_not_update_cache, source_get_tables,
    source_table_cache_key,
)


@pytest.fixture
def source():
    root = os.path.dirname(__file__)
    intake_sql_source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )
    return intake_sql_source


@pytest.fixture
def source_tables():
    df_test_sql = pd._testing.makeMixedDataFrame()
    df_test_sql_none = pd._testing.makeMixedDataFrame()
    df_test_sql_none['C'] = ['foo1', None, 'foo3', None, 'foo5']
    tables = {
        'test_sql': df_test_sql,
        'test_sql_with_none': df_test_sql_none,
    }
    return tables


@pytest.fixture
def source_schemas():
    test_sql_schema = {
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
    test_sql_with_none_schema = {
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
    schemas = {
        'test_sql': test_sql_schema,
        'test_sql_with_none': test_sql_with_none_schema,
    }
    return schemas


def test_intake_sql_resolve_module_type():
    assert IntakeSQLSource._get_type('lumen.sources.intake_sql.IntakeSQLSource') is IntakeSQLSource
    assert IntakeSQLSource.source_type == 'intake_sql'


def test_intake_sql_get_tables(source, source_tables):
    assert source_get_tables(source, source_tables)


def test_intake_sql_get_schema(source, source_schemas):
    for table in source_schemas:
        assert source.get_schema(table) == source_schemas[table]


def test_intake_sql_cache_key(source, source_tables):
    for table in source_tables:
        assert source_table_cache_key(source, table)


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
def test_intake_sql_filter(source, table_column_value_type, dask, expected_filtered_df):
    assert source_filter(
        source, table_column_value_type, dask, expected_filtered_df.reset_index(drop=True)
    )


def test_intake_sql_transforms(source, source_tables):
    df_test_sql = source_tables['test_sql']
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]
    transformed = source.get('test_sql', sql_transforms=transforms)
    expected = df_test_sql.groupby('B')['A'].sum().reset_index()
    pd.testing.assert_frame_equal(transformed, expected)


def test_intake_sql_transforms_cache(source, source_tables):
    df_test_sql = source_tables['test_sql']
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]

    source.get('test_sql', sql_transforms=transforms)
    cache_key = source._get_key('test_sql', sql_transforms=transforms)
    assert cache_key in source._cache

    expected = df_test_sql.groupby('B')['A'].sum().reset_index()
    pd.testing.assert_frame_equal(source._cache[cache_key], expected)

    cache_key = source._get_key('test_sql', sql_transforms=transforms)
    assert cache_key in source._cache


def test_intake_sql_get_schema_not_update_cache(source):
    assert source_get_schema_not_update_cache(source, table='test_sql')
