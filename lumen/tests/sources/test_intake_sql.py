import datetime as dt
import os

import pandas as pd
import pytest

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
def source_tables():
    df_test = pd._testing.makeMixedDataFrame()
    df_test_sql = pd._testing.makeMixedDataFrame()
    df_test_sql_none = pd._testing.makeMixedDataFrame()
    df_test_sql_none['C'] = ['foo1', None, 'foo3', None, 'foo5']
    tables = {
        'test': df_test,
        'test_sql': df_test_sql,
        'test_sql_with_none': df_test_sql_none,
    }
    return tables


def test_intake_sql_resolve_module_type():
    assert IntakeSQLSource._get_type('lumen.sources.intake_sql.IntakeSQLSource') is IntakeSQLSource
    assert IntakeSQLSource.source_type == 'intake_sql'


def test_intake_sql_get_tables(source, source_tables):
    tables = source.get_tables()
    assert tables == list(source_tables.keys())
    for table in tables:
        pd.testing.assert_frame_equal(source.get(table), source_tables[table])


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
    assert source.get_schema('test_sql') == expected_sql
    assert list(source._schema_cache.keys()) == ['test_sql']
    assert source.get_schema('test') == expected_csv
    assert list(source._schema_cache.keys()) == ['test_sql', 'test']


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
    assert list(source._schema_cache.keys()) == ['test_sql_with_none']


def test_intake_sql_get_schema_cache(source):
    source.get_schema('test_sql')
    assert 'test_sql' in source._schema_cache


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
    table, column, value, _ = table_column_value_type
    kwargs = {column: value}
    filtered = source.get(table, __dask=dask, **kwargs)
    pd.testing.assert_frame_equal(filtered, expected_filtered_df.reset_index(drop=True))


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


def test_intake_sql_clear_cache(source):
    source.get('test_sql')
    source.get_schema('test_sql')
    assert len(source._cache) == 1
    assert len(source._schema_cache) == 1
    source.clear_cache()
    assert len(source._cache) == 0
    assert len(source._schema_cache) == 0
