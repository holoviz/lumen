import datetime as dt
import os

import pandas as pd
import pytest
import yaml

from lumen.sources.intake import IntakeSource


@pytest.fixture
def source():
    root = os.path.dirname(__file__)
    return IntakeSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )


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


def test_intake_resolve_module_type():
    assert IntakeSource._get_type('lumen.sources.intake_sql.IntakeSource') is IntakeSource
    assert IntakeSource.source_type == 'intake'


def test_intake_source_from_file(source):
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(source.get('test'), df)


def test_intake_source_from_dict():
    root = os.path.dirname(__file__)
    with open(os.path.join(root, 'catalog.yml')) as f:
        catalog = yaml.load(f, Loader=yaml.Loader)
    source = IntakeSource(catalog=catalog, root=root)
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(source.get('test'), df)


@pytest.mark.parametrize(
    "table_column_value_type", [
        ('test', 'A', 1, 'single_value'),
        ('test', 'A', (1, 3), 'range'),
        ('test', 'A', [(0, 1), (3, 4)], 'range_list'),
        ('test', 'C', 'foo2', 'single_value'),
        ('test', 'C', ['foo1', 'foo3'], 'list'),
        ('test', 'D', dt.datetime(2009, 1, 2), 'single_value'),
        ('test', 'D', (dt.datetime(2009, 1, 2), dt.datetime(2009, 1, 5)), 'range'),
        ('test', 'D', dt.date(2009, 1, 2), 'date'),
        ('test', 'D', (dt.date(2009, 1, 2), dt.date(2009, 1, 5)), 'date_range'),
    ]
)
@pytest.mark.parametrize("dask", [True, False])
def test_intake_filter(source, table_column_value_type, dask, expected_filtered_df):
    table, column, value, _ = table_column_value_type
    kwargs = {column: value}
    filtered = source.get(table, __dask=dask, **kwargs)
    pd.testing.assert_frame_equal(filtered, expected_filtered_df)
