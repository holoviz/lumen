import os

import pandas as pd
import pytest

from lumen.sources import JSONSource

from .utils import source_filter, source_get_cache_no_query


@pytest.fixture
def source(make_jsonsource):
    root = os.path.dirname(__file__)
    return make_jsonsource(root)


@pytest.fixture
def source_tables():
    df = pd._testing.makeMixedDataFrame()
    df['D'] = pd.to_datetime(df['D']).dt.date
    df = df.astype({'A': int, 'B': int, "D": 'str'})
    return {'local': df}


def test_json_source_resolve_module_type():
    assert JSONSource._get_type('lumen.sources.JSONSource') is JSONSource
    assert JSONSource.source_type == 'json'


@pytest.mark.parametrize(
    "table_column_value_type", [
        ('local', 'A', 1.0, 'single_value'),
        ('local', 'A', (1, 3), 'range'),
        ('local', 'A', [(0, 1), (3, 4)], 'range_list'),
        ('local', 'C', 'foo2', 'single_value'),
        ('local', 'C', ['foo1', 'foo3'], 'list'),
    ]
)
@pytest.mark.parametrize("dask", [True, False])
def test_json_source_filter(source, table_column_value_type, dask, expected_filtered_df):
    assert source_filter(source, table_column_value_type, dask, expected_filtered_df)


@pytest.mark.parametrize("dask", [True, False])
def test_file_source_get_cache_no_query(source, dask, source_tables):
    assert source_get_cache_no_query(
        source, 'local', source_tables['local'], dask, use_dask=False
    )
