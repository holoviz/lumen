import datetime as dt
import os

import pandas as pd
import pytest
import yaml

from lumen.sources.intake import IntakeSource

from .utils import source_filter, source_get_tables


@pytest.fixture
def source():
    root = os.path.dirname(__file__)
    return IntakeSource(
        uri=os.path.join(root, 'catalog_intake.yml'), root=root
    )


@pytest.fixture
def source_tables():
    return {'test': pd._testing.makeMixedDataFrame()}


@pytest.fixture
def source_schemas():
    schemas = {
        'test': {
            'A': {'inclusiveMaximum': 4.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
            'B': {'inclusiveMaximum': 1.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
            'C': {'enum': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], 'type': 'string'},
            'D': {
                'format': 'datetime',
                'inclusiveMaximum': '2009-01-07T00:00:00',
                'inclusiveMinimum': '2009-01-01T00:00:00',
                'type': 'string'
            }
        }
    }
    return schemas


def test_intake_resolve_module_type():
    assert IntakeSource._get_type('lumen.sources.intake_sql.IntakeSource') is IntakeSource
    assert IntakeSource.source_type == 'intake'


def test_intake_source_from_file(source, source_tables):
    assert source_get_tables(source, source_tables)


def test_intake_source_from_dict():
    root = os.path.dirname(__file__)
    with open(os.path.join(root, 'catalog_intake.yml')) as f:
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
    assert source_filter(source, table_column_value_type, dask, expected_filtered_df)
