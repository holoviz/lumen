import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest

from lumen.sources import JoinedSource
from lumen.sources.intake_sql import IntakeSQLSource

from .utils import source_get_tables


@pytest.fixture
def file_source(make_filesource):
    root = os.path.dirname(__file__)
    return make_filesource(root)


@pytest.fixture
def intake_sql_source():
    root = os.path.dirname(__file__)
    intake_sql_source = IntakeSQLSource(
        uri=os.path.join(root, 'catalog_intake_sql.yml'), root=root
    )
    return intake_sql_source


@pytest.fixture
def source(file_source, intake_sql_source, index):
    sources = {
        'file_source': file_source,
        'intake_sql_source': intake_sql_source,
    }
    tables = {
        'joined_table': [
            {'source': 'file_source',
             'table': 'test',
            },
            {'source': 'intake_sql_source',
             'table': 'test_sql_with_none',
            }
        ]
    }
    if index is not None:
        tables['joined_table'][0]['index'] = index
        tables['joined_table'][1]['index'] = index

    intake_sql_source = JoinedSource(sources=sources, tables=tables)
    return intake_sql_source


@pytest.fixture
def source_tables_default_index():
    format = '%Y-%m-%dT%H:%M:%S'
    table = pd.DataFrame({
        'A': [0., 1., 2., 3., 4., 1., 3.],
        'B': [0., 1., 0., 1., 0., 1., 1.],
        'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5', None, None],
        'D': [
            dt.datetime.strptime('2009-01-01T00:00:00', format),
            dt.datetime.strptime('2009-01-02T00:00:00', format),
            dt.datetime.strptime('2009-01-05T00:00:00', format),
            dt.datetime.strptime('2009-01-06T00:00:00', format),
            dt.datetime.strptime('2009-01-07T00:00:00', format),
            dt.datetime.strptime('2009-01-02T00:00:00', format),
            dt.datetime.strptime('2009-01-06T00:00:00', format),
        ],
    })
    return {'joined_table': table}


@pytest.fixture
def source_tables_index_A():
    format = '%Y-%m-%dT%H:%M:%S'
    table = pd.DataFrame({
        'A': [0., 1., 2., 3., 4.],
        'B_x': [0., 1., 0., 1., 0.],
        'C_x': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'],
        'D_x': [
            dt.datetime.strptime('2009-01-01T00:00:00', format),
            dt.datetime.strptime('2009-01-02T00:00:00', format),
            dt.datetime.strptime('2009-01-05T00:00:00', format),
            dt.datetime.strptime('2009-01-06T00:00:00', format),
            dt.datetime.strptime('2009-01-07T00:00:00', format),
        ],
        'B_y': [0., 1., 0., 1., 0.],
        'C_y': ['foo1', None, 'foo3', None, 'foo5'],
        'D_y': [
            dt.datetime.strptime('2009-01-01T00:00:00', format),
            dt.datetime.strptime('2009-01-02T00:00:00', format),
            dt.datetime.strptime('2009-01-05T00:00:00', format),
            dt.datetime.strptime('2009-01-06T00:00:00', format),
            dt.datetime.strptime('2009-01-07T00:00:00', format),
        ],
    })
    return {'joined_table': table}


@pytest.fixture
def source_tables_index_C():
    format = '%Y-%m-%dT%H:%M:%S'
    table = pd.DataFrame({
        'A_x': [0., 1., 2., 3., 4.],
        'B_x': [0., 1., 0., 1., 0.],
        'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'],
        'D_x': [
            dt.datetime.strptime('2009-01-01T00:00:00', format),
            dt.datetime.strptime('2009-01-02T00:00:00', format),
            dt.datetime.strptime('2009-01-05T00:00:00', format),
            dt.datetime.strptime('2009-01-06T00:00:00', format),
            dt.datetime.strptime('2009-01-07T00:00:00', format),
        ],
        'A_y': [0., np.nan, 2., np.nan, 4.],
        'B_y': [0., np.nan, 0., np.nan, 0.],
        'D_y': [
            dt.datetime.strptime('2009-01-01T00:00:00', format),
            pd.NaT,
            dt.datetime.strptime('2009-01-05T00:00:00', format),
            pd.NaT,
            dt.datetime.strptime('2009-01-07T00:00:00', format),
        ],
    })
    return {'joined_table': table}


@pytest.fixture
def source_tables(
        index,
        source_tables_default_index,
        source_tables_index_A,
        source_tables_index_C,
):
    tables_dict = {
        None: source_tables_default_index,
        'A': source_tables_index_A,
        'C': source_tables_index_C,
    }
    return tables_dict[index]


def test_source_resolve_module_type():
    assert JoinedSource._get_type('lumen.sources.JoinedSource') is JoinedSource
    assert JoinedSource.source_type == 'join'


@pytest.mark.parametrize("index", [None, 'A', 'C'])
def test_source(source, source_tables, index):
    assert source_get_tables(source, source_tables)
