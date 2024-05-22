import pandas as pd
import pytest

from hvplot.tests.util import makeMixedDataFrame


@pytest.fixture
def source_tables():
    string = pd.get_option('mode.string_storage')
    pd.set_option('mode.string_storage', 'pyarrow')
    df_test = makeMixedDataFrame()
    df_test_sql = makeMixedDataFrame()
    df_test_sql_none = makeMixedDataFrame()
    df_test_sql_none['C'] = ['foo1', None, 'foo3', None, 'foo5']
    tables = {
        'test': df_test,
        'test_sql': df_test_sql,
        'test_sql_with_none': df_test_sql_none,
    }
    yield tables
    pd.set_option('mode.string_storage', string)
