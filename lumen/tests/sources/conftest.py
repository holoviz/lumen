import pandas as pd
import pytest


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
