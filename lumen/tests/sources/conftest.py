import tempfile

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hvplot.tests.util import makeMixedDataFrame

from lumen.sources.duckdb import DuckDBSource


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


@pytest.fixture
def duckdb_file_source():
    """
    Create a temporary DuckDB database file with a test table.

    Yields the DuckDBSource and cleans up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test.duckdb'
        conn = duckdb.connect(str(db_path))
        conn.execute('CREATE TABLE test (id INTEGER, name VARCHAR)')
        conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')")
        conn.close()

        source = DuckDBSource(
            uri=str(db_path),
            tables=['test'],
        )
        yield source
        source.close()
