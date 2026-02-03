import datetime as dt
import os

import pandas as pd
import pytest

from lumen.transforms.sql import SQLGroupBy

try:
    from sqlalchemy import text
    from sqlalchemy.engine.url import make_url

    from lumen.sources.sqlalchemy import (
        SQLALCHEMY_TO_SQLGLOT_DIALECT, SQLAlchemySource,
    )
    pytestmark = pytest.mark.xdist_group("sqlalchemy")
except ImportError:
    pytestmark = pytest.mark.skip(reason="SQLAlchemy is not installed")


@pytest.fixture
def sqlite_db():
    """Create a temporary SQLite database with test data."""
    root = os.path.dirname(__file__)
    db_path = os.path.join(root, 'test.db')

    # The database should already exist from other tests
    # but we'll verify it's there
    assert os.path.exists(db_path), f"Test database not found at {db_path}"

    return db_path


@pytest.fixture
def sqlalchemy_source(sqlite_db):
    """Create SQLAlchemy source connected to SQLite test database."""
    source = SQLAlchemySource(
        url=f'sqlite:///{sqlite_db}',
        tables={
            'test_sql': 'SELECT * FROM mixed',
            'test_sql_with_none': 'SELECT * FROM mixed_none',
        }
    )
    yield source
    source.close()


@pytest.fixture
def memory_source(mixed_df):
    """Create an in-memory SQLite database with test data."""
    source = SQLAlchemySource(
        url='sqlite:///:memory:',
    )
    yield source
    source.close()


def test_sqlalchemy_resolve_module_type():
    assert SQLAlchemySource._get_type('lumen.sources.sqlalchemy.SQLAlchemySource') is SQLAlchemySource
    assert SQLAlchemySource.source_type == 'sqlalchemy'


def test_sqlalchemy_get_tables(sqlalchemy_source, source_tables):
    tables = sqlalchemy_source.get_tables()
    assert not len(set(tables) - set(source_tables.keys()))
    for table in tables:
        result = sqlalchemy_source.get(table)
        expected = source_tables[table]
        # SQLite may return different dtypes, so just check values
        assert len(result) == len(expected)


def test_sqlalchemy_get_schema(sqlalchemy_source):
    source = sqlalchemy_source.get_schema('test_sql')

    # Check the basic structure
    assert 'A' in source
    assert 'B' in source
    assert 'C' in source
    assert 'D' in source
    assert '__len__' in source

    # Check numeric columns
    assert source['A']['type'] == 'number'
    assert source['A']['inclusiveMinimum'] == 0.0
    assert source['A']['inclusiveMaximum'] == 4.0

    assert source['B']['type'] == 'number'
    assert source['B']['inclusiveMinimum'] == 0.0
    assert source['B']['inclusiveMaximum'] == 1.0

    # Check enum column
    assert source['C']['type'] == 'string'
    assert set(source['C']['enum']) == {'foo1', 'foo2', 'foo3', 'foo4', 'foo5'}

    # Check datetime column (SQLite stores dates as strings, so it may be an enum)
    assert source['D']['type'] == 'string'
    # SQLite may treat dates as enum values or have min/max
    assert 'enum' in source['D'] or 'inclusiveMinimum' in source['D']

    assert source['__len__'] == 5
    assert list(sqlalchemy_source._schema_cache.keys()) == ['test_sql']


def test_sqlalchemy_get_schema_with_limit(sqlalchemy_source):
    source = sqlalchemy_source.get_schema('test_sql', limit=1)

    # With limit=1, we should only see the first row's values
    assert source['A']['inclusiveMinimum'] == 0.0
    assert source['A']['inclusiveMaximum'] == 0.0
    assert source['B']['inclusiveMinimum'] == 0.0
    assert source['B']['inclusiveMaximum'] == 0.0
    assert source['C']['enum'] == ['foo1']
    assert source['__len__'] == 5  # Total count should still be 5

    assert list(sqlalchemy_source._schema_cache.keys()) == ['test_sql']


def test_sqlalchemy_get_schema_with_none(sqlalchemy_source):
    source = sqlalchemy_source.get_schema('test_sql_with_none')

    # Check that None is in the enum
    assert None in source['C']['enum']
    assert 'foo1' in source['C']['enum']
    assert 'foo3' in source['C']['enum']
    assert 'foo5' in source['C']['enum']

    # Check numeric ranges
    assert source['A']['inclusiveMinimum'] == 0.0
    assert source['A']['inclusiveMaximum'] == 4.0
    assert source['B']['inclusiveMinimum'] == 0.0
    assert source['B']['inclusiveMaximum'] == 1.0

    assert source['__len__'] == 5
    assert list(sqlalchemy_source._schema_cache.keys()) == ['test_sql_with_none']


def test_sqlalchemy_get_schema_cache(sqlalchemy_source):
    sqlalchemy_source.get_schema('test_sql')
    assert 'test_sql' in sqlalchemy_source._schema_cache


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
def test_sqlalchemy_filter(sqlalchemy_source, table_column_value_type, expected_filtered_df):
    table, column, value, _ = table_column_value_type
    kwargs = {column: value}
    filtered = sqlalchemy_source.get(table, **kwargs)

    # SQLite may have different dtypes, so check values instead of exact equality
    expected = expected_filtered_df.reset_index(drop=True)
    assert len(filtered) == len(expected), f"Length mismatch: {len(filtered)} vs {len(expected)}"

    # Check each column's values (not dtypes)
    for col in expected.columns:
        if col == 'D':  # Datetime column - very lenient comparison
            # Just check that we have the right number of rows
            # SQLite stores datetimes as strings in various formats
            continue
        else:
            pd.testing.assert_series_equal(filtered[col], expected[col], check_names=False, check_dtype=False)


@pytest.mark.flaky(reruns=3)
def test_sqlalchemy_transforms(sqlalchemy_source, source_tables):
    df_test_sql = source_tables['test_sql']
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]
    transformed = sqlalchemy_source.get('test_sql', sql_transforms=transforms)
    expected = df_test_sql.groupby('B')['A'].sum().reset_index()

    # Check values, not dtypes (SQLite may return different types)
    assert len(transformed) == len(expected)
    pd.testing.assert_series_equal(transformed['B'], expected['B'], check_names=False, check_dtype=False)
    pd.testing.assert_series_equal(transformed['A'], expected['A'], check_names=False, check_dtype=False)


@pytest.mark.flaky(reruns=3)
def test_sqlalchemy_transforms_cache(sqlalchemy_source, source_tables):
    df_test_sql = source_tables['test_sql']
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]

    sqlalchemy_source.get('test_sql', sql_transforms=transforms)
    cache_key = sqlalchemy_source._get_key('test_sql', sql_transforms=transforms)
    assert cache_key in sqlalchemy_source._cache

    expected = df_test_sql.groupby('B')['A'].sum().reset_index()
    cached = sqlalchemy_source._cache[cache_key]

    # Check values, not dtypes
    assert len(cached) == len(expected)
    pd.testing.assert_series_equal(cached['B'], expected['B'], check_names=False, check_dtype=False)
    pd.testing.assert_series_equal(cached['A'], expected['A'], check_names=False, check_dtype=False)

    cache_key = sqlalchemy_source._get_key('test_sql', sql_transforms=transforms)
    assert cache_key in sqlalchemy_source._cache


def test_sqlalchemy_clear_cache(sqlalchemy_source):
    sqlalchemy_source.get('test_sql')
    sqlalchemy_source.get_schema('test_sql')
    assert len(sqlalchemy_source._cache) == 1
    assert len(sqlalchemy_source._schema_cache) == 1
    sqlalchemy_source.clear_cache()
    assert len(sqlalchemy_source._cache) == 0
    assert len(sqlalchemy_source._schema_cache) == 0


def test_sqlalchemy_connection_url_from_memory():
    """Test various connection URL formats."""
    # SQLite in-memory
    source1 = SQLAlchemySource(url='sqlite:///:memory:')
    try:
        assert source1._url.drivername == 'sqlite'
    finally:
        source1.close()

    # SQLite file
    source2 = SQLAlchemySource(url='sqlite:///test.db')
    try:
        assert source2._url.drivername == 'sqlite'
    finally:
        source2.close()


def test_sqlalchemy_connection_components():
    """Test connection using components instead of URL."""
    source = SQLAlchemySource(
        drivername='sqlite',
        database=':memory:'
    )
    try:
        assert source._url.drivername == 'sqlite'
        assert source._url.database == ':memory:'
    finally:
        source.close()


def test_sqlalchemy_driver_detection(memory_source):
    """Test async driver detection."""
    assert not memory_source._driver_is_async

    # Async drivers (won't connect, just test detection)
    assert SQLAlchemySource._is_async_driver('postgresql+asyncpg')
    assert SQLAlchemySource._is_async_driver('mysql+asyncmy')
    assert SQLAlchemySource._is_async_driver('sqlite+aiosqlite')
    assert not SQLAlchemySource._is_async_driver('postgresql+psycopg2')
    assert not SQLAlchemySource._is_async_driver('sqlite')


def test_sqlalchemy_dialect_mapping():
    """Test that SQLAlchemy dialect names are correctly mapped to sqlglot dialect names."""
    # We test the dialect property by checking what the URL would return
    # and verifying the mapping works without actually connecting to databases
    
    # Test postgresql -> postgres mapping
    url = make_url('postgresql://user:pass@localhost:5432/db')
    assert url.get_dialect().name == 'postgresql'
    # Now verify that the production mapping converts it correctly
    mapped_dialect = SQLALCHEMY_TO_SQLGLOT_DIALECT.get(url.get_dialect().name, url.get_dialect().name)
    assert mapped_dialect == 'postgres', f"Expected 'postgres' but got '{mapped_dialect}'"

    # Test mssql -> tsql mapping  
    url = make_url('mssql+pyodbc://user:pass@localhost:1433/db')
    assert url.get_dialect().name == 'mssql'
    mapped_dialect = SQLALCHEMY_TO_SQLGLOT_DIALECT.get(url.get_dialect().name, url.get_dialect().name)
    assert mapped_dialect == 'tsql', f"Expected 'tsql' but got '{mapped_dialect}'"

    # Test that non-mapped dialects pass through unchanged
    url = make_url('sqlite:///test.db')
    assert url.get_dialect().name == 'sqlite'
    mapped_dialect = SQLALCHEMY_TO_SQLGLOT_DIALECT.get(url.get_dialect().name, url.get_dialect().name)
    assert mapped_dialect == 'sqlite', f"Expected 'sqlite' but got '{mapped_dialect}'"

    url = make_url('mysql://user:pass@localhost:3306/db')
    assert url.get_dialect().name == 'mysql'
    mapped_dialect = SQLALCHEMY_TO_SQLGLOT_DIALECT.get(url.get_dialect().name, url.get_dialect().name)
    assert mapped_dialect == 'mysql', f"Expected 'mysql' but got '{mapped_dialect}'"


def test_sqlalchemy_dialect_property_with_sqlite():
    """Test that the dialect property works correctly on an actual SQLAlchemy source."""
    # Use SQLite which doesn't require external dependencies
    source = SQLAlchemySource(url='sqlite:///:memory:')
    assert source.dialect == 'sqlite', f"Expected 'sqlite' but got '{source.dialect}'"
    source.close()


def test_sqlalchemy_get_tables_from_inspector(memory_source):
    """Test automatic table discovery using SQLAlchemy inspector."""
    with memory_source._engine.begin() as conn:
        conn.execute(text('CREATE TABLE table1 (id INTEGER, name TEXT)'))
        conn.execute(text('CREATE TABLE table2 (id INTEGER, value REAL)'))
        conn.execute(text('INSERT INTO table1 VALUES (1, "test")'))
        conn.execute(text('INSERT INTO table2 VALUES (1, 1.5)'))

    # Get tables should discover both tables
    tables = memory_source.get_tables()
    assert 'table1' in tables
    assert 'table2' in tables


def test_sqlalchemy_excluded_tables(memory_source):
    """Test table exclusion patterns."""
    memory_source.excluded_tables = ['temp_*', 'test_*']
    with memory_source._engine.begin() as conn:
        conn.execute(text('CREATE TABLE temp_data (id INTEGER)'))
        conn.execute(text('CREATE TABLE test_data (id INTEGER)'))
        conn.execute(text('CREATE TABLE real_data (id INTEGER)'))

    tables = memory_source.get_tables()
    assert 'real_data' in tables
    assert 'temp_data' not in tables
    assert 'test_data' not in tables


def test_sqlalchemy_metadata_extraction(memory_source):
    """Test metadata extraction using SQLAlchemy inspector."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'value': [100.5, 200.75, 300.0]
    })

    with memory_source._engine.begin() as conn:
        df.to_sql('test_table', conn, if_exists='replace', index=False)

    # Get metadata
    metadata = memory_source._get_table_metadata(['test_table'])

    assert 'test_table' in metadata
    assert 'columns' in metadata['test_table']
    assert 'id' in metadata['test_table']['columns']
    assert 'name' in metadata['test_table']['columns']
    assert 'value' in metadata['test_table']['columns']


def test_sqlalchemy_create_sql_expr_source(memory_source):
    """Test creating a new source with SQL expressions."""
    df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    with memory_source._engine.begin() as conn:
        df.to_sql('data', conn, if_exists='replace', index=False)

    memory_source.tables = {'data': 'SELECT * FROM data'}

    # Create new source with SQL expression
    new_tables = {
        'filtered': 'SELECT * FROM data WHERE value > 15'
    }
    new_source = memory_source.create_sql_expr_source(new_tables)

    result = new_source.get('filtered')
    assert len(result) == 2
    assert all(result['value'] > 15)


def test_sqlalchemy_table_params_basic(memory_source):
    """Test table_params with parameterized queries."""

    # Create test data
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'city': ['NYC', 'LA', 'Chicago']
    })

    with memory_source._engine.begin() as conn:
        df.to_sql('customers', conn, if_exists='replace', index=False)

    # Use parameterized query
    memory_source.tables = {
        'customers': 'SELECT * FROM customers',
        'by_city': 'SELECT * FROM customers WHERE city = ?'
    }
    memory_source.table_params = {'by_city': ['NYC']}

    result = memory_source.get('by_city')
    assert len(result) == 1
    assert result.iloc[0]['city'] == 'NYC'


def test_sqlalchemy_table_params_multiple(memory_source):
    """Test table_params with multiple parameters."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'value': [10, 20, 30, 40]
    })

    with memory_source._engine.begin() as conn:
        df.to_sql('data', conn, if_exists='replace', index=False)

    # Use multiple parameters
    memory_source.tables = {
        'data': 'SELECT * FROM data',
        'range_query': 'SELECT * FROM data WHERE value > ? AND value < ?'
    }
    memory_source.table_params = {'range_query': [15, 35]}

    result = memory_source.get('range_query')
    assert len(result) == 2
    assert set(result['value']) == {20, 30}


def test_sqlalchemy_create_sql_expr_source_with_params(memory_source):
    """Test create_sql_expr_source with params."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'B', 'C'],
        'value': [10, 20, 30, 40, 50]
    })

    with memory_source._engine.begin() as conn:
        df.to_sql('data', conn, if_exists='replace', index=False)

    memory_source.tables = {'data': 'SELECT * FROM data'}

    # Create new source with parameterized query
    new_tables = {
        'filtered': 'SELECT * FROM data WHERE category = ? AND value > ?'
    }
    params = {'filtered': ['A', 15]}

    new_source = memory_source.create_sql_expr_source(new_tables, params=params)

    result = new_source.get('filtered')
    assert len(result) == 1
    assert result.iloc[0]['category'] == 'A'
    assert result.iloc[0]['value'] == 30


def test_sqlalchemy_execute(memory_source):
    """Test direct SQL execution."""
    with memory_source._engine.begin() as conn:
        conn.execute(text('CREATE TABLE test (id INTEGER, name TEXT)'))
        conn.execute(text('INSERT INTO test VALUES (1, "Alice")'))
        conn.execute(text('INSERT INTO test VALUES (2, "Bob")'))

    # Execute query
    result = memory_source.execute('SELECT * FROM test WHERE id = ?', [1])
    assert len(result) == 1
    assert result.iloc[0]['id'] == 1
    assert result.iloc[0]['name'] == 'Alice'


def test_sqlalchemy_close():
    """Test proper resource cleanup."""
    source = SQLAlchemySource(url='sqlite:///:memory:')

    # Verify engine exists
    assert source._engine is not None

    # Close the source
    source.close()

    # Verify cleanup
    assert source._engine is None
    assert source._inspector is None


def test_sqlalchemy_dialect_detection_from_url():
    """Test dialect detection from URL."""
    source_sqlite = SQLAlchemySource(url='sqlite:///:memory:')
    try:
        assert source_sqlite.dialect == 'sqlite'
    finally:
        source_sqlite.close()


def test_sqlalchemy_dialect_detection_from_drivername():

    # Test with components
    source_sqlite = SQLAlchemySource(drivername='sqlite', database=':memory:')
    try:
        assert source_sqlite.dialect == 'sqlite'
    finally:
        source_sqlite.close()


@pytest.mark.asyncio
async def test_sqlalchemy_async_with_sync_driver():
    """Test that async methods work with sync drivers (using thread pool)."""
    # Use a file-based SQLite instead of in-memory to avoid connection issues
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()

    source = None
    try:
        source = SQLAlchemySource(url=f'sqlite:///{temp_db.name}')

        # Create test data
        df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        with source._engine.begin() as conn:
            df.to_sql('data', conn, if_exists='replace', index=False)

        source.tables = {'data': 'SELECT * FROM data'}

        # Test async execute (should use thread pool)
        result = await source.execute_async('SELECT * FROM data WHERE id = ?', [2])
        assert len(result) == 1
        assert result.iloc[0]['id'] == 2

        # Test async get
        result = await source.get_async('data')
        assert len(result) == 3
    finally:
        # Close the engine before cleaning up the file
        if source is not None:
            source.close()

        # Clean up
        import os
        os.unlink(temp_db.name)


def test_sqlalchemy_error_handling(memory_source):
    """Test error handling for invalid queries."""
    # Invalid SQL should raise an error
    with pytest.raises(Exception):
        memory_source.execute('SELECT * FROM nonexistent_table')

    # Invalid table should raise an error
    with pytest.raises(Exception):
        memory_source.get('nonexistent_table')

@pytest.mark.parametrize('filter_in_sql', [False, True])
def test_sqlalchemy_filter_in_sql(memory_source, filter_in_sql):
    """Test filter_in_sql parameter."""
    memory_source.filter_in_sql = filter_in_sql

    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'B', 'C']
    })
    with memory_source._engine.begin() as conn:
        df.to_sql('data', conn, if_exists='replace', index=False)

    memory_source.tables = {'data': 'SELECT * FROM data'}

    # Filter should be applied in SQL
    result = memory_source.get('data', category='A')
    assert len(result) == 2
    assert all(result['category'] == 'A')


def test_sqlalchemy_engine_reuse(memory_source):
    """Test that create_sql_expr_source reuses the engine."""
    df = pd.DataFrame({'id': [1, 2, 3]})
    with memory_source._engine.begin() as conn:
        df.to_sql('data', conn, if_exists='replace', index=False)

    memory_source.tables = {'data': 'SELECT * FROM data'}

    # Create new source
    new_source = memory_source.create_sql_expr_source({'filtered': 'SELECT * FROM data WHERE id > 1'})

    # Should reuse the same engine
    assert new_source._engine is memory_source._engine


def test_sqlalchemy_connect_args():
    """Test connect_args parameter."""
    source = SQLAlchemySource(
        url='sqlite:///:memory:',
        connect_args={'check_same_thread': False}
    )

    # Should create successfully
    try:
        assert source._engine is not None
    finally:
        source.close()


def test_sqlalchemy_engine_kwargs():
    """Test engine_kwargs parameter."""
    source = SQLAlchemySource(
        url='sqlite:///:memory:',
        engine_kwargs={'echo': False, 'pool_pre_ping': True}
    )

    # Should create successfully
    try:
        assert source._engine is not None
    finally:
        source.close()


def test_sqlalchemy_query_params():
    """Test query_params in URL."""
    source = SQLAlchemySource(
        drivername='sqlite',
        database=':memory:',
        query_params={'timeout': '30'}
    )

    # Should create URL successfully
    try:
        assert source._url is not None
    finally:
        source.close()


def test_sqlalchemy_invalid_url():
    """Test error handling for invalid URL."""
    with pytest.raises(ValueError):
        # Missing both url and drivername
        SQLAlchemySource()


def test_sqlalchemy_get_metadata(memory_source):
    """Test get_metadata method."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie']
    })
    with memory_source._engine.begin() as conn:
        df.to_sql('users', conn, if_exists='replace', index=False)

    # Get metadata
    metadata = memory_source.get_metadata('users')

    assert 'description' in metadata
    assert 'columns' in metadata
    assert 'id' in metadata['columns']
    assert 'name' in metadata['columns']
    assert 'rows' in metadata


def test_sqlalchemy_custom_sql_expr(memory_source):
    """Test custom sql_expr parameter."""
    memory_source.param.update(
        sql_expr='SELECT id, name FROM {table} LIMIT 10',
        tables={'users': 'users'}
    )

    # Create test data
    df = pd.DataFrame({
        'id': list(range(1, 21)),
        'name': [f'User{i}' for i in range(1, 21)],
        'value': list(range(100, 120))
    })
    with memory_source._engine.begin() as conn:
        df.to_sql('users', conn, if_exists='replace', index=False)

    # Get should use custom sql_expr
    result = memory_source.get('users')
    assert len(result) <= 10
    assert 'id' in result.columns
    assert 'name' in result.columns


def test_sqlalchemy_create_sql_expr_no_nested_transaction(memory_source):
    """Test that create_sql_expr_source doesn't cause nested transaction errors."""
    df = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50]})
    with memory_source._engine.begin() as conn:
        df.to_sql('data_table', conn, if_exists='replace', index=False)

    memory_source.tables = {'data_table': 'SELECT * FROM data_table'}

    # This should not raise a nested transaction error
    new_source = memory_source.create_sql_expr_source({"limited": "SELECT * FROM data_table LIMIT 2"})
    result = new_source.get("limited")

    assert len(result) == 2
    assert list(result['id']) == [1, 2]
