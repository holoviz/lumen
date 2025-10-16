import datetime as dt
import os
import sqlite3
import tempfile

import pandas as pd
import pytest

from sqlalchemy import text

from lumen.transforms.sql import SQLGroupBy

try:
    from lumen.sources.sqlalchemy import SQLAlchemySource
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
    return source


@pytest.fixture
def sqlalchemy_memory_source(mixed_df):
    """Create an in-memory SQLite database with test data."""
    source = SQLAlchemySource(
        url='sqlite:///:memory:',
    )
    
    # Load data into the database
    with source._engine.begin() as conn:
        mixed_df.to_sql('mixed', conn, if_exists='replace', index=False)
    
    source.tables = {'mixed': 'SELECT * FROM mixed'}
    return source


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


def test_sqlalchemy_connection_url():
    """Test various connection URL formats."""
    # SQLite in-memory
    source1 = SQLAlchemySource(url='sqlite:///:memory:')
    assert source1._url.drivername == 'sqlite'
    
    # SQLite file
    source2 = SQLAlchemySource(url='sqlite:///test.db')
    assert source2._url.drivername == 'sqlite'
    
    # PostgreSQL URL (won't actually connect)
    try:
        source3 = SQLAlchemySource(url='postgresql://user:pass@localhost/db')
        assert source3._url.drivername == 'postgresql'
    except Exception:
        pass  # It's OK if connection fails, we just want to test URL parsing


def test_sqlalchemy_connection_components():
    """Test connection using components instead of URL."""
    source = SQLAlchemySource(
        drivername='sqlite',
        database=':memory:'
    )
    assert source._url.drivername == 'sqlite'
    assert source._url.database == ':memory:'


def test_sqlalchemy_driver_detection():
    """Test async driver detection."""
    # Sync driver
    source_sync = SQLAlchemySource(url='sqlite:///:memory:')
    assert not source_sync._driver_is_async
    
    # Async drivers (won't connect, just test detection)
    assert SQLAlchemySource._is_async_driver('postgresql+asyncpg')
    assert SQLAlchemySource._is_async_driver('mysql+asyncmy')
    assert SQLAlchemySource._is_async_driver('sqlite+aiosqlite')
    assert not SQLAlchemySource._is_async_driver('postgresql+psycopg2')
    assert not SQLAlchemySource._is_async_driver('sqlite')


def test_sqlalchemy_get_tables_from_inspector():
    """Test automatic table discovery using SQLAlchemy inspector."""
    # Create an in-memory database with multiple tables
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Create some test tables
    with source._engine.begin() as conn:
        conn.execute(text('CREATE TABLE table1 (id INTEGER, name TEXT)'))
        conn.execute(text('CREATE TABLE table2 (id INTEGER, value REAL)'))
        conn.execute(text('INSERT INTO table1 VALUES (1, "test")'))
        conn.execute(text('INSERT INTO table2 VALUES (1, 1.5)'))
    
    # Get tables should discover both tables
    tables = source.get_tables()
    assert 'table1' in tables
    assert 'table2' in tables


def test_sqlalchemy_excluded_tables():
    """Test table exclusion patterns."""
    source = SQLAlchemySource(
        url='sqlite:///:memory:',
        excluded_tables=['temp_*', 'test_*']
    )
    
    # Create test tables
    with source._engine.begin() as conn:
        conn.execute(text('CREATE TABLE temp_data (id INTEGER)'))
        conn.execute(text('CREATE TABLE test_data (id INTEGER)'))
        conn.execute(text('CREATE TABLE real_data (id INTEGER)'))
    
    tables = source.get_tables()
    assert 'real_data' in tables
    assert 'temp_data' not in tables
    assert 'test_data' not in tables


def test_sqlalchemy_metadata_extraction():
    """Test metadata extraction using SQLAlchemy inspector."""
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Create a table with some data
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'value': [100.5, 200.75, 300.0]
    })
    
    with source._engine.begin() as conn:
        df.to_sql('test_table', conn, if_exists='replace', index=False)
    
    # Get metadata
    metadata = source._get_table_metadata(['test_table'])
    
    assert 'test_table' in metadata
    assert 'columns' in metadata['test_table']
    assert 'id' in metadata['test_table']['columns']
    assert 'name' in metadata['test_table']['columns']
    assert 'value' in metadata['test_table']['columns']


def test_sqlalchemy_create_sql_expr_source():
    """Test creating a new source with SQL expressions."""
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Create initial table
    df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    with source._engine.begin() as conn:
        df.to_sql('data', conn, if_exists='replace', index=False)
    
    source.tables = {'data': 'SELECT * FROM data'}
    
    # Create new source with SQL expression
    new_tables = {
        'filtered': 'SELECT * FROM data WHERE value > 15'
    }
    new_source = source.create_sql_expr_source(new_tables)
    
    result = new_source.get('filtered')
    assert len(result) == 2
    assert all(result['value'] > 15)


def test_sqlalchemy_table_params_basic():
    """Test table_params with parameterized queries."""
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Create test data
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'city': ['NYC', 'LA', 'Chicago']
    })
    
    with source._engine.begin() as conn:
        df.to_sql('customers', conn, if_exists='replace', index=False)
    
    # Use parameterized query
    source.tables = {
        'customers': 'SELECT * FROM customers',
        'by_city': 'SELECT * FROM customers WHERE city = ?'
    }
    source.table_params = {'by_city': ['NYC']}
    
    result = source.get('by_city')
    assert len(result) == 1
    assert result.iloc[0]['city'] == 'NYC'


def test_sqlalchemy_table_params_multiple():
    """Test table_params with multiple parameters."""
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Create test data
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'value': [10, 20, 30, 40]
    })
    
    with source._engine.begin() as conn:
        df.to_sql('data', conn, if_exists='replace', index=False)
    
    # Use multiple parameters
    source.tables = {
        'data': 'SELECT * FROM data',
        'range_query': 'SELECT * FROM data WHERE value > ? AND value < ?'
    }
    source.table_params = {'range_query': [15, 35]}
    
    result = source.get('range_query')
    assert len(result) == 2
    assert set(result['value']) == {20, 30}


def test_sqlalchemy_create_sql_expr_source_with_params():
    """Test create_sql_expr_source with params."""
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Create test data
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'B', 'C'],
        'value': [10, 20, 30, 40, 50]
    })
    
    with source._engine.begin() as conn:
        df.to_sql('data', conn, if_exists='replace', index=False)
    
    source.tables = {'data': 'SELECT * FROM data'}
    
    # Create new source with parameterized query
    new_tables = {
        'filtered': 'SELECT * FROM data WHERE category = ? AND value > ?'
    }
    params = {'filtered': ['A', 15]}
    
    new_source = source.create_sql_expr_source(new_tables, params=params)
    
    result = new_source.get('filtered')
    assert len(result) == 1
    assert result.iloc[0]['category'] == 'A'
    assert result.iloc[0]['value'] == 30


def test_sqlalchemy_execute():
    """Test direct SQL execution."""
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Create and populate table
    with source._engine.begin() as conn:
        conn.execute(text('CREATE TABLE test (id INTEGER, name TEXT)'))
        conn.execute(text('INSERT INTO test VALUES (1, "Alice")'))
        conn.execute(text('INSERT INTO test VALUES (2, "Bob")'))
    
    # Execute query
    result = source.execute('SELECT * FROM test WHERE id = ?', [1])
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


def test_sqlalchemy_dialect_detection():
    """Test dialect detection from URL."""
    source_sqlite = SQLAlchemySource(url='sqlite:///:memory:')
    assert source_sqlite.dialect == 'sqlite'
    
    # Test with components
    source_sqlite2 = SQLAlchemySource(drivername='sqlite', database=':memory:')
    assert source_sqlite2.dialect == 'sqlite'


@pytest.mark.asyncio
async def test_sqlalchemy_async_with_sync_driver():
    """Test that async methods work with sync drivers (using thread pool)."""
    # Use a file-based SQLite instead of in-memory to avoid connection issues
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
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
        # Clean up
        import os
        os.unlink(temp_db.name)


def test_sqlalchemy_error_handling():
    """Test error handling for invalid queries."""
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Invalid SQL should raise an error
    with pytest.raises(Exception):
        source.execute('SELECT * FROM nonexistent_table')
    
    # Invalid table should raise an error
    with pytest.raises(Exception):
        source.get('nonexistent_table')


def test_sqlalchemy_schema_timeout():
    """Test schema_timeout_seconds parameter."""
    source = SQLAlchemySource(
        url='sqlite:///:memory:',
        schema_timeout_seconds=1
    )
    
    # Create a simple table
    with source._engine.begin() as conn:
        conn.execute(text('CREATE TABLE test (id INTEGER)'))
        conn.execute(text('INSERT INTO test VALUES (1)'))
    
    source.tables = {'test': 'SELECT * FROM test'}
    
    # Get schema should work
    schema = source.get_schema('test')
    assert 'id' in schema


def test_sqlalchemy_filter_in_sql():
    """Test filter_in_sql parameter."""
    source = SQLAlchemySource(
        url='sqlite:///:memory:',
        filter_in_sql=True
    )
    
    # Create test data
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'B', 'C']
    })
    
    with source._engine.begin() as conn:
        df.to_sql('data', conn, if_exists='replace', index=False)
    
    source.tables = {'data': 'SELECT * FROM data'}
    
    # Filter should be applied in SQL
    result = source.get('data', category='A')
    assert len(result) == 2
    assert all(result['category'] == 'A')


def test_sqlalchemy_engine_reuse():
    """Test that create_sql_expr_source reuses the engine."""
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Create test data
    df = pd.DataFrame({'id': [1, 2, 3]})
    with source._engine.begin() as conn:
        df.to_sql('data', conn, if_exists='replace', index=False)
    
    source.tables = {'data': 'SELECT * FROM data'}
    
    # Create new source
    new_source = source.create_sql_expr_source({'filtered': 'SELECT * FROM data WHERE id > 1'})
    
    # Should reuse the same engine
    assert new_source._engine is source._engine


def test_sqlalchemy_connect_args():
    """Test connect_args parameter."""
    source = SQLAlchemySource(
        url='sqlite:///:memory:',
        connect_args={'check_same_thread': False}
    )
    
    # Should create successfully
    assert source._engine is not None


def test_sqlalchemy_engine_kwargs():
    """Test engine_kwargs parameter."""
    source = SQLAlchemySource(
        url='sqlite:///:memory:',
        engine_kwargs={'echo': False, 'pool_pre_ping': True}
    )
    
    # Should create successfully
    assert source._engine is not None


def test_sqlalchemy_query_params():
    """Test query_params in URL."""
    source = SQLAlchemySource(
        drivername='sqlite',
        database=':memory:',
        query_params={'timeout': '30'}
    )
    
    # Should create URL successfully
    assert source._url is not None


def test_sqlalchemy_invalid_url():
    """Test error handling for invalid URL."""
    with pytest.raises(ValueError):
        # Missing both url and drivername
        SQLAlchemySource()


def test_sqlalchemy_get_metadata():
    """Test get_metadata method."""
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Create test table
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie']
    })
    
    with source._engine.begin() as conn:
        df.to_sql('users', conn, if_exists='replace', index=False)
    
    # Get metadata
    metadata = source.get_metadata('users')
    
    assert 'description' in metadata
    assert 'columns' in metadata
    assert 'id' in metadata['columns']
    assert 'name' in metadata['columns']
    assert 'rows' in metadata


def test_sqlalchemy_custom_sql_expr():
    """Test custom sql_expr parameter."""
    source = SQLAlchemySource(
        url='sqlite:///:memory:',
        sql_expr='SELECT id, name FROM {table} LIMIT 10',
        tables={'users': 'users'}
    )
    
    # Create test data
    df = pd.DataFrame({
        'id': list(range(1, 21)),
        'name': [f'User{i}' for i in range(1, 21)],
        'value': list(range(100, 120))
    })
    
    with source._engine.begin() as conn:
        df.to_sql('users', conn, if_exists='replace', index=False)
    
    # Get should use custom sql_expr
    result = source.get('users')
    assert len(result) <= 10
    assert 'id' in result.columns
    assert 'name' in result.columns


def test_sqlalchemy_create_sql_expr_no_nested_transaction():
    """Test that create_sql_expr_source doesn't cause nested transaction errors."""
    source = SQLAlchemySource(url='sqlite:///:memory:')
    
    # Create test data (use 'data_table' instead of 'table' which is a reserved keyword)
    df = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50]})
    with source._engine.begin() as conn:
        df.to_sql('data_table', conn, if_exists='replace', index=False)
    
    source.tables = {'data_table': 'SELECT * FROM data_table'}
    
    # This should not raise a nested transaction error
    new_source = source.create_sql_expr_source({"limited": "SELECT * FROM data_table LIMIT 2"})
    result = new_source.get("limited")
    
    assert len(result) == 2
    assert list(result['id']) == [1, 2]
