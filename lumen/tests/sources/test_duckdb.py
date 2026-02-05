import datetime as dt
import os
import tempfile

from pathlib import Path

import pandas as pd
import pytest

from lumen.transforms.sql import SQLGroupBy

try:
    import duckdb

    from lumen.sources.duckdb import DuckDBSource
    pytestmark = pytest.mark.xdist_group("duckdb")
except ImportError:
    pytestmark = pytest.mark.skip(reason="Duckdb is not installed")


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

@pytest.fixture
def duckdb_source():
    root = os.path.dirname(__file__)
    duckdb_source = DuckDBSource(
        initializers=[
            "INSTALL sqlite;",
            "LOAD sqlite;",
            f"SET home_directory='{root}';"
        ],
        root=root,
        sql_expr="SELECT A, B, C, D::TIMESTAMP_NS AS D FROM {table}",
        tables={
            'test_sql': f"sqlite_scan('{root + '/test.db'}', 'mixed')",
            'test_sql_with_none': f"sqlite_scan('{root + '/test.db'}', 'mixed_none')",
        }
    )
    return duckdb_source


@pytest.fixture
def duckdb_memory_source(mixed_df):
    source = DuckDBSource(uri=':memory:', ephemeral=True)
    # Convert pyarrow string dtype to object for DuckDB compatibility if necessary
    df = mixed_df.copy()
    for col in df.select_dtypes(include=['string']).columns:
        df[col] = df[col].astype(object)
    source._connection.from_df(df).to_view('mixed')
    return source


def test_duckdb_resolve_module_type():
    assert DuckDBSource._get_type('lumen.sources.duckdb.DuckDBSource') is DuckDBSource
    assert DuckDBSource.source_type == 'duckdb'


def test_duckdb_get_tables(duckdb_source, source_tables):
    tables = duckdb_source.get_tables()
    assert not len(set(tables) - set(source_tables.keys()))
    for table in tables:
        pd.testing.assert_frame_equal(
            duckdb_source.get(table),
            source_tables[table],
            check_dtype=False
        )


def test_duckdb_get_schema(duckdb_source):
    expected_sql = {
        'A': {'inclusiveMaximum': 4.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'B': {'inclusiveMaximum': 1.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'C': {'enum': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], 'type': 'string'},
        'D': {
            'format': 'datetime',
            'inclusiveMaximum': '2009-01-07 00:00:00',
            'inclusiveMinimum': '2009-01-01 00:00:00',
            'type': 'string'
        },
        '__len__': 5
    }
    source = duckdb_source.get_schema('test_sql')
    source["C"]["enum"].sort()
    assert source == expected_sql
    assert list(duckdb_source._schema_cache.keys()) == ['test_sql']


def test_duckdb_get_schema_with_limit(duckdb_source):
    expected_sql = {
        'A': {'inclusiveMaximum': 0.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'B': {'inclusiveMaximum': 0.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'C': {'enum': ['foo1'], 'type': 'string'},
        'D': {
            'format': 'datetime',
            'inclusiveMaximum': '2009-01-01T00:00:00',
            'inclusiveMinimum': '2009-01-01T00:00:00',
            'type': 'string'
        },
        '__len__': 5
    }
    source = duckdb_source.get_schema('test_sql', limit=1)
    source["C"]["enum"].sort()
    assert source == expected_sql
    assert list(duckdb_source._schema_cache.keys()) == ['test_sql']


def test_duckdb_get_schema_with_none(duckdb_source):
    enum = ['foo1', None, 'foo3', 'foo5']
    expected_sql = {
        'A': {'inclusiveMaximum': 4.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'B': {'inclusiveMaximum': 1.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'C': {'enum': enum, 'type': 'string'},
        'D': {
            'format': 'datetime',
            'inclusiveMaximum': '2009-01-07 00:00:00',
            'inclusiveMinimum': '2009-01-01 00:00:00',
            'type': 'string'
        },
        '__len__': 5
    }
    source = duckdb_source.get_schema('test_sql_with_none')
    source["C"]["enum"] = [None if pd.isna(e) else e for e in source["C"]["enum"]]
    source["C"]["enum"].sort(key=enum.index)
    assert source == expected_sql
    assert list(duckdb_source._schema_cache.keys()) == ['test_sql_with_none']


def test_duckdb_get_schema_cache(duckdb_source):
    duckdb_source.get_schema('test_sql')
    assert 'test_sql' in duckdb_source._schema_cache


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
def test_duckdb_filter(duckdb_source, table_column_value_type, dask, expected_filtered_df):
    table, column, value, _ = table_column_value_type
    kwargs = {column: value}
    filtered = duckdb_source.get(table, __dask=dask, **kwargs)

    expected = expected_filtered_df.reset_index(drop=True)
    if 'C' in filtered.columns and 'C' in expected.columns:
        filtered['C'] = [None if pd.isna(v) else v for v in filtered['C']]
        expected['C'] = [None if pd.isna(v) else v for v in expected['C']]

    # check_dtype=False because DuckDB may return strings as object dtype
    # while the original DataFrame might use the Arrow-backed StringDtype
    pd.testing.assert_frame_equal(filtered, expected, check_dtype=False)


def test_duckdb_transforms(duckdb_source, source_tables):
    df_test_sql = source_tables['test_sql']
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]
    transformed = duckdb_source.get('test_sql', sql_transforms=transforms)
    expected = df_test_sql.groupby('B')['A'].sum().reset_index()

    pd.testing.assert_frame_equal(
        transformed.sort_values("B").reset_index(drop=True),
        expected.sort_values("B").reset_index(drop=True)
    )


@pytest.mark.flaky(reruns=3)
def test_duckdb_transforms_cache(duckdb_source, source_tables):
    df_test_sql = source_tables['test_sql']
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]

    duckdb_source.get('test_sql', sql_transforms=transforms)
    cache_key = duckdb_source._get_key('test_sql', sql_transforms=transforms)
    assert cache_key in duckdb_source._cache

    expected = df_test_sql.groupby('B')['A'].sum().reset_index()
    pd.testing.assert_frame_equal(duckdb_source._cache[cache_key], expected)

    cache_key = duckdb_source._get_key('test_sql', sql_transforms=transforms)
    assert cache_key in duckdb_source._cache


def test_duckdb_clear_cache(duckdb_source):
    duckdb_source.get('test_sql')
    duckdb_source.get_schema('test_sql')
    assert len(duckdb_source._cache) == 1
    assert len(duckdb_source._schema_cache) == 1
    duckdb_source.clear_cache()
    assert len(duckdb_source._cache) == 0
    assert len(duckdb_source._schema_cache) == 0


def test_duckdb_source_ephemeral_roundtrips(duckdb_memory_source, mixed_df):
    source = DuckDBSource.from_spec(duckdb_memory_source.to_spec())
    df = source.get('mixed')
    # Use pd.testing to handle NA/None consistently
    # and check_dtype=False because DuckDB may return strings as object dtype
    expected = mixed_df.copy()
    pd.testing.assert_frame_equal(df, expected, check_dtype=False)


def test_duckdb_source_mirrors_source(duckdb_source):
    mirrored = DuckDBSource(uri=':memory:', mirrors={'mixed': (duckdb_source, 'test_sql')})
    pd.testing.assert_frame_equal(duckdb_source.get('test_sql'), mirrored.get('mixed'), check_dtype=False)


@pytest.fixture
def sample_csv_files():
    """Create temporary CSV files for testing."""
    test_dir = tempfile.mkdtemp()

    # Sample data
    customers_data = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'city': ['NYC', 'LA', 'Chicago']
    })

    orders_data = pd.DataFrame({
        'id': [101, 102, 103],
        'customer_id': [1, 2, 1],
        'total': [250.5, 150.75, 300.0]
    })

    # Save to CSV files
    customers_path = os.path.join(test_dir, 'customers.csv')
    orders_path = os.path.join(test_dir, 'orders.csv')
    test_path = os.path.join(test_dir, 'test.csv')
    test2_path = os.path.join(test_dir, 'test2.csv')

    customers_data.to_csv(customers_path, index=False)
    orders_data.to_csv(orders_path, index=False)
    customers_data.to_csv(test_path, index=False)  # For user's examples
    orders_data.to_csv(test2_path, index=False)   # For user's examples

    files_dict = {
        'dir': test_dir,
        'customers': customers_path,
        'orders': orders_path,
        'test': test_path,
        'test2': test2_path,
        'customers_data': customers_data,
        'orders_data': orders_data
    }

    yield files_dict

    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)


def test_file_path_detection():
    """Test the _is_file_path method with various inputs."""
    source = DuckDBSource(uri=':memory:', tables={})

    # Should detect as file paths
    assert source._is_file_path('customers.csv') == True
    assert source._is_file_path('data/orders.parquet') == True
    assert source._is_file_path('products.json') == True
    assert source._is_file_path('/absolute/path/data.csv') == True

    # Should detect as SQL
    assert source._is_file_path('SELECT * FROM customers') == False
    assert source._is_file_path('SELECT * FROM READ_CSV(\'data.csv\')') == False
    assert source._is_file_path('WITH cte AS (SELECT * FROM orders) SELECT * FROM cte') == False
    assert source._is_file_path('INSERT INTO customers VALUES (1, \'John\')') == False

    # Edge cases - these could go either way depending on SQLGlot parsing
    # Just ensure they don't crash
    try:
        source._is_file_path('customers')
        source._is_file_path('data')
        source._is_file_path('')
    except Exception as e:
        pytest.fail(f"_is_file_path should handle edge cases gracefully: {e}")


def test_automatic_csv_view_creation(sample_csv_files):
    """Test automatic view creation from CSV file paths."""
    files = sample_csv_files

    # Change to test directory so relative paths work
    original_cwd = os.getcwd()
    try:
        os.chdir(files['dir'])

        # Test automatic file detection - your simplified syntax
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',  # File path -> automatic view
                'orders': 'orders.csv'         # File path -> automatic view
            }
        )

        # Check tables are available
        tables = source.get_tables()
        assert 'customers' in tables
        assert 'orders' in tables

        # Test querying the views
        customers_result = source.execute("SELECT * FROM customers")
        orders_result = source.execute("SELECT * FROM orders")

        # Verify data matches
        pd.testing.assert_frame_equal(
            customers_result.sort_values('id').reset_index(drop=True),
            files['customers_data'].sort_values('id').reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            orders_result.sort_values('id').reset_index(drop=True),
            files['orders_data'].sort_values('id').reset_index(drop=True)
        )

    finally:
        os.chdir(original_cwd)


def test_create_sql_expr_source_preserves_original_sql(sample_csv_files):
    """Test that create_sql_expr_source preserves original SQL expressions, not generic SELECT * FROM."""
    files = sample_csv_files
    original_cwd = os.getcwd()
    
    try:
        os.chdir(files['dir'])
        
        # Create initial source with CSV files
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv'
            }
        )
        
        # Create a new source with SQL expressions
        new_tables = {
            'filtered_customers': 'SELECT * FROM customers WHERE id > 1',
            'order_summary': 'SELECT customer_id, COUNT(*) as order_count, SUM(total) as total_amount FROM orders GROUP BY customer_id'
        }
        
        new_source = source.create_sql_expr_source(new_tables)
        
        # Check that the new source has the tables
        tables = new_source.get_tables()
        assert 'filtered_customers' in tables
        assert 'order_summary' in tables
        
        # Most importantly: check that the stored SQL expressions are the original ones,
        # not generic "SELECT * FROM table_name"
        assert new_source.tables['filtered_customers'] == 'SELECT * FROM customers WHERE id > 1'
        assert new_source.tables['order_summary'] == 'SELECT customer_id, COUNT(*) as order_count, SUM(total) as total_amount FROM orders GROUP BY customer_id'
        
        # Verify the SQL expressions actually work
        filtered_result = new_source.get('filtered_customers')
        assert len(filtered_result) == 2  # Only customers with id > 1 (Bob and Charlie)
        assert all(filtered_result['id'] > 1)
        
        summary_result = new_source.get('order_summary')
        assert len(summary_result) == 2  # 2 unique customer_ids
        assert 'order_count' in summary_result.columns
        assert 'total_amount' in summary_result.columns
        
    finally:
        os.chdir(original_cwd)


def test_user_examples(sample_csv_files):
    """Test the exact examples from the user's code."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        # Example 1: Traditional approach (still works)
        source1 = DuckDBSource(
            uri=':memory:',
            tables={
                "test": "SELECT * FROM READ_CSV('test.csv')",
                "test2": "SELECT * FROM READ_CSV('test2.csv')",
                "test 3": "SELECT * FROM READ_CSV('test.csv')",  # Space in name
            }
        )

        # Example 2: New simplified approach
        source2 = DuckDBSource(
            uri=':memory:',
            tables={
                "test": "test.csv",    # File path detected -> creates view
                "test2": "test2.csv",  # File path detected -> creates view
                "test_3": "test.csv",  # Space in name -> creates view
            }
        )

        # All should work and produce same results
        result1_test = source1.execute("SELECT * FROM test")
        result1_test2 = source1.execute("SELECT * FROM test2")
        result1_test3 = source1.execute("SELECT * FROM 'test_3'")  # Space in name

        result2_test = source2.execute("SELECT * FROM test")
        result2_test2 = source2.execute("SELECT * FROM test2")
        result2_test3 = source2.execute("SELECT * FROM 'test_3'")

        # Compare results (should be identical)
        pd.testing.assert_frame_equal(
            result1_test.sort_values('id').reset_index(drop=True),
            result2_test.sort_values('id').reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            result1_test2.sort_values('id').reset_index(drop=True),
            result2_test2.sort_values('id').reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            result1_test3.sort_values('id').reset_index(drop=True),
            result2_test3.sort_values('id').reset_index(drop=True)
        )
    finally:
        os.chdir(original_cwd)


def test_mixed_file_paths_and_sql(sample_csv_files):
    """Test mixing file paths with SQL expressions."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        source = DuckDBSource(
            uri=':memory:',
            tables={
                # File paths - should create views
                'customers': 'customers.csv',
                'orders': 'orders.csv',

                # SQL expressions - should be processed normally
                'high_value_orders': 'SELECT * FROM orders WHERE total > 200',
                'customer_count': 'SELECT COUNT(*) as count FROM customers',

                # Complex SQL
                'customer_orders': '''
                    SELECT c.name, COUNT(o.id) as order_count
                    FROM customers c
                    LEFT JOIN orders o ON c.id = o.customer_id
                    GROUP BY c.name
                '''
            }
        )

        # Test all table types work
        tables = source.get_tables()
        expected_tables = {'customers', 'orders', 'high_value_orders', 'customer_count', 'customer_orders'}
        assert set(tables) == expected_tables

        # Test file-based views
        customers = source.execute("SELECT * FROM customers")
        assert len(customers) == 3

        # Test SQL-based tables
        high_value = source.execute("SELECT * FROM high_value_orders")
        assert len(high_value) == 2  # Orders with total > 200

        count_result = source.execute("SELECT * FROM customer_count")
        assert count_result.iloc[0, 0] == 3  # 3 customers

        # Test complex join
        customer_orders = source.execute("SELECT * FROM customer_orders ORDER BY name")
        assert len(customer_orders) == 3  # 3 customers

    finally:
        os.chdir(original_cwd)


def test_fallback_behavior_missing_files():
    """Test fallback behavior when files don't exist."""
    source = DuckDBSource(
        uri=':memory:',
        tables={
            'missing_file': 'nonexistent.csv',  # Should fallback to SQL expression
            'valid_sql': 'SELECT 1 as test_col'  # Should work normally
        }
    )

    # Valid SQL should work
    result = source.execute("SELECT * FROM valid_sql")
    assert result.iloc[0, 0] == 1

    # Missing file handling depends on implementation details
    # At minimum, it shouldn't crash the entire source creation
    tables = source.get_tables()
    assert 'valid_sql' in tables


def test_different_file_extensions():
    """Test detection of different file extensions."""
    source = DuckDBSource(uri=':memory:', tables={})

    # Test various extensions are detected as file paths
    file_extensions = [
        'data.csv',
        'data.parquet',
        'data.json',
        'data.jsonl',
        'data.ndjson'
    ]

    for file_path in file_extensions:
        assert source._is_file_path(file_path) == True, f"{file_path} should be detected as file path"


def test_table_key_naming():
    """Test that dict keys become table names correctly."""
    # Test with sample data since we can't rely on external files
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})

    source = DuckDBSource.from_df({'my_custom_table_name': df})

    tables = source.get_tables()
    assert 'my_custom_table_name' in tables

    result = source.execute("SELECT * FROM my_custom_table_name")
    # check_dtype=False because DuckDB may return strings as object dtype
    # while the original DataFrame might use the Arrow-backed StringDtype
    pd.testing.assert_frame_equal(result, df, check_dtype=False)


def test_absolute_vs_relative_paths(sample_csv_files):
    """Test handling of absolute vs relative file paths."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        source = DuckDBSource(
            uri=':memory:',
            tables={
                'relative_path': 'customers.csv',              # Relative path
                'absolute_path': files['customers'],           # Absolute path
            }
        )

        # Both should work and give same results
        result_relative = source.execute("SELECT * FROM relative_path")
        result_absolute = source.execute("SELECT * FROM absolute_path")

        pd.testing.assert_frame_equal(
            result_relative.sort_values('id').reset_index(drop=True),
            result_absolute.sort_values('id').reset_index(drop=True)
        )

    finally:
        os.chdir(original_cwd)

def test_mirrors_not_in_tables_from_spec():
    """Test that mirrors are not included in the tables from_spec."""
    spec = DuckDBSource(tables={"obs": "SELECT * FROM obs"}).to_spec()
    source = DuckDBSource.from_spec(spec)
    assert "mirrors" not in source.tables
    assert source.mirrors == {}


def test_create_sql_expr_source_with_params(sample_csv_files):
    """Test that create_sql_expr_source accepts and uses params correctly."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        # Create initial source with CSV files
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv'
            }
        )

        # Create a new source with parameterized SQL expressions
        new_tables = {
            'filtered_customers': 'SELECT * FROM customers WHERE id > ?',
            'high_value_orders': 'SELECT * FROM orders WHERE total > ? AND customer_id = ?'
        }

        # Define parameters for each table
        params = {
            'filtered_customers': [1],  # id > 1
            'high_value_orders': [200, 1]  # total > 200 AND customer_id = 1
        }

        new_source = source.create_sql_expr_source(new_tables, params=params)

        # Check that the new source has the tables
        tables = new_source.get_tables()
        assert 'filtered_customers' in tables
        assert 'high_value_orders' in tables

        # Verify the parameterized queries work correctly
        filtered_result = new_source.get('filtered_customers')
        assert len(filtered_result) == 2  # Only customers with id > 1 (Bob and Charlie)
        assert all(filtered_result['id'] > 1)
        assert set(filtered_result['name']) == {'Bob', 'Charlie'}

        high_value_result = new_source.get('high_value_orders')
        assert len(high_value_result) == 2  # Two orders match: customer_id=1 and total > 200
        assert all(high_value_result['customer_id'] == 1)
        assert all(high_value_result['total'] > 200)
        # Check specific values
        assert set(high_value_result['total']) == {250.5, 300.0}

    finally:
        os.chdir(original_cwd)


def test_table_params_parameter(sample_csv_files):
    """Test using table_params as a public parameter directly in constructor."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        # Create source with table_params directly
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'filtered_customers': 'SELECT * FROM customers WHERE city = ?'
            },
            table_params={
                'filtered_customers': ['NYC']
            }
        )

        # Query the parameterized table
        result = source.get('filtered_customers')
        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Alice'
        assert result.iloc[0]['city'] == 'NYC'

        # Regular table should still work
        all_customers = source.get('customers')
        assert len(all_customers) == 3

    finally:
        os.chdir(original_cwd)


def test_create_sql_expr_source_without_params(sample_csv_files):
    """Test that create_sql_expr_source works without params (backward compatibility)."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv'
            }
        )

        # Create a new source WITHOUT params - should work as before
        new_tables = {
            'all_customers': 'SELECT * FROM customers',
            'all_orders': 'SELECT * FROM orders'
        }

        new_source = source.create_sql_expr_source(new_tables)

        # Verify tables work without params
        customers_result = new_source.get('all_customers')
        assert len(customers_result) == 3

        orders_result = new_source.get('all_orders')
        assert len(orders_result) == 3

    finally:
        os.chdir(original_cwd)


def test_create_sql_expr_source_mixed_params(sample_csv_files):
    """Test create_sql_expr_source with some tables having params and others not."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv'
            }
        )

        # Mix of parameterized and non-parameterized queries
        new_tables = {
            'filtered_customers': 'SELECT * FROM customers WHERE id = ?',  # Has params
            'all_orders': 'SELECT * FROM orders'  # No params
        }

        params = {
            'filtered_customers': [2]  # Only customer with id=2 (Bob)
        }

        new_source = source.create_sql_expr_source(new_tables, params=params)

        # Verify parameterized query
        filtered_result = new_source.get('filtered_customers')
        assert len(filtered_result) == 1
        assert filtered_result.iloc[0]['id'] == 2
        assert filtered_result.iloc[0]['name'] == 'Bob'

        # Verify non-parameterized query
        orders_result = new_source.get('all_orders')
        assert len(orders_result) == 3

    finally:
        os.chdir(original_cwd)


def test_detour_roundtrip(sample_csv_files):
    """
    Test that creating a SQL expression source from an existing source
    preserves the original SQL file-based tables so that it can
    be re-serialized without error.
    """
    files = sample_csv_files
    original_cwd = os.getcwd()
    
    try:
        os.chdir(files['dir'])
        
        # Create source with file-based tables
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv'
            }
        )
        df = source.get("customers")
        
        # Create a derived source with a new SQL expression
        new_source = source.create_sql_expr_source(
            tables={"limited_customers": 'SELECT * FROM customers LIMIT 1'}
        )
        limited_df = new_source.get("limited_customers")
        assert len(limited_df) == 1
        assert limited_df.iloc[[0]].equals(df.iloc[[0]])

        # Serialize and deserialize
        spec = new_source.to_spec()
        
        read_source = DuckDBSource.from_spec(spec)
        read_df = read_source.get("limited_customers")
        assert len(read_df) == 1
        assert read_df.iloc[[0]].equals(df.iloc[[0]])
        assert read_source.tables["limited_customers"] == 'SELECT * FROM customers LIMIT 1'
        assert "customers" in read_source.tables
        assert "orders" in read_source.tables
    finally:
        os.chdir(original_cwd)


def test_table_params_basic(sample_csv_files):
    """Test table_params with single and multiple parameters."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv',
                'by_name': 'SELECT * FROM customers WHERE name = ?',
                'by_customer_total': 'SELECT * FROM orders WHERE customer_id = ? AND total > ?'
            },
            table_params={
                'by_name': ['Alice'],
                'by_customer_total': [1, 200]
            }
        )

        # Single parameter
        result = source.get('by_name')
        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Alice'

        # Multiple parameters
        result = source.get('by_customer_total')
        assert len(result) == 2
        assert all(result['customer_id'] == 1)
        assert all(result['total'] > 200)
    finally:
        os.chdir(original_cwd)


def test_table_params_data_types(sample_csv_files):
    """Test table_params with various data types and SQL patterns."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv',
                'by_int': 'SELECT * FROM orders WHERE customer_id = ?',
                'by_float': 'SELECT * FROM orders WHERE total > ?',
                'by_like': 'SELECT * FROM customers WHERE name LIKE ?',
                'by_in': 'SELECT * FROM customers WHERE city IN (?, ?)'
            },
            table_params={
                'by_int': [2],
                'by_float': [200.5],
                'by_like': ['%li%'],
                'by_in': ['NYC', 'LA']
            }
        )

        assert source.get('by_int').iloc[0]['customer_id'] == 2
        assert len(source.get('by_float')) == 2
        assert set(source.get('by_like')['name']) == {'Alice', 'Charlie'}
        assert len(source.get('by_in')) == 2
    finally:
        os.chdir(original_cwd)


def test_table_params_complex_queries(sample_csv_files):
    """Test table_params with JOINs, aggregations, CTEs, and subqueries."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv',
                'join_query': 'SELECT c.name, o.total FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.name = ?',
                'agg_query': 'SELECT customer_id, COUNT(*) as cnt, SUM(total) as sum FROM orders WHERE customer_id = ? GROUP BY customer_id',
                'cte_query': 'WITH totals AS (SELECT customer_id, SUM(total) as spent FROM orders GROUP BY customer_id) SELECT c.name, t.spent FROM customers c JOIN totals t ON c.id = t.customer_id WHERE t.spent > ?',
                'subquery': 'SELECT * FROM customers WHERE id IN (SELECT customer_id FROM orders WHERE total > ?)'
            },
            table_params={
                'join_query': ['Alice'],
                'agg_query': [1],
                'cte_query': [200],
                'subquery': [200]
            }
        )

        # JOIN
        join_result = source.get('join_query')
        assert len(join_result) == 2
        assert all(join_result['name'] == 'Alice')

        # Aggregation
        agg_result = source.get('agg_query')
        assert agg_result.iloc[0]['cnt'] == 2
        assert agg_result.iloc[0]['sum'] == 550.5

        # CTE
        cte_result = source.get('cte_query')
        assert cte_result.iloc[0]['name'] == 'Alice'

        # Subquery
        subquery_result = source.get('subquery')
        assert subquery_result.iloc[0]['name'] == 'Alice'
    finally:
        os.chdir(original_cwd)


def test_table_params_in_create_sql_expr_source(sample_csv_files):
    """Test params in create_sql_expr_source and propagation."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])
        base_source = DuckDBSource(
            uri=':memory:',
            tables={'customers': 'customers.csv'},
            table_params={'base_filter': [1]}
        )

        # Create new source with params
        new_source = base_source.create_sql_expr_source(
            tables={'new_filter': 'SELECT * FROM customers WHERE city = ?'},
            params={'new_filter': ['LA']}
        )

        result = new_source.get('new_filter')
        assert len(result) == 1
        assert result.iloc[0]['city'] == 'LA'
    finally:
        os.chdir(original_cwd)


def test_table_params_edge_cases(sample_csv_files):
    """Test edge cases: empty params, mismatched counts, multiple tables."""
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})

    # Empty params should work fine
    source1 = DuckDBSource.from_df({'test': df}, table_params={})
    pd.testing.assert_frame_equal(source1.get('test'), df, check_dtype=False)

    # Mismatched parameter count - need to create source directly, not from_df
    source2 = DuckDBSource(
        uri=':memory:',
        ephemeral=True,
        tables={
            'test': 'SELECT * FROM test',
            'bad': 'SELECT * FROM test WHERE col1 > ? AND col1 < ?'
        },
        table_params={'bad': [1]}  # Missing second parameter
    )
    # First load the test table
    source2._connection.from_df(df).to_view('test')
    
    # Now try to query with mismatched params
    with pytest.raises(Exception):  # DuckDB will raise an error about bind parameter count
        source2.get('bad')

    # Multiple tables with different params
    files = sample_csv_files
    original_cwd = os.getcwd()
    try:
        os.chdir(files['dir'])
        source3 = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'by_city': 'SELECT * FROM customers WHERE city = ?',
                'by_id': 'SELECT * FROM customers WHERE id = ?'
            },
            table_params={'by_city': ['NYC'], 'by_id': [2]}
        )
        assert source3.get('by_city').iloc[0]['city'] == 'NYC'
        assert source3.get('by_id').iloc[0]['id'] == 2
    finally:
        os.chdir(original_cwd)


def test_table_params_serialization(sample_csv_files):
    """Test that params survive to_spec/from_spec roundtrip."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])
        original = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'filtered': 'SELECT * FROM customers WHERE id = ?'
            },
            table_params={'filtered': [2]}
        )

        # Get the original result before serialization
        original_result = original.get('filtered')
        assert len(original_result) == 1
        assert original_result.iloc[0]['id'] == 2

        # Serialize to spec
        spec = original.to_spec()
        
        # For ephemeral in-memory sources with CSV files, we need to stay in the same directory
        # or use absolute paths for the restored source to find the files
        spec['tables'] = {
            'customers': files['customers'],
            'filtered': 'SELECT * FROM customers WHERE id = ?'
        }
        
        restored = DuckDBSource.from_spec(spec)
        restored_result = restored.get('filtered')
        
        pd.testing.assert_frame_equal(original_result, restored_result)
        assert len(restored_result) == 1
        assert restored_result.iloc[0]['id'] == 2
    finally:
        os.chdir(original_cwd)


def test_create_sql_expr_source_preserves_all_existing_tables(sample_csv_files):
    """Test that create_sql_expr_source preserves ALL existing tables (upsert behavior)."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        # Create initial source with multiple tables
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv',
                'existing_view': 'SELECT * FROM customers WHERE id > 1'
            }
        )

        # Verify initial state
        assert set(source.get_tables()) == {'customers', 'orders', 'existing_view'}

        # Create new source with additional tables - should preserve ALL existing ones
        new_tables = {
            'new_table': 'SELECT * FROM orders WHERE total > 200'
        }

        new_source = source.create_sql_expr_source(new_tables)

        # ALL tables should be present: original + new
        expected_tables = {'customers', 'orders', 'existing_view', 'new_table'}
        assert set(new_source.get_tables()) == expected_tables

        # Verify all tables are accessible and work correctly
        assert len(new_source.get('customers')) == 3
        assert len(new_source.get('orders')) == 3
        assert len(new_source.get('existing_view')) == 2  # id > 1
        assert len(new_source.get('new_table')) == 2  # total > 200

    finally:
        os.chdir(original_cwd)


def test_create_sql_expr_source_upserts_existing_tables(sample_csv_files):
    """Test that create_sql_expr_source overwrites tables with same name (upsert behavior)."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        # Create initial source
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv',
                'filtered_customers': 'SELECT * FROM customers WHERE id = 1'  # Original: just Alice
            }
        )

        # Verify initial state
        initial_result = source.get('filtered_customers')
        assert len(initial_result) == 1
        assert initial_result.iloc[0]['name'] == 'Alice'

        # Create new source that OVERWRITES filtered_customers but keeps others
        new_tables = {
            'filtered_customers': 'SELECT * FROM customers WHERE id > 1',  # New: Bob and Charlie
            'new_table': 'SELECT * FROM orders WHERE total > 200'
        }

        new_source = source.create_sql_expr_source(new_tables)

        # Should have all tables
        expected_tables = {'customers', 'orders', 'filtered_customers', 'new_table'}
        assert set(new_source.get_tables()) == expected_tables

        # filtered_customers should have NEW definition (id > 1, not id = 1)
        updated_result = new_source.get('filtered_customers')
        assert len(updated_result) == 2
        assert set(updated_result['name']) == {'Bob', 'Charlie'}

        # Original tables should still work
        assert len(new_source.get('customers')) == 3
        assert len(new_source.get('orders')) == 3
        assert len(new_source.get('new_table')) == 2

    finally:
        os.chdir(original_cwd)


def test_create_sql_expr_source_new_connection_only_new_tables(sample_csv_files):
    """Test that create_sql_expr_source with new connection only includes new tables."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        # Create initial source with multiple tables
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv',
                'existing_view': 'SELECT * FROM customers WHERE id > 1'
            }
        )

        # Verify initial state
        assert set(source.get_tables()) == {'customers', 'orders', 'existing_view'}

        # Create new source with a DIFFERENT URI - should NOT preserve old tables
        new_tables = {
            'products': files['customers']  # Reusing customers.csv as "products"
        }

        new_source = source.create_sql_expr_source(new_tables, uri=':memory:')

        # Should ONLY have the new table, not the old ones
        assert set(new_source.get_tables()) == {'products'}

        # Old tables should NOT be accessible
        assert 'customers' not in new_source.get_tables()
        assert 'orders' not in new_source.get_tables()
        assert 'existing_view' not in new_source.get_tables()

    finally:
        os.chdir(original_cwd)


def test_create_sql_expr_source_new_connection_includes_file_dependencies(sample_csv_files):
    """Test that new connection includes file-based tables referenced in SQL."""
    files = sample_csv_files
    original_cwd = os.getcwd()

    try:
        os.chdir(files['dir'])

        # Create initial source with file-based tables
        source = DuckDBSource(
            uri=':memory:',
            tables={
                'customers': 'customers.csv',
                'orders': 'orders.csv',
            }
        )

        # Create new source with new connection that REFERENCES file-based tables
        new_tables = {
            'summary': 'SELECT c.name, COUNT(o.id) as order_count FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.name'
        }

        new_source = source.create_sql_expr_source(new_tables, uri=':memory:')

        # Should have the new table AND the file-based dependencies
        expected_tables = {'summary', 'customers', 'orders'}
        assert set(new_source.get_tables()) == expected_tables

        # The summary query should actually work (dependencies are present)
        result = new_source.get('summary')
        assert len(result) == 3  # 3 customers
        assert 'order_count' in result.columns

        # File-based tables should be accessible
        assert len(new_source.get('customers')) == 3
        assert len(new_source.get('orders')) == 3

    finally:
        os.chdir(original_cwd)


def test_create_sql_expr_source_with_list_tables():
    """Test that create_sql_expr_source works when self.tables is a list."""
    # Create an in-memory source with actual data
    df = pd.DataFrame({
        'A': [0, 1, 2, 3, 4],
        'B': [0, 0, 1, 1, 1],
        'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5']
    })
    for col in df.select_dtypes(include=['string']).columns:
        df[col] = df[col].astype(object)
    
    # Use from_df which creates dict-based tables, then manually convert to list
    source = DuckDBSource.from_df({'test_table': df})
    # Simulate a list-based source (though unusual in practice)
    source.tables = ['test_table']  # Override with list
    
    # Verify it's a list
    assert isinstance(source.tables, list)
    
    # Create new source with SQL expressions
    new_tables = {
        'filtered': 'SELECT * FROM test_table WHERE A > 2'
    }
    
    new_source = source.create_sql_expr_source(new_tables)
    
    # Should only have the new table (list-based tables don't get preserved)
    assert 'filtered' in new_source.get_tables()
    assert isinstance(new_source.tables, dict)
    assert 'filtered' in new_source.tables
    
    # The new table should work
    result = new_source.get('filtered')
    assert len(result) == 2  # A values 3 and 4
    assert all(result['A'] > 2)


def test_create_sql_expr_source_reuse_connection_with_list_tables():
    """Test that reusing connection with list tables preserves them as dict entries."""
    # Create an in-memory source with actual data
    df = pd.DataFrame({
        'A': [0, 1, 2, 3, 4],
        'B': [0, 0, 1, 1, 1],
        'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5']
    })
    for col in df.select_dtypes(include=['string']).columns:
        df[col] = df[col].astype(object)

    source = DuckDBSource.from_df({'test_table': df})
    # Simulate a list-based source
    source.tables = ['test_table']  # Override with list

    # Create new source reusing connection
    new_tables = {
        'filtered': 'SELECT * FROM test_table WHERE A > 2'
    }

    # No uri or initializers provided - reusing connection
    new_source = source.create_sql_expr_source(new_tables)

    # Should have both the original table (converted from list) and the new table
    assert set(new_source.get_tables()) == {'test_table', 'filtered'}

    # Original table should still be queryable
    result = new_source.get('test_table')
    assert len(result) == 5

    # New filtered table should work
    filtered_result = new_source.get('filtered')
    assert len(filtered_result) == 2  # A values 3 and 4


def test_read_only_create_sql_expr_source_expands_tables(duckdb_file_source):
    """
    Test that create_sql_expr_source expands tables when not in read-only mode.

    This verifies that with read_only=False, new tables can be created and
    the existing tables are preserved.
    """
    source = duckdb_file_source

    # Verify initial state - should have the 'test' table
    initial_tables = source.get_tables()
    assert 'test' in initial_tables

    # Create new source with additional tables using create_sql_expr_source
    new_tables = {
        'filtered_test': 'SELECT * FROM test WHERE id > 1',
        'count_test': 'SELECT COUNT(*) as cnt FROM test'
    }

    new_source = source.create_sql_expr_source(new_tables)

    # Verify tables expanded - should have original + new tables
    expanded_tables = new_source.get_tables()
    assert 'test' in expanded_tables
    assert 'filtered_test' in expanded_tables
    assert 'count_test' in expanded_tables

    # Verify the new tables work correctly
    filtered_result = new_source.get('filtered_test')
    assert len(filtered_result) == 2  # id > 1 means Bob and Charlie

    count_result = new_source.get('count_test')
    assert count_result.iloc[0]['cnt'] == 3  # 3 total rows

    # Ensure read only
    assert source.read_only
    assert new_source.read_only

    # Ensure new tables did NOT modify the original source
    assert source.execute("SHOW TABLES").shape[0] == 1  # Only 'test' table in original source
