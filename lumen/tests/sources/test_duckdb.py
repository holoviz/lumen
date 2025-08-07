import datetime as dt
import os
import tempfile

import pandas as pd
import pytest

from lumen.transforms.sql import SQLGroupBy

try:
    from lumen.sources.duckdb import DuckDBSource
    pytestmark = pytest.mark.xdist_group("duckdb")
except ImportError:
    pytestmark = pytest.mark.skip(reason="Duckdb is not installed")



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
    source._connection.from_df(mixed_df).to_view('mixed')
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
    pd.testing.assert_frame_equal(filtered, expected_filtered_df.reset_index(drop=True))


@pytest.mark.flaky(reruns=3)
def test_duckdb_transforms(duckdb_source, source_tables):
    df_test_sql = source_tables['test_sql']
    transforms = [SQLGroupBy(by=['B'], aggregates={'SUM': 'A'})]
    transformed = duckdb_source.get('test_sql', sql_transforms=transforms)
    expected = df_test_sql.groupby('B')['A'].sum().reset_index()
    pd.testing.assert_frame_equal(transformed, expected)


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
    for col in df.columns:
        assert (df[col]==mixed_df[col]).all()


def test_duckdb_source_mirrors_source(duckdb_source):
    mirrored = DuckDBSource(uri=':memory:', mirrors={'mixed': (duckdb_source, 'test_sql')})
    pd.testing.assert_frame_equal(duckdb_source.get('test_sql'), mirrored.get('mixed'))


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
    pd.testing.assert_frame_equal(result, df)


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
    source = DuckDBSource.from_spec(spec).tables
    assert "mirrors" not in source.tables
    assert source.mirrors == {}
