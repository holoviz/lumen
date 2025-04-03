from unittest.mock import Mock, patch

import pandas as pd
import pytest

try:
    from lumen.sources.snowflake import SnowflakeSource
    pytestmark = pytest.mark.xdist_group("snowflake")
except ImportError:
    pytestmark = pytest.mark.skip(reason="Snowflake is not installed")


@pytest.fixture
def mock_snowflake_connection():
    """Fixture to create a mock Snowflake connection and cursor."""
    # Sample tables data
    tables_metadata = pd.DataFrame({
        'TABLE_NAME': ['CUSTOMERS', 'ORDERS', 'PRODUCTS', 'SALES', 'DEMOGRAPHICS', 'STATISTICS'],
        'TABLE_SCHEMA': ['PUBLIC', 'PUBLIC', 'ANALYTICS', 'ANALYTICS', 'TPCDS_SF100TCL', 'TPCDS_SF100TCL']
    })

    # Create mock cursor
    mock_cursor = Mock()
    mock_cursor.execute.return_value.fetch_pandas_all.return_value = tables_metadata

    # Create mock connection
    mock_conn = Mock()
    mock_conn.cursor.return_value = mock_cursor

    # Create patcher
    with patch('snowflake.connector.connect', return_value=mock_conn) as mock_connect:
        yield mock_connect, mock_conn, mock_cursor


def test_init(mock_snowflake_connection):
    """Test initialization of SnowflakeSource."""
    mock_connect, _, _ = mock_snowflake_connection

    SnowflakeSource(
        account='test_account',
        user='test_user',
        password='test_password',
        database='TEST_DB'
    )

    # Verify snowflake.connector.connect was called with the right parameters
    mock_connect.assert_called_once()
    call_kwargs = mock_connect.call_args[1]
    assert call_kwargs['account'] == 'test_account'
    assert call_kwargs['user'] == 'test_user'
    assert call_kwargs['password'] == 'test_password'
    assert call_kwargs['database'] == 'TEST_DB'


def test_get_tables_no_exclusions(mock_snowflake_connection):
    """Test get_tables method with no excluded tables."""
    _, _, mock_cursor = mock_snowflake_connection

    source = SnowflakeSource(database='TEST_DB')
    tables = source.get_tables()

    # Verify the mock was called with the correct SQL
    mock_cursor.execute.assert_called_with(
        'SELECT TABLE_NAME, TABLE_SCHEMA FROM TEST_DB.INFORMATION_SCHEMA.TABLES;'
    )

    # Check the returned table list
    expected_tables = [
        'TEST_DB.PUBLIC.CUSTOMERS',
        'TEST_DB.PUBLIC.ORDERS',
        'TEST_DB.ANALYTICS.PRODUCTS',
        'TEST_DB.ANALYTICS.SALES',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_with_exclusions_full_qualified(mock_snowflake_connection):
    """Test get_tables with fully qualified excluded tables."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['TEST_DB.PUBLIC.CUSTOMERS', 'TEST_DB.ANALYTICS.SALES']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.PUBLIC.ORDERS',
        'TEST_DB.ANALYTICS.PRODUCTS',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_with_exclusions_schema_qualified(mock_snowflake_connection):
    """Test get_tables with schema.table excluded tables."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['PUBLIC.ORDERS', 'ANALYTICS.PRODUCTS']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.PUBLIC.CUSTOMERS',
        'TEST_DB.ANALYTICS.SALES',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_with_exclusions_table_name_only(mock_snowflake_connection):
    """Test get_tables with just table names."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['CUSTOMERS', 'PRODUCTS']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.PUBLIC.ORDERS',
        'TEST_DB.ANALYTICS.SALES',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_with_mixed_exclusions(mock_snowflake_connection):
    """Test get_tables with mixed exclusion formats."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['TEST_DB.PUBLIC.CUSTOMERS', 'ANALYTICS.PRODUCTS', 'ORDERS']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.ANALYTICS.SALES',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_explicit_tables_with_exclusions(mock_snowflake_connection):
    """Test get_tables with explicitly defined tables and exclusions."""
    tables_list = [
        'TABLE1',
        'SCHEMA1.TABLE2',
        'DB1.SCHEMA2.TABLE3',
        'DB1.SCHEMA1.TABLE4'
    ]

    source = SnowflakeSource(
        tables=tables_list,
        excluded_tables=['TABLE1', 'SCHEMA1.TABLE2']
    )

    tables = source.get_tables()

    expected_tables = [
        'DB1.SCHEMA2.TABLE3',
        'DB1.SCHEMA1.TABLE4'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_dict_with_exclusions(mock_snowflake_connection):
    """Test get_tables with dict tables and exclusions."""
    tables_dict = {
        'table1': 'SQL1',
        'schema.table2': 'SQL2',
        'db.schema.table3': 'SQL3'
    }

    source = SnowflakeSource(
        tables=tables_dict,
        excluded_tables=['table1', 'schema.table2']
    )

    tables = source.get_tables()

    expected_tables = ['db.schema.table3']
    assert set(tables) == set(expected_tables)


def test_get_tables_with_schema_wildcard_exclusions(mock_snowflake_connection):
    """Test get_tables with schema wildcard exclusions (schema.*)."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['TPCDS_SF100TCL.*']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.PUBLIC.CUSTOMERS',
        'TEST_DB.PUBLIC.ORDERS',
        'TEST_DB.ANALYTICS.PRODUCTS',
        'TEST_DB.ANALYTICS.SALES'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_with_fully_qualified_schema_wildcard(mock_snowflake_connection):
    """Test get_tables with fully qualified schema wildcard exclusions (database.schema.*)."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['TEST_DB.ANALYTICS.*']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.PUBLIC.CUSTOMERS',
        'TEST_DB.PUBLIC.ORDERS',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)

def test_get_tables_with_empty_string_exclusions(mock_snowflake_connection):
    """Test that empty strings in excluded_tables are properly ignored."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['', 'CUSTOMERS', '']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.PUBLIC.ORDERS',
        'TEST_DB.ANALYTICS.PRODUCTS',
        'TEST_DB.ANALYTICS.SALES',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_with_complex_wildcard_patterns(mock_snowflake_connection):
    """Test exclusions with more complex wildcard patterns."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['*CUSTOMERS', 'ORDER*', '*STAT*']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.ANALYTICS.PRODUCTS',
        'TEST_DB.ANALYTICS.SALES',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_case_sensitivity(mock_snowflake_connection):
    """Test case sensitivity in excluded_tables patterns."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['customers', 'public.orders']  # lowercase patterns
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.ANALYTICS.PRODUCTS',
        'TEST_DB.ANALYTICS.SALES',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_with_none_in_exclusions(mock_snowflake_connection):
    """Test that None values in excluded_tables are properly handled."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['CUSTOMERS', None, 'ORDERS']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.ANALYTICS.PRODUCTS',
        'TEST_DB.ANALYTICS.SALES',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_with_overlapping_exclusions(mock_snowflake_connection):
    """Test with overlapping exclusion patterns."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['PUBLIC.*', 'TEST_DB.PUBLIC.CUSTOMERS', 'ORDERS']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.ANALYTICS.PRODUCTS',
        'TEST_DB.ANALYTICS.SALES',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)


def test_get_tables_with_nested_wildcard_exclusions(mock_snowflake_connection):
    """Test get_tables with nested wildcard patterns."""
    source = SnowflakeSource(
        database='TEST_DB',
        excluded_tables=['TEST_DB.*.CUST*']
    )

    tables = source.get_tables()

    expected_tables = [
        'TEST_DB.PUBLIC.ORDERS',
        'TEST_DB.ANALYTICS.PRODUCTS',
        'TEST_DB.ANALYTICS.SALES',
        'TEST_DB.TPCDS_SF100TCL.DEMOGRAPHICS',
        'TEST_DB.TPCDS_SF100TCL.STATISTICS'
    ]
    assert set(tables) == set(expected_tables)
