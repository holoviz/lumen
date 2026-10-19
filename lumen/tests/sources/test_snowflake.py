import asyncio
import contextlib
import itertools
import threading
import time

from unittest.mock import Mock, patch

import pandas as pd
import pytest

try:
    import snowflake.connector  # noqa: F401

    from snowflake.connector.constants import QueryStatus

    from lumen.sources.base import QueryTimeoutError
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


class FakeSnowflakeConnection:
    """A connection that hands out a fresh recorded cursor per statement."""

    def __init__(self):
        self.cursors = []
        self.frames = {}
        self.status = QueryStatus.SUCCESS
        self.fetch_delay = 0.0
        self.fetch_error = None
        self.fetch_gate = None
        self.fetching = False
        self.cancel_error = None
        self.submit_gate = None
        self.submit_error = None
        self.polled = threading.Event()
        self._submissions = itertools.count(1)

    def cursor(self):
        cursor = Mock()
        cursor.sfqid = None
        cursor.frame = None
        cursor.sql = None

        def execute(sql, *args, **kwargs):
            cursor.sql = sql
            if self.cancel_error is not None and "SYSTEM$CANCEL_QUERY" in sql:
                raise self.cancel_error
            cursor.frame = pd.DataFrame({"sql": [sql]})
            return cursor

        def execute_async(sql, *args, **kwargs):
            cursor.sql = sql
            if self.submit_gate is not None:
                entered, release = self.submit_gate
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("submit was never released")
            if self.submit_error is not None:
                raise self.submit_error
            cursor.sfqid = f"query-{next(self._submissions)}"
            self.frames[cursor.sfqid] = pd.DataFrame({"sql": [sql]})

        def get_results_from_sfqid(query_id):
            cursor.frame = self.frames[query_id]

        def fetch_pandas_all():
            if self.fetch_gate is not None:
                gated_sql, entered, release = self.fetch_gate
                if cursor.sql == gated_sql:
                    entered.set()
                    if not release.wait(10):
                        raise RuntimeError("fetch was never released")
            if self.fetch_delay:
                self.fetching = True
                try:
                    time.sleep(self.fetch_delay)
                finally:
                    self.fetching = False
            if self.fetch_error is not None:
                raise self.fetch_error
            return cursor.frame

        cursor.execute.side_effect = execute
        cursor.execute_async.side_effect = execute_async
        cursor.get_results_from_sfqid.side_effect = get_results_from_sfqid
        cursor.fetch_pandas_all.side_effect = fetch_pandas_all
        self.cursors.append(cursor)
        return cursor

    def get_query_status(self, query_id):
        self.polled.set()
        return self.status


@pytest.fixture
def snowflake_cursor_factory():
    """A Snowflake connection whose cursors are recorded per query."""
    return FakeSnowflakeConnection()


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

    expected_tables = ['DB.SCHEMA.TABLE3']
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


async def test_execute_with_timeout_cancels_the_query(snowflake_cursor_factory):
    snowflake_cursor_factory.status = QueryStatus.RUNNING
    source = SnowflakeSource(conn=snowflake_cursor_factory)

    with pytest.raises(QueryTimeoutError):
        await source.execute_with_timeout("SELECT 1", 0.1)

    submit_cursor, cancel_cursor = snowflake_cursor_factory.cursors
    submit_cursor.execute_async.assert_called_once_with("SELECT 1")
    cancel_cursor.execute.assert_called_once_with("SELECT SYSTEM$CANCEL_QUERY('query-1')")
    assert cancel_cursor is not submit_cursor
    submit_cursor.close.assert_called_once()
    cancel_cursor.close.assert_called_once()


async def test_cancel_failure_still_raises_original_cancellation(snowflake_cursor_factory):
    snowflake_cursor_factory.status = QueryStatus.RUNNING
    snowflake_cursor_factory.cancel_error = RuntimeError("cancel failed")
    source = SnowflakeSource(conn=snowflake_cursor_factory)

    query = asyncio.create_task(source.execute_async("SELECT 1"))
    try:
        assert await asyncio.to_thread(snowflake_cursor_factory.polled.wait, 5)
        query.cancel()
        with pytest.raises(asyncio.CancelledError):
            await query
    finally:
        if not query.done():
            query.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await query

    submit_cursor, cancel_cursor = snowflake_cursor_factory.cursors
    cancel_cursor.execute.assert_called_once_with("SELECT SYSTEM$CANCEL_QUERY('query-1')")
    assert cancel_cursor is not submit_cursor
    submit_cursor.close.assert_called_once()
    cancel_cursor.close.assert_called_once()


async def test_cancellation_during_submit_cancels_the_server_query(snowflake_cursor_factory):
    entered = threading.Event()
    release = threading.Event()
    snowflake_cursor_factory.submit_gate = (entered, release)
    source = SnowflakeSource(conn=snowflake_cursor_factory)

    query = asyncio.create_task(source.execute_async("SELECT 1"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        query.cancel()
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await query

    submit_cursor, cancel_cursor = snowflake_cursor_factory.cursors
    submit_cursor.execute_async.assert_called_once_with("SELECT 1")
    cancel_cursor.execute.assert_called_once_with("SELECT SYSTEM$CANCEL_QUERY('query-1')")
    submit_cursor.close.assert_called_once()
    cancel_cursor.close.assert_called_once()


async def test_submit_failure_during_cancellation_keeps_original_cancellation(snowflake_cursor_factory):
    entered = threading.Event()
    release = threading.Event()
    snowflake_cursor_factory.submit_gate = (entered, release)
    snowflake_cursor_factory.submit_error = RuntimeError("submit failed")
    source = SnowflakeSource(conn=snowflake_cursor_factory)

    query = asyncio.create_task(source.execute_async("SELECT 1"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        query.cancel()
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await query

    [submit_cursor] = snowflake_cursor_factory.cursors
    submit_cursor.execute_async.assert_called_once_with("SELECT 1")
    submit_cursor.execute.assert_not_called()
    submit_cursor.close.assert_called_once()


async def test_threaded_execute_and_async_query_use_their_own_cursors(snowflake_cursor_factory):
    entered = threading.Event()
    release = threading.Event()
    thread_sql = "SELECT 'thread' AS origin"
    snowflake_cursor_factory.fetch_gate = (thread_sql, entered, release)
    source = SnowflakeSource(conn=snowflake_cursor_factory)

    thread_query = asyncio.create_task(asyncio.to_thread(source.execute, thread_sql))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        async_df = await source.execute_async("SELECT 'async' AS origin")
    finally:
        release.set()
    thread_df = await thread_query

    assert thread_df["sql"].iloc[0] == thread_sql
    assert async_df["sql"].iloc[0] == "SELECT 'async' AS origin"
    assert len(snowflake_cursor_factory.cursors) == 3
    assert all(cursor.close.called for cursor in snowflake_cursor_factory.cursors)


async def test_gathered_async_queries_use_their_own_cursors(snowflake_cursor_factory):
    source = SnowflakeSource(conn=snowflake_cursor_factory)

    first, second = await asyncio.gather(
        source.execute_async("SELECT 1 AS value"),
        source.execute_async("SELECT 2 AS value"),
    )

    assert first["sql"].iloc[0] == "SELECT 1 AS value"
    assert second["sql"].iloc[0] == "SELECT 2 AS value"
    assert len(snowflake_cursor_factory.cursors) == 4
    assert all(cursor.close.called for cursor in snowflake_cursor_factory.cursors)


async def test_slow_fetch_does_not_block_the_event_loop(snowflake_cursor_factory):
    snowflake_cursor_factory.fetch_delay = 0.5
    source = SnowflakeSource(conn=snowflake_cursor_factory)
    ticks_during_fetch = 0

    async def ticker():
        nonlocal ticks_during_fetch
        while True:
            if snowflake_cursor_factory.fetching:
                ticks_during_fetch += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    try:
        await source.execute_async("SELECT 1 AS value")
    finally:
        ticker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker_task

    assert ticks_during_fetch >= 5


async def test_cursor_is_closed_when_fetch_raises(snowflake_cursor_factory):
    snowflake_cursor_factory.fetch_error = RuntimeError("fetch failed")
    source = SnowflakeSource(conn=snowflake_cursor_factory)

    with pytest.raises(RuntimeError, match="fetch failed"):
        await source.execute_async("SELECT 1")

    assert len(snowflake_cursor_factory.cursors) == 2
    assert all(cursor.close.called for cursor in snowflake_cursor_factory.cursors)


def test_schema_timeout_uses_separate_closed_cursors(snowflake_cursor_factory):
    source = SnowflakeSource(conn=snowflake_cursor_factory)

    with source._timeout_context(5):
        pass

    set_cursor, unset_cursor = snowflake_cursor_factory.cursors
    set_cursor.execute.assert_called_once_with(
        "ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 5"
    )
    unset_cursor.execute.assert_called_once_with(
        "ALTER SESSION UNSET STATEMENT_TIMEOUT_IN_SECONDS"
    )
    assert set_cursor is not unset_cursor
    assert set_cursor.close.called
    assert unset_cursor.close.called
