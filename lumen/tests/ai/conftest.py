"""
Fixtures for thread-safe DuckDB testing.

This module provides pytest fixtures that ensure DuckDB connections
are properly isolated between tests, preventing thread-safety issues
when tests run in parallel.
"""
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
import pytest


@pytest.fixture(scope="function")
def duckdb_connection():
    """
    Create an isolated DuckDB connection for each test.
    
    This fixture ensures that each test gets its own DuckDB connection,
    preventing thread-safety issues when pytest runs tests in parallel.
    """
    # Create a new in-memory connection for this test
    con = duckdb.connect(':memory:')
    
    yield con
    
    # Cleanup: close the connection after the test
    try:
        con.close()
    except:
        pass  # Connection might already be closed


@pytest.fixture(scope="function") 
def duckdb_source_factory(duckdb_connection):
    """
    Factory fixture for creating DuckDBSource instances with isolated connections.
    
    Usage:
        def test_something(duckdb_source_factory):
            source = duckdb_source_factory(
                tables={"table1": "path/to/file.csv"}
            )
    """
    def _create_source(**kwargs):
        from lumen.sources.duckdb import DuckDBSource

        # If uri is not specified, use our isolated connection
        if 'uri' not in kwargs:
            kwargs['uri'] = ':memory:'
        
        # Create the source with an isolated connection
        source = DuckDBSource(**kwargs)
        
        # Replace the connection with our isolated one if needed
        if hasattr(source, '_connection'):
            source._connection = duckdb_connection
            
        return source
    
    return _create_source


@contextmanager
def safe_duckdb_execute(source, max_retries=3):
    """
    Context manager for safely executing DuckDB queries with retry logic.
    
    Usage:
        with safe_duckdb_execute(source) as execute:
            result = execute("SELECT * FROM table")
    """
    import time
    
    def execute_with_retry(query):
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return source.execute(query)
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check if it's a connection error
                if any(x in error_msg for x in ['pending query result', 'closed', 'invalid']):
                    # Try to recreate the connection
                    try:
                        # Create new connection
                        new_con = duckdb.connect(':memory:')
                        
                        # Re-register tables if they exist
                        if hasattr(source, '_tables'):
                            for table_name, table_path in source._tables.items():
                                if Path(table_path).exists():
                                    df = pd.read_csv(table_path)
                                    new_con.register(table_name, df)
                        
                        # Replace the connection
                        source._connection = new_con
                        
                        # Small delay before retry
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    except:
                        pass
                
                # If it's not a connection error or we couldn't fix it, raise
                if attempt == max_retries - 1:
                    raise last_error
        
        raise last_error
    
    yield execute_with_retry


# Alternative: Mark tests that must run serially
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", 
        "serial: mark test to run serially (not in parallel with other tests)"
    )
