"""Tests for database file detection in UI._resolve_data"""
import tempfile

from pathlib import Path

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.ui import ExplorerUI
from lumen.sources.duckdb import DuckDBSource


def test_resolve_data_duckdb_connection_string():
    """Test that duckdb:// connection strings are correctly detected."""
    # Create a temporary DuckDB file (delete=True so file is removed but name is available)
    with tempfile.NamedTemporaryFile(suffix='.db', delete=True) as f:
        temp_path = f.name
    
    try:
        # Create a DuckDB database
        import duckdb
        conn = duckdb.connect(temp_path)
        conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
        conn.close()
        
        # Test with duckdb:// connection string
        sources = ExplorerUI._resolve_data(f'duckdb:///{temp_path}')
        
        assert len(sources) == 1
        assert isinstance(sources[0], DuckDBSource)
        # URI should be absolute path
        assert sources[0].uri == str(Path(temp_path).absolute())
        # Name should be just the filename
        assert sources[0].name == Path(temp_path).name
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_resolve_data_duckdb_file_magic_bytes():
    """Test that .db files with DuckDB magic bytes are detected as DuckDB."""
    # Create a temporary DuckDB file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=True) as f:
        temp_path = f.name
    
    try:
        # Create a DuckDB database
        import duckdb
        conn = duckdb.connect(temp_path)
        conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
        conn.close()
        
        # Test with just the file path (no protocol)
        sources = ExplorerUI._resolve_data(temp_path)
        
        assert len(sources) == 1
        assert isinstance(sources[0], DuckDBSource)
        # URI should be absolute path
        assert sources[0].uri == str(Path(temp_path).absolute())
        # Name should be just the filename
        assert sources[0].name == Path(temp_path).name
        
        # Verify the file actually has DUCK in header
        with open(temp_path, 'rb') as f:
            header = f.read(16)
            assert b'DUCK' in header
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_resolve_data_sqlite_file():
    """Test that .db files without DuckDB magic bytes are detected as SQLite."""
    import sqlite3

    # Create a temporary SQLite file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_path = f.name
    
    try:
        # Create a SQLite database
        conn = sqlite3.connect(temp_path)
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
        conn.commit()
        conn.close()
        
        # Test with just the file path (no protocol)
        sources = ExplorerUI._resolve_data(temp_path)
        
        assert len(sources) == 1
        # Should be SQLAlchemySource for SQLite
        from lumen.sources.sqlalchemy import SQLAlchemySource
        assert isinstance(sources[0], SQLAlchemySource)
        abs_path = str(Path(temp_path).absolute())
        assert f'sqlite:///{abs_path}' in sources[0].url or sources[0].url == f'sqlite:///{abs_path}'
        # Name should be just the filename
        assert sources[0].name == Path(temp_path).name
        
        # Verify the file does NOT have DUCK in header
        with open(temp_path, 'rb') as f:
            header = f.read(16)
            assert b'DUCK' not in header
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_resolve_data_sqlite_connection_string():
    """Test that sqlite:// connection strings are correctly detected."""
    import sqlite3

    # Create a temporary SQLite file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_path = f.name
    
    try:
        # Create a SQLite database
        conn = sqlite3.connect(temp_path)
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
        conn.commit()
        conn.close()
        
        # Test with sqlite:// connection string
        sources = ExplorerUI._resolve_data(f'sqlite:///{temp_path}')
        
        assert len(sources) == 1
        from lumen.sources.sqlalchemy import SQLAlchemySource
        assert isinstance(sources[0], SQLAlchemySource)
        assert sources[0].url == f'sqlite:///{temp_path}'
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_resolve_data_nonexistent_db_file():
    """Test that nonexistent .db files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        ExplorerUI._resolve_data('/nonexistent/path/file.db')
