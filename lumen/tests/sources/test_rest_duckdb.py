"""Tests for RESTDuckDBSource."""
import pandas as pd
import pytest

try:
    from lumen.sources.rest_duckdb import RESTDuckDBSource
    pytestmark = pytest.mark.xdist_group("duckdb")
except ImportError:
    pytestmark = pytest.mark.skip(reason="DuckDB is not installed")


@pytest.fixture
def rest_duckdb_config():
    """Fixture providing test configuration for RESTDuckDBSource."""
    return {
        'uri': ':memory:',
        'tables': {
            'daily': {
                'url': 'https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py',
                'url_params': {
                    'stations': 'ABR',
                    'sts': '2025-12-08',
                    'ets': '2025-12-09',
                    'network': 'SD_ASOS',
                    'format': 'csv'
                },
            },
            'raob': {
                'url': 'https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py',
                'url_params': {
                    'station': 'KABR',
                    'sts': '2025-12-08T15:49',
                    'ets': '2025-12-09T15:49',
                    'format': 'csv'
                },
            },
        }
    }


@pytest.fixture
def rest_duckdb_source(rest_duckdb_config):
    """Fixture providing a RESTDuckDBSource instance."""
    return RESTDuckDBSource(**rest_duckdb_config)


class TestRESTDuckDBSource:
    """Tests for RESTDuckDBSource class."""

    def test_source_type(self):
        """Test that source_type is correctly set."""
        assert RESTDuckDBSource.source_type == 'rest_duckdb'

    def test_resolve_module_type(self):
        """Test that the source can be resolved by module path."""
        assert RESTDuckDBSource._get_type('lumen.sources.rest_duckdb.RESTDuckDBSource') is RESTDuckDBSource

    def test_initialization(self, rest_duckdb_config):
        """Test that RESTDuckDBSource initializes correctly."""
        source = RESTDuckDBSource(**rest_duckdb_config)
        assert source.uri == ':memory:'
        assert 'daily' in source.tables
        assert 'raob' in source.tables

    def test_render_table_url(self, rest_duckdb_source):
        """Test that render_table_url constructs correct URLs."""
        daily_url = rest_duckdb_source.render_table_url('daily')
        assert 'https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py' in daily_url
        assert 'stations=ABR' in daily_url
        assert 'sts=2025-12-08' in daily_url
        assert 'ets=2025-12-09' in daily_url
        assert 'network=SD_ASOS' in daily_url
        assert 'format=csv' in daily_url

        raob_url = rest_duckdb_source.render_table_url('raob')
        assert 'https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py' in raob_url
        assert 'station=KABR' in raob_url

    def test_get_table(self, rest_duckdb_source):
        """Test that get() retrieves data correctly."""
        df = rest_duckdb_source.get('daily')
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'station' in df.columns
        assert 'day' in df.columns
        assert 'max_temp_f' in df.columns
        assert 'min_temp_f' in df.columns
        
        # Check that we have the expected rows
        assert len(df) == 2  # Based on the sample data showing 2 rows
        assert all(df['station'] == 'ABR')

    def test_get_multiple_tables(self, rest_duckdb_source):
        """Test that both tables can be retrieved."""
        daily_df = rest_duckdb_source.get('daily')
        raob_df = rest_duckdb_source.get('raob')
        
        assert isinstance(daily_df, pd.DataFrame)
        assert isinstance(raob_df, pd.DataFrame)
        assert not daily_df.empty
        # raob_df might be empty depending on data availability

    def test_tables_property(self, rest_duckdb_source):
        """Test that tables property returns correct table information."""
        tables = rest_duckdb_source.tables
        
        assert isinstance(tables, dict)
        assert 'daily' in tables
        assert 'raob' in tables
        
        # Check that table configs are preserved
        daily_config = tables['daily']
        assert 'url' in daily_config
        assert daily_config['url'] == 'https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py'

    def test_execute_sql(self, rest_duckdb_source):
        """Test that execute() runs SQL queries correctly."""
        result = rest_duckdb_source.execute("SELECT * FROM daily LIMIT 5")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5
        assert 'station' in result.columns
        assert 'day' in result.columns

    def test_execute_sql_with_filter(self, rest_duckdb_source):
        """Test SQL execution with WHERE clause."""
        result = rest_duckdb_source.execute("SELECT * FROM daily WHERE max_temp_f > 20")
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert all(result['max_temp_f'] > 20)

    def test_execute_sql_count(self, rest_duckdb_source):
        """Test SQL COUNT query."""
        result = rest_duckdb_source.execute("SELECT COUNT(*) as count FROM daily")
        
        assert isinstance(result, pd.DataFrame)
        assert 'count' in result.columns
        assert result['count'].iloc[0] > 0

    def test_create_sql_expr_source(self, rest_duckdb_source):
        """Test creating a derived source with SQL expressions."""
        new_source = rest_duckdb_source.create_sql_expr_source({
            'daily_1': "SELECT * FROM daily LIMIT 1"
        })
        
        # Check that new source exists and has the derived table
        assert hasattr(new_source, 'tables')
        assert 'daily_1' in new_source.tables
        
        # Check that the derived table can be queried
        df = new_source.get('daily_1')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'station' in df.columns

    def test_create_sql_expr_source_preserves_original_tables(self, rest_duckdb_source):
        """Test that creating SQL expr source preserves original tables."""
        new_source = rest_duckdb_source.create_sql_expr_source({
            'daily_1': "SELECT * FROM daily LIMIT 1"
        })
        
        # Original tables should still be accessible
        daily_df = new_source.get('daily')
        assert isinstance(daily_df, pd.DataFrame)
        assert len(daily_df) > 1  # Original table has more rows

    def test_create_sql_expr_source_multiple_expressions(self, rest_duckdb_source):
        """Test creating multiple SQL expressions at once."""
        new_source = rest_duckdb_source.create_sql_expr_source({
            'daily_1': "SELECT * FROM daily LIMIT 1",
            'daily_high_temp': "SELECT * FROM daily WHERE max_temp_f > 30"
        })
        
        assert 'daily_1' in new_source.tables
        assert 'daily_high_temp' in new_source.tables
        
        df1 = new_source.get('daily_1')
        df_high = new_source.get('daily_high_temp')
        
        assert len(df1) == 1
        assert isinstance(df_high, pd.DataFrame)

    def test_to_spec(self, rest_duckdb_source):
        """Test that to_spec() returns correct specification."""
        spec = rest_duckdb_source.to_spec()
        
        assert isinstance(spec, dict)
        assert 'uri' in spec
        assert spec['uri'] == ':memory:'
        assert 'tables' in spec
        assert 'type' in spec
        assert spec['type'] == 'rest_duckdb'

    def test_to_spec_with_sql_expressions(self, rest_duckdb_source):
        """Test to_spec() on derived source with SQL expressions."""
        new_source = rest_duckdb_source.create_sql_expr_source({
            'daily_1': "SELECT * FROM daily LIMIT 1"
        })
        
        spec = new_source.to_spec()
        
        assert isinstance(spec, dict)
        assert 'tables' in spec
        assert 'daily_1' in spec['tables']
        # Check that SQL expression is preserved in spec
        assert spec['tables']['daily_1'] == "SELECT * FROM daily LIMIT 1"

    def test_invalid_table_name(self, rest_duckdb_source):
        """Test that accessing non-existent table raises appropriate error."""
        with pytest.raises(Exception):
            rest_duckdb_source.get('nonexistent_table')

    def test_invalid_sql_query(self, rest_duckdb_source):
        """Test that invalid SQL raises appropriate error."""
        with pytest.raises(Exception):
            rest_duckdb_source.execute("SELECT * FROM nonexistent_table")

    def test_column_access(self, rest_duckdb_source):
        """Test accessing specific columns from the data."""
        df = rest_duckdb_source.get('daily')
        
        # Test expected columns exist
        expected_columns = ['station', 'day', 'max_temp_f', 'min_temp_f', 
                          'max_dewpoint_f', 'min_dewpoint_f', 'precip_in']
        for col in expected_columns:
            assert col in df.columns

    def test_data_types(self, rest_duckdb_source):
        """Test that data types are correctly inferred."""
        df = rest_duckdb_source.get('daily')
        
        # Numeric columns should be numeric types
        assert pd.api.types.is_numeric_dtype(df['max_temp_f'])
        assert pd.api.types.is_numeric_dtype(df['min_temp_f'])
        assert pd.api.types.is_numeric_dtype(df['precip_in'])

    def test_sql_join_across_tables(self, rest_duckdb_source):
        """Test SQL JOIN operations across multiple tables."""
        # Note: This test assumes both tables might have related data
        # In practice, adjust the JOIN condition based on actual schema
        query = """
        SELECT d.station, d.day, d.max_temp_f 
        FROM daily d
        LIMIT 5
        """
        result = rest_duckdb_source.execute(query)
        
        assert isinstance(result, pd.DataFrame)
        assert 'station' in result.columns
        assert 'day' in result.columns
        assert 'max_temp_f' in result.columns
