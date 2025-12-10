"""Tests for RESTDuckDBSource."""
import pandas as pd
import pytest

try:
    from lumen.sources.rest_duckdb import RESTDuckDBSource
    pytestmark = pytest.mark.xdist_group("duckdb")
except ImportError:
    pytestmark = pytest.mark.skip(reason="DuckDB is not installed")


# Table configurations as constants
DAILY_TABLE_CONFIG = {
    'url': 'https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py',
    'url_params': {
        'stations': 'ABR',
        'sts': '2025-12-08',
        'ets': '2025-12-09',
        'network': 'SD_ASOS',
        'format': 'csv'
    },
}

RAOB_TABLE_CONFIG = {
    'url': 'https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py',
    'url_params': {
        'station': 'KABR',
        'sts': '2025-12-08T15:49',
        'ets': '2025-12-09T15:49',
        'format': 'csv'
    },
}

PENGUINS_CSV_URL = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv'


@pytest.fixture(scope="session")
def single_table_source():
    """Fixture providing a RESTDuckDBSource with one REST table."""
    config = {
        'uri': ':memory:',
        'tables': {
            'daily': DAILY_TABLE_CONFIG,
        }
    }
    source = RESTDuckDBSource(**config)
    # Pre-materialize to avoid repeated API calls
    daily_df = source.get('daily')
    source._connection.from_df(daily_df).to_view('daily')
    source._cached_rest_tables["daily"] = DAILY_TABLE_CONFIG
    return source


@pytest.fixture(scope="session")
def multi_table_source():
    """Fixture providing a RESTDuckDBSource with two REST tables."""
    config = {
        'uri': ':memory:',
        'tables': {
            'daily': DAILY_TABLE_CONFIG,
            'raob': RAOB_TABLE_CONFIG,
        }
    }
    source = RESTDuckDBSource(**config)
    # Pre-materialize both tables
    daily_df = source.get('daily')
    raob_df = source.get('raob')
    source._connection.from_df(daily_df).to_view('daily')
    source._connection.from_df(raob_df).to_view('raob')
    source._cached_rest_tables["daily"] = DAILY_TABLE_CONFIG
    source._cached_rest_tables["raob"] = RAOB_TABLE_CONFIG
    return source


@pytest.fixture(scope="session")
def mixed_table_source():
    """Fixture providing a RESTDuckDBSource with REST table and CSV file."""
    config = {
        'uri': ':memory:',
        'tables': {
            'daily': DAILY_TABLE_CONFIG,
            'penguins': PENGUINS_CSV_URL,
        }
    }
    source = RESTDuckDBSource(**config)
    # Pre-materialize REST table
    daily_df = source.get('daily')
    source._connection.from_df(daily_df).to_view('daily')
    source._cached_rest_tables["daily"] = DAILY_TABLE_CONFIG
    # CSV table doesn't need pre-materialization
    return source


class TestRESTDuckDBSourceBasics:
    """Test basic functionality and initialization."""

    def test_source_type(self):
        """Test that source_type is correctly set."""
        assert RESTDuckDBSource.source_type == 'rest_duckdb'

    def test_resolve_module_type(self):
        """Test that the source can be resolved by module path."""
        assert RESTDuckDBSource._get_type('lumen.sources.rest_duckdb.RESTDuckDBSource') is RESTDuckDBSource

    def test_initialization(self, single_table_source):
        """Test that RESTDuckDBSource initializes correctly."""
        assert single_table_source.uri == ':memory:'
        assert 'daily' in single_table_source.tables
        assert isinstance(single_table_source.tables['daily'], dict)
        assert 'url' in single_table_source.tables['daily']


class TestRESTTableOperations:
    """Test REST-specific table operations."""

    def test_render_table_url(self, single_table_source):
        """Test that render_table_url constructs correct URLs."""
        url = single_table_source.render_table_url('daily')
        assert 'https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py' in url
        assert 'stations=ABR' in url
        assert 'sts=2025-12-08' in url
        assert 'format=csv' in url

    def test_get_table(self, single_table_source):
        """Test that get() retrieves data correctly."""
        df = single_table_source.get('daily')
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'station' in df.columns
        assert 'day' in df.columns
        assert 'max_temp_f' in df.columns
        assert len(df) == 2
        assert all(df['station'] == 'ABR')

    def test_get_multiple_tables(self, multi_table_source):
        """Test that both tables can be retrieved."""
        daily_df = multi_table_source.get('daily')
        raob_df = multi_table_source.get('raob')
        
        assert isinstance(daily_df, pd.DataFrame)
        assert isinstance(raob_df, pd.DataFrame)
        assert not daily_df.empty

    def test_invalid_table_name(self, single_table_source):
        """Test that accessing non-existent table raises appropriate error."""
        with pytest.raises(Exception):
            single_table_source.get('nonexistent_table')


class TestSQLExecution:
    """Test SQL query execution."""

    def test_execute_simple_query(self, single_table_source):
        """Test basic SQL execution."""
        result = single_table_source.execute("SELECT * FROM daily LIMIT 5")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5
        assert 'station' in result.columns

    def test_execute_with_filter(self, single_table_source):
        """Test SQL with WHERE clause."""
        result = single_table_source.execute("SELECT * FROM daily WHERE max_temp_f > 20")
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert all(result['max_temp_f'] > 20)

    def test_execute_aggregate(self, single_table_source):
        """Test SQL aggregation functions."""
        result = single_table_source.execute("SELECT COUNT(*) as count FROM daily")
        
        assert isinstance(result, pd.DataFrame)
        assert 'count' in result.columns
        assert result['count'].iloc[0] > 0

    def test_execute_materializes_rest_tables(self, single_table_source):
        """Test that execute() automatically materializes REST tables."""
        result = single_table_source.execute("SELECT COUNT(*) as cnt FROM daily")
        
        assert isinstance(result, pd.DataFrame)
        assert 'daily' in single_table_source._cached_rest_tables

    def test_invalid_sql_query(self, single_table_source):
        """Test that invalid SQL raises appropriate error."""
        with pytest.raises(Exception):
            single_table_source.execute("SELECT * FROM nonexistent_table")


class TestSQLExpressionSource:
    """Test create_sql_expr_source functionality."""

    def test_create_simple_expression(self, single_table_source):
        """Test creating a source with a simple SQL expression."""
        new_source = single_table_source.create_sql_expr_source({
            'daily_1': "SELECT * FROM daily LIMIT 1"
        })
        
        assert 'daily_1' in new_source.tables
        df = new_source.get('daily_1')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_create_multiple_expressions(self, single_table_source):
        """Test creating multiple SQL expressions at once."""
        new_source = single_table_source.create_sql_expr_source({
            'daily_1': "SELECT * FROM daily LIMIT 1",
            'daily_high_temp': "SELECT * FROM daily WHERE max_temp_f > 30"
        })
        
        assert 'daily_1' in new_source.tables
        assert 'daily_high_temp' in new_source.tables
        
        df1 = new_source.get('daily_1')
        df_high = new_source.get('daily_high_temp')
        
        assert len(df1) == 1
        assert isinstance(df_high, pd.DataFrame)

    def test_preserves_original_tables(self, single_table_source):
        """Test that creating SQL expr source preserves original tables."""
        new_source = single_table_source.create_sql_expr_source({
            'daily_1': "SELECT * FROM daily LIMIT 1"
        })
        
        # Original table should still be accessible
        daily_df = new_source.get('daily')
        assert isinstance(daily_df, pd.DataFrame)
        assert len(daily_df) > 1

    def test_rest_table_dependency_materialization(self, single_table_source):
        """Test that REST tables in SQL expressions are materialized."""
        new_source = single_table_source.create_sql_expr_source({
            'daily_filtered': "SELECT * FROM daily WHERE max_temp_f > 20"
        })
        
        # REST table should be accessible and materialized
        assert 'daily' in new_source.tables
        daily_df = new_source.get('daily')
        filtered_df = new_source.get('daily_filtered')
        
        assert len(filtered_df) <= len(daily_df)
        if not filtered_df.empty:
            assert all(filtered_df['max_temp_f'] > 20)

    def test_upsert_behavior(self, single_table_source):
        """Test that new tables with same name override existing ones."""
        source1 = single_table_source.create_sql_expr_source({
            'summary': "SELECT COUNT(*) as total_days FROM daily"
        })
        result1 = source1.get('summary')
        assert 'total_days' in result1.columns
        
        source2 = source1.create_sql_expr_source({
            'summary': "SELECT AVG(max_temp_f) as avg_temp FROM daily"
        })
        result2 = source2.get('summary')
        assert 'avg_temp' in result2.columns
        assert 'total_days' not in result2.columns

    def test_multiple_rest_dependencies(self, multi_table_source):
        """Test SQL expression that references multiple REST tables."""
        new_source = multi_table_source.create_sql_expr_source({
            'combined': """
                SELECT station, day, max_temp_f FROM daily
                UNION ALL
                SELECT station, validUTC as day, tmpc as max_temp_f FROM raob
                LIMIT 10
            """
        })
        
        # Both REST tables should be materialized
        assert 'daily' in new_source.tables
        assert 'raob' in new_source.tables
        assert 'combined' in new_source.tables
        
        result = new_source.get('combined')
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 10

    def test_preserves_rest_configs(self, single_table_source):
        """Test that REST table configs are preserved in derived sources."""
        new_source = single_table_source.create_sql_expr_source({
            'daily_subset': "SELECT * FROM daily LIMIT 5"
        })
        
        # REST config should be preserved
        assert isinstance(new_source.tables['daily'], dict)
        assert 'url' in new_source.tables['daily']
        url = new_source.render_table_url('daily')
        assert 'https://mesonet.agron.iastate.edu' in url


class TestSerialization:
    """Test source serialization."""

    def test_to_spec_basic(self, single_table_source):
        """Test that to_spec() returns correct specification."""
        spec = single_table_source.to_spec()
        
        assert isinstance(spec, dict)
        assert spec['uri'] == ':memory:'
        assert 'tables' in spec
        assert spec['type'] == 'rest_duckdb'
        assert '_cached_rest_tables' not in spec

    def test_to_spec_with_sql_expressions(self, single_table_source):
        """Test to_spec() on derived source with SQL expressions."""
        new_source = single_table_source.create_sql_expr_source({
            'daily_1': "SELECT * FROM daily LIMIT 1"
        })
        
        spec = new_source.to_spec()
        assert 'daily_1' in spec['tables']
        assert spec['tables']['daily_1'] == "SELECT * FROM daily LIMIT 1"


class TestDataValidation:
    """Test data type and content validation."""

    def test_column_presence(self, single_table_source):
        """Test that expected columns are present."""
        df = single_table_source.get('daily')
        
        expected_columns = ['station', 'day', 'max_temp_f', 'min_temp_f', 
                          'max_dewpoint_f', 'min_dewpoint_f', 'precip_in']
        for col in expected_columns:
            assert col in df.columns

    def test_data_types(self, single_table_source):
        """Test that data types are correctly inferred."""
        df = single_table_source.get('daily')
        
        assert pd.api.types.is_numeric_dtype(df['max_temp_f'])
        assert pd.api.types.is_numeric_dtype(df['min_temp_f'])
        assert pd.api.types.is_numeric_dtype(df['precip_in'])


class TestRequiredParams:
    """Test required_params validation."""

    def test_required_params_missing_raises_error(self):
        """Test that missing required params raises ValueError."""
        config = {
            'uri': ':memory:',
            'tables': {
                'daily': {
                    'url': 'https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py',
                    'url_params': {
                        'stations': 'ABR',
                        'network': 'SD_ASOS',
                        'format': 'csv'
                    },
                    'required_params': ['stations', 'sts', 'ets'],
                },
            }
        }
        source = RESTDuckDBSource(**config)

        with pytest.raises(ValueError, match="Missing required parameters.*sts.*ets"):
            source.get('daily')

    def test_required_params_provided_via_url_params_arg(self):
        """Test that required params can be provided via url_params argument."""
        config = {
            'uri': ':memory:',
            'tables': {
                'daily': {
                    'url': 'https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py',
                    'url_params': {
                        'network': 'SD_ASOS',
                        'format': 'csv'
                    },
                    'required_params': ['stations', 'sts', 'ets'],
                },
            }
        }
        source = RESTDuckDBSource(**config)
        df = source.get('daily', url_params={
            'stations': 'ABR',
            'sts': '2025-12-08',
            'ets': '2025-12-09',
        })

        assert isinstance(df, pd.DataFrame)
        assert not df.empty


class TestReadFnAndReadOptions:
    """Test read_fn and read_options configuration."""

    def test_read_fn_and_read_options_in_sql_expr(self):
        """Test that read_fn and read_options are included in the SQL expression."""
        config = {
            'uri': ':memory:',
            'tables': {
                'data': {
                    'url': 'https://example.com/data.csv',
                    'url_params': {},
                    'read_fn': 'csv',
                    'read_options': {
                        'header': True,
                        'delim': ',',
                    },
                },
            }
        }
        source = RESTDuckDBSource(**config)
        sql_expr = source.get_sql_expr('data')

        assert 'read_csv_auto' in sql_expr
        assert 'header=True' in sql_expr
        assert "delim=','" in sql_expr

    def test_read_fn_falls_back_to_url_params_format(self):
        """Test that read_fn falls back to url_params['format']."""
        config = {
            'uri': ':memory:',
            'tables': {
                'data': {
                    'url': 'https://example.com/data',
                    'url_params': {'format': 'csv'},
                },
            }
        }
        source = RESTDuckDBSource(**config)
        sql_expr = source.get_sql_expr('data')

        assert 'read_csv_auto' in sql_expr


class TestMixedTableTypes:
    """Test mixing REST tables with regular CSV tables."""

    def test_mixed_source_has_both_table_types(self, mixed_table_source):
        """Test that mixed source contains both REST and CSV tables."""
        assert 'daily' in mixed_table_source.tables
        assert 'penguins' in mixed_table_source.tables
        
        # daily is REST table (dict config)
        assert isinstance(mixed_table_source.tables['daily'], dict)
        assert 'url' in mixed_table_source.tables['daily']
        
        # penguins is CSV table (string URL)
        assert isinstance(mixed_table_source.tables['penguins'], str)

    def test_get_csv_table(self, mixed_table_source):
        """Test retrieving CSV table from mixed source."""
        df = mixed_table_source.get('penguins')
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'species' in df.columns
        assert 'island' in df.columns
        assert 'bill_length_mm' in df.columns

    def test_get_rest_table_from_mixed(self, mixed_table_source):
        """Test retrieving REST table from mixed source."""
        df = mixed_table_source.get('daily')
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'station' in df.columns

    def test_sql_join_rest_and_csv(self, mixed_table_source):
        """Test SQL query joining REST and CSV tables."""
        result = mixed_table_source.execute("""
            SELECT d.station, p.species, COUNT(*) as count
            FROM daily d
            CROSS JOIN penguins p
            WHERE p.species = 'Adelie'
            GROUP BY d.station, p.species
            LIMIT 5
        """)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'station' in result.columns
        assert 'species' in result.columns
        assert all(result['species'] == 'Adelie')

    def test_create_sql_expr_with_mixed_tables(self, mixed_table_source):
        """Test creating SQL expressions that reference both table types."""
        new_source = mixed_table_source.create_sql_expr_source({
            'rest_summary': "SELECT station, AVG(max_temp_f) as avg_temp FROM daily GROUP BY station",
            'csv_summary': "SELECT species, COUNT(*) as count FROM penguins GROUP BY species",
            'combined': """
                SELECT 'weather' as source_type, station as name FROM daily
                UNION ALL
                SELECT 'penguin' as source_type, species as name FROM penguins
                LIMIT 10
            """
        })
        
        # All tables should exist
        assert 'rest_summary' in new_source.tables
        assert 'csv_summary' in new_source.tables
        assert 'combined' in new_source.tables
        
        # Verify they work
        rest_df = new_source.get('rest_summary')
        csv_df = new_source.get('csv_summary')
        combined_df = new_source.get('combined')
        
        assert isinstance(rest_df, pd.DataFrame)
        assert isinstance(csv_df, pd.DataFrame)
        assert isinstance(combined_df, pd.DataFrame)
        assert 'avg_temp' in rest_df.columns
        assert 'species' in csv_df.columns
        assert 'source_type' in combined_df.columns
