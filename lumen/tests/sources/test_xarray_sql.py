"""
Tests for XArraySQLSource -- SQL-backed xarray source via DataFusion.

Tests cover:
- Construction from dataset, file path, and Zarr store
- SQL query execution via DataFusion
- Table listing, schema, and metadata generation
- Lumen integration (get, get_sql_expr, normalize_table)
- Serialization (to_spec / from_spec)
- Async operations
- Edge cases and error handling
"""

import warnings

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("xarray_sql")

from lumen.sources.xarray_sql import XArraySQLSource

# ---- Fixtures ----

@pytest.fixture
def synthetic_dataset():
    """
    Synthetic 3D dataset with known values for deterministic testing.
    Dimensions: time(10) x lat(5) x lon(4)
    Variables: temperature, pressure
    """
    times = pd.date_range("2020-01-01", periods=10, freq="D")
    lats = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lons = np.array([100.0, 110.0, 120.0, 130.0])

    rng = np.random.default_rng(42)
    temp = rng.uniform(250, 310, (10, 5, 4))
    pressure = rng.uniform(950, 1050, (10, 5, 4))

    return xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], temp, {
                "units": "K",
                "long_name": "Air Temperature",
            }),
            "pressure": (["time", "lat", "lon"], pressure, {
                "units": "hPa",
                "long_name": "Surface Pressure",
            }),
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={
            "title": "Synthetic Test Dataset",
            "source": "lumen-xarray tests",
        },
    )


@pytest.fixture
def nc_file(synthetic_dataset, tmp_path):
    """Write synthetic dataset to a temporary NetCDF file."""
    pytest.importorskip("netCDF4")
    path = tmp_path / "test_data.nc"
    synthetic_dataset.to_netcdf(path)
    return str(path)


@pytest.fixture
def zarr_path(synthetic_dataset, tmp_path):
    """Write synthetic dataset to a temporary Zarr store."""
    zarr = pytest.importorskip("zarr")  # noqa: F841
    path = tmp_path / "test_data.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        synthetic_dataset.to_zarr(path)
    return str(path)


# ---- Construction ----

class TestConstruction:

    def test_from_dataset(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        assert source.get_tables() == ["pressure", "temperature"]

    def test_from_netcdf(self, nc_file):
        source = XArraySQLSource(uri=nc_file)
        assert "temperature" in source.get_tables()
        assert "pressure" in source.get_tables()

    def test_from_zarr(self, zarr_path):
        source = XArraySQLSource(uri=zarr_path, engine="zarr")
        assert "temperature" in source.get_tables()

    def test_auto_engine_detection(self, nc_file):
        source = XArraySQLSource(uri=nc_file)
        assert source.dataset is not None

    def test_variable_filtering(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset, variables=["temperature"])
        assert source.get_tables() == ["temperature"]

    def test_invalid_variable_raises(self, synthetic_dataset):
        with pytest.raises(ValueError, match="not found"):
            XArraySQLSource(_dataset=synthetic_dataset, variables=["nonexistent"])

    def test_no_uri_no_dataset_raises(self):
        with pytest.raises(ValueError, match="Either 'uri' or '_dataset'"):
            XArraySQLSource()

    def test_dialect_is_postgres(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        assert source.dialect == "postgres"

    def test_supports_sql(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        assert source._supports_sql is True

    def test_source_type(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        assert source.source_type == "xarray"


# ---- SQL Execution ----

class TestSQLExecution:

    def test_select_all(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.execute("SELECT * FROM temperature LIMIT 5")
        assert len(df) == 5
        assert "temperature" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "time" in df.columns

    def test_select_columns(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.execute("SELECT lat, lon, temperature FROM temperature LIMIT 3")
        assert list(df.columns) == ["lat", "lon", "temperature"]
        assert len(df) == 3

    def test_where_filter(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.execute("SELECT * FROM temperature WHERE lat > 30")
        assert all(df["lat"] > 30)
        assert len(df) > 0

    def test_aggregation(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.execute("SELECT AVG(temperature) as avg_temp FROM temperature")
        assert len(df) == 1
        assert "avg_temp" in df.columns
        assert 250 < df["avg_temp"].iloc[0] < 310

    def test_group_by(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.execute(
            "SELECT lat, AVG(temperature) as avg_temp FROM temperature GROUP BY lat ORDER BY lat"
        )
        assert len(df) == 5  # 5 lat values
        assert df["lat"].is_monotonic_increasing

    def test_count(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.execute("SELECT COUNT(*) as cnt FROM temperature")
        assert df["cnt"].iloc[0] == 10 * 5 * 4  # time * lat * lon = 200

    def test_multiple_conditions(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.execute(
            "SELECT * FROM temperature WHERE lat >= 20 AND lat <= 40 AND lon > 110"
        )
        assert all(df["lat"] >= 20)
        assert all(df["lat"] <= 40)
        assert all(df["lon"] > 110)

    def test_order_by(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.execute(
            "SELECT lat, temperature FROM temperature ORDER BY temperature DESC LIMIT 10"
        )
        assert df["temperature"].is_monotonic_decreasing

    def test_cross_table_independence(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df_t = source.execute("SELECT COUNT(*) as cnt FROM temperature")
        df_p = source.execute("SELECT COUNT(*) as cnt FROM pressure")
        assert df_t["cnt"].iloc[0] == df_p["cnt"].iloc[0]


# ---- Schema & Metadata ----

class TestSchemaMetadata:

    def test_get_schema_single_table(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        schema = source.get_schema("temperature")
        assert "temperature" in schema
        assert "lat" in schema
        assert "lon" in schema
        assert "time" in schema
        assert "__len__" in schema

    def test_get_schema_all_tables(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        schemas = source.get_schema()
        assert "temperature" in schemas
        assert "pressure" in schemas

    def test_schema_has_types(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        schema = source.get_schema("temperature")
        assert "type" in schema["temperature"]
        assert "type" in schema["lat"]

    def test_schema_has_minmax(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        schema = source.get_schema("temperature")
        assert "inclusiveMinimum" in schema.get("lat", {}) or "inclusiveMinimum" in schema.get("temperature", {})

    def test_get_schema_with_limit(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        schema = source.get_schema("temperature", limit=5)
        assert "__len__" in schema

    def test_get_schema_with_shuffle(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        schema = source.get_schema("temperature", shuffle=True)
        assert "temperature" in schema

    def test_get_metadata(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        meta = source.get_metadata("temperature")
        assert "description" in meta
        assert "columns" in meta
        assert "rows" in meta
        assert meta["rows"] == 10 * 5 * 4  # time * lat * lon

    def test_metadata_has_descriptions(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        meta = source.get_metadata("temperature")
        assert "Air Temperature" in meta["description"]
        cols = meta["columns"]
        assert cols["temperature"]["description"] == "Air Temperature"
        assert cols["lat"]["data_type"] == "float64"

    def test_metadata_all_tables(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        meta = source.get_metadata()
        assert "temperature" in meta
        assert "pressure" in meta


# ---- Lumen Integration (get, get_sql_expr) ----

class TestLumenIntegration:

    def test_get_returns_dataframe(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.get("temperature")
        assert isinstance(df, pd.DataFrame)
        assert "temperature" in df.columns
        assert len(df) > 0

    def test_get_with_filter(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.get("temperature", lat=30.0)
        assert all(df["lat"] == 30.0)

    def test_get_sql_expr(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        expr = source.get_sql_expr("temperature")
        assert isinstance(expr, str)
        assert "temperature" in expr.lower()

    def test_get_sql_expr_dict_tables(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        derived = {"avg_temp": "SELECT lat, AVG(temperature) FROM temperature GROUP BY lat"}
        new_source = source.create_sql_expr_source(derived)
        expr = new_source.get_sql_expr("avg_temp")
        assert "AVG" in expr

    def test_get_sql_expr_missing_table_raises(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        derived = {"avg_temp": "SELECT 1"}
        new_source = source.create_sql_expr_source(derived)
        with pytest.raises(KeyError, match="not found"):
            new_source.get_sql_expr("nonexistent")

    def test_get_skips_dunder_keys(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.get("temperature", __limit__=10)
        assert len(df) > 0

    def test_get_with_list_filter(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.get("temperature", lat=[10.0, 30.0])
        assert set(df["lat"].unique()) <= {10.0, 30.0}

    def test_get_with_slice_filter(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = source.get("temperature", lat=slice(20.0, 40.0))
        assert all(df["lat"] >= 20.0)
        assert all(df["lat"] <= 40.0)

    def test_get_with_timestamp_filter(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        ts = pd.Timestamp("2020-01-03")
        df = source.get("temperature", time=ts)
        assert len(df) > 0


# ---- Normalize Table ----

class TestNormalizeTable:

    def test_exact_match(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        assert source.normalize_table("temperature") == "temperature"

    def test_case_insensitive(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        assert source.normalize_table("TEMPERATURE") == "temperature"
        assert source.normalize_table("Temperature") == "temperature"

    def test_strip_quotes(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        assert source.normalize_table('"temperature"') == "temperature"
        assert source.normalize_table("'temperature'") == "temperature"

    def test_unknown_returns_as_is(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        assert source.normalize_table("nonexistent") == "nonexistent"

    def test_normalize_derived_table(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        derived = {"avg_temp": "SELECT 1"}
        new_source = source.create_sql_expr_source(derived)
        assert new_source.normalize_table("AVG_TEMP") == "avg_temp"


# ---- get_tables ----

class TestGetTables:

    def test_includes_registered_tables(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        assert "temperature" in source.get_tables()
        assert "pressure" in source.get_tables()

    def test_includes_derived_tables(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        derived = {"avg_temp": "SELECT lat, AVG(temperature) FROM temperature GROUP BY lat"}
        new_source = source.create_sql_expr_source(derived)
        tables = new_source.get_tables()
        assert "avg_temp" in tables
        assert "temperature" in tables
        assert "pressure" in tables


# ---- Serialization ----

class TestSerialization:

    def test_to_spec(self, nc_file):
        source = XArraySQLSource(uri=nc_file)
        spec = source.to_spec()
        assert spec["type"] == "xarray"
        assert spec["uri"] == nc_file

    def test_to_spec_with_options(self, nc_file):
        source = XArraySQLSource(
            uri=nc_file, engine="netcdf4", variables=["temperature"]
        )
        spec = source.to_spec()
        assert spec["engine"] == "netcdf4"
        assert spec["variables"] == ["temperature"]

    def test_roundtrip_spec(self, nc_file):
        source1 = XArraySQLSource(uri=nc_file)
        spec = source1.to_spec()
        source2 = XArraySQLSource.from_spec(spec)
        assert source1.get_tables() == source2.get_tables()


# ---- Resource Management ----

class TestResourceManagement:

    def test_close(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        source.close()
        assert source._ctx is None

    def test_clear_cache(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        _ = source.get_schema("temperature")
        source.clear_cache()
        schema = source.get_schema("temperature")
        assert "temperature" in schema

    def test_dataset_property(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        assert source.dataset is synthetic_dataset

    def test_create_sql_expr_source(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        new_tables = {"avg_temp": "SELECT lat, AVG(temperature) FROM temperature GROUP BY lat"}
        new_source = source.create_sql_expr_source(new_tables)
        assert new_source._ctx is source._ctx  # shared context
        assert new_source._dataset is source._dataset

    def test_create_sql_expr_source_with_params(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        new_tables = {"filtered": "SELECT * FROM temperature WHERE lat > :min_lat"}
        params = {"filtered": {"min_lat": 30.0}}
        new_source = source.create_sql_expr_source(new_tables, params=params)
        assert new_source.table_params["filtered"] == {"min_lat": 30.0}

    def test_create_sql_expr_source_merges_tables(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        derived = {"avg_temp": "SELECT lat, AVG(temperature) FROM temperature GROUP BY lat"}
        new_source = source.create_sql_expr_source(derived)
        tables = new_source.get_tables()
        assert "temperature" in tables
        assert "pressure" in tables
        assert "avg_temp" in tables


# ---- Async ----

class TestAsync:

    @pytest.mark.asyncio
    async def test_execute_async(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = await source.execute_async("SELECT * FROM temperature LIMIT 5")
        assert len(df) == 5

    @pytest.mark.asyncio
    async def test_get_async(self, synthetic_dataset):
        source = XArraySQLSource(_dataset=synthetic_dataset)
        df = await source.get_async("temperature")
        assert len(df) > 0
