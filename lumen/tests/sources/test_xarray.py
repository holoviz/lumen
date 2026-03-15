import datetime as dt

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

xr = pytest.importorskip("xarray")

from lumen.dashboard import Dashboard
from lumen.sources.xarray import XarraySource
from lumen.validation import ValidationError


def assert_df_equal(df1, df2):
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False)


@pytest.fixture
def xr_dataset():
    time = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    lat = np.array([10.0, 20.0])
    lon = np.array([70.0, 80.0])

    temperature = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])
    pressure = np.array([
        [[101.0, 102.0], [103.0, 104.0]],
        [[105.0, 106.0], [107.0, 108.0]],
    ])

    ds = xr.Dataset(
        data_vars={
            "temperature": (("time", "lat", "lon"), temperature),
            "pressure": (("time", "lat", "lon"), pressure),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
    )
    ds["temperature"].attrs["units"] = "degC"
    ds["temperature"].attrs["long_name"] = "Air temperature"
    ds["lat"].attrs["long_name"] = "Latitude"
    return ds


def test_xarray_source_get_tables(xr_dataset):
    source = XarraySource(dataset=xr_dataset)
    assert source.get_tables() == ["temperature", "pressure"]


def test_xarray_source_loads_from_uri(tmp_path, xr_dataset):
    path = tmp_path / "sample.nc"
    xr_dataset.to_netcdf(path)

    source = XarraySource(uri=str(path))

    assert source.get_tables() == ["temperature", "pressure"]
    result = source.get("temperature", time="2024-01-01")
    assert len(result) == 4


def test_xarray_source_explicit_zarr_format_uses_open_zarr(monkeypatch):
    calls = []
    expected = object()

    def fake_open_zarr(uri, **kwargs):
        calls.append(("zarr", uri, kwargs))
        return expected

    def fake_open_dataset(uri, **kwargs):
        calls.append(("dataset", uri, kwargs))
        raise AssertionError("open_dataset should not be called for explicit zarr format")

    monkeypatch.setattr(xr, "open_zarr", fake_open_zarr)
    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    source = XarraySource(uri="memory-store", dataset_format="zarr", load_kwargs={"chunks": {}})
    resolved = source._resolve_uri()

    assert source._open_dataset() is expected
    assert calls == [("zarr", resolved, {"chunks": {}})]


def test_xarray_source_auto_detects_local_zarr_store_without_suffix(tmp_path, monkeypatch):
    store = tmp_path / "temperature_store"
    store.mkdir()
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    calls = []
    expected = object()

    def fake_open_zarr(uri, **kwargs):
        calls.append(("zarr", uri, kwargs))
        return expected

    def fake_open_dataset(uri, **kwargs):
        calls.append(("dataset", uri, kwargs))
        raise AssertionError("open_dataset should not be called for a detected local zarr store")

    monkeypatch.setattr(xr, "open_zarr", fake_open_zarr)
    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    source = XarraySource(uri=str(store))

    assert source._open_dataset() is expected
    assert calls == [("zarr", str(store), {})]


def test_xarray_source_windows_uri_treated_as_local_path(tmp_path, xr_dataset):
    path = tmp_path / "sample.nc"
    xr_dataset.to_netcdf(path)

    source = XarraySource(uri=r"C:\data\sample.nc", root=tmp_path)

    resolved = source._resolve_uri()

    assert resolved == r"C:\data\sample.nc"
    assert not source._is_remote_uri(r"C:\data\sample.nc")


def test_xarray_example_spec_renders_dashboard(tmp_path):
    example_path = Path(__file__).resolve().parents[3] / "examples" / "xarray_air_temperature.yaml"
    spec = yaml.safe_load(example_path.read_text(encoding="utf-8"))

    ds = xr.Dataset(
        data_vars={
            "air": (
                ("time", "lat", "lon"),
                np.array([
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]),
            ),
        },
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat": np.array([10.0, 20.0]),
            "lon": np.array([70.0, 80.0]),
        },
    )
    ds.to_netcdf(tmp_path / "air_temperature.nc")

    dashboard = Dashboard(spec, root=str(tmp_path))
    dashboard._render_dashboard()

    layout = dashboard.layouts[0]
    pipeline = next(iter(layout._pipelines.values()))

    assert isinstance(pipeline.source, XarraySource)
    assert [filt.field for filt in pipeline.filters] == ["time", "lat"]
    assert list(pipeline.data.columns) == ["time", "lat", "lon", "air"]
    assert len(pipeline.data) == 8


def test_xarray_source_get_schema_single_table(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    schema = source.get_schema("temperature")

    assert set(schema) == {"time", "lat", "lon", "temperature", "__len__"}
    assert schema["time"]["type"] == "string"
    assert schema["time"]["format"] == "datetime"
    assert schema["lat"]["type"] == "number"
    assert schema["lon"]["type"] == "number"
    assert schema["temperature"]["type"] == "number"
    assert schema["__len__"] == 8


def test_xarray_source_get_schema_only_exposes_queryable_1d_coords():
    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("x", "y"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
        },
        coords={
            "x": np.array([0, 1]),
            "y": np.array([0, 1]),
            "mesh": (("x", "y"), np.array([[10, 20], [30, 40]])),
        },
    )
    source = XarraySource(dataset=ds)

    schema = source.get_schema("temperature")

    assert set(schema) == {"x", "y", "temperature", "__len__"}


def test_xarray_source_get_schema_all_tables(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    schemas = source.get_schema()

    assert set(schemas) == {"temperature", "pressure"}
    assert set(schemas["temperature"]) == {"time", "lat", "lon", "temperature", "__len__"}
    assert set(schemas["pressure"]) == {"time", "lat", "lon", "pressure", "__len__"}
    assert schemas["temperature"]["__len__"] == 8
    assert schemas["pressure"]["__len__"] == 8


def test_xarray_source_get_schema_with_limit(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    schema = source.get_schema("temperature", limit=1)

    assert schema == {
        "time": {
            "type": "string",
            "format": "datetime",
            "inclusiveMinimum": "2024-01-01T00:00:00",
            "inclusiveMaximum": "2024-01-01T00:00:00",
        },
        "lat": {"type": "number", "inclusiveMinimum": 10.0, "inclusiveMaximum": 10.0},
        "lon": {"type": "number", "inclusiveMinimum": 70.0, "inclusiveMaximum": 70.0},
        "temperature": {"type": "number", "inclusiveMinimum": 1.0, "inclusiveMaximum": 1.0},
        "__len__": 8,
    }


def test_xarray_source_get_schema_with_shuffle(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    schema = source.get_schema("temperature", limit=1, shuffle=True)

    assert schema == {
        "time": {
            "type": "string",
            "format": "datetime",
            "inclusiveMinimum": "2024-01-02T00:00:00",
            "inclusiveMaximum": "2024-01-02T00:00:00",
        },
        "lat": {"type": "number", "inclusiveMinimum": 20.0, "inclusiveMaximum": 20.0},
        "lon": {"type": "number", "inclusiveMinimum": 70.0, "inclusiveMaximum": 70.0},
        "temperature": {"type": "number", "inclusiveMinimum": 7.0, "inclusiveMaximum": 7.0},
        "__len__": 8,
    }


def test_xarray_source_get_returns_dataframe(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    result = source.get("temperature")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["time", "lat", "lon", "temperature"]
    assert len(result) == 8


def test_xarray_source_get_scalar_coord_filter(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    result = source.get("temperature", time=np.datetime64("2024-01-01"))
    expected = pd.DataFrame(
        {
            "time": np.array(["2024-01-01"] * 4, dtype="datetime64[ns]"),
            "lat": [10.0, 10.0, 20.0, 20.0],
            "lon": [70.0, 80.0, 70.0, 80.0],
            "temperature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    assert_df_equal(result, expected)


def test_xarray_source_get_scalar_date_coord_filter(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    result = source.get("temperature", time=dt.date(2024, 1, 1))

    assert len(result) == 4
    assert result["time"].nunique() == 1
    assert pd.Timestamp(result["time"].iloc[0]) == pd.Timestamp("2024-01-01")


def test_xarray_source_get_scalar_string_datetime_coord_filter(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    result = source.get("temperature", time="2024-01-01")

    assert len(result) == 4
    assert result["time"].nunique() == 1
    assert pd.Timestamp(result["time"].iloc[0]) == pd.Timestamp("2024-01-01")


def test_xarray_source_get_range_coord_filter(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    result = source.get("temperature", lat=(10.0, 10.0))
    expected = pd.DataFrame(
        {
            "time": np.array(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"], dtype="datetime64[ns]"),
            "lat": [10.0, 10.0, 10.0, 10.0],
            "lon": [70.0, 80.0, 70.0, 80.0],
            "temperature": [1.0, 2.0, 5.0, 6.0],
        }
    )

    assert_df_equal(result, expected)


def test_xarray_source_get_date_range_coord_filter(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    result = source.get("temperature", time=(dt.date(2024, 1, 1), dt.date(2024, 1, 1)))

    assert len(result) == 4
    assert result["time"].nunique() == 1
    assert pd.Timestamp(result["time"].iloc[0]) == pd.Timestamp("2024-01-01")


def test_xarray_source_get_range_coord_filter_descending_coordinate():
    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("lat", "lon"),
                np.array([
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]),
            ),
        },
        coords={
            "lat": np.array([30.0, 20.0, 10.0]),
            "lon": np.array([70.0, 80.0]),
        },
    )
    source = XarraySource(dataset=ds)

    result = source.get("temperature", lat=(10.0, 20.0))

    assert list(result["lat"].unique()) == [20.0, 10.0]
    assert len(result) == 4


def test_xarray_source_get_range_list_coord_filter(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    result = source.get("temperature", time=[(np.datetime64("2024-01-01"), np.datetime64("2024-01-01"))])
    expected = pd.DataFrame(
        {
            "time": np.array(["2024-01-01"] * 4, dtype="datetime64[ns]"),
            "lat": [10.0, 10.0, 20.0, 20.0],
            "lon": [70.0, 80.0, 70.0, 80.0],
            "temperature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    assert_df_equal(result, expected)


def test_xarray_source_metadata_preserves_variable_attrs(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    metadata = source.get_metadata("temperature")

    assert metadata["description"] == "Air temperature"
    assert metadata["dims"] == ["time", "lat", "lon"]
    assert metadata["columns"]["temperature"]["unit"] == "degC"
    assert metadata["columns"]["lat"]["description"] == "Latitude"


def test_xarray_source_metadata_reports_auxiliary_coords():
    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("x", "y"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
        },
        coords={
            "x": np.array([0, 1]),
            "y": np.array([0, 1]),
            "mesh": (("x", "y"), np.array([[10, 20], [30, 40]])),
        },
    )
    source = XarraySource(dataset=ds)

    metadata = source.get_metadata("temperature")

    assert metadata["queryable_coords"] == ["x", "y"]
    assert metadata["auxiliary_coords"] == ["mesh"]


def test_xarray_source_unknown_table_raises(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    with pytest.raises(KeyError):
        source.get("not_a_table")


def test_xarray_source_unknown_schema_table_raises_validation_error(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    with pytest.raises(ValidationError, match="does not contain"):
        source.get_schema("not_a_table")


def test_xarray_source_unknown_query_field_raises(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    with pytest.raises(ValueError, match="Unsupported query key"):
        source.get("temperature", station=1)


def test_xarray_source_filterable_coords_restrict_query_surface(xr_dataset):
    source = XarraySource(dataset=xr_dataset, filterable_coords=["time"])

    schema = source.get_schema("temperature")

    assert set(schema) == {"time", "temperature", "__len__"}
    with pytest.raises(ValueError, match="Unsupported query key"):
        source.get("temperature", lat=(10.0, 20.0))


def test_xarray_source_invalid_range_tuple_raises(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    with pytest.raises(ValueError, match="exactly two elements"):
        source.get("temperature", lat=(10.0, 20.0, 30.0))


def test_xarray_source_empty_selection_returns_empty_dataframe(xr_dataset):
    source = XarraySource(dataset=xr_dataset)

    result = source.get("temperature", time=[])

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["time", "lat", "lon", "temperature"]
    assert result.empty


def test_xarray_source_handles_variable_with_different_dims():
    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "lat", "lon"),
                np.array([
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]),
            ),
            "station_bias": (
                ("lat", "lon"),
                np.array([
                    [0.1, 0.2],
                    [0.3, 0.4],
                ]),
            ),
        },
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat": np.array([10.0, 20.0]),
            "lon": np.array([70.0, 80.0]),
        },
    )
    source = XarraySource(dataset=ds)

    schema = source.get_schema("station_bias")
    result = source.get("station_bias")

    assert set(source.get_tables()) == {"temperature", "station_bias"}
    assert set(schema) == {"lat", "lon", "station_bias", "__len__"}
    assert list(result.columns) == ["lat", "lon", "station_bias"]
    assert len(result) == 4
    assert schema["__len__"] == 4


def test_xarray_source_max_rows_guard_raises(xr_dataset):
    source = XarraySource(dataset=xr_dataset, max_rows=2)

    with pytest.raises(ValueError, match="exceeds max_rows=2"):
        source.get("temperature")


def test_xarray_source_max_rows_guard_raises_before_materializing_dataframe(xr_dataset, monkeypatch):
    source = XarraySource(dataset=xr_dataset, max_rows=2)

    def fail_to_dataframe(*args, **kwargs):
        raise AssertionError("to_dataframe should not be called when max_rows is exceeded")

    monkeypatch.setattr(xr.DataArray, "to_dataframe", fail_to_dataframe)

    with pytest.raises(ValueError, match="exceeds max_rows=2"):
        source.get("temperature")


def test_xarray_source_non_1d_coordinate_filter_raises():
    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("x", "y"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
        },
        coords={
            "x": np.array([0, 1]),
            "y": np.array([0, 1]),
            "mesh": (("x", "y"), np.array([[10, 20], [30, 40]])),
        },
    )
    source = XarraySource(dataset=ds)

    with pytest.raises(ValueError, match="non-1D coordinate"):
        source.get("temperature", mesh=(10, 20))
