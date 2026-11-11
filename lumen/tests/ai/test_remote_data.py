import numpy as np
import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

import duckdb

import lumen.sources.data as data_module

from lumen.sources.data import read, resolve_data
from lumen.sources.duckdb import DuckDBSource
from lumen.sources.xarray_sql import XArraySQLSource


@pytest.mark.parametrize("url, expected", [
    ("gs://gcp-public-data-arco-era5/ar/x.zarr-v2", "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/x.zarr-v2"),
    ("s3://bucket/a/b.zarr", "https://bucket.s3.amazonaws.com/a/b.zarr"),
    ("https://data.source.coop/x.parquet", "https://data.source.coop/x.parquet"),
])
def test_public_buckets_are_read_over_https(url, expected):
    assert data_module._https(url) == expected


@pytest.mark.parametrize("url, expected", [
    ("s3://overturemaps-us-west-2/release/2026-10-21.0/theme=places/type=place/*.parquet", "place"),
    ("gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2", "1959_2022_full_37_1h_0p25deg_chunk_1"),
    ("https://data.source.coop/cholmes/eurocrops/fields.parquet", "fields"),
])
def test_a_remote_location_names_its_table(url, expected):
    assert data_module._remote_name(url) == expected


def test_a_zarr_store_is_cut_to_the_variables_and_coordinates_asked_for(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    ds = xr.Dataset(
        {"t2m": (["time", "latitude", "longitude"], np.arange(24.0).reshape(2, 3, 4), {"units": "K"}),
         "u10": (["time", "latitude", "longitude"], np.zeros((2, 3, 4)))},
        coords={"time": pd.date_range("2020-07-01", periods=2, freq="h"), "latitude": [60.0, 50.0, 40.0],
                "longitude": [0.0, 5.0, 10.0, 15.0]},
    )
    store = tmp_path / "era5.zarr-v2"
    ds.to_zarr(store, consolidated=True, zarr_format=2)
    source = read(str(store), variables=["t2m"], sel={"time": "2020-07-01T01:00", "latitude": slice(60, 50)})
    assert isinstance(source, XArraySQLSource) and source.get_tables() == ["t2m"]
    data = source.execute("SELECT latitude, longitude, t2m FROM t2m")
    assert len(data) == 2 * 4 and data["t2m"].min() == 12.0


def test_a_remote_zarr_url_is_read_lazily_not_as_a_local_path(monkeypatch):
    opened = []
    monkeypatch.setattr(data_module, "_remote_grid", lambda url, variables, sel: opened.append(url) or "grid")
    assert resolve_data("gs://gcp-public-data-arco-era5/ar/x.zarr-v2") == ["grid"]
    assert opened == ["gs://gcp-public-data-arco-era5/ar/x.zarr-v2"]


def _features(path, bbox_column):
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    box = ", {'xmin': x, 'xmax': x, 'ymin': y, 'ymax': y} AS bbox" if bbox_column else ""
    con.execute(f"""
        COPY (SELECT name, ST_Point(x, y) AS geometry{box}
              FROM (VALUES ('berlin', 13.4, 52.5), ('paris', 2.35, 48.86), ('potsdam', 13.06, 52.39)) t(name, x, y))
        TO '{path}' (FORMAT PARQUET)
    """)


@pytest.mark.parametrize("bbox_column", [True, False])
def test_a_box_keeps_the_features_inside_it(tmp_path, bbox_column):
    path = tmp_path / "places.parquet"
    _features(path, bbox_column)
    source = read(str(path), bbox=(12.5, 52.0, 14.0, 53.0))
    assert isinstance(source, DuckDBSource) and source.get_tables() == ["places"]
    assert sorted(source.execute("SELECT name FROM places")["name"]) == ["berlin", "potsdam"]


def test_options_say_what_they_apply_to(tmp_path):
    with pytest.raises(ValueError, match="cut a Zarr store with sel"):
        read("gs://b/x.zarr", bbox=(0, 0, 1, 1))
    with pytest.raises(ValueError, match="cut \\(Geo\\)Parquet with bbox"):
        read("s3://b/x.parquet", variables=["a"])
    with pytest.raises(ValueError, match="apply to one file"):
        read([1, 2], bbox=(0, 0, 1, 1))


def test_lumen_read_passes_the_options_on(tmp_path):
    import lumen
    path = tmp_path / "places.parquet"
    _features(path, True)
    assert sorted(lumen.read(str(path), bbox=(12.5, 52.0, 14.0, 53.0)).execute("SELECT name FROM places")["name"]) == ["berlin", "potsdam"]
