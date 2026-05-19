"""
Tests for STACSource -- a lazy STAC catalog source for Lumen.

The STAC API and the xpystac->xarray resolution are fully mocked so these
tests run deterministically with no network access. A chosen collection is
resolved to a synthetic xarray dataset and queried through the delegated
XArraySQLSource (DataFusion).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pystac_client")
pytest.importorskip("xpystac")
xr = pytest.importorskip("xarray")
pytest.importorskip("xarray_sql")

from lumen.sources.stac import STACSource


def test_guard_message_when_xarray_missing():
    """A minimal env (no xarray) must still get the lumen[stac] message,
    not a bare 'No module named xarray'."""
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "xarray" or name.startswith("xarray."):
            raise ImportError("No module named 'xarray'")
        return real_import(name, *args, **kwargs)

    saved = {
        k: sys.modules.pop(k)
        for k in [m for m in sys.modules if m == "lumen.sources.stac"]
    }
    try:
        with patch("builtins.__import__", side_effect=blocked):
            with pytest.raises(ImportError, match=r"pip install lumen\[stac\]"):
                importlib.import_module("lumen.sources.stac")
    finally:
        sys.modules.update(saved)


# ---- Fixtures ----


@pytest.fixture
def synthetic_dataset():
    """Synthetic 3D dataset with known values for deterministic testing."""
    times = pd.date_range("2020-01-01", periods=10, freq="D")
    lats = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lons = np.array([100.0, 110.0, 120.0, 130.0])

    rng = np.random.default_rng(42)
    temp = rng.uniform(250, 310, (10, 5, 4))
    pressure = rng.uniform(950, 1050, (10, 5, 4))

    return xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], temp, {
                "units": "K", "long_name": "Air Temperature",
            }),
            "pressure": (["time", "lat", "lon"], pressure, {
                "units": "hPa", "long_name": "Surface Pressure",
            }),
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={"title": "Synthetic Test Dataset"},
    )


def _fake_asset(href, media_type=None, roles=None):
    return SimpleNamespace(href=href, media_type=media_type, roles=roles or [])


def _fake_collection(cid, assets=None, title=None, description=None,
                     cube_variables=None):
    extent = SimpleNamespace(
        spatial=SimpleNamespace(bboxes=[[-180, -90, 180, 90]]),
        temporal=SimpleNamespace(intervals=[["2020-01-01T00:00:00Z", None]]),
    )
    return SimpleNamespace(
        id=cid,
        title=title or f"{cid} title",
        description=description or f"{cid} description",
        extent=extent,
        license="CC-BY-4.0",
        keywords=["climate", cid],
        assets=assets or {},
        stac_extensions=[],
        cube_variables=cube_variables,
    )


class _FakeDatacube:
    """Stand-in for pystac DatacubeExtension keyed off the fake collection."""

    @staticmethod
    def has_extension(collection):
        return collection.cube_variables is not None

    @staticmethod
    def ext(collection):
        return SimpleNamespace(variables=collection.cube_variables)


@pytest.fixture
def mock_stac_client():
    """Patch pystac_client.Client.open to return a fake catalog client."""
    zarr_asset = _fake_asset(
        "s3://bucket/airtemp.zarr",
        media_type="application/vnd+zarr",
        roles=["data"],
    )
    collections = {
        "air-temperature": _fake_collection(
            "air-temperature", assets={"zarr": zarr_asset},
            title="Air Temperature", description="Gridded air temperature",
        ),
        "no-asset-collection": _fake_collection("no-asset-collection"),
    }

    client = SimpleNamespace()
    client.get_collections = lambda: list(collections.values())
    client.get_collection = lambda cid: collections[cid]

    def _search(*args, **kwargs):
        first_item = SimpleNamespace(id="item-0")
        return SimpleNamespace(items=lambda: iter([first_item]))

    client.search = _search

    with patch("pystac_client.Client.open", return_value=client) as opener:
        yield opener, client, collections


# ---- Lazy construction ----


class TestLazyConstruction:
    def test_construction_does_not_fetch_data(self, mock_stac_client):
        opener, _, _ = mock_stac_client
        with patch.object(STACSource, "_open_dataset") as open_ds:
            source = STACSource(url="https://example.com/stac")
            source.get_tables()
            open_ds.assert_not_called()
        opener.assert_called_once_with("https://example.com/stac")

    def test_url_is_required(self, mock_stac_client):
        with pytest.raises(ValueError, match="url"):
            STACSource()

    def test_get_tables_lists_collections(self, mock_stac_client):
        source = STACSource(url="https://example.com/stac")
        assert source.get_tables() == ["air-temperature", "no-asset-collection"]

    def test_get_tables_honors_collections_filter(self, mock_stac_client):
        source = STACSource(
            url="https://example.com/stac", collections=["air-temperature"]
        )
        assert source.get_tables() == ["air-temperature"]


# ---- Resolution / delegation ----


class TestResolution:
    def test_resolve_delegates_and_is_memoized(self, mock_stac_client, synthetic_dataset):
        source = STACSource(url="https://example.com/stac")
        with patch.object(
            STACSource, "_open_dataset", return_value=synthetic_dataset
        ) as open_ds:
            schema1 = source.get_schema("air-temperature")
            schema2 = source.get_schema("air-temperature")
        assert open_ds.call_count == 1
        assert schema1 == schema2

    def test_resolve_drops_scalar_grid_mapping_vars(self, mock_stac_client):
        """CF grid-mapping/CRS scalars (0-dim vars, e.g.
        lambert_conformal_conic) are not tables and break chunking."""
        import numpy as np

        ds = xr.Dataset(
            {
                "prcp": (("time",), np.arange(4.0)),
                "lambert_conformal_conic": ((), np.int32(0)),
            },
            coords={"time": np.arange(4)},
        )
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset", return_value=ds):
            delegate = source._resolve("air-temperature")
        assert delegate.get_tables() == ["prcp"]

    def test_resolve_drops_cf_bounds_dims(self, mock_stac_client):
        """CF bounds vars (a size-2 'nv'/'bnds' dim, e.g. time_bnds) break
        XArraySQLSource chunking; _resolve must drop them."""
        import numpy as np

        ds = xr.Dataset(
            {
                "temp": (("time",), np.arange(4.0)),
                "time_bnds": (("time", "nv"), np.zeros((4, 2))),
            },
            coords={"time": np.arange(4)},
        )
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset", return_value=ds):
            delegate = source._resolve("air-temperature")
        assert "temp" in delegate.get_tables()
        assert "time_bnds" not in delegate.get_tables()

    def test_resolve_unifies_inconsistent_chunks(self, mock_stac_client):
        """Cloud Zarr (e.g. Planetary Computer daymet) can have inconsistent
        dask chunks; _resolve must unify them before XArraySQLSource."""
        pytest.importorskip("dask.array")
        import numpy as np

        # Two vars whose 'time' chunking differs -> inconsistent chunks.
        ds = xr.Dataset(
            {
                "a": (("time",), np.arange(6)),
                "b": (("time",), np.arange(6)),
            },
            coords={"time": np.arange(6)},
        ).chunk({"time": 2})
        ds["b"] = ds["b"].chunk({"time": 3})

        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset", return_value=ds):
            delegate = source._resolve("air-temperature")
        assert sorted(delegate.get_tables()) == ["a", "b"]

    def test_bare_collection_selects_primary_variable(self, mock_stac_client, synthetic_dataset):
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset", return_value=synthetic_dataset):
            schema = source.get_schema("air-temperature")
        # primary == first delegate table (alphabetical: 'pressure')
        assert "pressure" in schema

    def test_collection_variable_selects_named(self, mock_stac_client, synthetic_dataset):
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset", return_value=synthetic_dataset):
            df = source.get("air-temperature:temperature")
        assert "temperature" in df.columns
        assert len(df) == 10 * 5 * 4

    def test_execute_routes_to_resolved_delegate(self, mock_stac_client, synthetic_dataset):
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset", return_value=synthetic_dataset):
            source.get("air-temperature:temperature")  # resolve it
            df = source.execute("SELECT * FROM temperature LIMIT 5")
        assert len(df) == 5

    def test_get_on_unopenable_collection_raises_clear_error(self, mock_stac_client):
        """Data access on an unopenable collection must fail loudly & clearly."""
        source = STACSource(url="https://example.com/stac")
        with patch.object(
            STACSource, "_open_dataset",
            side_effect=AttributeError("'NoneType' object has no attribute 'variables'"),
        ):
            with pytest.raises(ValueError, match="air-temperature"):
                source.get("air-temperature")

    def test_get_on_none_dataset_raises_clear_error(self, mock_stac_client):
        """If xpystac returns None, surface a clear error, not an AttributeError."""
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset", return_value=None):
            with pytest.raises(ValueError, match="could not be opened as an xarray"):
                source.get("air-temperature")

    def test_get_schema_delegates_with_metadata_len(self, mock_stac_client, synthetic_dataset):
        """get_schema delegates to the resolved XArraySQLSource (no duplicated
        logic); __len__ is the O(1) metadata count from the delegate."""
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset", return_value=synthetic_dataset):
            schema = source.get_schema("air-temperature")
        # primary var == first data var ('pressure'); 10*5*4 = 200 elements
        assert schema["__len__"] == 200
        assert "pressure" in schema
        assert "time" in schema

    def test_schema_on_unopenable_collection_is_empty_not_raising(self, mock_stac_client):
        """Schema is best-effort: an unopenable collection must NOT abort
        catalog enumeration (real STAC catalogs mix xarray + non-xarray)."""
        source = STACSource(url="https://example.com/stac")
        with patch.object(
            STACSource, "_open_dataset", side_effect=RuntimeError("not xarray")
        ):
            assert source.get_schema("air-temperature") == {}

    def test_get_sql_expr_is_lazy(self, mock_stac_client):
        """get_sql_expr must be a cheap string op (no resolution) so the AI's
        catalog gathering does not resolve every collection."""
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset") as open_ds:
            expr = source.get_sql_expr("air-temperature")
            open_ds.assert_not_called()
        assert "air-temperature" in expr

    def test_create_sql_expr_source_resolves_and_remaps(self, mock_stac_client, synthetic_dataset):
        """The 'Select Data to Explore' path: create_sql_expr_source must
        resolve the collection and remap the SQL to the delegate's variable,
        producing a queryable source (no NotImplementedError)."""
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset", return_value=synthetic_dataset):
            new_source = source.create_sql_expr_source(
                {"select_air": 'SELECT * FROM "air-temperature"'}
            )
            df = new_source.get("select_air")
        assert len(df) == 10 * 5 * 4          # full primary variable
        assert "pressure" in df.columns       # delegate's first data var

    def test_execute_without_resolution_raises(self, mock_stac_client):
        source = STACSource(url="https://example.com/stac")
        with pytest.raises(ValueError, match="collection"):
            source.execute("SELECT 1")


# ---- Openable-asset selection ----


class TestAssetSelection:
    def test_prefers_xarray_openable_collection_asset(self, mock_stac_client):
        source = STACSource(url="https://example.com/stac")
        collection = mock_stac_client[2]["air-temperature"]
        obj = source._openable_candidates(collection)[0]
        assert obj.href == "s3://bucket/airtemp.zarr"

    def test_prefers_public_https_over_cloud_protocol_asset(self, mock_stac_client):
        """A public https zarr must win over an abfs/s3 zarr needing creds."""
        source = STACSource(url="https://example.com/stac")
        collection = _fake_collection(
            "multi",
            assets={
                "zarr-abfs": _fake_asset(
                    "abfs://bucket/data.zarr",
                    media_type="application/vnd+zarr",
                    roles=["data", "zarr", "abfs"],
                ),
                "zarr-https": _fake_asset(
                    "https://host/data.zarr",
                    media_type="application/vnd+zarr",
                    roles=["data", "zarr", "https"],
                ),
            },
        )
        obj = source._openable_candidates(collection)[0]
        assert obj.href == "https://host/data.zarr"

    def test_falls_back_to_first_item(self, mock_stac_client):
        source = STACSource(url="https://example.com/stac")
        collection = mock_stac_client[2]["no-asset-collection"]
        obj = source._openable_candidates(collection)[0]
        assert obj.id == "item-0"

    def test_raises_when_nothing_openable(self, mock_stac_client):
        source = STACSource(url="https://example.com/stac")
        collection = _fake_collection("empty")
        with patch.object(
            source._client, "search",
            return_value=SimpleNamespace(items=lambda: iter([])),
        ):
            with pytest.raises(ValueError, match="no xarray-openable"):
                source._openable_candidates(collection)


# ---- Lifecycle ----


class TestCatalogCaching:
    def test_listing_caches_collections_no_per_id_refetch(self, mock_stac_client):
        """get_metadata over all collections must not issue one network
        get_collection() call per collection after get_tables() listed them."""
        opener, client, collections = mock_stac_client
        calls = {"n": 0}
        real_get = client.get_collection

        def counting_get(cid):
            calls["n"] += 1
            return real_get(cid)

        client.get_collection = counting_get
        source = STACSource(url="https://example.com/stac")
        source.get_tables()           # warms the cache from get_collections()
        source.get_metadata(None)     # metadata for every collection
        assert calls["n"] == 0


class TestDiscoveryMetadata:
    def test_metadata_is_lazy_and_rich(self, mock_stac_client):
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset") as open_ds:
            md = source.get_metadata("air-temperature")
            open_ds.assert_not_called()
        desc = md["description"]
        assert "Air Temperature" in desc
        assert "Gridded air temperature" in desc
        assert "CC-BY-4.0" in desc
        assert "climate" in desc
        assert "-180" in desc            # spatial bbox folded in
        assert "2020-01-01" in desc      # temporal extent folded in

    def test_metadata_for_all_tables_does_not_resolve(self, mock_stac_client):
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset") as open_ds:
            md = source.get_metadata(None)
            open_ds.assert_not_called()
        assert set(md) == {"air-temperature", "no-asset-collection"}
        assert "Air Temperature" in md["air-temperature"]["description"]

    def test_metadata_columns_from_datacube_extension(self, mock_stac_client):
        from lumen.sources import stac as stac_mod
        cube = {
            "prcp": SimpleNamespace(
                dimensions=["time", "y", "x"], unit="mm/day",
                var_type="data", description="daily total precipitation",
            ),
            "tmax": SimpleNamespace(
                dimensions=["time", "y", "x"], unit="degC",
                var_type="data", description="daily maximum temperature",
            ),
        }
        coll = _fake_collection("daymet", cube_variables=cube)
        source = STACSource(url="https://example.com/stac")
        with patch.object(stac_mod, "DatacubeExtension", _FakeDatacube), \
             patch.object(STACSource, "_get_collection", return_value=coll), \
             patch.object(STACSource, "_open_dataset") as open_ds:
            cols = source.get_metadata("daymet")["columns"]
            open_ds.assert_not_called()  # still lazy
        assert set(cols) == {"prcp", "tmax"}
        assert "mm/day" in cols["prcp"]["description"]
        assert "precipitation" in cols["prcp"]["description"]

    def test_metadata_columns_empty_when_no_datacube(self, mock_stac_client):
        # Real DatacubeExtension on a plain fake collection -> no extension.
        source = STACSource(url="https://example.com/stac")
        assert source.get_metadata("air-temperature")["columns"] == {}

    def test_metadata_enrichment_never_raises(self, mock_stac_client):
        from lumen.sources import stac as stac_mod

        class _Broken:
            @staticmethod
            def has_extension(c):
                return True

            @staticmethod
            def ext(c):
                raise RuntimeError("malformed datacube extension")

        with patch.object(stac_mod, "DatacubeExtension", _Broken):
            source = STACSource(url="https://example.com/stac")
            md = source.get_metadata(None)
        assert set(md) == {"air-temperature", "no-asset-collection"}
        assert md["air-temperature"]["columns"] == {}


class TestSigning:
    def test_patch_url_is_forwarded_to_open_dataset(self, mock_stac_client):
        signer = lambda u: u  # noqa: E731
        source = STACSource(url="https://example.com/stac", patch_url=signer)
        captured = {}

        def fake_open(obj, engine=None, **kw):
            captured.update(kw)
            import xarray as xr
            return xr.Dataset()

        with patch("xarray.open_dataset", side_effect=fake_open):
            source._open_dataset(object())
        assert captured.get("patch_url") is signer

    def test_planetary_computer_url_auto_signs_object(self):
        from types import SimpleNamespace
        sign = lambda o: o  # noqa: E731
        fake_pc = SimpleNamespace(sign=sign)
        client = SimpleNamespace(get_collections=lambda: [])
        with patch("pystac_client.Client.open", return_value=client), \
             patch("lumen.sources.stac.planetary_computer", fake_pc):
            source = STACSource(
                url="https://planetarycomputer.microsoft.com/api/stac/v1"
            )
            assert source._object_signer() is sign

    def test_non_pc_url_has_no_default_signer(self, mock_stac_client):
        source = STACSource(url="https://example.com/stac")
        assert source._object_signer() is None

    def test_resolve_signs_object_and_falls_back(self, mock_stac_client, synthetic_dataset):
        """The object signer is applied, and resolution falls back to the
        next openable candidate when the first fails."""
        source = STACSource(url="https://example.com/stac")
        seen = {}

        def fake_signer(obj):
            seen["signed"] = obj
            return obj

        calls = {"n": 0}

        def fake_open(obj):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first asset unreadable")
            return synthetic_dataset

        collection = _fake_collection(
            "multi",
            assets={
                "a": _fake_asset("https://h/a.zarr", "application/vnd+zarr", ["data"]),
                "b": _fake_asset("https://h/b.zarr", "application/vnd+zarr", ["data"]),
            },
        )
        with patch.object(STACSource, "_object_signer", return_value=fake_signer), \
             patch.object(STACSource, "_get_collection", return_value=collection), \
             patch.object(STACSource, "_open_dataset", side_effect=fake_open):
            delegate = source._resolve("multi")
        assert "signed" in seen          # signer was applied
        assert calls["n"] == 2           # fell back to the 2nd candidate
        assert delegate.get_tables()     # produced a working delegate


class TestLifecycle:
    def test_close_closes_delegates(self, mock_stac_client, synthetic_dataset):
        source = STACSource(url="https://example.com/stac")
        with patch.object(STACSource, "_open_dataset", return_value=synthetic_dataset):
            source.get_schema("air-temperature")
        delegate = source._delegates["air-temperature"]
        with patch.object(delegate, "close") as delegate_close:
            source.close()
        delegate_close.assert_called_once()
