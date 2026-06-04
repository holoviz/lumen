"""Tests for STACCatalogControls -- the CatalogSourceControls subclass
that lets SourceAgent browse / search a STAC API in chat.

The STAC API and the asset->xarray resolution are mocked at the network
boundary (the same way test_stac.py does it) so tests run deterministically
with no real HTTP traffic. Each test exercises real STACCatalogControls
and real STACSource code.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

pytest.importorskip("pystac_client")
pytest.importorskip("xpystac")
pytest.importorskip("xarray")
pytest.importorskip("xarray_sql")

from lumen.ai.controls.ingest.stac import STACCatalogControls
from lumen.sources.stac import STACSource


def _fake_collection(cid, *, title=None, description=None,
                     keywords=None, license_="CC-BY-4.0"):
    extent = SimpleNamespace(
        spatial=SimpleNamespace(bboxes=[[-180, -90, 180, 90]]),
        temporal=SimpleNamespace(intervals=[["2020-01-01T00:00:00Z", None]]),
    )
    return SimpleNamespace(
        id=cid,
        title=title or f"{cid} title",
        description=description or f"{cid} description",
        extent=extent,
        license=license_,
        keywords=keywords or ["climate", cid],
        assets={},
        stac_extensions=[],
        cube_variables=None,
    )


@pytest.fixture
def mock_stac_api():
    """Patch pystac_client.Client.open to serve two collections."""
    collections = {
        "daymet": _fake_collection(
            "daymet", title="Daymet Daily HI",
            description="Daily surface weather data for Hawaii",
            keywords=["weather", "daily"],
        ),
        "era5": _fake_collection(
            "era5", title="ERA5 reanalysis",
            description="ECMWF reanalysis",
            license_="ECMWF-open",
        ),
    }
    client = SimpleNamespace(
        get_collections=lambda: list(collections.values()),
        get_collection=lambda cid: collections[cid],
        search=lambda *a, **k: SimpleNamespace(items=lambda: iter([])),
    )
    with patch("pystac_client.Client.open", return_value=client) as opener:
        yield opener, client, collections


class TestSTACCatalogControls:

    @pytest.mark.asyncio
    async def test_load_catalog_yields_collection_dataframe(self, mock_stac_api, context, source_catalog):
        controls = STACCatalogControls(
            url="https://example.com/stac",
            context=context,
            source_catalog=source_catalog,
        )
        df = await controls._load_catalog()
        assert isinstance(df, pd.DataFrame)
        assert set(df["id"]) == {"daymet", "era5"}
        # display / search columns must be present so the Tabulator and
        # vector store have something to render / embed
        for col in ("id", "title", "description", "license", "keywords"):
            assert col in df.columns
        row = df.set_index("id").loc["daymet"]
        assert "Daymet" in row["title"]
        assert "Hawaii" in row["description"]
        assert "weather" in row["keywords"]

    @pytest.mark.asyncio
    async def test_fetch_entry_returns_source_result_with_scoped_stac_source(
        self, mock_stac_api, context, source_catalog,
    ):
        controls = STACCatalogControls(
            url="https://example.com/stac",
            context=context,
            source_catalog=source_catalog,
        )
        df = await controls._load_catalog()
        result = await controls._fetch_entry(df.iloc[0])
        assert len(result.sources) == 1
        source = result.sources[0]
        assert isinstance(source, STACSource)
        assert source.url == "https://example.com/stac"
        assert source.collections == [df.iloc[0]["id"]]
        assert result.table == df.iloc[0]["id"]
        assert result.message and df.iloc[0]["id"] in result.message

    def test_as_tools_exposes_search_and_load_tool(self, mock_stac_api, context, source_catalog):
        controls = STACCatalogControls(
            url="https://example.com/stac",
            context=context,
            source_catalog=source_catalog,
        )
        tools = controls.as_tools()
        assert len(tools) == 1
        name, callable_ = tools[0]
        assert name.startswith("Search and load from")
        assert callable(callable_)
