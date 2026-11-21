import asyncio

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from lumen.ai.controls import CatalogSourceControls, SourceResult
from lumen.sources.duckdb import DuckDBSource


class _CatalogControls(CatalogSourceControls):
    search_columns = ["name"]

    def __init__(self, catalog_df, **params):
        self._catalog = catalog_df
        super().__init__(**params)

    async def _load_catalog(self):
        return self._catalog

    async def _fetch_entry(self, entry):
        return SourceResult(table=entry["name"], message=entry["name"])


async def _make_controls(catalog_df, **params):
    controls = _CatalogControls(catalog_df, **params)
    await asyncio.sleep(0)
    return controls


@pytest.mark.asyncio
async def test_load_from_query_uses_position_with_noncontiguous_index():
    controls = await _make_controls(
        pd.DataFrame({"name": ["first", "target"]}, index=[10, 20]),
        context={},
    )

    result = await controls._load_from_query("target")

    assert result.table == "target"


@pytest.mark.asyncio
async def test_load_from_query_uses_position_with_duplicate_index():
    first = pd.DataFrame({"name": ["first", "second"]})
    second = pd.DataFrame({"name": ["third", "target"]})
    controls = await _make_controls(pd.concat([first, second]), context={})

    result = await controls._load_from_query("target")

    assert result.table == "target"


@pytest.mark.asyncio
async def test_load_from_query_stores_positions_for_multiindex():
    index = pd.MultiIndex.from_tuples([("source", 1), ("source", 2)])
    controls = await _make_controls(
        pd.DataFrame({"name": ["first", "target"]}, index=index),
        context={},
    )
    controls.vector_store = AsyncMock()

    await controls._embed()

    items = controls.vector_store.upsert.await_args.args[0]
    assert [item["metadata"]["_row_idx"] for item in items] == [0, 1]

    controls.vector_store.query.return_value = [
        {
            "metadata": {
                "type": "catalog_entry",
                "_control_id": id(controls),
                "_row_idx": 1,
            }
        }
    ]
    result = await controls._load_from_query("target")

    assert result.table == "target"


@pytest.mark.asyncio
async def test_load_from_query_handles_stale_vector_position():
    controls = await _make_controls(
        pd.DataFrame({"name": ["target"]}),
        context={},
    )
    controls.vector_store = AsyncMock()
    controls.vector_store.query.return_value = [
        {
            "metadata": {
                "type": "catalog_entry",
                "_control_id": id(controls),
                "_row_idx": 10,
            }
        }
    ]
    controls.progress("Searching catalog")

    result = await controls._load_from_query("target")

    assert result.message == "No dataset matching 'target' found in catalog."
    assert controls.progress.visible is False


@pytest.mark.asyncio
async def test_load_from_query_registers_complete_output_once():
    source = DuckDBSource.from_df(tables={"target": pd.DataFrame({"value": [1]})})
    controls = await _make_controls(
        pd.DataFrame({"name": ["target"]}, index=[10]),
        context={"sources": [source]},
    )
    source_result = SourceResult.from_source(source, table="target")

    with patch.object(controls, "_fetch_entry", AsyncMock(return_value=source_result)):
        await controls._load_from_query("target")
        await controls._load_from_query("target")

    assert controls.outputs["source"] is source
    assert controls.outputs["sources"] == [source]
    assert controls.outputs["table"] == "target"
    assert controls._last_table == "target"
