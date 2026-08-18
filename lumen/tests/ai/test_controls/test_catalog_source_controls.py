from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from lumen.ai.controls import CatalogSourceControls, SourceResult
from lumen.sources.duckdb import DuckDBSource


class _CatalogControls(CatalogSourceControls):
    search_columns = ["name"]

    async def _load_catalog(self):
        return pd.DataFrame()

    async def _fetch_entry(self, entry):
        return SourceResult(table=entry["name"], message=entry["name"])


@pytest.mark.asyncio
async def test_load_from_query_uses_noncontiguous_integer_index_label():
    controls = _CatalogControls(context={})
    controls.catalog_df = pd.DataFrame(
        {"name": ["first", "target"]}, index=[10, 20]
    )

    result = await controls._load_from_query("target")

    assert result.table == "target"


@pytest.mark.asyncio
async def test_load_from_query_preserves_string_index_through_vector_store():
    controls = _CatalogControls(context={})
    controls.catalog_df = pd.DataFrame(
        {"name": ["first", "target"]}, index=["alpha", "beta"]
    )
    controls.vector_store = AsyncMock()

    await controls._embed()

    items = controls.vector_store.upsert.await_args.args[0]
    assert [item["metadata"]["_row_idx"] for item in items] == ["alpha", "beta"]

    controls.vector_store.query.return_value = [
        {
            "metadata": {
                "type": "catalog_entry",
                "_control_id": id(controls),
                "_row_idx": "beta",
            }
        }
    ]
    result = await controls._load_from_query("target")

    assert result.table == "target"


@pytest.mark.asyncio
async def test_load_from_query_registers_source_output_once_when_in_context():
    source = DuckDBSource.from_df(tables={"target": pd.DataFrame({"value": [1]})})
    controls = _CatalogControls(context={"sources": [source]})
    controls.catalog_df = pd.DataFrame({"name": ["target"]}, index=[10])
    source_result = SourceResult.from_source(source, table="target")

    with patch.object(controls, "_fetch_entry", AsyncMock(return_value=source_result)):
        await controls._load_from_query("target")
        await controls._load_from_query("target")

    assert controls.outputs["source"] is source
    assert controls.outputs["sources"] == [source]
