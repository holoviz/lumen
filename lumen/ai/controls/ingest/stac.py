"""STACCatalogControls -- make a STAC API browsable by SourceAgent.

Wraps a STAC API endpoint as a ``CatalogSourceControls`` subclass so each
cataloged collection becomes a clickable / searchable row in the chat UI.
Picking a row (either by click or via the ``SourceAgent`` tool returned
by ``as_tools()``) instantiates a ``STACSource`` scoped to that collection
and adds it to the session via the standard ``SourceResult`` channel.
"""
from __future__ import annotations

import pandas as pd
import param

try:
    import pystac_client
except ImportError as e:
    raise ImportError(
        "STACCatalogControls requires the 'pystac-client' package. "
        "Install it with: pip install 'lumen[stac]'"
    ) from e

from ....sources.stac import STACSource
from .catalog import CatalogSourceControls
from .result import SourceResult


class STACCatalogControls(CatalogSourceControls):
    """Browse a STAC API in chat.

    Every cataloged collection is exposed as a row; selecting one builds
    a ``STACSource`` for that collection alone and hands it to the
    surrounding UI through ``SourceResult``.
    """

    url = param.String(default=None, allow_None=True, doc="""
        Root URL of a STAC API endpoint (e.g. a STAC API landing page).""")

    label = "STAC catalog"

    display_columns = {
        "id": {"title": "Collection", "width": "25%"},
        "title": {"title": "Title", "width": "30%"},
        "description": {"title": "Description", "width": "35%"},
        "license": {"title": "License", "width": "10%"},
    }

    search_columns = ["id", "title", "description", "keywords"]

    filter_columns = {
        "license": {"type": "input", "func": "like", "placeholder": "License..."},
    }

    detail_columns = ["description", "keywords"]

    async def _load_catalog(self) -> pd.DataFrame:
        if not self.url:
            raise ValueError(
                "STACCatalogControls requires a 'url' pointing to a "
                "STAC API endpoint."
            )
        client = pystac_client.Client.open(self.url)
        rows = []
        for collection in client.get_collections():
            rows.append({
                "id": collection.id,
                "title": collection.title or collection.id,
                # Trim long descriptions so the Tabulator row stays readable;
                # the full description is available via the detail panel.
                "description": (collection.description or "")[:300],
                "license": collection.license or "",
                "keywords": ", ".join(collection.keywords or []),
            })
        return pd.DataFrame(rows)

    async def _fetch_entry(self, entry: pd.Series) -> SourceResult:
        # Scope a fresh STACSource to just the picked collection so the
        # session source map stays focused on what the user asked for.
        source = STACSource(url=self.url, collections=[entry["id"]])
        return SourceResult(
            sources=[source],
            table=entry["id"],
            message=f"Loaded STAC collection {entry['id']!r}",
        )
