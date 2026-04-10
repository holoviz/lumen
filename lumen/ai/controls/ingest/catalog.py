from __future__ import annotations

import asyncio

import pandas as pd
import param

from panel.pane import Markdown
from panel.widgets import Tabulator
from panel_material_ui import Column as MuiColumn

from .base import BaseSourceControls, SourceResult

# ─────────────────────────────────────────────────────────────────────────────
# CATALOG SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

class CatalogSourceControls(BaseSourceControls):
    """
    Base class for controls that browse a pre-fetched catalog of datasets,
    select entries, and fetch them on demand.

    Subclasses implement three methods:

    - ``_load_catalog() -> pd.DataFrame``  — fetch the registry at startup
    - ``_fetch_entry(entry: pd.Series) -> SourceResult``  — fetch one dataset
    - ``_entry_to_text(entry: pd.Series) -> str``  — text for vector embedding
      (default joins ``search_columns``; override for custom representations)

    Class-level attributes configure the Tabulator display:

    - ``display_columns``  — ``{col: {"title": ..., "width": ..., "formatter": ...}}``
    - ``filter_columns``   — ``{col: tabulator header_filter config dict}``
    - ``search_columns``   — column names concatenated for vector embedding
    - ``detail_columns``   — column names shown in expanded row detail view
    """

    catalog_df = param.DataFrame(default=None, doc="Full catalog DataFrame after loading.")

    display_columns = param.Dict(default={}, doc="""
        Mapping of column name to Tabulator config dict.
        Supported keys: 'title' (str), 'width' (str), 'formatter' (dict).""")

    search_columns = param.List(default=[], doc="""
        Column names whose values are concatenated to produce the text
        representation used for vector embedding.""")

    filter_columns = param.Dict(default={}, doc="""
        Mapping of column name to Tabulator header_filter config dict.
        e.g. {"name": {"type": "input", "func": "like", "placeholder": "Filter..."}}""")

    detail_columns = param.List(default=[], doc="""
        Column names displayed in the expanded row detail view.""")

    vector_store = param.Parameter(default=None, doc="""
        Optional VectorStore instance. When provided, catalog entries are
        embedded in the background after _load_catalog() completes so the
        agent's VectorLookupTool can search the catalog.""")

    load_mode = "manual"  # Row clicks trigger loading, not a button

    label = "Catalog"

    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        self._layout.loading = True
        import panel as pn
        pn.state.onload(self._load_catalog_wrapper)

    def _render_controls(self) -> list:
        """Build an empty Tabulator browser; columns are configured once data loads."""
        self._tabulator = Tabulator(
            page_size=5,
            pagination="local",
            sizing_mode="stretch_width",
            show_index=False,
            buttons={"download": '<i class="fa fa-download"></i>'},
            on_click=self._on_row_click,
        )

        return [
            Markdown("*Click on download icons to ingest datasets.*", margin=(0, 10)),
            self._tabulator,
        ]

    def _configure_tabulator(self):
        """Apply display_columns / filter_columns config to the Tabulator.

        Called from _load_catalog_wrapper AFTER the Tabulator's value has
        been set, so that columns already exist and the config (titles,
        widths, header_filters, etc.) is applied correctly.
        """
        display_cols = self.display_columns
        self._tabulator.param.update(
            titles={col: cfg.get("title", col) for col, cfg in display_cols.items()},
            widths={col: cfg.get("width", "auto") for col, cfg in display_cols.items()},
            formatters={col: cfg["formatter"] for col, cfg in display_cols.items() if "formatter" in cfg},
            editors={col: None for col in display_cols},
            header_filters=self.filter_columns,
            row_content=self._get_row_content if self.detail_columns else None,
        )

    def _render_layout(self):
        """Build layout — no load button for catalog controls."""
        controls = self._render_controls()
        return MuiColumn(
            *controls,
            self._error_placeholder,
            self._message_placeholder,
            self.progress.bar,
            self.progress.description,
            sizing_mode="stretch_width",
            margin=(10, 15),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Subclass contract
    # ──────────────────────────────────────────────────────────────────────────

    async def _load_catalog(self) -> pd.DataFrame:
        """
        Fetch the catalog. Called once at startup via ``pn.state.onload``.
        Must return a DataFrame whose columns include those listed in
        ``display_columns``, ``filter_columns``, ``search_columns``, and
        ``detail_columns``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _load_catalog()"
        )

    async def _fetch_entry(self, entry: pd.Series) -> SourceResult:
        """
        Download and process a single catalog entry.
        ``entry`` is the full row from ``catalog_df``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _fetch_entry(entry)"
        )

    def _entry_to_text(self, entry: pd.Series) -> str:
        """
        Convert a catalog entry to a searchable text string for vector
        embedding.  Default implementation joins ``search_columns``.
        Override for custom representations.
        """
        parts = []
        for col in self.search_columns:
            if col in entry.index and pd.notna(entry[col]):
                parts.append(f"{col}: {entry[col]}")
        return ". ".join(parts)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal machinery
    # ──────────────────────────────────────────────────────────────────────────

    async def _load_catalog_wrapper(self):
        """Load catalog at startup, populate the Tabulator, and start embedding."""
        try:
            self.catalog_df = await self._load_catalog()
            # Build the display DataFrame
            if self.display_columns:
                cols = [c for c in self.display_columns if c in self.catalog_df.columns]
                display_df = self.catalog_df[cols]
            else:
                display_df = self.catalog_df
            # Set value FIRST so the Tabulator knows its columns,
            # THEN apply titles/widths/filters/formatters on top.
            self._tabulator.value = display_df
            self._configure_tabulator()
        except Exception as e:
            self._show_message(f"Failed to load catalog: {e}", error=True)
        finally:
            self._layout.loading = False

        # Background embed — fire and forget, same pattern as SourceCatalog
        if self.vector_store is not None and self.catalog_df is not None:
            asyncio.create_task(self._embed_catalog())  # noqa: RUF006

    async def _on_row_click(self, event):
        """Handle Tabulator row click — delegates to _run_load."""
        await self._run_load(self._fetch_entry_by_index(event.row))

    async def _fetch_entry_by_index(self, row_idx: int) -> SourceResult:
        """Look up the full catalog row and delegate to _fetch_entry."""
        if self.catalog_df is None:
            return SourceResult.empty("Catalog not yet loaded.")
        entry = self.catalog_df.iloc[row_idx]
        return await self._fetch_entry(entry)

    def _get_row_content(self, row):
        """Render expanded row detail using detail_columns."""
        lines = []
        for col in self.detail_columns:
            if col in row and pd.notna(row[col]):
                lines.append(f"  {col}: {row[col]}")
        return Markdown(
            "\n".join(lines),
            sizing_mode="stretch_width",
        )

    async def _embed_catalog(self):
        """
        Embed all catalog entries into the vector store for agent search.
        Called as a background asyncio task after _load_catalog completes.
        Follows the SourceCatalog._sync_metadata_to_vector_store pattern.
        """
        if self.vector_store is None or self.catalog_df is None:
            return

        items = []
        for _, row in self.catalog_df.iterrows():
            text = self._entry_to_text(row)
            if not text:
                continue

            # Structured metadata for post-semantic filtering
            # VectorStore metadata values must be str/int/float/bool
            metadata = {"type": "catalog_entry"}
            for col in self.filter_columns:
                if col in row.index and pd.notna(row[col]):
                    val = row[col]
                    metadata[col] = val if isinstance(val, (str, int, float, bool)) else str(val)

            items.append({"text": text, "metadata": metadata})

        if items:
            await self.vector_store.upsert(items)
