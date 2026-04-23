from __future__ import annotations

import asyncio
import re

import pandas as pd
import panel as pn
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

    _supports_tools = True

    _cached_catalog_tools: list[tuple[str, callable]] | None = None

    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        self._layout.loading = True
        pn.state.onload(self._load_catalog_wrapper)

    def _render_controls(self) -> list:
        """Build a Tabulator browser pre-configured with column settings.

        An empty DataFrame with the correct column names is used so that
        the Tabulator creates its column definitions up-front.  When the
        real data arrives in ``_load_catalog_wrapper``, only the row data
        changes — columns are never rebuilt, which avoids a race where
        column configuration (titles, widths, formatters) is lost during
        a client-side column rebuild.
        """
        display_cols = self.display_columns
        if display_cols:
            empty_df = pd.DataFrame({col: pd.Series(dtype="object") for col in display_cols})
        else:
            empty_df = pd.DataFrame()

        tabulator_kwargs = dict(
            value=empty_df,
            page_size=5,
            pagination="local",
            sizing_mode="stretch_width",
            show_index=False,
            buttons={"download": '<i class="fa fa-download"></i>'},
            on_click=self._on_row_click,
        )
        if display_cols:
            tabulator_kwargs.update(
                titles={col: cfg.get("title", col) for col, cfg in display_cols.items()},
                widths={col: cfg.get("width", "auto") for col, cfg in display_cols.items()},
                formatters={
                    col: cfg["formatter"]
                    for col, cfg in display_cols.items()
                    if "formatter" in cfg
                },
                editors={col: None for col in display_cols},
                header_filters=self.filter_columns,
            )
        if self.detail_columns:
            tabulator_kwargs["row_content"] = self._get_row_content

        self._tabulator = Tabulator(**tabulator_kwargs)

        return [
            Markdown("*Click on download icons to ingest datasets.*", margin=(0, 10)),
            self._tabulator,
        ]

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
            # Only update the row data — columns were already
            # configured in _render_controls with an empty DataFrame
            # that has the correct column names.
            self._tabulator.value = display_df
        except Exception as e:
            self._show_message(f"Failed to load catalog: {e}", error=True)
        finally:
            self._layout.loading = False

        # Background embed — fire and forget, same pattern as SourceCatalog
        if self.vector_store is not None and self.catalog_df is not None:
            asyncio.create_task(self._embed())  # noqa: RUF006

    async def _on_row_click(self, event):
        """Handle Tabulator row click — delegates to _run_load."""
        await self._run_load(self._fetch_entry_by_index(event.row))

    async def _fetch_entry_by_index(self, row_idx: int) -> SourceResult:
        """Look up the full catalog row and delegate to _fetch_entry."""
        if self.catalog_df is None:
            return SourceResult.empty("Catalog not yet loaded.")
        entry = self.catalog_df.iloc[row_idx]
        return await self._fetch_entry(entry)

    async def load_entry(self, row_idx: int) -> SourceResult:
        """Load a catalog entry by DataFrame index.

        Public API for agents — runs the full load lifecycle
        (progress, source registration, outputs trigger) and
        returns the ``SourceResult`` so callers can inspect it.
        """
        return await self._run_load(self._fetch_entry_by_index(row_idx))

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

    async def _embed(self):
        """
        Embed all catalog entries into the vector store for agent search.
        Called as a background asyncio task after _load_catalog completes.
        Follows the SourceCatalog._sync_metadata_to_vector_store pattern.
        """
        if self.vector_store is None or self.catalog_df is None:
            return

        items = []
        for idx, row in self.catalog_df.iterrows():
            text = self._entry_to_text(row)
            if not text:
                continue

            # Structured metadata for post-semantic filtering
            # VectorStore metadata values must be str/int/float/bool
            metadata = {
                "type": "catalog_entry",
                "_row_idx": int(idx) if not isinstance(idx, int) else idx,
                "_control_id": id(self),
            }
            for col in self.filter_columns:
                if col in row.index and pd.notna(row[col]):
                    val = row[col]
                    metadata[col] = val if isinstance(val, (str, int, float, bool)) else str(val)

            items.append({"text": text, "metadata": metadata})

        if items:
            await self.vector_store.upsert(items)

    # ──────────────────────────────────────────────────────────────────────────
    # Agent / tool integration
    # ──────────────────────────────────────────────────────────────────────────

    def _tool_name(self) -> str:
        """Display name for the tool returned by ``as_tools()``.

        Override in subclasses for a custom name.
        """
        clean = re.sub(r'<[^>]+>', '', self.label).strip() if self.label else type(self).__name__
        return f"Search and load from {clean}"

    async def _load_from_query(self, query: str) -> SourceResult:
        """Search the catalog for a dataset matching the query and load it.

        Args:
            query: Natural language description, dataset name, collection
                name, or identifier to search for in the catalog.
        """
        if self.catalog_df is None or self.catalog_df.empty:
            return SourceResult.empty("Catalog not yet loaded.")

        match_idx = await self._search_catalog(query)
        if match_idx is None:
            return SourceResult.empty(
                f"No dataset matching '{query}' found in catalog."
            )

        entry = self.catalog_df.iloc[match_idx]
        try:
            result = await self._fetch_entry(entry)
            # Register the source on the control so the UI's
            # _sync_sources watcher fires and the SourceCatalog
            # ("Available Sources") is updated.  When invoked
            # via the Tabulator click path this happens inside
            # _run_load → _handle_success; the SourceAgent path
            # bypasses _run_load so we do it explicitly here.
            if result and result.sources:
                existing = self.context.get("sources", [])
                for src in result.sources:
                    if src not in existing:
                        self._register_source_output(src)
                self.param.trigger("outputs")
            return result
        finally:
            # _fetch_entry may update the progress bar (e.g.
            # "Processing …") but only _run_load clears it.
            # When called via SourceAgent we bypass _run_load,
            # so clean up here to avoid a stuck progress bar.
            self.progress.clear()

    def as_tools(
        self, query: str | None = None, top_k: int = 5,
    ) -> list[tuple[str, callable]]:
        """Return ``(name, callable)`` pairs for SourceAgent / SourceLookup.

        Exposes one tool that searches the catalog via vector similarity
        (or text fallback) and loads the best-matching entry.
        Results are cached so ``_control_hash`` remains stable.
        """
        if self._cached_catalog_tools is not None:
            return self._cached_catalog_tools
        tools = [(self._tool_name(), self._load_from_query)]
        self._cached_catalog_tools = tools
        return tools

    async def _search_catalog(self, query: str) -> int | None:
        """Find the best matching catalog row index for *query*.

        Uses vector search when a ``vector_store`` is available and
        falls back to keyword matching on ``search_columns``.
        """
        # ── vector search path ──
        if self.vector_store is not None:
            try:
                results = await self.vector_store.query(query, top_k=5)
                for r in results:
                    meta = r.get("metadata", {})
                    if (
                        meta.get("type") == "catalog_entry"
                        and meta.get("_control_id") == id(self)
                    ):
                        return meta["_row_idx"]
            except Exception:
                pass  # fall through to text search

        # ── text fallback ──
        if self.catalog_df is None or not self.search_columns:
            return None

        query_lower = query.lower()
        query_words = query_lower.split()
        best_idx = None
        best_score = 0

        for idx, row in self.catalog_df.iterrows():
            text = self._entry_to_text(row).lower()
            score = sum(1 for word in query_words if word in text)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx if best_score > 0 else None
