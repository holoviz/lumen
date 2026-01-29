from __future__ import annotations

import asyncio

import param

from panel.layout import Column
from panel.pane.markup import Markdown
from panel.viewable import Viewer
from panel_material_ui import Tree

from ..config import SOURCE_TABLE_SEPARATOR


class SourceCatalog(Viewer):
    """
    A component that displays all data sources and documents as hierarchical trees.
    """

    context = param.Dict(default={})

    sources = param.List(default=[], doc="""
        List of data sources to display in the catalog.""")

    vector_store = param.Parameter(default=None, doc="""
        The vector store to sync metadata documents to. If not provided,
        metadata files will only be stored locally.""")

    visibility_changed = param.Event(doc="""
        Triggered when table or document visibility changes.""")

    def __init__(self, /, context=None, **params):
        if context is None:
            raise ValueError("SourceCatalog must be given a context dictionary.")
        if "source" in context and "sources" not in context:
            context["sources"] = [context["source"]]

        super().__init__(context=context, **params)

        # === Sources Tree ===
        self._sources_title = Markdown("**Data Sources**", margin=(0, 10))
        self._sources_tree = Tree(
            items=[],
            checkboxes=True,
            propagate_to_child=True,
            color="primary",
            sizing_mode="stretch_width",
            margin=(0, 10),
        )
        self._sources_tree.param.watch(self._on_sources_active_change, "active")
        self._sources_tree.on_action("Delete", self._on_delete_source)

        # === Global Documents Tree ===
        self._docs_title = Markdown("**Apply to All Tables**", margin=(10, 10, 0, 10), visible=False)
        self._docs_tree = Tree(
            items=[],
            checkboxes=True,
            color="secondary",
            sizing_mode="stretch_width",
            margin=(0, 10),
            visible=False,
        )
        self._docs_tree.param.watch(self._on_docs_active_change, "active")

        # Combined layout
        self._layout = Column(
            self._docs_title,
            self._docs_tree,
            self._sources_title,
            self._sources_tree,
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0)
        )

        # Track the mapping from tree paths to source/table/metadata
        self._path_map = {}  # path tuple -> {"source": ..., "table": ..., "metadata": ...}
        self._docs_path_map = {}  # index -> filename
        self._suppress_sources_callback = False
        self._suppress_docs_callback = False

        # Store available metadata files
        self._available_metadata = []

    def _build_sources_items(self, sources: list) -> list[dict]:
        items = []
        available_metadata = self._available_metadata
        multiple_sources = len(sources) > 1
        self._path_map.clear()

        for src_idx, source in enumerate(sources):
            tables = source.get_tables()

            # Build table items
            table_items = []
            for tbl_idx, table in enumerate(tables):
                # Get table metadata for secondary text
                table_meta = source.metadata.get(table, {}) if source.metadata else {}
                secondary_parts = [
                    f"{k}: {v}" for k, v in table_meta.items()
                    if k not in ("docs", "columns")
                ]

                # Build metadata file items as children of this table
                metadata_items = []
                for meta_idx, meta in enumerate(available_metadata):
                    meta_path = (src_idx, tbl_idx, meta_idx)
                    self._path_map[meta_path] = {
                        "source": source,
                        "table": table,
                        "metadata": meta["filename"],
                    }
                    metadata_items.append({
                        "label": meta["filename"],
                        "icon": "description",
                    })

                table_path = (src_idx, tbl_idx)
                self._path_map[table_path] = {
                    "source": source,
                    "table": table,
                    "metadata": None,
                }

                table_item = {
                    "label": table,
                    "icon": "list",
                }
                if secondary_parts:
                    table_item["secondary"] = "; ".join(secondary_parts)
                if metadata_items:
                    table_item["items"] = metadata_items
                table_items.append(table_item)

            # Source-level path
            source_path = (src_idx,)
            self._path_map[source_path] = {
                "source": source,
                "table": None,
                "metadata": None,
            }

            source_item = {
                "label": source.name,
                "icon": "source",
            }
            if multiple_sources:
                source_item["actions"] = [{"label": "Delete", "icon": "delete"}]
            if table_items:
                source_item["items"] = table_items
            items.append(source_item)

        return items

    def _build_docs_items(self) -> list[dict]:
        items = []
        self._docs_path_map.clear()

        for idx, meta in enumerate(self._available_metadata):
            self._docs_path_map[idx] = meta["filename"]
            items.append({
                "label": meta["filename"],
                "icon": "description",
            })

        return items

    def _compute_sources_active_paths(self, sources: list) -> list[tuple]:
        active = []
        visible_slugs = self.context.get("visible_slugs", [])
        available_metadata = self._available_metadata
        meta_filenames = [m["filename"] for m in available_metadata]

        for src_idx, source in enumerate(sources):
            tables = source.get_tables()
            all_tables_visible = True

            for tbl_idx, table in enumerate(tables):
                table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}"
                table_is_visible = table_slug in visible_slugs

                if table_is_visible:
                    active.append((src_idx, tbl_idx))
                else:
                    all_tables_visible = False

                if source.metadata:
                    associations = source.metadata.get(table, {}).get("docs", [])
                    for meta_filename in associations:
                        if meta_filename in meta_filenames:
                            meta_idx = meta_filenames.index(meta_filename)
                            active.append((src_idx, tbl_idx, meta_idx))

            if all_tables_visible and len(tables) > 0:
                active.append((src_idx,))

        return active

    def _compute_docs_active_paths(self) -> list[tuple]:
        sources = self.sources or self.context.get("sources", [])
        if not sources:
            return []

        total_tables = sum(len(source.get_tables()) for source in sources)
        if total_tables == 0:
            return []

        active = []
        for idx, meta in enumerate(self._available_metadata):
            filename = meta["filename"]
            association_count = 0
            for source in sources:
                if not source.metadata:
                    continue
                for table in source.get_tables():
                    if table in source.metadata:
                        if filename in source.metadata[table].get("docs", []):
                            association_count += 1
            if association_count == total_tables:
                active.append((idx,))

        return active

    def _compute_expanded_paths(self, sources: list) -> list[tuple]:
        return [(src_idx,) for src_idx in range(len(sources))]

    def _on_sources_active_change(self, event):
        if self._suppress_sources_callback:
            return

        active_paths = set(event.new)

        for path, info in self._path_map.items():
            if info["table"] is None:
                continue

            if info["metadata"] is None:
                self._update_table_visibility(path, info, active_paths)
            else:
                self._update_table_doc_association(path, info, active_paths)

        self._sync_docs_tree()
        self.param.trigger('visibility_changed')

    def _update_table_visibility(self, path: tuple, info: dict, active_paths: set):
        source = info["source"]
        table = info["table"]
        table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}"

        if path in active_paths:
            self.context["visible_slugs"].add(table_slug)
        else:
            self.context["visible_slugs"].discard(table_slug)

    def _update_table_doc_association(self, path: tuple, info: dict, active_paths: set):
        source = info["source"]
        table = info["table"]
        metadata = info["metadata"]

        self._ensure_source_metadata(source)
        associations = self._get_table_docs(source, table)

        if path in active_paths:
            if metadata not in associations:
                associations.append(metadata)
        elif metadata in associations:
            associations.remove(metadata)

    def _on_docs_active_change(self, event):
        if self._suppress_docs_callback:
            return

        active_indices = {path[0] for path in event.new if len(path) == 1}
        sources = self.sources or self.context.get("sources", [])

        if "visible_docs" not in self.context:
            self.context["visible_docs"] = set()

        for idx, filename in self._docs_path_map.items():
            is_active = idx in active_indices
            if is_active:
                self._associate_doc_with_all_tables(filename, sources)
                self.context["visible_docs"].add(filename)
            else:
                self._remove_doc_from_all_tables(filename, sources)
                self.context["visible_docs"].discard(filename)

        self._sync_sources_tree_only()
        self.param.trigger('visibility_changed')

    def _associate_doc_with_all_tables(self, filename: str, sources: list):
        for source in sources:
            self._ensure_source_metadata(source)
            for table in source.get_tables():
                associations = self._get_table_docs(source, table)
                if filename not in associations:
                    associations.append(filename)

    def _remove_doc_from_all_tables(self, filename: str, sources: list):
        for source in sources:
            if not source.metadata:
                continue

            for table in source.get_tables():
                if table not in source.metadata or "docs" not in source.metadata[table]:
                    continue

                associations = source.metadata[table]["docs"]
                if filename in associations:
                    associations.remove(filename)

    def _ensure_source_metadata(self, source):
        if source.metadata is None:
            source.metadata = {}

    def _get_table_docs(self, source, table: str) -> list:
        if table not in source.metadata:
            source.metadata[table] = {}
        if "docs" not in source.metadata[table]:
            source.metadata[table]["docs"] = []
        return source.metadata[table]["docs"]

    def _on_delete_source(self, item: dict):
        source_name = item["label"]
        sources = self.context.get("sources", [])

        source_to_delete = self._find_source_by_name(sources, source_name)
        if not source_to_delete:
            return

        self._remove_source_tables_from_visible_slugs(source_to_delete)
        self._remove_source_from_context(sources, source_to_delete)

    def _find_source_by_name(self, sources: list, source_name: str):
        for source in sources:
            if source.name == source_name:
                return source
        return None

    def _remove_source_tables_from_visible_slugs(self, source):
        for table in source.get_tables():
            table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}"
            self.context["visible_slugs"].discard(table_slug)

    def _remove_source_from_context(self, sources: list, source_to_delete):
        self.context["sources"] = [s for s in sources if s is not source_to_delete]
        self.sources = self.context["sources"]

    async def _sync_metadata_to_vector_store(self, filename: str):
        if self.vector_store is None:
            return

        available = {
            m["filename"]: m for m in self._available_metadata
        }

        if filename not in available:
            return

        metadata_info = available[filename]
        content = metadata_info.get("content", "")

        doc_entry = {
            "text": content,
            "metadata": {
                "filename": filename,
                "type": "document",
            },
        }

        try:
            await self.vector_store.upsert([doc_entry])
        except Exception:
            raise

    @param.depends("sources", watch=True)
    def _on_sources_change(self):
        self.sync()

    def _sync_sources_tree_only(self):
        sources = self.sources or self.context.get("sources", [])
        if not sources:
            return
        items = self._build_sources_items(sources)
        if not items:
            return
        self._update_sources_tree_state(items, sources)

    def sync(self, context=None):
        context = context or self.context
        sources = self.sources or context.get("sources", [])
        self._sync_sources_tree(sources)
        self._sync_docs_tree()
        self._layout.loading = False

    def _sync_sources_tree(self, sources: list):
        if not sources:
            self._sources_title.object = "**Available Sources** *(no sources available)*"
            self._sources_tree.items = []
            return

        items = self._build_sources_items(sources)
        if not items:
            self._sources_title.object = "**Available Sources** *(no tables found)*"
            self._sources_tree.items = []
            return

        self._sources_title.object = "**Available Sources**"
        self._auto_associate_unassociated_metadata(sources)
        self._update_sources_tree_state(items, sources)

    def _auto_associate_unassociated_metadata(self, sources: list):
        meta_filenames = [m["filename"] for m in self._available_metadata]
        associated = self._collect_associated_metadata(sources)
        unassociated = set(meta_filenames) - associated

        if not unassociated:
            return

        if "visible_docs" not in self.context:
            self.context["visible_docs"] = set()

        for source in sources:
            for table in source.get_tables():
                table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}"
                self._ensure_source_metadata(source)
                associations = self._get_table_docs(source, table)
                for meta_filename in unassociated:
                    if meta_filename not in associations:
                        associations.append(meta_filename)
                self.context["visible_slugs"].add(table_slug)

        for meta_filename in unassociated:
            self.context["visible_docs"].add(meta_filename)

        for meta_filename in unassociated:
            asyncio.create_task(self._sync_metadata_to_vector_store(meta_filename))  # noqa: RUF006

    def _collect_associated_metadata(self, sources: list) -> set:
        associated = set()
        for source in sources:
            if not source.metadata:
                continue
            for table in source.get_tables():
                if table in source.metadata:
                    associations = source.metadata[table].get("docs", [])
                    associated.update(associations)
        return associated

    def _update_sources_tree_state(self, items: list, sources: list):
        active = self._compute_sources_active_paths(sources)
        expanded = self._compute_expanded_paths(sources)
        self._suppress_sources_callback = True
        try:
            self._sources_tree.items = items
            self._sources_tree.active = active
            self._sources_tree.expanded = expanded
        finally:
            self._suppress_sources_callback = False

    def _sync_docs_tree(self):
        if not self._available_metadata:
            self._docs_title.visible = False
            self._docs_tree.visible = False
            return

        self._docs_title.visible = True
        self._docs_title.object = "**Apply to All Tables**"
        self._docs_tree.visible = True

        docs_items = self._build_docs_items()
        docs_active = self._compute_docs_active_paths()

        self._suppress_docs_callback = True
        try:
            self._docs_tree.items = docs_items
            self._docs_tree.active = docs_active
        finally:
            self._suppress_docs_callback = False

    def __panel__(self):
        return self._layout
