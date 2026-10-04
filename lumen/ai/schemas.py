from __future__ import annotations

import asyncio
import warnings

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

from .config import SOURCE_TABLE_SEPARATOR
from .table_stats import (
    ESTIMATED, SAMPLED, TYPES_ONLY, ColumnStats, TableStats, get_stats_store,
    render_column, render_stats_header,
)
from .utils import (
    INDEXED_COLUMN_RE, collapse_indexed_columns, count_tokens, get_schema,
    log_debug, slug_to_table_name, truncate_string,
)

if TYPE_CHECKING:
    from ..sources import Source

# Pandas dtype names some sources report as column types; they say less than
# the statistics next to them, so they are not rendered as SQL types.
_PANDAS_DTYPES = {"object", "string", "str", "category"}

SAMPLED_NOTE = (
    "Stats from a sample or engine estimates can miss rare values; confirm "
    "literal values with run_exploration_sql before filtering on them."
)
# Catalog metadata keys that duplicate what the table rendering already shows.
_METADATA_EXCLUDE = frozenset({'columns', 'data_type', 'source_name', 'rows', 'description'})


def _extra_metadata(entry: TableCatalogEntry) -> dict[str, Any]:
    return {
        k: v for k, v in (entry.metadata or {}).items()
        if k not in _METADATA_EXCLUDE and v is not None and v != ''
    }


DEGRADED_NOTE = (
    "Some tables list names and types only; call load_table_schemas for their statistics."
)


@dataclass
class Column:
    name: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """A chunk of document text with relevance metadata."""
    filename: str
    text: str
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.text} (from: {self.filename}, relevance: {self.similarity:.2f})"


@dataclass
class TableCatalogEntry:
    table_slug: str
    similarity: float
    columns: list[Column]
    source: Source | None = None
    description: str | None = None
    sql_expr: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    derived_from: list[str] = field(default_factory=list)  # parent table slugs; empty = original
    created_order: int = 0  # session-scoped creation index; 0 = original, 1/2/3... = derived


@dataclass
class Metaset:
    """
    Schema container for table metadata with optional SQL enrichment and documents.

    Contains:
    - catalog: Table discovery results (descriptions, columns, similarity)
    - schemas: Optional SQL schema data (types, enums, row counts)
    - docs: Document chunks (already filtered by MetadataLookup based on visible_docs)
    - schema_tables: Tables to show full schemas for (if None, uses top n by similarity)
    """

    query: str | None
    catalog: dict[str, TableCatalogEntry]
    schemas: dict[str, dict[str, Any]] | None = None
    docs: list[DocumentChunk] | None = None
    schema_tables: list[str] | None = None
    stats: dict[str, TableStats] = field(default_factory=dict)

    @property
    def has_schemas(self) -> bool:
        return self.schemas is not None and len(self.schemas) > 0

    async def ensure_stats(
        self, table_slugs: list[str] | None = None, timeout: float | None = None
    ) -> dict[str, TableStats]:
        """
        Load column statistics for `table_slugs` (default: every catalog table),
        waiting at most `timeout` seconds; tables still being profiled render
        names and types until a later call finds them ready.
        """
        slugs = [s for s in (table_slugs or list(self.catalog)) if s not in self.stats]
        by_source: dict[int, tuple[Source, dict[str, str]]] = {}
        for slug in slugs:
            entry = self.catalog.get(slug)
            if entry is None or entry.source is None:
                continue
            _, by_table = by_source.setdefault(id(entry.source), (entry.source, {}))
            by_table[slug_to_table_name(slug)] = slug
        store = get_stats_store()
        results = await asyncio.gather(*(
            store.ensure(source, list(by_table), timeout=timeout)
            for source, by_table in by_source.values()
        ))
        for (_, by_table), found in zip(by_source.values(), results, strict=False):
            for table, stats in found.items():
                self.stats[by_table[table]] = stats
        return self.stats

    async def get_schema(self, table_slug: str) -> dict[str, Any] | None:
        """Deprecated: the legacy schema dict for one table; use `get_stats`."""
        warnings.warn(
            "Metaset.get_schema is deprecated, use Metaset.ensure_stats and "
            "Metaset.get_stats instead.", DeprecationWarning, stacklevel=2,
        )
        if self.schemas is None:
            self.schemas = {}
        if table_slug in self.schemas:
            return self.schemas[table_slug]
        entry = self.catalog.get(table_slug)
        if not entry or not entry.source:
            return None
        schema = await get_schema(entry.source, slug_to_table_name(table_slug), include_count=True)
        self.schemas[table_slug] = schema
        return schema

    async def ensure_schemas(self, table_slugs: list[str] | None = None) -> None:
        """Deprecated: use `ensure_stats`."""
        warnings.warn(
            "Metaset.ensure_schemas is deprecated, use Metaset.ensure_stats instead.",
            DeprecationWarning, stacklevel=2,
        )
        await self.ensure_stats(table_slugs)

    def get_stats(self, table_slug: str) -> TableStats | None:
        """Statistics for a table, adapting a legacy schema dict if that is all there is."""
        if table_slug in self.stats:
            return self.stats[table_slug]
        schema = (self.schemas or {}).get(table_slug)
        if isinstance(schema, dict) and schema:
            return TableStats.from_json_schema(slug_to_table_name(table_slug), schema)
        entry = self.catalog.get(table_slug)
        if entry is not None and entry.source is not None:
            # Profiled in the background since this metaset was built.
            try:
                stats = get_stats_store().get(entry.source, slug_to_table_name(table_slug))
            except Exception:
                stats = None
            if stats is not None:
                self.stats[table_slug] = stats
                return stats
        return None

    @property
    def has_docs(self) -> bool:
        return self.docs is not None and len(self.docs) > 0

    def get_docs(self) -> list[DocumentChunk]:
        """Get document chunks (already filtered by MetadataLookup)."""
        return self.docs if self.docs else []

    def docs_retrieval_stats(self) -> tuple[int, float, float] | None:
        """
        Return (chunk_count, min_similarity, max_similarity) for retrieved docs, or None if none.
        """
        docs = self.get_docs()
        if not docs:
            return None
        sims = [float(c.similarity) for c in docs]
        return (len(docs), min(sims), max(sims))

    def _build_table_data(
        self,
        catalog_entry: TableCatalogEntry,
        include_columns: bool,
        truncate: bool,
        include_sql: bool = True,
        include_metadata: bool = True,
        include_lineage: bool = False,
        max_order: int = 0,
    ) -> dict:
        """YAML-ready summary of a table without column statistics."""
        data = {}

        if include_sql:
            sql_expr = catalog_entry.sql_expr
            if truncate and sql_expr:
                sql_expr = truncate_string(sql_expr, max_length=200)
            data['read_with'] = sql_expr

        if catalog_entry.description:
            desc = catalog_entry.description
            if truncate:
                desc = truncate_string(desc, max_length=100)
            data['info'] = desc

        if include_metadata and (clean_metadata := _extra_metadata(catalog_entry)):
            data['metadata'] = clean_metadata

        if include_lineage and catalog_entry.derived_from:
            data['derived_from'] = [
                slug_to_table_name(p) for p in catalog_entry.derived_from
            ]
            data['step'] = catalog_entry.created_order
            if catalog_entry.created_order == max_order:
                data['latest'] = True

        if catalog_entry.columns and include_columns:
            # Collapse large numbered column series (e.g. embedding/PCA matrices).
            data['columns'] = collapse_indexed_columns([col.name for col in catalog_entry.columns])
        return data

    @property
    def single_source(self) -> bool:
        sources = {slug.split(SOURCE_TABLE_SEPARATOR, 1)[0] for slug in self.catalog if SOURCE_TABLE_SEPARATOR in slug}
        return len(sources) == 1

    def display_name(self, table_slug: str) -> str:
        """The identifier prompts, tools and errors use: the bare table name when there is one source."""
        if self.single_source and SOURCE_TABLE_SEPARATOR in table_slug:
            return table_slug.split(SOURCE_TABLE_SEPARATOR, 1)[1]
        return table_slug

    def table_detail(self, table_slug: str, include_lineage: bool = True) -> str:
        """Every statistic available for one table, untruncated."""
        entry = self.catalog[table_slug]
        max_order = max((e.created_order for e in self.catalog.values()), default=0)
        return self._render_table(
            table_slug, self.display_name(table_slug), entry, "full", truncate=False,
            include_sql=False, include_metadata=True, include_lineage=include_lineage,
            max_order=max_order,
        )

    def _render_table(
        self,
        table_slug: str,
        display_slug: str,
        entry: TableCatalogEntry,
        detail: str,
        truncate: bool,
        include_sql: bool,
        include_metadata: bool,
        include_lineage: bool,
        max_order: int,
    ) -> str:
        """
        Render one table as a header line followed by one line per column.

        `detail` is ``"full"``, ``"types"`` (names, types and keys) or
        ``"names"`` (a single line listing the columns).
        """
        stats = self.get_stats(table_slug)
        header = display_slug
        if note := render_stats_header(stats):
            header += f" ({note})"
        lines = [header]
        if entry.description:
            desc = " ".join(entry.description.split())
            lines.append(f"  info: {truncate_string(desc, max_length=100) if truncate else desc}")
        if include_lineage and entry.derived_from:
            parents = ", ".join(slug_to_table_name(p) for p in entry.derived_from)
            step = f"step {entry.created_order}" + (", latest" if entry.created_order == max_order else "")
            lines.append(f"  derived_from: {parents} ({step})")
        if include_metadata and (extra := _extra_metadata(entry)):
            lines.append("  metadata: " + ", ".join(f"{k}={v}" for k, v in extra.items()))
        if include_sql and entry.sql_expr:
            sql = " ".join(entry.sql_expr.split())
            lines.append(f"  read_with: {truncate_string(sql, max_length=200) if truncate else sql}")

        catalog_cols = {col.name: col for col in entry.columns}
        if stats is not None and stats.columns:
            columns = list(stats.columns)
            # Catalog columns the engine did not report, e.g. xarray coordinates.
            known = {c.name for c in columns}
            columns += [self._column_from_catalog(c) for c in entry.columns if c.name not in known]
        else:
            columns = [self._column_from_catalog(c) for c in entry.columns]
        if not columns:
            return "\n".join(lines)

        if detail == "names":
            names = collapse_indexed_columns([c.name for c in columns])
            lines.append("  columns: " + ", ".join(names))
            return "\n".join(lines)

        for label, members in self._column_groups(columns):
            col = members[0]
            description = None
            if len(members) == 1 and (cat := catalog_cols.get(col.name)) and cat.description:
                description = truncate_string(cat.description, max_length=100) if truncate else cat.description
            if len(members) > 1:
                col = self._merge_series(members)
            lines.append("  " + render_column(col, stats, description, detail=detail, name=label))
        return "\n".join(lines)

    @staticmethod
    def _column_from_catalog(column: Column) -> ColumnStats:
        data_type = (column.metadata or {}).get("data_type")
        if not data_type or str(data_type).lower() in _PANDAS_DTYPES:
            data_type = None
        return ColumnStats(name=column.name, type=str(data_type) if data_type else None)

    @staticmethod
    def _column_groups(columns: list[ColumnStats]) -> list[tuple[str | None, list[ColumnStats]]]:
        """Group numbered column series (embeddings, one-hot expansions) onto one line."""
        names = [c.name for c in columns]
        collapsed = collapse_indexed_columns(names)
        if len(collapsed) == len(names):
            return [(None, [c]) for c in columns]
        by_name = {c.name: c for c in columns}
        groups = []
        for label in collapsed:
            if label in by_name:
                groups.append((None, [by_name[label]]))
                continue
            stem = INDEXED_COLUMN_RE.match(label.split("..", 1)[0]).group("stem")
            members = [
                c for c in columns
                if (m := INDEXED_COLUMN_RE.match(c.name)) and m.group("stem") == stem
            ]
            groups.append((label, members))
        return groups

    @staticmethod
    def _merge_series(members: list[ColumnStats]) -> ColumnStats:
        first = members[0]
        merged = ColumnStats(name=first.name, type=first.type, kind=first.kind)
        mins = [m.min for m in members if m.min is not None]
        maxs = [m.max for m in members if m.max is not None]
        try:
            merged.min, merged.max = (min(mins), max(maxs)) if mins and maxs else (None, None)
        except TypeError:
            pass
        nulls = [m.nulls for m in members if m.nulls is not None]
        merged.nulls = max(nulls) if nulls else None
        return merged

    def _generate_schema_context(
        self,
        primary_slugs: list[str],
        display: dict[str, str],
        truncate: bool,
        include_sql: bool,
        include_metadata: bool,
        include_lineage: bool,
        max_order: int,
        max_tokens: int | None,
    ) -> tuple[str, list[str]]:
        """
        Render primary tables with full statistics until `max_tokens` is
        reached, then with names and types, then as bare column lists.
        Returns the text and the slugs that did not fit at all.
        """
        blocks, overflow, degraded, uncertain = [], [], False, False
        used = 0
        # Tables are in relevance order, so once one table had to drop
        # detail no less relevant table gets more.
        details = ("full", "types", "names")
        for slug in primary_slugs:
            entry = self.catalog[slug]
            rendered = None
            for i, detail in enumerate(details):
                block = self._render_table(
                    slug, display[slug], entry, detail, truncate, include_sql,
                    include_metadata, include_lineage, max_order,
                )
                cost = count_tokens(block) if max_tokens is not None else 0
                if max_tokens is None or used + cost <= max_tokens:
                    rendered = block
                    used += cost
                    degraded |= detail != "full"
                    details = details[i:]
                    break
            if rendered is None:
                overflow.append(slug)
                details = ()
                continue
            stats = self.get_stats(slug)
            if stats is not None and stats.method in (SAMPLED, ESTIMATED) and detail == "full":
                uncertain = True
            if stats is None or stats.method == TYPES_ONLY:
                degraded = degraded or bool(entry.columns or (stats and stats.columns))
            blocks.append(rendered)
        text = "\n\n".join(blocks)
        notes = []
        if uncertain:
            notes.append(SAMPLED_NOTE)
        if degraded or overflow:
            notes.append(DEGRADED_NOTE)
        if notes and text:
            text += "\n\n" + "\n".join(notes)
        return text, overflow

    def _deduplicated_slugs(self) -> list[str]:
        """
        Return catalog slugs deduplicated by table name, keeping the
        newest entry (highest ``created_order``) when the same table
        appears under multiple source names after materialization.

        Sorted by ``created_order`` descending so the most recent
        derived tables appear first in prompts.
        """
        best: dict[str, tuple[int, str]] = {}  # table_name -> (created_order, slug)
        for slug, entry in self.catalog.items():
            tname = slug_to_table_name(slug)
            if tname not in best or entry.created_order > best[tname][0]:
                best[tname] = (entry.created_order, slug)
        return [slug for _, slug in sorted(best.values(), reverse=True)]

    def _generate_context(
        self,
        include_columns: bool = False,
        include_schema: bool = True,
        truncate: bool = False,
        include_sql: bool = True,
        include_metadata: bool = True,
        include_docs: bool = False,
        include_lineage: bool = False,
        n: int | None = None,
        offset: int = 0,
        show_source: bool | None = None,
        n_others: int = 0,
        schema_tables: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        # Deduplicate catalog slugs, keeping only the newest entry per
        # table name so stale materialization generations don't appear.
        active_slugs = set(self._deduplicated_slugs())

        # Determine which tables to show with full details. A per-call
        # schema_tables scopes this render without mutating the shared
        # instance; None means "not provided" so an explicit [] can still
        # scope to zero primary tables.
        effective_schema_tables = (
            self.schema_tables if schema_tables is None else schema_tables
        )
        if effective_schema_tables is not None:
            # Use explicitly set schema_tables, filtered to active slugs
            primary_slugs = [s for s in effective_schema_tables if s in self.catalog and s in active_slugs]
        else:
            # Fall back to top n by similarity, filtered to active slugs
            primary_slugs = [s for s in self.get_top_tables(n, offset) if s in active_slugs]

        # Auto-detect show_source: hide source prefix when there's only one source
        if show_source is None:
            show_source = not self.single_source

        # Precompute max created_order for "latest" annotation
        max_order = max((e.created_order for e in self.catalog.values()), default=0)

        def display_name(slug: str) -> str:
            return slug if show_source else self.display_name(slug)

        result = ""
        overflow: list[str] = []
        primary_slugs = [s for s in primary_slugs if s in self.catalog]
        if include_schema:
            result, overflow = self._generate_schema_context(
                primary_slugs, {s: display_name(s) for s in primary_slugs}, truncate,
                include_sql, include_metadata, include_lineage, max_order, max_tokens,
            )
            if overflow:
                primary_slugs = [s for s in primary_slugs if s not in overflow]
                n_others = max(n_others, len(overflow))
        else:
            tables_data = {
                display_name(slug): self._build_table_data(
                    self.catalog[slug], include_columns, truncate,
                    include_sql, include_metadata, include_lineage, max_order
                )
                for slug in primary_slugs
            }
            if tables_data:
                if not include_lineage and all(not data for data in tables_data.values()):
                    result = yaml.dump(list(tables_data.keys()), default_flow_style=False, allow_unicode=True)
                else:
                    result = yaml.dump(tables_data, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # Add other tables section
        if n_others > 0:
            primary_set = set(primary_slugs)
            other_slugs = list(dict.fromkeys(overflow + [
                slug for slug in self.catalog.keys()
                if slug not in primary_set and slug in active_slugs
            ]))[:n_others]
            if other_slugs:
                result += "\n\nOthers available:\n"
                for slug in other_slugs:
                    display_slug = display_name(slug)
                    # Add a short lineage hint for derived tables
                    entry = self.catalog.get(slug)
                    if include_lineage and entry and entry.derived_from:
                        parents = ", ".join(
                            slug_to_table_name(p) for p in entry.derived_from
                        )
                        marker = " ★" if entry.created_order == max_order else ""
                        result += f"- {display_slug} (from {parents}){marker}\n"
                    else:
                        result += f"- {display_slug}\n"

        # Add docs section
        if include_docs:
            docs = self.get_docs()
            if docs:
                if result:
                    result += "\n---\n\n"
                result += "<documentation>\n"
                for chunk in docs:
                    text = truncate_string(chunk.text, 300) if truncate else truncate_string(chunk.text, 800)
                    text = " ".join(text.split())
                    result += f'<doc source="{chunk.filename}">\n{text}\n</doc>\n'
                result += "</documentation>"

        return result or "No data sources or documentation available."

    def _resolve_table_slug(self, table_name: str | None) -> str | None:
        """Resolve a bare table name to its full catalog slug."""
        if not table_name:
            return None
        if table_name in self.catalog:
            return table_name
        return next(
            (s for s in self.catalog if slug_to_table_name(s) == table_name),
            None,
        )

    def get_top_tables(self, n: int | None = None, offset: int = 0) -> list[str]:
        """Get top n table slugs sorted by similarity.

        Parameters
        ----------
        n : int | None
            Number of tables to return. If None, returns all.
        offset : int
            Number of tables to skip from the start.
        """
        sorted_slugs = sorted(
            self.catalog.keys(),
            key=lambda s: self.catalog[s].similarity,
            reverse=True
        )
        if offset:
            sorted_slugs = sorted_slugs[offset:]
        if n is not None:
            sorted_slugs = sorted_slugs[:n]
        return sorted_slugs

    def table_list(self, n: int | None = None, offset: int = 0, show_source: bool | None = None, n_others: int = 0, include_metadata: bool = False, include_lineage: bool = False, **override_kwargs) -> str:
        """Generate minimal table listing for planning - just table names and columns without schema details."""
        generate_kwargs = {
            "include_columns": False,
            "include_schema": False,
            "truncate": False,
            "include_sql": False,
            "include_metadata": include_metadata,
            "include_docs": False,
            "include_lineage": include_lineage,
        }
        generate_kwargs.update(override_kwargs)
        return self._generate_context(**generate_kwargs, n=n, offset=offset, show_source=show_source, n_others=n_others)

    def table_context(self, n: int | None = None, offset: int = 0, show_source: bool | None = None, n_others: int = 0, include_metadata: bool = True, include_lineage: bool = True, **override_kwargs) -> str:
        generate_kwargs = {
            "include_columns": True,
            "include_schema": False,
            "truncate": False,
            "include_sql": False,
            "include_metadata": include_metadata,
            "include_docs": False,
            "include_lineage": include_lineage,
        }
        generate_kwargs.update(override_kwargs)
        return self._generate_context(**generate_kwargs, n=n, offset=offset, show_source=show_source, n_others=n_others)

    def full_context(self, n: int | None = None, offset: int = 0, show_source: bool | None = None, n_others: int = 0, include_metadata: bool = True, include_lineage: bool = True, **override_kwargs) -> str:
        generate_kwargs = {
            "include_columns": True,
            "include_schema": True,
            "truncate": False,
            "include_sql": False,
            "include_metadata": include_metadata,
            "include_docs": True,
            "include_lineage": include_lineage,
        }
        generate_kwargs.update(override_kwargs)
        return self._generate_context(**generate_kwargs, n=n, offset=offset, show_source=show_source, n_others=n_others)

    def compact_context(self, n: int | None = None, offset: int = 0, show_source: bool | None = None, n_others: int = 0, include_metadata: bool = True, include_lineage: bool = True, **override_kwargs) -> str:
        generate_kwargs = {
            "include_columns": True,
            "include_schema": True,
            "truncate": True,
            "include_sql": False,
            "include_metadata": include_metadata,
            "include_docs": False,
            "include_lineage": include_lineage,
        }
        generate_kwargs.update(override_kwargs)
        return self._generate_context(**generate_kwargs, n=n, offset=offset, show_source=show_source, n_others=n_others)

    def __str__(self) -> str:
        return self.table_context()


async def get_metaset(
    sources: list[Source],
    tables: list[str],
    prev: Metaset | None = None,
    stats_timeout: float | None = 30,
) -> Metaset:
    """
    Get the metaset for the given sources and tables.

    Parameters
    ----------
    sources: list[Source]
        The sources to get the metaset for.
    tables: list[str]
        The tables to get the metaset for.
    prev: Metaset | None
        Previous metaset to reuse cached data from.
    stats_timeout: float | None
        Seconds to wait for column statistics; tables still being
        profiled render names and types and keep profiling in the
        background.

    Returns
    -------
    metaset: Metaset
        The metaset for the given sources and tables.
    """
    catalog_data = {}

    for table_slug in tables:
        if SOURCE_TABLE_SEPARATOR in table_slug:
            source_name, table_name = table_slug.split(SOURCE_TABLE_SEPARATOR)
        elif len(sources) > 1:
            raise ValueError(
                f"Cannot resolve table {table_slug} without providing "
                "the source, when multiple sources are provided. Ensure "
                f"that you qualify the table name as follows:\n\n"
                f"    <source>{SOURCE_TABLE_SEPARATOR}<table>"
            )
        else:
            source_name = next(iter(sources)).name
            table_name = table_slug
            table_slug = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"

        source = next((s for s in sources if s.name == source_name), None)

        if prev and table_slug in prev.catalog:
            catalog_entry = prev.catalog[table_slug]
            # Update source reference in case it changed
            catalog_entry.source = source
        else:
            try:
                metadata = source.get_metadata(table_name)
            except Exception as e:
                log_debug(f"Failed to get metadata for table {table_name} in source {source_name}: {e}")
                metadata = {}

            catalog_entry = TableCatalogEntry(
                table_slug=table_slug,
                similarity=1,
                columns=[
                    Column(
                        name=col_name,
                        description=col_values.pop("description", None),
                        metadata=col_values
                    )
                    for col_name, col_values in metadata.get("columns", {}).items()
                ],
                source=source,
                sql_expr=source.get_sql_expr(source.normalize_table(table_name)),
                description=metadata.get("description"),
            )
        catalog_data[table_slug] = catalog_entry

    # Preserve docs from previous metaset if available
    docs = prev.docs if prev else None

    metaset = Metaset(
        query=None,
        catalog=catalog_data,
        schemas={
            slug: schema for slug, schema in (prev.schemas or {}).items() if slug in catalog_data
        } if prev and prev.schemas else None,
        docs=docs,
    )
    # Statistics are not carried over from `prev`: the store revalidates
    # them, so a table replaced since then is profiled again.
    await metaset.ensure_stats(list(catalog_data), timeout=stats_timeout)
    return metaset


@dataclass
class DbtslMetadata:
    name: str
    similarity: float
    description: str | None = None
    dimensions: dict[str, list] = field(default_factory=dict)
    queryable_granularities: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        dimensions_str = ""
        for dimension, dimension_spec in self.dimensions.items():
            dtype = dimension_spec["type"]
            enums = dimension_spec.get("enum", [])
            if enums:
                dimensions_str += f"- {dimension} ({dtype}): {enums}\n"
            else:
                dimensions_str += f"- {dimension} ({dtype})\n"
        return (
            f"Metric: {self.name} (Similarity: {self.similarity:.3f})\n"
            f"Info: {self.description}\n"
            f"Dims:\n{dimensions_str}\n"
            f"Granularities: {', '.join(self.queryable_granularities)}\n\n"
        )


@dataclass
class DbtslMetaset:
    query: str
    metrics: dict[str, DbtslMetadata]

    def __str__(self) -> str:
        context = "Below are the relevant metrics to use with DbtslAgent:\n\n"
        for metric in self.metrics.values():
            context += str(metric)
        return context
