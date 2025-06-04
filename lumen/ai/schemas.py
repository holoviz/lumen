from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..sources import Source
from .config import SOURCE_TABLE_SEPARATOR
from .utils import (
    get_schema, log_debug, truncate_iterable, truncate_string,
)


@dataclass
class Column:
    """Schema for a column with its description."""

    name: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorMetadata:
    """Schema for vector lookup data for a single table."""

    table_slug: str  # Combined source_name and table_name with separator
    similarity: float
    columns: list[Column]
    description: str | None = None
    base_sql: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorMetaset:
    """Schema container for vector data for multiple tables_metadata."""

    query: str
    vector_metadata_map: dict[str, VectorMetadata]
    selected_columns: dict[str, list[str]] = field(default_factory=dict)

    def _generate_context(self, truncate: bool | None = None) -> str:
        """
        Generate formatted text representation of the context.

        Args:
            truncate: Controls truncation behavior.
                    None: Show all tables (max_context)
                    False: Filter by selected_columns without truncation (selected_context)
                    True: Filter by selected_columns with truncation (min_context)
        """
        context = "Below are the relevant tables and columns available:\n\n"

        # Use selected tables if specified, otherwise use all tables
        tables_to_show = self.selected_columns or self.vector_metadata_map.keys()

        for table_slug in self.vector_metadata_map.keys():
            # Skip tables not in the selected list if using selected_context or min_context
            if truncate is not None and table_slug not in tables_to_show:
                continue

            vector_metadata = self.vector_metadata_map[table_slug]
            context += f"\n\n{table_slug!r}\n\nSimilarity: ({vector_metadata.similarity:.3f})\n\n"

            if vector_metadata.description:
                context += f"Info: {vector_metadata.description}\n\n"

            if vector_metadata.base_sql:
                context += f"Base SQL: {vector_metadata.base_sql}\n\n"

            max_length = 20
            cols_to_show = vector_metadata.columns or []
            if truncate is not None:
                cols_to_show = [col for col in cols_to_show if col.name in self.selected_columns.get(table_slug, [])]

            show_ellipsis = False
            if truncate:
                cols_to_show, original_indices, show_ellipsis = truncate_iterable(cols_to_show, max_length)
            else:
                cols_to_show = list(cols_to_show)
                original_indices = list(range(len(cols_to_show)))
                show_ellipsis = False

            for i, (col, orig_idx) in enumerate(zip(cols_to_show, original_indices)):
                if show_ellipsis and i == len(cols_to_show) // 2:
                    context += "...\n"

                if i == 0:
                    context += "Cols:\n"

                col_name = truncate_string(col.name) if truncate else col.name
                context += f"{orig_idx}. {col_name!r}"
                if col.description:
                    col_desc = truncate_string(col.description, max_length=100) if truncate else col.description
                    context += f": {col_desc}"
                context += "\n"
        return context

    @property
    def max_context(self) -> str:
        """Generate formatted text representation of the context."""
        return self._generate_context(truncate=None)

    @property
    def selected_context(self) -> str:
        """Generate formatted text representation of the context with selected tables cols"""
        return self._generate_context(truncate=False)

    @property
    def min_context(self) -> str:
        """Generate formatted text representation of the context with selected tables cols and truncated strings"""
        return self._generate_context(truncate=True)

    def __str__(self) -> str:
        """String representation is the formatted context."""
        return self.selected_context


@dataclass
class SQLMetadata:
    """Schema for SQL schema data for a single table."""

    table_slug: str
    schema: dict[str, Any]


@dataclass
class SQLMetaset:
    """Schema container for SQL data for multiple tables_metadata that builds on vector context."""

    vector_metaset: VectorMetaset
    sql_metadata_map: dict[str, SQLMetadata]

    def _generate_context(self, truncate: bool | None = None) -> str:
        """
        Generate formatted context with both vector and SQL data.

        Args:
            truncate: Controls truncation behavior.
                      None: Show all tables (max_context)
                      False: Filter by selected_columns without truncation (selected_context)
                      True: Filter by selected_columns with truncation (min_context)

        Returns:
            Formatted context string
        """
        context = "Below are the relevant tables and columns available:\n\n"

        vector_metaset = self.vector_metaset
        tables_to_show = vector_metaset.selected_columns or vector_metaset.vector_metadata_map.keys()

        for table_slug in self.sql_metadata_map.keys():
            # Skip tables not in selected list for sub/min context
            if truncate is not None and table_slug not in tables_to_show:
                continue

            vector_metadata = vector_metaset.vector_metadata_map.get(table_slug)
            if not vector_metadata:
                continue

            context += f"{table_slug!r} Similarity: ({vector_metadata.similarity:.3f})\n"

            if vector_metadata.description:
                desc = truncate_string(vector_metadata.description, max_length=100) if truncate else vector_metadata.description
                context += f"Info: {desc}\n"

            base_sql = truncate_string(vector_metadata.base_sql, max_length=200) if truncate else vector_metadata.base_sql
            context += f"Base SQL: {base_sql}\n"

            sql_data: SQLMetadata = self.sql_metadata_map.get(table_slug)
            if sql_data:
                # Get the count from schema
                if sql_data.schema.get("__len__"):
                    context += f"Row count: {len(sql_data.schema)}\n"

            max_length = 20
            cols_to_show = vector_metadata.columns or []
            # Ensure that we subset at least some columns
            if truncate is not None and any(len(cols) > 0 for cols in vector_metaset.selected_columns.values()):
                cols_to_show = [col for col in cols_to_show if col.name in vector_metaset.selected_columns.get(table_slug, [])]

            original_indices = []
            show_ellipsis = False
            if truncate:
                cols_to_show, original_indices, show_ellipsis = truncate_iterable(cols_to_show, max_length)
            else:
                cols_to_show = list(cols_to_show)
                original_indices = list(range(len(cols_to_show)))
                show_ellipsis = False

            for i, (col, orig_idx) in enumerate(zip(cols_to_show, original_indices)):
                if show_ellipsis and i == len(cols_to_show) // 2:
                    context += "...\n"

                if i == 0:
                    context += "Columns:\n"

                schema_data = None
                if sql_data and col.name in sql_data.schema:
                    schema_data = sql_data.schema[col.name]
                    if truncate is not None and schema_data == "<null>":
                        continue

                # Get column name with optional truncation
                col_name = truncate_string(col.name) if truncate else col.name
                context += f"{orig_idx}. {col_name!r}"

                # Get column description with optional truncation
                if col.description:
                    col_desc = truncate_string(col.description, max_length=100) if truncate else col.description
                    context += f": {col_desc}"

                # Add schema info for the column if available
                if schema_data:
                    if truncate:
                        schema_data = truncate_string(str(schema_data), max_length=50)
                    context += f" `{schema_data}`"

                context += "\n"

        return context

    @property
    def max_context(self) -> str:
        """Generate comprehensive formatted context with both vector and SQL data."""
        return self._generate_context(truncate=None)

    @property
    def selected_context(self) -> str:
        """Generate context with selected tables and columns, without truncation."""
        return self._generate_context(truncate=False)

    @property
    def min_context(self) -> str:
        """Generate context with selected tables and columns, with truncation."""
        return self._generate_context(truncate=True)

    @property
    def query(self) -> str:
        """Get the original query that generated this context."""
        return self.vector_metaset.query

    def __str__(self) -> str:
        """String representation is the formatted context."""
        return self.selected_context


@dataclass
class PreviousState:
    """Schema for previous state data."""

    query: str
    selected_columns: dict[str, list[str]] = field(default_factory=dict)

    @property
    def max_context(self) -> str:
        """Generate formatted text representation of the previous state."""
        context = f"Previous query: {self.query}\n\n"
        if self.selected_columns:
            for table_slug, cols in self.selected_columns.items():
                context += f"Table: {table_slug}\n"
                for col in cols:
                    context += f"- {col!r}\n"
        return context

    def __str__(self) -> str:
        """String representation is the formatted context."""
        return self.max_context


async def get_metaset(sources: dict[str, Source], tables: list[str]) -> SQLMetaset:
    """
    Get the metaset for the given sources and tables.

    Parameters
    ----------
    sources: dict[str, Source]
        The sources to get the metaset for.
    tables: list[str]
        The tables to get the metaset for.

    Returns
    -------
    metaset: SQLMetaset
        The metaset for the given sources and tables.
    """
    tables_info, tables_metadata = {}, {}
    for table_slug in tables:
        if SOURCE_TABLE_SEPARATOR in table_slug:
            source_name, table_name = table_slug.split(SOURCE_TABLE_SEPARATOR)
        elif len(sources) > 1:
            raise ValueError(
                f"Cannot resolve table {table_slug} without providing "
                "the source, when multiple sources are provided. Ensure "
                f"that you qualify the table name as follows:\n\n"
                "    <source>{SOURCE_TABLE_SEPARATOR}<table>"
            )
        else:
            source_name = next(iter(sources))
            table_name = table_slug
        source = sources[source_name]
        schema = await get_schema(source, table_name, include_count=True)
        tables_info[table_name] = SQLMetadata(
            table_slug=table_slug,
            schema=schema,
        )
        try:
            metadata = source.get_metadata(table_name)
        except Exception as e:
            log_debug(f"Failed to get metadata for table {table_name} in source {source_name}: {e}")
            metadata = {}
        tables_metadata[table_name] = VectorMetadata(
            table_slug=table_slug,
            similarity=1,
            base_sql=source.get_sql_expr(source.normalize_table(table_name)),
            description=metadata.get("description"),
            columns=[
                Column(name=col_name, description=col_values.pop("description"), metadata=col_values)
                for col_name, col_values in metadata.get("columns").items()
            ],
        )
    vector_metaset = VectorMetaset(vector_metadata_map=tables_metadata, query=None)
    return SQLMetaset(
        vector_metaset=vector_metaset,
        sql_metadata_map=tables_info,
    )


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
