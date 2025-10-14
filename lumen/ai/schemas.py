from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

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

    def _generate_context(self, include_columns: bool = False, truncate: bool = False) -> str:
        """
        Generate YAML formatted representation of the context.

        Args:
            include_columns: Whether to include column details in the context
            truncate: Whether to truncate strings and columns for brevity
        """
        tables_data = {}

        for table_slug, vector_metadata in self.vector_metadata_map.items():
            base_sql = truncate_string(vector_metadata.base_sql, max_length=200) if truncate else vector_metadata.base_sql

            table_data = {'access': base_sql}

            if vector_metadata.description:
                desc = truncate_string(vector_metadata.description, max_length=100) if truncate else vector_metadata.description
                table_data['info'] = desc

            # Only include columns if explicitly requested
            if include_columns and vector_metadata.columns:
                max_length = 20
                cols_to_show = vector_metadata.columns

                show_ellipsis = False
                if truncate:
                    cols_to_show, original_indices, show_ellipsis = truncate_iterable(cols_to_show, max_length)
                else:
                    cols_to_show = list(cols_to_show)

                columns_data = {}
                for i, col in enumerate(cols_to_show):
                    if show_ellipsis and i == len(cols_to_show) // 2:
                        columns_data['...'] = '...'

                    col_name = truncate_string(col.name) if truncate else col.name
                    if col.description:
                        col_desc = truncate_string(col.description, max_length=100) if truncate else col.description
                        columns_data[col_name] = col_desc
                    else:
                        columns_data[col_name] = None

                if columns_data:
                    table_data['columns'] = columns_data

            tables_data[table_slug] = table_data

        return yaml.dump(tables_data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @property
    def table_context(self) -> str:
        """Generate formatted text representation showing only table info (no columns)."""
        return self._generate_context(include_columns=False, truncate=False)

    @property
    def full_context(self) -> str:
        """Generate formatted text representation with all table and column details."""
        return self._generate_context(include_columns=True, truncate=False)

    @property
    def compact_context(self) -> str:
        """Generate truncated context with tables and columns for brevity."""
        return self._generate_context(include_columns=True, truncate=True)

    def __str__(self) -> str:
        """String representation shows table context by default."""
        return self.table_context


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

    def _generate_context(self, include_columns: bool = False, truncate: bool = False) -> str:
        """
        Generate YAML formatted context with both vector and SQL data.

        Args:
            include_columns: Whether to include column details in the context
            truncate: Whether to truncate strings and columns for brevity

        Returns:
            YAML formatted context string
        """
        tables_data = {}

        for table_slug in self.sql_metadata_map.keys():
            vector_metadata = self.vector_metaset.vector_metadata_map.get(table_slug)
            if not vector_metadata:
                continue

            base_sql = truncate_string(vector_metadata.base_sql, max_length=200) if truncate else vector_metadata.base_sql

            table_data = {'access': base_sql}

            if vector_metadata.description:
                desc = truncate_string(vector_metadata.description, max_length=100) if truncate else vector_metadata.description
                table_data['info'] = desc

            sql_data: SQLMetadata = self.sql_metadata_map.get(table_slug)
            if sql_data:
                # Get the count from schema
                if sql_data.schema.get("__len__"):
                    table_data['row_count'] = len(sql_data.schema)

            # Only include columns if explicitly requested
            if include_columns and vector_metadata.columns:
                cols_to_show = vector_metadata.columns
                columns_data = {}

                for col in cols_to_show:
                    schema_data = None
                    if sql_data and col.name in sql_data.schema:
                        schema_data = sql_data.schema[col.name]
                        if truncate and schema_data == "<null>":
                            continue

                    col_info = {}

                    # Get column description with optional truncation
                    if col.description:
                        col_desc = truncate_string(col.description, max_length=100) if truncate else col.description
                        col_info['description'] = col_desc

                    # Add schema info for the column if available
                    if schema_data and schema_data != "<null>":
                        if isinstance(schema_data, dict):
                            # Remove 'type': 'str' for token efficiency
                            schema_copy = {k: v for k, v in schema_data.items() if not (k == 'type' and v == 'str')}
                            if truncate and schema_copy.get('type') == 'enum':
                                schema_str = str(schema_copy)
                                if len(schema_str) > 50:
                                    schema_copy = truncate_string(schema_str, max_length=50)
                            col_info.update(schema_copy)
                        else:
                            col_info['value'] = schema_data

                    columns_data[col.name] = col_info if col_info else None

                if columns_data:
                    table_data['columns'] = columns_data

            tables_data[table_slug] = table_data

        return yaml.dump(tables_data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @property
    def table_context(self) -> str:
        """Generate formatted text representation showing only table info (no columns)."""
        return self._generate_context(include_columns=False, truncate=False)

    @property
    def full_context(self) -> str:
        """Generate comprehensive formatted context with both vector and SQL data."""
        return self._generate_context(include_columns=True, truncate=False)

    @property
    def compact_context(self) -> str:
        """Generate context with tables and columns, with truncation."""
        return self._generate_context(include_columns=True, truncate=True)

    @property
    def query(self) -> str:
        """Get the original query that generated this context."""
        return self.vector_metaset.query

    def __str__(self) -> str:
        """String representation shows table context by default."""
        return self.table_context


async def get_metaset(sources: list[Source], tables: list[str]) -> SQLMetaset:
    """
    Get the metaset for the given sources and tables.

    Parameters
    ----------
    sources: list[Source]
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
            source_name = next(iter(sources)).name
            table_name = table_slug
        source = next((s for s in sources if s.name == source_name), None)
        schema = await get_schema(source, table_name, include_count=True)
        tables_info[table_slug] = SQLMetadata(
            table_slug=table_slug,
            schema=schema,
        )
        try:
            metadata = source.get_metadata(table_name)
        except Exception as e:
            log_debug(f"Failed to get metadata for table {table_name} in source {source_name}: {e}")
            metadata = {}
        tables_metadata[table_slug] = VectorMetadata(
            table_slug=table_slug,
            similarity=1,
            base_sql=source.get_sql_expr(source.normalize_table(table_name)),
            description=metadata.get("description"),
            columns=[
                Column(name=col_name, description=col_values.pop("description", None), metadata=col_values)
                for col_name, col_values in metadata.get("columns", {}).items()
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
