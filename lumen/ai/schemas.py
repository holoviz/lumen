from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lumen.ai.utils import truncate_iterable, truncate_string


@dataclass
class TableColumn:
    """Schema for a column with its description."""

    name: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableVectorMetadata:
    """Schema for vector lookup data for a single table."""

    table_slug: str  # Combined source_name and table_name with separator
    similarity: float
    description: str | None = None
    base_sql: str | None = None
    table_cols: list[TableColumn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableVectorMetaset:
    """Schema container for vector data for multiple tables_metadata."""

    query: str
    vector_metadata_map: dict[str, TableVectorMetadata]
    sel_tables_cols: dict[str, list[str]] = field(default_factory=dict)

    def _generate_context(self, truncate: bool | None = None) -> str:
        """
        Generate formatted text representation of the context.

        Args:
            truncate: Controls truncation behavior.
                    None: Show all tables (max_context)
                    False: Filter by sel_tables_cols without truncation (sel_context)
                    True: Filter by sel_tables_cols with truncation (min_context)
        """
        context = "Below are the relevant tables and columns to use:\n\n"

        # Use selected tables if specified, otherwise use all tables
        tables_to_show = self.sel_tables_cols or self.vector_metadata_map.keys()

        for table_slug in self.vector_metadata_map.keys():
            # Skip tables not in the selected list if using sel_context or min_context
            if truncate is not None and table_slug not in tables_to_show:
                continue

            vector_metadata = self.vector_metadata_map[table_slug]
            context += f"\n\n{table_slug!r} Similarity: ({vector_metadata.similarity:.3f})\n"

            if vector_metadata.description:
                context += f"Description: {vector_metadata.description}\n"

            if vector_metadata.base_sql:
                context += f"Base SQL: {vector_metadata.base_sql}\n"

            max_length = 20
            cols_to_show = vector_metadata.table_cols
            if truncate is not None:
                cols_to_show = [col for col in cols_to_show if col.name in self.sel_tables_cols.get(table_slug, [])]

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

                col_name = truncate_string(col.name) if truncate else col.name
                context += f"{orig_idx}. {col_name!r}"
                if col.description:
                    col_desc = (
                        truncate_string(col.description, max_length=100)
                        if truncate
                        else col.description
                    )
                    context += f": {col_desc}"
                context += "\n"
        return context

    @property
    def max_context(self) -> str:
        """Generate formatted text representation of the context."""
        return self._generate_context(truncate=None)

    @property
    def sel_context(self) -> str:
        """Generate formatted text representation of the context with selected tables cols"""
        return self._generate_context(truncate=False)

    @property
    def min_context(self) -> str:
        """Generate formatted text representation of the context with selected tables cols and truncated strings"""
        return self._generate_context(truncate=True)

    def __str__(self) -> str:
        """String representation is the formatted context."""
        return self.sel_context


@dataclass
class TableSQLMetadata:
    """Schema for SQL schema data for a single table."""

    table_slug: str
    schema: dict[str, Any]
    base_sql: str | None = None
    view_definition: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableSQLMetaset:
    """Schema container for SQL data for multiple tables_metadata that builds on vector context."""

    vector_metaset: TableVectorMetaset
    sql_metadata_map: dict[str, TableSQLMetadata]

    def _generate_context(self, truncate: bool | None = None) -> str:
        """
        Generate formatted context with both vector and SQL data.

        Args:
            truncate: Controls truncation behavior.
                      None: Show all tables (max_context)
                      False: Filter by sel_tables_cols without truncation (sel_context)
                      True: Filter by sel_tables_cols with truncation (min_context)

        Returns:
            Formatted context string
        """
        context = "Below are the relevant tables and columns to use:\n\n"

        vector_metaset = self.vector_metaset
        tables_to_show = vector_metaset.sel_tables_cols or vector_metaset.vector_metadata_map.keys()

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
                context += f"Description: {desc}\n"

            sql_data: TableSQLMetadata = self.sql_metadata_map.get(table_slug)
            if sql_data:
                base_sql = truncate_string(sql_data.base_sql, max_length=200) if truncate else sql_data.base_sql
                context += f"Base SQL: {base_sql}\n"

                # Get the count from schema
                if sql_data.schema.get("__len__"):
                    context += f"Row count: {len(sql_data.schema)}\n"

            max_length = 20
            cols_to_show = vector_metadata.table_cols
            if truncate is not None and vector_metaset.sel_tables_cols:
                cols_to_show = [col for col in cols_to_show if col.name in vector_metaset.sel_tables_cols.get(table_slug, [])]

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
    def sel_context(self) -> str:
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
        return self.sel_context

@dataclass
class PreviousState:
    """Schema for previous state data."""

    query: str
    sel_tables_cols: dict[str, list[str]] = field(default_factory=dict)

    @property
    def max_context(self) -> str:
        """Generate formatted text representation of the previous state."""
        context = f"Previous query: {self.query}\n\n"
        if self.sel_tables_cols:
            for table_slug, cols in self.sel_tables_cols.items():
                context += f"Table: {table_slug}\n"
                for col in cols:
                    context += f"- {col!r}\n"
        return context

    def __str__(self) -> str:
        """String representation is the formatted context."""
        return self.max_context
