from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    table_cols: list[TableColumn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableVectorMetaset:
    """Schema container for vector data for multiple tables_metadata."""

    query: str
    vector_metadata_map: dict[str, TableVectorMetadata]
    sel_tables_cols: dict[str, list[str]] = field(default_factory=dict)

    @property
    def context(self) -> str:
        """Generate formatted text representation of the context."""
        # TODO: maybe move this to Jinja2 templating
        context = f"Below are the relevant tables for query: {self.query}\n\n"

        # TODO: truncate list of columns up to a certain number
        sel_tables_cols = self.sel_tables_cols or self.vector_metadata_map.keys()
        for table_slug in sel_tables_cols:
            vector_metadata = self.vector_metadata_map[table_slug]
            context += f"{table_slug} Similarity: ({vector_metadata.similarity:.3f})\n"
            if vector_metadata.description:
                context += f"Description: {vector_metadata.description}\n"

            for index, col in enumerate(vector_metadata.table_cols):
                if table_slug not in sel_tables_cols:
                    continue
                context += f"{index + 1}. {col.name}"
                if col.description:
                    context += f": {col.description}"
                context += "\n"
        return context

    def __str__(self) -> str:
        """String representation is the formatted context."""
        return self.context


@dataclass
class TableSQLMetadata:
    """Schema for SQL schema data for a single table."""

    table_slug: str
    schema: dict[str, Any]
    count: int | None = None
    view_definition: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableSQLMetaset:
    """Schema container for SQL data for multiple tables_metadata that builds on vector context."""

    vector_metaset: TableVectorMetaset
    sql_metadata_map: dict[str, TableSQLMetadata]

    @property
    def context(self) -> str:
        """Generate comprehensive formatted context with both vector and SQL data."""
        context = f"Below are the relevant tables for query: {self.query}\n\n"

        table_slugs = self.sql_metadata_map.keys()
        vector_metaset = self.vector_metaset
        sel_tables_cols = vector_metaset.sel_tables_cols or {}
        for table_slug in table_slugs:
            if table_slug not in sel_tables_cols:
                continue

            vector_metadata = vector_metaset.vector_metadata_map[table_slug]
            context += f"{table_slug} Similarity: ({vector_metadata.similarity:.3f})\n"
            if vector_metadata.description:
                context += f"Description: {vector_metadata.description}\n"

            sql_data = self.sql_metadata_map.get(table_slug)
            for index, col in enumerate(vector_metadata.table_cols):
                if table_slug not in sel_tables_cols:
                    continue
                context += f"{index + 1}. {col.name}"
                if col.description:
                    context += f": {col.description}"
                if sql_data:
                    if col.name in sql_data["schema"]:
                        context += f" ({sql_data['schema'][col.name]})"
                context += "\n"
        return context

    @property
    def query(self) -> str:
        """Get the original query that generated this context."""
        return self.vector_metaset.query

    def __str__(self) -> str:
        """String representation is the formatted context."""
        return self.context


@dataclass
class PreviousState:
    """Schema for previous state data."""

    query: str
    sel_tables_cols: dict[str, list[str]] = field(default_factory=dict)

    @property
    def context(self) -> str:
        """Generate formatted text representation of the previous state."""
        context = f"Previous query: {self.query}\n\n"
        if self.sel_tables_cols:
            for table_slug, cols in self.sel_tables_cols.items():
                context += f"Table: {table_slug}\n"
                for col in cols:
                    context += f"- {col}\n"
        return context

    def __str__(self) -> str:
        """String representation is the formatted context."""
        return self.context
