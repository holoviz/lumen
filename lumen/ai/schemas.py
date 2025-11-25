from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from ..sources import Source
from .config import SOURCE_TABLE_SEPARATOR
from .utils import get_schema, log_debug, truncate_string


@dataclass
class Column:
    name: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableCatalogEntry:
    table_slug: str
    similarity: float
    columns: list[Column]
    description: str | None = None
    sql_expr: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Metaset:
    """
    Schema container for table metadata with optional SQL enrichment.

    Contains table catalog from discovery (descriptions, columns) and optionally
    SQL schema data (types, enums, row counts).
    """

    query: str | None
    catalog: dict[str, TableCatalogEntry]
    schemas: dict[str, dict[str, Any]] | None = None

    @property
    def has_schemas(self) -> bool:
        return self.schemas is not None and len(self.schemas) > 0

    def _build_table_data(
        self,
        table_slug: str,
        catalog_entry: TableCatalogEntry,
        include_columns: bool,
        truncate: bool
    ) -> dict:
        sql_expr = catalog_entry.sql_expr
        if truncate and sql_expr:
            sql_expr = truncate_string(sql_expr, max_length=200)

        data = {'read_with': sql_expr}

        if catalog_entry.description:
            desc = catalog_entry.description
            if truncate:
                desc = truncate_string(desc, max_length=100)
            data['info'] = desc

        if self.has_schemas:
            schema = self.schemas.get(table_slug)
            if schema and schema.get("__len__"):
                data['row_count'] = len(schema)

        if include_columns and catalog_entry.columns:
            data['columns'] = self._build_columns_data(
                table_slug, catalog_entry.columns, truncate
            )

        return data

    def _build_columns_data(
        self,
        table_slug: str,
        columns: list[Column],
        truncate: bool
    ) -> dict:
        columns_data = {}
        schema = self.schemas.get(table_slug) if self.has_schemas else None

        for col in columns:
            col_info = {}

            if col.description:
                desc = col.description
                if truncate:
                    desc = truncate_string(desc, max_length=100)
                col_info['description'] = desc

            if schema and col.name in schema:
                schema_data = schema[col.name]

                if truncate and schema_data == "<null>":
                    continue

                if schema_data != "<null>":
                    if isinstance(schema_data, dict):
                        schema_copy = {
                            k: v for k, v in schema_data.items()
                            if not (k == 'type' and v == 'str')
                        }
                        if truncate and schema_copy.get('type') == 'enum':
                            schema_str = str(schema_copy)
                            if len(schema_str) > 50:
                                schema_copy = truncate_string(schema_str, max_length=50)
                        col_info.update(schema_copy)
                    else:
                        col_info['value'] = schema_data

            columns_data[col.name] = col_info if col_info else None

        return columns_data

    def _generate_context(
        self,
        include_columns: bool = False,
        truncate: bool = False
    ) -> str:
        table_slugs = self.schemas.keys() if self.has_schemas else self.catalog.keys()

        tables_data = {}
        for table_slug in table_slugs:
            catalog_entry = self.catalog.get(table_slug)
            if not catalog_entry:
                continue

            tables_data[table_slug] = self._build_table_data(
                table_slug, catalog_entry, include_columns, truncate
            )

        return yaml.dump(
            tables_data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False
        )

    @property
    def table_context(self) -> str:
        return self._generate_context(include_columns=False, truncate=False)

    @property
    def full_context(self) -> str:
        return self._generate_context(include_columns=True, truncate=False)

    @property
    def compact_context(self) -> str:
        return self._generate_context(include_columns=True, truncate=True)

    def __str__(self) -> str:
        return self.table_context


async def get_metaset(
    sources: list[Source],
    tables: list[str],
    prev: Metaset | None = None
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

    Returns
    -------
    metaset: Metaset
        The metaset for the given sources and tables.
    """
    schemas_data, catalog_data = {}, {}

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

        if prev and prev.schemas and table_slug in prev.schemas:
            schema = prev.schemas[table_slug]
        else:
            schema = await get_schema(source, table_name, include_count=True)
        schemas_data[table_slug] = schema

        if prev and table_slug in prev.catalog:
            catalog_entry = prev.catalog[table_slug]
        else:
            try:
                metadata = source.get_metadata(table_name)
            except Exception as e:
                log_debug(f"Failed to get metadata for table {table_name} in source {source_name}: {e}")
                metadata = {}

            catalog_entry = TableCatalogEntry(
                table_slug=table_slug,
                similarity=1,
                sql_expr=source.get_sql_expr(source.normalize_table(table_name)),
                description=metadata.get("description"),
                columns=[
                    Column(
                        name=col_name,
                        description=col_values.pop("description", None),
                        metadata=col_values
                    )
                    for col_name, col_values in metadata.get("columns", {}).items()
                ],
            )
        catalog_data[table_slug] = catalog_entry

    return Metaset(
        query=None,
        catalog=catalog_data,
        schemas=schemas_data,
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
