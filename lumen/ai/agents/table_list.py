from typing import NotRequired

import panel as pn
import param

from ...sources.base import Source
from ..config import SOURCE_TABLE_SEPARATOR
from ..context import ContextModel, TContext
from ..schemas import Metaset
from .base_list import BaseListAgent


class TableListInputs(ContextModel):

    source: Source

    visible_slugs: NotRequired[set[str]]

    metaset: NotRequired[Metaset]


class TableListAgent(BaseListAgent):
    """
    The TableListAgent lists all available data and lets the user pick one.
    """

    conditions = param.List(default=[
        "Use when user explicitly asks to 'list data', 'show available data', or 'what data do you have'",
        "NOT for showing actual data contents, querying, or analyzing data",
        "NOT for describing data sources or telling what the data is about",
    ])

    not_with = param.List(default=["DbtslAgent", "SQLAgent"])

    purpose = param.String(default="""
        Displays a list of all available data & datasets. Not useful for identifying which dataset to use for analysis.""")

    _column_name = "Data"

    _message_format = "Show the data: {item}"

    input_schema = TableListInputs

    def _create_row_content(self, context: TContext, source_name: str):
        """Create row content function that displays metadata for a table."""
        metaset = context.get('metaset')

        def row_content(row):
            table_name = row[self._column_name]
            table_slug = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"

            if not metaset or table_slug not in metaset.catalog:
                return pn.pane.Markdown("*No metadata available*")

            entry = metaset.catalog[table_slug]

            content_parts = []
            if entry.description:
                content_parts.append(f"**Description:** {entry.description}")

            if entry.metadata:
                metadata_lines = ["**Metadata:**"]
                # Fields to exclude from display
                exclude_keys = {'description', 'columns', 'data_type', 'source_name', 'rows', 'updated_at', 'created_at'}
                for key, value in entry.metadata.items():
                    if key in exclude_keys or value is None or value == '':
                        continue
                    metadata_lines.append(f"- **{key}:** {value}")
                if len(metadata_lines) > 1:
                    content_parts.append("\n".join(metadata_lines))

            counts = []
            if entry.metadata and 'rows' in entry.metadata:
                counts.append(f"{entry.metadata['rows']} rows")
            if entry.columns:
                counts.append(f"{len(entry.columns)} columns")
            if counts:
                content_parts.append(" | ".join(counts))

            if not content_parts:
                return pn.pane.Markdown("*No metadata available*")

            return pn.pane.Markdown("\n\n".join(content_parts), sizing_mode='stretch_width')

        return row_content

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        return len(context.get('visible_slugs', set())) > 1

    def _get_items(self, context: TContext) -> dict[str, list[str]]:
        if "closest_tables" in context:
            # If we have closest_tables from search, return as a single group
            return {"Search Results": context["closest_tables"]}

        # Group tables by source
        visible_slugs = context.get('visible_slugs')
        if not visible_slugs:
            return {}

        # If metaset is available, sort by relevance (similarity score)
        metaset = context.get('metaset')
        if metaset:
            # Get tables sorted by relevance from metaset
            sorted_slugs = [
                slug for slug in metaset.catalog.keys()
                if slug in visible_slugs
            ]
        else:
            # Fallback to alphabetical sort
            sorted_slugs = sorted(visible_slugs)

        tables_by_source = {}
        for slug in sorted_slugs:
            if SOURCE_TABLE_SEPARATOR in slug:
                source_name, table_name = slug.split(SOURCE_TABLE_SEPARATOR, 1)
            else:
                # Fallback if separator not found
                source_name = "Unknown"
                table_name = slug

            if source_name not in tables_by_source:
                tables_by_source[source_name] = []
            tables_by_source[source_name].append(table_name)

        return tables_by_source
