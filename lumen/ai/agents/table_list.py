from typing import NotRequired

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
    ])

    not_with = param.List(default=["DbtslAgent", "SQLAgent"])

    purpose = param.String(default="""
        Displays a list of all available data & datasets. Not useful for identifying which dataset to use for analysis.""")

    _column_name = "Data"

    _message_format = "Show the data: {item}"

    input_schema = TableListInputs

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

    def summarize(self, outputs: list, out_ctx: dict) -> dict[str, str]:
        # TableListAgent's respond() doesn't populate out_ctx, but we can infer from the outputs
        # The outputs contain Tabs with Tabulators showing the tables

        if not outputs:
            return {
                "bare": "No tables available",
                "compact": "No tables found",
                "detailed": "No data tables are available",
            }

        # Try to extract table names from the Tabs widget in outputs
        tables_by_source = {}
        try:
            # The output is typically [Tabs widget]
            # Check if we have the _tabs stored on self from respond()
            if hasattr(self, '_tabs') and self._tabs is not None:
                for tab in self._tabs:
                    source_name = tab.name
                    # Each tab is a Tabulator with a DataFrame
                    if hasattr(tab, 'value') and tab.value is not None:
                        table_names = tab.value[self._column_name].tolist()
                        if table_names:
                            tables_by_source[source_name] = table_names
        except Exception:
            # Fallback if we can't extract from outputs
            pass

        if not tables_by_source:
            return {
                "bare": "Tables displayed but details unavailable",
                "compact": "Table list generated but summary unavailable",
                "detailed": "Table list widget created successfully",
            }

        # Flatten to get all table names
        all_tables = []
        for tables in tables_by_source.values():
            all_tables.extend(tables)

        n_tables = len(all_tables)
        n_sources = len(tables_by_source)

        # Build summaries
        if n_sources == 1:
            source_name = list(tables_by_source.keys())[0]
            return {
                "bare": f"{n_tables} tables from {source_name}: {', '.join(all_tables[:5])}{' ...' if n_tables > 5 else ''}",
                "compact": f"{n_tables} tables from {source_name}:\n" + "\n".join(f"- {t}" for t in all_tables),
                "detailed": f"{n_tables} tables from {source_name}:\n" + "\n".join(f"- {t}" for t in all_tables),
            }
        else:
            return {
                "bare": f"{n_tables} tables from {n_sources} sources: {', '.join(all_tables[:5])}{' ...' if n_tables > 5 else ''}",
                "compact": "\n".join(f"**{source}**: {', '.join(tables)}" for source, tables in tables_by_source.items()),
                "detailed": "\n".join(
                    f"**{source}** ({len(tables)} tables):\n" + "\n".join(f"  - {t}" for t in tables)
                    for source, tables in tables_by_source.items()
                ),
            }
