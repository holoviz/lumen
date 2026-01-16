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
        "NOT for describing data sources or telling what the data is about",
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
