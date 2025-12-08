
from functools import partial
from typing import Any, NotRequired

import pandas as pd
import panel as pn
import param

from panel_material_ui import Tabs

from ...sources.base import Source
from ..config import SOURCE_TABLE_SEPARATOR
from ..context import ContextModel, TContext
from ..llm import Message
from ..schemas import Metaset
from .base import Agent


class ListAgent(Agent):
    """
    Abstract base class for agents that display a list of items to the user.
    """

    purpose = param.String(default="""
        Renders a list of items to the user and lets the user pick one.""")

    _extensions = ("tabulator",)

    _column_name = None

    _message_format = None

    __abstract = True

    def _get_items(self, context: TContext) -> dict[str, list[str]]:
        """Return dict of items grouped by source/category"""

    def _use_item(self, event, interface=None):
        """Handle when a user clicks on an item in the list"""
        if event.column != "show":
            return

        tabulator = self._tabs[self._tabs.active]
        item = tabulator.value.iloc[event.row, 0]

        if self._message_format is None:
            raise ValueError("Subclass must define _message_format")

        message = self._message_format.format(item=repr(item))
        interface.send(message)

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        items = self._get_items(context)

        # Create tabs with one tabulator per source
        tabs = []
        for source_name, source_items in items.items():
            if not source_items:
                continue

            header_filters = False
            if len(source_items) > 10:
                column_filter = {"type": "input", "placeholder": f"Filter by {self._column_name.lower()}..."}
                header_filters = {self._column_name: column_filter}

            df = pd.DataFrame({self._column_name: source_items})
            item_list = pn.widgets.Tabulator(
                df,
                buttons={"show": '<i class="fa fa-eye"></i>'},
                show_index=False,
                min_height=150,
                min_width=350,
                widths={self._column_name: "90%"},
                disabled=True,
                page_size=10,
                pagination="remote",
                header_filters=header_filters,
                sizing_mode="stretch_width",
                name=source_name
            )
            item_list.on_click(partial(self._use_item, interface=self.interface))
            tabs.append(item_list)

        self._tabs = Tabs(*tabs, sizing_mode="stretch_width")

        self.interface.stream(
            pn.Column(
                f"The available {self._column_name.lower()}s are listed below. Click on the eye icon to show the {self._column_name.lower()} contents.",
                self._tabs
            ), user=self.__class__.__name__
        )
        return [self._tabs], {}


class TableListInputs(ContextModel):

    source: Source

    visible_slugs: NotRequired[set[str]]

    metaset: NotRequired[Metaset]


class TableListAgent(ListAgent):
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
        visible_slugs = context.get('visible_slugs', set())
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


class DocumentListInputs(ContextModel):

    document_sources: dict[str, Any]


class DocumentListAgent(ListAgent):
    """
    The DocumentListAgent lists all available documents provided by the user.
    """

    conditions = param.List(
        default=[
            "Use when user asks to list or see all available documents",
            "NOT when user asks about specific document content",
        ]
    )

    purpose = param.String(default="""
        Displays a list of all available documents.""")

    _column_name = "Documents"

    _message_format = "Tell me about: {item}"

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        sources = context.get("document_sources")
        if not sources:
            return False
        return len(sources) > 1

    def _get_items(self, context: TContext) -> dict[str, list[str]]:
        # extract the filename, following this pattern `Filename: 'filename'``
        documents = [doc["metadata"].get("filename", "untitled") for doc in context.get("document_sources", [])]
        return {"Documents": documents} if documents else {}
