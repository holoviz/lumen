
from functools import partial
from typing import Any

import pandas as pd
import panel as pn
import param

from panel_material_ui import Tabs

from ..context import TContext
from ..llm import Message
from .base import Agent


class BaseListAgent(Agent):
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
