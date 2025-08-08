from __future__ import annotations

import panel as pn
import param

from panel.viewable import Viewable

from ..base import Component
from .config import SOURCE_TABLE_SEPARATOR
from .controls import SourceControls
from .memory import _Memory, memory
from .utils import get_data


class Analysis(param.ParameterizedFunction):
    """
    The purpose of an Analysis is to perform custom actions tailored
    to a particular use case and/or domain. It is a ParameterizedFunction
    which implements an applies method to confirm whether the analysis
    applies to a particular dataset and a __call__ method which is given
    a pipeline and must return another valid component.
    """

    agents = param.List(default=[], doc="List of available agents that can be invoked by the analysis.")

    autorun = param.Boolean(default=True, doc="Whether to automatically run the analysis.")

    columns = param.List(default=[], constant=True, doc="""
       The columns required for the analysis. May use tuples to declare that one of
       the columns must be present.""")

    memory = param.ClassSelector(class_=_Memory, default=None, doc="""
        Local memory which will be used to provide the agent context.
        If None the global memory will be used.""")

    message = param.String(default="", doc="The message to display on interface when the analysis is run.")

    _run_button = param.Parameter(default=None)

     # Whether to allow the analysis to be called multiple times in a row
    _consecutive_calls = False

    _callable_by_llm = True

    _field_params = []

    @property
    def _memory(self):
        return memory if self.memory is None else self.memory

    @classmethod
    async def applies(cls, pipeline) -> bool:
        applies = True
        data = await get_data(pipeline)
        for col in cls.columns:
            if isinstance(col, tuple):
                applies &= any(c in data.columns for c in col)
            else:
                applies &= col in data.columns
        return applies

    def controls(self):
        config_options = [p for p in self.param if p not in Analysis.param]
        if config_options:
            return pn.Param(self.param, parameters=config_options)

    def __call__(self, pipeline) -> Component | Viewable:
        return pipeline


class Join(Analysis):

    autorun = param.Boolean(default=False, doc="Whether to automatically run the analysis.")

    table_name = param.String(doc="The name of the table to join.")

    index_col = param.String(doc="The column to join on in the left table.")

    context = param.String(doc="Additional context to provide to the LLM.")

    _callable_by_llm = False

    _previous_table = param.Parameter()

    _previous_source = param.Parameter()

    def _update_table_name(self, event):
        self.table_name = event.new

    def controls(self):
        self._source_controls = SourceControls(
            multiple=True, replace_controls=False, memory=self._memory
        )
        self._run_button = self._source_controls._add_button
        self._source_controls.param.watch(self._update_table_name, "_last_table")

        source = self._memory.get("source")
        table = self._memory.get("table")
        self._previous_source = source
        self._previous_table = table
        columns = list(source.get_schema(table).keys())
        index_col = pn.widgets.AutocompleteInput.from_param(
            self.param.index_col, options=columns, name="Join on",
            placeholder="Start typing column name", search_strategy="includes",
            case_sensitive=False, restrict=False
        )
        context = pn.widgets.TextInput.from_param(self.param.context, name="Context")
        controls = pn.FlexBox(
            index_col,
            context,
            self._source_controls,
        )
        return controls

    async def __call__(self, pipeline):
        if self.table_name:
            agent = next(agent for agent in self.agents if type(agent).__name__ == "SQLAgent")
            content = (
                "Join these tables: "
                f"'{self._previous_source}{SOURCE_TABLE_SEPARATOR}{self._previous_table}' "
                f"and '{self._memory['source']}{SOURCE_TABLE_SEPARATOR}{self.table_name}'"
            )
            if self.index_col:
                content += f" left join on {self.index_col}"
            else:
                content += " based on the closest matching columns"
            if self.context:
                content += f"\nadditional context:\n{self.context!r}"
            await agent.answer(messages=[{"role": "user", "content": content}])
            pipeline = self._memory["pipeline"]

        self.message = f"Joined {self._previous_table} with {self.table_name}."
        return pipeline
