import panel as pn
import param

from ..base import Component
from .controls import SourceControls
from .memory import memory
from .utils import get_schema


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

    _run_button = param.Parameter(default=None)

     # Whether to allow the analysis to be called multiple times in a row
    _consecutive_calls = False

    _callable_by_llm = True

    _field_params = []

    @classmethod
    def applies(cls, pipeline) -> bool:
        applies = True
        for col in cls.columns:
            if isinstance(col, tuple):
                applies &= any(c in pipeline.data.columns for c in col)
            else:
                applies &= col in pipeline.data.columns
        return applies

    def controls(self):
        config_options = [p for p in self.param if p not in Analysis.param]
        if config_options:
            return pn.Param(self.param, parameters=config_options)

    def __call__(self, pipeline) -> Component:
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
        self._source_controls = SourceControls()
        self._run_button = self._source_controls._add_button
        self._source_controls.param.watch(self._update_table_name, "_last_table")

        source = memory.get("current_source")
        table = memory.get("current_table")
        self._previous_source = source
        self._previous_table = table
        columns = list(get_schema(source, table=table).keys())
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

    async def __call__(self, pipeline) -> Component:
        if self.table_name:
            agent = next(agent for agent in self.agents if type(agent).__name__ == "SQLAgent")
            content = f"Join these tables: '//{self._previous_source}//{self._previous_table}' and '//{memory['current_source']}//{self.table_name}'"
            if self.index_col:
                content += f" left join on {self.index_col}"
            else:
                content += " based on the closest matching columns"
            if self.context:
                content += f"\nadditional context:\n{self.context!r}"
            await agent.answer(messages=[{"role": "user", "content": content}])
            pipeline = memory["current_pipeline"]
        return pipeline
