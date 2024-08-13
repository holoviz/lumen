import panel as pn
import param

from ..base import Component
from .controls import add_source_controls
from .memory import memory


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

    columns = param.List(default=[], doc="The columns required for the analysis.", readonly=True)

    _run_button = param.Parameter(default=None)

    @classmethod
    def applies(cls, pipeline) -> bool:
        return all(col in pipeline.data.columns for col in cls.columns)

    def controls(self):
        config_options = [p for p in self.param if p not in Analysis.param]
        if config_options:
            return pn.Param(self.param, parameters=config_options)

    def __call__(self, pipeline) -> Component:
        return pipeline


class JoinAnalysis(Analysis):

    autorun = param.Boolean(default=False, doc="Whether to automatically run the analysis.")

    table_name = param.String(doc="The name of the table to join.")

    index_col = param.String(doc="The column to join on in the left table.")

    context = param.String(doc="Additional context to provide to the LLM.")

    def controls(self):
        source_controls = add_source_controls(replace_controls=False)
        self._run_button = source_controls[0][-1]
        name_input = source_controls[0][1]
        name_input.link(self, value='table_name')
        index_col = pn.widgets.AutocompleteInput.from_param(
            self.param.index_col, options=memory["current_data"].columns.tolist(), name="Join on",
            placeholder="Start typing column name"
        )
        context = pn.widgets.TextInput.from_param(self.param.context, name="Context")
        source_controls[0].insert(0, index_col)
        source_controls[0].insert(1, context)
        return source_controls

    async def __call__(self, pipeline) -> Component:
        if self.table_name:
            agent = next((agent for agent in self.agents if type(agent).__name__ == "SQLAgent"), None)
            content = "Join the tables"
            if self.index_col:
                content += f" on {self.index_col}"
            if self.context:
                content += f"\nadditional context:\n{self.context!r}"
            await agent.answer(messages=[{"role": "user", "content": content}])
            pipeline = memory["current_pipeline"]
        return pipeline
