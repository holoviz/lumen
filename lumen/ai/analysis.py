import panel as pn
import param

from ..base import Component


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

    @classmethod
    def applies(cls, pipeline) -> bool:
        return all(col in pipeline.data.columns for col in cls.columns)

    def controls(self):
        config_options = [p for p in self.param if p not in Analysis.param]
        if config_options:
            return pn.Param(self.param, parameters=config_options)

    def __call__(self, pipeline) -> Component:
        return pipeline
