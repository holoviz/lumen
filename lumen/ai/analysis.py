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

    columns = param.List(default=[], doc="The columns required for the analysis.", readonly=True)

    @classmethod
    def applies(cls, pipeline) -> bool:
        return all(col in pipeline.data.columns for col in cls.columns)

    def __call__(self, pipeline) -> Component:
        return pipeline
