from __future__ import annotations

from typing import TYPE_CHECKING

import param

from panel.chat import ChatFeed
from panel.param import Param
from panel.viewable import Viewable

from ..base import Component
from .actor import TContext
from .utils import get_data

if TYPE_CHECKING:
    from ..pipeline import Pipeline


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

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The ChatInterface instance that will be used to stream messages.""")

    message = param.String(default="", doc="The message to display on interface when the analysis is run.")

    _run_button = param.Parameter(default=None)

    _dynamic_provides = param.Dict(default={}, instantiate=True, doc="""
        Context values the analysis publishes reactively after it has rendered,
        e.g. following an interactive selection. AnalysisOutput watches this and
        merges it into its render_context, so updates reach the out-context
        without mutating the (snapshot) input context. Unlike Tool.provides (a
        static list of context key names), this is a live map of values that can
        change on each interaction.""")

     # Whether to allow the analysis to be called multiple times in a row
    _consecutive_calls = False

    _callable_by_llm = True

    _field_params = []

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

    def controls(self, context: TContext) -> Param | None:
        config_options = [
            p for p in self.param
            if p not in Analysis.param
            and (self.param[p].precedence is None or self.param[p].precedence >= 0)
        ]
        if not config_options:
            return None
        return Param(self.param, parameters=config_options, margin=0, show_name=False)

    def __call__(self, pipeline: Pipeline, context: TContext) -> Component | Viewable:
        return pipeline
