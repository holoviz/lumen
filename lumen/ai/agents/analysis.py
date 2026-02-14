import asyncio

from collections.abc import Callable
from typing import Any, Literal, NotRequired

import param

from panel.pane import panel as as_panel
from panel.viewable import Viewable
from pydantic import create_model
from pydantic.fields import FieldInfo

from ...pipeline import Pipeline
from ...views import Panel
from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..editors import AnalysisOutput
from ..llm import Message
from ..utils import get_data, log_debug
from .base_lumen import BaseLumenAgent


def make_analysis_model(analyses: list[str]):
    return create_model(
        "Analysis",
        analysis=(Literal[tuple(analyses)], FieldInfo(
            description="The name of the analysis that is most appropriate given the user query."
        ))
    )

class AnalysisInputs(ContextModel):

    data: NotRequired[Any]

    pipeline: Pipeline


class AnalysisOutputs(ContextModel):

    analysis: Callable

    pipeline: NotRequired[Pipeline]

    view: NotRequired[Any]


class AnalysisAgent(BaseLumenAgent):

    analyses = param.List([])

    conditions = param.List(
        default=[
            "Use for custom analysis, advanced analytics, or domain-specific methods",
            "Use when the user query matches one of the available analyses' name or description below.",
            "Include the selected analysis' required cols in the instructions",
            "NOT for simple queries or basic visualizations",
        ]
    )

    purpose = param.String(default="Perform custom analyses that are reliable and repeatable.")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "AnalysisAgent" / "main.jinja2",
                "response_model": make_analysis_model
            },
        }
    )

    input_schema = AnalysisInputs
    output_schema = AnalysisOutputs

    _editor_type = AnalysisOutput

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None
    ) -> tuple[list[Any], TContext]:
        pipeline = context.get("pipeline")
        analyses = {a.name: a for a in self.analyses if await a.applies(pipeline)}
        if not analyses:
            log_debug("No analyses apply to the current data.")
            return [], {}

        # Short cut analysis selection if there's an exact match
        if len(messages):
            analysis = messages[0].get("content").replace("Apply ", "")
            if analysis in analyses:
                analyses = {analysis: analyses[analysis]}

        if len(analyses) > 1:
            with self._add_step(title="Choosing the most relevant analysis...", steps_layout=self._steps_layout) as step:
                analysis_name = await self.invoke_prompt(
                    "main",
                    messages,
                    analyses=list(analyses),
                    context=context,
                    data=context.get("data"),
                ).analysis
                step.stream(f"Selected {analysis_name}")
                step.success_title = f"Selected {analysis_name}"
        else:
            analysis_name = next(iter(analyses))
        analysis = analyses[analysis_name]

        view = None
        with self._add_step(title=step_title or "Creating view...", steps_layout=self._steps_layout) as step:
            await asyncio.sleep(0.1)  # necessary to give it time to render before calling sync function...
            analysis_callable = analysis.instance(agents=self.agents, interface=self.interface)

            data = await get_data(pipeline)
            for field in analysis._field_params:
                analysis_callable.param[field].objects = list(data.columns)

            if analysis.autorun:
                if asyncio.iscoroutinefunction(analysis_callable.__call__):
                    view = await analysis_callable(pipeline, context)
                else:
                    view = await asyncio.to_thread(analysis_callable, pipeline, context)
                view = as_panel(view)
                if isinstance(view, Viewable):
                    view = Panel(object=view, pipeline=context.get("pipeline"))
                step.stream(f"Generated view of type {type(view).__name__}")
                step.success_title = "Generated view"
            else:
                step.success_title = "Configure the analysis"

        if view is None and analysis.autorun:
            self.interface.stream("Failed to find an analysis that applies to this data.")
            return [], {}

        out = self._editor_type(
            component=view, title=step_title, analysis=analysis_callable,
            pipeline=context.get("pipeline"), context=context
        )
        out_context = await out.render_context()
        return [out], out_context
