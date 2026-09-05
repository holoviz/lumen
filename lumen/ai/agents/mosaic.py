"""
Mosaic Agent for generating interactive Mosaic/vgplot visualizations.

Generates declarative ``mosaic-spec`` specifications (YAML) that are rendered by
`MosaicView` through the `Mosaic` pane. Mosaic references tables by name and
pushes computation down to DuckDB, making it well suited to large,
cross-filtered and linked interactive views.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import param

from pydantic import Field

from ...config import load_yaml
from ...views.base import MosaicView
from ..config import PROMPTS_DIR
from ..editors import MosaicEditor
from ..models import EscapeBaseModel, RetrySpec
from ..utils import get_schema, log_debug, retry_llm_output
from .base_view import BaseViewAgent

if TYPE_CHECKING:
    from ...pipeline import Pipeline
    from ..context import TContext
    from ..llm import Message


class MosaicSpec(EscapeBaseModel):
    """Response model for Mosaic spec generation (declarative mode)."""

    chain_of_thought: str = Field(
        description="""Explain your design choices for the visualization:
        - What story does the data tell?
        - Which marks (lineY, barX, barY, rectY, dot, areaY, ...) best reveal it?
        - Should the view be interactive (selections, cross-filtering, inputs)?
        Keep response to 1-2 sentences.""",
        examples=[
            "Arrival delays are heavily right-skewed, so a binned rectY histogram with an intervalX crossfilter lets users brush the tail.",
            "Closing price over time is a simple trend, best shown with a single lineY mark.",
        ],
    )
    yaml_spec: str = Field(
        description="""A mosaic-spec YAML specification. Requirements:
        - Reference the data with `data: {from: <table>}` on every mark.
        - Do NOT include a top-level `data:` block declaring files or URLs; the
          table is provided automatically from the pipeline.
        - Use vgplot mark names (lineY, barX, barY, rectY, dot, areaY, ...).
        - Encodings are column names or SQL transforms ({sum: col}, {count: },
          {bin: col}, {avg: col}).
        """,
    )


class MosaicAgent(BaseViewAgent):
    """Agent for generating interactive Mosaic/vgplot visualizations.

    Generates declarative mosaic-specs which `MosaicView` renders through the
    `Mosaic` pane, with computation pushed down to DuckDB.
    """

    conditions = param.List(
        default=[
            "Use for cross-filtered or linked views, where brushing one plot filters the others",
            "Use for several coordinated plots that share a selection, rather than a single standalone chart",
            "Use for datasets too large to embed in a chart spec, since only query results reach the browser",
            "Use when the user requests Mosaic or vgplot by name",
        ]
    )

    purpose = param.String(
        default="Generates interactive Mosaic (vgplot) visualizations from the input data pipeline."
    )

    prompts = param.Dict(
        default={
            "main": {
                "response_model": MosaicSpec,
                "template": PROMPTS_DIR / "MosaicAgent" / "main.jinja2",
            },
            "revise_output": {
                "response_model": RetrySpec,
                "template": PROMPTS_DIR / "MosaicAgent" / "revise_output.jinja2",
            },
        }
    )

    user = param.String(default="Mosaic")

    view_type = MosaicView

    _editor_type = MosaicEditor

    def __init__(self, **params):
        self._last_output = None
        super().__init__(**params)

    @retry_llm_output()
    async def _generate_yaml_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc: str,
        errors: list | None = None,
    ) -> dict[str, Any]:
        """Generate a Mosaic spec via YAML (declarative mode)."""
        errors_context = self._build_errors_context(pipeline, context, errors)

        with self._add_step(title="Generating Mosaic specification", steps_layout=self._steps_layout) as step:
            response = self._stream_prompt(
                "main",
                messages,
                context,
                table=pipeline.table,
                doc=doc,
                **errors_context,
            )
            async for output in response:
                step.stream(output.chain_of_thought, replace=True)

            step.stream(f"\n```yaml\n{output.yaml_spec}\n```", replace=False)
            step.success_title = "Mosaic specification created"

        self._last_output = {"yaml_spec": output.yaml_spec}
        return await self._extract_spec(context, {"yaml_spec": output.yaml_spec})

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate a Mosaic spec, returning view parameters."""
        if yaml_spec := spec.get("yaml_spec"):
            mosaic_spec = load_yaml(yaml_spec)
        else:
            mosaic_spec = dict(spec)

        # Lumen injects the table from the pipeline, so a top-level `data:` block
        # declaring files/URLs would be ignored and can conflict with the
        # injected table; drop it. Marks still reference the table by name via
        # `data: {from: <table>}`.
        if isinstance(mosaic_spec, dict):
            mosaic_spec.pop("data", None)

        log_debug(f"{self.name} generated Mosaic spec:\n{mosaic_spec!r}")
        self._editor_type.validate_spec(mosaic_spec)

        return {
            "spec": mosaic_spec,
            "sizing_mode": "stretch_both",
            "min_height": 400,
        }

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], TContext]:
        """Generate a Mosaic visualization."""
        pipeline = context.get("pipeline")
        if not pipeline:
            raise ValueError("Context did not contain a pipeline.")

        schema = await get_schema(pipeline)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__

        full_dict = await self._generate_yaml_spec(messages, context, pipeline, doc)

        view = self.view_type(pipeline=pipeline, **full_dict)
        out = self._editor_type(component=view, title=step_title)
        out_context = await out.render_context()
        return [out], out_context
