from __future__ import annotations

from typing import Any

import param

from pydantic import BaseModel, Field

from ...pipeline import Pipeline
from ...sources.duckdb import DuckDBSource
from ...util import normalize_table_name
from ..code_executor import CodeSafetyCheck, PandasExecutor
from ..config import PROMPTS_DIR, UserCancelledError
from ..context import ContextModel, TContext
from ..llm import Message
from ..utils import describe_data, get_data
from ..views import LumenOutput
from .base_code import BaseCodeAgent


class TransformSpec(BaseModel):
    """LLM response model for pandas transformation code."""

    chain_of_thought: str = Field(
        description="Brief reasoning (1-2 sentences) for the transformation strategy."
    )
    table_slug: str = Field(
        description=(
            "Short, descriptive snake_case name for the transformed table "
            "(e.g. filtered_orders_2024)."
        )
    )
    code: str = Field(
        description=(
            "Python code that transforms the input DataFrame `df` into a new DataFrame "
            "assigned to `df_out`. Use pandas (pd) and optionally numpy (np). "
            "Do not perform any I/O."
        )
    )


class CodeTransformInputs(ContextModel):
    data: Any
    pipeline: Pipeline
    table: str


class CodeTransformOutputs(ContextModel):
    code: str
    data: Any
    pipeline: Pipeline
    source: DuckDBSource
    table: str


class CodeTransformAgent(BaseCodeAgent):
    """
    Generates pandas transformation code, executes it, and exposes the result as a DuckDB view.
    """

    conditions = param.List(
        default=[
            "If no pipeline is available you MUST use a SQL agent to make the data available first",
            "Use when the user asks to clean, reshape, or transform data that is not easily achievable in SQL",
            "Use for feature engineering, column creation, filtering, or aggregation in pandas",
            "Use when the user wants a new derived table from the current dataset",
        ]
    )

    purpose = param.String(
        default="Generates pandas code to transform the current DataFrame into a new table."
    )

    prompts = param.Dict(
        default={
            "main": {
                "response_model": TransformSpec,
                "template": PROMPTS_DIR / "CodeTransformAgent" / "main.jinja2",
            },
            "code_safety": {
                "response_model": CodeSafetyCheck,
                "template": PROMPTS_DIR / "CodeTransformAgent" / "code_safety.jinja2",
            },
        }
    )

    user = param.String(default="Transform")

    _executor_class = PandasExecutor

    _output_type = LumenOutput

    input_schema = CodeTransformInputs
    output_schema = CodeTransformOutputs

    async def _generate_code_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        errors: list[str] | None = None,
    ) -> dict[str, Any] | None:
        errors_context = self._build_errors_context(pipeline, context, errors)
        try:
            available_tables = pipeline.source.get_tables()
        except Exception:
            available_tables = []

        with self._add_step(title="Generating transformation code", steps_layout=self._steps_layout) as step:
            system_prompt = await self._render_prompt(
                "main",
                messages,
                context,
                table=pipeline.table,
                tables=available_tables,
                **errors_context,
            )

            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=TransformSpec,
            )

            async for output in response:
                step.stream(output.chain_of_thought, replace=True)

            step.stream(f"\n```python\n{output.code}\n```\n", replace=False)

            system = None
            if self.code_execution == "llm":
                system = await self._render_prompt(
                    "code_safety",
                    messages,
                    context,
                    code=output.code,
                )

            df = await get_data(pipeline)
            transformed = await self._execute_code(output.code, df, system=system, step=step)

        if transformed is None:
            raise UserCancelledError("Code execution rejected by user.")

        table_slug = normalize_table_name(output.table_slug or f"{pipeline.table}_transformed")

        source = DuckDBSource(uri=":memory:", mirrors={table_slug: transformed}, ephemeral=True)
        new_pipeline = Pipeline(source=source, table=table_slug)

        return {
            "code": output.code,
            "pipeline": new_pipeline,
            "source": source,
            "table": table_slug,
            "data": transformed,
        }

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], CodeTransformOutputs]:
        pipeline = context.get("pipeline")
        if not pipeline:
            raise ValueError("Context did not contain a pipeline.")

        result = await self._generate_code_spec(messages, context, pipeline)
        if result is None:
            return [], {}

        out = self._output_type(component=result["pipeline"], title=step_title)
        out_context = {
            "code": result["code"],
            "pipeline": result["pipeline"],
            "source": result["source"],
            "table": result["table"],
            "data": await describe_data(result["data"]),
        }
        return [out], out_context
