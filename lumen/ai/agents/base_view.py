import traceback

from typing import Any, NotRequired

import param

from ...config import dump_yaml, load_yaml
from ...pipeline import Pipeline
from ..config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR, MissingContextError
from ..context import ContextModel, TContext
from ..llm import Message
from ..schemas import Metaset
from ..utils import (
    get_root_exception, get_schema, log_debug, report_error, retry_llm_output,
)
from .base_lumen import BaseLumenAgent


class ViewInputs(ContextModel):

    data: Any

    pipeline: Pipeline

    metaset: NotRequired[Metaset]

    table: str


class ViewOutputs(ContextModel):

    view = Any


class BaseViewAgent(BaseLumenAgent):

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "BaseViewAgent" / "main.jinja2"},
        }
    )

    input_schema = ViewInputs
    output_schema = ViewOutputs

    def __init__(self, **params):
        self._last_output = None
        super().__init__(**params)

    def _build_errors_context(self, pipeline: Pipeline, context: TContext, errors: list[str] | None) -> dict:
        errors_context = {}
        if errors:
            errors = ("\n".join(f"{i + 1}. {error}" for i, error in enumerate(errors))).strip()
            try:
                last_output = load_yaml(self._last_output["yaml_spec"])
            except Exception:
                last_output = ""

            catalog = None
            if "metaset" in context:
                catalog = context["metaset"].catalog
            elif "dbtsl_metaset" in context:
                catalog = context["dbtsl_metaset"].catalog

            columns_context = ""
            if catalog is not None:
                for table_slug, catalog_entry in catalog.items():
                    table_name = table_slug.split(SOURCE_TABLE_SEPARATOR)[-1]
                    if table_name in pipeline.table:
                        columns = [col.name for col in catalog_entry.columns]
                        columns_context += f"\nSQL: {catalog_entry.sql_expr}\nColumns: {', '.join(columns)}\n\n"
            errors_context = {
                "errors": errors,
                "last_output": last_output,
                "columns_context": columns_context,
            }
        return errors_context

    @retry_llm_output()
    async def _generate_yaml_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        schema: dict[str, Any],
        step_title: str | None = None,
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        errors_context = self._build_errors_context(pipeline, context, errors)
        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__
        system = await self._render_prompt(
            "main",
            messages,
            context,
            table=pipeline.table,
            doc=doc,
            **errors_context,
        )

        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
        response = self.llm.stream(
            messages,
            system=system,
            model_spec=model_spec,
            response_model=self._get_model("main", schema=schema),
        )

        e = None
        error = ""
        spec = ""
        with self._add_step(
            title=step_title or "Generating view...", steps_layout=self._steps_layout
        ) as step:
            async for output in response:
                chain_of_thought = output.chain_of_thought or ""
                step.stream(chain_of_thought, replace=True)

            self._last_output = spec = dict(output)

            for i in range(3):
                try:
                    spec = await self._extract_spec(context, spec)
                    break
                except Exception as e:
                    e = get_root_exception(e, exceptions=(MissingContextError,))
                    if isinstance(e, MissingContextError):
                        raise e

                    error = str(e)
                    traceback.print_exception(e)
                    context = f"```\n{dump_yaml(load_yaml(self._last_output['yaml_spec']))}\n```"
                    report_error(e, step, language="json", context=context, status="failed")
                    with self._add_step(
                        title="Re-attempted view generation",
                        steps_layout=self._steps_layout,
                    ) as retry_step:
                        view = await self.revise(e, messages, context, dump_yaml(spec), language="yaml")
                        if "yaml_spec: " in view:
                            view = view.split("yaml_spec: ")[-1].rstrip('"').rstrip("'")
                        retry_step.stream(f"\n\n```yaml\n{view}\n```")
                    if i == 2:
                        raise

        self._last_output = spec

        if error:
            raise ValueError(error)

        log_debug(f"{self.name} settled on spec: {spec!r}.")
        return spec

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]):
        return dict(spec)

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], ViewOutputs]:
        """
        Generates a visualization based on user messages and the current data pipeline.
        """
        pipeline = context.get("pipeline")
        if not pipeline:
            raise ValueError("Context did not contain a pipeline.")

        schema = await get_schema(pipeline)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        spec = await self._generate_yaml_spec(messages, context, pipeline, schema, step_title)
        view = self.view_type(pipeline=pipeline, **spec)
        out = self._editor_type(component=view, title=step_title)
        return [out], {"view": dict(spec, type=self.view_type.view_type)}
