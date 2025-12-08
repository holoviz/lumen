import asyncio
import json
import traceback

from typing import Any

import param

from ...sources.base import BaseSQLSource, Source
from ...transforms.sql import SQLLimit
from ..config import PROMPTS_DIR, RetriesExceededError
from ..context import TContext
from ..llm import Message
from ..schemas import DbtslMetaset, get_metaset
from ..services import DbtslMixin
from ..shared.models import DbtslQueryParams, RetrySpec
from ..utils import (
    describe_data, get_data, get_pipeline, report_error, retry_llm_output,
    stream_details,
)
from ..views import LumenOutput
from .lumen import BaseLumenAgent
from .sql import SQLOutputs


class DbtslOutputs(SQLOutputs):

    dbtsl_metaset: DbtslMetaset


class DbtslAgent(BaseLumenAgent, DbtslMixin):
    """
    Responsible for creating and executing queries against a dbt Semantic Layer
    to answer user questions about business metrics.
    """

    conditions = param.List(
        default=[
            "Always use this when dbtsl_metaset is available",
        ]
    )

    purpose = param.String(
        default="""
        Responsible for displaying data to answer user queries about
        business metrics using dbt Semantic Layers. This agent can compile
        and execute metric queries against a dbt Semantic Layer."""
    )

    prompts = param.Dict(
        default={
            "main": {
                "response_model": DbtslQueryParams,
                "template": PROMPTS_DIR / "DbtslAgent" / "main.jinja2",
            },
            "revise_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "BaseLumenAgent" / "revise_output.jinja2"},
        }
    )

    requires = param.List(default=["source", "dbtsl_metaset"], readonly=True)

    source = param.ClassSelector(
        class_=BaseSQLSource,
        doc="""
        The source associated with the dbt Semantic Layer.""",
    )

    user = param.String(default="DBT")

    _extensions = ("codeeditor", "tabulator")

    _output_type = LumenOutput

    output_schema = DbtslOutputs

    def __init__(self, source: Source, **params):
        super().__init__(source=source, **params)

    @retry_llm_output()
    async def _create_valid_query(
        self, messages: list[Message], title: str | None = None, errors: list | None = None
    ) -> DbtslOutputs:
        """
        Create a valid dbt Semantic Layer query based on user messages.
        """
        system_prompt = await self._render_prompt(
            "main",
            messages,
            errors=errors,
        )

        out_context = {}
        with self._add_step(title=title or "dbt Semantic Layer query", steps_layout=self._steps_layout) as step:
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=DbtslQueryParams,
            )

            query_params = None
            try:
                async for output in response:
                    step_message = output.chain_of_thought or ""
                    step.stream(step_message, replace=True)

                query_params = {
                    "metrics": output.metrics,
                    "group_by": output.group_by,
                    "limit": output.limit,
                    "order_by": output.order_by,
                    "where": output.where,
                }
                expr_slug = output.expr_slug

                # Ensure required fields are present
                if "metrics" not in query_params or not query_params["metrics"]:
                    raise ValueError("Query must include at least one metric")

                # Ensure limit exists with a default value if not provided
                if "limit" not in query_params or not query_params["limit"]:
                    query_params["limit"] = 10000

                formatted_params = json.dumps(query_params, indent=2)
                step.stream(f"\n\n`{expr_slug}`\n```json\n{formatted_params}\n```")

                out_context["dbtsl_query_params"] = query_params
            except asyncio.CancelledError as e:
                step.failed_title = "Cancelled dbt Semantic Layer query generation"
                raise e

        if step.failed_title and step.failed_title.startswith("Cancelled"):
            raise asyncio.CancelledError()
        elif not query_params:
            raise ValueError("No dbt Semantic Layer query was generated.")

        try:
            # Execute the query against the dbt Semantic Layer
            client = self._instantiate_client()
            async with client.session():
                sql_query = await client.compile_sql(
                    metrics=query_params.get("metrics", []),
                    group_by=query_params.get("group_by"),
                    limit=query_params.get("limit"),
                    order_by=query_params.get("order_by"),
                    where=query_params.get("where"),
                )

                # Log the compiled SQL for debugging
                step.stream(f"\nCompiled SQL:\n```sql\n{sql_query}\n```", replace=False)

            sql_expr_source = self.source.create_sql_expr_source({expr_slug: sql_query})
            out_context["sql"] = sql_query

            # Apply transforms
            sql_transforms = [SQLLimit(limit=1_000_000, write=self.source.dialect, pretty=True, identify=False)]
            transformed_sql_query = sql_query
            for sql_transform in sql_transforms:
                transformed_sql_query = sql_transform.apply(transformed_sql_query)
                if transformed_sql_query != sql_query:
                    stream_details(f"```sql\n{transformed_sql_query}\n```", step, title=f"{sql_transform.__class__.__name__} Applied")

            # Create pipeline and get data
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms)

            df = await get_data(pipeline)
            metaset = await get_metaset([sql_expr_source], [expr_slug])

            # Update context
            out_context["data"] = await describe_data(df)
            out_context["source"] = sql_expr_source
            out_context["pipeline"] = pipeline
            out_context["table"] = pipeline.table
            out_context["dbtsl_metaset"] = metaset
        except Exception as e:
            report_error(e, step)
            raise e
        return out_context

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], DbtslOutputs]:
        """
        Responds to user messages by generating and executing a dbt Semantic Layer query.
        """
        try:
            out_context = await self._create_valid_query(messages, step_title)
        except RetriesExceededError as e:
            traceback.print_exception(e)
            context["__error__"] = str(e)
            return None

        pipeline = out_context["pipeline"]
        view = self._output_type(
            component=pipeline, title=step_title, spec=out_context["sql"]
        )
        return [view], out_context
