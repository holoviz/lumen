import asyncio
import json
import typing as t

import param

from pydantic import BaseModel, Field

from ...pipeline import Pipeline
from ...sources.base import BaseSQLSource, Source
from ...transforms.sql import SQLLimit
from ..config import PROMPTS_DIR
from ..context import TContext
from ..editors import SQLEditor
from ..llm import Message
from ..models import RetrySpec
from ..schemas import DbtslMetaset, get_metaset
from ..services import DbtslMixin
from ..utils import (
    get_pipeline, report_error, retry_llm_output, stream_details,
)
from .base_lumen import BaseLumenAgent
from .sql import SQLOutputs


class DbtslQueryParams(BaseModel):
    """
    Model for dbtsl.client.query() parameters.
    """

    chain_of_thought: str = Field(
        description="""You are a world-class dbt Semantic Layer expert. Think step by step about
        what metrics are needed, what dimensions to group by, what time granularity
        to use, and any filters that should be applied; if filters are applied, include those
        filtered dimensions in group_by. If there are errors, mention how you'll address the errors.
        """
    )

    expr_slug: str = Field(
        description="""Give the query a concise, but descriptive, slug that includes the metrics
        and dimensions used, e.g. monthly_revenue_by_region. The slug must be unique."""
    )

    where: list[str] = Field(
        default_factory=list,
        description="A list of conditions to filter the results; dimensions referenced here must also be in group_by, e.g. ['metric_time__month >= date_trunc('month', '2024-09-30'::date)']"
    )

    group_by: list[str] = Field(
        default_factory=list,
        description="A list of dimensions to group by, e.g. ['metric_time__month'], must include dimensions from where."
    )

    limit: int = Field(
        default=None,
        description="The maximum number of rows to return."
    )

    metrics: list[str] = Field(
        default_factory=list,
        description="A list of metrics to include in the query, e.g. ['revenue']"
    )

    order_by: list[str] = Field(
        default_factory=list,
        description="A list of columns or expressions to order the results by, e.g. ['metric_time__month']"
    )


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

    _editor_type = SQLEditor

    output_schema = DbtslOutputs

    def __init__(self, source: Source, **params):
        super().__init__(source=source, **params)

    @retry_llm_output()
    async def _create_valid_query(
        self, messages: list[Message], title: str | None = None, errors: list | None = None
    ) -> tuple[Pipeline, str]:
        """
        Create a valid dbt Semantic Layer query based on user messages.
        """
        with self._add_step(title=title or "dbt Semantic Layer query", steps_layout=self._steps_layout) as step:
            response = self._stream_prompt(
                "main",
                messages,
                errors=errors,
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

            # Apply transforms
            sql_transforms = [SQLLimit(limit=1_000_000, write=self.source.dialect, pretty=True, identify=False)]
            transformed_sql_query = sql_query
            for sql_transform in sql_transforms:
                transformed_sql_query = sql_transform.apply(transformed_sql_query)
                if transformed_sql_query != sql_query:
                    stream_details(f"```sql\n{transformed_sql_query}\n```", step, title=f"{sql_transform.__class__.__name__} Applied")

            # Create pipeline and get data
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms)
        except Exception as e:
            report_error(e, step)
            raise e
        return pipeline, sql_query

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[t.Any], DbtslOutputs]:
        """
        Responds to user messages by generating and executing a dbt Semantic Layer query.
        """
        pipeline, sql_query = await self._create_valid_query(messages, step_title)
        out = self._editor_type(
            component=pipeline, title=step_title, spec=sql_query
        )
        context = await out.render_context()
        context["dbtsl_metaset"] = await get_metaset([pipeline.source], [pipeline.table])
        return [out], t.cast(DbtslOutputs, context)
