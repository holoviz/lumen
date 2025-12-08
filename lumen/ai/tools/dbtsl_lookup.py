import asyncio

from typing import Any

import param

from panel.io import cache as pn_cache

from .config import PROMPTS_DIR
from .context import ContextModel, TContext
from .llm import Message
from .schemas import DbtslMetadata, DbtslMetaset
from .services import DbtslMixin
from .shared.models import ThinkingYesNo, make_refined_query_model
from .tools.vector_lookup import VectorLookupTool
from .utils import process_enums, stream_details, with_timeout


class DbtslLookupOutputs(ContextModel):
    dbtsl_metaset: DbtslMetaset


class DbtslLookup(VectorLookupTool, DbtslMixin):
    """
    DbtslLookup tool that creates a vector store of all available dbt semantic layers
    and responds with relevant metrics for user queries.
    """

    prompts = param.Dict(
        default={
            "refine_query": {
                "template": PROMPTS_DIR / "VectorLookupTool" / "refine_query.jinja2",
                "response_model": make_refined_query_model,
            },
            "main": {
                "template": PROMPTS_DIR / "DbtslLookup" / "main.jinja2",
                "response_model": ThinkingYesNo,
            },
        },
        doc="Dictionary of available prompts for the tool."
    )

    dimension_fetch_timeout = param.Number(default=15, doc="""
        Maximum time in seconds to wait for a dimension values fetch operation.""")

    max_concurrent = param.Integer(default=5, doc="""
        Maximum number of concurrent metadata fetch operations.""")

    min_similarity = param.Number(default=0.1, doc="""
        The minimum similarity to include a document.""")

    n = param.Integer(default=4, bounds=(1, None), doc="""
        The number of document results to return.""")

    purpose = param.String(default="""
        Looks up additional context by querying dbt semantic layers based on the user query with a vector store.
        Useful for quickly gathering information about dbt semantic layers and their metrics to plan the steps.
        Not useful for looking up what datasets are available. Likely useful for all queries.""")

    _item_type_name = "metrics"

    output_schema = DbtslLookupOutputs

    def __init__(self, environment_id: int, **params):
        params["environment_id"] = environment_id
        super().__init__(**params)
        self._metric_objs = {}

    async def _update_vector_store(self, context: TContext):
        """
        Updates the vector store with metrics from the dbt semantic layer client.

        This method fetches all metrics from the dbt semantic layer client and
        adds them to the vector store for semantic search.
        """
        client = self._instantiate_client()
        async with client.session():
            self._metric_objs = {metric.name: metric for metric in await client.metrics()}

        # Build a list of all metrics for a single upsert operation
        items_to_upsert = []
        for metric_name, metric in self._metric_objs.items():
            enriched_text = f"Metric: {metric_name}"
            if metric.description:
                enriched_text += f"\nInfo: {metric.description}"

            vector_metadata = {
                "type": self._item_type_name,
                "name": metric_name,
                "metric_type": str(metric.type.value) if metric.type else "UNKNOWN",
            }

            items_to_upsert.append({"text": enriched_text, "metadata": vector_metadata, "chunk_size": None})

        # Make a single upsert call with all metrics
        if items_to_upsert:
            await self.vector_store.upsert(items_to_upsert)

    async def respond(
        self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]
    ) -> tuple[list[Any], TContext]:
        """
        Fetches metrics based on the user query, populates the DbtslMetaset,
        and returns formatted context.
        """
        query = messages[-1]["content"]

        # Search for relevant metrics
        with self._add_step(title="dbt Semantic Layer Search") as search_step:
            search_step.stream(f"Searching for metrics relevant to: '{query}'")
            results = await self._perform_search_with_refinement(query, context)
            closest_metrics = [r for r in results if r['similarity'] >= self.min_similarity]

            if not closest_metrics:
                search_step.stream("\n⚠️ No metrics found with sufficient relevance to the query.")
                search_step.status = "failed"
                context.pop("dbtsl_metaset", None)
                return [], context

            metrics_info = [f"- {r['metadata'].get('name')}" for r in closest_metrics]
            stream_details("\n".join(metrics_info), search_step, title=f"Found {len(closest_metrics)} relevant chunks", auto=False)

        # Process metrics and fetch dimensions
        can_answer_query = False
        with self._add_step(title="Processing metrics") as step:
            step.stream("Processing metrics and their dimensions")

            # Collect unique metrics and prepare dimension fetch tasks
            metric_names = {r['metadata'].get('name', "") for r in closest_metrics}
            metric_objects = {name: self._metric_objs[name] for name in metric_names if name in self._metric_objs}

            # Create flat list of all dimension fetch tasks
            client = self._instantiate_client()
            async with client.session():
                num_cols = len(closest_metrics)

                # Build a flat list of all (metric_name, dimension) pairs
                all_dimensions = [
                    (metric_name, dim)
                    for metric_name, metric_obj in metric_objects.items()
                    for dim in metric_obj.dimensions
                ]

                # Log metric descriptions
                for metric_name, metric_obj in metric_objects.items():
                    step.stream(f"\n\n`{metric_name}`: {metric_obj.description}")

                # Fetch all dimensions in parallel
                step.stream("\n")
                if all_dimensions:
                    # Single gather call for all dimensions across all metrics
                    async with asyncio.Semaphore(self.max_concurrent):
                        dimension_tasks = [
                            with_timeout(
                                self._fetch_dimension_values(client, metric_name, dim, num_cols, step),
                                timeout_seconds=self.dimension_fetch_timeout,
                                default_value=(
                                    dim.name.upper(), {"type": dim.type.value}),
                                error_message=f"Dimension fetch timed out for {metric_name}.{dim.name}"
                            )
                            for metric_name, dim in all_dimensions
                        ]
                        all_results = await asyncio.gather(*dimension_tasks)

                    # Organize results by metric
                    dimension_results = {}
                    for i, (metric_name, _) in enumerate(all_dimensions):
                        if metric_name not in dimension_results:
                            dimension_results[metric_name] = {}

                        dim_name, dim_info = all_results[i]
                        dimension_results[metric_name][dim_name] = dim_info
                else:
                    dimension_results = {}

                # Build metrics dictionary with dimension data
                metrics = {}
                for result in closest_metrics:
                    metric_name = result['metadata']['name']
                    if metric_name in metrics:
                        continue

                    metric_obj = metric_objects[metric_name]
                    metrics[metric_name] = DbtslMetadata(
                        name=metric_name,
                        similarity=result['similarity'],
                        description=metric_obj.description,
                        dimensions=dimension_results.get(metric_name, {}),
                        queryable_granularities=[g.name for g in metric_obj.queryable_granularities]
                    )

                # Create metaset and evaluate if metrics can answer query
                metaset = DbtslMetaset(query=query, metrics=metrics)

                if metrics:
                    step.stream("\n\nEvaluating if found metrics can answer the query...")
                    try:
                        result = await self._invoke_prompt("main", messages, dbtsl_metaset=metaset)
                        can_answer_query = result.yes
                        step.stream(f"\n\n{result.chain_of_thought}")
                    except Exception as e:
                        step.stream(f"\n\nError evaluating metrics and dimensions: {e}")
                        step.status = "failed"

        if not can_answer_query:
            context.pop("dbtsl_metaset", None)
            return [], context

        context["dbtsl_metaset"] = metaset
        return [str(metaset)], context

    @pn_cache
    async def _fetch_dimension_values(self, client, metric_name, dim, num_cols, step):
        dim_name = dim.name.upper()
        dim_info = {"type": dim.type.value}

        if dim.type.value == "CATEGORICAL":
            try:
                enums = await client.dimension_values(
                    metrics=[metric_name], group_by=dim_name
                )
                if dim_name in enums.to_pydict():
                    enum_values = enums.to_pydict()[dim_name]
                    spec, _ = process_enums({"enum": enum_values}, num_cols)
                    dim_info["enum"] = spec["enum"]
                    dim_info["enum_count"] = len(enum_values)
                    # just to show its progressing...
                    step.stream(".")
                    # stream_details(spec["enum"], step, auto=False, title=f"{metric_name}.{dim_name} {dim_info["enum_count"]} values")
                else:
                    dim_info["enum"] = []
                    dim_info["enum_count"] = 0
                    dim_info["error"] = f"No values found for dimension {dim_name}"
            except Exception as e:
                dim_info["error"] = f"Error fetching dimension values: {e!s}"
        return dim_name, dim_info
