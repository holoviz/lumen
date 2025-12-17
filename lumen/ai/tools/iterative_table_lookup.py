import traceback

from typing import Any, Literal

import param

from pydantic import create_model
from pydantic.fields import FieldInfo

from ..config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from ..context import ContextModel, TContext
from ..llm import Message
from ..models import PartialBaseModel
from ..schemas import Metaset, get_metaset
from ..utils import (
    get_schema, log_debug, parse_table_slug, stream_details, truncate_string,
)
from .metadata_lookup import MetadataLookup
from .vector_lookup import make_refined_query_model


def make_iterative_selection_model(table_slugs):
    """
    Creates a model for table selection in the coordinator.
    This is separate from the SQLAgent's table selection model to focus on context gathering
    rather than query execution.
    """
    table_model = create_model(
        "IterativeTableSelection",
        chain_of_thought=(
            str,
            FieldInfo(
                description="""
            Consider which tables would provide the most relevant context for understanding the user query,
            and be specific about what columns are most relevant to the query. Please be concise.
            """
            ),
        ),
        is_done=(
            bool,
            FieldInfo(
                description="""
            Indicate whether you have sufficient context about the available data structures.
            Set to True if you understand enough about the data model to proceed with the query.
            Set to False if you need to examine additional tables to better understand the domain.
            Do not reach for perfection; if there are promising tables, but you are not 100% sure,
            still mark as done.
            """
            ),
        ),
        requires_join=(
            bool,
            FieldInfo(
                description="""
            Indicate whether a join is necessary to answer the user query.
            Set to True if a join is necessary to answer the query.
            Set to False if no join is necessary.
            """
            ),
        ),
        selected_slugs=(
            list[Literal[tuple(table_slugs)]],
            FieldInfo(
                description="""
            If is_done and a join is necessary, return a list of table names that will be used in the join.
            If is_done and no join is necessary, return a list of a single table name.
            If not is_done return a list of table names that you believe would provide
            the most relevant context for understanding the user query that wasn't already seen before.
            Focus on tables that contain key entities, relationships, or metrics mentioned in the query.
            """
            ),
        ),
        __base__=PartialBaseModel,
    )
    return table_model


class IterativeTableLookupOutputs(ContextModel):
    metaset: Metaset


class IterativeTableLookup(MetadataLookup):
    """
    Extended version of MetadataLookup that performs an iterative table selection process.
    This tool uses an LLM to select tables in multiple passes, examining schemas in detail.
    """

    conditions = param.List(
        default=[
            "Useful for answering data queries",
            "Avoid for follow-up questions when existing data is sufficient",
            "Use only when existing information is insufficient for the current query",
        ]
    )

    not_with = param.List(default=["MetadataLookup"])

    purpose = param.String(
        default="""
        Looks up the most relevant tables and provides SQL schemas of those tables."""
    )

    max_selection_iterations = param.Integer(
        default=3,
        doc="""
        Maximum number of iterations for the iterative table selection process.""",
    )

    table_similarity_threshold = param.Number(
        default=0.5,
        doc="""
        If any tables have a similarity score above this threshold,
        those tables will be automatically selected and there will not be an
        iterative table selection process.""",
    )

    prompts = param.Dict(
        default={
            "refine_query": {
                "template": PROMPTS_DIR / "VectorLookupTool" / "refine_query.jinja2",
                "response_model": make_refined_query_model,
            },
            "iterative_selection": {
                "template": PROMPTS_DIR / "IterativeTableLookup" / "iterative_selection.jinja2",
                "response_model": make_iterative_selection_model,
            },
        },
    )

    output_schema = IterativeTableLookupOutputs

    async def sync(self, context: TContext):
        """
        Update metaset when visible_slugs changes.
        This ensures SQL operations only work with visible tables by regenerating
        the metaset using get_metaset.
        """
        slugs = context.get("visible_slugs", set())

        # If no visible slugs or no sources, clear metaset
        if not slugs or not context.get("sources"):
            context.pop("metaset", None)
            return

        if "metaset" not in context:
            return

        try:
            sources = context["sources"]
            visible_tables = list(slugs)
            new_metaset = await get_metaset(sources, visible_tables, prev=context.get("metaset"))
            context["metaset"] = new_metaset
            log_debug(f"[IterativeTableLookup] Updated metaset with {len(visible_tables)} visible tables")
        except Exception as e:
            log_debug(f"[IterativeTableLookup] Error updating metaset: {e}")

    async def _gather_info(self, messages: list[dict[str, str]], context: TContext) -> IterativeTableLookupOutputs:
        """
        Performs an iterative table selection process to gather context.
        This function:
        1. From the top tables in MetadataLookup results, presents all tables and columns to the LLM
        2. Has the LLM select ~3 tables for deeper examination
        3. Gets complete schemas for these tables
        4. Repeats until the LLM is satisfied with the context
        """
        out_model = await super()._gather_info(messages, context)
        metaset = out_model["metaset"]
        catalog = metaset.catalog

        schemas = {}
        examined_slugs = set(schemas.keys())
        # Filter to only include visible tables
        all_slugs = list(catalog.keys())
        visible_slugs = context.get("visible_slugs", set())
        if visible_slugs:
            all_slugs = [slug for slug in all_slugs if slug in visible_slugs]

        if len(all_slugs) == 1:
            table_slug = all_slugs[0]
            source_obj, table_name = parse_table_slug(table_slug, context["sources"])
            schema = await get_schema(source_obj, table_name, reduce_enums=False)
            return {
                "metaset": Metaset(
                    catalog=catalog,
                    schemas={table_slug: schema},
                    query=metaset.query,
                )
            }

        failed_slugs = []
        selected_slugs = []
        chain_of_thought = ""
        max_iterations = self.max_selection_iterations
        sources = {source.name: source for source in context["sources"]}

        for iteration in range(1, max_iterations + 1):
            with self._add_step(
                title=f"Iterative table selection {iteration} / {max_iterations}",
                success_title=f"Selection iteration {iteration} / {max_iterations} completed",
                steps_layout=self.steps_layout,
            ) as step:
                available_slugs = [table_slug for table_slug in all_slugs if table_slug not in examined_slugs]
                log_debug(available_slugs, prefix=f"Available slugs for iteration {iteration}")
                if not available_slugs:
                    step.stream("No more tables available to examine.")
                    break

                fast_track = False
                if iteration == 1:
                    # For the first iteration, select tables based on similarity
                    # If any tables have a similarity score above the threshold, select up to 5 of those tables
                    any_matches = any(schema.similarity > self.table_similarity_threshold for schema in catalog.values())
                    limited_tables = len(catalog) <= 5
                    if any_matches or len(all_slugs) == 1 or limited_tables:
                        selected_slugs = sorted(catalog.keys(), key=lambda x: catalog[x].similarity, reverse=True)[:5]
                        step.stream("Selected tables based on similarity threshold.\n\n")
                        fast_track = True

                if not fast_track:
                    try:
                        step.stream(f"Selecting tables from {len(available_slugs)} available options\n\n")
                        # Render the prompt using the proper template system

                        system = await self._render_prompt(
                            "iterative_selection",
                            messages,
                            context,
                            chain_of_thought=chain_of_thought,
                            current_query=messages[-1]["content"],
                            available_slugs=available_slugs,
                            examined_slugs=examined_slugs,
                            selected_slugs=selected_slugs,
                            failed_slugs=failed_slugs,
                            schemas=schemas,
                            catalog=catalog,
                            iteration=iteration,
                            max_iterations=max_iterations,
                        )
                        model_spec = self.prompts["iterative_selection"].get("llm_spec", self.llm_spec_key)
                        find_tables_model = self._get_model("iterative_selection", table_slugs=available_slugs + list(examined_slugs))
                        output = await self.llm.invoke(
                            messages,
                            system=system,
                            model_spec=model_spec,
                            response_model=find_tables_model,
                        )

                        selected_slugs = output.selected_slugs or []
                        chain_of_thought = output.chain_of_thought
                        step.stream(chain_of_thought)
                        stream_details("\n\n".join(f"`{slug}`" for slug in selected_slugs), step, title=f"Selected {len(selected_slugs)} tables", auto=False)

                        # Check if we're done with table selection (do not allow on 1st iteration)
                        if (output.is_done and iteration != 1) or not available_slugs:
                            step.stream("\n\nSelection process complete - model is satisfied with selected tables")
                            fast_track = True
                        elif iteration != max_iterations:
                            step.stream("\n\nUnsatisfied with selected tables - continuing selection process...")
                        else:
                            step.stream("\n\nMaximum iterations reached - stopping selection process")
                            break
                    except Exception as e:
                        traceback.print_exc()
                        stream_details(f"\n\nError selecting tables: {e}", step, title="Error", auto=False)
                        continue

                step.stream("\n\nFetching detailed schema information for tables\n")
                for table_slug in selected_slugs:
                    stream_details(str(metaset.catalog[table_slug]), step, title="Table details", auto=False)
                    try:
                        view_definition = truncate_string(context["tables_metadata"].get(table_slug, {}).get("view_definition", ""), max_length=300)

                        if SOURCE_TABLE_SEPARATOR in table_slug:
                            source_name, table_name = table_slug.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)
                            source_obj = sources.get(source_name)
                        else:
                            source_obj = next(iter(sources.values()))
                            table_name = table_slug
                            table_slug = f"{source_obj.name}{SOURCE_TABLE_SEPARATOR}{table_name}"

                        if view_definition:
                            schema = {"view_definition": view_definition}
                        else:
                            schema = await get_schema(source_obj, table_name, reduce_enums=False)

                        schemas[table_slug] = schema

                        examined_slugs.add(table_slug)
                        stream_details(f"```yaml\n{schema}\n```", step, title="Schema")
                    except Exception as e:
                        failed_slugs.append(table_slug)
                        stream_details(f"Error fetching schema: {e}", step, title="Error", auto=False)

                if fast_track:
                    step.stream("Fast-tracked selection process.")
                    # based on similarity search alone, if we have selected tables, we're done!!
                    break

        metaset = Metaset(
            catalog=catalog,
            schemas=schemas,
            query=metaset.query,
            docs=metaset.docs,
        )
        return {"metaset": metaset}

    def _format_context(self, outputs: IterativeTableLookupOutputs) -> str:
        """Generate formatted text representation from schema objects."""
        metaset = outputs.get("metaset")
        return str(metaset)

    async def respond(self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]) -> tuple[list[Any], IterativeTableLookupOutputs]:
        """
        Fetches tables based on the user query and returns formatted context.
        """
        out_model = await self._gather_info(messages, context)
        return [self._format_context(out_model)], out_model
