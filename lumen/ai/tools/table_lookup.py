import asyncio
import traceback

from typing import Annotated, Any

import param

from ..sources.base import Source
from ..sources.duckdb import DuckDBSource
from .config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from .context import ContextModel, TContext
from .llm import Message
from .schemas import (
    Column, Metaset, TableCatalogEntry, get_metaset,
)
from .shared.models import (
    make_iterative_selection_model, make_refined_query_model,
)
from .utils import (
    get_schema, log_debug, parse_table_slug, stream_details, truncate_string,
)
from .vector_lookup_tool import VectorLookupTool


class TableLookupInputs(ContextModel):

    sources: Annotated[list[Source], ("accumulate", "source")]


class TableLookupOutputs(ContextModel):

    metaset: Metaset


class TableLookup(VectorLookupTool):
    """
    TableLookup tool that creates a vector store of all available tables
    and responds with relevant tables for user queries.
    """

    conditions = param.List(default=[
        "Best paired with ChatAgent for general conversation about data",
        "Avoid if table discovery already performed for same request",
        "Not useful for data related queries",
    ])

    prompts = param.Dict(
        default={
            "refine_query": {
                "template": PROMPTS_DIR / "VectorLookupTool" / "refine_query.jinja2",
                "response_model": make_refined_query_model,
            },
        },
        doc="Dictionary of available prompts for the tool."
    )

    exclusions = param.List(default=["dbtsl_metaset"])

    not_with = param.List(default=["IterativeTableLookup"])

    purpose = param.String(default="""
        Discovers relevant tables using vector search, providing context for other agents.
        Not to be used for finding tables for further analysis (e.g. SQL), because it does
        not provide a schema.""")

    include_metadata = param.Boolean(default=True, doc="""
        Whether to include table descriptions in the embeddings and responses.""")

    include_columns = param.Boolean(default=True, doc="""
        Whether to include column names and descriptions in the embeddings.""")

    include_misc = param.Boolean(default=False, doc="""
        Whether to include miscellaneous metadata in the embeddings,
        besides table and column descriptions.""")

    max_concurrent = param.Integer(default=1, doc="""
        Maximum number of concurrent metadata fetch operations.""")

    min_similarity = param.Number(default=0.05, doc="""
        The minimum similarity to include a document.""")

    n = param.Integer(default=10, bounds=(1, None), doc="""
        The number of document results to return.""")

    _ready = param.Boolean(default=False, allow_None=True, doc="""
        Whether the vector store is ready.""")

    _item_type_name = "tables"

    # Class variable to track which sources are currently being processed
    _sources_in_progress = {}
    _progress_lock = asyncio.Lock()  # Lock for thread-safe access

    input_schema = TableLookupInputs
    output_schema = TableLookupOutputs

    def __init__(self, **params):
        super().__init__(**params)
        self._raw_metadata = {}
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def sync(self, context: TContext):
        """
        Sync the vector store with updated context.
        Only processes new sources that haven't been added yet.
        """
        sources = context.get("sources", [])
        if not sources:
            log_debug("[TableLookup] sync called but no sources in context")
            return

        if "tables_metadata" not in context:
            context["tables_metadata"] = {}

        await self._update_vector_store(context)

    def _format_results_for_refinement(
        self, results: list[dict[str, Any]], context: TContext
    ) -> str:
        """
        Format table search results for inclusion in the refinement prompt.

        Parameters
        ----------
        results: list[dict[str, Any]]
            The search results from vector_store.query

        Returns
        -------
        str
            Formatted description of table results
        """
        visible_slugs = context.get('visible_slugs', set())
        formatted_results = []
        for result in results:
            source_name = result['metadata'].get("source", "unknown")
            table_name = result['metadata'].get("table_name", "unknown")
            table_slug = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"
            if visible_slugs and table_slug not in visible_slugs:
                continue

            text = result["text"]
            description = f"- {text} {table_slug} (Similarity: {result.get('similarity', 0):.3f})"
            if tables_vector_data := context["tables_metadata"].get(table_slug):
                if table_description := tables_vector_data.get("description"):
                    description += f"\n  Info: {table_description}"
            formatted_results.append(description)
        return "\n".join(formatted_results)

    async def _enrich_metadata(self, source, table_name: str, context: TContext):
        """Fetch metadata for a table and return enriched entry for batch processing."""
        async with self._semaphore:
            source_metadata = self._raw_metadata[source.name]
            if isinstance(source_metadata, asyncio.Task):
                source_metadata = await source_metadata
                self._raw_metadata[source.name] = source_metadata

        table_name_key = next((
            key for key in (table_name.upper(), table_name.lower(), table_name)
            if key in source_metadata), None
        )

        if not table_name_key:
            return None
        table_name = table_name_key
        vector_info = dict(source_metadata[table_name])

        # IMPORTANT: re-insert using table slug
        # we need to store a copy of tables_vector_data so it can be used to inject context into the LLM
        table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table_name}"
        context["tables_metadata"][table_slug] = vector_info.copy()
        context["tables_metadata"][table_slug]["source_name"] = source.name  # need this to rebuild the slug

        # Create column schema objects
        columns = []
        if self.include_columns and (vector_columns := vector_info.get('columns', {})):
            for col_name, col_info in vector_columns.items():
                col_desc = col_info.pop("description", "")
                column = Column(
                    name=col_name,
                    description=col_desc,
                    metadata=col_info.copy() if isinstance(col_info, dict) else {}
                )
                columns.append(column)

        vector_metadata = {"source": source.name, "table_name": table_name, "type": self._item_type_name}
        # the following is for the embeddings/vector store use only (i.e. filter from 1000s of tables)
        enriched_text = f"Table: {table_name}"
        if description := vector_info.pop('description', ''):
            enriched_text += f"\nInfo: {description}"
        if self.include_columns and (vector_columns := vector_info.pop('columns', {})):
            enriched_text += "\nCols:"
            for col_name, col_info in vector_columns.items():
                col_text = f"\n- {col_name}"
                if col_info.get("description"):
                    col_text += f": {col_info['description']}"
                enriched_text += col_text
            enriched_text += "\n"
        if self.include_misc:
            enriched_text += "\nMiscellaneous:"
            for key, value in vector_info.items():
                if key == "enriched":
                    continue
                enriched_text += f"\n- {key}: {value}"

        # Return the entry instead of upserting directly
        return {"text": enriched_text, "metadata": vector_metadata}

    async def _update_vector_store(self, context: TContext):
        sources = context.get("sources")
        if sources is None and "source" in context:
            sources = [context["source"]]
        elif sources is None or len(sources) == 0:
            log_debug("[TableLookup] _update_vector_store called but no sources available")
            return
        self._ready = False
        await asyncio.sleep(0.5)  # allow main thread time to load UI first
        tasks = []
        processed_sources = []

        # Use class variable to track which sources are currently being processed
        vector_store_id = id(self.vector_store)

        async with self._progress_lock:
            if vector_store_id not in self._sources_in_progress:
                self._sources_in_progress[vector_store_id] = set()

        log_debug(f"[TableLookup] Starting _update_vector_store for {len(sources)} sources")

        for source in sources:
            async with self._progress_lock:
                # Skip this source if it's already being processed for this vector store
                if source.name in self._sources_in_progress[vector_store_id]:
                    log_debug(f"[TableLookup] Skipping source {source.name} - already in progress")
                    continue

                # Mark this source as in progress for this vector store
                self._sources_in_progress[vector_store_id].add(source.name)
                processed_sources.append(source.name)
                log_debug(f"[TableLookup] Processing source {source.name}")

            if self.include_metadata and self._raw_metadata.get(source.name) is None:
                if isinstance(source, DuckDBSource):
                    self._raw_metadata[source.name] = source.get_metadata()
                else:
                    metadata_task = asyncio.create_task(
                        asyncio.to_thread(source.get_metadata)
                    )
                    self._raw_metadata[source.name] = metadata_task
                    tasks.append(metadata_task)

            tables = source.get_tables()

            if self.include_metadata:
                for table in tables:
                    task = asyncio.create_task(self._enrich_metadata(source, table, context))
                    tasks.append(task)
            else:
                await self.vector_store.upsert([
                    {"text": table_name, "metadata": {"source": source.name, "table_name": table_name, "type": "table"}}
                    for table_name in tables
                ])

        if tasks:
            # Pass the processed_sources to _mark_ready_when_done so it can clean up after tasks complete
            ready_task = asyncio.create_task(self._mark_ready_when_done(tasks, vector_store_id, processed_sources))
            ready_task.add_done_callback(self._handle_ready_task_done)
        else:
            self._ready = True
            # No tasks to wait for, so clean up immediately
            self._cleanup_sources_in_progress(vector_store_id, processed_sources)

    def _cleanup_sources_in_progress(self, vector_store_id, processed_sources):
        """Clean up the sources_in_progress tracking after tasks are completed."""
        async def _async_cleanup():
            async with self._progress_lock:
                try:
                    log_debug(f"[TableLookup] Cleaning up sources: {processed_sources}")
                    for source_name in processed_sources:
                        if vector_store_id in self._sources_in_progress:
                            self._sources_in_progress[vector_store_id].discard(source_name)
                            log_debug(f"[TableLookup] Removed {source_name} from in-progress")

                    # Remove the vector_store_id entry if empty
                    if vector_store_id in self._sources_in_progress and not self._sources_in_progress[vector_store_id]:
                        del self._sources_in_progress[vector_store_id]
                        log_debug(f"[TableLookup] Removed empty vector_store_id {vector_store_id}")
                except Exception as e:
                    log_debug(f"[TableLookup] Error cleaning up sources_in_progress: {e!s}")
                    # Force cleanup on error
                    if vector_store_id in self._sources_in_progress:
                        self._sources_in_progress[vector_store_id].clear()

        # Schedule the async cleanup and store the task reference
        # This prevents the task from being garbage collected
        cleanup_task = asyncio.create_task(_async_cleanup())
        # Optionally, we can add error handling for the task
        cleanup_task.add_done_callback(lambda t: t.exception() if t.done() else None)

    async def _mark_ready_when_done(self, tasks, vector_store_id=None, processed_sources=None):
        """Wait for all tasks to complete, collect results for batch upsert, and mark the tool as ready."""
        try:
            log_debug(f"[TableLookup] Waiting for {len(tasks)} tasks to complete")
            # Add timeout to prevent indefinite waiting (5 minutes)
            try:
                async def _gather_with_semaphore():
                    async with asyncio.Semaphore(self.max_concurrent):
                        return [
                            result for result in await asyncio.gather(*tasks, return_exceptions=True)
                            if isinstance(result, dict) and "text" in result
                        ]

                enriched_entries = await asyncio.wait_for(_gather_with_semaphore(), timeout=300)
            except TimeoutError:
                log_debug("[TableLookup] Timeout waiting for tasks to complete (5 minutes)")
                self._ready = None  # Set to error state
                return

            if enriched_entries:
                log_debug(f"[TableLookup] Upserting {len(enriched_entries)} enriched entries")
                # Add timeout to the upsert operation to prevent deadlock
                try:
                    await asyncio.wait_for(
                        self.vector_store.upsert(enriched_entries),
                        timeout=60
                    )
                    log_debug(f"[TableLookup] Successfully upserted {len(enriched_entries)} entries")
                except TimeoutError:
                    log_debug("[TableLookup] Timeout during upsert operation (60 seconds)")
                    self._ready = None  # Set to error state
                    return
            else:
                log_debug("[TableLookup] No enriched entries to upsert")
            log_debug("[TableLookup] All table metadata tasks completed.")
            self._ready = True
        except Exception as e:
            log_debug(f"[TableLookup] Error in _mark_ready_when_done: {e!s}")
            traceback.print_exc()
            self._ready = None  # Set to error state
            raise
        finally:
            # Clean up sources_in_progress after tasks are done, even if there was an error
            if vector_store_id is not None and processed_sources:
                self._cleanup_sources_in_progress(vector_store_id, processed_sources)

    async def _gather_info(self, messages: list[dict[str, str]], context: TContext) -> TableLookupOutputs:
        """Gather relevant information about the tables based on the user query."""
        query = messages[-1]["content"]

        # Count total number of tables available across all sources
        total_tables = 0
        for source in context.get("sources", []):
            total_tables += len(source.get_tables())

        # Skip query refinement if there are fewer than 5 tables
        if total_tables < 5:
            # Perform search without refinement
            filters = {}
            if self._item_type_name and "type" not in filters:
                filters["type"] = self._item_type_name
            results = await self.vector_store.query(query, top_k=self.n, filters=filters)
        else:
            results = await self._perform_search_with_refinement(query, context)

        any_matches = any(result["similarity"] >= self.min_similarity for result in results)
        same_table = len([result["metadata"]["table_name"] for result in results])

        catalog = {}
        for result in results:
            source_name = result["metadata"]["source"]
            table_name = result["metadata"]["table_name"]
            source_obj = None
            for source in context.get("sources", []):
                if source.name == source_name:
                    source_obj = source
                    sql = source.get_sql_expr(source.normalize_table(table_name))
                    break
            table_slug = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"
            similarity_score = result["similarity"]
            # Filter by visible_slugs if specified
            visible_slugs = context.get("visible_slugs", set())
            if visible_slugs and table_slug not in visible_slugs:
                continue

            if any_matches and result["similarity"] < self.min_similarity and not same_table:
                continue

            columns = []
            table_description = None

            if table_metadata := context["tables_metadata"].get(table_slug):
                table_description = table_metadata.get("description")
                column_metadata = table_metadata.get("columns", {})

                for col_name, col_info in column_metadata.items():
                    col_desc = col_info.pop("description", "")
                    column_schema = Column(
                        name=col_name,
                        description=col_desc,
                        metadata=col_info.copy() if isinstance(col_info, dict) else {}
                    )
                    columns.append(column_schema)

            catalog_entry = TableCatalogEntry(
                table_slug=table_slug,
                similarity=similarity_score,
                columns=columns,
                source=source_obj,
                description=table_description,
                sql_expr=sql,
                metadata=context["tables_metadata"].get(table_slug, {}).copy()
            )
            catalog[table_slug] = catalog_entry

        # If query contains an exact table name, mark it as max similarity
        visible_slugs = context.get("visible_slugs", set())
        tables_to_check = context["tables_metadata"]
        if visible_slugs:
            tables_to_check = {slug: data for slug, data in tables_to_check.items() if slug in visible_slugs}

        for table_slug in tables_to_check:
            if table_slug.split(SOURCE_TABLE_SEPARATOR)[-1].lower() in query.lower():
                if table_slug in catalog:
                    catalog[table_slug].similarity = 1

        metaset = Metaset(catalog=catalog, query=query)
        return {"metaset": metaset}

    def _format_context(self, outputs: TableLookupOutputs) -> str:
        """Generate formatted text representation from schema objects."""
        # Get schema objects from memory
        metaset = outputs.get("metaset")
        return str(metaset)

    async def respond(
        self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]
    ) -> tuple[list[Any], TableLookupOutputs]:
        """
        Fetches tables based on the user query and returns formatted context.
        """
        # Run the process to build schema objects
        out_model = await self._gather_info(messages, context)
        return [self._format_context(out_model)], out_model


class IterativeTableLookupOutputs(ContextModel):

    metaset: Metaset


class IterativeTableLookup(TableLookup):
    """
    Extended version of TableLookup that performs an iterative table selection process.
    This tool uses an LLM to select tables in multiple passes, examining schemas in detail.
    """

    conditions = param.List(default=[
        "Useful for answering data queries",
        "Avoid for follow-up questions when existing data is sufficient",
        "Use only when existing information is insufficient for the current query",
    ])

    not_with = param.List(default=["TableLookup"])

    purpose = param.String(default="""
        Looks up the most relevant tables and provides SQL schemas of those tables.""")

    max_selection_iterations = param.Integer(default=3, doc="""
        Maximum number of iterations for the iterative table selection process.""")

    table_similarity_threshold = param.Number(default=0.5, doc="""
        If any tables have a similarity score above this threshold,
        those tables will be automatically selected and there will not be an
        iterative table selection process.""")

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
        1. From the top tables in TableLookup results, presents all tables and columns to the LLM
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
        visible_slugs = context.get('visible_slugs', set())
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
                steps_layout=self.steps_layout
            ) as step:
                available_slugs = [
                    table_slug for table_slug in all_slugs
                    if table_slug not in examined_slugs
                ]
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
                        selected_slugs = sorted(
                            catalog.keys(),
                            key=lambda x: catalog[x].similarity,
                            reverse=True
                        )[:5]
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
                        stream_details(
                            '\n\n'.join(f'`{slug}`' for slug in selected_slugs),
                            step, title=f"Selected {len(selected_slugs)} tables", auto=False
                        )

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
                        view_definition = truncate_string(
                            context["tables_metadata"].get(table_slug, {}).get("view_definition", ""),
                            max_length=300
                        )

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
        )
        return {"metaset": metaset}

    def _format_context(self, outputs: IterativeTableLookupOutputs) -> str:
        """Generate formatted text representation from schema objects."""
        metaset = outputs.get("metaset")
        return str(metaset)

    async def respond(
        self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]
    ) -> tuple[list[Any], IterativeTableLookupOutputs]:
        """
        Fetches tables based on the user query and returns formatted context.
        """
        out_model = await self._gather_info(messages, context)
        return [self._format_context(out_model)], out_model
