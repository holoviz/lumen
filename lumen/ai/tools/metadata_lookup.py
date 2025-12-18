import asyncio
import traceback

from typing import Annotated, Any

import param

from ...sources.base import Source
from ...sources.duckdb import DuckDBSource
from ..config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from ..context import ContextModel, TContext
from ..llm import Message
from ..schemas import (
    Column, DocumentChunk, Metaset, TableCatalogEntry,
)
from ..utils import log_debug
from .vector_lookup import VectorLookupTool, make_refined_query_model


class MetadataLookupInputs(ContextModel):
    sources: Annotated[list[Source], ("accumulate", "source")]


class MetadataLookupOutputs(ContextModel):
    metaset: Metaset


class MetadataLookup(VectorLookupTool):
    """
    MetadataLookup tool that creates a vector store of all available tables
    and responds with relevant tables for user queries.
    """

    conditions = param.List(
        default=[
            "Best paired with ChatAgent for general conversation about data",
            "Avoid if table discovery already performed for same request",
            "Not useful for data related queries",
        ]
    )

    prompts = param.Dict(
        default={
            "refine_query": {
                "template": PROMPTS_DIR / "VectorLookupTool" / "refine_query.jinja2",
                "response_model": make_refined_query_model,
            },
        },
        doc="Dictionary of available prompts for the tool.",
    )

    exclusions = param.List(default=["dbtsl_metaset"])

    not_with = param.List(default=["IterativeTableLookup"])

    purpose = param.String(
        default="""
        Discovers relevant tables using vector search, providing context for other agents.
        Not to be used for finding tables for further analysis (e.g. SQL), because it does
        not provide a schema."""
    )

    include_metadata = param.Boolean(
        default=True,
        doc="""
        Whether to include table descriptions in the embeddings and responses.""",
    )

    include_columns = param.Boolean(
        default=True,
        doc="""
        Whether to include column names and descriptions in the embeddings.""",
    )

    include_misc = param.Boolean(
        default=False,
        doc="""
        Whether to include miscellaneous metadata in the embeddings,
        besides table and column descriptions.""",
    )

    max_concurrent = param.Integer(
        default=1,
        doc="""
        Maximum number of concurrent metadata fetch operations.""",
    )

    min_similarity = param.Number(
        default=0,
        doc="""
        The minimum similarity to include a document.""",
    )

    n = param.Integer(
        default=10,
        bounds=(1, None),
        doc="""
        The number of document results to return.""",
    )

    n_documents = param.Integer(
        default=2,
        bounds=(1, None),
        doc="""
        The number of document chunks to retrieve per table.""",
    )

    _ready = param.Boolean(
        default=False,
        allow_None=True,
        doc="""
        Whether the vector store is ready.""",
    )

    _item_type_name = "tables"

    # Class variable to track which sources are currently being processed
    _sources_in_progress = {}
    _progress_lock = asyncio.Lock()  # Lock for thread-safe access

    input_schema = MetadataLookupInputs
    output_schema = MetadataLookupOutputs

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
            return

        if "tables_metadata" not in context:
            context["tables_metadata"] = {}

        await self._update_vector_store(context)

    def _format_results_for_refinement(self, results: list[dict[str, Any]], context: TContext) -> str:
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
        visible_slugs = context.get("visible_slugs")
        formatted_results = []
        for result in results:
            source_name = result["metadata"].get("source", "unknown")
            table_name = result["metadata"].get("table_name", "unknown")
            table_slug = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"
            if visible_slugs is not None and table_slug not in visible_slugs:
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

        table_name_key = next((key for key in (table_name.upper(), table_name.lower(), table_name) if key in source_metadata), None)

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
        if self.include_columns and (vector_columns := vector_info.get("columns", {})):
            for col_name, col_info in vector_columns.items():
                col_desc = col_info.pop("description", "")
                column = Column(name=col_name, description=col_desc, metadata=col_info.copy() if isinstance(col_info, dict) else {})
                columns.append(column)

        vector_metadata = {"source": source.name, "table_name": table_name, "type": self._item_type_name}
        # the following is for the embeddings/vector store use only (i.e. filter from 1000s of tables)
        enriched_text = f"Table: {table_name}"
        if description := vector_info.pop("description", ""):
            enriched_text += f"\nInfo: {description}"
        if self.include_columns and (vector_columns := vector_info.pop("columns", {})):
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

        log_debug(f"[MetadataLookup] Starting _update_vector_store for {len(sources)} sources")

        for source in sources:
            async with self._progress_lock:
                # Skip this source if it's already being processed for this vector store
                if source.name in self._sources_in_progress[vector_store_id]:
                    log_debug(f"[MetadataLookup] Skipping source {source.name} - already in progress")
                    continue

                # Mark this source as in progress for this vector store
                self._sources_in_progress[vector_store_id].add(source.name)
                processed_sources.append(source.name)
                log_debug(f"[MetadataLookup] Processing source {source.name}")

            if self.include_metadata and self._raw_metadata.get(source.name) is None:
                if isinstance(source, DuckDBSource):
                    self._raw_metadata[source.name] = source.get_metadata()
                else:
                    metadata_task = asyncio.create_task(asyncio.to_thread(source.get_metadata))
                    self._raw_metadata[source.name] = metadata_task
                    tasks.append(metadata_task)

            tables = source.get_tables()

            if self.include_metadata:
                for table in tables:
                    task = asyncio.create_task(self._enrich_metadata(source, table, context))
                    tasks.append(task)
            else:
                await self.vector_store.upsert(
                    [{"text": table_name, "metadata": {"source": source.name, "table_name": table_name, "type": "table"}} for table_name in tables]
                )

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
                    log_debug(f"[MetadataLookup] Cleaning up sources: {processed_sources}")
                    for source_name in processed_sources:
                        if vector_store_id in self._sources_in_progress:
                            self._sources_in_progress[vector_store_id].discard(source_name)
                            log_debug(f"[MetadataLookup] Removed {source_name} from in-progress")

                    # Remove the vector_store_id entry if empty
                    if vector_store_id in self._sources_in_progress and not self._sources_in_progress[vector_store_id]:
                        del self._sources_in_progress[vector_store_id]
                        log_debug(f"[MetadataLookup] Removed empty vector_store_id {vector_store_id}")
                except Exception as e:
                    log_debug(f"[MetadataLookup] Error cleaning up sources_in_progress: {e!s}")
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
            log_debug(f"[MetadataLookup] Waiting for {len(tasks)} tasks to complete")
            # Add timeout to prevent indefinite waiting (5 minutes)
            try:

                async def _gather_with_semaphore():
                    async with asyncio.Semaphore(self.max_concurrent):
                        return [result for result in await asyncio.gather(*tasks, return_exceptions=True) if isinstance(result, dict) and "text" in result]

                enriched_entries = await asyncio.wait_for(_gather_with_semaphore(), timeout=300)
            except TimeoutError:
                log_debug("[MetadataLookup] Timeout waiting for tasks to complete (5 minutes)")
                self._ready = None  # Set to error state
                return

            if enriched_entries:
                log_debug(f"[MetadataLookup] Upserting {len(enriched_entries)} enriched entries")
                # Add timeout to the upsert operation to prevent deadlock
                try:
                    await asyncio.wait_for(self.vector_store.upsert(enriched_entries), timeout=60)
                    log_debug(f"[MetadataLookup] Successfully upserted {len(enriched_entries)} entries")
                except TimeoutError:
                    log_debug("[MetadataLookup] Timeout during upsert operation (60 seconds)")
                    self._ready = None  # Set to error state
                    return
            else:
                log_debug("[MetadataLookup] No enriched entries to upsert")
            log_debug("[MetadataLookup] All table metadata tasks completed.")
            self._ready = True
        except Exception as e:
            log_debug(f"[MetadataLookup] Error in _mark_ready_when_done: {e!s}")
            traceback.print_exc()
            self._ready = None  # Set to error state
            raise
        finally:
            # Clean up sources_in_progress after tasks are done, even if there was an error
            if vector_store_id is not None and processed_sources:
                self._cleanup_sources_in_progress(vector_store_id, processed_sources)

    async def _query_documents(
        self,
        query: str,
        context: TContext,
        relevant_tables: set[str] | None = None
    ) -> list[DocumentChunk]:
        """
        Query the vector store for relevant document chunks.

        Includes docs that are:
        - Associated with at least one relevant table
        - In the visible_docs set (if specified)

        Parameters
        ----------
        query : str
            The search query
        context : TContext
            The context containing sources
        relevant_tables : set[str] | None
            Set of relevant table slugs. If None, includes all table-associated docs.
        """
        doc_store = self.document_vector_store or self.vector_store

        # Get all document filenames from vector store
        all_doc_results = await doc_store.query(text="", top_k=1000, filters={"type": "document"})
        all_filenames = {r["metadata"]["filename"] for r in all_doc_results}

        # Get visible docs (if specified, only include these; otherwise include all)
        visible_docs = context.get("visible_docs")

        # Filter to only visible docs if specified
        if visible_docs is not None:
            all_filenames = all_filenames & visible_docs

        # Collect table-associated docs
        table_associated_docs = {}  # filename -> set of table slugs
        sources = context.get("sources", [])
        for source in sources:
            if source.metadata:
                for table in source.get_tables():
                    table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}"
                    if table in source.metadata:
                        for doc_filename in source.metadata[table].get("docs", []):
                            if doc_filename not in table_associated_docs:
                                table_associated_docs[doc_filename] = set()
                            table_associated_docs[doc_filename].add(table_slug)

        # Filter filenames - include only docs associated with relevant tables
        included_filenames = set()
        for filename in all_filenames:
            if filename in table_associated_docs:
                # Include if table-associated and at least one table is relevant
                if relevant_tables is None or any(tbl in relevant_tables for tbl in table_associated_docs[filename]):
                    included_filenames.add(filename)
            # If NOT associated with any table, it might be newly uploaded
            # Include it only if visible_docs is not set (backwards compat)
            # Once auto-association runs, it will be properly associated
            elif visible_docs is None:
                included_filenames.add(filename)

        log_debug(f"[_query_documents] query={query!r}, all={len(all_filenames)}, visible={len(visible_docs) if visible_docs else 'all'}, included={len(included_filenames)}")

        if not included_filenames:
            return []

        # Query each included document file
        all_docs: list[DocumentChunk] = []
        for filename in included_filenames:
            results = await doc_store.query(
                text=query, top_k=self.n_documents, filters={"type": "document", "filename": filename}
            )
            for r in results:
                if r["similarity"] < self.min_similarity:
                    continue
                all_docs.append(DocumentChunk(
                    filename=filename, text=r["text"], similarity=r["similarity"], metadata=r.get("metadata", {})
                ))

        # Sort by similarity and limit
        all_docs.sort(key=lambda c: c.similarity, reverse=True)
        return all_docs[:self.n_documents * 2]  # Return top chunks across all files

    async def _gather_info(self, messages: list[dict[str, str]], context: TContext) -> MetadataLookupOutputs:
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
            visible_slugs = context.get("visible_slugs")
            if visible_slugs is not None and table_slug not in visible_slugs:
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
                    column_schema = Column(name=col_name, description=col_desc, metadata=col_info.copy() if isinstance(col_info, dict) else {})
                    columns.append(column_schema)

            catalog_entry = TableCatalogEntry(
                table_slug=table_slug,
                similarity=similarity_score,
                columns=columns,
                source=source_obj,
                description=table_description,
                sql_expr=sql,
                metadata=context["tables_metadata"].get(table_slug, {}).copy(),
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

        # Query for relevant document chunks, passing relevant tables for filtering
        relevant_tables = set(catalog.keys())
        docs = await self._query_documents(query, context, relevant_tables)

        metaset = Metaset(
            catalog=catalog,
            query=query,
            docs=docs if docs else None,
        )
        return {"metaset": metaset}

    def _format_context(self, outputs: MetadataLookupOutputs) -> str:
        """Generate formatted text representation from schema objects."""
        # Get schema objects from memory
        metaset = outputs.get("metaset")
        return str(metaset)

    async def respond(self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]) -> tuple[list[Any], MetadataLookupOutputs]:
        """
        Fetches tables based on the user query and returns formatted context.
        """
        # Run the process to build schema objects
        out_model = await self._gather_info(messages, context)
        return [self._format_context(out_model)], out_model
