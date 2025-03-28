from __future__ import annotations

import asyncio
import traceback

from textwrap import indent
from types import FunctionType
from typing import Any

import param

from panel.viewable import Viewable

from ..views.base import View
from .actor import Actor, ContextProvider
from .config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from .embeddings import NumpyEmbeddings
from .llm import Message
from .models import make_iterative_selection_model, make_refined_query_model
from .translate import function_to_model
from .utils import (
    fetch_table_sql_info, format_info, log_debug, stream_details,
    truncate_string,
)
from .vector_store import NumpyVectorStore, VectorStore


class ToolUser(Actor):
    """
    ToolUser is a mixin class for actors that use tools.
    """

    tools = param.List(default=[], doc="""
        List of tools to use.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._tools = {}

        for prompt_name in self.prompts:
            self._tools[prompt_name] = []
            for tool in self._lookup_prompt_key(prompt_name, "tools"):
                if isinstance(tool, Actor):
                    if tool.llm is None:
                        tool.llm = self.llm
                    if tool.interface is None:
                        tool.interface = self.interface
                    self._tools[prompt_name].append(tool)
                elif isinstance(tool, FunctionType):
                    self._tools[prompt_name].append(FunctionTool(tool, llm=self.llm, interface=self.interface))
                else:
                    self._tools[prompt_name].append(tool(llm=self.llm, interface=self.interface))

    async def _use_tools(self, prompt_name: str, messages: list[Message]) -> str:
        tools_context = ""
        for tool in self._tools.get(prompt_name, []):
            if all(requirement in self._memory for requirement in tool.requires):
                with tool.param.update(memory=self.memory):
                    tool_context = await tool.respond(messages)
                    if tool_context:
                        tools_context += f"\n{tool_context}"
        return tools_context


class Tool(Actor, ContextProvider):
    """
    A Tool can be invoked by another Actor to provide additional
    context or respond to a question. Unlike an Agent they never
    interact with or on behalf of a user directly.
    """

    always_use = param.Boolean(default=False, doc="""
        Whether to always use this tool, even if it is not explicitly
        required by the current context.""")


class VectorLookupTool(Tool):
    """
    Baseclass for tools that search a vector database for relevant
    chunks.
    """
    enable_query_refinement = param.Boolean(default=True, doc="""
        Whether to enable query refinement for improving search results.""")

    min_similarity = param.Number(default=0.05, doc="""
        The minimum similarity to include a document.""")

    n = param.Integer(default=10, bounds=(1, None), doc="""
        The number of document results to return.""")

    prompts = param.Dict(
        default={
            "refine_query": {
                "template": PROMPTS_DIR / "VectorLookupTool" / "refine_query.jinja2",
                "response_model": make_refined_query_model,
            },
        },
        doc="Dictionary of available prompts for the tool."
    )

    refinement_similarity_threshold = param.Number(default=0.3, bounds=(0, 1), doc="""
        Similarity threshold below which query refinement is triggered.""")

    max_refinement_iterations = param.Integer(default=3, bounds=(1, 10), doc="""
        Maximum number of refinement iterations to perform.""")

    min_refinement_improvement = param.Number(default=0.05, bounds=(0, 1), doc="""
        Minimum improvement in similarity score required to keep refining.""")

    vector_store = param.ClassSelector(class_=VectorStore, constant=True, doc="""
        Vector store object which is queried to provide additional context
        before responding.""")

    _item_type_name: str = "items"

    __abstract = True

    def __init__(self, **params):
        if 'vector_store' not in params:
            params['vector_store'] = NumpyVectorStore(embeddings=NumpyEmbeddings())
        super().__init__(**params)

    def _format_results_for_refinement(self, results: list[dict[str, Any]]) -> str:
        """
        Format search results for inclusion in the refinement prompt.

        Subclasses can override this to provide more specific formatting.

        Parameters
        ----------
        results: list[dict[str, Any]]
            The search results from vector_store.query

        Returns
        -------
        str
            Formatted description of results
        """
        return "\n".join(
            f"- {result.get('text', 'No text')} (Similarity: {result.get('similarity', 0):.3f})"
            for i, result in enumerate(results)
        )

    async def _refine_query(
        self,
        original_query: str,
        results: list[dict[str, Any]]
    ) -> str:
        """
        Refines the search query based on initial search results.

        Parameters
        ----------
        original_query: str
            The original user query
        results: list[dict[str, Any]]
            The initial search results

        Returns
        -------
        str
            A refined search query
        """
        results_description = self._format_results_for_refinement(results)

        messages = [{"role": "user", "content": original_query}]

        try:
            refined_query_model = self._get_model("refine_query", item_type_name=self._item_type_name)
            system_prompt = await self._render_prompt(
                "refine_query",
                messages,
                results=results,
                results_description=results_description,
                original_query=original_query,
                item_type=self._item_type_name
            )

            model_spec = self.prompts["refine_query"].get("llm_spec", self.llm_spec_key)
            output = await self.llm.invoke(
                messages=messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=refined_query_model
            )

            return output.refined_search_query
        except Exception as e:
            with self._add_step(title="Query refinement error") as step:
                step.stream(f"Error refining query: {e}")
                step.status = "failed"
            return original_query

    async def _perform_search_with_refinement(self, query: str, **kwargs) -> list[dict[str, Any]]:
        """
        Performs a vector search with optional query refinement.

        Parameters
        ----------
        query: str
            The search query
        **kwargs:
            Additional arguments for vector_store.query

        Returns
        -------
        list[dict[str, Any]]
            The search results
        """
        final_query = query
        current_query = query
        iteration = 0

        with self._add_step(title="Vector Search with Refinement") as step:
            step.stream("Performing initial search\n\n")
            stream_details(query, step, title="Initial query", auto=False)
            results = self.vector_store.query(query, top_k=self.n, **kwargs)
            # check if all metadata is the same; if so, skip
            if all(result.get('metadata') == results[0].get('metadata') for result in results):
                return results

            best_similarity = max([result.get('similarity', 0) for result in results], default=0)
            best_results = results
            step.stream(f"Initial search found {len(results)} chunks with best similarity: {best_similarity:.3f}\n\n")
            stream_details("\n".join(
                f'{result["text"]} {result["metadata"]}' for result in results
            ), step, title=f"{len(results)} chunks", auto=False)

            refinement_history = []
            if not self.enable_query_refinement or best_similarity >= self.refinement_similarity_threshold or len(results) == 1:
                step.stream("Search complete - no refinement needed.")
                return best_results

            step.stream(f"Attempting to refine query (similarity {best_similarity:.3f} below threshold {self.refinement_similarity_threshold:.3f})\n\n")
            while iteration < self.max_refinement_iterations and best_similarity < self.refinement_similarity_threshold:
                iteration += 1
                step.stream(f"Processing refinement iteration {iteration}/{self.max_refinement_iterations}\n\n")

                refined_query = await self._refine_query(current_query, results)

                if refined_query == current_query:
                    step.stream("Refinement returned unchanged query, stopping iterations.")
                    break

                current_query = refined_query
                step.stream("Query refined.")
                stream_details(refined_query, step, title="Refined query", auto=False)

                new_results = self.vector_store.query(refined_query, top_k=self.n, **kwargs)
                new_best_similarity = max([result.get('similarity', 0) for result in new_results], default=0)

                improvement = new_best_similarity - best_similarity
                refinement_history.append({
                    "iteration": iteration,
                    "query": refined_query,
                    "similarity": new_best_similarity,
                    "improvement": improvement
                })

                step.stream(f"\n\nRefined search found {len(new_results)} results with best similarity: {new_best_similarity:.3f} (improvement: {improvement:.3f})\n")

                if new_best_similarity > best_similarity + self.min_refinement_improvement:
                    step.stream(f"Improved results (iteration {iteration}) with similarity {new_best_similarity:.3f}\n")
                    best_similarity = new_best_similarity
                    best_results = new_results
                    final_query = refined_query
                    results = new_results  # Update results for next iteration's refinement
                else:
                    step.stream(f"Insufficient improvement ({improvement:.3f} < {self.min_refinement_improvement:.3f}), stopping iterations\n")
                    break

                # Break if we've reached an acceptable similarity
                if best_similarity >= self.refinement_similarity_threshold:
                    step.stream(f"Reached acceptable similarity threshold: {best_similarity:.3f}\n")
                    break

            if refinement_history:
                step.stream(f"Final query after {iteration} iterations: '{final_query}' with similarity {best_similarity:.3f}\n")

        return best_results


class DocumentLookup(VectorLookupTool):
    """
    The DocumentLookup tool creates a vector store of all available documents
    and responds with a list of the most relevant documents given the user query.
    Always use this for more context.
    """

    purpose = param.String(default="""
        Looks up relevant documents based on the user query.""")

    requires = param.List(default=["document_sources"], readonly=True, doc="""
        List of context that this Tool requires to be run.""")

    # Override the item type name
    _item_type_name = "documents"

    def __init__(self, **params):
        super().__init__(**params)
        self._memory.on_change('document_sources', self._update_vector_store)

    def _format_results_for_refinement(self, results: list[dict[str, Any]]) -> str:
        """
        Format document search results for inclusion in the refinement prompt.

        Parameters
        ----------
        results: list[dict[str, Any]]
            The search results from vector_store.query

        Returns
        -------
        str
            Formatted description of document results
        """
        formatted_results = []
        for i, result in enumerate(results):
            metadata = result.get('metadata', {})
            filename = metadata.get('filename', 'Unknown document')
            text_preview = truncate_string(result.get('text', ''), max_length=150)

            description = f"- {filename} (Similarity: {result.get('similarity', 0):.3f})"
            if text_preview:
                description += f"\n  Preview: {text_preview}"

            formatted_results.append(description)

        return "\n".join(formatted_results)

    async def _update_vector_store(self, _, __, sources):
        for source in sources:
            metadata = source.get("metadata", {})
            filename = metadata.get("filename")
            if filename:
                # overwrite existing items with the same filename
                existing_items = self.vector_store.filter_by({'filename': filename})
                if existing_ids := [item['id'] for item in existing_items]:
                    self.vector_store.delete(existing_ids)
            self.vector_store.add([{"text": source["text"], "metadata": source.get("metadata", {})}])

    async def respond(self, messages: list[Message], **kwargs: Any) -> str:
        query = messages[-1]["content"]

        # Perform search with refinement
        results = await self._perform_search_with_refinement(query)
        closest_doc_chunks = [
            f"{result['text']} (Relevance: {result['similarity']:.1f} - "
            f"Metadata: {result['metadata']})"
            for result in results
            if result['similarity'] >= self.min_similarity
        ]

        if not closest_doc_chunks:
            return ""

        message = "Please augment your response with the following context if relevant:\n"
        message += "\n".join(f"- {doc}" for doc in closest_doc_chunks)
        return message


class TableLookup(VectorLookupTool):
    """
    TableLookup tool that creates a vector store of all available tables
    and responds with relevant tables for user queries.
    """

    purpose = param.String(default="""
        Looks up relevant tables based on the user query with a vector store.
        Useful for quickly gathering information about tables and their columns
        to chat about.""")

    requires = param.List(default=["sources"], readonly=True, doc="""
        List of context that this Tool requires to be run.""")

    provides = param.List(default=["tables_vector_info"], readonly=True, doc="""
        List of context values this Tool provides to current working memory.""")

    include_metadata = param.Boolean(default=True, doc="""
        Whether to include table descriptions in the embeddings and responses.""")

    include_columns = param.Boolean(default=True, doc="""
        Whether to include column names and descriptions in the embeddings.""")

    include_misc = param.Boolean(default=False, doc="""
        Whether to include miscellaneous metadata in the embeddings,
        besides table and column descriptions.""")

    max_concurrent = param.Integer(default=1, doc="""
        Maximum number of concurrent metadata fetch operations.""")

    _item_type_name = "database tables"

    _ready = param.Boolean(default=False, doc="""
        Whether the vector store is ready.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._tables_metadata = {}
        self._raw_metadata = {}
        self._metadata_tasks = set()  # Track ongoing metadata fetch tasks
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._memory.on_change('sources', self._update_vector_store)
        if "sources" in self._memory:
            self._memory.trigger('sources')

    def _format_results_for_refinement(self, results: list[dict[str, Any]]) -> str:
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
        formatted_results = []
        for i, result in enumerate(results):
            source_name = result['metadata'].get("source", "unknown")
            table_name = result['metadata'].get("table_name", "unknown")
            text = result["text"]
            table_slug = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"

            description = f"- {text} {table_slug} (Similarity: {result.get('similarity', 0):.3f})"
            if tables_vector_info := self._tables_metadata.get(table_slug):
                if table_description := tables_vector_info.get("description"):
                    description += f"\n  Description: {table_description}"
            formatted_results.append(description)
        return "\n".join(formatted_results)

    async def _fetch_and_store_metadata(self, source, table_name: str):
        """Fetch metadata for a table and store it in the vector store."""
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
            return
        table_name = table_name_key
        table_vector_info = dict(source_metadata[table_name])

        # IMPORTANT: re-insert using table slug
        # we need to store a copy of tables_vector_info so it can be used to inject context into the LLM
        table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table_name}"
        self._tables_metadata[table_slug] = table_vector_info.copy()
        self._tables_metadata[table_slug]["source_name"] = source.name  # need this to rebuild the slug

        vector_metadata = {"source": source.name, "table_name": table_name}
        if existing_items := self.vector_store.filter_by(vector_metadata):
            if existing_items and existing_items[0].get("metadata", {}).get("enriched"):
                return
            self.vector_store.delete([item['id'] for item in existing_items])

        # the following is for the embeddings/vector store use only (i.e. filter from 1000s of tables)
        vector_metadata["enriched"] = True
        enriched_text = f"Table: {table_name}"
        if description := table_vector_info.pop('description', ''):
            enriched_text += f"\nDescription: {description}"
        if self.include_columns and (columns := table_vector_info.pop('columns', {})):
            enriched_text += "\nCols:"
            for col_name, col_info in columns.items():
                col_text = f"\n- {col_name}"
                if col_info.get("description"):
                    col_text += f": {col_info['description']}"
                enriched_text += col_text
            enriched_text += "\n"
        if self.include_misc:
            enriched_text += "\nMiscellaneous:"
            for key, value in table_vector_info.items():
                if key == "enriched":
                    continue
                enriched_text += f"\n- {key}: {value}"
        self.vector_store.add([{"text": enriched_text, "metadata": vector_metadata}])

    async def _update_vector_store(self, _, __, sources):
        # Create a list to track all tasks we're starting in this update
        all_tasks = []

        for source in sources:
            if self.include_metadata and self._raw_metadata.get(source.name) is None:
                metadata_task = asyncio.create_task(
                    asyncio.to_thread(source.get_metadata)
                )
                self._raw_metadata[source.name] = metadata_task
                all_tasks.append(metadata_task)

            tables = source.get_tables()
            # since enriching the metadata might take time, first add basic metadata (table name)
            for table_name in tables:
                metadata = {"source": source.name, "table_name": table_name}
                if self.vector_store.filter_by(metadata):
                    # TODO: Add ability to update existing source table?
                    continue
                self.vector_store.add([{"text": table_name, "metadata": metadata}])

            # if metadata is included, upsert the existing
            if self.include_metadata:
                for table in tables:
                    task = asyncio.create_task(
                        self._fetch_and_store_metadata(source, table)
                    )
                    self._metadata_tasks.add(task)
                    task.add_done_callback(lambda t: self._metadata_tasks.discard(t))
                    all_tasks.append(task)

        if all_tasks:
            ready_task = asyncio.create_task(self._mark_ready_when_done(all_tasks))
            ready_task.add_done_callback(lambda t: None if t.exception() else None)

    async def _mark_ready_when_done(self, tasks):
        """Wait for all tasks to complete and then mark the tool as ready."""
        await asyncio.gather(*tasks, return_exceptions=True)
        log_debug("All table metadata tasks completed.")
        self._ready = True

    async def _gather_info(self, messages: list[dict[str, str]]) -> dict:
        """Gather relevant information about the tables based on the user query."""
        query = messages[-1]["content"]
        results = await self._perform_search_with_refinement(query)

        tables_vector_info = {}
        any_matches = any(result['similarity'] >= self.min_similarity for result in results)
        for result in results:
            if any_matches and result['similarity'] < self.min_similarity and len(results) > 1:
                continue
            source_name = result['metadata']["source"]
            table_name = result['metadata']["table_name"]
            table_slug = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"
            similarity_score = result['similarity']

            if table_slug not in tables_vector_info:
                tables_vector_info[table_slug] = {}
            tables_vector_info[table_slug]["similarity"] = similarity_score
            table_caption = ""
            if table_metadata := self._tables_metadata.get(table_slug):
                if table_description := table_metadata.get("description"):
                    table_caption += f"  Description: {table_description}"
                    tables_vector_info[table_slug]["description"] = table_description

                columns = table_metadata.get("columns", {})
                columns_description = "Cols:" if columns else ""
                tables_vector_info[table_slug]["columns_description"] = {}
                already_truncated = False
                for i, (col_name, col_info) in enumerate(columns.items()):
                    col_desc = f": {col_info['description']}" if col_info.get("description") else ""
                    tables_vector_info[table_slug]["columns_description"][col_name] = col_desc
                    if already_truncated:
                        # Skip if already truncated
                        continue
                    if i <= 15:
                        # Save on tokens by truncating long column names and letting the LLM infer the rest
                        col_label = truncate_string(col_name, max_length=20)
                        columns_description += f"\n- {col_label}{col_desc}"
                    else:
                        columns_description += f"\n  ... (out of {len(table_metadata['columns'])} columns)"
                        already_truncated = True

                table_caption += f"\n{indent(columns_description, ' ' * 2)}"
                tables_vector_info[table_slug]["caption"] = table_caption

        # If query contains an exact table name, mark it as max similarity
        for table_slug in self._tables_metadata:
            if table_slug.split(SOURCE_TABLE_SEPARATOR)[-1].lower() in query.lower():
                tables_vector_info[table_slug]["similarity"] = 1

        self._memory["tables_vector_info"] = tables_vector_info
        return tables_vector_info

    def _format_context(self) -> str:
        tables_vector_info = self._memory.get("tables_vector_info", {})
        context = "Below are the relevant tables and their descriptions."
        for table_slug, table_info in tables_vector_info.items():
            context += f"\n{table_slug} (Similarity: {table_info['similarity']:.3f})"
            if caption := table_info.get("caption", "").strip():
                context += f"\n```\n{caption}\n```\n"
        return context

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        """
        Fetches the closest tables based on the user query, with query refinement
        if enabled and initial results don't meet threshold.
        """
        await self._gather_info(messages)
        context = self._format_context()
        return context


class IterativeTableLookup(TableLookup):
    """
    Extended version of TableLookup that performs an iterative table selection process.
    This tool uses an LLM to select tables in multiple passes, examining schemas in detail.
    """

    purpose = param.String(default="""
        Looks up relevant tables based on the user query with a vector store and iterative selection process.
        Uses LLM to select tables in multiple passes, examining schemas in detail.
        Useful for SQLAgent, but not for ChatAgent.""")

    provides = param.List(default=["tables_vector_info", "tables_sql_info"], readonly=True, doc="""
        List of context values this Tool provides to current working memory.""")

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

    async def _gather_info(self, messages: list[dict[str, str]]) -> dict:
        """
        Performs an iterative table selection process to gather context.
        This function:
        1. From the top tables in TableLookup results, presents all tables and columns to the LLM
        2. Has the LLM select ~3 tables for deeper examination
        3. Gets complete schemas for these tables
        4. Repeats until the LLM is satisfied with the context
        """
        tables_vector_info = await super()._gather_info(messages)

        tables_sql_info = self._memory.get("tables_sql_info", {})
        examined_slugs = set(tables_sql_info.keys())
        all_slugs = list(tables_vector_info.keys())
        failed_slugs = []
        satisfied_slugs = []
        selected_slugs = []
        chain_of_thought = ""
        max_iterations = self.max_selection_iterations
        sources = {source.name: source for source in self._memory["sources"]}
        for iteration in range(1, max_iterations + 1):
            with self._add_step(title=f"Iterative table selection {iteration} / {max_iterations}", success_title=f"Selection iteration {iteration} / {max_iterations} completed") as step:
                available_slugs = [
                    table_slug for table_slug in all_slugs
                    if table_slug not in examined_slugs
                ]
                log_debug(available_slugs)
                if not available_slugs:
                    step.stream("No more tables available to examine.")
                    break

                fast_track = False
                if iteration == 1:
                    # For the first iteration, select tables based on similarity
                    # If any tables have a similarity score above the threshold, select up to 5 of those tables
                    if any(meta.get('similarity', 0) > self.table_similarity_threshold for meta in tables_vector_info.values()) or len(all_slugs) == 1:
                        selected_slugs = sorted(
                            tables_vector_info,
                            key=lambda x: tables_vector_info[x].get('similarity', 0),
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
                            chain_of_thought=chain_of_thought,
                            current_query=messages[-1]["content"],
                            available_slugs=available_slugs,
                            examined_slugs=examined_slugs,
                            selected_slugs=selected_slugs,
                            failed_slugs=failed_slugs,
                            separator=SOURCE_TABLE_SEPARATOR,
                            tables_sql_info=tables_sql_info,
                            tables_vector_info=tables_vector_info,
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
                        stream_details('\n'.join(selected_slugs), step, title=f"Selected {len(selected_slugs)} tables", auto=False)

                        # Check if we're done with table selection
                        if output.is_done or not available_slugs:
                            step.stream("Selection process complete - model is satisfied with selected tables")
                            satisfied_slugs = selected_slugs
                            break
                        if iteration != max_iterations:
                            step.stream("Unsatisfied with selected tables - continuing selection process...")
                        else:
                            step.stream("Maximum iterations reached - stopping selection process")
                            satisfied_slugs = selected_slugs or list(examined_slugs)
                            break
                    except Exception as e:
                        traceback.print_exc()
                        stream_details(f"Error selecting tables: {e}", step, title="Error", auto=False)
                        continue

                step.stream("\n\nFetching detailed schema information for tables\n")
                for table_slug in selected_slugs:
                    table_name = table_slug.split(SOURCE_TABLE_SEPARATOR)[-1]
                    vector_info = tables_vector_info[table_slug]
                    similarity = vector_info["similarity"]
                    caption = vector_info["caption"]
                    step.stream(f"`{table_name}` (Similarity: {similarity:.2f})")
                    stream_details(caption, step, title="Caption", auto=False)
                    try:
                        view_definition = truncate_string(
                            self._tables_metadata.get(table_slug, {}).get("view_definition", ""), max_length=300
                        )
                        table_sql_info = await fetch_table_sql_info(
                            sources,
                            table_slug,
                            view_definition=view_definition,
                            include_count=True
                        )
                        tables_sql_info[table_slug] = table_sql_info
                        examined_slugs.add(table_slug)
                        stream_details(f"```yaml\n{table_sql_info['schema']}\n```", step, title="Schema")
                    except Exception as e:
                        failed_slugs.append(table_slug)
                        stream_details(f"Error fetching schema: {e}", step, title="Error", auto=False)

                if fast_track:
                    step.stream("Fast-tracked selection process based on similarity threshold.")
                    satisfied_slugs = selected_slugs
                    # based on similarity search alone, if we have selected tables, we're done!!
                    break

        tables_sql_info = {
            table_slug: schema_data for table_slug, schema_data in tables_sql_info.items()
            # Only include tables that were selected in the final iteration or were fast-tracked
            if len(satisfied_slugs) == 0 or table_slug in satisfied_slugs
        }
        self._memory["tables_sql_info"] = tables_sql_info
        return tables_sql_info

    def _format_context(self) -> str:
        """Combines tables vector and sql info into a single context string."""
        tables_vector_info = self._memory.get("tables_vector_info", {})
        tables_sql_info = self._memory.get("tables_sql_info", {})
        return format_info(tables_vector_info, tables_sql_info)


class FunctionTool(Tool):

    """
    FunctionTool wraps arbitrary functions and makes them available as a tool
    for an LLM to call. It inspects the arguments of the function and generates
    a pydantic Model that the LLM will populate.

    The function may also consume information in memory, e.g. the current
    table or pipeline by declaring the requires parameter.

    The function may also return context to add to the current working memory
    by returning a dictionary and declaring the `provides` parameter. Any
    keys listed in the provides parameter will be copied into working memory.
    """

    formatter = param.Parameter(default="{function}({arguments}) returned: {output}", doc="""
        Formats the return value for inclusion in the global context.
        Accepts the 'function', 'arguments' and 'output' as formatting variables.""")

    function = param.Callable(default=None, allow_refs=False, doc="""
        The function to call.""")

    provides = param.List(default=[], readonly=False, constant=True, doc="""
        List of context values it provides to current working memory.""")

    requires = param.List(default=[], readonly=False, constant=True, doc="""
        List of context values it requires to be in memory.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "FunctionTool" / "main.jinja2"
            },
        }
    )

    def __init__(self, function, **params):
        model = function_to_model(function, skipped=self.requires)
        super().__init__(
            function=function,
            name=function.__name__,
            purpose=params.pop("purpose", model.__doc__) or "",
            **params
        )
        self._model = model

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        prompt = await self._render_prompt("main", messages)
        kwargs = {}
        if any(field not in self.requires for field in self._model.model_fields):
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            kwargs = await self.llm.invoke(
                messages,
                system=prompt,
                model_spec=model_spec,
                response_model=self._model,
                allow_partial=False,
                max_retries=3,
            )
        arguments = dict(kwargs, **{k: self._memory[k] for k in self.requires})
        if param.parameterized.iscoroutinefunction(self.function):
            result = await self.function(**arguments)
        else:
            result = self.function(**arguments)
        if isinstance(result, (View, Viewable)):
            return result
        if self.provides:
            if len(self.provides) == 1 and not isinstance(result, dict):
                self._memory[self.provides[0]] = result
            else:
                self._memory.update({result[key] for key in self.provides})
        return self.formatter.format(
            function=self.function.__name__,
            arguments=', '.join(f'{k}={v!r}' for k, v in arguments.items()),
            output=result
        )
