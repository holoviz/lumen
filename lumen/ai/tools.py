from __future__ import annotations

import asyncio

from textwrap import indent
from typing import Any

import param

from graphql import Source
from panel.chat import ChatInterface
from panel.viewable import Viewable

from ..views.base import View
from .actor import Actor, ContextProvider
from .config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from .embeddings import NumpyEmbeddings
from .llm import Message
from .models import make_coordinator_tables_model, make_refined_query_model
from .translate import function_to_model
from .utils import (
    fetch_table_schema, fetch_table_schemas, fuse_messages, log_debug,
    stream_details,
)
from .vector_store import NumpyVectorStore, VectorStore


class Tool(Actor, ContextProvider):
    """
    A Tool can be invoked by another Actor to provide additional
    context or respond to a question. Unlike an Agent they never
    interact with or on behalf of a user directly.
    """

    always_use = param.Boolean(default=False, doc="""
        Whether to always use this tool, even if it is not explicitly
        required by the current context.""")

    interface = param.ClassSelector(class_=ChatInterface, allow_None=True, doc="""
        The interface to report progress to.""")


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
            "table_selection": {
                "template": PROMPTS_DIR / "TableLookup" / "table_selection.jinja2",
                "response_model": make_coordinator_tables_model,
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
            with self.interface.add_step(title="Query refinement error") as step:
                step.stream(f"Error refining query: {e}")
                step.status = "failed"
            return original_query

    async def _perform_search_with_refinement(self, query: str, **kwargs) -> tuple[str, list[dict[str, Any]]]:
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
        tuple[str, list[dict[str, Any]], float]
            - The final query used (original or refined)
            - The search results
        """
        final_query = query
        current_query = query
        iteration = 0

        with self.interface.add_step(title="Initial search") as step:
            step.stream(f"Performing initial search with query: '{query}'")
            results = self.vector_store.query(query, top_k=self.n, **kwargs)
            best_similarity = max([result.get('similarity', 0) for result in results], default=0)
            best_results = results
            step.stream(f"Initial search found {len(results)} results with best similarity: {best_similarity:.3f}")

        refinement_history = []
        if not self.enable_query_refinement or best_similarity >= self.refinement_similarity_threshold:
            return best_results

        with self.interface.add_step(title="Refining query") as step:
            step.stream(f"Attempting to refine query (similarity {best_similarity:.3f} below threshold {self.refinement_similarity_threshold:.3f})")

        while iteration < self.max_refinement_iterations and best_similarity < self.refinement_similarity_threshold:
            iteration += 1

            with self.interface.add_step(title=f"Refinement iteration {iteration}/{self.max_refinement_iterations}") as step:
                step.stream(f"Processing refinement iteration {iteration}")

                refined_query = await self._refine_query(current_query, results)

                if refined_query == current_query:
                    step.stream("Refinement returned unchanged query, stopping iterations")
                    break

                current_query = refined_query
                step.stream(f"Query refined to: '{refined_query}'")

                new_results = self.vector_store.query(refined_query, top_k=self.n, **kwargs)
                new_best_similarity = max([result.get('similarity', 0) for result in new_results], default=0)

                improvement = new_best_similarity - best_similarity
                refinement_history.append({
                    "iteration": iteration,
                    "query": refined_query,
                    "similarity": new_best_similarity,
                    "improvement": improvement
                })

                step.stream(f"Refined search found {len(new_results)} results with best similarity: {new_best_similarity:.3f} (improvement: {improvement:.3f})")

                if new_best_similarity > best_similarity + self.min_refinement_improvement:
                    step.stream(f"Improved results (iteration {iteration}) with similarity {new_best_similarity:.3f}")
                    best_similarity = new_best_similarity
                    best_results = new_results
                    final_query = refined_query
                    results = new_results  # Update results for next iteration's refinement
                else:
                    step.stream(f"Insufficient improvement ({improvement:.3f} < {self.min_refinement_improvement:.3f}), stopping iterations")
                    break

                # Break if we've reached an acceptable similarity
                if best_similarity >= self.refinement_similarity_threshold:
                    step.stream(f"Reached acceptable similarity threshold: {best_similarity:.3f}")
                    break

        if refinement_history:
            with self.interface.add_step(title="Final query") as step:
                step.stream(f"Final query after {iteration} iterations: '{final_query}' with similarity {best_similarity:.3f}")

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
            text_preview = result.get('text', '')[:150] + '...' if len(result.get('text', '')) > 150 else result.get('text', '')

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
        Looks up relevant tables based on the user query. Unnecessary if
        the table doesn't need to change or the user is requesting a visualization
        from the previous results.""")

    requires = param.List(default=["sources"], readonly=True, doc="""
        List of context that this Tool requires to be run.""")

    provides = param.List(default=["tables_info"], readonly=True, doc="""
        List of context values this Tool provides to current working memory.""")

    iteratively_select = param.Boolean(default=True, doc="""
        Whether to enable iterative table selection process that uses LLM to
        select tables in multiple passes, examining schemas in detail.""")

    include_metadata = param.Boolean(default=True, doc="""
        Whether to include table descriptions in the embeddings and responses.""")

    include_columns = param.Boolean(default=True, doc="""
        Whether to include column names and descriptions in the embeddings.""")

    include_misc = param.Boolean(default=False, doc="""
        Whether to include miscellaneous metadata in the embeddings,
        besides table and column descriptions.""")

    max_concurrent = param.Integer(default=1, doc="""
        Maximum number of concurrent metadata fetch operations.""")

    max_selection_iterations = param.Integer(default=3, doc="""
        Maximum number of iterations for the iterative table selection process.""")

    table_similarity_threshold = param.Number(default=0.5, doc="""
        If any tables have a similarity score above this threshold,
        those tables will be automatically selected and there will not be an
        iterative table selection process.""")

    _item_type_name = "database tables"

    _ready = param.Boolean(default=False, doc="""
        Whether the vector store is ready.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._table_metadata = {}
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
            table_slug = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"

            description = f"- {table_slug} (Similarity: {result.get('similarity', 0):.3f})"
            if table_metadata := self._table_metadata.get(table_slug):
                if table_description := table_metadata.get("description"):
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
        table_metadata = dict(source_metadata[table_name])

        # IMPORTANT: re-insert using table slug
        # we need to store a copy of table_metadata so it can be used to inject context into the LLM
        table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table_name}"
        self._table_metadata[table_slug] = table_metadata.copy()
        self._table_metadata[table_slug]["source_name"] = source.name  # need this to rebuild the slug

        vector_metadata = {"source": source.name, "table_name": table_name}
        if existing_items := self.vector_store.filter_by(vector_metadata):
            if existing_items and existing_items[0].get("metadata", {}).get("enriched"):
                return
            self.vector_store.delete([item['id'] for item in existing_items])

        # the following is for the embeddings/vector store use only (i.e. filter from 1000s of tables)
        vector_metadata["enriched"] = True
        enriched_text = f"Table: {table_name}"
        if description := table_metadata.pop('description', ''):
            enriched_text += f"\nDescription: {description}"
        if self.include_columns and (columns := table_metadata.pop('columns', {})):
            enriched_text += "\nColumns:"
            for col_name, col_info in columns.items():
                col_text = f"\n- {col_name}"
                if col_info.get("description"):
                    col_text += f": {col_info['description']}"
                enriched_text += col_text
            enriched_text += "\n"
        if self.include_misc:
            enriched_text += "\nMiscellaneous:"
            for key, value in table_metadata.items():
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

    async def _iterative_table_selection(
        self,
        sources: dict[str, Source],
        messages: list[dict[str, str]],
        closest_tables: list[str],
        table_similarities: dict[str, float]
    ) -> dict:
        """
        Performs an iterative table selection process to gather context.
        This function:
        1. From the top tables in TableLookup results, presents all tables and columns to the LLM
        2. Has the LLM select ~3 tables for deeper examination
        3. Gets complete schemas for these tables
        4. Repeats until the LLM is satisfied with the context
        """
        if not sources:
            return {}

        max_iterations = self.max_selection_iterations
        tables_info = self._memory.get("tables_info", {})
        examined_tables = set(tables_info.keys())

        for iteration in range(1, max_iterations + 1):
            log_debug(f"Iterative table selection iteration {iteration} / {max_iterations}")

            available_tables = [
                table for table in closest_tables
                if table not in examined_tables
            ]
            if not available_tables:
                log_debug("No more tables available to examine")
                break

            fast_track = False
            if iteration == 1:
                # For the first iteration, select tables based on similarity
                # If any tables have a similarity score above the threshold, select up to 5 of those tables
                if any(similarity > self.table_similarity_threshold for similarity in table_similarities.values()):
                    closest_tables = selected_table_slugs = sorted(
                        table_similarities,
                        key=lambda x: table_similarities[x],
                        reverse=True
                    )[:5]
                    log_debug(f"Selected tables based on similarity threshold: {', '.join(selected_table_slugs)}")
                    fast_track = True
                else:
                    # If not, select the top 3 tables to examine
                    selected_table_slugs = available_tables[:3]
                    log_debug(f"Selected initial tables: {', '.join(selected_table_slugs)}")

            # For subsequent iterations, the LLM selects tables in the previous iteration
            with self.interface.add_step(title="Fetching detailed schemas") as step:
                step.stream(f"Fetching detailed schema information for {len(selected_table_slugs)} tables\n")
                tables_schema = []
                for table_slug in selected_table_slugs:
                    table_schema = await fetch_table_schema(sources, table_slug, include_count=True)
                    tables_schema.append(table_schema)
                    stream_details(f"\n- {table_slug}\n```json\n{table_schema}\n```", step)

            for table_slug, schema_data in tables_schema:
                tables_info[table_slug] = schema_data
                examined_tables.add(table_slug)

            if fast_track:
                # based on similarity search alone, if we have selected tables, we're done!!
                break

            try:
                # Render the prompt using the proper template system
                system = await self._render_prompt(
                    "table_selection",
                    messages,
                    current_query=messages[-1]["content"],
                    available_tables=available_tables,
                    examined_tables=examined_tables,
                    separator=SOURCE_TABLE_SEPARATOR,
                    current_schemas=tables_info,
                    iteration=iteration
                )

                log_debug(f"Selecting tables from {len(available_tables)} available options")

                # Get the model from the prompt definition
                model_spec = self.prompts["table_selection"].get("llm_spec", self.llm_spec_key)
                tables_model = self._get_model("table_selection", available_tables + list(examined_tables))

                output = await self.llm.invoke(
                    messages,
                    system=system,
                    model_spec=model_spec,
                    response_model=tables_model,
                )

                selected_table_slugs = output.selected_table_slugs or []
                log_debug(f"Selected tables: {', '.join(selected_table_slugs)}")

                # Check if we're done with table selection
                if output.is_satisfied or not available_tables:
                    log_debug("Selection process complete - model is satisfied with selected tables")
                    break
            except Exception as e:
                log_debug(f"Error during table selection: {e!s}")
                break

        tables_info = {
            table_slug: schema_data for table_slug, schema_data in tables_info.items()
            if table_slug in (selected_table_slugs or [])  # prevent it from getting too big and waste token
        }
        return tables_info

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        """
        Fetches the closest tables based on the user query, with query refinement
        if enabled and initial results don't meet threshold.
        """
        query = messages[-1]["content"]
        results = []
        table_similarities = {}
        if len(self._table_metadata) <= 5:
            table_similarities = {table_slug: 1 for table_slug in self._table_metadata}
        else:
            table_similarities = {
                table_slug: 1
                for table_slug in self._table_metadata
                if table_slug.split(SOURCE_TABLE_SEPARATOR)[-1].lower() in query.lower()
            }

        closest_tables = list(table_similarities)
        if not closest_tables:
            results = await self._perform_search_with_refinement(query)

        any_matches = any(result['similarity'] >= self.min_similarity for result in results)
        for result in results:
            if any_matches and result['similarity'] < self.min_similarity:
                continue
            source_name = result['metadata']["source"]
            table_name = result['metadata']["table_name"]
            table_slug = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"
            description = f"- `{table_slug}`: {result['similarity']:.3f} similarity"
            table_similarities[table_slug] = result['similarity']
            if table_metadata := self._table_metadata.get(table_slug):
                if table_description := table_metadata.get("description"):
                    description += f"  Description: {table_description}"
                columns_description = "Columns:"
                for i, (col_name, col_info) in enumerate(table_metadata.get("columns", {}).items()):
                    # Save on tokens by truncating long column names and letting the LLM infer the rest
                    col_label = col_name if len(col_name) <= 50 else f"{col_name[:15]}...{col_name[-35:]}"
                    col_desc = f": {col_info['description']}" if col_info.get("description") else ""
                    columns_description += f"\n- {col_label}{col_desc}"
                    if i > 10:
                        columns_description += f"\n  ... (out of {len(table_metadata['columns'])} columns)"
                        break
                description += f"\n{indent(columns_description, ' ' * 2)}"
            closest_tables.append(table_slug)

        sources = {source.name: source for source in self._memory.get("sources", [])}
        if self.iteratively_select and len(closest_tables) > 5:
            fused_messages = fuse_messages(messages, max_user_messages=3)
            tables_info = await self._iterative_table_selection(
                sources, fused_messages, closest_tables, table_similarities)
        else:
            with self.interface.add_step(title="Fetching schemas") as step:
                step.stream(f"Fetching schemas for {len(closest_tables)} tables")
                tables_info = await fetch_table_schemas(
                    sources, closest_tables, include_count=True)
                for table_slug, table_schema in tables_info.items():
                    stream_details(f"\n- {table_slug}\n```json\n{table_schema}\n```", step)

        self._memory["tables_info"] = tables_info
        context = (
            f"\n\nHere are the relevant tables and their schemas:"
            f"\n\n```json\n{tables_info}\n```"
        )
        return context


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
