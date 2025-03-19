from __future__ import annotations

import asyncio

from textwrap import indent
from typing import Any

import param

from panel.viewable import Viewable

from ..views.base import View
from .actor import Actor, ContextProvider
from .config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from .embeddings import NumpyEmbeddings
from .llm import Message
from .models import make_refined_query_model
from .translate import function_to_model
from .utils import log_debug
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
            log_debug(f"Error refining query: {e}")
            return original_query

    async def _perform_search_with_refinement(self, query: str, **kwargs) -> tuple[str, list[dict[str, Any]], float]:
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
            - The best similarity score
        """
        original_query = query
        final_query = query
        current_query = query
        iteration = 0

        log_debug(f"Performing initial search with query: '{query}'")
        results = self.vector_store.query(query, top_k=self.n, **kwargs)
        best_similarity = max([result.get('similarity', 0) for result in results], default=0)
        best_results = results
        log_debug(f"Initial search found {len(results)} results with best similarity: {best_similarity:.3f}")

        self._memory["original_query"] = original_query
        self._memory["original_similarity"] = best_similarity

        refinement_history = []
        if not self.enable_query_refinement or best_similarity >= self.refinement_similarity_threshold:
            return final_query, best_results, best_similarity

        log_debug(f"Attempting to refine query (similarity {best_similarity:.3f} below threshold {self.refinement_similarity_threshold:.3f})")

        while iteration < self.max_refinement_iterations and best_similarity < self.refinement_similarity_threshold:
            iteration += 1
            log_debug(f"Refinement iteration {iteration}/{self.max_refinement_iterations}")

            refined_query = await self._refine_query(current_query, results)

            if refined_query == current_query:
                log_debug("Refinement returned unchanged query, stopping iterations")
                break

            current_query = refined_query
            log_debug(f"Query refined to: \033[92m'{refined_query}'\033[0m")

            new_results = self.vector_store.query(refined_query, top_k=self.n, **kwargs)
            new_best_similarity = max([result.get('similarity', 0) for result in new_results], default=0)

            improvement = new_best_similarity - best_similarity
            refinement_history.append({
                "iteration": iteration,
                "query": refined_query,
                "similarity": new_best_similarity,
                "improvement": improvement
            })

            log_debug(f"Refined search found {len(new_results)} results with best similarity: {new_best_similarity:.3f} (improvement: {improvement:.3f})")

            if new_best_similarity > best_similarity + self.min_refinement_improvement:
                log_debug(f"Improved results (iteration {iteration}) with similarity {new_best_similarity:.3f}")
                best_similarity = new_best_similarity
                best_results = new_results
                final_query = refined_query
                results = new_results  # Update results for next iteration's refinement
            else:
                log_debug(f"Insufficient improvement ({improvement:.3f} < {self.min_refinement_improvement:.3f}), stopping iterations")
                break

            # Break if we've reached an acceptable similarity
            if best_similarity >= self.refinement_similarity_threshold:
                log_debug(f"Reached acceptable similarity threshold: {best_similarity:.3f}")
                break

        if refinement_history:
            self._memory["refined_search_query"] = final_query
            log_debug(f"Final query after {iteration} iterations: '{final_query}' with similarity {best_similarity:.3f}")

        return final_query, best_results, best_similarity


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
        final_query, results, best_similarity = await self._perform_search_with_refinement(query)

        closest_doc_chunks = [
            f"{result['text']} (Relevance: {result['similarity']:.1f} - "
            f"Metadata: {result['metadata']})"
            for result in results
            if result['similarity'] >= self.min_similarity
        ]

        if not closest_doc_chunks:
            return ""

        message = "Please augment your response with the following context if relevant:\n"

        # If refined query was used, note this in the response
        if "refined_search_query" in self._memory and final_query != query:
            message = f"Refined search query '{final_query}' yielded these results:\n\n" + message

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

    provides = param.List(default=["closest_tables", "table_similarities"], readonly=True, doc="""
        List of context values this Tool provides to current working memory.""")

    batched = param.Boolean(default=True, doc="""
        Whether to batch the table metadata fetches.""")

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

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        """
        Fetches the closest tables based on the user query, with query refinement
        if enabled and initial results don't meet threshold.
        """
        query = messages[-1]["content"]

        closest_tables, descriptions = [], []
        table_similarities = {}

        # Check if any tables are mentioned verbatim
        for table in self._table_metadata:
            if table.split(SOURCE_TABLE_SEPARATOR)[-1].lower() in query.lower():
                closest_tables.append(table)
                table_similarities[table] = 1

        # If so skip search
        if closest_tables:
            final_query = query
            results = []
            any_matches = True
            best_similarity = 1
        else:
            # otherwise perform search with refinement
            final_query, results, best_similarity = await self._perform_search_with_refinement(query)
            any_matches = any(result['similarity'] >= self.min_similarity for result in results)

        # Process results as before
        for result in results:
            if any_matches and result['similarity'] < self.min_similarity:
                continue
            source_name = result['metadata']["source"]
            table_name = result['metadata']["table_name"]
            table_slug = f"{source_name} - {table_name}"
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
            descriptions.append(description)

        if any_matches:
            message = "Based on the vector search, the most relevant tables are:\n"
        else:
            message = "Based on the vector search, was unable to find tables relevant to the query.\n"
            if closest_tables:
                message += "Here are some other tables:\n"

        self._memory["closest_tables"] = closest_tables
        self._memory["table_similarities"] = table_similarities
        if "refined_search_query" in self._memory and final_query != query:
            message = f"\n\nRefined search query: '{final_query}'\n\n" + message
        return message + "\n".join(descriptions)


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
