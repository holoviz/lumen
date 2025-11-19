import asyncio
import traceback

from types import FunctionType
from typing import Annotated, Any, TypedDict

import param

from panel.io import cache as pn_cache
from panel.pane import HoloViews as HoloViewsPanel, panel as as_panel
from panel.viewable import Viewable

from ..sources.base import Source
from ..sources.duckdb import DuckDBSource
from ..views.base import HoloViews, View
from .actor import Actor, ContextProvider
from .config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from .context import ContextModel, TContext
from .embeddings import NumpyEmbeddings
from .llm import Message
from .models import (
    ThinkingYesNo, make_iterative_selection_model, make_refined_query_model,
)
from .schemas import (
    Column, DbtslMetadata, DbtslMetaset, SQLMetadata, SQLMetaset,
    VectorMetadata, VectorMetaset, get_metaset,
)
from .services import DbtslMixin
from .translate import function_to_model
from .utils import (
    get_schema, log_debug, parse_table_slug, process_enums, stream_details,
    truncate_string, with_timeout,
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
            self._tools[prompt_name] = self._initialize_tools_for_prompt(prompt_name, **params)

    def _get_tool_kwargs(self, tool, prompt_tools, **params):
        """
        Get kwargs for initializing a tool.
        Subclasses can override to provide additional kwargs.

        Parameters
        ----------
        tool : object
            The tool (class or instance) being initialized
        prompt_tools : list
            List of all tools for this prompt

        Returns
        -------
        dict
            Keyword arguments for tool initialization
        """
        return {"llm": self.llm or params.get("llm"), "interface": self.interface or params.get("interface")}

    def _initialize_tools_for_prompt(self, tools_or_key: str | list, **params) -> list:
        """
        Initialize tools for a specific prompt.

        Parameters
        ----------
        tools : str | list
            The list of tools to initialize. If None, tools are looked up from the prompt.
        **params : dict
            Additional parameters for tool initialization

        Returns
        -------
        list
            List of instantiated tools
        """
        if isinstance(tools_or_key, str):
            prompt_tools = self._lookup_prompt_key(tools_or_key, "tools")
        else:
            prompt_tools = tools_or_key
        instantiated_tools = []

        # Initialize each tool
        for tool in prompt_tools:
            if isinstance(tool, Actor):
                # For already instantiated Actors, set properties directly
                if tool.llm is None:
                    tool.llm = self.llm or params.get("llm")
                if tool.interface is None:
                    tool.interface = self.interface or params.get("interface")

                # Apply any additional configuration from subclasses
                tool_kwargs = self._get_tool_kwargs(tool, instantiated_tools or prompt_tools, **params)
                for key, value in tool_kwargs.items():
                    if key not in ('llm', 'interface') and hasattr(tool, key):
                        setattr(tool, key, value)

                instantiated_tools.append(tool)
            elif isinstance(tool, FunctionType):
                # Function tools only get basic kwargs
                tool_kwargs = self._get_tool_kwargs(tool, instantiated_tools or prompt_tools, **params)
                instantiated_tools.append(FunctionTool(tool, **tool_kwargs))
            else:
                # For classes that need to be instantiated
                tool_kwargs = self._get_tool_kwargs(tool, instantiated_tools or prompt_tools, **params)
                instantiated_tools.append(tool(**tool_kwargs))
        return instantiated_tools

    async def _use_tools(self, prompt_name: str, messages: list[Message], context: TContext) -> str:
        tools_context = ""
        # TODO: INVESTIGATE WHY or self.tools is needed
        for tool in self._tools.get(prompt_name, []) or self.tools:
            if all(requirement in context for requirement in tool.input_schema.__required_keys__):
                tool_context = await tool.respond(messages, context)
                if tool_context:
                    tools_context += f"\n{tool_context}"
        return tools_context


class VectorLookupToolUser(ToolUser):
    """
    VectorLookupToolUser is a mixin class for actors that use vector lookup tools.
    """

    document_vector_store = param.ClassSelector(
        class_=VectorStore, default=None, doc="""
        The vector store to use for document tools. If not provided, a new one will be created
        or inferred from the tools provided."""
    )

    vector_store = param.ClassSelector(
        class_=VectorStore, default=None, doc="""
        The vector store to use for the tools. If not provided, a new one will be created
        or inferred from the tools provided."""
    )

    def _get_tool_kwargs(self, tool, prompt_tools, **params):
        """
        Override to provide vector_store to applicable tools.

        Parameters
        ----------
        tool : object
            The tool (class or instance) being initialized
        prompt_tools : list
            List of all tools for this prompt
        **params : dict
            Additional parameters for tool initialization

        Returns
        -------
        dict
            Keyword arguments for tool initialization including vector_store if applicable
        """
        # Get base kwargs from parent
        kwargs = super()._get_tool_kwargs(tool, prompt_tools, **params)

        # If the tool is already instantiated and has a vector_store, use it
        if (
            ((isinstance(tool, type) and not issubclass(tool, VectorLookupTool)) and not isinstance(tool, VectorLookupTool)) or
            (isinstance(tool, VectorLookupTool) and tool.vector_store is not None)
        ):
            return kwargs
        elif isinstance(tool, VectorLookupTool) and tool._item_type_name == "document" and self.document_vector_store is not None:
            kwargs["vector_store"] = self.document_vector_store
            return kwargs
        elif self.vector_store is not None:
            kwargs["vector_store"] = self.vector_store
            return kwargs

        vector_store = next(
            (t.vector_store for t in prompt_tools
             if isinstance(t, VectorLookupTool)),
            None
        )
        # Only inherit vector_store if the _item_type_name is the same
        if hasattr(tool, "vector_store") and vector_store is not None:
            # Find the source tool that provided the vector_store
            source_tool = next(
                (t for t in prompt_tools
                 if isinstance(t, VectorLookupTool) and hasattr(t, "vector_store")
                 and t.vector_store is vector_store),
                None
            )

            # Get item types for comparison
            # Handle both class types and instances
            tool_item_type = getattr(tool, "_item_type_name", None)
            source_item_type = getattr(source_tool, "_item_type_name", None) if source_tool else None

            # Only set vector_store if item types match (e.g., IterativeTableLookup and TableLookup both use "tables")
            # or if either doesn't specify a type (None)
            if tool_item_type is None or source_item_type is None or tool_item_type == source_item_type:
                kwargs["vector_store"] = vector_store
        else:
            # default to NumpyVectorStore if not provided
            kwargs["vector_store"] = NumpyVectorStore()
        return kwargs


class Tool(Actor, ContextProvider):
    """
    A Tool can be invoked by another Actor to provide additional
    context or respond to a question. Unlike an Agent they never
    interact with or on behalf of a user directly.
    """

    always_use = param.Boolean(default=False, doc="""
        Whether to always use this tool, even if it is not explicitly
        required by the current context.""")

    conditions = param.List(default=[
        "Always requires a supporting agent to interpret results"
    ])

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        """
        Additional checks to determine if the tool should be used.
        """
        return True

    async def sync(self, context: TContext):
        """
        Allows the tool to update when the provided context changes.
        """



class VectorLookupOutputs(ContextModel):
    document_chunks: list[str]


class VectorLookupTool(Tool):
    """
    Baseclass for tools that search a vector database for relevant
    chunks.
    """

    enable_query_refinement = param.Boolean(default=True, doc="""
        Whether to enable query refinement for improving search results.""")

    max_refinement_iterations = param.Integer(default=3, bounds=(1, 10), doc="""
        Maximum number of refinement iterations to perform.""")

    min_similarity = param.Number(default=0.3, doc="""
        The minimum similarity to include a document.""")

    min_refinement_improvement = param.Number(default=0.05, bounds=(0, 1), doc="""
        Minimum improvement in similarity score required to keep refining.""")

    n = param.Integer(default=5, bounds=(1, None), doc="""
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

    vector_store = param.ClassSelector(class_=VectorStore, constant=True, doc="""
        Vector store object which is queried to provide additional context
        before responding.""")

    _item_type_name: str = None

    # Class variable to track which sources are currently being processed
    _sources_in_progress = {}

    __abstract = True

    output_schema = VectorLookupOutputs

    def __init__(self, **params):
        if 'vector_store' not in params:
            params['vector_store'] = NumpyVectorStore(embeddings=NumpyEmbeddings())
        super().__init__(**params)

    async def prepare(self, context: TContext):
        if "tables_metadata" not in context:
            context["tables_metadata"] = {}
        await self._update_vector_store(context)

    def _handle_ready_task_done(self, task):
        """Properly handle exceptions from async ready tasks."""
        try:
            # This will re-raise the exception if one occurred
            if task.exception():
                raise task.exception()
        except Exception as e:
            log_debug(f"Error in ready task: {type(e).__name__} - {e!s}")
            traceback.print_exc()

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
        results: list[dict[str, Any]],
        context: TContext
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
        results_description = self._format_results_for_refinement(results, context)

        messages = [{"role": "user", "content": original_query}]

        try:
            refined_query_model = self._get_model("refine_query", item_type_name=self._item_type_name)
            system_prompt = await self._render_prompt(
                "refine_query",
                messages,
                context,
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

    async def _perform_search_with_refinement(self, query: str, context: TContext, **kwargs) -> list[dict[str, Any]]:
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

        filters = kwargs.pop("filters", {})
        if self._item_type_name and "type" not in filters:
            filters["type"] = self._item_type_name
        kwargs["filters"] = filters
        results = await self.vector_store.query(query, top_k=self.n, **kwargs)

        # check if all metadata is the same; if so, skip
        if all(result.get('metadata') == results[0].get('metadata') for result in results) or self.llm is None:
            return results

        with self._add_step(title="Vector Search with Refinement", steps_layout=self.steps_layout) as step:
            best_similarity = max([result.get('similarity', 0) for result in results], default=0)
            best_results = results
            step.stream(f"Initial search found {len(results)} chunks with best similarity: {best_similarity:.3f}\n\n")
            stream_details("\n".join(
                f'```\n{result["text"]} (Similarity: {result.get("similarity", 0):.3f})\n\n{result["metadata"]}\n```'
                for result in results
            ), step, title=f"{len(results)} chunks", auto=False)

            refinement_history = []
            if not self.enable_query_refinement or best_similarity >= self.refinement_similarity_threshold or len(results) == 1:
                step.stream("Search complete - no refinement needed.")
                return best_results

            step.stream(f"Attempting to refine query (similarity {best_similarity:.3f} below threshold {self.refinement_similarity_threshold:.3f})\n\n")
            while iteration < self.max_refinement_iterations and best_similarity < self.refinement_similarity_threshold:
                iteration += 1
                step.stream(f"Processing refinement iteration {iteration}/{self.max_refinement_iterations}\n\n")

                refined_query = await self._refine_query(current_query, results, context)

                if refined_query == current_query:
                    step.stream("Refinement returned unchanged query, stopping iterations.")
                    break

                current_query = refined_query
                new_results = await self.vector_store.query(refined_query, top_k=self.n, **kwargs)
                new_best_similarity = max([result.get('similarity', 0) for result in new_results], default=0)

                improvement = new_best_similarity - best_similarity
                refinement_history.append({
                    "iteration": iteration,
                    "query": refined_query,
                    "similarity": new_best_similarity,
                    "improvement": improvement
                })

                # Build combined details message
                details_msg = f"Query refined:\n\n{refined_query}\n\nRefined search found {len(new_results)} results with best similarity: {new_best_similarity:.3f} (improvement: {improvement:.3f})\n"

                if new_best_similarity > best_similarity + self.min_refinement_improvement:
                    details_msg += f"\nImproved results (iteration {iteration}) with similarity {new_best_similarity:.3f}"
                    best_similarity = new_best_similarity
                    best_results = new_results
                    final_query = refined_query
                    results = new_results  # Update results for next iteration's refinement
                else:
                    details_msg += f"\nInsufficient improvement ({improvement:.3f} < {self.min_refinement_improvement:.3f}), stopping iterations"
                    stream_details(details_msg, step, title=f"Refinement iteration {iteration}", auto=False)
                    break

                # Break if we've reached an acceptable similarity
                if best_similarity >= self.refinement_similarity_threshold:
                    details_msg += f"\nReached acceptable similarity threshold: {best_similarity:.3f}"
                    stream_details(details_msg, step, title=f"Refinement iteration {iteration}", auto=False)
                    break

            if refinement_history:
                stream_details(f"Final query after {iteration} iterations: '{final_query}' with similarity {best_similarity:.3f}\n", step, auto=False)

        return best_results

    async def respond(self, messages: list[Message], context: TContext, **kwargs: Any) -> tuple[list[Any], ]:
        """
        Respond to a user query using the vector store.

        Parameters
        ----------
        messages: list[Message]
            The user query and any additional context
        **kwargs: Any
            Additional arguments for the response

        Returns
        -------
        str
            The response from the vector store
        """
        query = messages[-1]["content"]

        # Perform search with refinement
        results = await self._perform_search_with_refinement(query, context)
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
        return [message], {"document_chunks": closest_doc_chunks}


class VectorLookupInputs(ContextModel):
    document_sources: dict[str, dict[str, str]]


class DocumentLookup(VectorLookupTool):
    """
    The DocumentLookup tool creates a vector store of all available documents
    and responds with a list of the most relevant documents given the user query.
    Always use this for more context.
    """

    purpose = param.String(default="""
        Looks up relevant documents based on the user query.""")

    # Override the item type name
    _item_type_name = "documents"

    input_schema = VectorLookupInputs

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
        for result in results:
            metadata = result.get('metadata', {})
            filename = metadata.get('filename', 'Unknown document')
            text_preview = truncate_string(result.get('text', ''), max_length=150)

            description = f"- {filename} (Similarity: {result.get('similarity', 0):.3f})"
            if text_preview:
                description += f"\n  Preview: {text_preview}"

            formatted_results.append(description)

        return "\n".join(formatted_results)

    async def _update_vector_store(self, context: TContext):
        # Build a list of document items for a single upsert operation
        items_to_upsert = []
        for source in context.get("sources"):
            metadata = source.get("metadata", {})
            metadata["type"] = self._item_type_name
            items_to_upsert.append({"text": source["text"], "metadata": metadata})

        # Make a single upsert call with all documents
        if items_to_upsert:
            await self.vector_store.upsert(items_to_upsert)


class TableLookupInputs(ContextModel):

    sources: Annotated[list[Source], ("accumulate", "source")]


class TableLookupOutputs(ContextModel):

    vector_metaset: VectorMetaset


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
        elif sources is None:
            raise ValueError("Context does not contain a \"source\" or \"sources\".")
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
            except asyncio.TimeoutError:
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
                except asyncio.TimeoutError:
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

        vector_metadata_map = {}
        for result in results:
            source_name = result["metadata"]["source"]
            table_name = result["metadata"]["table_name"]
            for source in context.get("sources", []):
                if source.name == source_name:
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

            vector_metadata = VectorMetadata(
                table_slug=table_slug,
                similarity=similarity_score,
                description=table_description,
                base_sql=sql,
                columns=columns,
                metadata=context["tables_metadata"].get(table_slug, {}).copy()
            )
            vector_metadata_map[table_slug] = vector_metadata

        # If query contains an exact table name, mark it as max similarity
        visible_slugs = context.get("visible_slugs", set())
        tables_to_check = context["tables_metadata"]
        if visible_slugs:
            tables_to_check = {slug: data for slug, data in tables_to_check.items() if slug in visible_slugs}

        for table_slug in tables_to_check:
            if table_slug.split(SOURCE_TABLE_SEPARATOR)[-1].lower() in query.lower():
                if table_slug in vector_metadata_map:
                    vector_metadata_map[table_slug].similarity = 1

        vector_metaset = VectorMetaset(
            vector_metadata_map=vector_metadata_map, query=query)
        return {"vector_metaset": vector_metaset}

    def _format_context(self, outputs: TableLookupOutputs) -> str:
        """Generate formatted text representation from schema objects."""
        # Get schema objects from memory
        vector_metaset = outputs.get("vector_metaset")
        return str(vector_metaset)

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

    sql_metaset: SQLMetaset


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
        Update sql_metaset when visible_slugs changes.
        This ensures SQL operations only work with visible tables by regenerating
        the sql_metaset using get_metaset.
        """
        slugs = context.get("visible_slugs", set())

        # If no visible slugs or no sources, clear sql_metaset
        if not slugs or not context.get("sources"):
            context.pop("sql_metaset", None)
            return

        if "sql_metaset" not in context:
            return

        try:
            sources = context["sources"]
            visible_tables = list(slugs)
            new_sql_metaset = await get_metaset(sources, visible_tables, prev=context.get("sql_metaset"))
            context["sql_metaset"] = new_sql_metaset
            log_debug(f"[IterativeTableLookup] Updated sql_metaset with {len(visible_tables)} visible tables")
        except Exception as e:
            log_debug(f"[IterativeTableLookup] Error updating sql_metaset: {e}")

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
        vector_metaset = out_model["vector_metaset"]
        vector_metadata_map = vector_metaset.vector_metadata_map

        sql_metadata_map = {}
        examined_slugs = set(sql_metadata_map.keys())
        # Filter to only include visible tables
        all_slugs = list(vector_metadata_map.keys())
        visible_slugs = context.get('visible_slugs', set())
        if visible_slugs:
            all_slugs = [slug for slug in all_slugs if slug in visible_slugs]

        if len(all_slugs) == 1:
            table_slug = all_slugs[0]
            source_obj, table_name = parse_table_slug(table_slug, context["sources"])
            schema = await get_schema(source_obj, table_name, reduce_enums=False)
            sql_metadata = SQLMetadata(
                table_slug=table_slug,
                schema=schema,
            )
            return {
                "sql_metaset": SQLMetaset(
                    vector_metaset=vector_metaset,
                    sql_metadata_map={table_slug: sql_metadata},
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
                    any_matches = any(schema.similarity > self.table_similarity_threshold for schema in vector_metadata_map.values())
                    limited_tables = len(vector_metadata_map) <= 5
                    if any_matches or len(all_slugs) == 1 or limited_tables:
                        selected_slugs = sorted(
                            vector_metadata_map.keys(),
                            key=lambda x: vector_metadata_map[x].similarity,
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
                            sql_metadata_map=sql_metadata_map,
                            vector_metadata_map=vector_metadata_map,
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
                    stream_details(str(vector_metaset.vector_metadata_map[table_slug]), step, title="Table details", auto=False)
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

                        sql_metadata = SQLMetadata(
                            table_slug=table_slug,
                            schema=schema,
                        )
                        sql_metadata_map[table_slug] = sql_metadata

                        examined_slugs.add(table_slug)
                        stream_details(f"```yaml\n{schema}\n```", step, title="Schema")
                    except Exception as e:
                        failed_slugs.append(table_slug)
                        stream_details(f"Error fetching schema: {e}", step, title="Error", auto=False)

                if fast_track:
                    step.stream("Fast-tracked selection process.")
                    # based on similarity search alone, if we have selected tables, we're done!!
                    break

        sql_metadata_map = {table_slug: schema_data for table_slug, schema_data in sql_metadata_map.items()}

        sql_metaset = SQLMetaset(
            vector_metaset=vector_metaset,
            sql_metadata_map=sql_metadata_map,
        )
        return {"sql_metaset": sql_metaset}

    def _format_context(self, outputs: IterativeTableLookupOutputs) -> str:
        """Generate formatted text representation from schema objects."""
        sql_metaset = outputs.get("sql_metaset")
        return str(sql_metaset)

    async def respond(
        self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]
    ) -> tuple[list[Any], IterativeTableLookupOutputs]:
        """
        Fetches tables based on the user query and returns formatted context.
        """
        out_model = await self._gather_info(messages, context)
        return [self._format_context(out_model)], out_model


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

            items_to_upsert.append({"text": enriched_text, "metadata": vector_metadata})

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

    render_output = param.Boolean(default=False, doc="""
        Whether to render the tool output directly, even if it is not already a Lumen View or Panel Viewable.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "FunctionTool" / "main.jinja2"
            },
        }
    )

    def __init__(self, function, **params):
        model = function_to_model(function, skipped=self.requires)
        if "purpose" not in params:
            params["purpose"] = f"{model.__name__}: {model.__doc__}" if model.__doc__ else model.__name__
        super().__init__(
            function=function,
            name=function.__name__,
            **params
        )
        self._model = model

    @property
    def inputs(self):
        return TypedDict(f"{self.function.__name__}Inputs", {f: Any for f in self.requires})

    async def respond(
        self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]
    ) -> tuple[list[Any], ContextModel]:
        prompt = await self._render_prompt("main", messages, context)
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
        arguments = dict(kwargs, **{k: context[k] for k in self.requires})
        if param.parameterized.iscoroutinefunction(self.function):
            result = await self.function(**arguments)
        else:
            result = self.function(**arguments)
        if isinstance(result, (View, Viewable)):
            return [result], {}
        elif self.render_output:
            p = as_panel(result)
            if isinstance(p, HoloViewsPanel):
                p = HoloViews(object=p.object)
            return [p], {}
        out_model = {}
        if self.provides:
            if len(self.provides) == 1 and not isinstance(result, dict):
                out_model[self.provides[0]] = result
            else:
                out_model.update({result[key] for key in self.provides})
        return [self.formatter.format(
            function=self.function.__name__,
            arguments=', '.join(f'{k}={v!r}' for k, v in arguments.items()),
            output=result
        )], out_model
