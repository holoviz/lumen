import traceback

from typing import Any

import param

from pydantic import Field, create_model

from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..embeddings import NumpyEmbeddings
from ..llm import Message
from ..models import PartialBaseModel
from ..utils import log_debug, stream_details
from ..vector_store import NumpyVectorStore, VectorStore
from .base import Tool, ToolUser


def make_refined_query_model(item_type_name: str = "items"):
    """
    Creates a model for refining search queries in vector lookup tools.
    """
    return create_model(
        "RefinedQuery",
        chain_of_thought=(str, Field(
            description=f"""
            Analyze the current search results for {item_type_name}. Consider whether the terms used
            in the query match the terms that might be found in {item_type_name} names and descriptions.
            Think about more specific or alternative terms that might yield better results.
            """
        )),
        refined_search_query=(str, Field(
            description=f"""
            A refined search query that would help find more relevant {item_type_name}.
            This should be a Google-style keyword search query, focusing on terms
            that might appear in relevant {item_type_name}.
            """,
        )),
        __base__=PartialBaseModel
    )


class VectorLookupInputs(ContextModel):
    document_sources: dict[str, dict[str, str]]


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

    refinement_similarity_threshold = param.Number(default=0.05, bounds=(0, 1), doc="""
        Similarity threshold below which query refinement is triggered.""")

    vector_store = param.ClassSelector(class_=VectorStore, constant=True, doc="""
        Vector store object which is queried to provide additional context
        before responding.""")

    document_vector_store = param.ClassSelector(
        class_=VectorStore,
        default=None,
        constant=True,
        doc="""\n        Separate vector store for documents. If None, uses the same vector store as tables.\n        This allows for different embedding strategies or storage backends for documents vs tables.""")

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

    def _format_results_for_refinement(
        self, results: list[dict[str, Any]], context: TContext
    ) -> str:
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

        # Always pass document_vector_store if available
        if self.document_vector_store is not None:
            kwargs["document_vector_store"] = self.document_vector_store

        # First, try to inherit vector_store from another tool with the same _item_type_name
        # This takes precedence over self.vector_store to allow tools to share stores
        inherited_vector_store = None
        tool_item_type = tool._item_type_name
        for t in prompt_tools:
            if not isinstance(t, VectorLookupTool) or t.vector_store is None:
                continue
            source_item_type = t._item_type_name
            # Only inherit if item types match or either is None
            if tool_item_type is None or source_item_type is None or tool_item_type == source_item_type:
                inherited_vector_store = t.vector_store
                break

        if inherited_vector_store is not None:
            kwargs["vector_store"] = inherited_vector_store
        elif self.vector_store is not None:
            # Fall back to self.vector_store if no inheritance source found
            kwargs["vector_store"] = self.vector_store
        else:
            # Default to NumpyVectorStore if nothing else is available
            kwargs["vector_store"] = NumpyVectorStore()
        return kwargs
