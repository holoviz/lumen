
from typing import Any

import param

from ..context import TContext
from ..tools.vector_lookup import VectorLookupInputs, VectorLookupTool
from ..utils import truncate_string


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

    def _format_results_for_refinement(
        self, results: list[dict[str, Any]], context: TContext
    ) -> str:
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
