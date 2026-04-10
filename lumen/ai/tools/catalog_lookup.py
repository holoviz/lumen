from __future__ import annotations

from typing import Any, NotRequired

import param

from ..context import ContextModel, TContext
from ..embeddings import NumpyEmbeddings
from ..llm import Message
from ..utils import log_debug
from ..vector_store import NumpyVectorStore, VectorStore
from .vector_lookup import VectorLookupTool


class CatalogLookupOutputs(ContextModel):
    document_chunks: list[str]
    matches: NotRequired[list[dict]]


class CatalogLookupTool(VectorLookupTool):
    """
    Searches catalog entries embedded by CatalogSourceControls.

    CatalogSourceControls._embed_catalog() populates the vector store
    with items whose metadata includes ``{"type": "catalog_entry"}``.
    This tool queries that same vector store using the ``_item_type_name``
    filter so it only finds catalog entries, not table schemas or documents.

    Wire both this tool and the CatalogSourceControls to the **same**
    VectorStore instance so that the control writes and the tool reads.
    """

    purpose = param.String(default="""
        Discovers relevant datasets from browsable catalogs (e.g. CELLxGENE,
        data portals) using semantic search over catalog metadata. Returns
        ranked matches so the user can pick which dataset to load.""")

    conditions = param.List(default=[
        "Use when the user wants to find, discover, search for, or browse datasets",
        "Use when the user describes a research question and needs to find relevant data",
        "Use when the user asks what data or datasets are available before any sources are loaded",
        "NOT when the user already has data loaded and wants to query, visualize, or analyze it",
    ])

    always_use = param.Boolean(default=False)

    n = param.Integer(default=8, bounds=(1, None), doc="""
        Number of catalog results to return.""")

    min_similarity = param.Number(default=0.05, doc="""
        Minimum similarity to include a catalog entry.""")

    _item_type_name = "catalog_entry"

    output_schema = CatalogLookupOutputs

    def __init__(self, **params):
        if "vector_store" not in params:
            params["vector_store"] = NumpyVectorStore(embeddings=NumpyEmbeddings())
        super().__init__(**params)

    async def prepare(self, context: TContext):
        # CatalogLookupTool does not need to populate the vector store
        # itself — CatalogSourceControls._embed_catalog does that.
        # Override to skip VectorLookupTool.prepare which calls
        # _update_vector_store (meant for MetadataLookup).
        pass

    async def _update_vector_store(self, context: TContext):
        # No-op: the catalog controls handle embedding.
        pass

    def _format_results_for_refinement(
        self, results: list[dict[str, Any]], context: TContext
    ) -> str:
        """Format catalog results for the query-refinement prompt."""
        lines = []
        for r in results:
            text = r.get("text", "")
            sim = r.get("similarity", 0)
            lines.append(f"- {text[:200]} (similarity: {sim:.3f})")
        return "\n".join(lines)

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        **kwargs: Any,
    ) -> tuple[list[Any], CatalogLookupOutputs]:
        query = messages[-1]["content"]

        # Check whether the store has any catalog entries at all.
        # If not, the catalog may still be loading in the background.
        store_size = len(self.vector_store) if self.vector_store else 0
        if store_size == 0:
            log_debug("[CatalogLookupTool] Vector store is empty — catalog may still be loading.")
            return ["No catalog entries available yet — the catalog may still be loading."], {"document_chunks": []}

        results = await self._perform_search_with_refinement(
            query, context, filters={"type": "catalog_entry"}
        )

        matches = [
            r for r in results
            if r.get("similarity", 0) >= self.min_similarity
        ]

        if not matches:
            return ["No matching datasets found in the catalog."], {"document_chunks": []}

        # Build a readable summary the agent prompt can use.
        parts = ["The following datasets were found in the catalog:\n"]
        for i, m in enumerate(matches, 1):
            text = m["text"]
            sim = m["similarity"]
            meta = m.get("metadata", {})
            parts.append(f"{i}. {text}")
            # Surface any structured metadata beyond type
            extra = {k: v for k, v in meta.items() if k != "type"}
            if extra:
                parts.append(f"   Metadata: {extra}")
            parts.append(f"   (relevance: {sim:.2f})")
            parts.append("")

        message = "\n".join(parts)
        chunks = [m["text"] for m in matches]

        # Build structured match info for the agent to construct buttons
        structured_matches = []
        for m in matches:
            meta = m.get("metadata", {})
            required_str = meta.get("_required_params", "")
            structured_matches.append({
                "label": m["text"][:80],
                "row_idx": meta.get("_row_idx"),
                "control_id": meta.get("_control_id"),
                "similarity": m["similarity"],
                "required_params": required_str.split(",") if required_str else [],
                "params_schema": meta.get("_params_schema", ""),
            })

        return [message], {"document_chunks": chunks, "matches": structured_matches}
