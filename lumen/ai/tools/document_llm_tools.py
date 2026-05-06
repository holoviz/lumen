"""
LLM-callable tools for listing and semantically searching the document vector store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..context import TContext
from ..utils import truncate_string
from .base import FunctionTool

if TYPE_CHECKING:
    from ..vector_store import VectorStore


def _doc_metadata_filters(
    context: TContext,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    When the document store is shared with table metadata, restrict to document chunks.
    """
    filters: dict[str, str] = {}
    if context.get("document_vector_store") is context.get("vector_store"):
        filters["type"] = "document"
    if extra:
        filters.update(extra)
    return filters


def make_document_vector_llm_tools(context: TContext) -> list[Any]:
    """
    Build ``FunctionTool`` instances for the document vector store in *context*.

    Expects ``context['document_vector_store']`` (set by :class:`~lumen.ai.coordinator.base.Coordinator`).
    Returns an empty list if no store is configured.
    """

    store: VectorStore | None = context.get("document_vector_store")
    if store is None or len(store) == 0:
        return []

    # When the document store is shared with the table-metadata store,
    # len(store) > 0 can be true even with zero document entries.
    # Check for actual document-type chunks before exposing the tools.
    if context.get("document_vector_store") is context.get("vector_store"):
        if not store.filter_by({"type": "document"}, limit=1):
            return []

    visible_docs: set[str] | None = context.get("visible_docs")
    list_cap = 4000

    def list_indexed_documents(
        limit: int = 40,
        offset: int = 0,
    ) -> str:
        """
        List distinct document filenames currently indexed in the document vector store (paginated).
        Does not run semantic search — use search_document_chunks for that.
        Call this first to see which files you can narrow searches to.
        """
        filters = _doc_metadata_filters(context)
        rows = store.filter_by(filters if filters else {}, limit=list_cap, offset=0)
        filenames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            meta = row.get("metadata") or {}
            fn = meta.get("filename")
            if not fn:
                continue
            if visible_docs is not None and fn not in visible_docs:
                continue
            if fn not in seen:
                seen.add(fn)
                filenames.append(str(fn))
        filenames.sort()
        total = len(filenames)
        if total == 0:
            return "No indexed documents found (store empty or no document-type chunks yet)."
        window = filenames[offset : offset + limit]
        lines = "\n".join(f"- {f}" for f in window)
        tail = f"\n\nShowing {len(window)} of {total} unique file(s); offset={offset}."
        return lines + tail

    async def search_document_chunks(
        query: str,
        top_k: int = 8,
        min_similarity: float = 0.25,
        filename: str | None = None,
    ) -> str:
        """
        Semantic search over embedded document chunks. Supply a natural-language or keyword query.
        Optionally restrict to one file name from list_indexed_documents.
        """
        q = (query or "").strip()
        if not q:
            return "Empty query; provide search text."

        extra: dict[str, str] = {}
        if filename:
            extra["filename"] = filename
        filters = _doc_metadata_filters(context, extra if extra else None)
        use_filters: dict[str, str] | None = filters if filters else None

        results = await store.query(
            text=q,
            top_k=max(1, min(top_k, 50)),
            filters=use_filters,
            threshold=min_similarity,
        )

        if visible_docs is not None:
            results = [
                r
                for r in results
                if (r.get("metadata") or {}).get("filename") in visible_docs
            ]

        if not results:
            return (
                "No chunks matched the query above the similarity threshold. "
                "Try different keywords, lower min_similarity, or list_indexed_documents."
            )

        parts: list[str] = []
        for i, r in enumerate(results, start=1):
            meta = r.get("metadata") or {}
            fn = meta.get("filename", "?")
            sim = float(r.get("similarity", 0.0))
            text = r.get("text") or ""
            text = truncate_string(" ".join(text.split()), 900)
            parts.append(f"### {i}. {fn} (similarity={sim:.3f})\n{text}")
        out = "\n\n".join(parts)
        if len(out) > 14000:
            out = out[:14000] + "\n... (truncated)"
        return out

    list_indexed_documents.__doc__ = (list_indexed_documents.__doc__ or "").strip()
    search_document_chunks.__doc__ = (search_document_chunks.__doc__ or "").strip()

    return [
        FunctionTool(
            list_indexed_documents,
            purpose=(
                "List distinct filenames in the document vector index (paginated). "
                "Use before search_document_chunks when you need to scope by file."
            ),
        ),
        FunctionTool(
            search_document_chunks,
            purpose=(
                "Run semantic search over embedded documentation chunks; optional filename filter. "
                "Use for definitions, policies, or domain text not in SQL catalogs."
            ),
        ),
    ]
