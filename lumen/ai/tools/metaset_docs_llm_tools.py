"""
LLM tool for on-demand retrieval of documentation chunks attached to the current :class:`~lumen.ai.schemas.Metaset`.

Injected for all actors when ``context['metaset']`` has matched docs (via :class:`~lumen.ai.tools.metadata_lookup.MetadataLookup`).
"""

from __future__ import annotations

from ..context import TContext
from ..utils import truncate_string
from .base import FunctionTool


def make_load_metaset_relevant_docs_tool(context: TContext) -> list[FunctionTool]:
    """
    When MetadataLookup attached documentation chunks to the metaset, expose an LLM tool
    to fetch full chunk text on demand. The tool purpose summarizes count and similarity range.
    """
    metaset = context.get("metaset", None)
    if metaset is None:
        return []
    stats = metaset.docs_retrieval_stats()
    if stats is None:
        return []
    n_chunks, sim_lo, sim_hi = stats

    def load_relevant_metaset_docs(max_chunks: int = 12) -> str:
        """
        Return full text of the most relevant documentation chunks for this query (by similarity).
        """
        cap = max(1, min(int(max_chunks), 40))
        ranked = sorted(metaset.get_docs(), key=lambda c: c.similarity, reverse=True)[:cap]
        parts: list[str] = [
            f"## Documentation ({len(ranked)} of {n_chunks} chunk(s), "
            f"similarity range across all retrieved chunks: {sim_lo:.3f}-{sim_hi:.3f})\n"
        ]
        for i, chunk in enumerate(ranked, start=1):
            body = truncate_string(" ".join((chunk.text or "").split()), 2000)
            parts.append(
                f'### {i}. {chunk.filename} (similarity={float(chunk.similarity):.3f})\n{body}\n'
            )
        text = "\n".join(parts)
        if len(text) > 16000:
            text = text[:16000] + "\n... (truncated)"
        return text

    load_relevant_metaset_docs.__doc__ = (load_relevant_metaset_docs.__doc__ or "").strip()
    purpose = (
        f"Documentation: {n_chunks} chunk(s) matched this request (vector similarity min={sim_lo:.3f}, "
        f"max={sim_hi:.3f}). Fetches full text of the top chunks by relevance — use when definitions, "
        "business rules, or field meanings in those docs should inform your answer (SQL, charts, or prose)."
    )
    return [FunctionTool(load_relevant_metaset_docs, purpose=purpose)]
