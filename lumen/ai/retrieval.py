"""Utilities for reranking documentation retrieved by vector similarity."""

from __future__ import annotations

import re

from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset({
    "a", "an", "and", "by", "chart", "create", "data", "for", "from",
    "in", "make", "of", "on", "plot", "show", "the", "to", "use", "with",
})
_INTENT_ALIASES = {
    "area chart": ({"area"},),
    "bar chart": ({"bar"},),
    "box plot": ({"box", "boxplot", "quartile", "whisker"},),
    "brush selection": ({"brush", "interval"}, {"selection"}),
    "heat map": ({"heatmap", "rect"},),
    "heatmap": ({"heatmap", "rect"},),
    "histogram": ({"bin", "binned", "histogram"},),
    "line chart": ({"line"},),
    "linked views": ({"crossfilter", "linked"}, {"filter", "selection"}),
    "scatter plot": ({"circle", "point", "scatter"},),
    "slider": ({"input", "range", "slider"},),
    "tooltip": ({"hover", "tooltip"},),
}
_COVERAGE_REQUIREMENTS = {
    "slider": (frozenset({"input", "binding"}),),
    "brush selection": (frozenset({"interval", "selection"}),),
    "tooltip": (frozenset({"tooltip", "channel"}),),
    "scatter plot": (
        frozenset({"scatterplot"}),
        frozenset({"scatter", "plot"}),
        frozenset({"point", "plot"}),
    ),
}
_TARGETED_DOCUMENT_QUERIES = {
    "slider": "Vega-Lite input element binding slider",
    "brush selection": "Vega-Lite interval selection brush",
    "tooltip": "Vega-Lite tooltip channel",
    "scatter plot": "Vega-Lite scatterplot circle mark",
}
_COVERAGE_SLUGS = {
    "slider": frozenset({"parameter-bind"}),
    "brush selection": frozenset({"parameter-select"}),
    "tooltip": frozenset({"tooltip"}),
    "scatter plot": frozenset({"mark-circle", "mark-point"}),
}


def _tokens(value: Any) -> set[str]:
    return set(_TOKEN_RE.findall(str(value).lower()))


def _candidate_text(document: dict) -> str:
    metadata = document.get("metadata") or {}
    values = [
        document.get("text", ""), metadata.get("category", ""),
        metadata.get("description", ""), metadata.get("page_title", ""),
        metadata.get("section_title", ""), metadata.get("slug", ""),
    ]
    values.extend(metadata.get("capabilities") or ())
    for example in metadata.get("examples") or ():
        values.append(example.get("name", ""))
        values.extend(example.get("constructs") or ())
        values.extend(example.get("capabilities") or ())
    return " ".join(str(value) for value in values if value)


def _query_concepts(query: str) -> list[set[str]]:
    """Return independently matchable concepts, including phrase aliases."""
    lowered = query.lower().replace("-", " ")
    concepts: list[set[str]] = []
    consumed: set[str] = set()
    for phrase, aliases in _INTENT_ALIASES.items():
        if phrase in lowered:
            concepts.extend(aliases)
            consumed.update(_tokens(phrase))
    concepts.extend(
        {token} for token in _tokens(lowered) - consumed - _STOP_WORDS
    )
    return concepts


def _intent_score(concepts: list[set[str]], candidate_tokens: set[str]) -> float:
    if not concepts:
        return 0
    return sum(bool(concept & candidate_tokens) for concept in concepts) / len(concepts)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0


def _requested_capabilities(query: str) -> list[str]:
    """Return explicitly requested capabilities in the order they appear."""
    lowered = query.lower().replace("-", " ")
    requested = [
        (lowered.index(phrase), phrase)
        for phrase in _COVERAGE_REQUIREMENTS
        if phrase in lowered
    ]
    return [phrase for _, phrase in sorted(requested)]


def targeted_document_queries(query: str) -> list[str]:
    """Return focused retrieval queries for explicit chart capabilities.

    A broad natural-language request can rank the exact interaction page far
    below the normal candidate pool. These small follow-up queries retrieve one
    strong candidate for each capability; reranking still decides whether and
    where to include it in the final context.
    """
    return [
        _TARGETED_DOCUMENT_QUERIES[capability]
        for capability in _requested_capabilities(query)
    ]


def targeted_document_searches(query: str) -> list[tuple[str, str]]:
    """Return capability labels paired with their focused retrieval queries."""
    return [
        (capability, _TARGETED_DOCUMENT_QUERIES[capability])
        for capability in _requested_capabilities(query)
    ]


def _requested_coverage(query: str) -> list[tuple[frozenset[str], ...]]:
    return [_COVERAGE_REQUIREMENTS[capability] for capability in _requested_capabilities(query)]


def _matches_requirement(
    document: dict,
    capability: str,
    alternatives: tuple[frozenset[str], ...],
) -> bool:
    metadata = document.get("metadata") or {}
    if metadata.get("_retrieval_capability") == capability:
        return True
    if metadata.get("slug") not in _COVERAGE_SLUGS[capability]:
        return False
    title_tokens = _tokens(metadata.get("section_title", ""))
    if capability == "tooltip" and "disable" in title_tokens:
        return False
    values = [metadata.get("slug", ""), metadata.get("section_title", "")]
    structured_tokens = _tokens(" ".join(str(value) for value in values if value))
    return any(alternative <= structured_tokens for alternative in alternatives)


def rerank_documents(
    query: str,
    candidates: list[dict],
    *,
    limit: int,
    group_key: str = "slug",
    max_per_group: int = 2,
    reserve_base_form: bool = True,
    diversity_weight: float = 0.12,
) -> list[dict]:
    """Rerank vector results by intent, canonicality, and diversity.

    Vector similarity remains the strongest signal. Exact matches against titles,
    descriptions, example names, and optional capability metadata correct close
    semantic rankings, while a small maximal-marginal-relevance penalty prevents
    the final context from collapsing onto near-duplicate sections.
    """
    if limit <= 0 or not candidates:
        return []

    concepts = _query_concepts(query)
    similarities = [float(candidate.get("similarity", 0)) for candidate in candidates]
    ranked = []
    for position, candidate in enumerate(candidates):
        metadata = candidate.get("metadata") or {}
        tokens = _tokens(_candidate_text(candidate))
        # Preserve the magnitude of the vector score. Min-max normalization is
        # actively harmful here: 0.91 versus 0.89 is inside the retrieval noise
        # floor, but normalization turns that tiny difference into 1 versus 0.
        semantic = similarities[position]
        intent = _intent_score(concepts, tokens)
        title_tokens = _tokens(" ".join(str(metadata.get(key, "")) for key in (
            "page_title", "section_title", "slug",
        )))
        title_match = _intent_score(concepts, title_tokens)
        canonical = 0.08 if reserve_base_form and metadata.get("base_form") and intent else 0
        example = 0.03 if metadata.get("examples") else 0
        score = 0.52 * semantic + 0.30 * intent + 0.10 * title_match + canonical + example
        ranked.append((score, position, candidate, tokens))

    selected: list[tuple[float, int, dict, set[str]]] = []
    groups: Counter = Counter()
    remaining = ranked.copy()

    def eligible_items():
        return [
            item for item in remaining
            if (item[2].get("metadata") or {}).get(group_key) is None
            or groups[(item[2].get("metadata") or {}).get(group_key)] < max_per_group
        ]

    def selection_score(item):
        redundancy = max(
            (_jaccard(item[3], chosen[3]) for chosen in selected),
            default=0,
        )
        return item[0] - diversity_weight * redundancy, -item[1]

    def select(item):
        selected.append(item)
        remaining.remove(item)
        group = (item[2].get("metadata") or {}).get(group_key)
        if group is not None:
            groups[group] += 1

    # A request with multiple explicit capabilities needs documentation for each
    # of them. Without this pass, several high-scoring selection pages can evict
    # the scatter or tooltip example that makes the final specification usable.
    for capability, requirement in zip(_requested_capabilities(query), _requested_coverage(query), strict=True):
        if len(selected) >= limit:
            break
        matching = [
            item for item in eligible_items()
            if _matches_requirement(item[2], capability, requirement)
        ]
        if matching:
            select(max(matching, key=selection_score))

    while remaining and len(selected) < limit:
        eligible = eligible_items()
        if not eligible:
            break
        select(max(eligible, key=selection_score))

    return [item[2] for item in selected]
