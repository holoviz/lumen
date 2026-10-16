"""Utilities for reranking documentation retrieved by vector similarity."""

from __future__ import annotations

import re

from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset({
    "a", "an", "and", "by", "controlled", "create", "data", "for", "from",
    "in", "make", "of", "on", "show", "the", "to", "use", "with",
})
_CLAUSE_SPLIT_RE = re.compile(r"[,;]|\b(?:and|with|plus|including|using)\b")


def normalize_retrieval_token(token: str) -> str:
    """Apply lightweight English inflection normalization for retrieval."""
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    for suffix in ("ing", "ed"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            stem = token[:-len(suffix)]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def retrieval_tokens(value: Any) -> list[str]:
    """Tokenize text consistently for reranking and lexical retrieval."""
    return [
        normalize_retrieval_token(token)
        for token in _TOKEN_RE.findall(str(value).lower())
    ]


def _tokens(value: Any) -> set[str]:
    return set(retrieval_tokens(value))


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
    """Return independently matchable terms from a natural-language request."""
    return [{token} for token in _tokens(query) - _STOP_WORDS]


def _intent_score(concepts: list[set[str]], candidate_tokens: set[str]) -> float:
    if not concepts:
        return 0
    return sum(bool(concept & candidate_tokens) for concept in concepts) / len(concepts)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0


def focused_document_queries(query: str, limit: int = 4) -> list[str]:
    """Break a request into focused, document-searchable clauses.

    The clauses come entirely from the user's words. Each one is queried
    separately so that a specific requirement is not lost behind a broader
    chart request. This works for any documented feature rather than a fixed
    list of chart types or interaction names.
    """
    clauses = []
    for clause in _CLAUSE_SPLIT_RE.split(query.lower().replace("-", " ")):
        terms = [term for term in _TOKEN_RE.findall(clause) if term not in _STOP_WORDS]
        # ``chart`` and ``plot`` add little when a clause already names a more
        # specific feature, but retain them for a request that contains nothing
        # else (for example, "make a chart").
        if len(terms) > 1:
            terms = [term for term in terms if term not in {"chart", "plot"}]
        # Long clauses usually contain dataset fields and values rather than a
        # separable documentation need. The broad semantic + lexical searches
        # already handle those; focused searches are for concise sub-requests.
        if terms and len(terms) <= 3:
            clauses.append(" ".join(terms))

    # De-duplicate without changing the order in which the user stated needs.
    return list(dict.fromkeys(clauses))[:limit]


def fuse_ranked_documents(
    searches: list[tuple[str | None, list[dict]]],
    *,
    rank_constant: int = 60,
) -> list[dict]:
    """Combine incomparable semantic and lexical rankings using RRF.

    ``query`` is ``None`` for semantic retrieval and a focused user clause for
    lexical retrieval. The latter is retained as transient metadata so the
    final selection can preserve requirements supported by exact textual
    evidence, without knowing anything about individual visualization features.
    """
    fused: dict[tuple, dict] = {}
    for query, documents in searches:
        for rank, document in enumerate(documents, start=1):
            metadata = document.get("metadata") or {}
            key = (
                ("id", document["id"])
                if document.get("id") is not None
                else (
                    "document",
                    metadata.get("slug"),
                    metadata.get("section_title"),
                    document.get("text", ""),
                )
            )
            if key not in fused:
                fused[key] = {
                    "document": {
                        **document,
                        "metadata": {**metadata},
                    },
                    "score": 0.0,
                    "queries": [],
                    "query_scores": {},
                }
            entry = fused[key]
            contribution = 1 / (rank_constant + rank)
            entry["score"] += contribution
            if query and query not in entry["queries"]:
                entry["queries"].append(query)
            if query:
                entry["query_scores"][query] = (
                    entry["query_scores"].get(query, 0) + contribution
                )

    if not fused:
        return []
    maximum = max(entry["score"] for entry in fused.values())
    ranked = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)
    results = []
    for entry in ranked:
        document = entry["document"]
        metadata = document["metadata"]
        if entry["queries"]:
            metadata["_retrieval_queries"] = entry["queries"]
            metadata["_retrieval_query_scores"] = entry["query_scores"]
        document["similarity"] = entry["score"] / maximum
        results.append(document)
    return results


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
    covered_queries: set[str] = set()
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

    def focused_selection_score(item, focused_query):
        metadata = item[2].get("metadata") or {}
        focused_relevance = (metadata.get("_retrieval_query_scores") or {}).get(
            focused_query, 0
        )
        focused_concepts = _query_concepts(focused_query)
        literal_match = _intent_score(focused_concepts, item[3])
        title_tokens = _tokens(" ".join(str(metadata.get(key, "")) for key in (
            "page_title", "section_title", "slug",
        )))
        title_match = _intent_score(focused_concepts, title_tokens)
        return (
            focused_relevance + literal_match + 0.5 * title_match,
            selection_score(item),
        )

    def select(item):
        selected.append(item)
        remaining.remove(item)
        metadata = item[2].get("metadata") or {}
        covered_queries.update(metadata.get("_retrieval_queries") or ())
        group = metadata.get(group_key)
        if group is not None:
            groups[group] += 1

    # Preserve one result for each focused clause. A candidate only receives
    # this marker after a separate keyword lookup for words the user supplied;
    # it is not tied to a predefined list of visualization features.
    focused_queries = list(dict.fromkeys(
        focused_query
        for item in ranked
        for focused_query in (item[2].get("metadata") or {}).get("_retrieval_queries", ())
    ))
    for focused_query in focused_queries:
        if len(selected) >= limit:
            break
        if focused_query in covered_queries:
            continue
        matching = [
            item for item in eligible_items()
            if focused_query in (item[2].get("metadata") or {}).get("_retrieval_queries", ())
        ]
        if matching:
            select(max(matching, key=lambda item: focused_selection_score(item, focused_query)))

    while remaining and len(selected) < limit:
        eligible = eligible_items()
        if not eligible:
            break
        select(max(eligible, key=selection_score))

    return [item[2] for item in selected]
