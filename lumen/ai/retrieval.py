"""Query preparation, rank fusion, and documentation selection."""

from __future__ import annotations

import re

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
RETRIEVAL_STOP_WORDS = frozenset({
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


def doc_title(metadata: dict) -> str:
    return " ".join(str(metadata.get(key, "")) for key in (
        "page_title", "section_title", "slug",
    ))


def doc_key(document: dict) -> tuple:
    metadata = document.get("metadata") or {}
    if document.get("id") is not None:
        return ("id", document["id"])
    return ("document", metadata.get("slug"), metadata.get("section_title"), document.get("text", ""))


def _intent_score(query_terms: set[str], candidate_tokens: set[str]) -> float:
    return len(query_terms & candidate_tokens) / len(query_terms) if query_terms else 0


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
        terms = [term for term in _TOKEN_RE.findall(clause) if term not in RETRIEVAL_STOP_WORDS]
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


def documentation_coverage(queries: list[str], pages: list[dict]) -> dict[str, dict]:
    """Return the first section with positive textual evidence for each covered clause."""
    evidence = []
    for page in pages:
        metadata = page.get("metadata") or {}
        heading = metadata.get("section_title") or ""
        # A setting or a section describing removal is weak evidence for
        # implementing the operation itself.
        if heading.startswith("`") or _tokens(heading) & {"disable", "remove", "suppress"}:
            continue
        evidence.append((page, _tokens(" ".join((
            metadata.get("page_title") or "", heading,
            metadata.get("description") or "", page["text"],
        )))))
    coverage = {}
    for query in queries:
        query_tokens = _tokens(query)
        for page, terms in evidence:
            if query_tokens <= terms:
                coverage[query] = page
                break
    return coverage


def documentation_shortlist(candidates: list[dict], preferred: list[dict]) -> list[dict]:
    """Include alternate sections from the pages represented in the first pass."""
    shortlist = preferred[:16]
    groups = defaultdict(list)
    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        if metadata.get("slug") and not str(metadata.get("section_title", "")).startswith("`"):
            groups[metadata["slug"]].append(candidate)

    slugs = list(dict.fromkeys(
        slug for doc in shortlist if (slug := (doc.get("metadata") or {}).get("slug"))
    ))[:3]
    alternatives = [[doc for doc in groups[slug] if doc not in shortlist][:4] for slug in slugs]
    # Interleave pages so one long reference page cannot take all the slots.
    for row in zip_longest(*alternatives):
        shortlist.extend(doc for doc in row if doc is not None)
    return shortlist[:16]


def fuse_ranked_documents(
    searches: list[tuple[str | None, list[dict]]],
    *,
    rank_constant: int = 60,
) -> list[dict]:
    """Combine incomparable semantic and lexical rankings using RRF.

    Focused lexical rankings carry the user clause into the result metadata
    for coverage selection. Other primary rankings use ``None``.
    """
    fused: dict[tuple, dict] = {}
    for query, documents in searches:
        for rank, document in enumerate(documents, start=1):
            metadata = document.get("metadata") or {}
            key = doc_key(document)
            if key not in fused:
                fused[key] = {
                    "document": {
                        **document,
                        "metadata": {**metadata},
                    },
                    "score": 0.0,
                    "query_scores": {},
                }
            entry = fused[key]
            contribution = 1 / (rank_constant + rank)
            entry["score"] += contribution
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
        query_scores = entry["query_scores"]
        if query_scores:
            metadata["_retrieval_query_scores"] = query_scores
        document["fusion_score"] = entry["score"] / maximum
        results.append(document)
    return results


@dataclass
class _RankedDocument:
    score: float
    position: int
    document: dict
    tokens: set[str]
    title_tokens: set[str]
    group: Any
    query_scores: dict[str, float]


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
    """Select relevant sections using fused rank, lexical evidence and diversity.

    Give fused rank the largest weight. Token and title overlap break close
    rankings, with smaller bonuses for base forms and runnable examples. These
    weights are heuristic and apply to relative ranks within this request.
    """
    if limit <= 0 or not candidates:
        return []

    query_terms = _tokens(query) - RETRIEVAL_STOP_WORDS
    ranked = []
    for position, candidate in enumerate(candidates):
        metadata = candidate.get("metadata") or {}
        tokens = _tokens(_candidate_text(candidate))
        relevance = float(candidate.get("fusion_score", candidate.get("similarity", 0)))
        intent = _intent_score(query_terms, tokens)
        title_tokens = _tokens(doc_title(metadata))
        title_match = _intent_score(query_terms, title_tokens)
        canonical = 0.08 if reserve_base_form and metadata.get("base_form") and intent else 0
        example = 0.03 if metadata.get("examples") else 0
        score = 0.52 * relevance + 0.30 * intent + 0.10 * title_match + canonical + example
        ranked.append(_RankedDocument(
            score, position, candidate, tokens, title_tokens, metadata.get(group_key),
            metadata.get("_retrieval_query_scores") or {},
        ))

    selected: list[_RankedDocument] = []
    groups: Counter = Counter()
    covered_queries: set[str] = set()
    remaining = ranked.copy()

    def eligible_items():
        return [
            item for item in remaining
            if item.group is None or groups[item.group] < max_per_group
        ]

    def selection_score(item):
        redundancy = max(
            (_jaccard(item.tokens, chosen.tokens) for chosen in selected),
            default=0,
        )
        return item.score - diversity_weight * redundancy, -item.position

    def focused_selection_score(item, focused_query):
        focused_relevance = item.query_scores.get(focused_query, 0)
        focused_terms = _tokens(focused_query) - RETRIEVAL_STOP_WORDS
        literal_match = _intent_score(focused_terms, item.tokens)
        title_match = _intent_score(focused_terms, item.title_tokens)
        return (
            focused_relevance + literal_match + 0.5 * title_match,
            selection_score(item),
        )

    def select(item):
        selected.append(item)
        remaining.remove(item)
        covered_queries.update(item.query_scores)
        if item.group is not None:
            groups[item.group] += 1

    # Reserve space for clauses that a focused keyword search matched.
    focused_queries = list(dict.fromkeys(
        focused_query
        for item in ranked
        for focused_query in item.query_scores
    ))
    for focused_query in focused_queries:
        if len(selected) >= limit:
            break
        if focused_query in covered_queries:
            continue
        matching = [
            item for item in eligible_items()
            if focused_query in item.query_scores
        ]
        if matching:
            select(max(matching, key=lambda item: focused_selection_score(item, focused_query)))

    while remaining and len(selected) < limit:
        eligible = eligible_items()
        if not eligible:
            break
        select(max(eligible, key=selection_score))

    return [item.document for item in selected]
