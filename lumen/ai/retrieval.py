"""Utilities for reranking documentation retrieved by vector similarity."""

from __future__ import annotations

import re

from collections import Counter
from collections.abc import Iterable
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
    while remaining and len(selected) < limit:
        eligible = [
            item for item in remaining
            if (item[2].get("metadata") or {}).get(group_key) is None
            or groups[(item[2].get("metadata") or {}).get(group_key)] < max_per_group
        ]
        if not eligible:
            break

        def selection_score(item):
            redundancy = max(
                (_jaccard(item[3], chosen[3]) for chosen in selected),
                default=0,
            )
            return item[0] - diversity_weight * redundancy, -item[1]

        chosen = max(eligible, key=selection_score)
        selected.append(chosen)
        remaining.remove(chosen)
        group = (chosen[2].get("metadata") or {}).get(group_key)
        if group is not None:
            groups[group] += 1

    return [item[2] for item in selected]
