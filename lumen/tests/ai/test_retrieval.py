from lumen.ai.retrieval import (
    focused_document_queries, fuse_ranked_documents, rerank_documents,
)


def candidate(
    title,
    similarity,
    *,
    slug=None,
    base_form=False,
    capabilities=(),
    examples=(),
    retrieval_query=None,
):
    return {
        "text": f"Documentation for {title}",
        "similarity": similarity,
        "metadata": {
            "page_title": title,
            "section_title": title,
            "slug": slug or title.lower().replace(" ", "-"),
            "base_form": base_form,
            "capabilities": list(capabilities),
            "examples": list(examples),
            **({"_retrieval_queries": [retrieval_query]} if retrieval_query else {}),
        },
    }


def test_rerank_promotes_exact_intent_over_close_vector_score():
    items = [
        candidate("Line Chart", 0.91),
        candidate("Box Plot", 0.89, capabilities=("quartile", "whisker")),
        candidate("Axis Formatting", 0.88),
    ]

    result = rerank_documents("Make a box plot by species", items, limit=2)

    assert result[0]["metadata"]["section_title"] == "Box Plot"


def test_rerank_prefers_matching_base_form_to_specialized_variant():
    items = [
        candidate("Normalized Stacked Bar Chart", 0.94, slug="mark-bar"),
        candidate("Bar Chart", 0.91, slug="mark-bar", base_form=True),
        candidate("Sorting", 0.90, slug="encoding-sort"),
    ]

    result = rerank_documents("Create a bar chart", items, limit=2)

    assert result[0]["metadata"]["section_title"] == "Bar Chart"


def test_rerank_uses_capability_metadata():
    items = [
        candidate("Point Selection", 0.91, capabilities=("click",)),
        candidate("Interval Selection", 0.89, capabilities=("brush", "selection")),
    ]

    result = rerank_documents("Add a brush selection", items, limit=1)

    assert result[0]["metadata"]["section_title"] == "Interval Selection"


def test_rerank_caps_pages_and_keeps_diverse_context():
    items = [
        candidate("Bar Chart", 0.95, slug="mark-bar"),
        candidate("Stacked Bar", 0.94, slug="mark-bar"),
        candidate("Normalized Bar", 0.93, slug="mark-bar"),
        candidate("Aggregate Transform", 0.90, slug="transform-aggregate"),
    ]

    result = rerank_documents(
        "Bar chart of aggregate totals", items, limit=3, max_per_group=2
    )

    slugs = [item["metadata"]["slug"] for item in result]
    assert slugs.count("mark-bar") == 2
    assert "transform-aggregate" in slugs


def test_rerank_preserves_similarity_for_contentless_query():
    items = [candidate("First", 0.9), candidate("Second", 0.8)]

    result = rerank_documents("visualize the data", items, limit=2)

    assert [item["metadata"]["section_title"] for item in result] == ["First", "Second"]


def test_rerank_reserves_a_document_retrieved_for_a_focused_clause():
    items = [
        candidate("Overview", 0.95),
        candidate(
            "Widget Binding", 0.60, slug="widget-bind",
            retrieval_query="controlled dial",
        ),
    ]

    result = rerank_documents("Create a chart controlled by a dial", items, limit=1)

    assert result[0]["metadata"]["section_title"] == "Widget Binding"


def test_focused_document_queries_follow_user_clauses():
    queries = focused_document_queries(
        "Make a chart with a range control, threshold line, and data labels"
    )

    assert queries == [
        "chart",
        "range control",
        "threshold line",
        "labels",
    ]


def test_rank_fusion_combines_semantic_and_lexical_evidence():
    semantic = [
        {**candidate("Overview", 0.9), "id": 1},
        {**candidate("Input Binding", 0.8), "id": 2},
    ]
    lexical = [
        {**candidate("Input Binding", 4.2), "id": 2},
        {**candidate("Reference", 3.1), "id": 3},
    ]

    result = fuse_ranked_documents([
        (None, semantic),
        ("slider", lexical),
    ])

    assert result[0]["metadata"]["section_title"] == "Input Binding"
    assert result[0]["metadata"]["_retrieval_queries"] == ["slider"]


def test_rerank_covers_each_focused_clause():
    items = [
        candidate("Overview", 0.99, slug="overview"),
        candidate(
            "Range Control", 0.70, slug="range-control",
            retrieval_query="range control",
        ),
        candidate(
            "Threshold Line", 0.65, slug="threshold-line",
            retrieval_query="threshold line",
        ),
        candidate(
            "Data Labels", 0.60, slug="data-labels", retrieval_query="labels"
        ),
    ]

    result = rerank_documents(
        "Make a chart with a range control, threshold line, and data labels", items, limit=3
    )

    titles = {item["metadata"]["section_title"] for item in result}
    assert titles == {"Range Control", "Threshold Line", "Data Labels"}


def test_rerank_preserves_a_focused_result_over_a_better_broad_match():
    items = [
        candidate("Disable Widgets", 0.99, slug="widgets"),
        candidate("Widget Binding", 0.60, slug="widgets", retrieval_query="widget control"),
    ]

    result = rerank_documents("Add a widget control", items, limit=1)

    assert result[0]["metadata"]["section_title"] == "Widget Binding"


def test_rerank_does_not_group_documents_without_a_slug():
    items = [candidate(f"Document {index}", 1 - index / 10) for index in range(4)]
    for item in items:
        item["metadata"]["slug"] = None

    result = rerank_documents("visualize the data", items, limit=4, max_per_group=1)

    assert len(result) == 4
