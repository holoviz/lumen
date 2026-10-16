from lumen.ai.retrieval import rerank_documents, targeted_document_queries


def candidate(
    title,
    similarity,
    *,
    slug=None,
    base_form=False,
    capabilities=(),
    examples=(),
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


def test_rerank_reserves_input_binding_for_a_slider_request():
    items = [
        candidate("Ranged Area", 0.95),
        candidate("Input Element Binding", 0.60, slug="parameter-bind"),
    ]

    result = rerank_documents("Create a chart controlled by a slider", items, limit=1)

    assert result[0]["metadata"]["section_title"] == "Input Element Binding"


def test_targeted_document_queries_follow_requested_capabilities():
    queries = targeted_document_queries(
        "Make a scatter plot with a brush selection, tooltip, and slider"
    )

    assert queries == [
        "Vega-Lite scatterplot circle mark",
        "Vega-Lite interval selection brush",
        "Vega-Lite tooltip channel",
        "Vega-Lite input element binding slider",
    ]


def test_rerank_covers_each_requested_interactive_capability():
    items = [
        candidate("Common Selection Properties", 0.99, slug="parameter-select"),
        candidate("Scatterplot with Circle", 0.70, slug="mark-circle"),
        candidate("Interval Selection", 0.65, slug="parameter-select"),
        candidate("Tooltip Channel", 0.60, slug="tooltip"),
    ]

    result = rerank_documents(
        "Make a scatter plot with a brush selection and tooltip", items, limit=3
    )

    titles = {item["metadata"]["section_title"] for item in result}
    assert titles == {"Scatterplot with Circle", "Interval Selection", "Tooltip Channel"}


def test_rerank_does_not_choose_disable_tooltips_as_tooltip_guidance():
    items = [
        candidate("Disable Tooltips", 0.99, slug="tooltip"),
        candidate("Tooltip Channel", 0.60, slug="tooltip"),
    ]

    result = rerank_documents("Add a tooltip", items, limit=1)

    assert result[0]["metadata"]["section_title"] == "Tooltip Channel"


def test_rerank_does_not_group_documents_without_a_slug():
    items = [candidate(f"Document {index}", 1 - index / 10) for index in range(4)]
    for item in items:
        item["metadata"]["slug"] = None

    result = rerank_documents("visualize the data", items, limit=4, max_per_group=1)

    assert len(result) == 4
