"""Tests for Vega-Lite docs vector store retrieval."""

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

import datetime

from lumen.ai.config import PROMPTS_DIR
from lumen.ai.utils import render_template


def _base_context(doc_pages=None):
    """Create base context for template rendering."""
    class _FakeSource:
        dialect = None

    class _FakePipeline:
        table = "test_table"
        source = _FakeSource()

    class _FakeMemory(dict):
        pass

    memory = _FakeMemory({
        "pipeline": _FakePipeline(),
        "data": "col1,col2\n1,2\n3,4",
        "table": "test_table",
    })
    return dict(
        memory=memory,
        doc="VegaLiteView description",
        doc_pages=doc_pages,
        current_datetime=datetime.datetime(2026, 1, 1),
        actor_name="VegaLiteAgent",
    )


def test_vegalite_prompt_renders_doc_pages():
    """Test that doc_pages are rendered in the prompt when provided."""
    doc_pages = [
        {
            "text": "# Bar - Examples\n\nBar marks are useful for bar charts.",
            "metadata": {
                "page_title": "Bar",
                "section_title": "Examples",
                "category": "Marks",
                "url": "https://vega.github.io/vega-lite/docs/mark-bar.html",
            },
        },
        {
            "text": "# Aggregate - Introduction\n\nAggregate transform groups data.",
            "metadata": {
                "page_title": "Aggregate",
                "section_title": "Introduction",
                "category": "Transforms",
                "url": "https://vega.github.io/vega-lite/docs/transform-aggregate.html",
            },
        },
    ]

    rendered = render_template(
        PROMPTS_DIR / "VegaLiteAgent" / "main.jinja2",
        **_base_context(doc_pages=doc_pages),
    )

    # Check that docs block is rendered
    assert "Relevant Vega-Lite Documentation" in rendered
    assert "Examples — Bar" in rendered
    assert "Introduction — Aggregate" in rendered
    assert "https://vega.github.io/vega-lite/docs/mark-bar.html" in rendered
    assert "https://vega.github.io/vega-lite/docs/transform-aggregate.html" in rendered
    assert "Bar marks are useful for bar charts" in rendered
    assert "Aggregate transform groups data" in rendered


def test_vegalite_prompt_no_doc_pages():
    """Test that prompt renders without doc_pages (empty list or None)."""
    rendered = render_template(
        PROMPTS_DIR / "VegaLiteAgent" / "main.jinja2",
        **_base_context(doc_pages=[]),
    )

    # Docs block should not appear when empty
    assert "Relevant Vega-Lite Documentation" not in rendered


def test_vegalite_altair_prompt_renders_doc_pages():
    """Test that altair prompt also renders doc_pages."""
    doc_pages = [
        {
            "text": "# Line - Examples\n\nLine charts show trends over time.",
            "metadata": {
                "page_title": "Line",
                "section_title": "Examples",
                "category": "Marks",
                "url": "https://vega.github.io/vega-lite/docs/mark-line.html",
            },
        },
    ]

    rendered = render_template(
        PROMPTS_DIR / "VegaLiteAgent" / "main_altair.jinja2",
        **_base_context(doc_pages=doc_pages),
    )

    assert "Relevant Vega-Lite Documentation" in rendered
    assert "Examples — Line" in rendered
    assert "Line charts show trends over time" in rendered


def test_vegalite_revise_prompt_renders_doc_pages():
    """Test that revise_output prompt renders doc_pages."""
    doc_pages = [
        {
            "text": "# Config - Style Configuration\n\nStyling options for charts.",
            "metadata": {
                "page_title": "Config",
                "section_title": "Style Configuration",
                "category": "Config",
                "url": "https://vega.github.io/vega-lite/docs/config.html",
            },
        },
    ]

    context = _base_context(doc_pages=doc_pages)
    context["language"] = "yaml"
    context["numbered_text"] = "1: mark: bar\n2: encoding:\n3:   x: {field: A}"
    context["feedback"] = "Make it look better"

    rendered = render_template(
        PROMPTS_DIR / "VegaLiteAgent" / "revise_output.jinja2",
        **context,
    )

    assert "Relevant Vega-Lite Documentation" in rendered
    assert "Style Configuration — Config" in rendered


async def test_get_doc_pages_returns_results(llm):
    """Test that _get_doc_pages method returns results from vector store."""
    from pathlib import Path

    from lumen.ai.agents.vega_lite import VegaLiteAgent
    from lumen.ai.config import LUMEN_CACHE_DIR, VEGA_LITE_DOCS_OPENAI_DB_FILE
    from lumen.ai.vector_store import DuckDBVectorStore

    # Skip if vector store doesn't exist
    db_path = LUMEN_CACHE_DIR / VEGA_LITE_DOCS_OPENAI_DB_FILE
    if not Path(db_path).exists():
        pytest.skip(f"Vector store not found at {db_path}. Run create_vega_lite_docs_embeddings.py first.")

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    # Test with a natural language query
    results = await agent._get_doc_pages("bar chart")

    assert len(results) > 0
    assert len(results) <= 5  # Default top_k

    # Check structure
    for result in results:
        assert "text" in result
        assert "metadata" in result
        assert "page_title" in result["metadata"]
        assert "section_title" in result["metadata"]
        assert "category" in result["metadata"]
        assert "url" in result["metadata"]


async def test_get_doc_pages_bar_chart_query(llm):
    """Test that bar chart query retrieves bar mark docs."""
    from pathlib import Path

    from lumen.ai.agents.vega_lite import VegaLiteAgent
    from lumen.ai.config import LUMEN_CACHE_DIR, VEGA_LITE_DOCS_OPENAI_DB_FILE
    from lumen.ai.vector_store import DuckDBVectorStore

    db_path = LUMEN_CACHE_DIR / VEGA_LITE_DOCS_OPENAI_DB_FILE
    if not Path(db_path).exists():
        pytest.skip(f"Vector store not found at {db_path}")

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")
    results = await agent._get_doc_pages("bar chart")

    assert len(results) > 0
    # At least one result should be about bar marks
    bar_results = [r for r in results if "bar" in r["metadata"]["page_title"].lower()]
    assert len(bar_results) > 0, "Expected at least one bar mark result"


async def test_get_doc_pages_line_chart_query(llm):
    """Test that line chart query retrieves line mark docs."""
    from pathlib import Path

    from lumen.ai.agents.vega_lite import VegaLiteAgent
    from lumen.ai.config import LUMEN_CACHE_DIR, VEGA_LITE_DOCS_OPENAI_DB_FILE

    db_path = LUMEN_CACHE_DIR / VEGA_LITE_DOCS_OPENAI_DB_FILE
    if not Path(db_path).exists():
        pytest.skip(f"Vector store not found at {db_path}")

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")
    results = await agent._get_doc_pages("line chart time series")

    assert len(results) > 0
    # At least one result should be about line marks
    line_results = [r for r in results if "line" in r["metadata"]["page_title"].lower()]
    assert len(line_results) > 0, "Expected at least one line mark result"


async def test_get_doc_pages_aggregate_query(llm):
    """Test that aggregate query retrieves transform docs."""
    from pathlib import Path

    from lumen.ai.agents.vega_lite import VegaLiteAgent
    from lumen.ai.config import LUMEN_CACHE_DIR, VEGA_LITE_DOCS_OPENAI_DB_FILE

    db_path = LUMEN_CACHE_DIR / VEGA_LITE_DOCS_OPENAI_DB_FILE
    if not Path(db_path).exists():
        pytest.skip(f"Vector store not found at {db_path}")

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")
    results = await agent._get_doc_pages("aggregate group by")

    assert len(results) > 0
    # At least one result should be about aggregate transform
    agg_results = [r for r in results if "aggregate" in r["metadata"]["page_title"].lower()]
    assert len(agg_results) > 0, "Expected at least one aggregate transform result"


async def test_get_doc_pages_geographic_query(llm):
    """Test that geographic query retrieves projection docs."""
    from pathlib import Path

    from lumen.ai.agents.vega_lite import VegaLiteAgent
    from lumen.ai.config import LUMEN_CACHE_DIR, VEGA_LITE_DOCS_OPENAI_DB_FILE

    db_path = LUMEN_CACHE_DIR / VEGA_LITE_DOCS_OPENAI_DB_FILE
    if not Path(db_path).exists():
        pytest.skip(f"Vector store not found at {db_path}")

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")
    results = await agent._get_doc_pages("geographic map projection")

    assert len(results) > 0
    # At least one result should be about projection
    proj_results = [r for r in results if "projection" in r["metadata"]["page_title"].lower()]
    assert len(proj_results) > 0, "Expected at least one projection result"


async def test_get_doc_pages_empty_query(llm):
    """Test that empty query returns empty list."""
    from pathlib import Path

    from lumen.ai.agents.vega_lite import VegaLiteAgent
    from lumen.ai.config import LUMEN_CACHE_DIR, VEGA_LITE_DOCS_OPENAI_DB_FILE

    db_path = LUMEN_CACHE_DIR / VEGA_LITE_DOCS_OPENAI_DB_FILE
    if not Path(db_path).exists():
        pytest.skip(f"Vector store not found at {db_path}")

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")
    results = await agent._get_doc_pages("")

    assert results == []


async def test_get_doc_pages_natural_language(llm):
    """Test that natural language queries work."""
    from pathlib import Path

    from lumen.ai.agents.vega_lite import VegaLiteAgent
    from lumen.ai.config import LUMEN_CACHE_DIR, VEGA_LITE_DOCS_OPENAI_DB_FILE

    db_path = LUMEN_CACHE_DIR / VEGA_LITE_DOCS_OPENAI_DB_FILE
    if not Path(db_path).exists():
        pytest.skip(f"Vector store not found at {db_path}")

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    # Test various natural language queries
    queries = [
        "Can you create a bar chart of the Olympics data",
        "Show me a line chart of stock prices over time",
        "Make a pie chart showing the distribution by category",
        "Add tooltips to show values on hover",
    ]

    for query in queries:
        results = await agent._get_doc_pages(query)
        assert len(results) > 0, f"Expected results for query: {query}"
        # Verify structure
        for result in results:
            assert "text" in result
            assert "metadata" in result
            assert len(result["text"]) > 0
            assert len(result["metadata"]) > 0
