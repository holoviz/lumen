import json

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lumen.ai.agents.document_list import DocumentListAgent
from lumen.ai.schemas import DocumentChunk

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.pane import Markdown

from lumen.ai.agents import (
    AnalysisAgent, ChatAgent, SQLAgent, VegaLiteAgent,
)
from lumen.ai.agents.analysis import make_analysis_model
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.agents.vega_lite import VegaLiteSpec, VegaLiteSpecUpdate
from lumen.ai.analysis import Analysis
from lumen.ai.editors import AnalysisOutput, SQLEditor, VegaLiteEditor
from lumen.ai.llm import Llm
from lumen.ai.schemas import Metaset, get_metaset
from lumen.config import dump_yaml
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource
from lumen.views import Panel

root = str(Path(__file__).parent.parent / "sources")

@pytest.fixture
def duckdb_source():
    duckdb_source = DuckDBSource(
        initializers=["INSTALL sqlite;", "LOAD sqlite;", f"SET home_directory='{root}';"],
        root=root,
        tables={"test_sql": f"SELECT A, B, C, D::TIMESTAMP_NS AS D FROM READ_CSV('{root + '/test.csv'}')"},
    )
    return duckdb_source


@pytest.fixture
def test_messages():
    """Create test messages for agent respond method"""
    return [{"role": "user", "content": "Test message"}]

async def test_chat_agent(llm, test_messages):
    agent = ChatAgent(llm=llm)

    llm.set_responses([
        "Test Response"
    ])

    out, out_context = await agent.respond(test_messages, {})
    assert out[0].object == "Test Response"

async def test_chat_agent_with_data(llm, duckdb_source, test_messages):
    """Test ChatAgent in analyst mode (with data)"""
    agent = ChatAgent(llm=llm)
    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "data": [{"A": 1, "B": 2}],
        "sql": "SELECT * FROM test_sql"
    }
    llm.set_responses([
        "Analysis of data"
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 1
    assert out[0].object == "Analysis of data"
    assert out_context == {}

async def test_sql_agent(llm, duckdb_source, test_messages):
    agent = SQLAgent(llm=llm)

    context = {
        "source": duckdb_source,
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
    }
    # Create the proper SQL model with tables field for single source
    SQLQueryWithTables = make_sql_model([(duckdb_source.name, "test_sql")])
    llm.set_responses([
        SQLQueryWithTables(
            query="SELECT SUM(A) as A_sum FROM test_sql",
            table_slug="test_sql_agg",
            tables=["test_sql"]
        ),
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 1
    assert isinstance(out[0], SQLEditor)
    assert out[0].spec == (
        "SELECT\n"
        "  SUM(A) AS A_sum\n"
        "FROM test_sql"
    )
    assert set(out_context) == {"data", "pipeline", "sql", "table", "source"}

async def test_vegalite_agent(llm, duckdb_source, test_messages):
    """Test VegaLiteAgent instantiation and respond"""

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "table": "test_sql",
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
        "data": duckdb_source.get("test_sql")
    }

    spec = {
        "data": {
            "values": [
                {"A": 1, "B": 2, "C": 3, "D": "2023-01-01T00:00:00Z"},
                {"A": 4, "B": 5, "C": 6, "D": "2023-01-02T00:00:00Z"},
            ]
        },
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"},
        }
    }
    llm.set_responses([
        VegaLiteSpec(
            chain_of_thought="Test plot",
            yaml_spec=dump_yaml(spec),
            insufficient_context=False,
            insufficient_context_reason="none"
        ),
        VegaLiteSpecUpdate(
            chain_of_thought="All good",
            yaml_update=""
        )
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 1
    assert isinstance(out[0], VegaLiteEditor)
    assert out[0].spec == "$schema: https://vega.github.io/schema/vega-lite/v5.json\ndata:\n  values:\n  - A: 1\n    B: 2\n    C: 3\n    D: '2023-01-01T00:00:00Z'\n  - A: 4\n    B: 5\n    C: 6\n    D: '2023-01-02T00:00:00Z'\nencoding:\n  x:\n    field: A\n    type: quantitative\n  y:\n    field: B\n    type: quantitative\nheight: container\nmark: bar\nwidth: container\n"


async def test_analysis_agent(llm, duckdb_source, test_messages):

    class TestAnalysis(Analysis):

        def __call__(self, pipeline, context):
            return f"Test Analysis"

    agent = AnalysisAgent(
        analyses=[TestAnalysis.instance(name='foo'), TestAnalysis.instance(name='bar')],
        llm=llm
    )
    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql")
    }

    model = make_analysis_model([analysis.name for analysis in agent.analyses])
    llm.set_responses([
        model(analysis="bar")
    ])
    out, out_context = await agent.respond(test_messages, context)

    assert len(out) == 1
    assert isinstance(out[0], AnalysisOutput)
    assert isinstance(out[0].component, Panel)
    assert isinstance(out[0].component.object, Markdown)

    assert "view" in out_context
    assert out_context["view"]["type"] == "panel"
    assert out_context["view"]["object"]["object"] == "Test Analysis"



@pytest.mark.asyncio
class TestDocumentListAgentIntegration:
    """Tests for DocumentListAgent with metaset."""

    async def test_document_list_agent_with_metaset(self):
        """Test that DocumentListAgent works with metaset.docs."""
        # Create metaset with document chunks
        metaset = Metaset(
            query="test",
            catalog={},
            docs=[
                DocumentChunk(filename="readme.md", text="chunk 1", similarity=0.9),
                DocumentChunk(filename="readme.md", text="chunk 2", similarity=0.8),
                DocumentChunk(filename="schema.md", text="chunk 3", similarity=0.7),
            ]
        )
        
        context = {"metaset": metaset}
        
        # Test applies
        applies = await DocumentListAgent.applies(context)
        assert applies is True  # More than 1 unique document
        
        # Test _get_items
        agent = DocumentListAgent()
        items = agent._get_items(context)
        
        # Should return unique, sorted filenames
        assert items == {"Documents": ["readme.md", "schema.md"]}

    async def test_document_list_agent_no_docs(self):
        """Test that DocumentListAgent doesn't apply when no docs."""
        # Metaset without docs
        metaset = Metaset(query="test", catalog={}, docs=None)
        context = {"metaset": metaset}
        
        applies = await DocumentListAgent.applies(context)
        assert applies is False

    async def test_document_list_agent_single_doc(self):
        """Test that DocumentListAgent doesn't apply for single doc."""
        # Metaset with only one unique document
        metaset = Metaset(
            query="test",
            catalog={},
            docs=[DocumentChunk(filename="readme.md", text="chunk", similarity=0.9)]
        )
        context = {"metaset": metaset}

        applies = await DocumentListAgent.applies(context)
        assert applies is True


@pytest.mark.asyncio
class TestTemplateOverrides:
    """Tests for template_overrides on Agent classes."""

    async def test_subclass_override(self, llm):
        """Subclass with class-level template_overrides injects extra instructions."""

        class CustomAgent(ChatAgent):
            template_overrides = {
                "main": {"instructions": "{{ super() }}\nCustom subclass rule."}
            }

        agent = CustomAgent(llm=llm)
        messages = [{"role": "user", "content": "test"}]
        prompt = await agent._render_prompt("main", messages, {})
        assert "Custom subclass rule." in prompt


    async def test_instance_override(self, llm):
        """Instance-level template_overrides injects extra instructions."""
        agent = ChatAgent(
            llm=llm,
            template_overrides={
                "main": {"instructions": "Instance override rule."}
            }
        )
        messages = [{"role": "user", "content": "test"}]
        prompt = await agent._render_prompt("main", messages, {})
        assert "Instance override rule." in prompt

    async def test_super_preserves_parent_content(self, llm):
        """{{ super() }} keeps the parent block content when present."""

        class ExtendedChatAgent(ChatAgent):
            template_overrides = {
                "main": {"footer": "{{ super() }}\nFooter appended."}
            }

        agent = ExtendedChatAgent(llm=llm)
        messages = [{"role": "user", "content": "test"}]
        prompt = await agent._render_prompt("main", messages, {})
        assert "Footer appended." in prompt


class TestFixArcLabels:
    """Tests for VegaLiteAgent._fix_arc_labels post-processing."""

    @pytest.fixture
    def agent(self, llm):
        return VegaLiteAgent(llm=llm, code_execution="disabled")

    def _make_pie_spec(self, categories=None, outer_radius=None, with_text_layer=False):
        """Helper to build a pie chart spec for testing."""
        if categories is None:
            categories = [
                {"cat": "A", "val": 10},
                {"cat": "B", "val": 20},
                {"cat": "C", "val": 30},
            ]
        arc_mark = {"type": "arc"}
        if outer_radius is not None:
            arc_mark["outerRadius"] = outer_radius
        layers = [
            {
                "mark": arc_mark,
                "encoding": {
                    "theta": {"field": "val", "type": "quantitative", "stack": True},
                    "color": {"field": "cat", "type": "nominal"},
                },
            }
        ]
        if with_text_layer:
            layers.append({
                "mark": {"type": "text", "radiusOffset": 10},
                "encoding": {
                    "theta": {"field": "val", "type": "quantitative", "stack": True},
                    "text": {"field": "cat", "type": "nominal"},
                },
            })
        return {"data": {"values": categories}, "layer": layers}

    def test_adds_text_layer_with_correct_radius(self, agent):
        spec = self._make_pie_spec(outer_radius=80)
        result = agent._fix_arc_labels(spec)
        text_layers = [l for l in result["layer"] if agent._get_layer_mark_type(l) == "text"]
        assert len(text_layers) == 1
        assert text_layers[0]["mark"]["radius"] == 110  # 80 + 30

    def test_replaces_existing_text_layer(self, agent):
        spec = self._make_pie_spec(with_text_layer=True)
        result = agent._fix_arc_labels(spec)
        text_layers = [l for l in result["layer"] if agent._get_layer_mark_type(l) == "text"]
        assert len(text_layers) == 1
        # Should not have the old radiusOffset
        assert "radiusOffset" not in text_layers[0]["mark"]

    def test_preserves_non_arc_specs(self, agent):
        bar_spec = {"layer": [{"mark": "bar", "encoding": {"x": {"field": "a"}}}]}
        result = agent._fix_arc_labels(bar_spec)
        assert result == bar_spec

    def test_no_layers_returns_unchanged(self, agent):
        spec = {"mark": "arc", "encoding": {"theta": {"field": "v"}}}
        result = agent._fix_arc_labels(spec)
        assert result is spec

    def test_normalizes_string_arc_mark(self, agent):
        spec = {
            "data": {"values": [{"c": "X", "v": 1}]},
            "layer": [
                {
                    "mark": "arc",
                    "encoding": {
                        "theta": {"field": "v", "type": "quantitative"},
                        "color": {"field": "c", "type": "nominal"},
                    },
                }
            ],
        }
        result = agent._fix_arc_labels(spec)
        arc = [l for l in result["layer"] if agent._get_layer_mark_type(l) == "arc"][0]
        assert isinstance(arc["mark"], dict)
        assert arc["mark"]["outerRadius"] == 80

    def test_skips_labels_for_many_categories(self, agent):
        categories = [{"cat": chr(65 + i), "val": 10} for i in range(10)]
        spec = self._make_pie_spec(categories=categories)
        result = agent._fix_arc_labels(spec)
        text_layers = [l for l in result["layer"] if agent._get_layer_mark_type(l) == "text"]
        assert len(text_layers) == 0

    def test_adds_labels_for_few_categories(self, agent):
        categories = [{"cat": chr(65 + i), "val": 10} for i in range(5)]
        spec = self._make_pie_spec(categories=categories)
        result = agent._fix_arc_labels(spec)
        text_layers = [l for l in result["layer"] if agent._get_layer_mark_type(l) == "text"]
        assert len(text_layers) == 1

    def test_adds_labels_when_data_is_named(self, agent):
        """When data is referenced by name (not inline), labels are added."""
        spec = {
            "data": {"name": "my_table"},
            "layer": [
                {
                    "mark": {"type": "arc", "outerRadius": 80},
                    "encoding": {
                        "theta": {"field": "val", "type": "quantitative"},
                        "color": {"field": "cat", "type": "nominal"},
                    },
                }
            ],
        }
        result = agent._fix_arc_labels(spec)
        text_layers = [l for l in result["layer"] if agent._get_layer_mark_type(l) == "text"]
        assert len(text_layers) == 1

    def test_removes_legend_none(self, agent):
        spec = self._make_pie_spec()
        spec["layer"][0]["encoding"]["color"]["legend"] = None
        result = agent._fix_arc_labels(spec)
        arc = [l for l in result["layer"] if agent._get_layer_mark_type(l) == "arc"][0]
        assert "legend" not in arc["encoding"]["color"]

    def test_donut_preserves_inner_radius(self, agent):
        spec = self._make_pie_spec(outer_radius=80)
        spec["layer"][0]["mark"]["innerRadius"] = 40
        result = agent._fix_arc_labels(spec)
        arc = [l for l in result["layer"] if agent._get_layer_mark_type(l) == "arc"][0]
        assert arc["mark"]["innerRadius"] == 40
