import json

from pathlib import Path
from types import SimpleNamespace
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
from lumen.ai.agents.deck_gl import DeckGLAgent
from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.agents.vega_lite import (
    AltairChartSpec, AltairSpec, ChartSpec, VegaLiteSpec, VegaLiteSpecUpdate,
)
from lumen.ai.analysis import Analysis
from lumen.ai.editors import (
    AnalysisOutput, MultiChartEditor, SQLEditor, VegaLiteEditor,
)
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
    assert out_context == {'chat': 'Analysis of data'}

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
            charts=[ChartSpec(title="A vs B", yaml_spec=dump_yaml(spec))],
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


async def test_vegalite_agent_multiple(llm, duckdb_source, test_messages):
    """Multiple charts: a view-only 'All' overview plus one editable editor per chart."""

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "table": "test_sql",
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
        "data": duckdb_source.get("test_sql")
    }

    spec_ab = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"},
        }
    }
    spec_ac = {
        "mark": "line",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "C", "type": "quantitative"},
        }
    }
    llm.set_responses([
        VegaLiteSpec(
            chain_of_thought="Two distinct relationships",
            charts=[
                ChartSpec(title="A vs B", yaml_spec=dump_yaml(spec_ab)),
                ChartSpec(title="A vs C", yaml_spec=dump_yaml(spec_ac)),
            ],
            insufficient_context=False,
            insufficient_context_reason="none"
        ),
        # One polish response per editable editor.
        VegaLiteSpecUpdate(chain_of_thought="ok", yaml_update=""),
        VegaLiteSpecUpdate(chain_of_thought="ok", yaml_update=""),
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 3
    # One editable VegaLiteEditor per chart first...
    assert isinstance(out[0], VegaLiteEditor)
    assert isinstance(out[1], VegaLiteEditor)
    assert "field: B" in out[0].spec
    assert "field: C" in out[1].spec
    # ...then the view-only "All" overview as the last tab (no code editor).
    assert isinstance(out[-1], MultiChartEditor)
    assert out[-1].title == "All"
    assert out[-1]._render_editor() is None
    assert isinstance(out[-1].component, Panel)
    assert len(out[-1].component.object) == 2  # both plots stacked in the overview


async def test_vegalite_agent_skips_unparseable_chart(llm, duckdb_source, test_messages):
    """A chart whose spec fails to parse is skipped; valid charts still render."""

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "table": "test_sql",
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
        "data": duckdb_source.get("test_sql")
    }

    good = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"},
        }
    }
    llm.set_responses([
        VegaLiteSpec(
            chain_of_thought="one good, one malformed",
            charts=[
                ChartSpec(title="good", yaml_spec=dump_yaml(good)),
                # Unbalanced braces -> YAML parse error.
                ChartSpec(title="bad", yaml_spec="mark: bar\nencoding:\n  x: {field: A, type: quantitative}}"),
            ],
            insufficient_context=False,
            insufficient_context_reason="none"
        ),
        VegaLiteSpecUpdate(chain_of_thought="ok", yaml_update=""),
    ])
    out, out_context = await agent.respond(test_messages, context)
    # Only the valid chart survives -> a single editor, no "All" overview.
    assert len(out) == 1
    assert isinstance(out[0], VegaLiteEditor)
    assert "field: B" in out[0].spec


async def test_vegalite_agent_altair_multiple(llm, duckdb_source, test_messages):
    """VegaLiteAgent Altair (code) mode returns one editor per generated chart."""

    agent = VegaLiteAgent(llm=llm, code_execution="allow")

    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "table": "test_sql",
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
        "data": duckdb_source.get("test_sql")
    }

    chart_ab = SimpleNamespace(to_dict=lambda: {
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"},
        },
    })
    chart_ac = SimpleNamespace(to_dict=lambda: {
        "mark": "line",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "C", "type": "quantitative"},
        },
    })
    # Bypass real code execution; return a chart object per generated code block.
    agent._execute_code = AsyncMock(side_effect=[chart_ab, chart_ac])

    llm.set_responses([
        AltairSpec(
            chain_of_thought="Two independent charts",
            charts=[
                AltairChartSpec(
                    title="A vs B",
                    code="chart = alt.Chart(df).mark_bar().encode(x='A:Q', y='B:Q')"
                ),
                AltairChartSpec(
                    title="A vs C",
                    code="chart = alt.Chart(df).mark_line().encode(x='A:Q', y='C:Q')"
                ),
            ],
        ),
    ])
    out, out_context = await agent.respond(test_messages, context)
    # One editable editor per chart, then a view-only "All" overview last.
    assert len(out) == 3
    assert isinstance(out[0], VegaLiteEditor)
    assert isinstance(out[1], VegaLiteEditor)
    assert "field: B" in out[0].spec
    assert "field: C" in out[1].spec
    assert isinstance(out[-1], MultiChartEditor)
    assert out[-1].title == "All"


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


def test_map_agents_route_geometry_columns():
    """hvPlot and DeckGL agents advertise a geometry-column condition so the
    coordinator routes GeoDataFrame data to a map-capable view."""
    assert any("geometry" in c.lower() for c in hvPlotAgent.conditions)
    assert any("geometry" in c.lower() for c in DeckGLAgent.conditions)


def test_sqlagent_active_filters_describes_conditions():
    """SQLAgent._active_filters turns the interactive slider filters into
    WHERE-style conditions (skipping inactive/full-range ones) so a follow-up
    query can preserve the subset."""
    class _Filter:
        def __init__(self, field, query):
            self.field, self._query = field, query

        @property
        def query(self):
            return self._query

    pipeline = SimpleNamespace(filters=[
        _Filter("game_year", (2000, 2016)),
        _Filter("game_season", ["Summer", "Winter"]),
        _Filter("game_location", "Japan"),
        _Filter("game_slug", None),  # full range / nothing selected -> skipped
    ])
    assert SQLAgent._active_filters(pipeline) == [
        "game_year between 2000 and 2016",
        "game_season in ('Summer', 'Winter')",
        "game_location = 'Japan'",
    ]
    assert SQLAgent._active_filters(None) is None
    assert SQLAgent._active_filters(SimpleNamespace(filters=[])) is None


async def test_sqlagent_prompt_surfaces_active_filters(llm):
    """Active exploration filters are surfaced in the SQL agent's prompt so a
    follow-up query keeps the subset."""
    agent = SQLAgent(llm=llm)
    messages = [{"role": "user", "content": "top 5 rows"}]
    prompt = await agent._render_prompt(
        "main", messages, {},
        dialect="duckdb", is_final_step=True, step_number=1, current_step="",
        sql_query_history={}, current_iteration=1, sql_plan_context=None,
        errors=None, discovery_context=None, source_names=["src"],
        active_filters=["game_year between 2000 and 2016", "game_season in ('Summer')"],
    )
    assert "Active exploration filters" in prompt
    assert "game_year between 2000 and 2016" in prompt
    assert "game_season in ('Summer')" in prompt
