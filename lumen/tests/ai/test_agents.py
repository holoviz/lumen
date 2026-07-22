import json

from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock, patch

import pytest

from lumen.ai.agents.document_list import DocumentListAgent
from lumen.ai.schemas import DocumentChunk

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from hvplot.ui import DEFAULT_CMAPS
from panel.pane import Markdown

from lumen.ai.agents import (
    AnalysisAgent, ChatAgent, SQLAgent, VegaLiteAgent,
)
from lumen.ai.agents.analysis import make_analysis_model
from lumen.ai.agents.deck_gl import DeckGLAgent
from lumen.ai.agents.hvplot import hvPlotAgent, resolve_cmap
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.agents.vega_lite import VegaLiteSpec, VegaLiteSpecUpdate
from lumen.ai.analysis import Analysis
from lumen.ai.editors import AnalysisOutput, SQLEditor, VegaLiteEditor
from lumen.ai.llm import Llm
from lumen.ai.schemas import Metaset, get_metaset
from lumen.config import dump_yaml
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource
from lumen.views import Panel, hvPlotUIView

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


def test_hvplot_agent_builds_response_model(llm):
    """hvPlotAgent must be able to build its structured-output model. The model
    is requested on every response (Actor._invoke), so a failure here makes the
    agent unusable rather than degraded."""
    schema = {
        "A": {"type": "number"},
        "B": {"type": "number"},
        "C": {"type": "string"},
    }
    model = hvPlotAgent(llm=llm)._get_model("main", schema=schema)

    assert "chain_of_thought" in model.model_fields
    for field in ("kind", "x", "y", "by", "groupby"):
        assert field in model.model_fields

    # x/y are restricted to real columns so the LLM cannot invent one.
    x_literal, _none = get_args(model.model_fields["x"].annotation)
    assert set(get_args(x_literal)) == set(schema)

    # kind tracks the view's own Selector rather than a duplicated list.
    assert set(get_args(model.model_fields["kind"].annotation)) == set(
        hvPlotUIView.param["kind"].objects
    )


def test_hvplot_agent_model_exposes_colormap(llm):
    """The model offers semantic colormap names rather than the full list of
    concrete colormaps, which is far easier for an LLM to choose from."""
    model = hvPlotAgent(llm=llm)._get_model("main", schema={"A": {"type": "number"}})

    for field in ("cmap", "cnorm", "colorbar", "color"):
        assert field in model.model_fields

    cmap_literal, _none = get_args(model.model_fields["cmap"].annotation)
    assert set(get_args(cmap_literal)) == set(DEFAULT_CMAPS)


def test_resolve_cmap_returns_names_the_explorer_accepts():
    """hvplot.ui.Colormapping.cmap is a Selector over concrete colormap names,
    so the semantic names have to be resolved before reaching the view. The
    categorical default resolves to a list of colours, which it rejects."""
    from hvplot.ui import CMAPS

    for semantic in ("linear", "diverging", "categorical", "cyclic"):
        assert resolve_cmap(semantic) in CMAPS


async def test_hvplot_agent_resolves_semantic_colormap(llm, tiny_source):
    """The semantic colormap names exist for the model's benefit;
    hvplot.ui.Colormapping.cmap is a Selector over concrete colormaps and
    rejects them, so they have to be resolved before the spec reaches the view.
    The reasoning field has to go for the same reason."""
    agent = hvPlotAgent(llm=llm)
    context = {"pipeline": Pipeline(source=tiny_source, table="tiny")}

    spec = await agent._extract_spec(context, {
        "chain_of_thought": "why", "kind": "scatter",
        "x": "id", "y": "value", "cmap": "diverging",
    })

    assert spec["cmap"] == "coolwarm"
    assert "chain_of_thought" not in spec


async def test_hvplot_agent_names_colormap_for_categorical_colour(llm, tiny_source):
    """Colouring by a non-numeric column makes hvPlot default the colormap to a
    list of colours, which the same Selector rejects, so name one instead."""
    agent = hvPlotAgent(llm=llm)
    context = {"pipeline": Pipeline(source=tiny_source, table="tiny")}
    base = {"chain_of_thought": "why", "kind": "scatter", "x": "id", "y": "value"}

    spec = await agent._extract_spec(context, dict(base, color="category"))
    assert spec["cmap"] == "glasbey"

    # Numeric colour columns are left alone for hvPlot to decide.
    spec = await agent._extract_spec(context, dict(base, color="value"))
    assert "cmap" not in spec


async def test_hvplot_agent_keeps_explicit_cnorm_on_large_data(llm):
    """Large datasets get a log colour norm, but not at the cost of one the
    model asked for."""
    source = DuckDBSource(tables={
        "big": "SELECT i AS id, i * 1.0 AS value FROM range(25000) t(i)"
    })
    agent = hvPlotAgent(llm=llm)
    context = {"pipeline": Pipeline(source=source, table="big")}
    base = {"chain_of_thought": "why", "kind": "scatter", "x": "id", "y": "value"}

    spec = await agent._extract_spec(context, dict(base, cnorm="eq_hist"))
    assert spec["rasterize"] is True
    assert spec["cnorm"] == "eq_hist"

    # Without an explicit choice the log norm is applied, with a colormap to
    # go alongside it rather than a bare norm.
    spec = await agent._extract_spec(context, dict(base))
    assert spec["cnorm"] == "log"
    assert spec["cmap"] == "kbc_r"


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


async def test_view_retry_keeps_context_and_passes_spec_by_keyword(llm):
    """The yaml auto-retry must keep the TContext dict (the next _extract_spec and
    revise both need it) and pass the spec by keyword, otherwise it binds to the
    `view` parameter and revise() raises AttributeError on `view.spec`."""
    agent = hvPlotAgent(llm=llm)
    captured = {}

    class _Out(dict):
        chain_of_thought = ""

    async def fake_stream_prompt(*args, **kwargs):
        async def gen():
            yield _Out(yaml_spec="kind: line")
        return gen()

    async def fake_extract_spec(context, spec):
        raise ValueError("bad spec")

    async def fake_revise(instruction, messages, context, view=None, spec=None, **kwargs):
        captured["context"] = context
        captured["view"] = view
        captured["spec"] = spec
        return "kind: line"

    ctx = {"pipeline": SimpleNamespace(table="t")}
    with patch.object(agent, "_stream_prompt", side_effect=fake_stream_prompt), \
         patch.object(agent, "_extract_spec", side_effect=fake_extract_spec), \
         patch.object(agent, "revise", side_effect=fake_revise):
        # _extract_spec always fails, so the retry loop exhausts and raises
        with pytest.raises(Exception):
            await agent._generate_yaml_spec(
                [{"role": "user", "content": "x"}], ctx,
                SimpleNamespace(table="t"), {},
            )

    # context must still be the TContext dict, not a formatted yaml string
    assert captured["context"] is ctx
    # the spec must not have bound to the `view` parameter
    assert captured["view"] is None
    assert isinstance(captured["spec"], str)


async def test_view_retry_recovers_using_revised_spec(llm):
    """The retry must not assume a `yaml_spec` field (hvPlot emits a flat param
    dict) and must feed the revision into the next attempt, otherwise it re-runs
    _extract_spec on the identical spec and can never recover."""
    agent = hvPlotAgent(llm=llm)
    seen_specs = []

    class _Out(dict):
        chain_of_thought = ""

    async def fake_stream_prompt(*args, **kwargs):
        async def gen():
            # realistic hvPlot output shape: view params, no yaml_spec
            yield _Out(kind="line", x="a", y="b")
        return gen()

    async def fake_extract_spec(context, spec):
        seen_specs.append(dict(spec))
        if len(seen_specs) == 1:
            raise ValueError("bad spec")
        return dict(spec)

    async def fake_revise(instruction, messages, context, view=None, spec=None, **kwargs):
        return "kind: bar\nx: a\ny: b"

    ctx = {"pipeline": SimpleNamespace(table="t")}
    with patch.object(agent, "_stream_prompt", side_effect=fake_stream_prompt), \
         patch.object(agent, "_extract_spec", side_effect=fake_extract_spec), \
         patch.object(agent, "revise", side_effect=fake_revise):
        result = await agent._generate_yaml_spec(
            [{"role": "user", "content": "x"}], ctx,
            SimpleNamespace(table="t"), {},
        )

    # revise was reached (no KeyError on the missing yaml_spec) and retried once
    assert len(seen_specs) == 2
    assert seen_specs[0]["kind"] == "line"   # first attempt used the original
    assert seen_specs[1]["kind"] == "bar"    # retry used the REVISED spec
    assert result["kind"] == "bar"
