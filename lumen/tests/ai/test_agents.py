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

import lumen.ai.agents.sql as sql_agent_module

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


async def test_sql_agent_repairs_where_before_from(llm, duckdb_source, test_messages):
    agent = SQLAgent(llm=llm)

    context = {
        "source": duckdb_source,
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
    }
    SQLQueryWithTables = make_sql_model([(duckdb_source.name, "test_sql")])
    llm.set_responses([
        SQLQueryWithTables(
            query='SELECT * WHERE "A" > 0 FROM test_sql',
            table_slug="test_sql_filtered",
            tables=["test_sql"],
        ),
    ])

    out, out_context = await agent.respond(test_messages, context)

    assert len(out) == 1
    assert isinstance(out[0], SQLEditor)
    assert "FROM test_sql" in out[0].spec
    assert out[0].spec.index("FROM test_sql") < out[0].spec.index("WHERE")
    assert set(out_context) == {"data", "pipeline", "sql", "table", "source"}


async def test_validate_sql_logs_cleaning_failure_and_continues(monkeypatch, llm):
    agent = SQLAgent(llm=llm)

    class DummySource:
        dialect = "duckdb"

        def execute(self, _sql):
            return None

    class DummyStep:
        def __init__(self):
            self.messages = []

        def stream(self, message):
            self.messages.append(message)

    def always_fail_clean_sql(*_args, **_kwargs):
        raise ValueError("bad formatting")

    monkeypatch.setattr(sql_agent_module, "clean_sql", always_fail_clean_sql)
    step = DummyStep()

    result = await agent._validate_sql(
        context={},
        sql_query="SELECT 1",
        expr_slug="expr",
        source=DummySource(),
        messages=[],
        step=step,
        max_retries=1,
    )

    assert result == "SELECT 1"
    assert any("SQL cleaning failed" in message for message in step.messages)


async def test_validate_sql_raises_after_final_attempt_without_repair(monkeypatch, llm):
    agent = SQLAgent(llm=llm)

    class DummySource:
        dialect = "duckdb"

        def execute(self, _sql):
            raise RuntimeError("invalid sql")

    class DummyStep:
        def __init__(self):
            self.messages = []

        def stream(self, message):
            self.messages.append(message)

    monkeypatch.setattr(sql_agent_module, "clean_sql", lambda sql, *_args, **_kwargs: sql)
    monkeypatch.setattr(sql_agent_module, "repair_common_sql_clause_order", lambda sql: sql)
    step = DummyStep()

    with pytest.raises(RuntimeError, match="invalid sql"):
        await agent._validate_sql(
            context={},
            sql_query="SELECT * FROM missing",
            expr_slug="expr",
            source=DummySource(),
            messages=[],
            step=step,
            max_retries=1,
        )

    assert any("failed after 1 attempts" in message for message in step.messages)


async def test_validate_sql_retries_with_revise_after_non_final_failure(monkeypatch, llm):
    agent = SQLAgent(llm=llm)

    class DummySource:
        dialect = "duckdb"

        def __init__(self):
            self.calls = 0

        def execute(self, _sql):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("invalid sql")
            return None

    class DummyStep:
        def __init__(self):
            self.messages = []

        def stream(self, message):
            self.messages.append(message)

    async def fake_revise(*_args, **_kwargs):
        return "SELECT 1"

    monkeypatch.setattr(sql_agent_module, "clean_sql", lambda sql, *_args, **_kwargs: sql)
    monkeypatch.setattr(sql_agent_module, "repair_common_sql_clause_order", lambda sql: sql)
    monkeypatch.setattr(agent, "revise", fake_revise)
    source = DummySource()
    step = DummyStep()

    result = await agent._validate_sql(
        context={},
        sql_query="SELECT * FROM missing",
        expr_slug="expr",
        source=source,
        messages=[],
        step=step,
        max_retries=2,
    )

    assert result == "SELECT 1"
    assert any("failed (attempt 1/2)" in message for message in step.messages)

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
