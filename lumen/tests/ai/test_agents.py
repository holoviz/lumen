import json

from pathlib import Path

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai import memory
from lumen.ai.agents import (
    AnalystAgent, ChatAgent, SQLAgent, VegaLiteAgent,
)
from lumen.ai.llm import Llm
from lumen.ai.schemas import get_metaset
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource

try:
    from lumen.sources.duckdb import DuckDBSource

    pytestmark = pytest.mark.xdist_group("duckdb")
except ImportError:
    pytestmark = pytest.mark.skip(reason="Duckdb is not installed")

root = str(Path(__file__).parent.parent / "sources")


@pytest.fixture
def duckdb_source():
    duckdb_source = DuckDBSource(
        initializers=["INSTALL sqlite;", "LOAD sqlite;", f"SET home_directory='{root}';"],
        root=root,
        sql_expr="SELECT A, B, C, D::TIMESTAMP_NS AS D FROM {table}",
        tables={"test_sql": f"(SELECT * FROM READ_CSV('{root + '/test.csv'}'))"},
    )
    return duckdb_source


class MockLLM(Llm):
    """Mock LLM for testing"""

    _supports_stream = True
    _supports_model_stream = True

    def __init__(self, **params):
        super().__init__(model_kwargs={"default": {"model": "test"}}, **params)

    async def invoke(self, messages, system="", response_model=None, allow_partial=False, model_spec="default", **input_kwargs):
        return "Test response"

    async def stream(self, messages, system="", response_model=None, field=None, model_spec="default", **kwargs):
        if response_model:
            print(f"Response model: {response_model.__name__}")
            if response_model.__name__ == "Sql":
                yield response_model(
                    query=f"(SELECT * FROM READ_CSV('{root + '/test.csv'}'))",
                    expr_slug="test_query",
                    chain_of_thought="Sample SQL thought process"
                )
                return
            elif response_model.__name__ == "VegaLiteSpec":
                vega_spec = {
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
                    },
                }
                yield response_model(
                    chain_of_thought="Sample thought process",
                    json_spec=json.dumps(vega_spec)
                )
                return
        else:
            yield "Test response"  # Only yield this if no response_model is provided


    async def run_client(self, model_spec, messages, **kwargs):
        # Mock implementation
        return "Test client response"

    async def get_client(self, model_spec, response_model=None, **kwargs):
        # Mock implementation
        async def mock_client(**kwargs):
            return "Test client response"

        return mock_client

    async def initialize(self, log_level):
        # Do nothing for initialization
        pass


class TestSoloAgentRespond:

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory object with necessary data for agents"""
        return memory

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance"""
        return MockLLM()

    @pytest.fixture
    def test_messages(self):
        """Create test messages for agent respond method"""
        return [{"role": "user", "content": "Test message"}]

    async def test_chat_agent(self, mock_llm, mock_memory, test_messages):
        agent = ChatAgent(llm=mock_llm, memory=mock_memory)
        response = await agent.respond(test_messages)
        assert response is not None

    async def test_analyst_agent(self, mock_llm, mock_memory, duckdb_source, test_messages):
        mock_memory["source"] = duckdb_source
        mock_memory["pipeline"] = Pipeline(source=duckdb_source, table="test_sql")
        agent = AnalystAgent(llm=mock_llm, memory=mock_memory)
        response = await agent.respond(test_messages)
        assert response is not None

    async def test_sql_agent(self, mock_llm, mock_memory, duckdb_source, test_messages):
        """Test SQLAgent instantiation and respond"""
        mock_memory["source"] = duckdb_source
        mock_memory["sources"] = sources = {"duckdb": duckdb_source}
        mock_memory["sql_metaset"] = await get_metaset(sources, ["test_sql"])
        agent = SQLAgent(llm=mock_llm, memory=mock_memory)
        response = await agent.respond(test_messages)
        assert response is not None

    async def test_vegalite_agent(self, mock_llm, mock_memory, duckdb_source, test_messages):
        """Test VegaLiteAgent instantiation and respond"""
        # Set up necessary data in memory
        mock_memory["source"] = duckdb_source
        mock_memory["pipeline"] = Pipeline(source=duckdb_source, table="test_sql")
        mock_memory["table"] = "test_sql"
        mock_memory["sources"] = sources = {"duckdb": duckdb_source}
        mock_memory["sql_metaset"] = await get_metaset(sources, ["test_sql"])
        mock_memory["data"] = duckdb_source.get("test_sql")

        agent = VegaLiteAgent(llm=mock_llm, memory=mock_memory)

        response = await agent.respond(test_messages)
        assert response is not None
