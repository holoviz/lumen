import asyncio

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.viewable import Viewable
from panel_material_ui import (
    Button, Column, RadioButtonGroup, TextInput,
)

from lumen.ai.coordinator import Coordinator
from lumen.ai.schemas import DocumentChunk, Metaset, TableCatalogEntry
from lumen.ai.tools import FunctionTool, MetadataLookup, define_tool
from lumen.ai.tools.clarification_llm_tool import make_clarification_llm_tool
from lumen.ai.tools.document_llm_tools import make_document_vector_llm_tools
from lumen.ai.vector_store import NumpyVectorStore
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.sources.duckdb import DuckDBSource


def add(a: int, b: int) -> int:
    """
    Adds two integers

    Parameters
    ----------
    a : int
        Integer A
    b : int
        Integer B

    Returns
    -------
    int
        Result of addition
    """
    return a + b


@define_tool(
    requires=["a", "b"],
    provides=["c"],
    purpose="Add two numbers from context"
)
def add_from_context(a: int, b: int) -> int:
    """
    Adds two integers from context
    """
    return a + b


@define_tool(render_output=True)
def render_greeting() -> str:
    return "Hello from tool"


def test_function_tool_constructor():
    tool = FunctionTool(add)

    assert tool.name == "add"
    assert tool.purpose == "add: Adds two integers"


async def test_function_tool_requires():
    tool = FunctionTool(add, requires=["a", "b"])
    context = {"a": 1, "b": 3}
    views, out_context = await tool.respond([], context)
    assert views[0] == "add(a=1, b=3) returned: 4"


async def test_function_tool_define_tool_decorator():
    tool = FunctionTool(add_from_context)
    context = {"a": 2, "b": 5}
    views, out_context = await tool.respond([], context)
    assert tool.requires == ["a", "b"]
    assert tool.provides == ["c"]
    assert tool.purpose == "Add two numbers from context"
    assert out_context["c"] == 7
    assert views[0] == "add_from_context(a=2, b=5) returned: 7"


async def test_function_tool_define_tool_render_output():
    tool = FunctionTool(render_greeting)
    views, out_context = await tool.respond([], {})
    assert tool.render_output is True
    assert out_context == {}
    assert isinstance(views[0], Viewable)


async def test_function_tool_provides():
    tool = FunctionTool(add, requires=["a", "b"], provides=["c"])
    context = {"a": 1, "b": 3}
    views, out_context = await tool.respond([], context)
    assert out_context["c"] == 4


class TestVectorLookupToolUser:
    async def test_inherit_vector_store_all_tools_instantiated(self):
        # Create three separate vector stores
        vs1 = NumpyVectorStore()
        vs2 = NumpyVectorStore()
        vs3 = NumpyVectorStore()

        # Explicitly instantiate all tools with their own vector stores
        coordinator = Coordinator(
            tools=[
                MetadataLookup(vector_store=vs1),
            ]
        )

        # Get the tools
        table_tool = coordinator._tools["main"][0]

        # Verify they're the correct types
        assert isinstance(table_tool, MetadataLookup)

        # Each tool should have its own vector store that was passed in
        assert id(table_tool.vector_store) == id(vs1)


async def test_metadata_lookup_source_keeps_first_provenance():
    source = DuckDBSource(uri=":memory:", tables={"test_table": "SELECT 1 as id"})
    tool = MetadataLookup(vector_store=NumpyVectorStore())

    context_global = {"sources": [source], "provenance_chain": ["global"], "tables_metadata": {}}
    await tool.sync(context_global)

    context_exploration = {
        "sources": [source],
        "provenance_chain": ["global", "exploration_1"],
        "tables_metadata": {},
    }
    await tool.sync(context_exploration)

    entries = tool.vector_store.filter_by({"source": source.name, "type": "tables"})
    assert len(entries) == 1
    assert entries[0]["metadata"]["provenance_chain"] == ["global"]


async def test_metadata_lookup_exploration_finds_global_tables():
    """Tables stored at global provenance should be discoverable from an exploration context."""
    source = DuckDBSource(uri=":memory:", tables={"test_table": "SELECT 1 as id"})
    tool = MetadataLookup(vector_store=NumpyVectorStore())

    # Sync at global scope — tables get stored with provenance ['global']
    context = {
        "sources": [source],
        "provenance_chain": ["global"],
        "tables_metadata": {},
    }
    await tool.sync(context)

    # Query from an exploration context with longer provenance chain
    exploration_context = {
        "sources": [source],
        "provenance_chain": ["global", "exploration_123"],
        "tables_metadata": context["tables_metadata"],
    }
    messages = [{"role": "user", "content": "Show me test_table"}]
    _, outputs = await tool.respond(messages, exploration_context)
    metaset = outputs["metaset"]

    assert len(metaset.catalog) > 0, (
        "Exploration context should find global-scoped tables via provenance fallback"
    )
    slugs = list(metaset.catalog.keys())
    assert any("test_table" in s for s in slugs)


async def test_metadata_lookup_exploration_finds_global_tables_deeply_nested():
    """Provenance fallback should work even for deeply nested exploration chains."""
    source = DuckDBSource(uri=":memory:", tables={"sales": "SELECT 1 as revenue"})
    tool = MetadataLookup(vector_store=NumpyVectorStore())

    context = {
        "sources": [source],
        "provenance_chain": ["global"],
        "tables_metadata": {},
    }
    await tool.sync(context)

    # Three levels deep
    deep_context = {
        "sources": [source],
        "provenance_chain": ["global", "exploration_1", "exploration_2"],
        "tables_metadata": context["tables_metadata"],
    }
    messages = [{"role": "user", "content": "Show sales data"}]
    _, outputs = await tool.respond(messages, deep_context)
    metaset = outputs["metaset"]

    assert len(metaset.catalog) > 0, (
        "Deeply nested exploration should still find global-scoped tables"
    )


async def test_metadata_lookup_global_context_still_works():
    """Querying from global context should continue to work without fallback."""
    source = DuckDBSource(uri=":memory:", tables={"orders": "SELECT 1 as id"})
    tool = MetadataLookup(vector_store=NumpyVectorStore())

    context = {
        "sources": [source],
        "provenance_chain": ["global"],
        "tables_metadata": {},
    }
    await tool.sync(context)

    messages = [{"role": "user", "content": "What data is available?"}]
    _, outputs = await tool.respond(messages, context)
    metaset = outputs["metaset"]

    assert len(metaset.catalog) > 0
    slugs = list(metaset.catalog.keys())
    assert any("orders" in s for s in slugs)


async def test_document_llm_tools_list_and_search():
    store = NumpyVectorStore()
    await store.add(
        [{"text": "penguins like cold water", "metadata": {"filename": "zoo.md", "type": "document"}}]
    )
    ctx: dict = {"document_vector_store": store}
    tools = make_document_vector_llm_tools(ctx)
    assert len(tools) == 2
    listed = tools[0].function(limit=10, offset=0)
    assert "zoo.md" in listed
    searched = await tools[1].function(query="penguins", top_k=3, min_similarity=-1.0)
    assert "cold water" in searched


class _DummyInterface:

    def __init__(self):
        self.objects = []

    def send(self, obj, **kwargs):
        self.objects.append(obj)


async def test_clarification_tool_requires_confirm_click_for_options():
    context: dict = {}
    interface = _DummyInterface()
    tool = make_clarification_llm_tool(interface=interface, context=context)

    task = asyncio.create_task(
        tool.function("Pick one", options=["Option A", "Option B"])
    )

    await asyncio.sleep(0.05)
    assert interface.objects
    layout = interface.objects[-1]
    assert isinstance(layout, Column)
    widget = layout[1]
    confirm = layout[2]
    assert isinstance(widget, RadioButtonGroup)
    assert isinstance(confirm, Button)
    assert confirm.icon == "check"

    widget.value = "Option A"
    await asyncio.sleep(0.2)
    assert not task.done()

    confirm.clicks += 1
    result = await asyncio.wait_for(task, timeout=1)
    assert result == "User clarification: Option A"
    assert context["clarification"] == "Option A"


async def test_clarification_tool_requires_confirm_click_for_text():
    context: dict = {}
    interface = _DummyInterface()
    tool = make_clarification_llm_tool(interface=interface, context=context)

    task = asyncio.create_task(tool.function("Describe preference"))

    await asyncio.sleep(0.05)
    layout = interface.objects[-1]
    assert isinstance(layout, Column)
    widget = layout[1]
    confirm = layout[2]
    assert isinstance(widget, TextInput)
    assert isinstance(confirm, Button)
    assert confirm.icon == "check"

    widget.value = "   concise answers   "
    await asyncio.sleep(0.2)
    assert not task.done()

    confirm.clicks += 1
    result = await asyncio.wait_for(task, timeout=1)
    assert result == "User clarification: concise answers"
    assert context["clarification"] == "concise answers"
