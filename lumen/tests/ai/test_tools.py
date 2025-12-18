import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.coordinator import Coordinator
from lumen.ai.tools import FunctionTool, MetadataLookup
from lumen.ai.vector_store import NumpyVectorStore


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


def test_function_tool_constructor():
    tool = FunctionTool(add)

    assert tool.name == "add"
    assert tool.purpose == "add: Adds two integers"


async def test_function_tool_requires():
    tool = FunctionTool(add, requires=["a", "b"])
    context = {"a": 1, "b": 3}
    views, out_context = await tool.respond([], context)
    assert views[0] == "add(a=1, b=3) returned: 4"


async def test_function_tool_provides():
    tool = FunctionTool(add, requires=["a", "b"], provides=["c"])
    context = {"a": 1, "b": 3}
    views, out_context = await tool.respond([], context)
    assert out_context["c"] == 4


class TestVectorLookupToolUser:
    async def test_inherit_vector_store_uninstantiated(self):
        coordinator = Coordinator(tools=[MetadataLookup])
        table_tool = coordinator._tools["main"][0]
        table_vector_store = table_tool.vector_store
        assert isinstance(table_tool, MetadataLookup)
        iterative_table_tool = coordinator._tools["main"][1]
        assert isinstance(iterative_table_tool)
        assert id(table_vector_store) == id(iterative_table_tool.vector_store)

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
        iterative_table_tool = coordinator._tools["main"][1]

        # Verify they're the correct types
        assert isinstance(table_tool, MetadataLookup)

        # Each tool should have its own vector store that was passed in
        assert id(table_tool.vector_store) == id(vs1)
        assert id(iterative_table_tool.vector_store) == id(vs2)

        # Verify that MetadataLookup have different vector stores
        # since they were explicitly created with different instances
        assert id(table_tool.vector_store) != id(iterative_table_tool.vector_store)

    async def test_inherit_vector_store_instantiated_with_vector_store(self):
        # Create a shared vector store at the coordinator level
        shared_vs = NumpyVectorStore()

        # Create coordinator with shared vector store and uninstantiated tools
        coordinator = Coordinator(vector_store=shared_vs, tools=[MetadataLookup])

        # Get the tools
        table_tool = coordinator._tools["main"][0]
        iterative_table_tool = coordinator._tools["main"][1]

        # Verify they're the correct types
        assert isinstance(table_tool, MetadataLookup)

        # All tools should use the shared vector store from the coordinator
        assert id(table_tool.vector_store) == id(shared_vs)
        assert id(iterative_table_tool.vector_store) == id(shared_vs)
