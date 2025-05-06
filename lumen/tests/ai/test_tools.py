import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.coordinator import Coordinator
from lumen.ai.memory import _Memory
from lumen.ai.tools import (
    DocumentLookup, FunctionTool, IterativeTableLookup, TableLookup,
)
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
    assert tool.purpose == "Adds two integers"


async def test_function_tool_requires():
    tool = FunctionTool(add, requires=["a", "b"])
    with tool.param.update(memory=_Memory()):
        tool.memory["a"] = 1
        tool.memory["b"] = 3
        result = await tool.respond([])
    assert result == "add(a=1, b=3) returned: 4"


async def test_function_tool_provides():
    tool = FunctionTool(add, requires=["a", "b"], provides=["c"])
    memory = _Memory()
    with tool.param.update(memory=memory):
        tool.memory["a"] = 1
        tool.memory["b"] = 3
        await tool.respond([])
    assert memory["c"] == 4


class TestVectorLookupToolUser:
    async def test_inherit_vector_store_uninstantiated(self):
        coordinator = Coordinator(tools=[TableLookup, IterativeTableLookup, DocumentLookup])
        table_tool = coordinator._tools["main"][0]
        table_vector_store = table_tool.vector_store
        assert isinstance(table_tool, TableLookup)
        iterative_table_tool = coordinator._tools["main"][1]
        assert isinstance(iterative_table_tool, IterativeTableLookup)
        assert id(table_vector_store) == id(iterative_table_tool.vector_store)
        document_tool = coordinator._tools["main"][2]
        assert isinstance(document_tool, DocumentLookup)
        assert id(table_vector_store) != id(document_tool.vector_store)

    async def test_inherit_vector_store_all_tools_instantiated(self):
        # Create three separate vector stores
        vs1 = NumpyVectorStore()
        vs2 = NumpyVectorStore()
        vs3 = NumpyVectorStore()

        # Explicitly instantiate all tools with their own vector stores
        coordinator = Coordinator(
            tools=[
                TableLookup(vector_store=vs1),
                IterativeTableLookup(vector_store=vs2),
                DocumentLookup(vector_store=vs3),
            ]
        )

        # Get the tools
        table_tool = coordinator._tools["main"][0]
        iterative_table_tool = coordinator._tools["main"][1]
        document_tool = coordinator._tools["main"][2]

        # Verify they're the correct types
        assert isinstance(table_tool, TableLookup)
        assert isinstance(iterative_table_tool, IterativeTableLookup)
        assert isinstance(document_tool, DocumentLookup)

        # Each tool should have its own vector store that was passed in
        assert id(table_tool.vector_store) == id(vs1)
        assert id(iterative_table_tool.vector_store) == id(vs2)
        assert id(document_tool.vector_store) == id(vs3)

        # Verify that TableLookup and IterativeTableLookup have different vector stores
        # since they were explicitly created with different instances
        assert id(table_tool.vector_store) != id(iterative_table_tool.vector_store)

    async def test_inherit_vector_store_instantiated_with_vector_store(self):
        # Create a shared vector store at the coordinator level
        shared_vs = NumpyVectorStore()

        # Create coordinator with shared vector store and uninstantiated tools
        coordinator = Coordinator(vector_store=shared_vs, tools=[TableLookup, IterativeTableLookup, DocumentLookup])

        # Get the tools
        table_tool = coordinator._tools["main"][0]
        iterative_table_tool = coordinator._tools["main"][1]
        document_tool = coordinator._tools["main"][2]

        # Verify they're the correct types
        assert isinstance(table_tool, TableLookup)
        assert isinstance(iterative_table_tool, IterativeTableLookup)
        assert isinstance(document_tool, DocumentLookup)

        # All tools should use the shared vector store from the coordinator
        assert id(table_tool.vector_store) == id(shared_vs)
        assert id(iterative_table_tool.vector_store) == id(shared_vs)
        assert id(document_tool.vector_store) == id(shared_vs)

    async def test_inherit_vector_store_some_tools_instantiated(self):
        # Create a vector store for the first tool
        table_vs = NumpyVectorStore()

        # Mix of instantiated and uninstantiated tools
        coordinator = Coordinator(
            tools=[
                TableLookup(vector_store=table_vs),  # Explicitly instantiated with vector_store
                IterativeTableLookup,  # Uninstantiated, should inherit from TableLookup
                DocumentLookup(),  # Instantiated without vector_store
            ]
        )

        # Get the tools
        table_tool = coordinator._tools["main"][0]
        iterative_table_tool = coordinator._tools["main"][1]
        document_tool = coordinator._tools["main"][2]

        # Verify they're the correct types
        assert isinstance(table_tool, TableLookup)
        assert isinstance(iterative_table_tool, IterativeTableLookup)
        assert isinstance(document_tool, DocumentLookup)

        # The first tool should use the explicitly provided vector store
        assert id(table_tool.vector_store) == id(table_vs)

        # IterativeTableLookup should inherit vector store from TableLookup
        # because they have the same _item_type_name
        assert id(iterative_table_tool.vector_store) == id(table_tool.vector_store)

        # DocumentLookup should have a new vector store because:
        # 1. It was instantiated without one
        # 2. It has a different _item_type_name than TableLookup/IterativeTableLookup
        assert id(document_tool.vector_store) != id(table_tool.vector_store)
