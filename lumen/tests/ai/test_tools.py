import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.viewable import Viewable

from lumen.ai.coordinator import Coordinator
from lumen.ai.tools import FunctionTool, MetadataLookup, define_tool
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
