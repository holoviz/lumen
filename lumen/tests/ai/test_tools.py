import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.memory import _Memory
from lumen.ai.tools import FunctionTool


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
    return a+b


def test_function_tool_constructor():
    tool = FunctionTool(add)

    assert tool.name == 'add'
    assert tool.purpose == 'Adds two integers'

async def test_function_tool_requires():
    tool = FunctionTool(add, requires=['a', 'b'])
    with tool.param.update(memory=_Memory()):
        tool.memory['a'] = 1
        tool.memory['b'] = 3
        result = await tool.respond([])
    assert result == 'add(a=1, b=3) returned: 4'

async def test_function_tool_provides():
    tool = FunctionTool(add, requires=['a', 'b'], provides=['c'])
    memory = _Memory()
    with tool.param.update(memory=memory):
        tool.memory['a'] = 1
        tool.memory['b'] = 3
        await tool.respond([])
    assert memory['c'] == 4
