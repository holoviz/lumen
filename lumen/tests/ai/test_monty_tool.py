from importlib.util import find_spec

import pytest

pydantic_monty = pytest.importorskip("pydantic_monty")

from lumen.ai.actor import _expand_llm_tool_entries
from lumen.ai.agents.chat import ChatAgent
from lumen.ai.schemas import Metaset
from lumen.ai.tools import make_monty_llm_tool
from lumen.ai.tools.base import FunctionTool


async def test_monty_tool_runs_code_and_captures_print():
    tool = make_monty_llm_tool()
    assert await tool.function("print('hello')\n2 + 3") == "hello\n5"


async def test_monty_tool_does_not_share_state():
    tool = make_monty_llm_tool()
    await tool.function("secret = 42")
    result = await tool.function("secret")
    assert "Error:" in result
    assert "secret" in result


async def test_monty_tool_rejects_host_access():
    tool = make_monty_llm_tool()
    result = await tool.function("open('/etc/passwd').read()")
    assert "Error:" in result
    assert "Permission" in result


async def test_monty_tool_rejects_third_party_imports():
    tool = make_monty_llm_tool()
    result = await tool.function("import pandas")
    assert "Error:" in result
    assert "pandas" in result


async def test_chat_prompt_with_metaset_without_visible_slugs():
    agent = ChatAgent()
    prompt = await agent._render_prompt(
        "main", [{"role": "user", "content": "Calculate 2 + 3 with Python"}],
        {"metaset": Metaset(query="calculation", catalog={})},
    )
    assert "No Data Loaded" not in prompt


def test_monty_tool_factory_registers_when_installed():
    assert find_spec("pydantic_monty") is not None
    tools = _expand_llm_tool_entries([make_monty_llm_tool], {})
    assert len(tools) == 1
    assert isinstance(tools[0], FunctionTool)


def test_monty_tool_factory_skips_when_not_installed(monkeypatch):
    monkeypatch.setattr("lumen.ai.tools.monty.find_spec", lambda _: None)
    assert _expand_llm_tool_entries([make_monty_llm_tool], {}) == []
