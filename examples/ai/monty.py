from lumen.ai import ExplorerUI
from lumen.ai.agents import ChatAgent
from lumen.ai.tools import make_monty_llm_tool

ExplorerUI(
    title="Monty sandbox demo",
    default_agents=[ChatAgent],
    llm_tools=[make_monty_llm_tool()],
    suggestions=["Use Python to calculate the sum of squares from 1 to 100."],
).servable()
