from lumen.ai import ExplorerUI
from lumen.ai.agents import ChatAgent

ExplorerUI(
    title="Monty sandbox demo",
    default_agents=[ChatAgent],
    suggestions=["Use Python to calculate the sum of squares from 1 to 100."],
).servable()
