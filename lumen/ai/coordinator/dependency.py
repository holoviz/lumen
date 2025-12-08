from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING, Any

import param

from .agents import Agent
from .base import Coordinator, Plan
from .config import PROMPTS_DIR
from .context import TContext
from .llm import Message
from .models import make_agent_model
from .report import ActorTask
from .tools import Tool
from .utils import log_debug

if TYPE_CHECKING:
    pass


class DependencyResolver(Coordinator):
    """
    DependencyResolver is a type of Coordinator that chooses the agent
    to answer the query and then recursively resolves all the
    information required for that agent until the answer is available.
    """

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "DependencyResolver" / "main.jinja2",
                "response_model": make_agent_model,
            },
        },
    )

    async def _choose_agent(
        self, messages: list[Message], context: TContext, agents: list[Agent] | None = None, primary: bool = False, unmet_dependencies: tuple[str] | None = None
    ):
        if agents is None:
            agents = self.agents
        applies = await asyncio.gather(*[agent.applies(context) for agent in agents])
        agents = [agent for agent, aapply in zip(agents, applies, strict=False) if aapply]
        applies = await asyncio.gather(*[tool.applies(context) for tool in self._tools["main"]])
        tools = [tool for tool, tapply in zip(self._tools["main"], applies, strict=False) if tapply]

        agent_names = tuple(sagent.name[:-5] for sagent in agents) + tuple(tool.name for tool in tools)
        agent_model = self._get_model("main", agent_names=agent_names, primary=primary)
        if len(agent_names) == 0:
            raise ValueError("No agents available to choose from.")
        if len(agent_names) == 1:
            return agent_model(agent=agent_names[0], chain_of_thought="")
        system = await self._render_prompt("main", messages, context, agents=agents, tools=tools, primary=primary, unmet_dependencies=unmet_dependencies)
        return await self._fill_model(messages, system, agent_model)

    async def _compute_plan(
        self, messages: list[Message], context: TContext, agents: dict[str, Agent], tools: dict[str, Tool], pre_plan_output: dict[str, Any]
    ) -> Plan | None:
        if len(agents) == 1:
            agent = next(iter(agents.values()))
        else:
            agent = None
            with self._add_step(title="Selecting primary agent...", user="Assistant") as step:
                try:
                    output = await self._choose_agent(messages, context, self.agents, primary=True)
                except Exception as e:
                    if self.interface.callback_exception not in ("raise", "verbose"):
                        step.failed_title = "Failed to select main agent..."
                        step.stream(str(e), replace=True)
                    else:
                        raise e
                step.stream(output.chain_of_thought, replace=True)
                step.success_title = f"Selected {output.agent_or_tool}"
                if output.agent_or_tool in tools:
                    agent = tools[output.agent_or_tool]
                else:
                    agent = agents[output.agent_or_tool]

        if agent is None:
            return None

        cot = output.chain_of_thought
        subagent = agent
        tasks = []
        while unmet_dependencies := tuple(r for r in await subagent.requirements(messages) if r not in context):
            with self._add_step(title="Resolving dependencies...", user="Assistant") as step:
                step.stream(f"Found {len(unmet_dependencies)} unmet dependencies: {', '.join(unmet_dependencies)}")
                log_debug(f"\033[91m### Unmet dependencies: {unmet_dependencies}\033[0m")
                subagents = [agent for agent in self.agents if any(ur in agent.provides for ur in unmet_dependencies)]
                output = await self._choose_agent(messages, subagents, unmet_dependencies)
                if output.agent_or_tool is None:
                    continue
                if output.agent_or_tool in tools:
                    subagent = tools[output.agent_or_tool]
                else:
                    subagent = agents[output.agent_or_tool]
                tasks.append(ActorTask(subagent, instruction=output.chain_of_thought, title=output.agent_or_tool))
                step.success_title = f"Solved a dependency with {output.agent_or_tool}"
        tasks = tasks[::-1] + [ActorTask(agent, instruction=cot)]
        return Plan(*tasks, history=messages, context=context, coordinator=self)
