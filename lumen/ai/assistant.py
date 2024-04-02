from __future__ import annotations

from typing import Literal, Type

import param

from panel.chat import ChatInterface, ChatSteps
from panel.layout import Column, Tabs
from panel.pane import Markdown
from panel.viewable import Viewer
from pydantic import create_model

from .agents import Agent, ChatAgent
from .llm import Llama, Llm
from .memory import memory


class Assistant(Viewer):
    """
    An Assistant handles multiple agents.
    """

    agents = param.List(default=[ChatAgent])

    llm = param.ClassSelector(class_=Llm, default=Llama(model="default"))

    interface = param.ClassSelector(class_=ChatInterface)

    steps = param.ClassSelector(class_=ChatSteps)

    def __init__(
        self,
        llm: Llm | None = None,
        interface: ChatInterface | None = None,
        steps: ChatSteps | None = None,
        agents: list[Agent | Type[Agent]] | None = None,
        **params,
    ):
        if interface is None:
            interface = ChatInterface(
                callback=self._chat_invoke, callback_exception="verbose"
            )
        else:
            interface.callback = self._chat_invoke
        if steps is None:
            steps = ChatSteps()
        llm = llm or self.llm
        instantiated = []
        for agent in agents or self.agents:
            if not isinstance(agent, Agent):
                kwargs = {"llm": llm} if agent.llm is None else {}
                agent = agent(interface=interface, steps=steps, **kwargs)
            instantiated.append(agent)
        super().__init__(llm=llm, agents=instantiated, interface=interface, steps=steps, **params)
        self._current_agent = Markdown("## No agent active", margin=0)
        self._controls = Column(self._current_agent, Tabs(("Memory", memory)))

    def _invoke_on_init(self):
        self.invoke("Initializing chat")

    def _generate_picker_prompt(self, agents):
        # prompt = f'Current you have the following items in memory: {list(memory)}'
        prompt = "\nSelect most relevant agent for the user's query:\n" + "\n".join(
            f"- {agent.name[:-5]}: {agent.__doc__.strip()}" for agent in agents
        )
        return prompt

    def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface):
        return self.invoke(contents)

    def _choose_agent(self, messages: list | str, agents: list[Agent]):
        agent_names = tuple(sagent.name[:-5] for sagent in agents)
        if len(agent_names) == 0:
            raise ValueError("No agents available to choose from.")
        if len(agent_names) == 1:
            return agent_names[0]
        agent_model = create_model(
            "Agent", chain_of_thought=(str, ...), agent=(Literal[agent_names], ...)
        )
        self._current_agent.object = "## **Current Agent**: Lumen.ai"
        for _ in range(3):
            out = self.llm.invoke(
                messages=messages,
                system=self._generate_picker_prompt(agents),
                response_model=agent_model,
                allow_partial=False,
            )
            if out:
                return out.agent

    def invoke(self, messages: list | str) -> str:
        if len(self.agents) == 1:
            return self.agents[0]
        agent_types = tuple(agent.name[:-5] for agent in self.agents)
        agents = {agent.name[:-5]: agent for agent in self.agents}
        if len(agent_types) == 1:
            agent = agent_types[0]
        else:
            agent = self._choose_agent(messages, self.agents)

        steps = self.interface.steps(
            chat_steps=self.steps,
            title="Solving dependencies...",
            status="running",
        )

        selected = subagent = agents[agent]
        steps.step(title="Selected {subagent.name[:-5]}...")

        agent_chain = []
        while unmet_dependencies := tuple(
            r for r in subagent.requires if r not in memory
        ):
            steps.title = f"Solving {len(unmet_dependencies)} dependencies..."
            subagents = [
                agent
                for agent in self.agents
                if any(ur in agent.provides for ur in unmet_dependencies)
            ]
            subagent_name = self._choose_agent(messages, subagents)
            if subagent_name is None:
                continue
            subagent = agents[subagent_name]
            steps.step(title=f"Preparing to use {subagent_name}...")
            agent_chain.append((subagent, unmet_dependencies))
            if not (
                unmet_dependencies := tuple(
                    r for r in subagent.requires if r not in memory
                )
            ):
                break

        for i, (subagent, deps) in enumerate(agent_chain[::-1]):
            self._current_agent.object = f"## **Current Agent**: {subagent.name[:-5]}"
            steps.title = f"Using {subagent.name[:-5]} to solve {deps}..."
            with steps[-i - 1] as step:
                subagent.answer(messages, step=step)

        self._current_agent.object = f"## **Current Agent**: {selected.name[:-5]}"
        with steps[0] as step:
            result = selected.invoke(messages, step=step)
        self._current_agent.object = "## No agent active"
        steps.status = "completed"
        steps.title = "Completed!"
        return result

    def controls(self):
        return self._controls

    def __panel__(self):
        return self.interface
