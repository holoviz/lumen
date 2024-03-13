from __future__ import annotations

from typing import Literal, Type

import param

from panel.chat import ChatInterface
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

    llm = param.ClassSelector(class_=Llm, default=Llama(model='default'))

    interface = param.ClassSelector(class_=ChatInterface)

    def __init__(
        self,
        llm: Llm | None = None,
        interface: ChatInterface | None = None,
        agents: list[Agent | Type[Agent]] | None = None,
        **params
    ):
        if interface is None:
            interface = ChatInterface(callback=self._chat_invoke, callback_exception='verbose')
        else:
            interface.callback = self._chat_invoke
        llm = llm or self.llm
        instantiated = []
        for agent in (agents or self.agents):
            if not isinstance(agent, Agent):
                kwargs = {'llm': llm} if agent.llm is None else {}
                agent = agent(interface=interface, **kwargs)
            instantiated.append(agent)
        super().__init__(llm=llm, agents=instantiated, interface=interface, **params)

    def _generate_picker_prompt(self, agents):
        prompt = "Select most relevant agent for the user's query:\n" + "\n".join(
            f"- {agent.name}: {agent.__doc__.strip()}" for agent in self.agents
        )
        print(prompt)
        return prompt

    def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface):
        return self.invoke(contents)

    def _choose_agent(self, messages: list | str, agents: list[Agent]):
        agent_names = tuple(sagent.name for sagent in agents)
        agent_model = create_model("Agent", agent=(Literal[agent_names], ...))
        return self.llm.invoke(
            messages=messages,
            system=self._generate_picker_prompt(agents),
            response_model=agent_model,
        ).agent

    def _get_agent(self, messages: list | str):
        if len(self.agents) == 1:
            return self.agents[0]
        agent_types = tuple(agent.name for agent in self.agents)
        agents = {agent.name: agent for agent in self.agents}
        if len(agent_types) == 1:
            agent = agent_types[0]
        else:
            agent = self._choose_agent(messages, self.agents)
        print(f'Assistant decided on {agent} between the available options: {agent_types}')
        selected = subagent = agents[agent]
        agent_chain = []
        while (unmet_dependencies:=tuple(r for r in subagent.requires if r not in memory)):
            subagents = [
                agent for agent in self.agents if any(ur in agent.provides for ur in unmet_dependencies)
            ]
            subagent_name = self._choose_agent(messages, subagents)
            subagent = agents[subagent_name]
            agent_chain.append((subagent, unmet_dependencies))
            if not (unmet_dependencies:= tuple(r for r in subagent.requires if r not in memory)):
                break
        for (subagent, deps) in agent_chain[::-1]:
            print(f"Assistant decided the {agent} will provide {deps}.")
            subagent.answer(messages)
        return selected

    def invoke(self, messages: list | str) -> str:
        return self._get_agent(messages).invoke(messages)

    def __panel__(self):
        return self.interface
