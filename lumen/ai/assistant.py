from __future__ import annotations

import json

from io import StringIO
from typing import Literal, Type

import param

from panel.chat import ChatInterface, ChatMessage
from panel.layout import Column, FlexBox, Tabs
from panel.pane import Markdown
from panel.viewable import Viewer
from panel.widgets import Button, FileDownload
from pydantic import create_model
from pydantic.fields import FieldInfo

from .agents import Agent, ChatAgent
from .llm import Llama, Llm
from .memory import memory
from .models import PipelineValidity

GETTING_STARTED_SUGGESTIONS = [
    "What datasets do you have?",
    "Tell me about the dataset.",
    "Create a plot of the dataset.",
    "Find the min and max of the values.",
]

GETTING_STARTED_SUGGESTIONS = [
    "What datasets do you have?",
    "Tell me about the dataset.",
    "Create a plot of the dataset.",
    "Find the min and max of the values.",
]


class Assistant(Viewer):
    """
    An Assistant handles multiple agents.
    """

    agents = param.List(default=[ChatAgent])

    llm = param.ClassSelector(class_=Llm, default=Llama(model="default"))

    interface = param.ClassSelector(class_=ChatInterface)

    def __init__(
        self,
        llm: Llm | None = None,
        interface: ChatInterface | None = None,
        agents: list[Agent | Type[Agent]] | None = None,
        **params,
    ):
        def download_messages():
            def filter_by_reactions(messages):
                return [
                    message for message in messages if "favorite" in message.reactions
                ]

            messages = self.interface.serialize(filter_by=filter_by_reactions)
            if len(messages) == 0:
                messages = self.interface.serialize()
            if len(messages) == 0:
                raise ValueError("No messages to download.")
            return StringIO(json.dumps(messages))

        if interface is None:
            interface = ChatInterface(callback=self._chat_invoke)
        else:
            interface.callback = self._chat_invoke
        interface.callback_exception = "raise"

        llm = llm or self.llm
        instantiated = []
        for agent in agents or self.agents:
            if not isinstance(agent, Agent):
                kwargs = {"llm": llm} if agent.llm is None else {}
                agent = agent(interface=interface, **kwargs)
            instantiated.append(agent)
        super().__init__(llm=llm, agents=instantiated, interface=interface, **params)
        interface.send(
            "Welcome to LumenAI; get started by clicking a suggestion or type your own query below!", user="Help", respond=False
        )
        interface.button_properties={
            "suggest": {"callback": self._create_suggestion, "icon": "wand"}
        }
        self._add_suggestions_to_footer(GETTING_STARTED_SUGGESTIONS)

        self._current_agent = Markdown("## No agent active", margin=0)
        download_button = FileDownload(
            icon="download",
            button_type="success",
            callback=download_messages,
            filename="favorited_messages.json",
            sizing_mode="stretch_width",
        )
        self._controls = Column(download_button, self._current_agent, Tabs(("Memory", memory)))

    def _add_suggestions_to_footer(self, suggestions: list[str], inplace: bool = True):
        def use_suggestion(event):
            contents = event.obj.name
            suggestion_buttons.visible = False
            self.interface.send(contents)

        suggestion_buttons = FlexBox(
            *[
                Button(
                    name=suggestion,
                    button_style="outline",
                    on_click=use_suggestion,
                    margin=5,
                )
                for suggestion in suggestions
            ],
            margin=(5, 5),
        )

        message = self.interface.objects[-1]
        message = ChatMessage(
            footer_objects=[suggestion_buttons],
            user=message.user,
            object=message.object,
        )
        if inplace:
            self.interface.objects[-1] = message
        return message

    async def _invalidate_pipeline(self, messages):
        pipeline = memory.get("current_pipeline")
        if not pipeline:
            return
        system = f"""
        Based on the latest user's query, does the pipeline include all the necessary datasets or columns
        to answer the user's query? If an does the SQL transform is included, do those
        transformations contain all the necessary columns to answer the user's query?
        FOCUS only on the columns, and **disregard** everything else,
        like whether the pipeline contains visualization or not.

        ### Current Pipeline:
        ```
        {pipeline.to_spec()}
        ```
        """
        validity = await self.llm.invoke(
            messages=messages,
            system=system,
            response_model=PipelineValidity,
            allow_partial=False,
            model="gpt-4-turbo-preview",
        )
        if not validity.valid:
            memory.pop("current_pipeline", None)
            memory.pop("current_sql", None)
            memory.pop("current_transform", None)
        print(f"Current memory: {memory.keys()}")

    def _create_suggestion(self, instance, event):
        messages = self.interface.serialize(custom_serializer=self._serialize)[-3:-1]
        string = self.llm.stream(
            messages,
            system="Generate a follow-up question that a user might ask; ask from the user POV",
            allow_partial=True,
        )
        try:
            self.interface.disabled = True
            for chunk in string:
                self.interface.active_widget.value_input = chunk
        finally:
            self.interface.disabled = False

    def _generate_picker_prompt(self, agents):
        # prompt = f'Current you have the following items in memory: {list(memory)}'
        prompt = (
            "\nYou are the leader of a team of expert agents. "
            "Select most relevant expert for the user's query:\n'''\n"
            + "\n".join(
                f"- `{agent.name[:-5]}`: {' '.join(agent.__doc__.strip().split())}"
                for agent in agents
            )
            + "\n'''\nEach agent can request other agents to fill in the blanks, so pick the agent that can best answer the entire query."
        )
        if "current_agent" in memory:
            prompt += f"If possible, continue using the existing agent to perform the query: {self._current_agent.object}"
        return prompt

    async def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface):
        print("NEW" + "-" * 100)
        await self.invoke(contents)

    async def _choose_agent(self, messages: list | str, agents: list[Agent]):
        agent_names = tuple(sagent.name[:-5] for sagent in agents)
        if len(agent_names) == 0:
            raise ValueError("No agents available to choose from.")
        if len(agent_names) == 1:
            return agent_names[0]
        agent_model = create_model(
            "Agent",
            chain_of_thought=(
                str,
                FieldInfo(
                    description=(
                        "Concise thoughts focused on application and how it could be useful to the user, "
                        "specifically focusing on the prior messages and the user's query. Brevity is key."
                    )
                ),
            ),
            agent=(Literal[agent_names], ...),
        )
        self._current_agent.object = "## **Current Agent**: Lumen.ai"
        for _ in range(3):
            out = await self.llm.invoke(
                messages=messages,
                system=self._generate_picker_prompt(agents),
                response_model=agent_model,
                allow_partial=False
            )
            if out:
                return out.agent

    async def _get_agent(self, messages: list | str):
        if len(self.agents) == 1:
            return self.agents[0]
        agent_types = tuple(agent.name[:-5] for agent in self.agents)
        agents = {agent.name[:-5]: agent for agent in self.agents}
        if len(agent_types) == 1:
            agent = agent_types[0]
        else:
            agent = await self._choose_agent(messages, self.agents)
        print(
            f"Assistant decided on {agent} between the available options: {agent_types}"
        )
        selected = subagent = agents[agent]
        agent_chain = []
        while unmet_dependencies := tuple(
            r for r in subagent.requires if r not in memory
        ):
            print(f"Unmet dependencies: {unmet_dependencies}")
            subagents = [
                agent
                for agent in self.agents
                if any(ur in agent.provides for ur in unmet_dependencies)
            ]
            subagent_name = await self._choose_agent(messages, subagents)
            if subagent_name is None:
                continue
            subagent = agents[subagent_name]
            agent_chain.append((subagent, unmet_dependencies))
            if not (
                unmet_dependencies := tuple(
                    r for r in subagent.requires if r not in memory
                )
            ):
                break
        for subagent, deps in agent_chain[::-1]:
            print(f"Assistant decided the {subagent} will provide {deps}.")
            self._current_agent.object = f"## **Current Agent**: {subagent.name[:-5]}"
            await subagent.answer(messages)
        return selected

    def _serialize(self, obj):
        if isinstance(obj, (Tabs, Column)):
            return self._serialize(obj[0])
        if hasattr(obj, "object"):
            return obj.object
        elif hasattr(obj, "value"):
            return obj.value
        return str(obj)

    async def invoke(self, messages: list | str) -> str:
        messages = self.interface.serialize(custom_serializer=self._serialize)[-4:]
        await self._invalidate_pipeline(messages[-2:])
        agent = await self._get_agent(messages[-2:])
        self._current_agent.object = f"## **Current Agent**: {agent.name[:-5]}"

        for message in messages:
            print(f"{message['role']!r}: {message['content']}")
            print("ENTRY" + "-" * 10)

        result = await agent.invoke(messages[-2:])
        self._current_agent.object = "## No agent active"
        print("DONE", "\n\n")
        return result

    def controls(self):
        return self._controls

    def __panel__(self):
        return self.interface
