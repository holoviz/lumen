from __future__ import annotations

import json

from io import StringIO
from typing import Literal, Type

import param

from panel import bind
from panel.chat import ChatInterface, ChatMessage
from panel.layout import Column, FlexBox, Tabs
from panel.pane import HTML, Markdown
from panel.viewable import Viewer
from panel.widgets import Button, FileDownload
from pydantic import create_model
from pydantic.fields import FieldInfo

from .agents import Agent, ChatAgent
from .llm import Llama, Llm
from .logs import ChatLogs
from .memory import memory
from .models import Validity
from .utils import render_template

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

    llm = param.ClassSelector(class_=Llm, default=Llama())

    interface = param.ClassSelector(class_=ChatInterface)

    logs_filename = param.String()

    def __init__(
        self,
        llm: Llm | None = None,
        interface: ChatInterface | None = None,
        agents: list[Agent | Type[Agent]] | None = None,
        logs_filename: str = "",
        **params,
    ):

        def on_message(message, instance):
            def update_on_reaction(reactions):
                self._logs.update_status(
                    message_id=message_id,
                    liked="like" in reactions,
                    disliked="dislike" in reactions,
                )

            bind(update_on_reaction, message.param.reactions, watch=True)
            message_id = id(message)
            message_index = len(instance) - 1
            self._logs.upsert(
                session_id=self._session_id,
                message_id=message_id,
                message_index=message_index,
                message_user=message.user,
                message_content=message.object,
            )

        def on_undo(instance, _):
            count = instance._get_last_user_entry_index()
            messages = instance[-count:]
            for message in messages:
                self._logs.update_status(message_id=id(message), removed=True)

        def on_rerun(instance, _):
            count = instance._get_last_user_entry_index() - 1
            messages = instance[-count:]
            for message in messages:
                self._logs.update_status(message_id=id(message), removed=True)

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
            interface = ChatInterface(
                callback=self._chat_invoke,
            )
        else:
            interface.callback = self._chat_invoke
        interface.callback_exception = "raise"
        interface.message_params["reaction_icons"] = {"like": "thumb-up", "dislike": "thumb-down"}

        self._session_id = id(self)

        if logs_filename is not None:
            self._logs = ChatLogs(filename=logs_filename)
            interface.post_hook = on_message

        llm = llm or self.llm
        instantiated = []
        for agent in agents or self.agents:
            if not isinstance(agent, Agent):
                kwargs = {"llm": llm} if agent.llm is None else {}
                agent = agent(interface=interface, **kwargs)
            instantiated.append(agent)

        super().__init__(llm=llm, agents=instantiated, interface=interface, logs_filename=logs_filename, **params)
        interface.send(
            "Welcome to LumenAI; get started by clicking a suggestion or type your own query below!",
            user="Help", respond=False,
        )
        interface.button_properties={
            "suggest": {"callback": self._create_suggestion, "icon": "wand"},
            "undo": {"callback": on_undo},
            "rerun": {"callback": on_rerun},
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

    async def _invalidate_memory(self, messages):
        table = memory.get("current_table")
        if not table:
            return

        current_data = memory["current_data"]
        if isinstance(current_data, dict):
            columns = list(current_data["stats"].keys())
        else:
            columns = list(current_data.columns)
        system = render_template("check_validity.jinja2", table=table, columns=columns)
        validity = await self.llm.invoke(
            messages=messages,
            system=system,
            response_model=Validity,
            allow_partial=False,
        )
        if validity and validity.is_invalid:
            memory.pop("current_table", None)
            memory.pop("current_data", None)
            memory.pop("current_pipeline", None)
            memory.pop("closest_tables", None)
            print("\033[91mInvalidated from memory.\033[0m")

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

    async def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface):
        print("\033[94mNEW\033[0m" + "-" * 100)
        await self.invoke(contents)

    @staticmethod
    def _create_agent_model(agent_names):
        agent_model = create_model(
            "RelevantAgent",
            chain_of_thought=(
                str,
                FieldInfo(
                    description="Explain in your own words, what the user wants."
                ),
            ),
            agent=(
                Literal[agent_names],
                FieldInfo(default=..., description="The most relevant agent to use.")
            ),
        )
        return agent_model

    async def _choose_agent(self, messages: list | str, agents: list[Agent], return_reasoning: bool = False):
        agent_names = tuple(sagent.name[:-5] for sagent in agents)
        if len(agent_names) == 0:
            raise ValueError("No agents available to choose from.")
        if len(agent_names) == 1:
            return agent_names[0]
        self._current_agent.object = "## **Current Agent**: Lumen.ai"
        agent_model = self._create_agent_model(agent_names)
        system = render_template(
            "pick_agent.jinja2", agents=agents, current_agent=self._current_agent.object
        )
        for _ in range(3):
            out = await self.llm.invoke(
                messages=messages,
                system=system,
                response_model=agent_model,
                allow_partial=False
            )
            if not (out and out.agent):
                continue
            elif return_reasoning:
                return out.agent, out.chain_of_thought
            return out.agent
        return (None, None) if return_reasoning else None

    async def _get_agent(self, messages: list | str):
        if len(self.agents) == 1:
            return self.agents[0]
        agent_types = tuple(agent.name[:-5] for agent in self.agents)
        agents = {agent.name[:-5]: agent for agent in self.agents}

        if len(agent_types) == 1:
            agent = agent_types[0]
        else:
            agent, reasoning = await self._choose_agent(messages, self.agents, return_reasoning=True)
            messages.append({"role": "assistant", "content": reasoning})

        if agent is None:
            return None

        print(
            f"Assistant decided on \033[95m{agent!r}\033[0m"
        )
        selected = subagent = agents[agent]
        agent_chain = []
        while unmet_dependencies := tuple(
            r for r in await subagent.requirements(messages) if r not in memory
        ):
            print(f"\033[91m### Unmet dependencies: {unmet_dependencies}\033[0m")
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
        for subagent, deps in agent_chain[::-1]:
            print(f"Assistant decided the {subagent.name[:-5]!r} will provide {deps}.")
            self._current_agent.object = f"## **Current Agent**: {subagent.name[:-5]}"
            await subagent.answer(messages)
        return selected

    def _serialize(self, obj):
        if isinstance(obj, (Tabs, Column)):
            return self._serialize(obj[0])
        if isinstance(obj, HTML) and 'catalog' in obj.tags:
            return f"Summarized table listing: {obj.object[:30]}"
        elif hasattr(obj, "object"):
            return obj.object
        elif hasattr(obj, "value"):
            return obj.value
        return str(obj)

    async def invoke(self, messages: list | str) -> str:
        messages = self.interface.serialize(custom_serializer=self._serialize)[-4:]
        await self._invalidate_memory(messages[-2:])
        agent = await self._get_agent(messages[-3:])
        if agent is None:
            msg = (
                "Assistant could not settle on an agent to perform the requested query. "
                "Please restate your request."
            )
            self.interface.stream(msg, user='Lumen')
            return msg

        self._current_agent.object = f"## **Current Agent**: {agent.name[:-5]}"
        print("\n\033[95mMESSAGES:\033[0m")
        for message in messages:
            print(f"{message['role']!r}: {message['content']}")
            print("ENTRY" + "-" * 10)

        result = await agent.invoke(messages[-2:])
        self._current_agent.object = "## No agent active"
        print("\033[92mDONE\033[0m", "\n\n")
        return result

    def controls(self):
        return self._controls

    def __panel__(self):
        return self.interface
