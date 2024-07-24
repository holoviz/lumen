from __future__ import annotations

import asyncio

from io import StringIO
from typing import Literal, Type

import param

from panel import bind
from panel.chat import ChatInterface
from panel.layout import Column, FlexBox, Tabs
from panel.pane import HTML, Markdown
from panel.viewable import Viewer
from panel.widgets import Button, FileDownload
from pydantic import create_model
from pydantic.fields import FieldInfo

from .agents import (
    Agent, AnalysisAgent, ChatAgent, SQLAgent,
)
from .export import export_notebook
from .llm import Llama, Llm
from .logs import ChatLogs
from .memory import memory
from .models import Validity
from .utils import get_schema, render_template, retry_llm_output

GETTING_STARTED_SUGGESTIONS = [
    "What datasets do you have?",
    "Tell me about the dataset.",
    "Create a plot of the dataset.",
    "Find the min and max of the values.",
]

DEMO_MESSAGES = [
    "What data is available?",
    "Can I see the first one?",
    "Tell me about the dataset.",
    "What could be interesting to analyze?",
    "Perform a SQL query on one of these.",
    "Show it to me as a scatter plot."
]


class Assistant(Viewer):
    """
    An Assistant handles multiple agents.
    """

    agents = param.List(default=[ChatAgent])

    llm = param.ClassSelector(class_=Llm, default=Llama())

    interface = param.ClassSelector(class_=ChatInterface)

    suggestions = param.List(default=GETTING_STARTED_SUGGESTIONS)

    demo_inputs = param.List(default=DEMO_MESSAGES)

    notebook_preamble = param.String()

    sidebar_widgets = param.List()

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

        def download_notebook():
            nb = export_notebook(self, preamble=self.notebook_preamble)
            return StringIO(nb)

        if interface is None:
            interface = ChatInterface(
                callback=self._chat_invoke, load_buffer=10,
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
        self._analyses = []
        for agent in agents or self.agents:
            if not isinstance(agent, Agent):
                kwargs = {"llm": llm} if agent.llm is None else {}
                agent = agent(interface=interface, **kwargs)
            if agent.llm is None:
                agent.llm = llm
            # must use the same interface or else nothing shows
            agent.interface = interface
            if isinstance(agent, AnalysisAgent):
                self._analyses.extend(agent.analyses)
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
        self._add_suggestions_to_footer(self.suggestions)

        self._current_agent = Markdown("## No agent active", margin=0)

        notebook_button = FileDownload(
            icon="notebook",
            button_type="success",
            callback=download_notebook,
            filename="Lumen_ai.ipynb",
            sizing_mode="stretch_width",
        )

        self._controls = Column(
            notebook_button, *self.sidebar_widgets, self._current_agent, Tabs(("Memory", memory))
        )

    def _add_suggestions_to_footer(
        self,
        suggestions: list[str],
        num_objects: int = 1,
        inplace: bool = True,
        analysis: bool = False,
        append_demo: bool = True
    ):
        async def hide_suggestions(_=None):
            if len(self.interface.objects) > num_objects:
                suggestion_buttons.visible = False

        async def use_suggestion(event):
            contents = event.obj.name
            await hide_suggestions()
            if analysis:
                for agent in self.agents:
                    if isinstance(agent, AnalysisAgent):
                        break
                else:
                    return
                await agent.invoke([{'user': 'User', 'contents': contents}])
            else:
                self.interface.send(contents)

        async def run_demo(event):
            await hide_suggestions()
            with self.interface.active_widget.param.update(loading=True):
                for demo_message in self.demo_inputs:
                    while self.interface.disabled:
                        await asyncio.sleep(1.25)
                    self.interface.active_widget.value = demo_message
                    await asyncio.sleep(2)

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

        if append_demo and self.demo_inputs:
            suggestion_buttons.append(Button(
                name="Show a demo",
                button_type="primary",
                on_click=run_demo,
                margin=5,
            ))

        message = self.interface.objects[-1]
        if inplace:
            message.footer_objects = [suggestion_buttons]

        self.interface.param.watch(hide_suggestions, "objects")
        return message

    async def _invalidate_memory(self, messages):
        table = memory.get("current_table")
        if not table:
            return

        source = memory.get("current_source")
        spec = get_schema(source, table=table)
        sql = memory.get("current_sql")
        system = render_template("check_validity.jinja2", table=table, spec=spec, sql=sql, analyses=self._analyses)
        with self.interface.add_step(title="Checking memory...", user="Assistant") as step:
            response = self.llm.stream(
                messages=messages,
                system=system,
                response_model=Validity,
            )
            async for output in response:
                step.stream(output.correct_assessment, replace=True)
            step.success_title = f"{output.is_invalid.title()} needs refresh" if output.is_invalid else "Memory still valid"

        if output and output.is_invalid:
            if output.is_invalid == "table":
                memory.pop("current_table", None)
                memory.pop("current_data", None)
                memory.pop("current_sql", None)
                memory.pop("current_pipeline", None)
                memory.pop("closest_tables", None)
                print("\033[91mInvalidated from memory.\033[0m")
            elif output.is_invalid == "sql":
                memory.pop("current_sql", None)
                memory.pop("current_data", None)
                memory.pop("current_pipeline", None)
                print("\033[91mInvalidated SQL from memory.\033[0m")

    async def _create_suggestion(self, instance, event):
        messages = self.interface.serialize(custom_serializer=self._serialize)[-3:-1]
        response = self.llm.stream(
            messages,
            system="Generate a follow-up question that a user might ask; ask from the user POV",
        )
        try:
            self.interface.disabled = True
            async for output in response:
                self.interface.active_widget.value_input = output
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

    @retry_llm_output()
    async def _create_valid_agent(self, messages, system, agent_model, errors=None):
        if errors:
            errors = '\n'.join(errors)
            messages += [{"role": "user", "content": f"\nExpertly resolve these issues:\n{errors}"}]

        out = self.llm.stream(
            messages=messages,
            system=system,
            response_model=agent_model
        )
        return out

    async def _choose_agent(self, messages: list | str, agents: list[Agent]):
        agent_names = tuple(sagent.name[:-5] for sagent in agents)
        if len(agent_names) == 0:
            raise ValueError("No agents available to choose from.")
        if len(agent_names) == 1:
            return agent_names[0]
        self._current_agent.object = "## **Current Agent**: [Lumen.ai](https://lumen.holoviz.org/)"
        agent_model = self._create_agent_model(agent_names)

        for agent in agents:
            if isinstance(agent, AnalysisAgent):
                analyses = "\n".join(f"- `{analysis.__name__}`: {analysis.__doc__.strip()}" for analysis in agent.analyses)
                agent.__doc__ = f"Available analyses include:\n{analyses}\nSelect this agent to perform one of these analyses."

        system = render_template(
            "pick_agent.jinja2", agents=agents, current_agent=self._current_agent.object
        )
        return await self._create_valid_agent(messages, system, agent_model)

    async def _get_agent(self, messages: list | str):
        if len(self.agents) == 1:
            return self.agents[0]
        agent_types = tuple(agent.name[:-5] for agent in self.agents)
        agents = {agent.name[:-5]: agent for agent in self.agents}

        if len(agent_types) == 1:
            agent = agent_types[0]
        else:
            with self.interface.add_step(title="Selecting relevant agent...", user="Assistant") as step:
                response = await self._choose_agent(messages, self.agents)
                async for output in response:
                    step.stream(output.chain_of_thought, replace=True)
                agent = output.agent
                step.success_title = f"Selected {agent}"
            messages.append({"role": "assistant", "content": output.chain_of_thought})

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
            with self.interface.add_step(title="Solving dependency chain...") as step:
                step.stream(f"Found {len(unmet_dependencies)} unmet dependencies: {', '.join(unmet_dependencies)}")
                print(f"\033[91m### Unmet dependencies: {unmet_dependencies}\033[0m")
                subagents = [
                    agent
                    for agent in self.agents
                    if any(ur in agent.provides for ur in unmet_dependencies)
                ]
                response = await self._choose_agent(messages, subagents)
                if isinstance(response, str):
                    subagent_name = response
                else:
                    async for output in response:
                        subagent_name = output.agent
                if subagent_name is None:
                    continue
                subagent = agents[subagent_name]
                agent_chain.append((subagent, unmet_dependencies))
                step.success_title = f"Solved a dependency with {subagent_name}"
        for subagent, deps in agent_chain[::-1]:
            with self.interface.add_step(title="Choosing subagent...") as step:
                step.stream(f"Assistant decided the {subagent.name[:-5]!r} will provide {', '.join(deps)}.")
                self._current_agent.object = f"## **Current Agent**: {subagent.name[:-5]}"

                if isinstance(subagent, SQLAgent):
                    custom_messages = messages.copy()
                    custom_agent = next((agent for agent in self.agents if isinstance(agent, AnalysisAgent)), None)
                    if custom_agent:
                        custom_analysis_doc = custom_agent.__doc__.replace("Available analyses include:\n", "")
                        custom_message = (
                            f"Avoid doing the same analysis as any of these custom analyses: {custom_analysis_doc} "
                            f"Most likely, you'll just need to do a simple SELECT * FROM {{table}};"
                        )
                        custom_messages.append({"role": "user", "content": custom_message})
                        print(custom_messages)
                    await subagent.answer(custom_messages)
                else:
                    await subagent.answer(messages)
                step.success_title = f"Selected {subagent.name[:-5]}"
        return selected

    def _serialize(self, obj):
        if isinstance(obj, (Tabs, Column)):
            for o in obj:
                if hasattr(o, "visible") and o.visible:
                    break
            return self._serialize(o)

        if isinstance(obj, HTML) and 'catalog' in obj.tags:
            return f"Summarized table listing: {obj.object[:30]}"

        if hasattr(obj, "object"):
            obj = obj.object
        elif hasattr(obj, "value"):
            obj = obj.value
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

        await agent.invoke(messages[-2:])
        self._current_agent.object = "## No agent active"

        if "current_pipeline" in agent.provides:
            pipeline = memory['current_pipeline']
            applicable_analyses = [
                analysis for analysis in self._analyses if analysis.applies(pipeline)
            ]
            self._add_suggestions_to_footer(
                [f"Apply {analysis.__name__}" for analysis in applicable_analyses],
                append_demo=False,
                analysis=True,
                num_objects=len(self.interface.objects),
            )

        print("\033[92mDONE\033[0m", "\n\n")

    def controls(self):
        return self._controls

    def __panel__(self):
        return self.interface
