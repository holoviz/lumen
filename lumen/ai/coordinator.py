from __future__ import annotations

import asyncio
import re

from typing import TYPE_CHECKING

import param
import yaml

from panel import Card, bind
from panel.chat import ChatInterface, ChatStep
from panel.layout import Column, FlexBox, Tabs
from panel.pane import HTML
from panel.viewable import Viewer
from panel.widgets import Button
from pydantic import BaseModel

from .agents import (
    Agent, AnalysisAgent, ChatAgent, SQLAgent,
)
from .config import DEMO_MESSAGES, GETTING_STARTED_SUGGESTIONS
from .llm import Llama, Llm, Message
from .logs import ChatLogs
from .memory import _Memory, memory
from .models import Validity, make_agent_model, make_plan_models
from .utils import get_schema, render_template, retry_llm_output

if TYPE_CHECKING:
    from panel.chat.step import ChatStep

    from ..sources import Source


class ExecutionNode(param.Parameterized):
    """
    Defines a node in an execution graph.
    """

    agent = param.ClassSelector(class_=Agent)

    provides = param.ClassSelector(class_=(set, list), default=set())

    instruction = param.String(default="")

    title = param.String(default="")

    render_output = param.Boolean(default=False)


class Coordinator(Viewer):
    """
    A Coordinator is responsible for coordinating the actions
    of a number of agents towards the user defined query by
    computing an execution graph and then executing each
    step along the graph.
    """

    agents = param.List(default=[ChatAgent], doc="""
        List of agents to coordinate.""")

    demo_inputs = param.List(default=DEMO_MESSAGES, doc="""
        List of instructions to demo the Coordinator.""")

    interface = param.ClassSelector(class_=ChatInterface, doc="""
        The ChatInterface for the Coordinator to interact with.""")

    llm = param.ClassSelector(class_=Llm, default=Llama(), doc="""
        LLM used by the assistant to plan the execution.""")

    logs_filename = param.String(default=None, doc="""
        Log file to write to.""")

    memory = param.ClassSelector(class_=_Memory, default=None, doc="""
        Local memory which will be used to provide the agent context.
        If None the global memory will be used.""")

    render_output = param.Boolean(default=True, doc="""
        Whether to write outputs to the ChatInterface.""")

    suggestions = param.List(default=GETTING_STARTED_SUGGESTIONS, doc="""
        Initial list of suggestions of actions the user can take.""")

    __abstract = True

    def __init__(
        self,
        llm: Llm | None = None,
        interface: ChatInterface | None = None,
        agents: list[Agent | type[Agent]] | None = None,
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

        if interface is None:
            interface = ChatInterface(
                callback=self._chat_invoke, load_buffer=5,
            )
        else:
            interface.callback = self._chat_invoke
        interface.callback_exception = "verbose"
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
            if isinstance(agent, AnalysisAgent):
                analyses = "\n".join(
                    f"- `{analysis.__name__}`: {(analysis.__doc__ or '').strip()}"
                    for analysis in agent.analyses if analysis._callable_by_llm
                )
                agent.__doc__ = f"Available analyses include:\n{analyses}\nSelect this agent to perform one of these analyses."
                agent.interface = interface
                self._analyses.extend(agent.analyses)
            if agent.llm is None:
                agent.llm = llm
            # must use the same interface or else nothing shows
            agent.interface = interface
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

        if "current_source" in self._memory and "available_sources" not in self._memory:
            self._memory["available_sources"] = [self._memory["current_source"]]
        elif "current_source" not in self._memory and self._memory.get("available_sources"):
            self._memory["current_source"] = self._memory["available_sources"][0]
        elif "available_sources" not in self._memory:
            self._memory["available_sources"] = []

    @property
    def _memory(self):
        return memory if self.memory is None else self.memory

    def __panel__(self):
        return self.interface

    def _add_suggestions_to_footer(
        self,
        suggestions: list[str],
        num_objects: int = 1,
        inplace: bool = True,
        analysis: bool = False,
        append_demo: bool = True,
        hide_after_use: bool = True
    ):
        async def hide_suggestions(_=None):
            if len(self.interface.objects) > num_objects:
                suggestion_buttons.visible = False

        async def use_suggestion(event):
            button = event.obj
            with button.param.update(loading=True), self.interface.active_widget.param.update(loading=True):
                contents = button.name
                if hide_after_use:
                    suggestion_buttons.visible = False
                    if event.new > 1:  # prevent double clicks
                        return
                if analysis:
                    for agent in self.agents:
                        if isinstance(agent, AnalysisAgent):
                            break
                    else:
                        print("No analysis agent found.")
                        return
                    messages = [{'role': 'user', 'content': contents}]
                    await agent.respond(
                        messages, render_output=self.render_output, agents=self.agents
                    )
                    await self._add_analysis_suggestions()
                else:
                    self.interface.send(contents)

        async def run_demo(event):
            if hide_after_use:
                suggestion_buttons.visible = False
                if event.new > 1:  # prevent double clicks
                    return
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
        disable_js = "cb_obj.origin.disabled = true; setTimeout(() => cb_obj.origin.disabled = false, 3000)"
        for b in suggestion_buttons:
            b.js_on_click(code=disable_js)

        message = self.interface.objects[-1]
        if inplace:
            message.footer_objects = [suggestion_buttons]

        self.interface.param.watch(hide_suggestions, "objects")
        return message

    async def _add_analysis_suggestions(self):
        pipeline = self._memory['current_pipeline']
        current_analysis = self._memory.get("current_analysis")
        allow_consecutive = getattr(current_analysis, '_consecutive_calls', True)
        applicable_analyses = []
        for analysis in self._analyses:
            if await analysis.applies(pipeline) and (allow_consecutive or analysis is not type(current_analysis)):
                applicable_analyses.append(analysis)
        self._add_suggestions_to_footer(
            [f"Apply {analysis.__name__}" for analysis in applicable_analyses],
            append_demo=False,
            analysis=True,
            hide_after_use=False,
            num_objects=len(self.interface.objects),
        )

    async def _invalidate_memory(self, messages):
        table = self._memory.get("current_table")
        if not table:
            return

        source = self._memory.get("current_source")
        if table not in source:
            sources = [src for src in self._memory.get('available_sources', []) if table in src]
            if sources:
                self._memory['current_source'] = source = sources[0]
            else:
                raise KeyError(f'Table {table} could not be found in available sources.')

        try:
            spec = await get_schema(source, table=table, include_count=True)
        except Exception:
            # If the selected table cannot be fetched we should invalidate it
            spec = None

        sql = self._memory.get("current_sql")
        analyses_names = [analysis.__name__ for analysis in self._analyses]
        system = render_template("check_validity.jinja2", table=table, spec=yaml.dump(spec), sql=sql, analyses=analyses_names)
        with self.interface.add_step(title="Checking memory...", user="Assistant") as step:
            output = await self.llm.invoke(
                messages=messages,
                system=system,
                response_model=Validity,
            )
            step.stream(output.correct_assessment, replace=True)
            step.success_title = f"{output.is_invalid.title()} needs refresh" if output.is_invalid else "Memory still valid"

        if output and output.is_invalid:
            if output.is_invalid == "table":
                self._memory.pop("current_table", None)
                self._memory.pop("current_data", None)
                self._memory.pop("current_sql", None)
                self._memory.pop("current_pipeline", None)
                self._memory.pop("closest_tables", None)
                print("\033[91mInvalidated from memory.\033[0m")
            elif output.is_invalid == "sql":
                self._memory.pop("current_sql", None)
                self._memory.pop("current_data", None)
                self._memory.pop("current_pipeline", None)
                print("\033[91mInvalidated SQL from memory.\033[0m")
            return output.correct_assessment

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
        await self.respond(contents)

    @retry_llm_output()
    async def _fill_model(self, messages, system, agent_model, errors=None):
        if errors:
            errors = '\n'.join(errors)
            messages += [{"role": "user", "content": f"\nExpertly resolve these issues:\n{errors}"}]

        out = await self.llm.invoke(
            messages=messages,
            system=system,
            response_model=agent_model
        )
        return out

    async def _execute_graph_node(self, node: ExecutionNode, messages: list[Message]):
        subagent = node.agent
        instruction = node.instruction
        title = node.title.capitalize()
        render_output = node.render_output and self.render_output
        agent_name = type(subagent).name.replace('Agent', '')

        # attach the new steps to the existing steps--used when there is intermediate Lumen output
        steps_layout = None
        for step_message in reversed(self.interface.objects[-5:]):
            if step_message.user == "Assistant" and isinstance(step_message.object, Card):
                steps_layout = step_message.object
                break

        with self.interface.add_step(title=f"Querying {agent_name} agent...", steps_layout=steps_layout) as step:
            step.stream(f"`{agent_name}` agent is working on the following task:\n\n{instruction}")
            custom_messages = messages.copy()
            if isinstance(subagent, SQLAgent):
                custom_agent = next((a for a in self.agents if isinstance(a, AnalysisAgent)), None)
                if custom_agent:
                    custom_analysis_doc = custom_agent.__doc__.replace("Available analyses include:\n", "")
                    custom_message = (
                        f"Avoid doing the same analysis as any of these custom analyses: {custom_analysis_doc} "
                        f"Most likely, you'll just need to do a simple SELECT * FROM {{table}};"
                    )
                    custom_messages.append({"role": "user", "content": custom_message})
            if instruction:
                custom_messages.append({"role": "user", "content": instruction})

            respond_kwargs = {"agents": self.agents} if isinstance(subagent, AnalysisAgent) else {}
            with subagent.param.update(memory=self.memory, steps_layout=steps_layout):
                await subagent.respond(custom_messages, step_title=title, render_output=render_output, **respond_kwargs)
            step.stream(f"`{agent_name}` agent successfully completed the following task:\n\n- {instruction}", replace=True)
            step.success_title = f"{agent_name} agent successfully responded"

    def _serialize(self, obj, exclude_passwords=True):
        if isinstance(obj, (Column, Tabs)):
            for o in obj:
                if isinstance(obj, ChatStep) and not obj.title.startswith("Selected"):
                    # only want the chain of thoughts from the selected agent
                    continue
                if hasattr(o, "visible") and o.visible:
                    break
            else:
                return ""
            string = self._serialize(o)
            if exclude_passwords:
                string = re.sub(r"password:.*\n", "", string)
            return string

        if isinstance(obj, HTML) and 'catalog' in obj.tags:
            return f"Summarized table listing: {obj.object[:30]}"

        if hasattr(obj, "object"):
            obj = obj.object
        elif hasattr(obj, "value"):
            obj = obj.value
        return str(obj)

    async def respond(self, messages: list[Message]) -> str:
        messages = self.interface.serialize(custom_serializer=self._serialize)[-4:]
        invalidation_assessment = await self._invalidate_memory(messages[-2:])
        context_length = 3
        if invalidation_assessment:
            messages.append({"role": "assistant", "content": invalidation_assessment + " so do not choose that table."})
            context_length += 1
        agents = {agent.name[:-5]: agent for agent in self.agents}
        execution_graph = await self._compute_execution_graph(messages[-context_length:], agents)
        if execution_graph is None:
            msg = (
                "Assistant could not settle on a plan of action to perform the requested query. "
                "Please restate your request."
            )
            self.interface.stream(msg, user='Lumen')
            return msg
        for node in execution_graph:
            await self._execute_graph_node(node, messages[-context_length:])
        if "current_pipeline" in self._memory:
            await self._add_analysis_suggestions()
        print("\033[92mDONE\033[0m", "\n\n")


class Resolver(Coordinator):
    """
    Resolver is a type of Coordinator that chooses the agent to answer the
    query and then recursively resolves all the information required for
    that agent until the answer is available.
    """

    async def _choose_agent(
        self,
        messages: list[Message],
        agents: list[Agent] | None = None,
        primary: bool = False,
        unmet_dependencies: tuple[str] | None = None
    ):
        if agents is None:
            agents = self.agents
        agents = [agent for agent in agents if await agent.applies(self._memory)]
        agent_names = tuple(sagent.name[:-5] for sagent in agents)
        agent_model = make_agent_model(agent_names, primary=primary)
        if len(agent_names) == 0:
            raise ValueError("No agents available to choose from.")
        if len(agent_names) == 1:
            return agent_model(agent=agent_names[0], chain_of_thought='')
        system = render_template(
            'pick_agent.jinja2', agents=agents, primary=primary, unmet_dependencies=unmet_dependencies
        )
        return await self._fill_model(messages, system, agent_model)

    async def _compute_execution_graph(self, messages, agents: dict[str, Agent]) -> list[ExecutionNode]:
        if len(agents) == 1:
            agent = next(iter(agents.values()))
        else:
            with self.interface.add_step(title="Selecting primary agent...", user="Assistant") as step:
                output = await self._choose_agent(messages, self.agents, primary=True)
                step.stream(output.chain_of_thought, replace=True)
                step.success_title = f"Selected {output.agent}"
                agent = agents[output.agent]

        if agent is None:
            return []

        subagent = agent
        execution_graph = []
        while (unmet_dependencies := tuple(
            r for r in await subagent.requirements(messages) if r not in self._memory
        )):
            with self.interface.add_step(title="Resolving dependencies...", user="Assistant") as step:
                step.stream(f"Found {len(unmet_dependencies)} unmet dependencies: {', '.join(unmet_dependencies)}")
                print(f"\033[91m### Unmet dependencies: {unmet_dependencies}\033[0m")
                subagents = [
                    agent
                    for agent in self.agents
                    if any(ur in agent.provides for ur in unmet_dependencies)
                ]
                output = await self._choose_agent(messages, subagents, unmet_dependencies)
                if output.agent is None:
                    continue
                subagent = agents[output.agent]
                execution_graph.append(
                    ExecutionNode(
                        agent=subagent,
                        provides=unmet_dependencies,
                        instruction=output.chain_of_thought,
                    )
                )
                step.success_title = f"Solved a dependency with {output.agent}"
        return execution_graph[::-1] + [ExecutionNode(agent=agent)]


class Planner(Coordinator):
    """
    The Planner develops a plan to solve the user query step-by-step
    and then executes it.
    """

    @classmethod
    async def _lookup_schemas(
        cls,
        tables: dict[str, Source],
        requested: list[str],
        provided: list[str],
        cache: dict[str, dict] | None = None
    ) -> str:
        cache = cache or {}
        to_query, queries = [], []
        for table in requested:
            if table in provided or table in cache:
                continue
            to_query.append(table)
            queries.append(get_schema(tables[table], table, limit=3))
        for table, schema in zip(to_query, await asyncio.gather(*queries)):
            cache[table] = schema
        schema_info = ''
        for table in requested:
            if table in provided:
                continue
            provided.append(table)
            schema_info += f'- {table}: {cache[table]}\n\n'
        return schema_info

    async def _make_plan(
        self,
        messages: list[Message],
        agents: dict[str, Agent],
        tables: dict[str, Source],
        unmet_dependencies: set[str],
        reason_model: type[BaseModel],
        plan_model: type[BaseModel],
        step: ChatStep,
        schemas: dict[str, dict] | None = None
    ) -> BaseModel:
        user_msg = messages[-1]
        info = ''
        reasoning = None
        requested, provided = [], []
        if 'current_table' in self._memory:
            requested.append(self._memory['current_table'])
        elif len(tables) == 1:
            requested.append(next(iter(tables)))
        while reasoning is None or requested:
            info += await self._lookup_schemas(tables, requested, provided, cache=schemas)
            available = [t for t in tables if t not in provided]
            system = render_template(
                'plan_agent.jinja2',
                agents=list(agents.values()),
                unmet_dependencies=unmet_dependencies,
                memory=self._memory,
                table_info=info,
                tables=available
            )
            async for reasoning in self.llm.stream(
                messages=messages,
                system=system,
                response_model=reason_model,
            ):
                if reasoning.chain_of_thought:  # do not replace with empty string
                    step.stream(reasoning.chain_of_thought, replace=True)
            requested = [
                t for t in getattr(reasoning, 'tables', [])
                if t and t not in provided
            ]
        new_msg = dict(role=user_msg['role'], content=f"<user query>{user_msg['content']}</user query> {reasoning.chain_of_thought}")
        messages = messages[:-1] + [new_msg]
        plan = await self._fill_model(messages, system, plan_model)
        return plan

    async def _resolve_plan(self, plan, agents, messages) -> tuple[list[ExecutionNode], set[str]]:
        step = plan.steps[-1]
        subagent = agents[step.expert]
        unmet_dependencies = {
            r for r in await subagent.requirements(messages) if r not in self._memory
        }
        execution_graph = [
            ExecutionNode(
                agent=subagent,
                provides=unmet_dependencies,
                instruction=step.instruction,
                title=step.title,
                render_output=step.render_output
            )
        ]
        for step in plan.steps[:-1][::-1]:
            subagent = agents[step.expert]
            requires = set(await subagent.requirements(messages))
            unmet_dependencies = {
                dep for dep in (unmet_dependencies | requires)
                if dep not in subagent.provides and dep not in self._memory
            }
            execution_graph.append(
                ExecutionNode(
                    agent=subagent,
                    provides=subagent.provides,
                    instruction=step.instruction,
                    title=step.title,
                    render_output=step.render_output
                )
            )
        return execution_graph, unmet_dependencies

    async def _compute_execution_graph(self, messages: list[Message], agents: dict[str, Agent]) -> list[ExecutionNode]:
        agent_names = tuple(sagent.name[:-5] for sagent in agents.values())
        tables = {}
        for src in self._memory['available_sources']:
            for table in src.get_tables():
                tables[table] = src

        reason_model, plan_model = make_plan_models(agent_names, list(tables))
        planned = False
        unmet_dependencies = set()
        schemas = {}
        with self.interface.add_step(title="Planning how to solve user query...", user="Assistant") as istep:
            while not planned:
                plan = await self._make_plan(
                    messages, agents, tables, unmet_dependencies, reason_model, plan_model, istep, schemas
                )
                execution_graph, unmet_dependencies = await self._resolve_plan(plan, agents, messages)
                if unmet_dependencies:
                    istep.stream(f"The plan didn't account for {unmet_dependencies!r}", replace=True)
                else:
                    planned = True
            self._memory['current_plan'] = plan.title
            istep.stream('\n\nHere are the steps:\n\n')
            for i, step in enumerate(plan.steps):
                istep.stream(f"{i+1}. {step.expert}: {step.instruction}\n")
            istep.success_title = "Successfully came up with a plan"
        return execution_graph[::-1]
