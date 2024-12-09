from __future__ import annotations

import asyncio
import re

from types import FunctionType
from typing import TYPE_CHECKING, Any

import param
import yaml

from panel import Card, bind
from panel.chat import ChatInterface, ChatStep
from panel.layout import Column, FlexBox, Tabs
from panel.pane import HTML
from panel.viewable import Viewer
from panel.widgets import Button
from pydantic import BaseModel

from .actor import Actor
from .agents import (
    Agent, AnalysisAgent, ChatAgent, SQLAgent,
)
from .config import DEMO_MESSAGES, GETTING_STARTED_SUGGESTIONS, PROMPTS_DIR
from .llm import Llama, Llm, Message
from .logs import ChatLogs
from .models import Validity, make_agent_model, make_plan_models
from .tools import FunctionTool, Tool
from .utils import get_schema, retry_llm_output

if TYPE_CHECKING:
    from panel.chat.step import ChatStep

    from ..sources import Source


class ExecutionNode(param.Parameterized):
    """
    Defines a node in an execution graph.
    """

    actor = param.ClassSelector(class_=Actor)

    provides = param.ClassSelector(class_=(set, list), default=set())

    instruction = param.String(default="")

    title = param.String(default="")

    render_output = param.Boolean(default=False)


class Coordinator(Viewer, Actor):
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

    logs_filename = param.String(default=None, doc="""
        Log file to write to.""")

    render_output = param.Boolean(default=True, doc="""
        Whether to write outputs to the ChatInterface.""")

    suggestions = param.List(default=GETTING_STARTED_SUGGESTIONS, doc="""
        Initial list of suggestions of actions the user can take.""")

    tools = param.List(default=[])

    prompts = param.Dict(
        default={
            "check_validity": {
                "template": PROMPTS_DIR / "Coordinator" / "check_validity.jinja2",
                "response_model": Validity,
            },
        }
    )

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
                agent.purpose = f"Available analyses include:\n\n{analyses}\nSelect this agent to perform one of these analyses."
                agent.interface = interface
                self._analyses.extend(agent.analyses)
            if agent.llm is None:
                agent.llm = llm
            # must use the same interface or else nothing shows
            agent.interface = interface
            instantiated.append(agent)

        super().__init__(llm=llm, agents=instantiated, interface=interface, logs_filename=logs_filename, **params)

        self._tools["__main__"] = [
            tool if isinstance(tool, Actor) else (FunctionTool(tool, llm=llm) if isinstance(tool, FunctionType) else tool(llm=llm))
            for tool in self.tools
        ]
        interface.send(
            "Welcome to LumenAI; get started by clicking a suggestion or type your own query below!",
            user="Help", respond=False,
        )
        interface.button_properties={
            "undo": {"callback": on_undo},
            "rerun": {"callback": on_rerun},
        }
        self._add_suggestions_to_footer(self.suggestions)

        if "source" in self._memory and "sources" not in self._memory:
            self._memory["sources"] = [self._memory["source"]]
        elif "source" not in self._memory and self._memory.get("sources"):
            self._memory["source"] = self._memory["sources"][0]
        elif "sources" not in self._memory:
            self._memory["sources"] = []

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

        memory = self._memory
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
                    with agent.param.update(memory=memory):
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
                    disabled=self.interface.param.loading
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
                disabled=self.interface.param.loading
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
        pipeline = self._memory["pipeline"]
        current_analysis = self._memory.get("analysis")
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

    async def _invalidate_memory(self, messages: list[Message]):
        table = self._memory.get("table")
        if not table:
            return

        source = self._memory.get("source")
        if table not in source:
            sources = [src for src in self._memory.get('sources', []) if table in src]
            if sources:
                self._memory["source"] = source = sources[0]
            else:
                raise KeyError(f'Table {table} could not be found in available sources.')

        try:
            spec = await get_schema(source, table=table, include_count=True)
        except Exception:
            # If the selected table cannot be fetched we should invalidate it
            spec = None

        sql = self._memory.get("sql")
        analyses_names = [analysis.__name__ for analysis in self._analyses]
        system = await self._render_prompt(
            "check_validity", messages, table=table, spec=yaml.dump(spec), sql=sql, analyses=analyses_names
        )
        with self.interface.add_step(title="Checking memory...", user="Assistant") as step:
            model_spec = self.prompts["check_validity"].get("llm_spec", "default")
            output = await self.llm.invoke(
                messages=messages,
                system=system,
                model_spec=model_spec,
                response_model=self._get_model("check_validity"),
            )
            step.stream(output.correct_assessment, replace=True)
            step.success_title = f"{output.is_invalid.title()} needs refresh" if output.is_invalid else "Memory still valid"

        if output and output.is_invalid:
            if output.is_invalid == "table":
                self._memory.pop("table", None)
                self._memory.pop("data", None)
                self._memory.pop("sql", None)
                self._memory.pop("pipeline", None)
                self._memory.pop("closest_tables", None)
                print("\033[91mInvalidated from memory.\033[0m")
            elif output.is_invalid == "sql":
                self._memory.pop("sql", None)
                self._memory.pop("data", None)
                self._memory.pop("pipeline", None)
                print("\033[91mInvalidated SQL from memory.\033[0m")
            return output.correct_assessment

    async def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface):
        print("\033[94mNEW\033[0m" + "-" * 100)
        await self.respond(contents)

    @retry_llm_output()
    async def _fill_model(self, messages, system, agent_model, errors=None):
        if errors:
            errors = '\n'.join(errors)
            messages += [{"role": "user", "content": f"\nExpertly resolve these issues:\n{errors}"}]

        model_spec = self.prompts["main"].get("llm_spec", "default")
        out = await self.llm.invoke(
            messages=messages,
            system=system,
            model_spec=model_spec,
            response_model=agent_model
        )
        return out

    async def _execute_graph_node(self, node: ExecutionNode, messages: list[Message]):
        subagent = node.actor
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
                    custom_analysis_doc = custom_agent.purpose.replace("Available analyses include:\n", "")
                    custom_message = (
                        f"Avoid doing the same analysis as any of these custom analyses: {custom_analysis_doc} "
                        f"Most likely, you'll just need to do a simple SELECT * FROM {{table}};"
                    )
                    custom_messages.append({"role": "user", "content": custom_message})
            if instruction:
                custom_messages.append({"role": "assistant", "content": instruction})

            shared_ctx = {"memory": self.memory}
            respond_kwargs = {"agents": self.agents} if isinstance(subagent, AnalysisAgent) else {}
            if isinstance(subagent, Agent):
                shared_ctx["steps_layout"] = steps_layout
            with subagent.param.update(**shared_ctx):
                result = await subagent.respond(
                    custom_messages,
                    step_title=title,
                    render_output=render_output,
                    **respond_kwargs
                )
            step.stream(f"\n\n`{agent_name}` agent successfully completed the following task:\n\n> {instruction}", replace=True)
            if isinstance(subagent, Tool) and result:
                self._memory["tool_context"] += '\n\n'+result
                step.stream('\n\n'+result)
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

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        self._memory["tool_context"] = ""
        with self.interface.param.update(loading=True):
            if isinstance(self.llm, Llama):
                with self.interface.add_step(title="Loading Llama model...", success_title="Using the cached Llama model", user="Assistant") as step:
                    default_kwargs = self.llm.model_kwargs["default"]
                    step.stream(f"Model: `{default_kwargs['repo']}/{default_kwargs['model_file']}`")
                    await self.llm.get_client("default")  # caches the model for future use

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
            if "pipeline" in self._memory:
                await self._add_analysis_suggestions()
            print("\033[92mDONE\033[0m", "\n\n")


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
            "check_validity": {
                "template": PROMPTS_DIR / "Coordinator" / "check_validity.jinja2",
                "response_model": Validity,
            },
        },
    )

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
        tools = self._tools["__main__"]
        agent_names = tuple(sagent.name[:-5] for sagent in agents) + tuple(tool.name for tool in tools)
        agent_model = self._get_model("main", agent_names=agent_names, primary=primary)
        if len(agent_names) == 0:
            raise ValueError("No agents available to choose from.")
        if len(agent_names) == 1:
            return agent_model(agent=agent_names[0], chain_of_thought='')
        system = await self._render_prompt(
            "main", messages, agents=agents, tools=tools, primary=primary,
            unmet_dependencies=unmet_dependencies
        )
        return await self._fill_model(messages, system, agent_model)

    async def _compute_execution_graph(self, messages, agents: dict[str, Agent]) -> list[ExecutionNode]:
        if len(agents) == 1:
            agent = next(iter(agents.values()))
        else:
            tools = {tool.name: tool for tool in self._tools["__main__"]}
            agent = None
            with self.interface.add_step(title="Selecting primary agent...", user="Assistant") as step:
                try:
                    output = await self._choose_agent(messages, self.agents, primary=True)
                except Exception as e:
                    if self.interface.callback_exception not in ('raise', 'verbose'):
                        step.failed_title = 'Failed to select main agent...'
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
            return []

        cot = output.chain_of_thought
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
                if output.agent_or_tool is None:
                    continue
                if output.agent_or_tool in tools:
                    subagent = tools[output.agent_or_tool]
                else:
                    subagent = agents[output.agent_or_tool]
                execution_graph.append(
                    ExecutionNode(
                        actor=subagent,
                        provides=unmet_dependencies,
                        instruction=output.chain_of_thought,
                    )
                )
                step.success_title = f"Solved a dependency with {output.agent_or_tool}"
        return execution_graph[::-1] + [ExecutionNode(actor=agent, instruction=cot)]


class Planner(Coordinator):
    """
    The Planner develops a plan to solve the user query step-by-step
    and then executes it.
    """

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "Planner" / "main.jinja2",
                "response_model": make_plan_models,
            },
            "check_validity": {
                "template": PROMPTS_DIR / "Coordinator" / "check_validity.jinja2",
                "response_model": Validity,
            },
        }
    )

    @classmethod
    async def _lookup_schemas(
        cls,
        tables: dict[str, Source],
        requested: list[str],
        provided: list[str],
        cache: dict[str, dict] | None = None
    ) -> str:
        cache = cache or {}
        for table in requested:
            if table in provided or table in cache:
                continue
            cache[table] = await get_schema(tables[table], table, limit=1000)
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
        info = ''
        reasoning = None
        tools = self._tools["__main__"]
        requested, provided = [], []
        if "table" in self._memory:
            requested.append(self._memory["table"])
        elif len(tables) == 1:
            requested.append(next(iter(tables)))
        while reasoning is None or requested:
            info += await self._lookup_schemas(tables, requested, provided, cache=schemas)
            available = [t for t in tables if t not in provided]
            system = await self._render_prompt(
                "main",
                messages,
                agents=list(agents.values()),
                tools=tools,
                unmet_dependencies=unmet_dependencies,
                table_info=info,
                tables=available
            )
            model_spec = self.prompts["main"].get("llm_spec", "default")
            reasoning = await self.llm.invoke(
                messages=messages,
                system=system,
                model_spec=model_spec,
                response_model=reason_model,
                max_retries=3,
            )
            if reasoning.chain_of_thought:  # do not replace with empty string
                step.stream(reasoning.chain_of_thought, replace=True)
            requested = [
                t for t in getattr(reasoning, 'tables', [])
                if t and t not in provided
            ]
        new_msg = dict(
            role='assistant',
            content=reasoning.chain_of_thought
        )
        messages = messages + [new_msg]
        return await self._fill_model(messages, system, plan_model)

    async def _resolve_plan(self, plan, agents, messages) -> tuple[list[ExecutionNode], set[str]]:
        table_provided = False
        execution_graph = []
        provided = set(self._memory)
        unmet_dependencies = set()
        tools = {tool.name: tool for tool in self._tools["__main__"]}
        steps = []
        for step in plan.steps:
            key = step.expert_or_tool
            if key in agents:
                subagent = agents[key]
            elif key in tools:
                subagent = tools[key]
            requires = set(await subagent.requirements(messages))
            provided |= set(subagent.provides)
            unmet_dependencies = (unmet_dependencies | requires) - provided
            if "table" in unmet_dependencies and not table_provided and "SQLAgent" in agents:
                provided |= set(agents['SQLAgent'].provides)
                sql_step = type(step)(
                    expert_or_tool='SQLAgent',
                    instruction='Load the table',
                    title='Loading table',
                    render_output=False
                )
                execution_graph.append(
                    ExecutionNode(
                        actor=agents['SQLAgent'],
                        provides=['table'],
                        instruction=sql_step.instruction,
                        title=sql_step.title,
                        render_output=False
                    )
                )
                steps.append(sql_step)
                table_provided = True
                unmet_dependencies -= provided
            execution_graph.append(
                ExecutionNode(
                    actor=subagent,
                    provides=subagent.provides,
                    instruction=step.instruction,
                    title=step.title,
                    render_output=step.render_output
                )
            )
            steps.append(step)
        last_node = execution_graph[-1]
        if isinstance(last_node.actor, Tool):
            if "AnalystAgent" in agents and all(r in provided for r in agents["AnalystAgent"].requires):
                expert = "AnalystAgent"
            else:
                expert = "ChatAgent"
            summarize_step = type(step)(
                expert_or_tool=expert,
                instruction='Summarize the results.',
                title='Summarizing results',
                render_output=False
            )
            steps.append(summarize_step)
            execution_graph.append(
                ExecutionNode(
                    actor=agents[expert],
                    provides=[],
                    instruction=summarize_step.instruction,
                    title=summarize_step.title,
                    render_output=summarize_step.render_output
                )
            )
        plan.steps = steps
        return execution_graph, unmet_dependencies

    async def _compute_execution_graph(self, messages: list[Message], agents: dict[str, Agent]) -> list[ExecutionNode]:
        tool_names = [tool.name for tool in self._tools["__main__"]]
        agent_names = [sagent.name[:-5] for sagent in agents.values()]
        tables = {}
        for src in self._memory['sources']:
            for table in src.get_tables():
                tables[table] = src

        reason_model, plan_model = self._get_model(
            "main",
            experts_or_tools=agent_names+tool_names,
            tables=list(tables)
        )
        planned = False
        unmet_dependencies = set()
        schemas = {}
        execution_graph = []
        attempts = 0
        with self.interface.add_step(title="Planning how to solve user query...", user="Assistant") as istep:
            while not planned:
                try:
                    plan = await self._make_plan(
                        messages, agents, tables, unmet_dependencies, reason_model, plan_model, istep, schemas
                    )
                except Exception as e:
                    if self.interface.callback_exception not in ('raise', 'verbose'):
                        istep.failed_title = 'Failed to make plan. Ensure LLM is configured correctly and/or try again.'
                        istep.stream(str(e), replace=True)
                    else:
                        raise e
                execution_graph, unmet_dependencies = await self._resolve_plan(plan, agents, messages)
                if unmet_dependencies:
                    istep.stream(f"The plan didn't account for {unmet_dependencies!r}", replace=True)
                    attempts += 1
                else:
                    planned = True
                if attempts > 5:
                    raise ValueError(f"Could not find a suitable plan for {messages[-1]['content']!r}")
            self._memory['plan'] = plan
            istep.stream('\n\nHere are the steps:\n\n')
            for i, step in enumerate(plan.steps):
                istep.stream(f"{i+1}. {step.expert_or_tool}: {step.instruction}\n")
            istep.success_title = "Successfully came up with a plan"
        return execution_graph
