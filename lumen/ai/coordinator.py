from __future__ import annotations

import asyncio
import re
import traceback

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import param

from panel import bind
from panel.chat import ChatFeed, ChatInterface, ChatStep
from panel.layout import (
    Card, Column, FlexBox, Tabs,
)
from panel.pane import HTML
from panel.viewable import Viewable, Viewer
from panel.widgets import Button
from pydantic import BaseModel

from ..views.base import Panel, View
from .actor import Actor
from .agents import (
    Agent, AnalysisAgent, ChatAgent, SQLAgent,
)
from .config import DEMO_MESSAGES, GETTING_STARTED_SUGGESTIONS, PROMPTS_DIR
from .llm import LlamaCpp, Llm, Message
from .logs import ChatLogs
from .models import YesNo, make_agent_model, make_plan_models
from .tools import (
    IterativeTableLookup, TableLookup, Tool, ToolUser,
)
from .utils import (
    fuse_messages, log_debug, mutate_user_message, retry_llm_output,
    stream_details,
)
from .views import LumenOutput

if TYPE_CHECKING:
    from panel.chat.step import ChatStep


UI_INTRO_MESSAGE = """
👋 Click a suggestion below or upload a data source to get started!

Lumen AI combines large language models (LLMs) with specialized agents to help you explore, analyze,
and visualize data without writing code.

On the chat interface...

💬 Ask questions in plain English to generate SQL queries and visualizations
🔍 Inspect and validate results through conversation
📝 Get summaries and key insights from your data
🧩 Apply custom analyses with a click of a button

Click the toggle, or drag the edge, to expand the sidebar and...

📚 Upload sources (tables and documents) by dragging or selecting files
🌐 Explore data with [Graphic Walker](https://docs.kanaries.net/graphic-walker) - filter, sort, download
💾 Access all generated tables and visualizations under tabs
📤 Export your session as a reproducible notebook

📖 Learn more about [Lumen AI](https://lumen.holoviz.org/lumen_ai/getting_started/using_lumen_ai.html)
"""


class ExecutionNode(param.Parameterized):
    """
    Defines a node in an execution graph.
    """

    actor = param.ClassSelector(class_=Actor)

    provides = param.ClassSelector(class_=(set, list), default=set())

    instruction = param.String(default="")

    title = param.String(default="")

    render_output = param.Boolean(default=False)


class Coordinator(Viewer, ToolUser):
    """
    A Coordinator is responsible for coordinating the actions
    of a number of agents towards the user defined query by
    computing an execution graph and then executing each
    step along the graph.
    """

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "Coordinator" / "main.jinja2",
            },
            "yesno_update": {
                "template": PROMPTS_DIR / "Coordinator" / "yesno_update.jinja2",
                "response_model": YesNo,
            },
            "tool_relevance": {
                "template": PROMPTS_DIR / "Coordinator" / "tool_relevance.jinja2",
                "response_model": YesNo,
            },
        },
    )

    within_ui = param.Boolean(default=False, constant=True, doc="""
        Whether this coordinator is being used within the UI.""")

    agents = param.List(default=[ChatAgent], doc="""
        List of agents to coordinate.""")

    demo_inputs = param.List(default=DEMO_MESSAGES, doc="""
        List of instructions to demo the Coordinator.""")

    history = param.Integer(default=3, doc="""
        Number of previous user-assistant interactions to include in the chat history.""")

    logs_db_path = param.String(default=None, doc="""
        The path to the log file that will store the messages exchanged with the LLM.""")

    render_output = param.Boolean(default=True, doc="""
        Whether to write outputs to the ChatInterface.""")

    suggestions = param.List(default=GETTING_STARTED_SUGGESTIONS, doc="""
        Initial list of suggestions of actions the user can take.""")

    tools = param.List(default=[], doc="""
        List of tools to use to provide context.""")

    __abstract = True

    def __init__(
        self,
        llm: Llm | None = None,
        interface: ChatFeed | ChatInterface | None = None,
        agents: list[Agent | type[Agent]] | None = None,
        tools: list[Tool | type[Tool]] | None = None,
        logs_db_path: str = "",
        **params,
    ):
        log_debug("New Session: \033[92mStarted\033[0m", show_sep="above")

        def on_message(message, instance):
            def update_on_reaction(reactions):
                if not self._logs:
                    return
                self._logs.update_status(
                    message_id=message_id,
                    liked="like" in reactions,
                    disliked="dislike" in reactions,
                )

            bind(update_on_reaction, message.param.reactions, watch=True)
            message_id = id(message)
            message_index = instance.objects.index(message)
            self._logs.upsert(
                session_id=self._session_id,
                message_id=message_id,
                message_index=message_index,
                message_user=message.user,
                message_content=message.serialize(),
            )

        def on_undo(instance, _):
            if not self._logs:
                return
            count = instance._get_last_user_entry_index()
            messages = instance[-count:]
            for message in messages:
                self._logs.update_status(message_id=id(message), removed=True)

        def on_rerun(instance, _):
            if not self._logs:
                return
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

        self._session_id = id(self)

        if logs_db_path:
            interface.message_params["reaction_icons"] = {"like": "thumb-up", "dislike": "thumb-down"}
            self._logs = ChatLogs(filename=logs_db_path)
            interface.post_hook = on_message
        else:
            interface.message_params["show_reaction_icons"] = False
            self._logs = None

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

        # Add user-provided tools to the list of tools of the coordinator
        self.prompts["main"]["tools"] += [tool for tool in tools]
        super().__init__(llm=llm, agents=instantiated, interface=interface, logs_db_path=logs_db_path, **params)

        welcome_message = UI_INTRO_MESSAGE if self.within_ui else "Welcome to LumenAI; get started by clicking a suggestion or type your own query below!"
        interface.send(
            welcome_message,
            user="Help", respond=False, show_reaction_icons=False, show_copy_icon=False
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
        num_objects: int = 2,
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
                        log_debug("No analysis agent found.")
                        return
                    messages = [{"role": "user", "content": contents}]
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

    async def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface):
        log_debug(f"New Message: \033[91m{contents!r}\033[0m", show_sep="above")
        await self.respond(contents)

    @retry_llm_output()
    async def _fill_model(self, messages, system, agent_model, errors=None):
        if errors:
            errors = '\n'.join(errors)
            messages = mutate_user_message(
                f"\n\nThe following are errors that previously came up; be sure to keep them in mind:\n{errors}",
                messages
            )

        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
        out = await self.llm.invoke(
            messages=messages,
            system=system,
            model_spec=model_spec,
            response_model=agent_model
        )
        return out

    async def _execute_graph_node(self, node: ExecutionNode, messages: list[Message], execution_graph: list[ExecutionNode] | None = None) -> bool:
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

        mutated_messages = deepcopy(messages)
        with self.interface.add_step(title=f"Querying {agent_name} agent...", steps_layout=steps_layout) as step:
            step.stream(f"`{agent_name}` agent is working on the following task:\n\n{instruction}")
            if isinstance(subagent, SQLAgent):
                custom_agent = next((a for a in self.agents if isinstance(a, AnalysisAgent)), None)
                if custom_agent:
                    custom_analysis_doc = custom_agent.purpose.replace("Available analyses include:\n", "")
                    custom_message = (
                        f"Avoid doing the same analysis as any of these custom analyses: {custom_analysis_doc!r} "
                        f"Most likely, you'll just need to do a simple SELECT * FROM {{table}};"
                    )
                    mutated_messages = mutate_user_message(custom_message, mutated_messages)
            if instruction:
                content = f"-- For context...\nHere's part of the multi-step plan: {instruction!r}, but as the expert, you may need to deviate from it if you notice any inconsistencies or issues."
                mutate_user_message(
                    content,
                    mutated_messages, suffix=True, wrap=True
                )

            shared_ctx = {"memory": self.memory}
            respond_kwargs = {"agents": self.agents} if isinstance(subagent, AnalysisAgent) else {}
            if isinstance(subagent, Agent):
                shared_ctx["steps_layout"] = steps_layout
            with subagent.param.update(**shared_ctx):
                try:
                    result = await subagent.respond(
                        mutated_messages,
                        step_title=title,
                        render_output=render_output,
                        **respond_kwargs
                    )
                except asyncio.CancelledError as e:
                    step.failed_title = f"{agent_name} agent was cancelled"
                    raise e
                except Exception as e:
                    self._memory['__error__'] = str(e)
                    raise e
                log_debug(f"\033[96mCompleted: {agent_name}\033[0m", show_length=False)

            unprovided = [p for p in subagent.provides if p not in self._memory]
            if (unprovided and agent_name != "Source") or (len(unprovided) > 1 and agent_name == "Source"):
                step.failed_title = f"{agent_name}Agent did not provide {', '.join(unprovided)}. Aborting the plan."
                raise RuntimeError(f"{agent_name} failed to provide declared context.")
            step.stream(f"\n\n`{agent_name}` agent successfully completed the following task:\n\n> {instruction}", replace=True)
            # Handle Tool specific behaviors
            if isinstance(subagent, Tool):
                # Handle string results from tools
                if isinstance(result, str) and result:
                    await self._handle_tool_result(subagent, result, step, execution_graph, messages)
                # Handle View/Viewable results regardless of agent type
                if isinstance(result, (View, Viewable)):
                    if isinstance(result, Viewable):
                        result = Panel(object=result, pipeline=self._memory.get('pipeline'))
                    out = LumenOutput(
                        component=result, render_output=render_output, title=title
                    )
                    if 'outputs' in self._memory:
                        # We have to create a new list to trigger an event
                        # since inplace updates will not trigger updates
                        # and won't allow diffing between old and new values
                        self._memory['outputs'] = self._memory['outputs']+[out]
                    message_kwargs = dict(value=out, user=subagent.name)
                    self.interface.stream(**message_kwargs)
            step.success_title = f"{agent_name} agent successfully responded"
        return step.status == "success"

    def _serialize(self, obj: Any, exclude_passwords: bool = True) -> str:
        if isinstance(obj, (Column, Card, Tabs)):
            string = ""
            for o in obj:
                if isinstance(o, ChatStep):
                    # Drop context from steps; should be irrelevant now
                    continue
                string += self._serialize(o)
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
        self._memory["agent_tool_contexts"] = {}
        with self.interface.param.update(loading=True):
            if isinstance(self.llm, LlamaCpp):
                with self.interface.add_step(title="Loading LlamaCpp model...", success_title="Using the cached LlamaCpp model", user="Assistant") as step:
                    default_kwargs = self.llm.model_kwargs["default"]
                    step.stream(f"Model: `{default_kwargs['repo']}/{default_kwargs['model_file']}`")
                    await self.llm.get_client("default")  # caches the model for future use

            messages = fuse_messages(
                self.interface.serialize(custom_serializer=self._serialize, limit=10),
                max_user_messages=self.history
            )

            agents = {agent.name[:-5]: agent for agent in self.agents}
            execution_graph = await self._compute_execution_graph(messages, agents)
            if execution_graph is None:
                msg = (
                    "Assistant could not settle on a plan of action to perform the requested query. "
                    "Please restate your request."
                )
                self.interface.stream(msg, user='Lumen')
                return msg
            for i, node in enumerate(execution_graph):
                succeeded = await self._execute_graph_node(node, messages, execution_graph=execution_graph)
                if not succeeded:
                    break
            if "pipeline" in self._memory:
                await self._add_analysis_suggestions()
            log_debug("\033[92mCompleted: Coordinator\033[0m", show_sep="below")

        for message_obj in self.interface.objects[::-1]:
            if isinstance(message_obj.object, Card):
                message_obj.object.collapsed = True
                break

    async def _check_tool_relevance(self, tool_name: str, tool_output: str, agent: Agent, agent_task: str, messages: list[Message]) -> bool:
        tool = next((t for t in self._tools["main"] if t.name == tool_name), None)
        result = await self._invoke_prompt(
            "tool_relevance",
            messages,
            tool_name=tool_name,
            tool_description=getattr(tool, "description", ""),
            tool_output=tool_output,
            agent_name=agent.name,
            agent_purpose=agent.purpose,
            agent_task=agent_task,
        )

        return result.yes

    async def _handle_tool_result(self, tool: Tool, result: str, step: ChatStep, execution_graph: list[ExecutionNode] | None = None, messages: list[Message] | None = None):
        """Handle string tool results and determine relevance for future agents."""
        # Caller should ensure result is a non-empty string before calling this method

        # Display the result
        stream_details(result, step, title="Results", auto=False)

        # Early exit if no execution graph provided
        if not execution_graph:
            return

        # Find agents that will be used in future nodes
        future_agents = {}
        for agent in self.agents:
            # Skip tools, only interested in non-tool agents
            if isinstance(agent, Tool):
                continue

            # Find if this agent appears in future nodes
            for future_node in execution_graph:
                if future_node.actor is agent:
                    future_agents[agent] = future_node.instruction
                    break

        # Early exit if no future agents found
        if not future_agents:
            return

        # Check relevance and store context for each future agent
        for agent, task in future_agents.items():
            # Get tool and agent provides/requires lists
            tool_provides = getattr(tool, "provides", [])
            agent_requires = getattr(agent, "requires", [])

            # If tool provides at least one thing the agent requires, consider it relevant
            # without performing the more expensive relevance check
            direct_dependency = any(provided in agent_requires for provided in tool_provides)

            is_relevant = False
            if direct_dependency:
                log_debug(f"Direct dependency detected: {tool.name} provides at least one requirement for {agent.name}")
                # The agent already has it formatted in its template
                continue
            else:
                # Otherwise, check semantic relevance
                is_relevant = await self._check_tool_relevance(
                    tool.name, result, agent, task, messages
                )

            if is_relevant:
                # Initialize agent_tool_contexts if needed
                if agent.name not in self._memory["agent_tool_contexts"]:
                    self._memory["agent_tool_contexts"][agent.name] = {}

                # Store the tool output in agent's context
                self._memory["agent_tool_contexts"][agent.name][tool.name] = result
                log_debug(f"Added {tool.name} output to {agent.name}'s context")


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
        self,
        messages: list[Message],
        agents: list[Agent] | None = None,
        primary: bool = False,
        unmet_dependencies: tuple[str] | None = None
    ):
        if agents is None:
            agents = self.agents
        agents = [agent for agent in agents if await agent.applies(self._memory)]
        tools = self._tools["main"]
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
            tools = {tool.name: tool for tool in self._tools["main"]}
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
                log_debug(f"\033[91m### Unmet dependencies: {unmet_dependencies}\033[0m")
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
                "tools": [TableLookup, IterativeTableLookup]
            },
        }
    )

    async def _make_plan(
        self,
        messages: list[Message],
        agents: dict[str, Agent],
        unmet_dependencies: set[str],
        previous_plans: list[str],
        reason_model: type[BaseModel],
        plan_model: type[BaseModel],
        step: ChatStep,
    ) -> BaseModel:
        tools = self._tools["main"]
        reasoning = None
        while reasoning is None:
            # candidates = agents and tools that can provide
            # the unmet dependencies
            agent_candidates = [
                agent for agent in agents.values()
                if not unmet_dependencies or set(agent.provides) & unmet_dependencies
            ]
            tool_candidates = [
                tool for tool in tools
                if not unmet_dependencies or set(tool.provides) & unmet_dependencies
            ]
            system = await self._render_prompt(
                "main",
                messages,
                agents=list(agents.values()),
                tools=list(tools),
                unmet_dependencies=unmet_dependencies,
                # Gather candidates that can provide unmet dependencies
                candidates = agent_candidates + tool_candidates,
                previous_plans=previous_plans,
            )
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            async for reasoning in self.llm.stream(
                messages=messages,
                system=system,
                model_spec=model_spec,
                response_model=reason_model,
                max_retries=3,
            ):
                if reasoning.chain_of_thought:  # do not replace with empty string
                    self._memory["reasoning"] = reasoning.chain_of_thought
                    step.stream(reasoning.chain_of_thought, replace=True)
            previous_plans.append(reasoning.chain_of_thought)
        mutated_messages = mutate_user_message(
            f"Follow this latest plan: {reasoning.chain_of_thought!r} to finish answering: ",
            deepcopy(messages),
            suffix=False,
            wrap=True,
        )
        return await self._fill_model(mutated_messages, system, plan_model)

    async def _resolve_plan(self, plan, agents, messages) -> tuple[list[ExecutionNode], set[str]]:
        table_provided = False
        execution_graph = []
        provided = set(self._memory)
        unmet_dependencies = set()
        tools = {tool.name: tool for tool in self._tools["main"]}
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
            has_table_lookup = any(
                isinstance(node.actor, TableLookup)
                for node in execution_graph
            )
            if "table" in unmet_dependencies and not table_provided and "SQLAgent" in agents and has_table_lookup:
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
        if isinstance(last_node.actor, Tool) and not isinstance(last_node.actor, TableLookup):
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
        tool_names = [tool.name for tool in self._tools["main"]]
        agent_names = [sagent.name[:-5] for sagent in agents.values()]

        reason_model, plan_model = self._get_model(
            "main",
            agents=agent_names,
            tools=tool_names,
        )
        planned = False
        unmet_dependencies = set()
        execution_graph = []
        previous_plans = []
        attempts = 0

        with self.interface.add_step(title="Planning how to solve user query...", user="Assistant") as istep:
            while not planned:
                if attempts > 0:
                    log_debug(f"\033[91m!! Attempt {attempts}\033[0m")
                plan = None
                try:
                    plan = await self._make_plan(
                        messages, agents, unmet_dependencies, previous_plans, reason_model, plan_model, istep
                    )
                except asyncio.CancelledError as e:
                    istep.failed_title = 'Planning was cancelled, please try again.'
                    traceback.print_exception(e)
                    raise e
                except Exception as e:
                    istep.failed_title = 'Failed to make plan. Ensure LLM is configured correctly and/or try again.'
                    traceback.print_exception(e)
                    raise e
                execution_graph, unmet_dependencies = await self._resolve_plan(plan, agents, messages)
                if unmet_dependencies:
                    istep.stream(f"The plan didn't account for {unmet_dependencies!r}", replace=True)
                    attempts += 1
                else:
                    planned = True
                if attempts > 5:
                    istep.failed_title = "Planning failed to come up with viable plan, please restate the problem and try again."
                    e = RuntimeError("Planner failed to come up with viable plan after 5 attempts.")
                    traceback.print_exception(e)
                    raise e
            self._memory["plan"] = plan
            istep.stream('\n\nHere are the steps:\n\n')
            for i, step in enumerate(plan.steps):
                istep.stream(f"{i+1}. {step.expert_or_tool}: {step.instruction}\n")
            if attempts > 0:
                istep.success_title = f"Plan with {len(plan.steps)} steps created after {attempts + 1} attempts"
            else:
                istep.success_title = f"Plan with {len(plan.steps)} steps created"
        return execution_graph
