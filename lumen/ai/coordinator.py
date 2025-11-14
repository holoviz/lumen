from __future__ import annotations

import asyncio
import re
import traceback

from typing import TYPE_CHECKING, Any

import param

from panel.chat import ChatFeed
from panel.pane import HTML
from panel.viewable import Viewer
from panel_material_ui import (
    Card, ChatInterface, ChatStep, Column, Tabs, Typography,
)
from pydantic import BaseModel
from typing_extensions import Self

from .actor import Actor
from .agents import Agent, AnalysisAgent, ChatAgent
from .config import PROMPTS_DIR, MissingContextError
from .context import ContextError, TContext
from .llm import LlamaCpp, Llm, Message
from .models import (
    RawPlan, Reasoning, ThinkingYesNo, make_agent_model, make_plan_model,
)
from .report import ActorTask, Section, TaskWrapper
from .tools import (
    IterativeTableLookup, TableLookup, Tool, VectorLookupToolUser,
)
from .utils import (
    fuse_messages, get_root_exception, log_debug, mutate_user_message,
    normalized_name, wrap_logfire,
)

if TYPE_CHECKING:
    from panel.chat.step import ChatStep

    from .vector_store import VectorStore


class Plan(Section):
    """
    A Plan is a Task that is a collection of other Tasks.
    """

    abort_on_error = param.Boolean(default=True, doc="""
        If True, the report will abort if an error occurs.""")

    agents = param.List(item_type=Actor, default=[])

    coordinator = param.ClassSelector(class_=param.Parameterized)

    interface = param.ClassSelector(class_=ChatFeed)

    _tasks = param.List(item_type=ActorTask)

    def render_task_history(self, i: int, failed: bool = False) -> tuple[list[Message], str]:
        user_query = None
        for msg in reversed(self.history):
            if msg.get('role') == 'user':
                user_query = msg
                break

        todos_list = []
        for idx, task in enumerate(self):
            instruction = task.instruction
            if failed and idx == i:
                status = "❌"
            else:
                if i == idx:
                    instruction = f"<u>{instruction}</u>"
                if idx < i:
                    status = "x"
                else:
                    status = " "
                status = f"[{status}]"
            todos_list.append(f"- {status} {instruction}")
        todos = "\n".join(todos_list)

        formatted_content = f"User Request: {user_query['content']!r}\n\nComplete the underlined todo:\n{todos}"
        rendered_history = []
        for msg in self.history:
            if msg is user_query:
                rendered_history.append({'content': formatted_content, 'role': 'user'})
            else:
                rendered_history.append(msg)
        return rendered_history, todos

    async def _run_task(self, i: int, task: Self | Actor, context: TContext, **kwargs):
        outputs = []
        with self._add_step(
            title=f"{task.title}...",
            user="Runner",
            layout_params={"title": "🏗️ Running "},
            steps_layout=self.steps_layout
        ) as step:
            self.steps_layout.header[0].object = f"⚙️ Working on task {task.title!r}..."
            step.stream(task.instruction)
            history, todos = self.render_task_history(i)
            subcontext = self._get_context(i, context, task)
            self.steps_layout.header[1].object = todos
            try:
                kwargs = {"agents": self.agents} if 'agents' in task.param else {}
                with task.param.update(
                    interface=self.interface, steps_layout=self.steps_layout[-1],
                    history=history, **kwargs
                ):
                    outputs, task_context = await task.execute(subcontext, **kwargs)
            except Exception as e:
                outputs, task_context = await self._handle_task_execution_error(
                    e, task, context, step, i
                )
            unprovided = [
                p for actor in task for p in actor.output_schema.__required_keys__
                if p not in task_context
            ]
            if unprovided:
                step.failed_title = (
                    f"{task.title!r} failed to provide declared context values {', '.join(unprovided)}. "
                    "Aborting the plan."
                )
                raise RuntimeError(f"{task.title!r} task failed to provide declared context.")
            elif task.status == "error":
                step.failed_title = f"{task.title} errored during execution."
                raise RuntimeError(step.failed_title)
            context_keys = ", ".join(f'`{k}`' for k in task_context)
            step.stream(f"Generated {len(outputs)} and provided {context_keys}.")
            step.success_title = f"Task {task.title!r} successfully completed"
            log_debug(f"\033[96mCompleted: {task.title}\033[0m", show_length=False)
        return outputs, task_context

    async def _handle_task_execution_error(
        self, e: Exception, task: Self | Actor, context: TContext, step: ChatStep, i: int
    ) -> tuple[list[Any], TContext]:
        """
        Handle exceptions that occur during task execution.
        Returns outputs if the error was handled successfully, None if the error should be re-raised.
        """
        # Check if this is a MissingContextError (possibly wrapped)
        root_exception = get_root_exception(e, exceptions=(MissingContextError,))

        if isinstance(root_exception, MissingContextError):
            # Handle missing context by finding and re-running the provider
            step.failed_title = f"{task.title} - Missing context, retrying..."
            log_debug(f"\033[93mMissing context detected: {root_exception!s}\033[0m")

            # Find which agent provided the pipeline or relevant context
            provider_index = self._find_context_provider(i, context)
            if provider_index is not None:
                # Re-run from the provider with the error as feedback
                outputs, out_context = await self._retry_from_provider(
                    provider_index, i, str(root_exception), context
                )
                step.success_title = f"{task.title!r} task successfully completed after retry"
                return outputs, out_context  # Return after successful retry
            else:
                # If we can't find a provider, raise the original error
                step.failed_title = f"{task.title!r} - Cannot resolve missing context"
                traceback.print_exception(e)
                raise e
        elif isinstance(e, asyncio.CancelledError):
            step.failed_title = f"{task.title!r} task was cancelled"
            raise e
        else:
            traceback.print_exception(e)
            step.failed_title = f"{task.title!r} task failed due to an internal error"
            raise e

    def _find_context_provider(self, failed_index: int, context: TContext) -> int | None:
        """
        Find the task that provided the pipeline or other relevant context.
        Search backwards from the failed task.
        """
        pipeline_exists = 'pipeline' in context
        if not pipeline_exists:
            return

        for idx in reversed(range(failed_index)):
            task = self._tasks[idx]
            # Check if this task provides pipeline or other relevant context
            if isinstance(task, TaskWrapper):
                for actor in task:
                    if 'pipeline' in actor.output_schema.__annotations__:
                        return idx

    async def _retry_from_provider(
        self, provider_index: int, failed_index: int, error_message: str, context: TContext
    ) -> tuple[list[Any], TContext]:
        """
        Re-run the plan from the provider task with additional feedback about the error.
        """

        outputs = []
        # Create feedback suffix to add to user message
        feedback_suffix = (
            f"Be mindful of: {self._tasks[failed_index].title!r}"
            f"Previously, it was reported that: {error_message!r}. "
            "Try a different approach."
        )
        # Re-run from the provider until the failed task
        for idx in range(provider_index, failed_index+1):
            task = self[idx]
            task.reset()
            if idx < failed_index:
                out, out_context = await self._run_task(idx, task, context)
                continue

            # For the provider task, mutate the user message to include feedback
            retry_history = mutate_user_message(feedback_suffix, self.history.copy())
            # Show retry in UI
            with self._add_step(
                title=f"🔄 Retrying {task.title} with feedback...",
                user="Runner",
                steps_layout=self.steps_layout
            ) as retry_step:
                retry_step.stream(f"Retrying with feedback about: {error_message}")

                # Update todos to show retry
                todos = '\n'.join(
                    f"- [{'x' if tidx < idx else '🔄' if tidx == idx else ' '}] {'<u>' + t.instruction + '</u>' if tidx == idx else t.instruction}"
                    for tidx, t in enumerate(self)
                )
                self.steps_layout.headers[1].object = todos

                # Run with mutated history
                kwargs = {"agents": self.agents} if 'agents' in task.param else {}
                subcontext = self._get_context(idx, context, task)
                with task.param.update(
                    interface=self.interface, steps_layout=self.steps_layout,
                    history=retry_history, **kwargs
                ):
                    out, out_context = await task.execute(subcontext, **kwargs)
                retry_step.success_title = f"✅ {task.title} successfully completed on retry"
        return outputs, out_context

    async def execute(self, context: TContext = None, **kwargs) -> list[Any]:
        context = context or self.context
        if '__error__' in context:
            del context['__error__']
        outputs, out_context = await super().execute(context, **kwargs)
        _, todos = self.render_task_history(self._current, failed=self.status == "error")
        steps_title, todo_list = self.steps_layout.header
        todo_list.object = todos
        if self.status == 'success':
            steps_title.object = f"✅ Sucessfully completed {self.title!r}"
        else:
            steps_title.object = f"❌ Failed to execute {self.title!r}"
        log_debug("\033[92mCompleted: Plan\033[0m", show_sep="below")
        if self.interface is not None:
            self.steps_layout.collapsed = True
        return outputs, out_context


class Coordinator(Viewer, VectorLookupToolUser):
    """
    A Coordinator is responsible for coordinating the actions
    of a number of agents towards the user defined query by
    computing an execution graph and then executing each
    step along the graph.
    """

    agents = param.List(default=[ChatAgent], doc="""
        List of agents to coordinate.""")

    context = param.Dict(default={})

    history = param.Integer(default=3, doc="""
        Number of previous user-assistant interactions to include in the chat history.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "Coordinator" / "main.jinja2",
            },
            "tool_relevance": {
                "template": PROMPTS_DIR / "Coordinator" / "tool_relevance.jinja2",
                "response_model": ThinkingYesNo,
            },
        },
    )

    verbose = param.Boolean(default=False, allow_refs=True, doc="""
        Whether to show verbose output.""")

    within_ui = param.Boolean(default=False, constant=True, doc="""
        Whether this coordinator is being used within the UI.""")

    validation_enabled = param.Boolean(default=True, doc="""
        Whether to enable the ValidationAgent in the planning process.""")

    __abstract = True

    def __init__(
        self,
        llm: Llm | None = None,
        interface: ChatFeed | None = None,
        agents: list[Agent | type[Agent]] | None = None,
        tools: list[Tool | type[Tool]] | None = None,
        context: TContext | None = None,
        vector_store: VectorStore | None = None,
        document_vector_store: VectorStore | None = None,
        **params,
    ):
        if context is None:
            context = {}

        if interface is None:
            interface = ChatInterface(
                callback=self._chat_invoke,
                callback_exception="raise",
                load_buffer=5,
                show_button_tooltips=True,
                show_button_name=False,
                sizing_mode="stretch_both"
            )

        llm = llm or self.llm
        instantiated = []
        self._analyses = []
        for agent in agents or self.agents:
            if not isinstance(agent, Agent):
                agent = agent()
            if isinstance(agent, AnalysisAgent):
                analyses = "\n".join(
                    f"- `{analysis.__name__}`: {(analysis.__doc__ or '').strip()}"
                    for analysis in agent.analyses if analysis._callable_by_llm
                )
                agent.purpose = f"Available analyses include:\n\n{analyses}\nSelect this agent to perform one of these analyses."
                self._analyses.extend(agent.analyses)
            # must use the same interface or else nothing shows
            if agent.llm is None:
                agent.llm = llm
            instantiated.append(agent)

        params["tools"] = tools = self._process_tools(tools)
        params["prompts"] = self._process_prompts(params.get("prompts"), tools)

        super().__init__(
            llm=llm,
            agents=instantiated,
            interface=interface,
            vector_store=vector_store,
            document_vector_store=document_vector_store,
            context=context,
            **params
        )

    @wrap_logfire(span_name="Chat Invoke")
    async def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface) -> Plan:
        log_debug(f"New Message: \033[91m{contents!r}\033[0m", show_sep="above")
        return await self.respond(contents, self.context)

    def _process_tools(self, tools: list[type[Tool] | Tool] | None) -> list[type[Tool] | Tool]:
        tools = list(tools) if tools else []

        # If none of the tools provide vector_metaset, add tablelookup
        provides_vector_metaset = any(
            "vector_metaset" in tool.output_schema.__annotations__
            for tool in tools or []
        )
        provides_sql_metaset = any(
            "sql_metaset" in tool.output_schema.__annotations__
            for tool in tools or []
        )
        if not provides_vector_metaset and not provides_sql_metaset:
            # Add both tools - they will share the same vector store through VectorLookupToolUser
            # Both need to be added as classes, not instances, for proper initialization
            tools += [TableLookup, IterativeTableLookup]
        elif not provides_vector_metaset:
            tools += [TableLookup]
        elif not provides_sql_metaset:
            tools += [IterativeTableLookup]
        return tools

    def _process_prompts(
        self, prompts: dict[str, dict[str, Any]], tools: list[type[Tool] | Tool]
    ) -> dict[str, dict[str, Any]]:
        # Add user-provided tools to the list of tools of the coordinator
        if prompts is None:
            prompts = {}

        if "main" not in prompts:
            prompts["main"] = {}

        if "tools" not in prompts["main"]:
            prompts["main"]["tools"] = []
        prompts["main"]["tools"] += [tool for tool in tools]
        return prompts

    async def _fill_model(self, messages, system, agent_model):
        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
        out = await self.llm.invoke(
            messages=messages,
            system=system,
            model_spec=model_spec,
            response_model=agent_model,
        )
        return out

    async def _pre_plan(
        self, messages: list[Message], context: TContext, agents: dict[str, Agent], tools: dict[str, Tool]
    ) -> tuple[dict[str, Agent], dict[str, Tool], dict[str, Any]]:
        """
        Pre-plan step to prepare the agents and tools for the execution graph.
        This is where we can modify the agents and tools based on the messages.
        """
        # Filter agents by exclusions and applies
        agents = {agent_name: agent for agent_name, agent in agents.items() if not any(excluded_key in context for excluded_key in agent.exclusions)}
        applies = await asyncio.gather(*[agent.applies(context) for agent in agents.values()])
        agents = {agent_name: agent for (agent_name, agent), aapply in zip(agents.items(), applies, strict=False) if aapply}

        # Filter tools by exclusions and applies
        tools = {tool_name: tool for tool_name, tool in tools.items() if not any(excluded_key in context for excluded_key in tool.exclusions)}
        applies = await asyncio.gather(*[tool.applies(context) for tool in tools.values()])
        tools = {tool_name: tool for (tool_name, tool), tapply in zip(tools.items(), applies, strict=False) if tapply}

        return agents, tools, {}

    async def _compute_plan(
        self,
        messages: list[Message],
        context: TContext,
        agents: dict[str, Agent],
        tools: dict[str, Tool],
        pre_plan_output: dict
    ) -> Plan:
        """
        Compute the execution graph for the given messages and agents.
        The graph is a list of ExecutionNode objects that represent
        the actions to be taken by the agents.
        """

    def _serialize(self, obj: Any, exclude_passwords: bool = True) -> str:
        if isinstance(obj, (Column, Card, Tabs)):
            string = ""
            for o in obj:
                if isinstance(o, ChatStep) or o.name == "TableSourceCard":
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

    async def respond(
        self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]
    ) -> Plan | None:
        context = {"agent_tool_contexts": [], **context}
        with self.interface.param.update(loading=True):
            if isinstance(self.llm, LlamaCpp):
                with self._add_step(
                    success_title="Using the cached LlamaCpp model",
                    title="Loading LlamaCpp model...",
                    user="Assistant"
                ) as step:
                    default_kwargs = self.llm.model_kwargs["default"]
                    if 'repo' in default_kwargs and 'model_file' in default_kwargs:
                        step.stream(f"Model: `{default_kwargs['repo']}/{default_kwargs['model_file']}`")
                    elif 'model_path' in default_kwargs:
                        step.stream(f"Model: `{default_kwargs['model_path']}`")
                    await self.llm.get_client("default")  # caches the model for future use

            # TODO INVESTIGATE
            messages = fuse_messages(
                self.interface.serialize(custom_serializer=self._serialize, limit=10) or messages,
                max_user_messages=self.history
            )

            # the master dict of agents / tools to be used downstream
            # change this for filling models' literals
            agents = {normalized_name(agent): agent for agent in self.agents}
            tools = {normalized_name(tool): tool for tool in self._tools["main"]}

            agents, tools, pre_plan_output = await self._pre_plan(
                messages, context, agents, tools
            )
            plan = await self._compute_plan(
                messages, context, agents, tools, pre_plan_output
            )
        if plan is not None:
            self.steps_layout.header[0].object = "🧾 Checklist ready..."
            _, todos = plan.render_task_history(-1)
            self.steps_layout.header[1].object = todos
        return plan

    async def _check_tool_relevance(
        self, tool: Tool, tool_output: str, actor: Actor, actor_task: str,
        messages: list[Message], context: TContext
    ) -> bool:
        result = await self._invoke_prompt(
            "tool_relevance",
            messages,
            context,
            tool_name=tool.name,
            tool_purpose=getattr(tool, "purpose", ""),
            tool_output=tool_output,
            actor_name=actor.name,
            actor_purpose=getattr(actor, "purpose", actor.__doc__),
            actor_task=actor_task,
        )

        return result.yes

    def __panel__(self):
        return self.interface

    async def sync(self, context: TContext | None):
        context = context or self.context
        for tools in self._tools.values():
            for tool in tools:
                await tool.sync(context)


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
        context: TContext,
        agents: list[Agent] | None = None,
        primary: bool = False,
        unmet_dependencies: tuple[str] | None = None
    ):
        if agents is None:
            agents = self.agents
        applies = await asyncio.gather(*[agent.applies(context) for agent in agents])
        agents = [agent for agent, aapply in zip(agents, applies, strict=False) if aapply]
        applies = await asyncio.gather(*[tool.applies(context) for tool in self._tools['main']])
        tools = [tool for tool, tapply in zip(self._tools['main'], applies, strict=False) if tapply]

        agent_names = tuple(sagent.name[:-5] for sagent in agents) + tuple(tool.name for tool in tools)
        agent_model = self._get_model("main", agent_names=agent_names, primary=primary)
        if len(agent_names) == 0:
            raise ValueError("No agents available to choose from.")
        if len(agent_names) == 1:
            return agent_model(agent=agent_names[0], chain_of_thought='')
        system = await self._render_prompt(
            "main",
            messages,
            context,
            agents=agents,
            tools=tools,
            primary=primary,
            unmet_dependencies=unmet_dependencies
        )
        return await self._fill_model(messages, system, agent_model)

    async def _compute_plan(
        self,
        messages: list[Message],
        context: TContext,
        agents: dict[str, Agent],
        tools: dict[str, Tool],
        pre_plan_output: dict[str, Any]
    ) -> Plan | None:
        if len(agents) == 1:
            agent = next(iter(agents.values()))
        else:
            agent = None
            with self._add_step(title="Selecting primary agent...", user="Assistant") as step:
                try:
                    output = await self._choose_agent(messages, context, self.agents, primary=True)
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
            return None

        cot = output.chain_of_thought
        subagent = agent
        tasks = []
        while (unmet_dependencies := tuple(
            r for r in await subagent.requirements(messages) if r not in self._memory
        )):
            with self._add_step(title="Resolving dependencies...", user="Assistant") as step:
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
                tasks.append(
                    ActorTask(
                        subagent,
                        instruction=output.chain_of_thought,
                        title=output.agent_or_tool
                    )
                )
                step.success_title = f"Solved a dependency with {output.agent_or_tool}"
        tasks = tasks[::-1] + [ActorTask(agent, instruction=cot)]
        return Plan(*tasks, history=messages, context=context, coordinator=self)


class Planner(Coordinator):
    """
    The Planner develops a plan to solve the user query step-by-step
    and then executes it.
    """

    planner_tools = param.List(default=[], doc="""
        List of tools to use to provide context for the planner prior
        to making a plan.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "Planner" / "main.jinja2",
                "response_model": make_plan_model,
            },
            "follow_up": {
                "template": PROMPTS_DIR / "Planner" / "follow_up.jinja2",
                "response_model": ThinkingYesNo,
            },
        }
    )

    def __init__(self, **params):
        if 'planner_tools' in params:
            params["planner_tools"] = self._initialize_tools_for_prompt(params["planner_tools"], **params)
        super().__init__(**params)

    async def _check_follow_up_question(self, messages: list[Message], context: TContext) -> bool:
        """Check if the user's query is a follow-up question about the previous dataset."""
        # Only check if data is in memory
        if "data" not in context:
            return False

        # Use the follow_up prompt to check
        result = await self._invoke_prompt(
            "follow_up",
            messages,
            context
        )

        is_follow_up = result.yes
        if not is_follow_up:
            context.pop("pipeline", None)

        return is_follow_up

    async def _execute_planner_tools(self, messages: list[Message], context: TContext):
        """Execute planner tools to gather context before planning."""
        if not self.planner_tools:
            return

        user_query = next((
            msg["content"] for msg in reversed(messages)
            if msg.get("role") == "user"), ""
        )

        with self._add_step(
            title="Gathering context for planning...", user="Assistant", steps_layout=self.steps_layout
        ) as step:
            for tool in self.planner_tools:
                is_relevant = await self._check_tool_relevance(
                    tool, "", self, f"Gather context for planning to answer {user_query}",
                    messages, context
                )

                if not is_relevant:
                    continue

                tool_name = getattr(tool, "name", type(tool).__name__)
                step.stream(f"Using {tool_name} to gather planning context...")
                task = ActorTask(
                    tool,
                    interface=self.interface,
                    instruction=user_query,
                    title=f"Gathering context with {tool_name}",
                    steps_layout=self.steps_layout,
                )
                await task.execute(context)
                if task.status != "error":
                    step.stream(f"\n\n✗ Failed to gather context from {tool_name}")
                    continue

    async def _pre_plan(
        self,
        messages: list[Message],
        context: TContext,
        agents: dict[str, Agent],
        tools: dict[str, Tool],
    ) -> tuple[dict[str, Agent], dict[str, Tool], dict[str, Any]]:
        is_follow_up = await self._check_follow_up_question(messages, context)
        if not is_follow_up:
            await self._execute_planner_tools(messages, context)
        else:
            log_debug("\033[92mDetected follow-up question, using existing context\033[0m")
            with self._add_step(
                title="Using existing data context...", user="Assistant", steps_layout=self.steps_layout
            ) as step:
                step.stream("Detected that this is a follow-up question related to the previous dataset.")
                step.stream("\n\nUsing the existing data in memory to answer without re-executing data retrieval.")
                step.success_title = "Using existing data for follow-up question"
        agents, tools, pre_plan_output = await super()._pre_plan(messages, context, agents, tools)
        pre_plan_output["is_follow_up"] = is_follow_up
        return agents, tools, pre_plan_output

    @wrap_logfire(span_name="Make Plan", extract_args=["messages", "previous_plans", "is_follow_up"])
    async def _make_plan(
        self,
        messages: list[Message],
        context: TContext,
        agents: dict[str, Agent],
        tools: dict[str, Tool],
        unmet_dependencies: set[str],
        previous_actors: list[str],
        previous_plans: list[str],
        plan_model: type[RawPlan],
        step: ChatStep,
        is_follow_up: bool = False,
    ) -> BaseModel:
        agents = list(agents.values())
        tools = list(tools.values())
        all_provides = set()
        for provider in agents + tools:
            all_provides |= set(provider.output_schema.__annotations__)
        all_provides |= set(context)

        # filter agents using applies
        agents = [agent for agent in agents if await agent.applies(context)]

        # ensure these candidates are satisfiable
        # e.g. DbtslAgent is unsatisfiable if DbtslLookup was used in planning
        # but did not provide dbtsl_metaset
        # also filter out agents where excluded keys exist in memory
        agents = [
            agent for agent in agents if len(set(agent.input_schema.__required_keys__) - all_provides) == 0
            and type(agent).__name__ != "ValidationAgent"
        ]
        tools = [
            tool for tool in tools if len(set(tool.input_schema.__required_keys__) - all_provides) == 0
        ]
        reasoning = None
        while reasoning is None:
            # candidates = agents and tools that can provide
            # the unmet dependencies
            agent_candidates = [
                agent for agent in agents
                if not unmet_dependencies or set(agent.output_schema.__annotations__) & unmet_dependencies
            ]
            tool_candidates = [
                tool for tool in tools
                if not unmet_dependencies or set(tool.output_schema.__annotations__) & unmet_dependencies
            ]
            system = await self._render_prompt(
                "main",
                messages,
                context,
                agents=agents,
                tools=tools,
                unmet_dependencies=unmet_dependencies,
                candidates=agent_candidates + tool_candidates,
                previous_actors=previous_actors,
                previous_plans=previous_plans,
                is_follow_up=is_follow_up,
            )
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            async for reasoning in self.llm.stream(
                messages=messages,
                system=system,
                model_spec=model_spec,
                response_model=Reasoning,
                max_retries=3,
            ):
                if reasoning.chain_of_thought:  # do not replace with empty string
                    context["reasoning"] = reasoning.chain_of_thought
                    step.stream(reasoning.chain_of_thought, replace=True)
                    previous_plans.append(reasoning.chain_of_thought)

        return await self._fill_model(messages, reasoning.chain_of_thought, plan_model)

    async def _resolve_plan(
        self,
        raw_plan: RawPlan,
        agents: dict[str, Agent],
        tools: dict[str, Tool],
        messages: list[Message],
        context: TContext,
        previous_actors: list[str],
    ) -> tuple[Plan, set[str], list[str]]:
        table_provided = False
        tasks = []
        provided = set(context)
        unmet_dependencies = set()
        steps = []
        actors = []
        actors_in_graph = set()

        # Check for consecutive agents of the same type and merge them
        merged_steps = []
        i = 0
        while i < len(raw_plan.steps):
            current_step = raw_plan.steps[i]

            # Look ahead to see if there are consecutive steps with the same actor
            consecutive_steps = [current_step]
            j = i + 1
            while j < len(raw_plan.steps) and raw_plan.steps[j].actor == current_step.actor:
                consecutive_steps.append(raw_plan.steps[j])
                j += 1

            if len(consecutive_steps) > 1:
                # Join the consecutive steps into one
                combined_instruction = f"{current_step.instruction} Additionally: " + ". ".join(
                    step.instruction for step in consecutive_steps[1:]
                )
                combined_title = f"{current_step.title} (combined {len(consecutive_steps)} steps)"

                # Create a new step with combined instruction
                merged_step = type(current_step)(
                    actor=current_step.actor,
                    instruction=combined_instruction,
                    title=combined_title,
                )
                merged_steps.append(merged_step)
            else:
                merged_steps.append(current_step)

            i = j

        # Update the raw_plan with merged steps
        raw_plan.steps = merged_steps

        for step in raw_plan.steps:
            key = step.actor
            actors.append(key)
            if key in agents:
                subagent = agents[key]
            elif key in tools:
                subagent = tools[key]
            else:
                # Skip if the agent or tool doesn't exist
                log_debug(f"Warning: Agent or tool '{key}' not found in available agents/tools")
                continue

            requires = set(await subagent.requirements(messages))
            provided |= set(subagent.output_schema.__annotations__)
            unmet_dependencies = (unmet_dependencies | requires) - provided
            has_table_lookup = any(
                any(isinstance(st, TableLookup) for st in task)
                for task in tasks
            )
            if "table" in unmet_dependencies and not table_provided and "SQLAgent" in agents and has_table_lookup:
                provided |= set(agents['SQLAgent'].output_schema.__annotations__)
                sql_step = type(step)(
                    actor='SQLAgent',
                    instruction='Load the table',
                    title='Loading table',
                )
                tasks.append(
                    ActorTask(
                        agents['SQLAgent'],
                        instruction=sql_step.instruction,
                        title=sql_step.title
                    )
                )
                steps.append(sql_step)
                table_provided = True
                unmet_dependencies -= provided
                actors_in_graph.add('SQLAgent')
            tasks.append(
                ActorTask(
                    subagent,
                    instruction=step.instruction,
                    title=step.title
                )
            )
            steps.append(step)
            actors_in_graph.add(key)

        last_task = tasks[-1] if tasks else None
        if last_task and isinstance(last_task.actor, Tool):
            if "AnalystAgent" in agents and all(r in provided for r in agents["AnalystAgent"].input_schema.__annotations__):
                actor = "AnalystAgent"
            else:
                actor = "ChatAgent"

            # Check if the actor conflicts with any actor in the graph
            not_with = getattr(agents[actor], 'not_with', [])
            conflicts = [actor for actor in actors_in_graph if actor in not_with]
            if conflicts:
                # Skip the summarization step if there's a conflict
                log_debug(f"Skipping summarization with {actor} due to conflicts: {conflicts}")
                raw_plan.steps = steps
                previous_actors = actors
                return Plan(*tasks, title=raw_plan.title, history=messages, context=context, coordinator=self, steps_layout=self.steps_layout), previous_actors

            summarize_step = type(step)(
                actor=actor,
                instruction='Summarize the results.',
                title='Summarizing results',
            )
            steps.append(summarize_step)
            tasks.append(
                ActorTask(
                    agents[actor],
                    instruction=summarize_step.instruction,
                    title=summarize_step.title,
                )
            )
            actors_in_graph.add(actor)

        if "ValidationAgent" in agents and len(actors_in_graph) > 1 and self.validation_enabled:
            validation_step = type(step)(
                actor="ValidationAgent",
                instruction='Validate whether the executed plan fully answered the user\'s original query.',
                title='Validating results',
            )
            steps.append(validation_step)
            tasks.append(
                ActorTask(
                    agents["ValidationAgent"],
                    instruction=validation_step.instruction,
                    title=validation_step.title
                )
            )
            actors_in_graph.add("ValidationAgent")

        raw_plan.steps = steps
        return Plan(*tasks, title=raw_plan.title, history=messages, context=context, coordinator=self, steps_layout=self.steps_layout), actors

    async def _compute_plan(
        self,
        messages: list[Message],
        context: TContext,
        agents: dict[str, Agent],
        tools: dict[str, Tool],
        pre_plan_output: dict[str, Any]
    ) -> Plan:
        tool_names = list(tools)
        agent_names = list(agents)
        plan_model = self._get_model("main", agents=agent_names, tools=tool_names)

        planned = False
        unmet_dependencies = set()
        previous_plans, previous_actors = [], []
        attempts = 0
        plan = None

        todos_title = Typography(
            "📋 Building checklist...", css_classes=["todos-title"], margin=0,
            styles={"font-weight": "normal", "font-size": "1.1em"}
        )
        todos = Typography(
            css_classes=["todos"], margin=0, styles={"font-weight": "normal"}
        )
        todos_layout = Column(todos_title, todos)

        with self._add_step(
            title="Planning how to solve user query...", user="Planner",
            layout_params={"header": todos_layout, "collapsed": not self.verbose, "elevation": 2}
        ) as istep:
            await asyncio.sleep(0.1)
            self.steps_layout = self.interface.objects[-1].object
            while not planned:
                if attempts > 0:
                    log_debug(f"\033[91m!! Attempt {attempts}\033[0m")
                plan = None
                try:
                    raw_plan = await self._make_plan(
                        messages, context, agents, tools, unmet_dependencies, previous_actors, previous_plans,
                        plan_model, istep, is_follow_up=pre_plan_output["is_follow_up"]
                    )
                except asyncio.CancelledError as e:
                    todos_title.object = istep.failed_title = 'Planning was cancelled, please try again.'
                    traceback.print_exception(e)
                    raise e
                except Exception as e:
                    istep.failed_title = "Internal execution error during planning stage."
                    todos_title.object = (
                        "Planner could not settle on a plan of action to perform the requested query. "
                        "Please restate your request."
                    )
                    traceback.print_exception(e)
                    raise e
                plan, previous_actors = await self._resolve_plan(
                    raw_plan, agents, tools, messages, context, previous_actors
                )
                try:
                    plan.validate()
                except ContextError as e:
                    istep.stream(str(e), replace=True)
                    attempts += 1
                else:
                    planned = True
                if attempts > 5:
                    istep.failed_title = "Planning failed to come up with viable plan, please restate the problem and try again."
                    todos_title.object = "❌ Planning failed"
                    e = RuntimeError("Planner failed to come up with viable plan after 5 attempts.")
                    traceback.print_exception(e)
                    raise e

            # Store the todo message reference for later updates
            if attempts > 0:
                istep.success_title = f"Plan with {len(raw_plan.steps)} steps created after {attempts + 1} attempts"
            else:
                istep.success_title = f"Plan with {len(raw_plan.steps)} steps created"
        return plan
