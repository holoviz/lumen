from __future__ import annotations

import asyncio
import re
import traceback

from functools import partial
from textwrap import dedent, indent
from types import FunctionType
from typing import TYPE_CHECKING, Any, Self

import param

from panel.chat import ChatFeed
from panel.io.state import state
from panel.layout.base import ListLike, NamedListLike
from panel.pane import HTML, Image
from panel.viewable import Viewer
from panel_material_ui import (
    Card, ChatInterface, ChatStep, Column, Typography,
)

from ..actor import Actor
from ..agents import Agent, AnalysisAgent, ChatAgent
from ..config import PROMPTS_DIR, MissingContextError
from ..context import TContext
from ..llm import LlamaCpp, Llm, Message
from ..models import ThinkingYesNo
from ..report import ActorTask, Section, TaskGroup
from ..tools import MetadataLookup, Tool, VectorLookupToolUser
from ..utils import (
    fuse_messages, get_root_exception, log_debug, mutate_user_message,
    normalized_name, wrap_logfire,
)
from ..vector_store import NumpyVectorStore

if TYPE_CHECKING:
    from panel.chat.step import ChatStep

    from .vector_store import VectorStore


class Plan(Section):
    """
    A Plan is a Task that is a collection of other Tasks.
    """

    abort_on_error = param.Boolean(
        default=True,
        doc="""
        If True, the report will abort if an error occurs.""",
    )

    agents = param.List(item_type=Actor, default=[])

    coordinator = param.ClassSelector(class_=param.Parameterized)

    interface = param.ClassSelector(class_=ChatFeed)

    is_followup = param.Boolean(default=False)

    _tasks = param.List(item_type=ActorTask)

    def render_task_history(self, i: int | None = None, failed: bool = False) -> tuple[list[Message], str]:
        i = self._current if i is None else i
        user_query = None
        for msg in reversed(self.history):
            if msg.get("role") == "user":
                user_query = msg
                break

        todos_list = []
        for idx, task in enumerate(self):
            instruction = task.instruction
            if failed and idx == i:
                status = "🔴"
            elif idx < i:
                status = "🟢"
            elif i == idx:
                status = "🟡"
            else:
                status = "⚪"
            todos_list.append(f"- {status} {instruction}")
        todos = "\n".join(todos_list)

        if isinstance(user_query["content"], list):
            user_content = '\n'.join([content for content in user_query["content"] if isinstance(content, str)])
        else:
            user_content = user_query["content"]

        formatted_content = (
            f"User: {user_content}\n"
            f"Roadmap:\n{indent(todos, '    ')}\n"
            f"Tasks marked ⚪ are scheduled for others later. "
            f"Your EXCLUSIVE goal is to focus on the 🟡 task"
        )
        rendered_history = []
        for msg in self.history:
            if msg is user_query:
                rendered_history.append({"content": formatted_content, "role": "user"})
            else:
                rendered_history.append(msg)
        return rendered_history, todos

    async def _run_task(self, i: int, task: Self | Actor, context: TContext, **kwargs):
        if task.status == "success":
            return task.views, task.out_context

        outputs = []
        with self._add_step(title=f"{task.title}...", user="Runner", layout_params={"title": "🏗️ Running "}, steps_layout=self.steps_layout) as step:
            history, todos = self.render_task_history(i)
            subcontext = self._get_context(i, context, task)
            if self.steps_layout is not None:
                self.steps_layout.header[0].object = f"⚙️ Working on task {task.title!r}..."
                self.steps_layout.header[1].object = todos
            step.stream(task.instruction)
            try:
                kwargs = {"agents": self.agents} if "agents" in task.param else {}
                with task.param.update(
                    interface=self.interface, llm=task.llm or self.llm, steps_layout=self.steps_layout[-1] if self.steps_layout else None, history=history, **kwargs
                ):
                    outputs, task_context = await task.execute(subcontext, **kwargs)
            except Exception as e:
                outputs, task_context = await self._handle_task_execution_error(e, task, context, step, i)

            # Check if task errored before validating context to have richer error info
            if task.status == "error":
                # Get the error message from context if available
                error_msg = task_context.get("__error__", "unknown error")
                step.failed_title = f"{task.title} errored during execution."
                raise RuntimeError(error_msg)

            unprovided = [p for p in task.output_schema.__required_keys__ if p not in task_context]
            if unprovided:
                step.failed_title = f"{task.title!r} failed to provide declared context values {', '.join(unprovided)}. Aborting the plan."
                raise RuntimeError(f"{task.title!r} task failed to provide declared context.")

            context_keys = ", ".join(f"`{k}`" for k in task_context)
            step.stream(f"Generated {len(outputs)} and provided {context_keys}.")
            step.success_title = f"Task {task.title!r} successfully completed"
            log_debug(f"\033[96mCompleted: {task.title}\033[0m", show_length=False)
        return outputs, task_context

    async def _handle_task_execution_error(self, e: Exception, task: Self | Actor, context: TContext, step: ChatStep, i: int) -> tuple[list[Any], TContext]:
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
                outputs, out_context = await self._retry_from_provider(provider_index, i, str(root_exception), context)
                step.success_title = f"{task.title!r} task successfully completed after retry"
                return outputs, out_context  # Return after successful retry
            else:
                # If we can't find a provider, this error is unrecoverable
                # Show it clearly to the user instead of re-raising
                error_msg = str(root_exception)
                step.failed_title = f"{task.title!r} failed: {error_msg}"
                step.stream(f"\n\n❌ **{error_msg}**")
                task.status = "error"
                task.out_context["__error__"] = error_msg
                return [], {"__error__": error_msg}  # Return with error message in context
        elif isinstance(e, asyncio.CancelledError):
            traceback.print_exception(e)
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
        pipeline_exists = "pipeline" in context
        if not pipeline_exists:
            return

        for idx in reversed(range(failed_index)):
            task = self._tasks[idx]
            # Check if this task provides pipeline or other relevant context
            if isinstance(task, ActorTask):
                if "pipeline" in task.output_schema.__annotations__:
                    return idx
            elif isinstance(task, TaskGroup):
                for subtask in task:
                    if isinstance(subtask, ActorTask) and "pipeline" in subtask.output_schema.__annotations__:
                        return idx
        return

    async def _retry_from_provider(self, provider_index: int, failed_index: int, error_message: str, context: TContext) -> tuple[list[Any], TContext]:
        """
        Re-run the plan from the provider task with additional feedback about the error.
        """

        outputs = []
        # Create feedback suffix to add to user message
        feedback_suffix = f"Be mindful of: {self._tasks[failed_index].title!r}Previously, it was reported that: {error_message!r}. Try a different approach."
        # Re-run from the provider until the failed task
        for idx in range(provider_index, failed_index + 1):
            task = self[idx]
            task.reset()
            if idx < failed_index:
                out, out_context = await self._run_task(idx, task, context)
                continue

            # For the provider task, mutate the user message to include feedback
            retry_history = mutate_user_message(feedback_suffix, self.history.copy())
            # Show retry in UI
            with self._add_step(title=f"🔄 Retrying {task.title} with feedback...", user="Runner", steps_layout=self.steps_layout) as retry_step:
                retry_step.stream(f"Retrying with feedback about: {error_message}")

                # Update todos to show retry
                if self.steps_layout is not None:
                    todos = "\n".join(
                        f"- [{'x' if tidx < idx else '🔄' if tidx == idx else ' '}] {'<u>' + t.instruction + '</u>' if tidx == idx else t.instruction}"
                        for tidx, t in enumerate(self)
                    )
                    self.steps_layout.header[1].object = todos

                # Run with mutated history
                kwargs = {"agents": self.agents} if "agents" in task.param else {}
                subcontext = self._get_context(idx, context, task)
                with task.param.update(interface=self.interface, steps_layout=self.steps_layout, history=retry_history, **kwargs):
                    out, out_context = await task.execute(subcontext, **kwargs)
                retry_step.success_title = f"✅ {task.title} successfully completed on retry"
        return outputs, out_context

    async def execute(self, context: TContext = None, **kwargs) -> list[Any]:
        context = context or self.context
        if "__error__" in context:
            del context["__error__"]
        outputs, out_context = await super().execute(context, **kwargs)
        _, todos = self.render_task_history(failed=self.status == "error")
        if self.steps_layout is not None:
            steps_title, todo_list = self.steps_layout.header
            todo_list.object = todos
            if self.status == "success":
                steps_title.object = f"✅ Sucessfully completed {self.title!r}"
            else:
                steps_title.object = f"❌ Failed to execute {self.title!r}"
            self.steps_layout.collapsed = True
        log_debug("\033[92mCompleted: Plan\033[0m", show_sep="below")
        return outputs, out_context


class Coordinator(Viewer, VectorLookupToolUser):
    """
    A Coordinator is responsible for coordinating the actions
    of a number of agents towards the user defined query by
    computing an execution graph and then executing each
    step along the graph.
    """

    agents = param.List(
        default=[ChatAgent],
        doc="""
        List of agents to coordinate.""",
    )

    context = param.Dict(default={})

    history = param.Integer(
        default=3,
        doc="""
        Number of previous user-assistant interactions to include in the chat history.""",
    )

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

    verbose = param.Boolean(
        default=False,
        allow_refs=True,
        doc="""
        Whether to show verbose output.""",
    )

    within_ui = param.Boolean(
        default=False,
        constant=True,
        doc="""
        Whether this coordinator is being used within the UI.""",
    )

    validation_enabled = param.Boolean(
        default=False,
        allow_refs=True,
        doc="""
        Whether to enable the ValidationAgent in the planning process.""",
    )

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
                callback=self._chat_invoke, callback_exception="raise", load_buffer=5, show_button_tooltips=True, show_button_name=False, sizing_mode="stretch_both"
            )

        # Create default vector stores if not provided
        if vector_store is None:
            vector_store = NumpyVectorStore()
        # Use the same vector_store for documents if not explicitly provided
        if document_vector_store is None:
            document_vector_store = vector_store

        llm = llm or self.llm
        instantiated = []
        self._analyses = []
        for agent in agents or self.agents:
            if not isinstance(agent, Agent):
                agent = agent()
            if isinstance(agent, AnalysisAgent):
                analyses = indent("\n".join(
                    f"- `{analysis.__name__}`:\n{indent(dedent(analysis.__doc__ or '').strip(), ' ' * 4)} (required cols: {', '.join(analysis.columns)})"
                    for analysis in agent.analyses if analysis._callable_by_llm
                ), " " * 4)
                agent.conditions.append(
                    f"The following analyses can be performed by AnalysisAgent:\n {analyses}\n"
                )
                self._analyses.extend(agent.analyses)
            # must use the same interface or else nothing shows
            if agent.llm is None:
                agent.llm = llm
            instantiated.append(agent)

        params["tools"] = tools = self._process_tools(tools)
        params["prompts"] = self._process_prompts(params.get("prompts"), tools)

        super().__init__(
            llm=llm, agents=instantiated, interface=interface, vector_store=vector_store, document_vector_store=document_vector_store, context=context, **params
        )
        for tools in self._tools.values():
            for tool in tools:
                state.execute(partial(tool.prepare, context))

    @wrap_logfire(span_name="Chat Invoke")
    async def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface) -> Plan:
        log_debug(f"New Message: \033[91m{contents!r}\033[0m", show_sep="above")
        return await self.respond(contents, self.context)

    def _process_tools(self, tools: list[type[Tool] | Tool] | None) -> list[type[Tool] | Tool | FunctionType]:
        tools = list(tools) if tools else []

        # If none of the tools provide metaset, add MetadataLookup
        provides_metaset = any(
            "metaset" in tool.output_schema.__annotations__ for tool in tools
            if isinstance(tool, Tool) or (isinstance(tool, type) and issubclass(tool, Tool))
        )
        if not provides_metaset:
            # Add both tools - they will share the same vector store through VectorLookupToolUser
            # Both need to be added as classes, not instances, for proper initialization
            tools += [MetadataLookup]
        return tools

    def _process_prompts(self, prompts: dict[str, dict[str, Any]], tools: list[type[Tool] | Tool]) -> dict[str, dict[str, Any]]:
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

    async def _compute_plan(self, messages: list[Message], context: TContext, agents: dict[str, Agent], tools: dict[str, Tool], pre_plan_output: dict) -> Plan:
        """
        Compute the execution graph for the given messages and agents.
        The graph is a list of ExecutionNode objects that represent
        the actions to be taken by the agents.
        """

    def _serialize(self, obj: Any, exclude_passwords: bool = True) -> str | list:
        if isinstance(obj, str):
            return obj

        if isinstance(obj, (ListLike, NamedListLike)):
            return self._serialize(list(obj), exclude_passwords=exclude_passwords)

        if isinstance(obj, list):
            contents = []
            for o in obj:
                if isinstance(o, ChatStep):
                    # Drop context from steps; should be irrelevant now
                    continue
                if isinstance(o, Image):
                    contents.append(o)
                    continue
                result = self._serialize(o, exclude_passwords=exclude_passwords)
                if isinstance(result, list):
                    contents.extend(result)
                else:
                    contents.append(result)
            return contents

        if isinstance(obj, HTML) and "catalog" in obj.tags:
            return f"Summarized table listing: {obj.object[:30]}"

        if hasattr(obj, "object"):
            obj = obj.object
        elif hasattr(obj, "value"):
            obj = obj.value

        if exclude_passwords:
            if isinstance(obj, str):
                obj = re.sub(r"password:.*\n", "", obj)
            elif isinstance(obj, list):
                obj = [re.sub(r"password:.*\n", "", item) if isinstance(item, str) else item for item in obj]
        return obj

    async def respond(self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]) -> Plan | None:
        context = {"agent_tool_contexts": [], **context}
        with self.interface.param.update(loading=True):
            if isinstance(self.llm, LlamaCpp):
                with self._add_step(success_title="Using the cached LlamaCpp model", title="Loading LlamaCpp model...", user="Assistant") as step:
                    default_kwargs = self.llm.model_kwargs["default"]
                    if "repo" in default_kwargs and "model_file" in default_kwargs:
                        step.stream(f"Model: `{default_kwargs['repo']}/{default_kwargs['model_file']}`")
                    elif "model_path" in default_kwargs:
                        step.stream(f"Model: `{default_kwargs['model_path']}`")
                    await self.llm.get_client("default")  # caches the model for future use

            messages = fuse_messages(self.interface.serialize(custom_serializer=self._serialize, limit=10) or messages, max_user_messages=self.history)

            # the master dict of agents / tools to be used downstream
            # change this for filling models' literals
            agents = {normalized_name(agent): agent for agent in self.agents}
            tools = {normalized_name(tool): tool for tool in self._tools["main"]}

            self._todos_title = Typography("📋 Building checklist...", css_classes=["todos-title"], margin=0, styles={"font-weight": "normal", "font-size": "1.1em"})
            todos = Typography(css_classes=["todos"], margin=0, styles={"font-weight": "normal"})
            todos_layout = Column(self._todos_title, todos, stylesheets=[".markdown { padding-inline: unset; }"])
            self.steps_layout = Card(header=todos_layout, collapsed=not self.verbose, elevation=2)
            self.interface.stream(self.steps_layout, user="Planner")
            agents, tools, pre_plan_output = await self._pre_plan(messages, context, agents, tools)
            plan = await self._compute_plan(messages, context, agents, tools, pre_plan_output)
        if plan is not None:
            self.steps_layout.header[0].object = "🧾 Checklist ready..."
            _, todos = plan.render_task_history(-1)
            self.steps_layout.header[1].object = todos
        return plan

    async def _check_tool_relevance(self, tool: Tool, tool_output: str, actor: Actor, actor_task: str, messages: list[Message], context: TContext) -> bool:
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
        synced_tools = set()
        for tools in self._tools.values():
            for tool in tools:
                if id(tool) not in synced_tools:
                    await tool.sync(context)
                    synced_tools.add(id(tool))
