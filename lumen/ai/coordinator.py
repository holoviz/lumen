from __future__ import annotations

import asyncio
import re
import traceback

from typing import TYPE_CHECKING, Any

import param

from panel import bind
from panel.chat import ChatFeed
from panel.io.document import hold
from panel.pane import HTML, Markdown
from panel.viewable import Viewer
from panel_material_ui import (
    Accordion, Button, Card, ChatInterface, ChatStep, Column, FlexBox, Paper,
    Tabs, Typography,
)
from pydantic import BaseModel
from typing_extensions import Self

from .actor import Actor
from .agents import Agent, AnalysisAgent, ChatAgent
from .config import (
    DEMO_MESSAGES, GETTING_STARTED_SUGGESTIONS, PROMPTS_DIR,
    MissingContextError,
)
from .context import ContextError, TContext
from .controls import SourceControls, TableSourceCard
from .llm import LlamaCpp, Llm, Message
from .logs import ChatLogs
from .models import (
    RawPlan, Reasoning, ThinkingYesNo, make_agent_model, make_plan_model,
)
from .report import Section, TaskGroup
from .tools import (
    IterativeTableLookup, TableLookup, Tool, VectorLookupToolUser,
)
from .utils import (
    fuse_messages, get_root_exception, log_debug, mutate_user_message,
    normalized_name, wrap_logfire,
)
from .views import AnalysisOutput

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

    def _render_task_history(self, i: int) -> tuple[list[Message], str]:
        user_query = None
        for msg in reversed(self.history):
            if msg.get('role') == 'user':
                user_query = msg
                break
        todos = '\n'.join(
            f"- [{'x' if idx < i else ' '}] {'<u>' + task.instruction + '</u>' if idx == i else task.instruction}" for idx, task in enumerate(self)
        )
        formatted_content = f"User Request: {user_query['content']!r}\n\nComplete the underlined todo:\n{todos}"
        return [
            {'content': formatted_content, 'role': 'user'} if msg is user_query else msg
            for msg in self.history
        ], todos

    async def _run_task(self, i: int, task: Self | Actor, context: TContext, **kwargs):
        outputs = []
        with self.interface.add_step(
            title=f"{task.title}...", user="Runner", layout_params={"title": "🏗️ Plan Execution Steps"},
            steps_layout=self.coordinator.steps_layout
        ) as step:
            self.coordinator._todos_title.object = f"⚙️ Working on task {task.title!r}..."
            step.stream(f"`Working on task {task.title}`:\n\n{task.instruction}")
            history, todos = self._render_task_history(i)
            self.coordinator._todos.object = todos
            try:
                kwargs = {"agents": self.agents} if 'agents' in task.param else {}
                with task.param.update(
                    interface=self.interface, steps_layout=self.coordinator.steps_layout,
                    history=history, **kwargs
                ):
                    new, task_context = await task.execute(context, **kwargs)
                    outputs += new
            except Exception as e:
                # Handle the exception using the dedicated error handler
                task_context = {}
                error_outputs = await self._handle_task_execution_error(e, task, task_context, step, i)
                if error_outputs is not None:
                    return error_outputs, task_context
            if isinstance(task, TaskGroup):
                unprovided = [
                    p for actor in task for p in actor.output_schema.__annotations__
                    if p not in task_context
                ]
            else:
                unprovided = []
            if unprovided:
                step.failed_title = f"{task.title} did not provide {', '.join(unprovided)}. Aborting the plan."
                raise RuntimeError(f"{task.title!r} task failed to provide declared context.")
            log_debug(f"\033[96mCompleted: {task.title}\033[0m", show_length=False)
            step.stream(f"\n\nSuccessfully completed task {task.title}:\n\n> {task.instruction}", replace=True)
            step.success_title = f"{task.title} successfully completed"
        return outputs, task_context

    async def _handle_task_execution_error(self, e: Exception, task: Self | Actor, context: TContext, step: ChatStep, i: int) -> list | None:
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
                outputs = await self._retry_from_provider(provider_index, i, str(root_exception))
                step.success_title = f"{task.title} successfully completed after retry"
                return outputs  # Return after successful retry
            else:
                # If we can't find a provider, raise the original error
                step.failed_title = f"{task.title} - Cannot resolve missing context"
                traceback.print_exception(e)
                self.memory['__error__'] = str(e)
                raise e
        elif isinstance(e, asyncio.CancelledError):
            step.failed_title = f"{task.title} agent was cancelled"
            raise e
        else:
            traceback.print_exception(e)
            context['__error__'] = str(e)
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
            if isinstance(task, TaskGroup):
                for actor in task:
                    if 'pipeline' in actor.output_schema.__annotations__:
                        return idx

    async def _retry_from_provider(self, provider_index: int, failed_index: int, error_message: str) -> list:
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
        # Re-run from the provider onwards
        for idx in range(provider_index, len(self)):
            task = self[idx]
            if idx < provider_index:
                outputs += await self._run_task(idx, task)
                continue
            elif idx > failed_index:
                break # Stop after re-running the failed task

            # For the provider task, mutate the user message to include feedback
            retry_history = mutate_user_message(feedback_suffix, self.history.copy())
            # Show retry in UI
            with self.interface.add_step(
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
                self.coordinator._todos.object = todos

                # Run with mutated history
                kwargs = {"agents": self.agents} if 'agents' in task.param else {}
                with task.param.update(
                    interface=self.interface, steps_layout=self.steps_layout,
                    history=retry_history, **kwargs
                ):
                    outputs += await task.execute(**kwargs)
                retry_step.success_title = f"✅ {task.title} successfully completed on retry"
        return outputs

    async def execute(self, context: TContext = None, **kwargs) -> list[Any]:
        context = context or self.context
        if '__error__' in context:
            del context['__error__']
        outputs, out_context = await super().execute(context, **kwargs)
        _, todos = self._render_task_history(len(self))
        self.coordinator._todos.object = todos
        if self.status == 'success':
            self.coordinator._todos_title.object = f"✅ Sucessfully completed {self.title!r}"
        else:
            self.coordinator._todos_title.object = f"❌ Failed to execute {self.title!r}"
        if "pipeline" in out_context:
            await self.coordinator._add_analysis_suggestions(out_context)
            log_debug("\033[92mCompleted: Plan\033[0m", show_sep="below")
        if self.interface is not None:
            for message_obj in self.interface.objects[::-1]:
                if isinstance(message_obj.object, Card):
                    message_obj.object.collapsed = True
        return outputs, out_context


class Coordinator(Viewer, VectorLookupToolUser):
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
            "tool_relevance": {
                "template": PROMPTS_DIR / "Coordinator" / "tool_relevance.jinja2",
                "response_model": ThinkingYesNo,
            },
        },
    )

    agents = param.List(default=[ChatAgent], doc="""
        List of agents to coordinate.""")

    context = param.Dict(default={})

    demo_inputs = param.List(default=DEMO_MESSAGES, doc="""
        List of instructions to demo the Coordinator.""")

    history = param.Integer(default=3, doc="""
        Number of previous user-assistant interactions to include in the chat history.""")

    logs_db_path = param.String(default=None, doc="""
        The path to the log file that will store the messages exchanged with the LLM.""")

    suggestions = param.List(default=GETTING_STARTED_SUGGESTIONS, doc="""
        Initial list of suggestions of actions the user can take.""")

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
        logs_db_path: str = "",
        **params,
    ):
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

        def on_clear(instance, _):
            pass

        def on_submit(event=None, instance=None):
            chat_input = self.interface.active_widget
            uploaded = chat_input.value_uploaded
            user_prompt = chat_input.value_input
            if not user_prompt and not uploaded:
                return

            with self.interface.param.update(disabled=True, loading=True):
                if self._main[0] is not self.interface:
                    # Reset value input because reset has no time to propagate
                    self._main[:] = [self.interface]

                old_sources = context.get("sources", [])
                if uploaded:
                    # Process uploaded files through SourceControls if any exist
                    source_controls = SourceControls(
                        downloaded_files={key: value["value"] for key, value in uploaded.items()},
                        context=context,
                        replace_controls=False,
                        show_input=False,
                        clear_uploads=True  # Clear the uploads after processing
                    )
                    source_controls.param.trigger("add")
                    chat_input.value_uploaded = {}
                    source_cards = [
                        TableSourceCard(source=source, name=source.name)
                        for source in context.get("sources", []) if source not in old_sources
                    ]
                    if len(source_cards) > 1:
                        source_view = Accordion(*source_cards, sizing_mode="stretch_width", name="TableSourceCard")
                    else:
                        source_view = source_cards[0]
                    msg = Column(chat_input.value, source_view) if user_prompt else source_view
                else:
                    msg = Markdown(user_prompt)

                with hold():
                    self.interface.send(msg, respond=bool(user_prompt))
                    chat_input.value_input = ""

        log_debug("New Session: \033[92mStarted\033[0m", show_sep="above")

        if interface is None:
            interface = ChatInterface(
                callback=self._chat_invoke,
                load_buffer=5,
                on_submit=on_submit,
                show_button_tooltips=True,
                show_button_name=False,
                sizing_mode="stretch_both"
            )
        else:
            interface.callback = self._chat_invoke
            interface.on_submit = on_submit

        welcome_title = Typography(
            "Illuminate your data",
            disable_anchors=True,
            variant="h1"
        )

        if context is None:
            context = {}
        num_sources = len(context.get("sources", []))
        prefix_text = "Add your dataset to begin, then" if num_sources == 0 else f"{num_sources} source{'s' if num_sources > 1 else ''} connected;"
        welcome_text = Typography(
            f"{prefix_text} ask any question, or select a quick action below."
        )

        welcome_screen = Paper(
            welcome_title,
            welcome_text,
            interface._widget,
            max_width=850,
            styles={'margin': 'auto'},
            sx={'p': '0 20px 20px 20px'}
        )

        self._main = Column(
            welcome_screen,
            sx={'display': 'flex', 'align-items': 'center'},
            height_policy='max'
        )

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
        tools = tools or []
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

        # If none of the tools provide vector_metaset, add tablelookup
        provides_vector_metaset = any(
            "vector_metaset" in tool.provides
            for tool in tools or []
        )
        provides_sql_metaset = any(
            "sql_metaset" in tool.provides
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

        # Add user-provided tools to the list of tools of the coordinator
        if "prompts" not in params:
            params["prompts"] = {}
        if "main" not in params["prompts"]:
            params["prompts"]["main"] = {}

        if "tools" not in params["prompts"]["main"]:
            params["prompts"]["main"]["tools"] = []
        params["prompts"]["main"]["tools"] += [tool for tool in tools]
        super().__init__(
            llm=llm, agents=instantiated, interface=interface, logs_db_path=logs_db_path,
            vector_store=vector_store, document_vector_store=document_vector_store, context=context,
            **params
        )

        interface.button_properties = {
            "undo": {"callback": on_undo},
            "rerun": {"callback": on_rerun},
            "clear": {"callback": on_clear},
        }


        # Use existing method to create suggestions
        self._add_suggestions_to_footer(
            self.suggestions,
            num_objects=1,
            inplace=True,
            analysis=False,
            append_demo=True,
            hide_after_use=False
        )

    def __panel__(self):
        return self._main

    def _add_suggestions_to_footer(
        self,
        suggestions: list[str],
        num_objects: int = 2,
        inplace: bool = True,
        analysis: bool = False,
        append_demo: bool = True,
        hide_after_use: bool = True,
        context: TContext | None = None
    ):
        if not suggestions:
            return

        async def hide_suggestions(_=None):
            if len(self.interface.objects) > num_objects:
                suggestion_buttons.visible = False

        async def use_suggestion(event):
            if self._main[0] is not self.interface:
                self._main[:] = [self.interface]

            button = event.obj
            with button.param.update(loading=True), self.interface.active_widget.param.update(loading=True):
                # Find the original suggestion tuple to get the text
                button_index = suggestion_buttons.objects.index(button)
                if button_index < len(suggestions):
                    suggestion = suggestions[button_index]
                    contents = suggestion[1] if isinstance(suggestion, tuple) else suggestion
                else:
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
                    with agent.param.update(agents=self.agents):
                        await agent.respond(messages, context)
                        # Pass the same memory to _add_analysis_suggestions
                        await self._add_analysis_suggestions(context)
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
                    name=suggestion[1] if isinstance(suggestion, tuple) else suggestion,
                    icon=suggestion[0] if isinstance(suggestion, tuple) else None,
                    button_style="outlined",
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
                icon="play_arrow",
                button_type="primary",
                variant="outlined",
                on_click=run_demo,
                margin=5,
                disabled=self.interface.param.loading
            ))
        disable_js = "cb_obj.origin.disabled = true; setTimeout(() => cb_obj.origin.disabled = false, 3000)"
        for b in suggestion_buttons:
            b.js_on_click(code=disable_js)

        if len(self.interface) != 0:
            message = self.interface.objects[-1]
            if inplace:
                footer_objects = message.footer_objects or []
                message.footer_objects = footer_objects + [suggestion_buttons]
        else:
            self._main[0].append(suggestion_buttons)

        self.interface.param.watch(hide_suggestions, "objects")

    async def _add_analysis_suggestions(self, context: TContext):
        pipeline = context["pipeline"]
        current_analysis = context.get("analysis")

        # Clear current_analysis unless the last message is the same AnalysisOutput
        if current_analysis and self.interface.objects:
            last_message_obj = self.interface.objects[-1].object
            if not (isinstance(last_message_obj, AnalysisOutput) and last_message_obj.analysis is current_analysis):
                current_analysis = None

        allow_consecutive = getattr(current_analysis, '_consecutive_calls', True)
        applicable_analyses = []
        for analysis in self._analyses:
            if await analysis.applies(pipeline) and (allow_consecutive or analysis is not type(current_analysis)):
                applicable_analyses.append(analysis)
        self._add_suggestions_to_footer(
            [f"Apply {analysis.__name__}" for analysis in applicable_analyses],
            append_demo=False,
            analysis=True,
            context=context,
            hide_after_use=False,
            num_objects=len(self.interface.objects),
        )

    @wrap_logfire(span_name="Chat Invoke")
    async def _chat_invoke(self, messages: list[Message], context: TContext, user: str, instance: ChatInterface) -> Plan:
        log_debug(f"New Message: \033[91m{messages!r}\033[0m", show_sep="above")
        return await self.respond(messages, context)

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

    async def respond(self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]) -> Plan | None:
        context = {"agent_tool_contexts": [], **context}
        with self.interface.param.update(loading=True):
            if isinstance(self.llm, LlamaCpp):
                with self.interface.add_step(title="Loading LlamaCpp model...", success_title="Using the cached LlamaCpp model", user="Assistant") as step:
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

            agents, tools, pre_plan_output = await self._pre_plan(messages, context, agents, tools)
            context["plan"] = plan = await self._compute_plan(messages, context, agents, tools, pre_plan_output)
            if plan is None:
                msg = (
                    "Assistant could not settle on a plan of action to perform the requested query. "
                    "Please restate your request."
                )
                self.interface.stream(msg, user='Lumen')
                return

        return plan

    async def _check_tool_relevance(
        self, tool: Tool, tool_output: str, actor: Actor, actor_task: str, messages: list[Message], context: TContext
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
            with self.interface.add_step(title="Selecting primary agent...", user="Assistant") as step:
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
                tasks.append(
                    TaskGroup(
                        subagent,
                        provides=unmet_dependencies,
                        instruction=output.chain_of_thought,
                        title=output.agent_or_tool
                    )
                )
                step.success_title = f"Solved a dependency with {output.agent_or_tool}"
        return Plan(*(tasks[::-1] + [TaskGroup(agent, instruction=cot)]), history=messages, context=context, coordinator=self)


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

        steps_layout = self.steps_layout
        if self.interface and steps_layout is None:
            steps_layout = None
            for step_message in reversed(self.interface.objects[-5:]):
                if step_message.user == "Planner" and isinstance(step_message.object, Card):
                    steps_layout = step_message.object
                    break

        with self.interface.add_step(title="Gathering context for planning...", user="Assistant", steps_layout=steps_layout) as step:
            for tool in self.planner_tools:
                is_relevant = await self._check_tool_relevance(
                    tool, "", self, f"Gather context for planning to answer {user_query}", messages, context
                )

                if not is_relevant:
                    continue

                tool_name = getattr(tool, "name", type(tool).__name__)
                step.stream(f"Using {tool_name} to gather planning context...")
                task = TaskGroup(
                    tool,
                    interface=self.interface,
                    instruction=user_query,
                    title=f"Gathering context with {tool_name}",
                    steps_layout=steps_layout,
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
            with self.interface.add_step(
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
        agents = [agent for agent in agents if len(set(agent.input_schema.__annotations__) - all_provides) == 0 and type(agent).__name__ != "ValidationAgent"]
        tools = [tool for tool in tools if len(set(tool.input_schema.__annotations__) - all_provides) == 0]
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
                self.steps_layout.title = "🧠 Reasoning about the plan..."
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
                    TaskGroup(
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
                TaskGroup(
                    subagent,
                    instruction=step.instruction,
                    title=step.title
                )
            )
            steps.append(step)
            actors_in_graph.add(key)

        last_task = tasks[-1]
        if isinstance(last_task[0], Tool):
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
                return Plan(*tasks, title=raw_plan.title, history=messages, context=context, coordinator=self), previous_actors

            summarize_step = type(step)(
                actor=actor,
                instruction='Summarize the results.',
                title='Summarizing results',
            )
            steps.append(summarize_step)
            tasks.append(
                TaskGroup(
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
                TaskGroup(
                    agents["ValidationAgent"],
                    instruction=validation_step.instruction,
                    title=validation_step.title
                )
            )
            actors_in_graph.add("ValidationAgent")

        raw_plan.steps = steps
        return Plan(*tasks, title=raw_plan.title, history=messages, context=context, coordinator=self), actors

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

        self._todos_title = Typography("📋 Building checklist...", css_classes=["todos-title"], margin=0, styles={"font-weight": "normal", "font-size": "1.1em"})
        self._todos = Typography(css_classes=["todos"], margin=0, styles={"font-weight": "normal"})
        with self.interface.param.update(callback_exception="raise"):
            with self.interface.add_step(
                title="Planning how to solve user query...", user="Planner",
                layout_params={"header": Column(self._todos_title, self._todos), "collapsed": not self.verbose}
            ) as istep:
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
                        self._todos_title.object = istep.failed_title = 'Planning was cancelled, please try again.'
                        traceback.print_exception(e)
                        raise e
                    except Exception as e:
                        self._todos_title.object = istep.failed_title = 'Failed to make plan. Ensure LLM is configured correctly and/or try again.'
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
                        self._todos_title.object = "❌ Planning failed"
                        e = RuntimeError("Planner failed to come up with viable plan after 5 attempts.")
                        traceback.print_exception(e)
                        raise e

            # Store the todo message reference for later updates
            self._todo_step = istep
            if attempts > 0:
                istep.success_title = f"Plan with {len(raw_plan.steps)} steps created after {attempts + 1} attempts"
            else:
                istep.success_title = f"Plan with {len(raw_plan.steps)} steps created"
        return plan
